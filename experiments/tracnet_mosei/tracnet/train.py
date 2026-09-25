from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .config import TRACNetConfig
from .data import TemporalCorruptor, build_datasets, move_batch
from .losses import LossWeights, TRACNetCriterion
from .metrics import all_metrics
from .model import TRACNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TRAC-Net on aligned CMU-MOSEI features")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--bert", type=Path, default=Path("E题数据/model"))
    parser.add_argument("--text-mode", choices=("bert", "features"), default="bert")
    parser.add_argument("--output-dir", type=Path, default=Path("tracnet/runs/default"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--bert-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--local-window", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--unfreeze-bert-layers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--validation-corruptions", type=int, default=3)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_optimizer(
    model: TRACNet, learning_rate: float, bert_learning_rate: float, weight_decay: float
) -> torch.optim.Optimizer:
    bert_parameters = []
    other_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (bert_parameters if name.startswith("bert.") else other_parameters).append(parameter)
    groups: list[dict[str, Any]] = [
        {"params": other_parameters, "lr": learning_rate, "weight_decay": weight_decay}
    ]
    if bert_parameters:
        groups.append(
            {"params": bert_parameters, "lr": bert_learning_rate, "weight_decay": weight_decay}
        )
    return torch.optim.AdamW(groups)


@torch.no_grad()
def evaluate(
    model: TRACNet,
    loader: DataLoader,
    device: torch.device,
    corruptor: TemporalCorruptor | None = None,
) -> dict[str, float]:
    model.eval()
    class_truth: list[int] = []
    class_prediction: list[int] = []
    regression_truth: list[float] = []
    regression_prediction: list[float] = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        if corruptor is not None:
            batch, _ = corruptor(batch)
        output = model(batch)
        class_truth.extend(batch["classification_label"].cpu().tolist())
        class_prediction.extend(output.logits.argmax(dim=-1).cpu().tolist())
        regression_truth.extend(batch["regression_label"].cpu().tolist())
        regression_prediction.extend(output.regression.cpu().tolist())
    return all_metrics(class_truth, class_prediction, regression_truth, regression_prediction)


def evaluate_fixed_missing_average(
    model: TRACNet,
    loader: DataLoader,
    device: torch.device,
    seeds: list[int],
) -> dict[str, float]:
    """Average deterministic missing-pattern evaluations.

    Recreating each seeded corruptor on every epoch guarantees that validation
    sample i receives the same missing spans throughout training.
    """

    runs = [
        evaluate(model, loader, device, corruptor=TemporalCorruptor(seed=seed))
        for seed in seeds
    ]
    return {
        key: float(np.mean([metrics[key] for metrics in runs])) for key in runs[0]
    }


def save_checkpoint(
    path: Path,
    model: TRACNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: TRACNetConfig,
    normalizer_state: dict[str, Any],
    metrics: dict[str, float],
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "config": config.to_dict(),
            "normalizer": normalizer_state,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    config = TRACNetConfig(
        text_mode=args.text_mode,
        bert_path=str(args.bert.resolve()),
        freeze_bert=args.unfreeze_bert_layers == 0,
        bert_trainable_layers=args.unfreeze_bert_layers,
        d_model=args.d_model,
        temporal_layers=args.temporal_layers,
        local_window=args.local_window,
        dropout=args.dropout,
        feedforward_dim=args.d_model * 2,
    )
    datasets, normalizer = build_datasets(args.data, text_mode=args.text_mode)
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        for name, dataset in datasets.items()
    }
    model = TRACNet(config).to(device)
    criterion = TRACNetCriterion(LossWeights())
    optimizer = make_optimizer(
        model, args.learning_rate, args.bert_learning_rate, args.weight_decay
    )
    corruptor = TemporalCorruptor(seed=args.seed + 17)
    validation_seeds = [args.seed + 1000 + index for index in range(args.validation_corruptions)]
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_learning_rate,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    best_score = -float("inf")
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        running: dict[str, float] = defaultdict(float)
        steps = 0
        for raw_batch in loaders["train"]:
            batch = move_batch(raw_batch, device)
            corrupted_batch, artificial_missing = corruptor(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                reference = model(batch)
                corrupted = model(corrupted_batch)
                loss, parts = criterion(reference, corrupted, batch, artificial_missing)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            for key, value in parts.items():
                running[key] += float(value)
            steps += 1

        clean_metrics = evaluate(model, loaders["valid"], device)
        missing_metrics = evaluate_fixed_missing_average(
            model, loaders["valid"], device, validation_seeds
        )
        # Higher is better; MAE is subtracted to keep all official metrics represented.
        score = (
            missing_metrics["macro_f1"]
            + missing_metrics["pearson"]
            - 0.25 * missing_metrics["mae"]
        )
        scheduler.step(score)
        record = {
            "epoch": epoch,
            "seconds": round(time.time() - started, 2),
            "train": {key: value / max(1, steps) for key, value in running.items()},
            "valid_clean": clean_metrics,
            "valid_missing": missing_metrics,
            "selection_score": score,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "validation_corruption_seeds": validation_seeds,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        (args.output_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        save_checkpoint(
            args.output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            config,
            normalizer.state_dict(),
            record,
            scheduler,
        )
        if score > best_score:
            best_score = score
            stale_epochs = 0
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                config,
                normalizer.state_dict(),
                record,
                scheduler,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, loaders["test"], device)
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("test", json.dumps(test_metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
