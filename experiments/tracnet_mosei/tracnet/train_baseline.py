from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .baselines import BaselineOutput, build_baseline
from .config import TRACNetConfig
from .data import TemporalCorruptor, build_datasets, move_batch
from .losses import pearson_loss
from .metrics import all_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train diagnostic baselines")
    parser.add_argument(
        "--baseline", choices=("text_only", "simple_multimodal"), required=True
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--bert", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--validation-corruptions", type=int, default=3)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def task_loss(
    output: BaselineOutput,
    class_target: torch.Tensor,
    regression_target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    classification = F.cross_entropy(output.logits, class_target)
    regression = F.smooth_l1_loss(output.regression, regression_target)
    correlation = pearson_loss(output.regression, regression_target)
    total = classification + regression + 0.2 * correlation
    return total, {
        "classification": classification.detach(),
        "regression": regression.detach(),
        "pearson": correlation.detach(),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    corruption_seed: int | None = None,
) -> dict[str, float]:
    model.eval()
    corruptor = (
        TemporalCorruptor(seed=corruption_seed) if corruption_seed is not None else None
    )
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
    return all_metrics(
        class_truth, class_prediction, regression_truth, regression_prediction
    )


def average_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([item[key] for item in metrics])) for key in metrics[0]
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    config: TRACNetConfig,
    normalizer: dict[str, Any],
    record: dict[str, Any],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config.to_dict(),
            "normalizer": normalizer,
            "metrics": record,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    config = TRACNetConfig(
        text_mode="bert",
        bert_path=str(args.bert.resolve()),
        freeze_bert=True,
        d_model=128,
        temporal_layers=1,
        feedforward_dim=256,
        local_window=2,
        dropout=0.15,
    )
    datasets, normalizer = build_datasets(args.data, text_mode="bert")
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            pin_memory=device.type == "cuda",
        )
        for name, dataset in datasets.items()
    }
    model = build_baseline(args.baseline, config).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.lr_factor, patience=args.lr_patience, min_lr=1e-5
    )
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    corruptor = TemporalCorruptor(seed=args.seed + 17)
    validation_seeds = [args.seed + 1000 + i for i in range(args.validation_corruptions)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {"baseline": args.baseline, **config.to_dict()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        totals = {"total": 0.0, "classification": 0.0, "regression": 0.0, "pearson": 0.0}
        steps = 0
        for raw_batch in loaders["train"]:
            batch = move_batch(raw_batch, device)
            corrupted, _ = corruptor(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                clean_output = model(batch)
                missing_output = model(corrupted)
                clean_loss, clean_parts = task_loss(
                    clean_output,
                    batch["classification_label"],
                    batch["regression_label"],
                )
                missing_loss, missing_parts = task_loss(
                    missing_output,
                    batch["classification_label"],
                    batch["regression_label"],
                )
                loss = 0.5 * (clean_loss + missing_loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            totals["total"] += float(loss.detach())
            for key in ("classification", "regression", "pearson"):
                totals[key] += 0.5 * float(clean_parts[key] + missing_parts[key])
            steps += 1

        clean_metrics = evaluate(model, loaders["valid"], device)
        missing_metrics = average_metrics(
            [evaluate(model, loaders["valid"], device, seed) for seed in validation_seeds]
        )
        score = (
            missing_metrics["macro_f1"]
            + missing_metrics["pearson"]
            - 0.25 * missing_metrics["mae"]
        )
        scheduler.step(score)
        record = {
            "epoch": epoch,
            "seconds": round(time.time() - started, 2),
            "train": {key: value / max(1, steps) for key, value in totals.items()},
            "valid_clean": clean_metrics,
            "valid_missing": missing_metrics,
            "selection_score": score,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation_corruption_seeds": validation_seeds,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        (args.output_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        save_checkpoint(
            args.output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            config,
            normalizer.state_dict(),
            record,
        )
        if score > best_score:
            best_score = score
            stale = 0
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                config,
                normalizer.state_dict(),
                record,
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping after {epoch} epochs.", flush=True)
                break

    checkpoint = torch.load(
        args.output_dir / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, loaders["test"], device)
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("test", json.dumps(test_metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
