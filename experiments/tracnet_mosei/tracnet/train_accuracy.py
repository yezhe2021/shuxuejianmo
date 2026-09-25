from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .baselines import BaselineOutput, build_baseline
from .config import TRACNetConfig
from .data import build_datasets, move_batch
from .losses import pearson_loss
from .metrics import all_metrics
from .train_baseline import save_checkpoint, seed_everything


VARIANTS = {
    "simple_clean": "simple_multimodal",
    "hierarchical_clean": "hierarchical_multimodal",
    "anchor_hierarchical_clean": "text_anchor_hierarchical",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train accuracy-first diagnostic models")
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--bert", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def accuracy_first_loss(
    output: BaselineOutput,
    class_target: torch.Tensor,
    regression_target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if output.neutral_logit is None:
        three_class = F.cross_entropy(output.logits, class_target)
        classification = three_class
        neutral = three_class.new_zeros(())
        polarity = three_class.new_zeros(())
    else:
        three_class = F.nll_loss(output.logits, class_target)
        neutral_target = (class_target == 1).float()
        neutral = F.binary_cross_entropy_with_logits(
            output.neutral_logit, neutral_target
        )
        non_neutral = class_target != 1
        if non_neutral.any():
            polarity_target = (class_target[non_neutral] == 2).float()
            polarity = F.binary_cross_entropy_with_logits(
                output.polarity_logit[non_neutral], polarity_target
            )
        else:
            polarity = neutral.new_zeros(())
        classification = neutral + polarity + 0.3 * three_class

    regression = F.smooth_l1_loss(output.regression, regression_target)
    correlation = pearson_loss(output.regression, regression_target)
    total = classification + 0.2 * regression + 0.05 * correlation
    return total, {
        "classification": classification.detach(),
        "three_class": three_class.detach(),
        "neutral": neutral.detach(),
        "polarity": polarity.detach(),
        "regression": regression.detach(),
        "pearson": correlation.detach(),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    class_truth: list[int] = []
    class_prediction: list[int] = []
    regression_truth: list[float] = []
    regression_prediction: list[float] = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        output = model(batch)
        class_truth.extend(batch["classification_label"].cpu().tolist())
        class_prediction.extend(output.logits.argmax(dim=-1).cpu().tolist())
        regression_truth.extend(batch["regression_label"].cpu().tolist())
        regression_prediction.extend(output.regression.cpu().tolist())
    metrics = all_metrics(
        class_truth, class_prediction, regression_truth, regression_prediction
    )
    matrix = confusion_matrix(class_truth, class_prediction, labels=[0, 1, 2])
    return metrics, matrix


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
    model_name = VARIANTS[args.variant]
    model = build_baseline(model_name, config).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-5
    )
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "variant": args.variant,
                "model": model_name,
                "loss": {
                    "regression_weight": 0.2,
                    "pearson_weight": 0.05,
                    "hierarchical_three_class_weight": 0.3,
                },
                **config.to_dict(),
            },
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
        totals = {
            key: 0.0
            for key in (
                "total",
                "classification",
                "three_class",
                "neutral",
                "polarity",
                "regression",
                "pearson",
            )
        }
        steps = 0
        for raw_batch in loaders["train"]:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                output = model(batch)
                loss, parts = accuracy_first_loss(
                    output,
                    batch["classification_label"],
                    batch["regression_label"],
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            totals["total"] += float(loss.detach())
            for key, value in parts.items():
                totals[key] += float(value)
            steps += 1

        valid_metrics, valid_matrix = evaluate(model, loaders["valid"], device)
        # Accuracy is primary; Macro-F1 only breaks practically exact ties.
        score = valid_metrics["accuracy"] + 1e-3 * valid_metrics["macro_f1"]
        scheduler.step(score)
        record = {
            "epoch": epoch,
            "seconds": round(time.time() - started, 2),
            "train": {key: value / max(1, steps) for key, value in totals.items()},
            "valid": valid_metrics,
            "valid_confusion_matrix": valid_matrix.tolist(),
            "selection_score": score,
            "learning_rate": optimizer.param_groups[0]["lr"],
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
    test_metrics, test_matrix = evaluate(model, loaders["test"], device)
    result = {
        **test_metrics,
        "confusion_matrix": test_matrix.tolist(),
        "best_epoch": checkpoint["metrics"]["epoch"],
        "best_validation": checkpoint["metrics"]["valid"],
    }
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("test", json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
