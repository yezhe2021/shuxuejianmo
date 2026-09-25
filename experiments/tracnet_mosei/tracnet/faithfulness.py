from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import TRACNetConfig
from .data import FeatureNormalizer, MOSEIDataset, load_pickle, move_batch
from .model import TRACNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evidence deletion/retention faithfulness test")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bert", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def mask_entries(
    batch: dict[str, Any],
    selected: torch.Tensor,
    retain_only: bool,
) -> dict[str, Any]:
    result = {
        key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()
    }
    observed = batch["observed_mask"].clone()
    change = (observed & ~selected) if retain_only else (observed & selected)
    result["text_bert"][:, :, :].masked_fill_(change[:, 0, None, :], 0)
    if "text" in result:
        result["text"].masked_fill_(change[:, 0, :, None], 0)
    result["audio"].masked_fill_(change[:, 1, :, None], 0)
    result["vision"].masked_fill_(change[:, 2, :, None], 0)
    result["observed_mask"] = observed & ~change
    return result


def select_top(evidence: torch.Tensor, observed: torch.Tensor, ratio: float) -> torch.Tensor:
    batch_size = evidence.shape[0]
    selected = torch.zeros_like(observed)
    for b in range(batch_size):
        valid_indices = torch.nonzero(observed[b].reshape(-1), as_tuple=False).flatten()
        count = max(1, round(valid_indices.numel() * ratio))
        scores = evidence[b].reshape(-1)[valid_indices]
        chosen = valid_indices[torch.topk(scores, min(count, scores.numel())).indices]
        selected[b].view(-1)[chosen] = True
    return selected


def select_random(observed: torch.Tensor, selected_top: torch.Tensor) -> torch.Tensor:
    selected = torch.zeros_like(observed)
    for b in range(observed.shape[0]):
        valid = torch.nonzero(observed[b].reshape(-1), as_tuple=False).flatten()
        count = int(selected_top[b].sum().item())
        permutation = torch.randperm(valid.numel(), device=valid.device)[:count]
        selected[b].view(-1)[valid[permutation]] = True
    return selected


def main() -> None:
    args = parse_args()
    if not 0 < args.ratio < 1:
        raise ValueError("ratio must be between 0 and 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = TRACNetConfig.from_dict(checkpoint["config"])
    if args.bert is not None:
        config.bert_path = str(args.bert.resolve())
    normalizer = FeatureNormalizer.from_state_dict(checkpoint["normalizer"])
    data = load_pickle(args.data)
    dataset = MOSEIDataset(data[args.split], normalizer, text_mode=config.text_mode)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = TRACNet(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    totals = {
        "baseline_confidence": 0.0,
        "top_deletion_confidence": 0.0,
        "random_deletion_confidence": 0.0,
        "top_retention_confidence": 0.0,
        "top_deletion_regression_change": 0.0,
        "random_deletion_regression_change": 0.0,
    }
    sample_count = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            baseline = model(batch)
            predicted = baseline.logits.argmax(dim=-1)
            top = select_top(baseline.evidence, batch["observed_mask"], args.ratio)
            random_mask = select_random(batch["observed_mask"], top)
            deleted = model(mask_entries(batch, top, retain_only=False))
            random_deleted = model(mask_entries(batch, random_mask, retain_only=False))
            retained = model(mask_entries(batch, top, retain_only=True))

            def confidence(output: Any) -> torch.Tensor:
                return torch.softmax(output.logits, dim=-1).gather(1, predicted[:, None]).squeeze(1)

            size = predicted.numel()
            totals["baseline_confidence"] += float(confidence(baseline).sum())
            totals["top_deletion_confidence"] += float(confidence(deleted).sum())
            totals["random_deletion_confidence"] += float(confidence(random_deleted).sum())
            totals["top_retention_confidence"] += float(confidence(retained).sum())
            totals["top_deletion_regression_change"] += float(
                torch.abs(deleted.regression - baseline.regression).sum()
            )
            totals["random_deletion_regression_change"] += float(
                torch.abs(random_deleted.regression - baseline.regression).sum()
            )
            sample_count += size

    metrics = {key: value / max(1, sample_count) for key, value in totals.items()}
    metrics["top_deletion_confidence_drop"] = (
        metrics["baseline_confidence"] - metrics["top_deletion_confidence"]
    )
    metrics["random_deletion_confidence_drop"] = (
        metrics["baseline_confidence"] - metrics["random_deletion_confidence"]
    )
    metrics["ratio"] = args.ratio
    metrics["samples"] = sample_count
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
