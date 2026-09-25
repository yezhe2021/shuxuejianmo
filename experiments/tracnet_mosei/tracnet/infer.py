from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import TRACNetConfig
from .data import FeatureNormalizer, infer_masks, iter_pickle_samples, move_batch
from .model import TRACNet


POLARITY = ("Negative", "Neutral", "Positive")
MODALITIES = ("text", "audio", "vision")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TRAC-Net on attachment 3 or 4")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="A pickle file or directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bert", type=Path, default=None, help="Optional local BERT path override")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def prepare_sample(
    sample: dict[str, Any],
    normalizer: FeatureNormalizer,
    text_mode: str,
) -> dict[str, Any]:
    text_bert = np.asarray(sample["text_bert"])
    audio = np.asarray(sample["audio"])
    vision = np.asarray(sample["vision"])
    padding, observed = infer_masks(text_bert, audio, vision)
    batch: dict[str, Any] = {
        "text_bert": torch.as_tensor(text_bert, dtype=torch.long).unsqueeze(0),
        "audio": torch.from_numpy(normalizer.transform(audio, "audio", observed[1])).unsqueeze(0),
        "vision": torch.from_numpy(normalizer.transform(vision, "vision", observed[2])).unsqueeze(0),
        "padding_mask": torch.from_numpy(padding).unsqueeze(0),
        "observed_mask": torch.from_numpy(observed).unsqueeze(0),
    }
    if text_mode == "features":
        if "text" not in sample:
            raise KeyError("Checkpoint uses text features, but this sample contains no 'text' field")
        batch["text"] = torch.as_tensor(sample["text"], dtype=torch.float32).unsqueeze(0)
    return batch


def input_files(path: Path) -> list[Path]:
    return [path] if path.is_file() else sorted(path.rglob("*.pkl"))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = TRACNetConfig.from_dict(checkpoint["config"])
    if args.bert is not None:
        config.bert_path = str(args.bert.resolve())
    normalizer = FeatureNormalizer.from_state_dict(checkpoint["normalizer"])
    model = TRACNet(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for path in input_files(args.input):
            for sample_id, sample in iter_pickle_samples(path):
                batch = move_batch(prepare_sample(sample, normalizer, config.text_mode), device)
                output = model(batch)
                probabilities = torch.softmax(output.logits, dim=-1)[0].cpu().numpy()
                evidence = output.evidence[0].cpu().numpy()
                modality_importance = evidence.sum(axis=1)
                main_modality = int(modality_importance.argmax())
                valid_length = int(batch["padding_mask"][0].sum().item())
                flat = evidence[:, :valid_length].reshape(-1)
                top_count = min(args.top_k, flat.size)
                top_indices = np.argsort(flat)[-top_count:][::-1]
                top_evidence = []
                for index in top_indices:
                    modality = int(index // valid_length)
                    position = int(index % valid_length)
                    top_evidence.append(
                        {
                            "modality": MODALITIES[modality],
                            "position": position + 1,
                            "score": round(float(flat[index]), 8),
                        }
                    )
                predicted_class = int(probabilities.argmax())
                rows.append(
                    {
                        "id": sample_id,
                        "polarity": POLARITY[predicted_class],
                        "class_index": predicted_class,
                        "negative_probability": float(probabilities[0]),
                        "neutral_probability": float(probabilities[1]),
                        "positive_probability": float(probabilities[2]),
                        "sentiment_intensity": float(output.regression[0].cpu()),
                        "main_modality": MODALITIES[main_modality],
                        "text_importance": float(modality_importance[0]),
                        "audio_importance": float(modality_importance[1]),
                        "vision_importance": float(modality_importance[2]),
                        "top_evidence": json.dumps(top_evidence, ensure_ascii=False),
                        "source_file": str(path),
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No pickle samples found under {args.input}")
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
