from __future__ import annotations

import copy
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


MODALITIES = ("text", "audio", "vision")


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _row_nonzero(array: np.ndarray, atol: float = 1e-8) -> np.ndarray:
    return np.any(np.abs(array) > atol, axis=-1)


def infer_masks(
    text_bert: np.ndarray,
    audio: np.ndarray,
    vision: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer left-aligned valid positions and per-modality observations.

    Padding is the suffix after the final non-zero position across all modalities.
    Internal all-zero runs remain valid so simultaneous local missingness is not
    mistaken for padding.
    """

    if text_bert.ndim == 2:
        text_bert = text_bert[None, ...]
        audio = audio[None, ...]
        vision = vision[None, ...]
        squeeze = True
    else:
        squeeze = False

    text_ids = text_bert[:, 0, :]
    text_observed = np.abs(text_ids) > 0
    audio_observed = _row_nonzero(audio)
    vision_observed = _row_nonzero(vision)
    any_observed = text_observed | audio_observed | vision_observed
    batch, seq_len = any_observed.shape
    positions = np.arange(seq_len)[None, :]
    last = np.where(any_observed, positions, -1).max(axis=1)
    padding_mask = positions <= last[:, None]
    padding_mask[last < 0] = False
    observed_mask = np.stack(
        [text_observed, audio_observed, vision_observed], axis=1
    ) & padding_mask[:, None, :]
    if squeeze:
        return padding_mask[0], observed_mask[0]
    return padding_mask, observed_mask


@dataclass
class FeatureNormalizer:
    audio_mean: np.ndarray
    audio_std: np.ndarray
    vision_mean: np.ndarray
    vision_std: np.ndarray

    @staticmethod
    def _stats(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        selected = values[mask]
        if selected.size == 0:
            return np.zeros(values.shape[-1], dtype=np.float32), np.ones(
                values.shape[-1], dtype=np.float32
            )
        mean = selected.mean(axis=0, dtype=np.float64)
        std = selected.std(axis=0, dtype=np.float64)
        std = np.where(std < 1e-6, 1.0, std)
        return mean.astype(np.float32), std.astype(np.float32)

    @classmethod
    def fit(cls, split: dict[str, Any]) -> "FeatureNormalizer":
        padding, observed = infer_masks(
            split["text_bert"], split["audio"], split["vision"]
        )
        audio_mean, audio_std = cls._stats(split["audio"], observed[:, 1] & padding)
        vision_mean, vision_std = cls._stats(split["vision"], observed[:, 2] & padding)
        return cls(audio_mean, audio_std, vision_mean, vision_std)

    def transform(self, values: np.ndarray, modality: str, observed: np.ndarray) -> np.ndarray:
        if modality == "audio":
            mean, std = self.audio_mean, self.audio_std
        elif modality == "vision":
            mean, std = self.vision_mean, self.vision_std
        else:
            raise ValueError(f"Unsupported normalized modality: {modality}")
        normalized = (values.astype(np.float32) - mean) / std
        return normalized * observed[..., None].astype(np.float32)

    def state_dict(self) -> dict[str, list[float]]:
        return {
            "audio_mean": self.audio_mean.tolist(),
            "audio_std": self.audio_std.tolist(),
            "vision_mean": self.vision_mean.tolist(),
            "vision_std": self.vision_std.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "FeatureNormalizer":
        return cls(**{key: np.asarray(value, dtype=np.float32) for key, value in state.items()})


class MOSEIDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        split: dict[str, Any],
        normalizer: FeatureNormalizer,
        text_mode: str = "bert",
    ) -> None:
        self.split = split
        self.normalizer = normalizer
        self.text_mode = text_mode
        self.padding_mask, self.observed_mask = infer_masks(
            split["text_bert"], split["audio"], split["vision"]
        )

    def __len__(self) -> int:
        return len(self.split["audio"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        observed = self.observed_mask[index]
        item: dict[str, Any] = {
            "id": str(self.split["id"][index]),
            "text_bert": torch.as_tensor(self.split["text_bert"][index], dtype=torch.long),
            "audio": torch.from_numpy(
                self.normalizer.transform(self.split["audio"][index], "audio", observed[1])
            ),
            "vision": torch.from_numpy(
                self.normalizer.transform(self.split["vision"][index], "vision", observed[2])
            ),
            "padding_mask": torch.from_numpy(self.padding_mask[index]),
            "observed_mask": torch.from_numpy(observed),
            "classification_label": torch.tensor(
                int(self.split["classification_labels"][index]), dtype=torch.long
            ),
            "regression_label": torch.tensor(
                float(self.split["regression_labels"][index]), dtype=torch.float32
            ),
        }
        if self.text_mode == "features":
            item["text"] = torch.as_tensor(self.split["text"][index], dtype=torch.float32)
        return item


def build_datasets(
    pickle_path: str | Path, text_mode: str = "bert"
) -> tuple[dict[str, MOSEIDataset], FeatureNormalizer]:
    data = load_pickle(pickle_path)
    normalizer = FeatureNormalizer.fit(data["train"])
    datasets = {
        name: MOSEIDataset(split, normalizer, text_mode=text_mode)
        for name, split in data.items()
        if name in {"train", "valid", "test"}
    }
    return datasets, normalizer


class TemporalCorruptor:
    """Create contiguous, label-independent missing spans inside valid positions."""

    def __init__(
        self,
        min_ratio: float = 0.1,
        max_ratio: float = 0.5,
        min_modalities: int = 1,
        max_modalities: int = 3,
        spans_per_sample: int = 1,
        seed: int | None = None,
    ) -> None:
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_modalities = min_modalities
        self.max_modalities = max_modalities
        self.spans_per_sample = spans_per_sample
        self.rng = random.Random(seed) if seed is not None else random

    def __call__(self, batch: dict[str, Any]) -> tuple[dict[str, Any], torch.Tensor]:
        corrupted = {
            key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
            for key, value in batch.items()
        }
        padding = batch["padding_mask"]
        observed = batch["observed_mask"].clone()
        artificial = torch.zeros_like(observed)
        batch_size, seq_len = padding.shape

        for b in range(batch_size):
            valid_positions = torch.nonzero(padding[b], as_tuple=False).flatten()
            if valid_positions.numel() < 3:
                continue
            first = int(valid_positions[0])
            last = int(valid_positions[-1]) + 1
            valid_len = last - first
            for _ in range(self.spans_per_sample):
                count = self.rng.randint(self.min_modalities, self.max_modalities)
                modalities = self.rng.sample(range(3), k=count)
                ratio = self.rng.uniform(self.min_ratio, self.max_ratio)
                span_len = max(1, min(valid_len - 1, round(valid_len * ratio)))
                # Keep the first token available for BERT stability when possible.
                low = min(first + 1, last - span_len)
                high = max(low, last - span_len)
                start = self.rng.randint(low, high) if high >= low else first
                stop = start + span_len
                for modality in modalities:
                    mask = observed[b, modality, start:stop]
                    artificial[b, modality, start:stop] |= mask
                    observed[b, modality, start:stop] = False
                    if modality == 0:
                        corrupted["text_bert"][b, :, start:stop] = 0
                        if "text" in corrupted:
                            corrupted["text"][b, start:stop] = 0
                    elif modality == 1:
                        corrupted["audio"][b, start:stop] = 0
                    else:
                        corrupted["vision"][b, start:stop] = 0

        corrupted["observed_mask"] = observed
        return corrupted, artificial


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def iter_pickle_samples(path: str | Path) -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield normalized dictionary shapes from attachment 3/4 pickle layouts."""

    source = Path(path)
    data = load_pickle(source)
    if set(data) == {"test"}:
        split = data["test"]
        size = int(np.asarray(next(iter(split.values()))).shape[0])
        for index in range(size):
            yield f"{source.stem}_{index + 1}", {
                key: np.asarray(value[index]) for key, value in split.items()
            }
    else:
        yield str(data.get("id", source.stem)), data
