from __future__ import annotations

import random

import numpy as np
import torch

from .config import TRACNetConfig
from .data import TemporalCorruptor
from .losses import TRACNetCriterion
from .model import TRACNet


def make_batch(batch_size: int = 4, seq_len: int = 12) -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    lengths = torch.tensor([12, 10, 9, 7])[:batch_size]
    positions = torch.arange(seq_len)[None, :]
    padding = positions < lengths[:, None]
    observed = padding[:, None, :].expand(-1, 3, -1).clone()
    observed[0, 1, 4:7] = False
    observed[1, 2, 2:5] = False
    text = torch.randn(batch_size, seq_len, 768) * observed[:, 0, :, None]
    audio = torch.randn(batch_size, seq_len, 74) * observed[:, 1, :, None]
    vision = torch.randn(batch_size, seq_len, 35) * observed[:, 2, :, None]
    text_bert = torch.zeros(batch_size, 3, seq_len, dtype=torch.long)
    text_bert[:, 0] = (torch.arange(seq_len) + 100)[None, :] * padding
    text_bert[:, 1] = padding.long()
    return {
        "text": text,
        "text_bert": text_bert,
        "audio": audio,
        "vision": vision,
        "padding_mask": padding,
        "observed_mask": observed,
        "classification_label": torch.tensor([0, 1, 2, 2])[:batch_size],
        "regression_label": torch.tensor([-1.0, 0.0, 1.5, 0.5])[:batch_size],
    }


def main() -> None:
    random.seed(7)
    np.random.seed(7)
    config = TRACNetConfig(
        text_mode="features",
        d_model=32,
        num_heads=4,
        temporal_layers=1,
        feedforward_dim=64,
        max_seq_len=12,
        local_window=1,
        dropout=0.0,
    )
    model = TRACNet(config)
    batch = make_batch()
    corruptor = TemporalCorruptor(min_ratio=0.2, max_ratio=0.3)
    corrupted_batch, artificial_missing = corruptor(batch)
    reference = model(batch)
    corrupted = model(corrupted_batch)
    loss, parts = TRACNetCriterion()(reference, corrupted, batch, artificial_missing)
    loss.backward()

    assert reference.logits.shape == (4, 3)
    assert reference.regression.shape == (4,)
    assert reference.evidence.shape == (4, 3, 12)
    assert torch.isfinite(loss)
    assert torch.isfinite(reference.evidence).all()
    assert torch.all(reference.effective_reliability[~batch["observed_mask"] & batch["padding_mask"][:, None, :]] > 0)
    evidence_sum = reference.evidence.sum(dim=(1, 2))
    assert torch.allclose(evidence_sum, torch.ones_like(evidence_sum), atol=1e-5)
    print("TRAC-Net smoke test passed")
    print({key: round(float(value), 5) for key, value in parts.items()})


if __name__ == "__main__":
    main()
