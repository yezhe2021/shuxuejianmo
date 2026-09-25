"""Benchmark representative TRAC-Net training steps on real samples."""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from tracnet.config import TRACNetConfig
from tracnet.data import TemporalCorruptor, build_datasets, move_batch
from tracnet.losses import TRACNetCriterion
from tracnet.model import TRACNet
from tracnet.train import make_optimizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--bert", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    device = torch.device("cuda")
    datasets, _ = build_datasets(args.data, text_mode="bert")
    loader = DataLoader(
        datasets["train"], batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    batches = itertools.cycle(loader)
    config = TRACNetConfig(
        text_mode="bert",
        bert_path=str(args.bert.resolve()),
        d_model=128,
        num_heads=4,
        temporal_layers=2,
        feedforward_dim=256,
        local_window=2,
        dropout=0.15,
    )
    model = TRACNet(config).to(device).train()
    optimizer = make_optimizer(model, 3e-4, 2e-5, 1e-4)
    criterion = TRACNetCriterion()
    corruptor = TemporalCorruptor()
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()

    durations: list[float] = []
    for step in range(args.warmup + args.steps):
        batch = move_batch(next(batches), device)
        corrupted, artificial = corruptor(batch)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            reference = model(batch)
            damaged = model(corrupted)
            loss, _ = criterion(reference, damaged, batch, artificial)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        duration = time.perf_counter() - started
        if step >= args.warmup:
            durations.append(duration)
        print(f"step={step + 1} seconds={duration:.4f} loss={float(loss.detach()):.4f}")

    mean = sum(durations) / len(durations)
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    print(f"mean_step_seconds={mean:.6f}")
    print(f"peak_allocated_gib={peak_gib:.3f}")


if __name__ == "__main__":
    main()
