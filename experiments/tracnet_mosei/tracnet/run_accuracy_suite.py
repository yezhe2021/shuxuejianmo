from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run accuracy-first models sequentially")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--bert", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    variants = (
        "simple_clean",
        "hierarchical_clean",
        "anchor_hierarchical_clean",
    )
    for variant in variants:
        command = [
            sys.executable,
            "-u",
            "-m",
            "tracnet.train_accuracy",
            "--variant",
            variant,
            "--data",
            str(args.data.resolve()),
            "--bert",
            str(args.bert.resolve()),
            "--output-dir",
            str((args.output_root / variant).resolve()),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
            "--amp",
        ]
        print(f"START {variant}", flush=True)
        completed = subprocess.run(command, check=False)
        print(f"END {variant} exit_code={completed.returncode}", flush=True)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
