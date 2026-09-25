from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run diagnostic baselines sequentially")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--bert", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    for baseline in ("text_only", "simple_multimodal"):
        output_dir = args.output_root / baseline
        command = [
            sys.executable,
            "-u",
            "-m",
            "tracnet.train_baseline",
            "--baseline",
            baseline,
            "--data",
            str(args.data.resolve()),
            "--bert",
            str(args.bert.resolve()),
            "--output-dir",
            str(output_dir.resolve()),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
            "--amp",
        ]
        print(f"START {baseline}", flush=True)
        completed = subprocess.run(command, check=False)
        print(f"END {baseline} exit_code={completed.returncode}", flush=True)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
