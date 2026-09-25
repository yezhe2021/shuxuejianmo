"""Print a compact, non-mutating summary of a competition pickle file."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def describe(value: Any, depth: int = 0) -> None:
    indent = "  " * depth
    if isinstance(value, dict):
        print(f"{indent}dict[{len(value)}] keys={list(value.keys())}")
        for key, child in value.items():
            print(f"{indent}- {key!r}:")
            describe(child, depth + 1)
        return
    if isinstance(value, np.ndarray):
        finite = value[np.isfinite(value)] if np.issubdtype(value.dtype, np.number) else None
        stats = ""
        if finite is not None and finite.size:
            stats = f", min={finite.min():.5g}, max={finite.max():.5g}"
        print(f"{indent}ndarray shape={value.shape}, dtype={value.dtype}{stats}")
        if value.size and value.ndim <= 2:
            print(f"{indent}sample={value.reshape(-1)[:8].tolist()}")
        return
    if isinstance(value, (list, tuple)):
        print(f"{indent}{type(value).__name__}[{len(value)}]")
        if value:
            describe(value[0], depth + 1)
        return
    print(f"{indent}{type(value).__name__}: {str(value)[:200]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    with args.path.open("rb") as handle:
        data = pickle.load(handle)
    print(args.path)
    describe(data)


if __name__ == "__main__":
    main()
