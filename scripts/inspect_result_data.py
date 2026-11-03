#!/usr/bin/env python3
"""Summarize supplied published-figure arrays without needing a 3-D solver."""

from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in sorted((ROOT / "results").rglob("*")):
    if path.suffix not in (".npy", ".npz"):
        continue
    print(f"\n{path.relative_to(ROOT)}")
    if path.suffix == ".npy":
        try:
            data = np.load(path, allow_pickle=False)
            print(f"  shape={data.shape}, dtype={data.dtype}")
        except ValueError as exc:
            print("  Legacy object data (not loaded because loading requires pickle):", str(exc))
    else:
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")
