#!/usr/bin/env python3
"""Safely export an IP-Net PyTorch state_dict to a plain JAX-compatible NPZ."""

import argparse
from pathlib import Path
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("Expected a plain state_dict[str, Tensor]")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, **{key: value.detach().cpu().numpy() for key, value in state.items()})
    print(f"Saved {len(state)} arrays to {a.output}")


if __name__ == "__main__":
    main()
