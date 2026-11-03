#!/usr/bin/env python3
"""Measure 10,000-input IP-Net JAX inference time after JIT warm-up."""

from pathlib import Path
import sys
import time
import numpy as np
import jax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tem_ip_ipnet.jax_model import JaxIPNet

model = JaxIPNet(ROOT / "models/ipnet_14poles_100k_jax.npz")
rng = np.random.default_rng(42)
x = np.stack([rng.uniform(-4, 1, 10000), rng.uniform(0.1, 1.0, 10000)], axis=1).astype("float32")
# Compile before measuring; synchronize with JAX's asynchronous execution.
compiled = model(x)
jax.block_until_ready(compiled)
timings = []
for _ in range(5):
    t0 = time.perf_counter()
    outputs = model(x)
    jax.block_until_ready(outputs)
    timings.append(time.perf_counter() - t0)
print("Device:", jax.devices()[0])
print("Inputs:", len(x))
print("Post-JIT timings (s):", [round(s, 6) for s in timings])
print("Median post-JIT time (s):", round(float(np.median(timings)), 6))
print("This benchmark excludes compilation and data-loading overhead.")
