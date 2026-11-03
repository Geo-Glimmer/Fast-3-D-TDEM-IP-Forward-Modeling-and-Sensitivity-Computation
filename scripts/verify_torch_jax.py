#!/usr/bin/env python3
"""Check exact checkpoint tensors and output agreement across PyTorch and JAX."""

import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tem_ip_ipnet.torch_model import IPNet
from tem_ip_ipnet.jax_model import JaxIPNet

state = torch.load(ROOT / "models/ipnet_14poles_100k_torch.pth", map_location="cpu", weights_only=True)
with np.load(ROOT / "models/ipnet_14poles_100k_jax.npz", allow_pickle=False) as archive:
    if set(state) != set(archive.files):
        raise AssertionError("PyTorch and JAX checkpoint key sets differ")
    max_weight_diff = max(float(np.max(np.abs(t.detach().numpy() - archive[k]))) for k, t in state.items())
model_torch = IPNet(n_poles=14, width=512, fourier_size=64, n_blocks=4)
model_torch.load_state_dict(state)
model_torch.eval()
model_jax = JaxIPNet(ROOT / "models/ipnet_14poles_100k_jax.npz")
points = np.array([[-4.0, 0.2], [-2.0, 0.5], [0.0, 0.9]], dtype=np.float32)
with torch.inference_mode():
    tau_pt, w_pt, _, _ = model_torch(torch.tensor(points))
tau_jax, w_jax = model_jax(points)
err_tau = float(np.max(np.abs(tau_pt.numpy() - np.asarray(tau_jax))))
err_w = float(np.max(np.abs(w_pt.numpy() - np.asarray(w_jax))))
print("Identical checkpoint tensor keys:", len(state))
print("Max checkpoint parameter absolute difference:", max_weight_diff)
print("Max tau output absolute difference:", err_tau)
print("Max weight output absolute difference:", err_w)
if max_weight_diff != 0.0 or err_tau > 2e-3 or err_w > 2e-5:
    raise AssertionError("Checkpoint or output mismatch")
print("PASS: identical weights and numerically consistent inference")
