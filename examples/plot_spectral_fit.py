#!/usr/bin/env python3
"""Plot dimensionless complex Cole-Cole and IP-Net Debye spectra."""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tem_ip_ipnet.jax_model import JaxIPNet
from tem_ip_ipnet.physics import cole_cole_kernel, multi_debye_kernel

model = JaxIPNet(ROOT / "models/ipnet_14poles_100k_jax.npz")
cases = [(0.01, 0.2), (0.01, 0.5), (0.01, 0.8), (0.001, 0.4)]
freq = np.logspace(-2, 6, 100)
fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
for ax, (tau, c) in zip(axes.flat, cases):
    tk, w = model(np.array([[np.log10(tau), c]], dtype=np.float32))
    reference = cole_cole_kernel(tau, c, freq)
    reconstructed = multi_debye_kernel(np.asarray(tk[0]), np.asarray(w[0]), freq)
    ax.semilogx(freq, reference.real, "-", label="Cole-Cole: Re")
    ax.semilogx(freq, reconstructed.real, "--", label="Debye: Re")
    ax.semilogx(freq, reference.imag, "-", label="Cole-Cole: Im")
    ax.semilogx(freq, reconstructed.imag, "--", label="Debye: Im")
    mae = np.mean(np.abs(reference - reconstructed))
    ax.set_title(f"tau={tau:g} s, c={c:g}, MAE={mae:.3g}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Dimensionless spectrum")
    ax.grid(True, alpha=0.3)
axes.flat[0].legend(fontsize=8)
output = ROOT / "outputs/ipnet_spectral_validation.png"
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=180)
print("Wrote:", output)
