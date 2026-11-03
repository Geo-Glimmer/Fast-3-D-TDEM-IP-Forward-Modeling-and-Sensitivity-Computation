#!/usr/bin/env python3
"""Load the 14-pole trained model and evaluate one Cole-Cole sample."""

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tem_ip_ipnet.jax_model import JaxIPNet
from tem_ip_ipnet.physics import cole_cole_kernel, multi_debye_kernel

model = JaxIPNet(ROOT / "models" / "ipnet_14poles_100k_jax.npz")
print("Detected network architecture:", model.config)

# IMPORTANT: input columns must be [log10(tau in seconds), c].
tau = 0.01
c = 0.5
poles, weights = model(np.array([[np.log10(tau), c]], dtype=np.float32))
poles = np.asarray(poles[0])
weights = np.asarray(weights[0])

frequency = np.logspace(-2, 6, 100)
true_spectrum = cole_cole_kernel(tau, c, frequency)
fit_spectrum = multi_debye_kernel(poles, weights, frequency)
print("Number of Debye poles:", len(poles))
print("Minimum Debye time (s):", float(np.min(poles)))
print("Maximum Debye time (s):", float(np.max(poles)))
print("Weight sum:", float(np.sum(weights)))
print("Complex spectral MAE:", float(np.mean(np.abs(true_spectrum - fit_spectrum))))
print("This is a Debye-spectrum example, NOT a 3-D electromagnetic simulation.")
