"""Dimensionless Cole-Cole and multi-Debye dispersive kernels.

Both spectra are the same loss-function terms as in the original training
notebook; they are not the complete electromagnetic forward response.
"""

import numpy as np


def cole_cole_kernel(tau, c, frequencies_hz):
    """Calculate 1/[1 + (i 2*pi*f*tau)^c] at the specified frequencies."""
    s = 1j * 2.0 * np.pi * np.asarray(frequencies_hz)
    return 1.0 / (1.0 + (s * tau) ** c)


def multi_debye_kernel(tau_k, weights, frequencies_hz):
    """Calculate sum_k w_k/[1 + i 2*pi*f*tau_k]."""
    poles = np.asarray(tau_k, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if poles.shape != w.shape:
        raise ValueError("Debye times and weights must have identical shapes")
    s = 1j * 2.0 * np.pi * np.asarray(frequencies_hz)
    return np.sum(w[:, None] / (1.0 + poles[:, None] * s[None, :]), axis=0)
