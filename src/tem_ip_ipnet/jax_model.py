"""JAX inference for the released IP-Net `.npz` parameter arrays.

This is a standalone extraction of the PureJAXIPNet forward operation in the
supplied conversion notebook, with architecture automatically read from weights.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import jax.nn as jnn
import numpy as np


def load_parameters(path):
    """Load a plain numeric NPZ, never Python pickle/object arrays."""
    with np.load(Path(path), allow_pickle=False) as archive:
        return {key: jnp.asarray(archive[key], dtype=jnp.float32) for key in archive.files}


def architecture(parameters):
    """Inspect actual weight shapes to infer (poles, width, Fourier size, blocks)."""
    size = int(parameters["out_layer.bias"].shape[0])
    if size % 2:
        raise ValueError("Output weights must contain equal time/weight halves")
    pole_count = size // 2
    width = int(parameters["in_layer.bias"].shape[0])
    fourier = int(parameters["fourier_map.B"].shape[1])
    blocks = sorted({int(k.split(".")[1]) for k in parameters if k.startswith("res_layers.")})
    if blocks != list(range(len(blocks))):
        raise ValueError("Residual blocks must be contiguous and numbered from zero")
    if int(parameters["in_layer.weight"].shape[1]) != 2 * fourier + 2:
        raise ValueError("Unexpected input projection shape")
    return {"n_poles": pole_count, "width": width, "fourier_size": fourier, "n_blocks": len(blocks)}


def _layer_norm(x, weight, bias, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * weight + bias


def _forward(params, x, n_poles: int, n_blocks: int):
    fourier = (2.0 * jnp.pi * x) @ params["fourier_map.B"]
    h0 = jnp.concatenate([x, jnp.sin(fourier), jnp.cos(fourier)], axis=-1)
    h = jnn.silu(h0 @ params["in_layer.weight"].T + params["in_layer.bias"])
    for i in range(n_blocks):
        prefix = f"res_layers.{i}"
        residual = h
        norm1 = _layer_norm(h, params[f"{prefix}.norm1.weight"], params[f"{prefix}.norm1.bias"])
        h = jnn.silu(norm1 @ params[f"{prefix}.fc1.weight"].T + params[f"{prefix}.fc1.bias"])
        norm2 = _layer_norm(h, params[f"{prefix}.norm2.weight"], params[f"{prefix}.norm2.bias"])
        h = jnn.silu(norm2 @ params[f"{prefix}.fc2.weight"].T + params[f"{prefix}.fc2.bias"])
        h = h + residual
    out = h @ params["out_layer.weight"].T + params["out_layer.bias"]
    tau = 10.0 ** out[:, :n_poles]
    raw = 10.0 ** out[:, n_poles:]
    weights = raw / (jnp.sum(raw, axis=-1, keepdims=True) + 1e-12)
    return tau, weights


class JaxIPNet:
    """Load and evaluate the saved IP-Net using JAX on CPU or GPU."""

    def __init__(self, npz_file):
        self.params = load_parameters(npz_file)
        self.config = architecture(self.params)
        self._run = jax.jit(_forward, static_argnames=("n_poles", "n_blocks"))

    def __call__(self, inputs):
        inputs = jnp.asarray(inputs, dtype=jnp.float32)
        if inputs.ndim != 2 or inputs.shape[1] != 2:
            raise ValueError("Inputs must be (batch,2) in [log10(tau), c] order")
        return self._run(self.params, inputs, self.config["n_poles"], self.config["n_blocks"])
