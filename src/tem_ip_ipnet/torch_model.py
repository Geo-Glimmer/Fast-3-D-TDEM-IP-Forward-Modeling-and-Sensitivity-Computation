"""PyTorch IP-Net architecture reconstructed from the supplied training notebook.

The reference configuration uses 14 Debye poles, four residual blocks, a
512-unit hidden layer, and 64 Fourier frequencies. Its original source is
`notebooks/training/DL-net-torch-100000.ipynb`.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class FourierFeatureMapping(nn.Module):
    def __init__(self, in_dims=2, mapping_size=64, scale=5.0):
        super().__init__()
        self.register_buffer("B", torch.randn(in_dims, mapping_size) * scale)

    def forward(self, x):
        projection = (2.0 * math.pi * x) @ self.B
        return torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)


class DenseResBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)

    def forward(self, x):
        residual = x
        x = F.silu(self.fc1(self.norm1(x)))
        x = F.silu(self.fc2(self.norm2(x)))
        return x + residual


class IPNet(nn.Module):
    """Map [log10(tau), c] to positive Debye times and normalized weights."""

    def __init__(self, n_poles=14, width=512, fourier_size=64, n_blocks=4):
        super().__init__()
        self.n_poles = n_poles
        self.fourier_map = FourierFeatureMapping(2, fourier_size, scale=5.0)
        self.in_layer = nn.Linear(fourier_size * 2 + 2, width)
        self.res_layers = nn.ModuleList([DenseResBlock(width) for _ in range(n_blocks)])
        self.out_layer = nn.Linear(width, 2 * n_poles)

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError("Expected input shape (batch, 2): [log10(tau), c]")
        fourier = self.fourier_map(x)
        h = F.silu(self.in_layer(torch.cat([x, fourier], dim=-1)))
        for block in self.res_layers:
            h = block(h)
        out = self.out_layer(h)
        log_tau_k, log_w_k = out[:, : self.n_poles], out[:, self.n_poles :]
        tau_k = 10.0 ** log_tau_k
        raw_w = 10.0 ** log_w_k
        w_k = raw_w / (torch.sum(raw_w, dim=1, keepdim=True) + 1e-12)
        return tau_k, w_k, log_tau_k, log_w_k
