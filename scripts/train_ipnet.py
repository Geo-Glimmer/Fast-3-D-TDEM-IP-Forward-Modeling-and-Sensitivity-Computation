#!/usr/bin/env python3
"""Retrain IP-Net from randomly sampled, unlabeled Cole-Cole parameters.

Defaults follow `DL-net-torch-100000.ipynb`: 100,000 input samples, 14 poles,
64 Fourier frequencies, four residual blocks of width 512, 100 log-spaced
frequencies between 1e-2 and 1e6 Hz, 10,000 epochs with early stopping.
Training is costly; run with --epochs 2 --samples 256 for an installation test.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

try:
    from tem_ip_ipnet.torch_model import IPNet
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tem_ip_ipnet.torch_model import IPNet


def spectral_loss(model_inputs, times, weights, complex_frequencies):
    log_tau = model_inputs[:, :1]
    c = model_inputs[:, 1:2]
    s = complex_frequencies[None, :]
    true = 1.0 / (1.0 + (s * 10.0 ** log_tau) ** c)
    pred = torch.sum(weights[:, None, :] / (1.0 + complex_frequencies[None, :, None] * times[:, None, :]), dim=-1)
    return torch.mean(torch.abs(true - pred) ** 2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=int, default=100000)
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=500)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--output", type=Path, default=Path("outputs/ipnet_retrained.pth"))
    a = p.parse_args()
    device = "cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device)
    if a.samples < 1 or a.epochs < 1 or a.batch_size < 1:
        raise ValueError("samples, epochs and batch size must be positive")
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(a.seed)
    x = np.stack([
        np.random.uniform(-4.0, 1.0, a.samples),
        np.random.uniform(0.1, 1.0, a.samples),
    ], axis=1).astype(np.float32)
    x_tensor = torch.as_tensor(x, device=device)
    batches = DataLoader(TensorDataset(x_tensor), batch_size=a.batch_size, shuffle=True)
    model = IPNet(n_poles=14, width=512, fourier_size=64, n_blocks=4).to(device)
    frequency = torch.logspace(-2, 6, 100, device=device)
    s_bench = 1j * 2.0 * torch.pi * frequency
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs, eta_min=1e-6)
    best_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    for epoch in range(a.epochs):
        model.train()
        loss_acc = 0.0
        for (inputs,) in batches:
            optimizer.zero_grad()
            tau, w, _, _ = model(inputs)
            loss = spectral_loss(inputs, tau, w, s_bench)
            loss.backward()
            optimizer.step()
            loss_acc += float(loss.item())
        avg = loss_acc / len(batches)
        scheduler.step()
        if avg < best_loss:
            best_loss = avg
            best_state = {k: value.detach().cpu().clone() for k, value in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epoch == 0 or (epoch + 1) % 100 == 0 or epoch == a.epochs - 1:
            print(f"epoch={epoch+1}/{a.epochs} spectral_loss={avg:.6e} best={best_loss:.6e}")
        if epochs_no_improve >= a.patience:
            print("Early stopping: no improvement in", a.patience, "epochs")
            break
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, a.output)
    print("Saved best weights:", a.output)
    print("Note: results depend on package versions, device and training randomness.")


if __name__ == "__main__":
    main()
