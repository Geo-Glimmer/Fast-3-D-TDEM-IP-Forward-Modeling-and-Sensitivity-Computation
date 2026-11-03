# Packaging-environment verification (not the original paper hardware)

| Item | Value |
| --- | --- |
| Operating system | Linux, CPU verification |
| Python | 3.13.5 |
| numpy | 2.3.5 |
| jax | 0.9.0.1 |
| jaxlib | 0.9.0.1 |
| torch | 2.10.0+cpu |
| scipy | 1.17.0 |
| matplotlib | 3.10.8 |
| optuna | 4.8.0 |
| pandas | 2.2.3 |
| scikit-rf | not installed |

Validation performed here includes 14-pole JAX inference, checkpoint parity, spectral sanity checks, short PyTorch training and checkpoint conversion. No claim is made that these are the original experimental package versions or that the full paper's GPU performance was reproduced. See `REPRODUCIBILITY_STATUS.md`.
