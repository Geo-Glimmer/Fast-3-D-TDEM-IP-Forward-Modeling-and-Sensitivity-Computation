# User guide: core IP-Net inference and training

The repository provides IP-Net only, not a complete 3-D electromagnetic simulator. To run commands below, use the repository root as your working directory.

## Inputs

A batch of dimensionless feature pairs with shape `(n, 2)` in **this exact order**:

1. `log10(tau_seconds)`, the base-10 logarithm of the Cole–Cole characteristic relaxation time expressed in seconds;
2. `c`, the Cole–Cole frequency exponent.

Reference training domain: `log10(tau_seconds)` in [-4, 1], `c` in [0.1, 1]. The model may extrapolate poorly outside that domain. It is self-supervised: no target Debye times or weights are needed for retraining.

## Outputs

`JaxIPNet(batch)` returns `(debye_tau_seconds, debye_weights)` as JAX arrays of shape `(n, 14)` for the primary model. `debye_tau_seconds > 0`; the nonnegative weights sum to approximately 1 within floating-point precision. These correspond to the normalized dimensionless spectral kernel in `src/tem_ip_ipnet/physics.py`, not to a complete Cole–Cole conductivity model with baseline conductivity or resistivity.

## Model setup

The main checkpoint is `models/ipnet_14poles_100k_jax.npz`. The Torch state_dict with matching tensor values is `models/ipnet_14poles_100k_torch.pth`. Network architecture is inferred from the JAX arrays; the default Torch network configuration is explicitly specified in `torch_model.IPNet`.

```python
import numpy as np
from tem_ip_ipnet.jax_model import JaxIPNet
model = JaxIPNet('models/ipnet_14poles_100k_jax.npz')
poles, weights = model(np.asarray([[-2.0, 0.5]], dtype='float32'))
```

## Main command-line programs

| Command | Inputs/options | Outputs/expected behavior |
| --- | --- | --- |
| `python examples/quickstart.py` | Fixed synthetic `(tau=0.01 s, c=0.5)` | Prints model architecture, output positivity/normalization and spectral mean absolute error |
| `python examples/plot_spectral_fit.py` | Four synthetic `(tau,c)` combinations | Writes a new spectral comparison image to `outputs/ipnet_spectral_validation.png` |
| `python examples/benchmark_ipnet.py` | 10,000 synthetic pairs, fixed RNG seed 42 | Prints first available JAX device, per-repeat times and median post-JIT inference time; does not include compilation |
| `python scripts/train_ipnet.py` | `--samples`, `--epochs`, `--batch-size`, `--seed`, `--patience`, `--device`, `--output` | Runs physics-constrained supervised-target-free training; writes best PyTorch `.pth` weights |
| `python scripts/convert_weights.py INPUT.pth OUTPUT.npz` | Plain Torch `state_dict` | Writes a numeric NPZ usable by `JaxIPNet`, with no pickled data |
| `python scripts/verify_torch_jax.py` | Included primary Torch and JAX checkpoints | Verifies tensor matching and comparable outputs for test samples |
| `python scripts/inspect_result_data.py` | All `results/` NPZ/NPY files | Prints array keys, shapes and dtypes; does not invoke or recreate the 3-D solver |
| `python tests.py` | Included pretrained JAX model | Runs deterministic structural and spectral sanity tests |

## Scientific conventions

The training spectral target is `F_CC(s) = 1/(1+(s*tau)^c)` for `s=i*2*pi*f` and the predicted Debye approximation is `F_D(s) = sum_k w_k/(1+s*tau_k)`. The actual frequency-dependent conductivity in the manuscript also depends on baseline conductivity and chargeability; these functions deliberately represent the **normalized dispersive kernel** only.

## Known limitations

- The paper's multi-Debye time-domain auxiliary state updates, GPU FDTD solver, adjoint integration and full 3-D mesh/time scheme are **not implemented in this release** because the relevant source code was not in the supplied archives.
- The stored 3-D arrays are incomplete without numerical mesh, mapping, survey and time metadata and cannot be passed directly to `JaxIPNet` as a 3-D forward model.
- JAX and PyTorch execution environments must be selected and tested separately for GPUs; CPU demonstration performance is not a reproduction of paper GPU benchmarks.
- Legacy research notebooks are retained with their experimental structure. A number have historical configuration mismatches or machine-specific assumptions, so start with the supported scripts above.
