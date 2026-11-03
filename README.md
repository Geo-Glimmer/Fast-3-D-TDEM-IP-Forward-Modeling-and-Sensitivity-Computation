# IP-Net: self-supervised Debye decomposition for 3-D TEM-IP modeling

**Status:** curated research-code and result-data release candidate for a *Computers & Geosciences* manuscript. **This release currently contains a reproducible IP-Net component .** 

## Scope

The supplied sources include PyTorch IP-Net training notebooks, neural-network architecture and hyperparameter studies, a Torch-to-JAX weight export, the trained 14-Debye-pole model, and original arrays/images related to manuscript Figures 2–8. The release reorganizes these into runnable core scripts, supplementary research notebooks, pretrained weights, results, figures, and documentation. It does **not** add unavailable electromagnetic modeling source code or fabricate missing experiment configuration files.

**Manuscript context:** fast three-dimensional transient electromagnetic modeling with induced polarization effects (**TEM-IP**), using self-supervised Cole–Cole-to-multi-Debye mapping and matrix-free sensitivities.

### Released reference IP-Net

| Parameter | Value | Basis |
| --- | --- | --- |
| Debye relaxation terms | 14 | Released 14-pole checkpoint / training notebook |
| Fourier feature dimension | 64 | Checkpoint shape and `DL-net-torch-100000.ipynb` |
| Hidden width | 512 | Checkpoint shape and source notebook |
| Residual blocks | 4 | Checkpoint shape and source notebook |
| Input columns | `[log10(tau in s), c]` | Source training data and spectral loss |
| Output | 14 positive Debye relaxation times (s), 14 normalized nonnegative weights | Source network forward operation |
| Training objective | Squared complex spectral difference between Cole–Cole and multi-Debye kernels | Source self-supervised loss |
| Training example count | 100,000 unlabeled parameter pairs | Source training notebook |
| Training parameter domain | `log10(tau/s)` uniform on [-4, 1]; `c` uniform on [0.1, 1] | Source training notebook |
| Training frequencies | 100 log-spaced frequencies, 0.01–1,000,000 Hz | Source training notebook |

The included PyTorch `.pth` checkpoint and JAX `.npz` checkpoint contain **exactly the same 37 tensors**. The JAX code computes the network output only; it is **not** a standalone 3-D TEM-IP forward simulator.

## Install

Requires Python 3.10+ and NumPy/JAX. For CPU demonstrations:

```bash
python -m venv .venv
# Activate your virtual environment as appropriate for your operating system.
python -m pip install -e .
python -m pip install matplotlib
```

For PyTorch training and conversion:

```bash
python -m pip install -e ".[train]"
```

For optional legacy notebooks (including Optuna and vector fitting):

```bash
python -m pip install -r requirements-research.txt
```

**GPU installation:** install a CUDA-compatible version of JAX and (if training) PyTorch following the official framework documentation for your hardware. Do not assume the CPU dependencies above enable CUDA acceleration. The original paper's workstation and its full 3-D solver are not part of this environment test; runtime results can vary across devices and versions.

## Quick start: reproducible IP-Net example

Run from the repository root:

```bash
python examples/quickstart.py
python tests.py
python scripts/verify_torch_jax.py
python examples/plot_spectral_fit.py
python examples/benchmark_ipnet.py
```

- `quickstart.py` loads pretrained **14-pole** weights and outputs relaxation times and weights for one `(tau, c)` pair; computes a spectral approximation error.
- `tests.py` verifies architecture, positive poles, nonnegative normalized weights, and one spectral-fit sanity check.
- `verify_torch_jax.py` compares the original PyTorch and JAX tensors and evaluates a few inputs in both frameworks. Requires PyTorch.
- `plot_spectral_fit.py` generates a **new independent validation plot** under `outputs/`; it does not recreate the original Figure 3 exactly.
- `benchmark_ipnet.py` times JAX network inference on 10,000 parameter pairs **after JIT compilation**, with asynchronous work synchronized. This is NOT a 3-D forward/adjoint timing benchmark.

The input format is always `[[log10(tau_seconds), c], ...]`; for `tau=0.01 s, c=0.5`, supply `[[-2.0, 0.5]]`. Do **not** reverse the input columns, even though some legacy notebook comments may suggest a different order.

## Retraining the IP-Net

The complete reference setup is large (100,000 samples and up to 10,000 training epochs). A minimal installation/training smoke test is:

```bash
python scripts/train_ipnet.py --samples 256 --epochs 2 --batch-size 64 --device cpu
python scripts/convert_weights.py outputs/ipnet_retrained.pth outputs/ipnet_retrained.npz
```

To use the paper's source-notebook training settings:

```bash
python scripts/train_ipnet.py --samples 100000 --epochs 10000 --batch-size 512
```

The script uses unlabeled sampled Cole–Cole inputs with a physics-based complex spectral loss. The saved checkpoint is the best training-loss checkpoint with early stopping; exact retraining trajectories and test timings may differ across hardware/software stacks.

## Repository structure

```text
src/tem_ip_ipnet/            Clean Python modules for IP-Net Torch/JAX + spectral kernels
scripts/                    Training, checkpoint conversion/verification, array inspection
examples/                   Standalone demonstrations and inference benchmarking
models/                     Verified 14-pole pretrained weights
models/experiments/         Original alternative-pole/architecture/training-size checkpoints
notebooks/                  Original research notebooks, regrouped and cleared of outputs
figures/figure_02..08/      Source diagrams, plotted comparisons, and paper-result images
results/figure_04..08/      Uploaded numerical result arrays
results/hyperparameter_search/ Original Optuna study databases
results/legacy/            Legacy serialized array (requires extra caution)
docs/                      Source provenance, data dictionary and reproducibility audit
```

See [`docs/FIGURE_INDEX.md`](docs/FIGURE_INDEX.md) for a mapping between the manuscript figures and files, and [`docs/SOURCE_FILE_MAP.csv`](docs/SOURCE_FILE_MAP.csv) for an exact file-by-file mapping back to the two user-provided archives, with SHA-256 hashes.


