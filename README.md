# IP-Net: Self-supervised Debye decomposition for 3-D TEM-IP modeling

**Status:** Research code, pretrained models, and reproducible results associated with a *Computers & Geosciences* manuscript.

This repository provides the IP-Net implementation, pretrained network weights, training and validation workflows, numerical results, and figures supporting fast three-dimensional transient electromagnetic modeling with induced polarization effects (TEM-IP).

## Scope

The repository includes PyTorch IP-Net training notebooks, neural-network architecture and hyperparameter studies, Torch-to-JAX weight conversion, the pretrained 14-term Debye model, and numerical arrays and images associated with manuscript Figures 2–8.

The research materials are organized into reusable Python modules, executable scripts, supplementary notebooks, pretrained weights, numerical results, figures, and documentation.

**Manuscript context:** Fast three-dimensional transient electromagnetic modeling with induced polarization effects (TEM-IP), using self-supervised Cole–Cole-to-multi-Debye mapping and matrix-free sensitivity calculations.

### Reference IP-Net configuration

| Parameter | Value | Basis |
| --- | --- | --- |
| Debye relaxation terms | 14 | Pretrained checkpoint and training notebook |
| Fourier feature dimension | 64 | Checkpoint and `DL-net-torch-100000.ipynb` |
| Hidden width | 512 | Checkpoint and source notebook |
| Residual blocks | 4 | Checkpoint and source notebook |
| Input parameters | `[log10(tau in s), c]` | Training data and spectral loss |
| Network output | 14 positive relaxation times (s) and 14 normalized nonnegative weights | Network implementation |
| Training objective | Squared complex spectral difference between Cole–Cole and multi-Debye kernels | Self-supervised spectral loss |
| Training samples | 100,000 unlabeled parameter pairs | Source training notebook |
| Training parameter domain | `log10(tau/s)` uniform on [-4, 1], `c` uniform on [0.1, 1] | Source training notebook |
| Training frequencies | 100 logarithmically spaced frequencies from 0.01 to 1,000,000 Hz | Source training notebook |

The pretrained PyTorch (`.pth`) and JAX (`.npz`) checkpoints contain the same 37 parameter tensors, enabling consistent network inference across both frameworks.

## Installation

Requires Python 3.10+ and NumPy/JAX.

For CPU demonstrations:

```bash
python -m venv .venv
# Activate the virtual environment for your operating system.
python -m pip install -e .
python -m pip install matplotlib
```

For PyTorch training and weight conversion:

```bash
python -m pip install -e ".[train]"
```

For optional research notebooks, including Optuna and vector fitting:

```bash
python -m pip install -r requirements-research.txt
```

**GPU installation:** Install CUDA-compatible versions of JAX and PyTorch according to their official documentation. Computational performance depends on hardware, framework versions, and execution settings.

## Quick start

Run the following commands from the repository root:

```bash
python examples/quickstart.py
python tests.py
python scripts/verify_torch_jax.py
python examples/plot_spectral_fit.py
python examples/benchmark_ipnet.py
```

The scripts provide the following functionality:

- `quickstart.py`: Loads pretrained 14-term Debye weights, predicts relaxation times and corresponding weights for a specified `(tau, c)` pair, and evaluates the spectral approximation error.
- `tests.py`: Validates network architecture, positive relaxation times, nonnegative normalized weights, and spectral-fitting consistency.
- `verify_torch_jax.py`: Compares PyTorch and JAX parameter tensors and evaluates cross-framework inference consistency. Requires PyTorch.
- `plot_spectral_fit.py`: Generates an independent spectral validation figure under `outputs/`.
- `benchmark_ipnet.py`: Evaluates JAX inference performance on 10,000 parameter pairs after JIT compilation, with explicit synchronization for accurate timing.

### Input format

The IP-Net input consists of two parameters:

```python
[[log10(tau_seconds), c], ...]
```

For example, given `tau = 0.01 s` and `c = 0.5`:

```python
inputs = [[-2.0, 0.5]]
```

The input parameter order must remain `[log10(tau_seconds), c]`.

## Retraining IP-Net

IP-Net uses self-supervised training based on the spectral consistency between Cole–Cole and multi-Debye representations. Training samples are generated from the specified Cole–Cole parameter domain without requiring labeled Debye decomposition targets.

### Minimal training example

For a CPU installation and training test:

```bash
python scripts/train_ipnet.py --samples 256 --epochs 2 --batch-size 64 --device cpu
python scripts/convert_weights.py outputs/ipnet_retrained.pth outputs/ipnet_retrained.npz
```

### Reference training configuration

To reproduce the training configuration used in the source notebook:

```bash
python scripts/train_ipnet.py --samples 100000 --epochs 10000 --batch-size 512
```

The training procedure minimizes the complex spectral discrepancy between the Cole–Cole response and its multi-Debye approximation.

The best-performing checkpoint is selected according to training loss, with early stopping supported by the training implementation. Exact training trajectories may vary with random initialization, numerical precision, and software or hardware configurations.

## Repository structure

```text
src/tem_ip_ipnet/
    IP-Net implementations and spectral kernels
    PyTorch and JAX inference modules

scripts/
    Training and checkpoint conversion
    Cross-framework verification
    Numerical array inspection

examples/
    IP-Net inference demonstrations
    Spectral approximation validation
    Network inference benchmarks

models/
    Pretrained 14-term Debye model weights

models/experiments/
    Alternative network architectures
    Debye-term configurations
    Training-size experiments

notebooks/
    Original research notebooks
    Architecture and hyperparameter studies

figures/figure_02..08/
    Manuscript figures and source images
    Network and numerical comparison figures

results/figure_04..08/
    Numerical result arrays
    Data supporting manuscript figures

results/hyperparameter_search/
    Original Optuna study databases

results/legacy/
    Legacy serialized research arrays

docs/
    Source provenance
    Figure-to-file mapping
    Data dictionary
    Reproducibility audit
```

## Manuscript figures and numerical results

The repository preserves the original numerical arrays and figure materials associated with manuscript Figures 2–8.

The corresponding files are indexed in:

- [`docs/FIGURE_INDEX.md`](docs/FIGURE_INDEX.md): Mapping between manuscript figures and their source files.
- [`docs/SOURCE_FILE_MAP.csv`](docs/SOURCE_FILE_MAP.csv): File-level provenance mapping to the original research archives, including SHA-256 hashes.

These records facilitate verification of numerical results, inspection of source data, and reproduction of the associated IP-Net experiments.

## Reproducibility

The repository supports reproducibility through documented network configurations, pretrained model parameters, executable examples, cross-framework weight verification, and archived numerical results.

The supplied scripts enable independent evaluation of spectral approximation accuracy, network inference consistency, and computational performance under different execution environments.
