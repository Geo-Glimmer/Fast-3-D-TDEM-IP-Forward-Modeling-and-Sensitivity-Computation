# How-to: typical public-repository use cases

## A. Check the pretrained 14-pole IP-Net

```bash
python -m pip install -e .
python examples/quickstart.py
python tests.py
```

Expect a reported 14-pole architecture, nonnegative weights summing to ~1, and a finite spectral approximation error. The example is a 1-point spectral test; it is not a 3-D TEM-IP field response.

## B. Verify PyTorch-to-JAX conversion

```bash
python -m pip install -e ".[train]"
python scripts/verify_torch_jax.py
```

For the provided primary checkpoint, the 37 parameter tensors are identical. Small floating-point differences in the final outputs are expected because PyTorch and JAX may evaluate operations differently.

## C. Create a new spectral fit figure

```bash
python examples/plot_spectral_fit.py
```

The output appears in `outputs/ipnet_spectral_validation.png`. This is an independent demonstration (not an exact recreation of paper Figure 3); the original uploaded figures are retained under `figures/figure_03/`.

## D. Run a brief self-supervised training demonstration

```bash
python -m pip install -e ".[train]"
python scripts/train_ipnet.py --samples 256 --epochs 2 --batch-size 64 --device cpu
python scripts/convert_weights.py outputs/ipnet_retrained.pth outputs/ipnet_retrained.npz
```

A short run is a software check only, not the published high-accuracy model. For the full reference training settings, see `README.md`.

## E. Inspect numerical arrays associated with manuscript figures

```bash
python scripts/inspect_result_data.py
```

See `docs/DATA_DICTIONARY.csv` for the stored array keys and shapes. Figures 4–8 require additional original source code/metadata for end-to-end re-simulation; the saved images are viewable in `figures/`.
