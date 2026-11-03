# Reproducibility audit (based on files in the two uploaded archives)

This audit distinguishes **source-code reproduction**, **saved-result inspection**, and **visual material**. It is not an independent validation of the paper's reported 3-D accuracy or GPU performance.

| Manuscript result | Uploaded materials | What can currently be done | Critical missing elements |
| --- | --- | --- | --- |
| IP-Net architecture, 14-pole inference | Original PyTorch/JAX notebooks, model weights | Load model, verify weights, run spectral inference and a new 10,000-point network-only benchmark | Exact original dependency lockfile; original GPU compilation/timing protocol |
| Training sample-size and hyperparameter study (Fig. 2) | Training-size notebooks, Optuna notebook and databases, figures | Inspect original notebooks and saved plots; optionally retrain / rerun subset | Matched original GPU software environment, saved random trial conditions across all runs |
| Cole–Cole spectral approximation (Fig. 3) | Four saved images and comparison notebooks | Generate new Cole–Cole vs IP-Net spectra; compare against stored images | Original figure-specific complete regression outputs and exact LS/VF runtime setup |
| Four source configurations (Fig. 4) | Four paired arrays (`jax`, `simpeg`) and four comparison plots | Inspect saved comparison arrays and images | Full JAX solver, survey/mesh/time-gate/configuration scripts and 1-D reference setup |
| 1-D vs 3-D sensitivity (Fig. 5) | Comparison figure; NPZ with different stored array lengths | Inspect saved figure, array shape and values | Matching axis interpretation, grids, aggregation, reference construction and adjoint solver |
| Sulfide-body survey/model (Fig. 6) | Two static images | View the geometry/model illustrations | Machine-readable model parameter grids, survey positions, mesh and simulator |
| Sulfide-body decay and line profiles (Fig. 7) | Two plots and two 187×41 response arrays | Inspect images and array statistics | Full coordinate/time-gate metadata, source configuration and forward solver |
| 3-D sensitivity maps (Fig. 8) | Four images and one flattened gradient array | Inspect images and gradient values | Parameter-to-array mapping, 3-D mesh/active-cell indexing and adjoint solver |
| Forward/adjoint mesh-size scaling (Table 1) | Timings in manuscript, not separate benchmark source files | Compare reported timings as reported in the manuscript | Benchmark harness, JAX compilation/synchronization policy, saved logs/configs and full solver |

## Objective verification carried out on this release

- The reference checkpoint contains 14 poles, width 512, Fourier size 64, four residual blocks.
- PyTorch and JAX checkpoint parameter keys and 37 tensors are exactly equal (maximum absolute checkpoint difference = 0).
- A JAX inference smoke test, normalized-weight check, standalone plotting example and a minimal PyTorch training/NPZ-export check completed successfully in the packaging environment.
- The JAX/PyTorch forward outputs agree numerically for tested synthetic inputs (floating-point differences possible).
- The original NPZ and NPY data files are preserved byte-for-byte and mapped back to their inputs in `SOURCE_FILE_MAP.csv`.
- The uploaded Figure 5 `.npz` includes a `jax` vector with 249,104 elements and a `simpeg` vector with 12 elements; **do not compare element by element** without the missing aggregation and matching metadata.
- The `p_10` folder contains a file named `jax_ip_model_14poles.npz` whose shape is indeed a 14-pole, 6-residual-block model; this is kept in `models/experiments/pole_count/10/` for provenance, **not** installed as the primary 10-pole checkpoint.
- The `p_6` folder also contains an 8-pole export; it is retained under its original experiment folder rather than silently relabeled as a 6-pole network.
- A legacy `.npy` contains an object array and would need `allow_pickle=True`; it is retained only as a research artifact and is not loaded by the provided scripts.

## What is NOT verified

- No original 3-D FDTD or adjoint computations were rerun: their source code was not provided.
- No exact paper Figure 4–8 regeneration has been verified from executable solver code.
- No claim of matching the paper's GPU timings has been made. JAX inference tests in the packaging environment used CPU, excluding JIT compilation from the measured steady-state timings.
- Other legacy variant checkpoints and notebooks have been inventoried but not all retrained or individually executed.

The main item preventing classification as a **full-paper Code availability release** is the missing original JAX FDTD / discrete-adjoint implementation and the associated executable survey/mesh/model/benchmark settings.

Additional checkpoint cross-checks from `MODEL_REGISTRY.csv`: the file `models/experiments/pole_count/06/model_trained.pth` contains a 16-pole, five-residual-block network even though its experiment folder is named `p_6`. The `models/experiments/training_size/model_trained-15000.pth` file is a 10-pole, two-residual-block network, not the 14-pole reference architecture. These variants remain in their original experimental categories and must not be substituted for the primary trained model without verifying compatibility.
