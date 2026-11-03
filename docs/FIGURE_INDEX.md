# Figure and experiment index

File-to-figure grouping is derived from the uploaded archive directory names and the scientific content of the images. It is not a claim that each image exactly matches the final journal figure panel or publishing layout. All provenance is recorded in `SOURCE_FILE_MAP.csv`.

| Figure or analysis | Images | Numeric source(s) | Related research notebooks |
| --- | --- | --- | --- |
| Fig. 1 — overall TEM-IP workflow | Not supplied | Not supplied | Not supplied in these archives |
| Fig. 2 — IP-Net architecture, training-size and hyperparameter studies | `figures/figure_02/{network_architecture,training_sample_error_plot,optuna_trials_curve}.png` | `results/hyperparameter_search/optuna_history.db*`; model variants | `notebooks/hyperparameter_search/`, `notebooks/training_size/` |
| Fig. 3 — Cole–Cole vs LS/VF/IP-Net spectral comparisons | `figures/figure_03/spectral_fit_case_{1..4}.png` | Original notebook calculations; no separate numeric output was supplied | `notebooks/spectral_comparison/` |
| Fig. 4 — four acquisition configurations and 3-D/1-D response comparison | `figures/figure_04/*.png` | `results/figure_04/*.npz` | No corresponding 3-D solver notebooks supplied |
| Fig. 5 — 3-D/1-D sensitivity comparison | `figures/figure_05/sensitivity_comparison.png` | `results/figure_05/Jvec_compare_magnetic_loop.npz` | No adjoint solver notebook supplied |
| Fig. 6 — 3-D sulfide-body model and slices | `figures/figure_06/*.png` | No machine-readable model geometry supplied | No 3-D model generation script supplied |
| Fig. 7 — site and survey-line responses | `figures/figure_07/*.png` | `results/figure_07/{ip_data,no_ip_data}.npy` | No 3-D forward solver supplied |
| Fig. 8 — 3-D sensitivity map | `figures/figure_08/{rho,eta,tau,c}.png` | `results/figure_08/grad.npy` | No 3-D adjoint solver supplied |
| Table 1 — runtimes at different meshes | Not supplied in archives | No timing table CSV or raw benchmark logs | No timing runner supplied |

The original `.drawio` file for the network structure is in `figures/figure_02/network_architecture_source.drawio`. The 4-pole/6-pole/8-pole/10-pole/12-pole/14-pole experiment names are not sufficient to infer true architecture: actual pole count is determined from model checkpoint tensor shapes where possible.
