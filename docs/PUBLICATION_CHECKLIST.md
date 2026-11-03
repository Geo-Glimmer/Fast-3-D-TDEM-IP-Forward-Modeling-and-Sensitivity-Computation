# Final actions before submission as a public *Computers & Geosciences* repository

- [ ] Confirm author ownership/permissions for the released code and figures; approve an actual OSI-compatible license, replace `LICENSE.template` with `LICENSE`, and identify the actual copyright holder. This package **does not assign a license on your behalf**.
- [ ] Publish all folders and files in a public Git repository (e.g. GitHub). **Do not upload only the transfer ZIP**. Verify reviewers can access it without authentication.
- [ ] Place the permanent repository URL in the paper's `Code availability` section; optionally preserve an exact version in Zenodo or another citable archive and provide a DOI.
- [ ] Publish the original JAX 3-D TEM-IP FDTD solver and discrete-adjoint sensitivity code, or disclose and justify its absence. Without that, this release does **not** fully reproduce the study's principal 3-D claims and could fail the journal's open-code criteria.
- [ ] Supply at least one machine-readable 3-D synthetic model, survey geometries (transmitters/receivers), mesh and active-cell masks, time gates, physics settings, weighting rules, initial-field settings, and commands to regenerate principal results.
- [ ] Add full numerical test/benchmark scripts (forward vs 1-D references, adjoint/Jacobian products, gradient tests and timing protocol). Include hardware information, JAX/torch/CUDA versions, JIT warm-up and `block_until_ready` synchronization policy.
- [ ] Investigate Figure 5's nonmatching array lengths and add documented, reproducible layer aggregation and comparison procedures.
- [ ] Verify every manuscript figure and data panel maps to an accurate dataset and source script; inspect scientific axis labels and units, and consider final English-label figures.
- [ ] Inspect and run all notebooks intended as public tutorials; legacy notebooks preserve research history but are **not** guaranteed portable or runnable in the relocated directory structure.
- [ ] Decide whether the legacy object `.npy`, duplicated experimental weights, and raw Optuna database files should be kept public; confirm no proprietary information or sensitive research details are inadvertently disclosed.
- [ ] Verify documentation for licensed third-party components and model/data usage, package installation on a second environment, and actual numerical reproducibility before describing the project as submission-ready.

## Current verified result

The released 14-pole IP-Net model, weight compatibility tests, simple spectral inference and miniature training test work in a CPU environment. The publication blockers above are author decisions or missing original research materials, not items that can be inferred or recreated reliably from the saved figures alone.
