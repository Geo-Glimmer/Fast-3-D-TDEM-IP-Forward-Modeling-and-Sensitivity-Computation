# Manuscript Code availability text (to finalize after GitHub publication)

**Current release scope — accurate wording (do not claim full solver availability):**

> **Code availability.** The source code for the self-supervised IP-Net Debye-parameter mapping, including PyTorch training, JAX inference, pretrained network weights, and spectral validation examples, is available from [PUBLIC_GITHUB_REPOSITORY_URL]. Supporting numerical result arrays and figures for the TEM-IP forward-modeling and sensitivity experiments are also provided in the repository. The complete three-dimensional electromagnetic forward and discrete-adjoint sensitivity solver source used to generate those results is not included in the current release.

**Caution:** The above reflects the supplied files but may NOT satisfy *Computers & Geosciences* requirements for the paper's full 3-D forward/sensitivity claims. This paragraph cannot be used until the repository is actually public and a license has been assigned by the rights holder.

**Alternative text ONLY after the full solver, benchmarks, and example configurations have genuinely been released and tested:**

> **Code availability.** The source code for IP-Net, the JAX-accelerated 3-D TEM-IP forward solver, and matrix-free sensitivity computation is publicly available at [PUBLIC_GITHUB_REPOSITORY_URL] ([ARCHIVED_RELEASE_DOI], if applicable). The repository includes pretrained network weights, model configurations, example datasets, reproducibility scripts, and benchmark instructions for the principal results reported in this study. The code is released under [APPROVED_OPEN_SOURCE_LICENSE].

Only use the second version after verifying the named materials exist publicly. No URL, DOI, author name, copyright owner, or license has been invented in this package.
