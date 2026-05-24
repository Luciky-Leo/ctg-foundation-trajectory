# CTGDL External Validation Plan

Last checked: 2026-05-21.

## Source

CTGDL v5 Zenodo record:

- DOI: https://doi.org/10.5281/zenodo.19510407
- Record: https://zenodo.org/records/19510407
- GitHub: https://github.com/naomifridman/CTGDL

## Terms and Dataset Boundaries

Use CTGDL carefully because the subsets do not all have the same terms.

1. CTU-UHB: ODC-BY-1.0, but it duplicates the primary cohort. Use it only for preprocessing reproducibility checks, not as independent external validation.
2. FHRMA: GPL-3.0 according to the CTGDL record. Use it as an external morphology/representation validation source, especially for deceleration or annotation-related representation checks.
3. SPaM: governed by a Data Use Agreement. Do not include it in automatic download or redistribution. Treat it as a future optional validation set after a separate DUA step.

## Current Downloaded Open Files

Script:

```powershell
.\.venv312\Scripts\python.exe .\scripts\01_download\download_ctgdl_open_files.py
.\.venv312\Scripts\python.exe .\scripts\02_preprocess\inspect_ctgdl_open_files.py
```

Downloaded into:

```text
data/raw/ctgdl/v5_open/
```

Current processed outputs:

```text
data/processed/ctgdl_fhrma_windows.npz
data/processed/ctgdl_fhrma_window_index.csv
results/tables/ctgdl_fhrma_record_embeddings.csv
results/tables/ctgdl_external_embedding_validation.csv
results/figures/ctgdl_external_embedding_pca.png
```

## Validation Uses

### Immediate External Validation

Use FHRMA processed CSV files to test whether the transformer encoder produces stable morphology-related embeddings on non-CTU data.

Candidate analyses:

1. Embed FHRMA processed windows using the CTU-UHB-trained encoder.
2. Compare embedding distribution against CTU-UHB with PCA/UMAP.
3. Test whether SSL dimensions separate annotated morphology patterns if annotation channels are available.
4. Report this as external representation validation, not neonatal outcome validation.

Current first-pass result:

- FHRMA records: 135.
- FHRMA windows: 2604.
- CTU-vs-FHRMA domain classifier AUROC: 0.9088, indicating strong dataset shift.
- FHRMA records map mainly to CTU SSL phenotypes 1 and 3.

### Not Yet Justified

Do not call FHRMA an external neonatal risk validation set unless outcome labels equivalent to CTU-UHB pH/Apgar are verified.

Do not use SPaM files until the DUA pathway is completed.
