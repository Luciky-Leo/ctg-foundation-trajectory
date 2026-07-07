# CTG Foundation Trajectory

Code, manuscript source, and submission package for:

Self-supervised biomedical signal representation learning from intrapartum cardiotocography for neonatal risk enrichment

## Overview

This repository contains the reproducible analysis scripts, manuscript source files, and Frontiers submission package for an open-data intrapartum cardiotocography (CTG) biomedical signal representation-learning study. The analysis evaluates whether self-supervised fetal heart rate (FHR) and uterine contraction (UC) trajectory representations can support calibrated neonatal risk enrichment, clinically interpretable dynamic phenotypes and explicit domain-shift assessment.

The repository is intended as a code and source-data archive, not as a deployable clinical device. It does not include raw public datasets, generated model weights, Python virtual environments, or large intermediate tensors.

## Primary Data Strategy

| Role | Dataset | Use |
|---|---|---|
| Primary signal cohort | CTU-UHB / CTU-CHB Intrapartum CTG Database | Raw FHR and UC time-series, clinical outcomes |
| External/harmonized validation | CTGDL | Multi-source processed CTG files; license and DUA checks required for each subset |
| Feature-level baseline | UCI Cardiotocography | Traditional CTG features and expert labels |
| Annotation support | CTU-CHB annotation dataset | Optional expert annotation layer for interpretability analysis |

## Analysis modules

1. Data ingestion and source audit.
2. CTG signal preprocessing and quality control.
3. Conventional CTG feature extraction.
4. Self-supervised representation learning.
5. Dynamic phenotype discovery.
6. Neonatal risk prediction and calibration.
7. Cross-dataset validation.
8. Interpretability, subgroup robustness, and fairness checks.

## Repository layout

```text
config/                  Pipeline configuration
scripts/                 Data download, preprocessing, modelling and reporting scripts
manuscript_overleaf/     Manuscript source, figures, source data and supplementary tables
submission_packages/     Journal submission packages, including the current R1 Frontiers package
reproducibility/         Fixed split files, locked-replay summaries and reviewer-facing sensitivity tables
docs/PROJECT_STRUCTURE.md Detailed active/archive path guide
archive/                 Deprecated experiments retained for traceability
requirements.txt         Minimal Python package list
run_pipeline.ps1         Synthetic smoke-test pipeline
R1_RELEASE_NOTES_20260707.md R1 revision release notes and locked-replay summary
ZENODO_R1_NEW_VERSION_UPLOAD_20260707.md Manual Zenodo new-version upload metadata
```

## Fast smoke test

The current pipeline can run without downloaded public data. It generates synthetic CTG-like traces and checks whether feature extraction, risk scoring, and reporting work.

```powershell
powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1
```

## Python environment

The working deep-learning environment is:

```powershell
.\.venv312\Scripts\python.exe
```

It uses uv-managed CPython 3.12.13 and PyTorch `2.5.1+cpu`. The default Python 3.14 environment installed PyTorch successfully but failed to import `torch` because `c10.dll` initialization failed, so do not use `.venv` for model training.

## Real-data workflow

Download a small CTU-UHB subset:

```powershell
python .\scripts\01_download\download_ctu_uhb.py --limit 20
```

Download the complete CTU-UHB dataset:

```powershell
python .\scripts\01_download\download_ctu_uhb.py --limit 0
```

The complete CTU-UHB raw dataset is about 38 MB uncompressed according to PhysioNet.

Build record-level features after download:

```powershell
python .\scripts\02_preprocess\parse_ctu_uhb_headers.py
python .\scripts\03_features\extract_ctu_uhb_raw_features.py
```

Build self-supervised window tensors:

```powershell
python .\scripts\02_preprocess\build_ctu_uhb_windows.py
```

Train the current masked reconstruction encoder and downstream analyses:

```powershell
.\.venv312\Scripts\python.exe .\scripts\04_models\train_ssl_encoder.py --epochs 15
.\.venv312\Scripts\python.exe .\scripts\04_models\extract_ssl_embeddings.py
.\.venv312\Scripts\python.exe .\scripts\04_models\cluster_phenotypes.py
.\.venv312\Scripts\python.exe .\scripts\04_models\cluster_signal_phenotypes.py
.\.venv312\Scripts\python.exe .\scripts\04_models\evaluate_downstream_tasks.py
```

Train the stronger PatchTST-style masked transformer encoder:

```powershell
.\.venv312\Scripts\python.exe .\scripts\04_models\train_transformer_ssl_encoder.py --epochs 8
.\.venv312\Scripts\python.exe .\scripts\04_models\extract_transformer_embeddings.py
.\.venv312\Scripts\python.exe .\scripts\04_models\cluster_phenotypes.py
.\.venv312\Scripts\python.exe .\scripts\04_models\evaluate_downstream_tasks.py
```

Run a compact transformer sweep:

```powershell
.\.venv312\Scripts\python.exe .\scripts\04_models\run_transformer_sweep.py --epochs 4
```

## Current manuscript-candidate result

The fixed record-level train/test validation used CTU-UHB/CTU-CHB as the primary raw-signal cohort. The primary endpoint was umbilical artery pH <7.15 or 5-minute Apgar score <7.

- Signal features: AUROC 0.6303, AUPRC 0.3366.
- SSL embeddings alone: AUROC 0.4947, AUPRC 0.2125.
- Signal + SSL embeddings: AUROC 0.6025, AUPRC 0.3765.
- Signal + SSL + signal phenotype manuscript candidate: AUROC 0.6168, AUPRC 0.4016.
- Development-only model-selection sensitivity selected a smaller transformer configuration with held-out AUPRC 0.3872.
- SSL PCA sensitivity retained similar enrichment with fewer predictors: PCA10 AUPRC 0.4241 and PCA20 AUPRC 0.4256.

These values correspond to the fixed-seed locked replay used for the R1 revision. The originally submitted unseeded values (signal + SSL AUPRC 0.4432; final AUPRC 0.4324) were not retained because the exact run could not be reproduced. Bootstrap AUPRC-difference intervals versus signal-only features crossed zero, so SSL is framed as an interpretable representation, phenotype-enrichment and recalibration layer rather than as a stable discrimination-improvement result.

Interpretation: the model is framed as a retrospective risk-enrichment and phenotyping workflow, not as a clinically deployable diagnostic predictor.

## Current Best Transformer Sweep

The best compact sweep result so far is:

```text
patch_len=80
d_model=64
layers=3
mask_strategy=channel
contrastive_weight=0.2
epochs=4
```

Validation:

- signal + SSL embeddings: AUROC 0.6025, AUPRC 0.3765.
- signal + SSL + signal phenotype: AUROC 0.6168, AUPRC 0.4016.

The selected architecture's 12-epoch refit is reported for training-dynamics inspection only; downstream representation quality was judged from fixed held-out embeddings, phenotype structure and reviewer-requested sensitivity analyses.

## CTGDL External Representation Validation

CTGDL FHRMA processed files have been downloaded and embedded using the current best transformer. This is representation/domain-shift validation, not neonatal outcome validation.

- CTGDL FHRMA records: 135.
- CTGDL FHRMA windows: 2604.
- CTU-vs-FHRMA embedding domain classifier AUROC: 0.9088.

Interpretation: external FHRMA data show substantial dataset shift, which is important evidence for the paper's robustness section.

## Manuscript and reporting files

The manuscript source package is in `manuscript_overleaf/`. The current R1 Frontiers package is `submission_packages/frontiers_signal_processing_r1_20260707/`, including the official Frontiers LaTeX template, compiled manuscript PDF, compiled supplementary PDF, separate Reviewer 1/Reviewer 2 response PDFs, source-data CSV files, supplementary tables S1-S11 and standalone supplementary-table PDFs. The previous first-submission package is retained under `submission_packages/frontiers_signal_processing_20260601/` for traceability.

## Code archive

[![DOI](https://zenodo.org/badge/1248137386.svg)](https://doi.org/10.5281/zenodo.20364141)

Repository URL: https://github.com/Luciky-Leo/ctg-foundation-trajectory

Latest published Zenodo version DOI before the R1 update: https://doi.org/10.5281/zenodo.20485068

Zenodo concept DOI for all versions: https://doi.org/10.5281/zenodo.20364141

This repository includes `.zenodo.json` and `CITATION.cff` metadata for Zenodo/GitHub release archiving.

## License

The analysis code is released under the MIT License. Public source datasets remain governed by their original database licenses and access terms.
