# CTG Foundation Trajectory

Code and manuscript package for:

Self-supervised dynamic cardiotocography phenotyping for calibrated neonatal risk enrichment: an open-data decision-modelling study

## Overview

This repository contains the reproducible analysis scripts and manuscript source files for an open-data intrapartum cardiotocography (CTG) study. The analysis evaluates whether self-supervised fetal heart rate (FHR) and uterine contraction (UC) representations can support calibrated neonatal risk enrichment and clinically interpretable dynamic phenotypes.

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
manuscript_overleaf/     Final Springer Nature / BMC-ready LaTeX package and figure source data
reproducibility/         Fixed split files and reviewer-facing sensitivity tables
requirements.txt         Minimal Python package list
run_pipeline.ps1         Synthetic smoke-test pipeline
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
- Signal + SSL embeddings: AUROC 0.6482, AUPRC 0.4432.
- Signal + SSL + signal phenotype manuscript candidate: AUROC 0.6560, AUPRC 0.4324.
- Development-only model-selection sensitivity selected a smaller transformer configuration with held-out AUPRC 0.3872.
- SSL PCA sensitivity retained similar enrichment with fewer predictors: PCA10 AUPRC 0.4241 and PCA20 AUPRC 0.4256.

Interpretation: the model is framed as a risk-enrichment and phenotyping workflow, not as a clinically deployable diagnostic predictor.

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

- signal + SSL embeddings: AUROC 0.6482, AUPRC 0.4432.
- signal + SSL + signal phenotype: AUROC 0.6560, AUPRC 0.4324.

The same configuration trained for 12 epochs did not improve validation performance, so longer training should not be assumed to help.

## CTGDL External Representation Validation

CTGDL FHRMA processed files have been downloaded and embedded using the current best transformer. This is representation/domain-shift validation, not neonatal outcome validation.

- CTGDL FHRMA records: 135.
- CTGDL FHRMA windows: 2604.
- CTU-vs-FHRMA embedding domain classifier AUROC: 0.9088.

Interpretation: external FHRMA data show substantial dataset shift, which is important evidence for the paper's robustness section.

## Manuscript and reporting files

The final LaTeX source package is in `manuscript_overleaf/`. Supplementary tables include reviewer-facing sensitivity analyses and a TRIPOD+AI checklist mapping.

## Code archive

Repository URL: https://github.com/Luciky-Leo/ctg-foundation-trajectory

This repository includes `.zenodo.json` and `CITATION.cff` metadata for Zenodo/GitHub release archiving. After enabling the Zenodo GitHub integration and creating a GitHub release, add the resulting DOI to the manuscript Code availability and Data availability statements.

## License

The analysis code is released under the MIT License. Public source datasets remain governed by their original database licenses and access terms.
