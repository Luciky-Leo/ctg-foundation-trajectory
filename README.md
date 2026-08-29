# CTG Foundation Trajectory

Code, manuscript source, and submission package for:

Self-supervised biomedical signal representation learning from intrapartum cardiotocography: stability, ablation, and exploratory neonatal risk enrichment

## Current release

The current Reviewer 5 revision is `v1.4.1-r2` (2026-08-29). Its authoritative public payload is under `releases/r5_jnu_fig1v2_20260829/`. Version `v1.4.0-r2` and earlier manuscript candidates remain available for traceability but are not the current presentation package.

## Overview

This repository contains the reproducible analysis scripts, manuscript source files, and Frontiers submission package for an open-data intrapartum cardiotocography (CTG) biomedical signal representation-learning study. The analysis evaluates whether self-supervised fetal heart rate (FHR) and uterine contraction (UC) trajectory representations can support calibrated neonatal risk enrichment, clinically interpretable dynamic phenotypes and explicit domain-shift assessment.

The repository is intended as a code and source-data archive, not as a deployable clinical device. It does not include raw public datasets, generated model weights, Python virtual environments, or large intermediate tensors.

## Primary Data Strategy

| Role | Dataset | Use |
|---|---|---|
| Primary signal cohort | CTU-UHB / CTU-CHB Intrapartum CTG Database | Raw FHR and UC time-series, clinical outcomes |
| Cross-domain label-free pretraining | JNU-CTG | 20,769 antepartum records from 12,606 patient groups; labels withheld during SSL |
| External representation context | CTGDL-FHRMA | Representation/domain-shift analysis only; not neonatal-outcome validation |
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
releases/r5_jnu_fig1v2_20260829  Current R2 manuscript, scripts, aggregate results, Figure 1 v2 and audits
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

## Current R2 evidence

The current analysis freezes the 386/166 CTU-UHB record split, a compact p80-d48-l2 mixed-mask encoder, five training seeds and a downstream model with 10 classical signal features plus five training-fitted SSL principal components. The fixed test set had historical configuration exposure and is therefore reported as exploratory rather than fully independent.

- Signal baseline fixed-test AUPRC: 0.3765; AUROC: 0.6428.
- Reduced SSL five-seed median fixed-test AUPRC: 0.3935; AUROC: 0.5954.
- All paired AUPRC intervals versus the signal baseline crossed zero.
- Development-only objective ablations did not outperform a random encoder.

JNU-CTG supplied 62,307 non-overlapping 10-minute windows for a prespecified public cross-domain sensitivity. JNU-to-CTU adaptation did not improve development risk enrichment over CTU-only SSL (paired mean AUPRC difference 0.0030, 95% CI -0.0304 to 0.0316) and had lower exploratory fixed-test AUPRC (0.3420 versus 0.4017). It did transfer expert-labelled morphology information: median macro-AUROC increased from 0.6392 to 0.6802 versus CTU-only SSL, but remained below classical signal features (0.7357).

The prespecified promotion decision is therefore `SUPPLEMENT_ONLY`. The allowed claim is transfer of morphology information, not improved neonatal-risk prediction, external outcome validation or clinical deployment.

## CTGDL external representation context

The selected encoder was applied to 135 CTGDL-FHRMA records. The revised domain classifier AUROC is 0.8811. Because a harmonized neonatal endpoint was unavailable, this analysis tests representation transport and source-domain shift only.

## Manuscript and reporting files

The current R2 manuscript, supplement, Reviewer 5 response, scripts, aggregate results, figures, supplementary tables S1-S18 and reproducibility audits are under `releases/r5_jnu_20260828/`. The R1 and first-submission packages remain under `submission_packages/` for traceability.

## Code archive

[![DOI](https://zenodo.org/badge/1248137386.svg)](https://doi.org/10.5281/zenodo.20364141)

Repository URL: https://github.com/Luciky-Leo/ctg-foundation-trajectory

Previous R1 Zenodo version DOI: https://doi.org/10.5281/zenodo.21242760

Zenodo concept DOI for all versions: https://doi.org/10.5281/zenodo.20364141

This repository includes `.zenodo.json` and `CITATION.cff` metadata for Zenodo/GitHub release archiving.

## License

The analysis code is released under the MIT License. Public source datasets remain governed by their original database licenses and access terms.
