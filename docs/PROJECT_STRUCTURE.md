# CTG Foundation Trajectory Project Structure

This repository keeps the reproducible analysis code, source data derivatives,
manuscript assets, and submission-ready Overleaf files separated.

## Active Paths

- `config/`: analysis and pipeline configuration.
- `data/`: raw, interim, and processed CTG data products.
- `scripts/00_setup/`: setup and environment checks.
- `scripts/01_download/`: data acquisition scripts.
- `scripts/02_preprocess/`: CTG preprocessing and window construction.
- `scripts/03_features/`: feature engineering.
- `scripts/04_models/`: model training, SSL representation, clustering, and validation.
- `scripts/05_reports/`: accepted report-building scripts for manuscript text, figures, tables, and project status.
- `results/tables/`: analysis result tables used by the manuscript.
- `results/manuscript/figures/`: current manuscript figure files.
- `results/manuscript/source_data/`: current figure source-data files.
- `results/manuscript/supplementary_tables/`: supplementary table outputs.
- `manuscript_overleaf/`: current LaTeX manuscript package for Overleaf.
- `submission_packages/frontiers_signal_processing_20260601/`: current
  Frontiers in Signal Processing submission package, including the official
  Frontiers LaTeX template version.
- `reproducibility/`: split files and reproducibility metadata.

## Archive Paths

- `archive/deprecated_redraw_20260531/`: abandoned redraw experiments and related source-data copies. These files are retained for traceability but should not be used for the active manuscript package.

## Current Figure Policy

Use `scripts/05_reports/build_manuscript_figures.py` and the outputs in
`results/manuscript/figures/` for the active manuscript figures.

Do not use archived redraw outputs unless a future revision explicitly revives
that route.
