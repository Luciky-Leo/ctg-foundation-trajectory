# R1 Release Notes (2026-07-07)

This repository snapshot supports the Frontiers in Signal Processing R1 revision package.

## Scientific Reproducibility Change

The R1 revision adopts the fixed-seed locked replay values throughout the manuscript, figures, supplementary tables and reviewer responses. The originally submitted unseeded values (signal + SSL AUPRC 0.4432; final AUPRC 0.4324) were not retained because the exact run could not be reproduced.

Current locked replay values:

- Signal features: AUROC 0.6303, AUPRC 0.3366.
- SSL embeddings alone: AUROC 0.4947, AUPRC 0.2125.
- Signal + SSL embeddings: AUROC 0.6025, AUPRC 0.3765.
- Signal + SSL + signal phenotype manuscript model: AUROC 0.6168, AUPRC 0.4016.
- Development-only model-selection sensitivity: held-out AUPRC 0.3872.

Both SSL-containing AUPRC-difference confidence intervals versus signal-only features crossed zero. SSL is therefore framed as an interpretable representation, phenotype-enrichment and recalibration layer rather than as a stable discrimination-improvement result.

## Current Submission Package

The current R1 package is:

`submission_packages/frontiers_signal_processing_r1_20260707/`

It contains:

- Frontiers original-template LaTeX source.
- Compiled main manuscript PDF.
- Compiled supplementary material PDF.
- Separate Reviewer 1 and Reviewer 2 response PDFs.
- Main figures 1-4 and Supplementary Figures S1-S3.
- Figure source-data CSV files.
- Supplementary Tables S1-S11 as CSV/XLSX plus standalone table PDFs.
- Submission precheck report.

Preflight status: PASS.

## Local Pre-Zenodo Archive

For manual Zenodo upload, use the release archive created from this repository snapshot:

`ctg-foundation-trajectory_v1.3.0-r1_20260707.zip`

The archive should be uploaded as a new version under the existing Zenodo concept DOI:

`10.5281/zenodo.20364141`

The previous published version DOI before this R1 update was:

`10.5281/zenodo.20485068`

After publishing the new Zenodo version, update `CITATION.cff`, `README.md`, and the manuscript Data Availability statement if the exact new version DOI rather than the concept DOI is required.

