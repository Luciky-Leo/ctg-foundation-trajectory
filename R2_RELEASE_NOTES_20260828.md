# R2 Reviewer 5 release notes

Version: `1.4.0-r2`

Date: 2026-08-29

## Added

- JNU-CTG label-free cross-domain pretraining using 20,769 records, 12,606 patient groups and 62,307 non-overlapping 10-minute windows.
- Five-seed comparisons of random, CTU-only, JNU-only and JNU-to-CTU encoders.
- Frozen-encoder expert-morphology linear probes.
- FHR-only, UC-only, 1-Hz and channel-asymmetric masking sensitivities.
- Hierarchical paired bootstrap intervals and a prespecified main-figure promotion decision.
- Rebuilt Figure 1, Supplementary Figure S3, supplementary tables S14-S18 and a complete Reviewer 5 response.

## Corrected interpretation

The fixed test set had historical configuration exposure and is not represented as fully independent. The main downstream model uses 10 classical signal features plus five training-fitted SSL principal components; phenotype variables are secondary sensitivity analyses.

JNU-to-CTU pretraining did not provide stable neonatal-risk enrichment. Its development AUPRC difference versus CTU-only SSL was 0.0030 (95% CI -0.0304 to 0.0316), and exploratory fixed-test AUPRC was 0.3420 versus 0.4017. The route did improve frozen-encoder morphology macro-AUROC from 0.6392 to 0.6802, while remaining below classical signal features at 0.7357.

The release therefore supports a bounded morphology-transfer result, not predictive superiority, external neonatal-outcome validation, a fixed clinical prediction horizon or deployment readiness.

## Reproducibility gates

- JNU completion audit: 56/56 PASS.
- Method four-gate audit: PASS.
- Canonical submission preflight: PASS.
- Public release excludes raw CTG data, window arrays, embeddings and model weights.
