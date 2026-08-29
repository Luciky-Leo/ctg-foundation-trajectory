# CTG R5 JNU cross-domain revision payload

This payload accompanies the 2026-08-28 Reviewer 5 revision of the public CTG representation-learning study.

## Revision-specific addition

JNU-CTG (Zenodo `10.5281/zenodo.21800730`) was used only for label-free antepartum pretraining. All 20,769 records were readable and mapped to 12,606 patient groups; 62,307 non-overlapping 10-minute windows were generated. Clinical labels were withheld during SSL, and grouped splits prevented patient overlap.

The prespecified promotion gate was not met for neonatal-risk enrichment. JNU-to-CTU adaptation transferred expert-labelled morphology information but did not stably outperform CTU-only SSL for risk enrichment. These results remain supplementary and are not presented as external outcome validation.

## Contents

- `manuscript`: final Frontiers source, bibliography, supplement, and PDFs.
- `figures`: manuscript figures and reproducible Figure 1/S3 vector exports.
- `supplement`: machine-readable supplementary tables S1-S18.
- `scripts`: JNU preparation, frozen cross-domain training, analysis, figure, and audit code.
- `protocol`: frozen JNU cross-domain addendum.
- `results`: compact aggregate tables and promotion decision; no patient-level raw signals.
- `reproducibility`: leakage, numeric, figure, reference, and four-gate audits.

## Deliberate exclusions

Raw CTG records, window arrays, model checkpoints, window-level embeddings, credentials, and temporary build files are excluded. Obtain public data from the original providers under their licenses. JNU-CTG is CC BY 4.0.

## Persistent links

- Repository: https://github.com/Luciky-Leo/ctg-foundation-trajectory
- Zenodo concept DOI: https://doi.org/10.5281/zenodo.20364141

The included source code is distributed under `LICENSE`. Dataset licenses remain with the original providers.
