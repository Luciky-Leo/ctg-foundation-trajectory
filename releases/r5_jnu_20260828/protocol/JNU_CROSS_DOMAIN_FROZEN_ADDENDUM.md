# JNU-CTG Cross-Domain Enhancement: Frozen Addendum

Status: FROZEN BEFORE JNU MODEL RUNS

Frozen on: 2026-08-27

## Purpose

This addendum tests whether larger-scale public antepartum CTG pretraining
improves transferable representation quality. It does not redefine the CTU-UHB
clinical endpoint and does not treat JNU-CTG as external neonatal-outcome
validation.

## Locked CTU route

- Cohort: 552 CTU-UHB records; 386 development and 166 fixed test records.
- Primary endpoint: cord pH below 7.15 or 5-minute Apgar below 7.
- Frozen encoder: patch length 80, d_model 48, two transformer layers, four
  attention heads, mixed masking, reconstruction weight 1.0, contrastive
  weight 0.1, jitter standard deviation 0.02.
- Primary supervised comparison: ten prespecified signal features versus the
  same ten features plus five training-fitted SSL principal components in an
  L2-regularized logistic model.
- The fixed CTU test set remains exploratory because it was viewed during
  historical model work. It is never used for JNU preprocessing, encoder
  selection, epoch selection, feature selection, PCA selection, or threshold
  selection.

## JNU data contract

- Source: JNU-CTG Zenodo record 21800730.
- Expected archive MD5: ac1cfcba2f1b3336596544211d776771.
- Expected scale: 20,769 recordings from 12,606 patient groups.
- License: CC BY 4.0, verified from the archive LICENSE.txt.
- Each 30-minute 4-Hz record is converted to three non-overlapping 10-minute
  windows with two channels (FHR and UC).
- Apgar, neonatal asphyxia, FIGO labels, maternal variables, and all other
  clinical metadata are excluded from SSL pretraining.
- Any supervised JNU analysis must split by patient_id. Record-level random
  splitting is prohibited.

## Prespecified encoder comparison

The following four routes are compared without architecture search:

1. Randomly initialized frozen encoder.
2. CTU-development-only SSL encoder.
3. JNU-only SSL encoder.
4. JNU SSL pretraining followed by CTU-development-only SSL adaptation.

Seeds are 20260521 through 20260525. Primary evaluation uses the existing CTU
development folds. The fixed test set is evaluated once per seed only after all
development outputs have been written.

## Prespecified secondary analyses

- Frozen-encoder linear probes for available CTU expert morphology/FIGO labels,
  using grouped record-level cross-validation.
- Input ablation: FHR+UC, FHR-only, and UC-only.
- Sampling sensitivity: 4 Hz primary and 1 Hz secondary.
- Channel-asymmetric reconstruction, with FHR masked and UC visible, as a
  development-only sensitivity analysis.

## Promotion rules

- Main-text promotion requires consistent direction across five seeds and a
  paired development-fold interval supporting a meaningful representation
  improvement over both the random encoder and CTU-only SSL.
- A risk-model difference whose paired interval crosses zero is described as
  inconclusive and cannot support an improved-prediction claim.
- Morphology-probe improvement may support representation transfer but not
  clinical outcome transportability.
- Negative and null results are retained and reported.
- JNU and CTU outcomes are never pooled.
- The study remains a retrospective delivery-level risk-enrichment analysis
  without a prespecified lead-time horizon.

