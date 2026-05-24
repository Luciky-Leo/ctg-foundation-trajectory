# Manuscript Figure Package Report

Date: 2026-05-23.

## Purpose

This report documents the manuscript-ready figure package assembled from the current CTG foundation-style trajectory analysis. The package converts analysis outputs into publication-oriented multi-panel figures with source-data tables.

## Figure Package Location

- Figures: `results/manuscript/figures`.
- Source data: `results/manuscript/source_data`.
- Manifest: `results/manuscript/source_data/manuscript_figure_manifest.csv`.
- Build script: `scripts/05_reports/build_manuscript_figures.py`.

Each figure is exported as PNG, PDF, SVG, and TIFF. PDF/SVG are intended for manuscript editing; TIFF is intended for journal upload; PNG is intended for quick review.

## Main Figures

### Figure 1. Study Design and Evidence Chain

Claim: open CTG datasets support a reproducible dynamic phenotyping and validation workflow.

Panels:

- A: data-to-model-to-validation workflow.
- B: CTU-UHB and CTGDL-FHRMA cohort scale.
- C: record-level outcome balance in the train/test split.
- D: evidence map for outcome, interpretability, robustness, and external-domain claims.

Source data: `Figure_1_source_data.csv`.

### Figure 2. Model Validation and Explanation

Claim: self-supervised CTG representations improve neonatal risk enrichment beyond classical signal summaries, with interpretable feature-level evidence.

Panels:

- A: precision-recall curve.
- B: ROC curve.
- C: bootstrap AUROC/AUPRC validation.
- D: bootstrap AUPRC enrichment versus signal-only features.
- E: missingness sensitivity after excluding missingness features.
- F: permutation importance for the signal-plus-SSL model.

Source data: `Figure_2_source_data.csv`.

### Figure 3. Dynamic Phenotype Prototypes

Claim: signal-derived phenotypes provide clinically interpretable dynamic CTG prototypes and risk gradients.

Panels:

- A: neonatal risk gradient across signal phenotypes.
- B: clinical outcome profile across signal phenotypes.
- C: standardized phenotype profile heatmap.
- D: physiological signal burden across signal phenotypes.
- E-H: representative CTG prototype records for signal phenotypes 1-4.

Source data: `Figure_3_source_data.csv`.

### Figure 4. Recalibration and Clinical Utility

Claim: cross-fitted Platt recalibration improves absolute risk calibration and threshold behavior without changing discrimination.

Panels:

- A: recalibration curve for the final signal-plus-SSL-plus-phenotype model.
- B: Brier and expected calibration error by calibration method.
- C: risk-score compression after Platt recalibration.
- D: bin-level calibration error.
- E: decision curve analysis after recalibration.
- F: mean net benefit in the 0.10-0.30 threshold range.

Source data: `Figure_4_source_data.csv`.

## Supplementary Figure

### Figure S1. External Representation Validation

Claim: CTGDL-FHRMA supports external representation/domain-shift validation, not external outcome validation.

Panels:

- A: CTU-UHB versus CTGDL-FHRMA embedding space.
- B: CTU-versus-CTGDL domain separability.
- C: CTGDL-FHRMA records assigned to CTU SSL phenotypes.
- D: CTU-UHB SSL phenotype risk gradient.

Source data: `Figure_S1_source_data.csv`.

### Figure S2. Model Selection and Sensitivity

Claim: transformer configuration, enrichment-calibration tradeoffs, SSL phenotype gradients, and feature-family contributions support the final modelling story.

Panels:

- A: transformer sweep ranking by AUPRC.
- B: enrichment-calibration frontier across model families.
- C: SSL phenotype risk gradient.
- D: feature-family contribution to AUPRC permutation importance.

Source data: `Figure_S2_source_data.csv`.

## Interpretation Boundary

The current public-data evidence supports a risk-enrichment and dynamic-phenotyping manuscript, not a deployable clinical risk calculator.

Defensible claims:

- The SSL representation improves AUPRC/risk enrichment beyond classical signal features in held-out CTU-UHB testing.
- Signal-derived phenotypes are the strongest clinical interpretation anchor.
- Raw risk scores are poorly calibrated, but cross-fitted Platt recalibration materially improves Brier score and expected calibration error.
- CTGDL-FHRMA demonstrates representation transport and domain shift.

Claims to avoid:

- Do not claim stable AUROC superiority, because AUROC difference intervals cross zero.
- Do not claim missingness-independent superiority as statistically secure, because no-missingness bootstrap intervals cross zero.
- Do not claim external outcome validation on CTGDL-FHRMA unless neonatal outcome labels are added.
- Do not claim immediate clinical deployment, because this remains retrospective public-data validation.

## Next Manuscript Step

The figure package now supports manuscript drafting and target-journal polishing:

- Use the expanded Figure 1-4, Figure S1, and Figure S2 captions in the LaTeX draft.
- Keep source-data CSVs with the submitted figure package.
- Add journal-specific citations, reporting-checklist language, and final author/declaration details before submission.
