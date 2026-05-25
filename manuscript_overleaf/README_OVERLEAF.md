# CTG Foundation Trajectory - Methods of Information in Medicine oriented source package

This folder is a clean LaTeX source package built from the final manuscript state on 2026-05-24 and revised for Methods of Information in Medicine positioning.

## Main file

- `main.tex` is the current LaTeX source version of the manuscript, with double spacing, line numbering, and Methods of Information in Medicine oriented health-informatics language.
- Upload the source zip to Overleaf and set `main.tex` as the main document if Overleaf does not detect it automatically. The local compile-check PDF is for audit only and does not need to be uploaded.

## Final figure set

The main manuscript uses the PDF files in `figures/`:

- `Figure_1_study_design_pipeline.pdf`
- `Figure_2_model_validation_and_explanation.pdf`
- `Figure_3_dynamic_phenotype_prototypes.pdf`
- `Figure_4_recalibration_and_clinical_utility.pdf`

The supplementary figures are also in `figures/` and are numbered in first-citation order:

- `Supplementary_Figure_S1_model_selection_and_sensitivity.pdf`
- `Supplementary_Figure_S2_external_domain_shift.pdf`

`Figure_1A_method_flowchart_source.png` is retained only as the source asset for the generated Fig. 1A panel.

## Data attachments

- `source_data/` contains source-data CSV files for the figures.
- `supplementary_tables/` contains supplementary CSV files and the combined supplementary workbook. `Supplementary_Table_9_reviewer_sensitivity_analyses.csv` contains the added reviewer-facing sensitivity checks for model-selection leakage, SSL dimensionality, regularization and alternative outcome definitions. `Supplementary_Table_10_TRIPOD_AI_checklist.csv` maps the manuscript to TRIPOD+AI reporting items.

## Preview

- `BMC_MIDD_COMPILED_CHECK_REVIEW_READY.pdf` is the local compile check generated from this package. It is included for audit only and is not required by `main.tex`.

## Final audit notes

- Only the final manuscript PDF preview is retained in this package. Historical PDFs are excluded.
- The text has been reframed around clinical decision relevance, self-supervised time-series method design, AUPRC-based risk enrichment, recalibration, and explicit external-validation boundaries.
- Supplementary Figure S2 no longer contains the in-figure validation-boundary banner; the boundary is described in the text and caption.
- Main figures and supplementary figures are separated in `main.tex`; supplementary figures use S-numbering and follow first-citation order.
- Figures are forced to appear immediately after the `Figures` and `Supplementary figures` headings in the compiled review PDF.
- Figure 2C legend was moved above the panel to avoid overlap with confidence-interval lines.
- Reviewer-facing sensitivity analyses have been added for development-only model selection, PCA-reduced SSL embeddings, elastic-net regularization and pH-only outcome thresholds.
- Author names, affiliation, funding and contribution statements have been updated for Xiaoqian Gui, Kaisi Zhu, Ziyi Jiang and Feifan Lu, including equal-contribution statements for the first three authors.
