# Frontiers in Signal Processing upload text

Journal: Frontiers in Signal Processing
Specialty: Biomedical Signal Processing
Article type: Original Research

## Title
Self-supervised biomedical signal representation learning from intrapartum cardiotocography for neonatal risk enrichment

## Running title
Self-supervised CTG signal representation

## Abstract
Intrapartum cardiotocography records long paired fetal heart rate and uterine contraction signals, but many decision-support approaches still compress these physiological trajectories into static summaries. We evaluated whether self-supervised biomedical signal representation learning can support interpretable neonatal risk enrichment using open cardiotocography data. CTU-UHB/CTU-CHB was used as the primary raw-signal cohort and CTGDL-FHRMA for external representation/domain-shift analysis. The primary endpoint was umbilical artery pH <7.15 or 5-minute Apgar score <7. CTU-UHB records were split at the record level into 386 training and 166 held-out test records. A compact PatchTST-style transformer encoder was trained only on training-record fetal heart rate and uterine contraction windows using patch tokenization, masked reconstruction and NT-Xent contrastive consistency, and record-level embeddings were combined with classical signal features and dynamic phenotypes. In the held-out test set, signal-only features achieved an AUPRC of 0.3366 (95% CI 0.2173-0.4941), whereas adding self-supervised embeddings increased AUPRC to 0.4432 (95% CI 0.2974-0.5891). The signal-plus-embedding-plus-phenotype model achieved AUROC 0.6560 (95% CI 0.5490-0.7577) and AUPRC 0.4324 (95% CI 0.2867-0.5877), but these estimates should be interpreted cautiously because the transformer sweep used held-out AUPRC for exploration and the full model had a low event-to-predictor ratio. Development-only model selection gave a more conservative held-out AUPRC of 0.3872. Cross-fitted Platt recalibration improved Brier score from 0.2516 to 0.1524, and CTGDL-FHRMA showed substantial representation domain shift (domain AUROC 0.9088). These findings support self-supervised cardiotocography representation learning as a reproducible biomedical signal-processing workflow for hypothesis-generating risk enrichment and phenotype discovery, but they do not establish external clinical validity or clinical deployability.

## Keywords
cardiotocography, biomedical signal processing, self-supervised learning, time-series representation learning, fetal heart rate, uterine contraction, neonatal risk, calibration

## Cover letter
See 03_cover_letter/Cover_Letter_Frontiers.docx.

## Recommended file mapping
- Manuscript: 01_manuscript_frontiers_full/Frontiers_Original_Research_Manuscript.docx
- Title page: 02_title_page/Frontiers_Title_Page.docx, if the portal requests one separately.
- Cover letter: 03_cover_letter/Cover_Letter_Frontiers.docx
- Figures: upload 04_figures_images/*.jpg as individual figures.
- Supplementary material: upload 05_supplementary_files/Supplementary_Tables_CTGFHT.xlsx and source-data CSVs if the portal allows source data.

## Retargeting note
This version is retargeted to Frontiers in Signal Processing, Biomedical Signal Processing specialty. The key framing is biomedical signal representation learning from CTG time series, not a deployable clinical diagnostic model.
