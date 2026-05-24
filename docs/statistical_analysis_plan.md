# Statistical Analysis Plan

## Objective

Develop and validate a self-supervised dynamic representation of intrapartum fetal heart rate and uterine contraction signals, then use the learned representations to identify fetal risk phenotypes and predict neonatal compromise.

## Primary Research Question

Can self-supervised CTG representations identify dynamic fetal risk phenotypes that predict adverse neonatal status better than conventional CTG features and standard supervised time-series baselines?

## Target Population

Intrapartum singleton CTG recordings with available FHR and UC channels and neonatal outcome metadata.

## Primary Dataset

CTU-UHB / CTU-CHB Intrapartum Cardiotocography Database.

## Candidate External Validation Sources

1. CTGDL CTU-UHB harmonized files, used only as a preprocessing reproducibility check if it duplicates CTU-UHB.
2. CTGDL non-CTU subsets, subject to license and data-use terms.
3. UCI Cardiotocography, used as a feature-level baseline only.

## Outcomes

Primary outcome candidates:

1. Low umbilical artery pH, with thresholds evaluated at pH < 7.05, pH < 7.10, and pH < 7.15.
2. Low Apgar score at 5 minutes.
3. Composite neonatal compromise, defined from pH, Apgar, base deficit, and available clinical indicators.

The final primary outcome must be selected before model comparison to avoid outcome shopping.

## Signal Preprocessing

1. Parse FHR and UC from WFDB records.
2. Align the last 60 minutes before delivery or record end.
3. Resample to a common frequency if required.
4. Mark missingness and signal quality.
5. Winsorize or mask implausible FHR and UC values rather than silently deleting traces.
6. Store both cleaned values and missingness masks.

## Baseline Models

1. Conventional CTG feature model using summary FHR/UC statistics.
2. XGBoost or random forest on engineered features.
3. CNN baseline on raw or windowed traces.
4. LSTM/GRU baseline.
5. Transformer supervised baseline.

## Self-Supervised Representation Learning

Candidate objectives:

1. Masked time-series reconstruction: hide segments of FHR/UC and reconstruct them.
2. Contrastive segment learning: pull segments from the same recording together and push different recordings apart.
3. Future-window prediction: predict the next segment representation from preceding CTG context.
4. Channel-asymmetric masking: reconstruct FHR from UC context and vice versa, where clinically meaningful.

## Dynamic Phenotype Discovery

Use embeddings from the pretrained model to derive phenotype classes by:

1. Gaussian mixture models.
2. Hierarchical clustering.
3. HDBSCAN or density-based clustering.
4. Latent trajectory class models if low-dimensional longitudinal features are used.

Phenotypes must be described clinically using:

1. FHR baseline and variability.
2. Deceleration burden.
3. UC frequency and intensity.
4. FHR-UC temporal coupling.
5. Missingness and signal quality.

## Model Evaluation

Discrimination:

1. AUROC.
2. AUPRC.
3. Sensitivity and specificity at clinically relevant thresholds.

Calibration:

1. Calibration slope and intercept.
2. Brier score.
3. Calibration plots.

Clinical utility:

1. Decision curve analysis.
2. Net benefit across plausible risk thresholds.

Robustness:

1. Temporal holdout.
2. Record-level bootstrap confidence intervals.
3. Sensitivity to analysis-window length.
4. Sensitivity to missingness handling.

## Fairness and Subgroup Analysis

Evaluate model performance by:

1. Gestational age strata.
2. Birthweight strata.
3. Delivery mode.
4. Sex of newborn if available.
5. Maternal age if available.
6. Signal quality strata.

If race/ethnicity is unavailable or not comparable, do not invent fairness claims. Use available clinically relevant subgroup axes.

## Interpretability

1. Segment attribution to identify high-risk windows.
2. Prototype traces for each phenotype.
3. Alignment of high-risk attribution with decelerations, variability loss, and UC peaks.
4. Comparison against expert CTG concepts where available.

## Reporting Standards

Use TRIPOD+AI and PROBAST+AI principles for prediction-model reporting and risk-of-bias review. Provide source code, preprocessing scripts, dataset version numbers, and a model card.

