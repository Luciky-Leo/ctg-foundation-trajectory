# Explainability Report

Date: 2026-05-21

## Purpose

This report adds model-interpretability evidence for the CTG foundation trajectory project. Because SHAP is not installed in the current Python environment, the implemented explanation method is permutation importance on the fixed held-out CTU-UHB test split.

## Inputs

- Signal features: `data/processed/ctu_uhb_record_features.csv`
- Best SSL embeddings: `results/tables/sweep/p80_d64_l3_channel_cw02_record_embeddings.csv`
- Best SSL phenotypes: `results/tables/sweep/p80_d64_l3_channel_cw02_ssl_phenotypes.csv`
- Signal phenotypes: `results/tables/ctu_uhb_signal_phenotypes.csv`
- Record split: `data/processed/ctu_uhb_record_split.csv`

## Models Explained

1. `signal_features`
2. `signal_plus_ssl`
3. `signal_plus_ssl_plus_signal_phenotype`

Permutation importance was computed for AUPRC and AUROC. AUPRC is the primary explanation target because bootstrap validation showed that SSL representations most clearly improve risk enrichment rather than AUROC or calibration.

## Key Findings

The strongest individual predictors for AUPRC in the current best model are a mixture of clinical signal features and SSL embedding dimensions.

Top recurring features include:

- `fhr_missing_fraction`
- SSL dimensions including `emb_55`, `emb_32`, `emb_17`, `emb_19`, and `emb_43`
- `fhr_mean`

This supports the manuscript claim that SSL representations add information beyond classical features. However, the SSL dimensions are latent features and should be interpreted as representation evidence rather than direct clinical mechanisms.

## Feature-Family Interpretation

For `signal_plus_ssl`, positive AUPRC permutation importance was:

- Clinical signal features: total 0.4874, mean per feature 0.0406 across 12 features.
- SSL embeddings: total 2.6003, mean per feature 0.0406 across 64 features.

For `signal_plus_ssl_plus_signal_phenotype`, positive AUPRC permutation importance was:

- Clinical signal features: total 0.3688, mean per feature 0.0307.
- SSL embeddings: total 1.8933, mean per feature 0.0296.
- Signal phenotype: total 0.0235.

The total SSL contribution is larger because there are 64 embedding dimensions. The mean-per-feature contribution is similar to clinical signal features, which is a more conservative interpretation.

## Representative Records

The high-risk representative records selected by `signal_plus_ssl` have low cord pH and high predicted risk:

- Record `2003`: pH 6.96, predicted risk 0.9689, signal phenotype 2, SSL phenotype 2.
- Record `1199`: pH 7.02, predicted risk 0.9540, signal phenotype 2, SSL phenotype 4.
- Record `1361`: pH 7.02, predicted risk 0.9498, signal phenotype 1, SSL phenotype 3.
- Record `1108`: pH 7.02, predicted risk 0.9456, signal phenotype 2, SSL phenotype 3.
- Record `1495`: pH 7.03, predicted risk 0.9350, signal phenotype 2, SSL phenotype 4.

These records generally show clinically concerning signal patterns such as lower FHR percentiles, higher FHR variability, and higher deceleration-proxy burden.

Low-risk representative records have higher pH, lower predicted risk, and generally less extreme signal summaries:

- Record `1203`: pH 7.34, predicted risk 0.0688.
- Record `1378`: pH 7.30, predicted risk 0.0592.
- Record `1315`: pH 7.19, predicted risk 0.0473.
- Record `1407`: pH 7.26, predicted risk 0.0346.
- Record `1155`: pH 7.34, predicted risk 0.0255.

## Important Caveat

`fhr_missing_fraction` is highly important in the current permutation analysis. This must be handled carefully:

- It may reflect poor signal quality or signal loss during clinically difficult monitoring.
- It may also act as a proxy for severe fetal/maternal events, motion, intervention, or technical artifact.
- It should not be presented as a direct physiological mechanism.

Recommended sensitivity analysis:

1. Refit models without missingness features.
2. Compare AUROC/AUPRC/Brier and bootstrap CIs.
3. Report whether SSL and signal phenotype findings remain directionally stable.

## Generated Outputs

- `results/tables/permutation_feature_importance.csv`
- `results/tables/permutation_feature_family_importance.csv`
- `results/tables/model_coefficients.csv`
- `results/tables/explanation_representative_records.csv`
- `results/figures/permutation_importance_top_features.png`
- `results/figures/permutation_importance_feature_families.png`

## Manuscript Use

Use this section conservatively:

"Permutation-based model interpretation indicated that the improvement in risk enrichment was supported by both classical CTG signal summaries and latent self-supervised representation dimensions. The most influential clinical summaries included fetal heart rate missingness, mean fetal heart rate, and related signal-distribution features. Because signal missingness may reflect both clinical difficulty and technical artifact, we treated it as an interpretable but non-mechanistic risk marker and planned sensitivity analyses excluding missingness features."
