# Missingness Sensitivity Report

Date: 2026-05-21

## Purpose

Permutation importance showed that `fhr_missing_fraction` was one of the strongest predictors in the current manuscript candidate model. This sensitivity analysis tests whether the apparent value of SSL representations depends mainly on FHR/UC missingness features.

## Design

Two feature scenarios were compared on the fixed CTU-UHB train/test split:

1. `full_features`: all 12 clinical signal features, including:
   - `fhr_missing_fraction`
   - `uc_missing_fraction`
2. `no_missingness_features`: the same signal feature set after removing both missingness features.

The same best SSL embeddings were used:

`results/tables/sweep/p80_d64_l3_channel_cw02_record_embeddings.csv`

Models tested:

- `signal_features`
- `signal_plus_ssl`
- `signal_plus_ssl_plus_signal_phenotype`

Bootstrap confidence intervals used 2000 held-out test-set resamples.

## Main Results

| Scenario | Model | AUROC (95% CI) | AUPRC (95% CI) | Brier (95% CI) |
|---|---|---:|---:|---:|
| Full features | Signal features | 0.6303 (0.5254-0.7284) | 0.3366 (0.2173-0.4941) | 0.2525 (0.2234-0.2834) |
| Full features | Signal + SSL | 0.6482 (0.5386-0.7491) | 0.4432 (0.2974-0.5891) | 0.2554 (0.2199-0.2901) |
| Full features | Signal + SSL + signal phenotype | 0.6560 (0.5490-0.7577) | 0.4324 (0.2867-0.5877) | 0.2516 (0.2150-0.2863) |
| No missingness | Signal features | 0.6428 (0.5348-0.7461) | 0.3765 (0.2418-0.5493) | 0.2482 (0.2216-0.2754) |
| No missingness | Signal + SSL | 0.6312 (0.5226-0.7369) | 0.4284 (0.2800-0.5823) | 0.2632 (0.2278-0.2970) |
| No missingness | Signal + SSL + signal phenotype | 0.6348 (0.5262-0.7405) | 0.4319 (0.2858-0.5855) | 0.2601 (0.2232-0.2939) |

## Model-Difference Results

With full features:

- `signal_plus_ssl - signal_features` AUPRC difference: 0.0978 (95% CI 0.0148-0.1864).
- `signal_plus_ssl_plus_signal_phenotype - signal_features` AUPRC difference: 0.0891 (95% CI 0.0073-0.1766).

After removing missingness features:

- `signal_plus_ssl - signal_features` AUPRC difference: 0.0448 (95% CI -0.0373 to 0.1275).
- `signal_plus_ssl_plus_signal_phenotype - signal_features` AUPRC difference: 0.0489 (95% CI -0.0308 to 0.1309).

The SSL-enhanced models remain directionally better for AUPRC after removing missingness, but the confidence intervals cross zero. Therefore the manuscript should not claim missingness-independent statistical superiority.

## Within-Model Robustness

Removing missingness features changed model performance as follows:

- `signal_plus_ssl` AUPRC change: -0.0152 (95% CI -0.0658 to 0.0355).
- `signal_plus_ssl_plus_signal_phenotype` AUPRC change: -0.0024 (95% CI -0.0587 to 0.0562).
- `signal_features` AUPRC change: +0.0378 (95% CI -0.0380 to 0.1149).

This suggests the SSL-enhanced models are not wholly dependent on missingness features, but the incremental SSL advantage is weaker when missingness is excluded.

## Interpretation for Manuscript

The appropriate wording is:

"Because fetal heart rate missingness was identified as an influential predictor, we repeated the validation after excluding missingness-related features. The SSL-enhanced models retained directionally higher AUPRC than signal-only models, but the bootstrap confidence intervals for the incremental differences crossed zero. These findings indicate that the representation signal is not solely driven by missingness, while also showing that the evidence for missingness-independent superiority remains modest."

Do not write:

- "The model is independent of signal missingness."
- "SSL remains significantly superior after excluding missingness."
- "Missingness is a biological fetal-risk mechanism."

## Generated Outputs

- `results/tables/missingness_sensitivity_predictions.csv`
- `results/tables/missingness_sensitivity_metrics.csv`
- `results/tables/missingness_sensitivity_differences.csv`
- `results/figures/missingness_sensitivity_performance.png`
