# Calibration and Decision Curve Report

Date: 2026-05-21

## Purpose

This report evaluates whether the current manuscript candidate models provide well-calibrated risk probabilities and clinically useful threshold-based decisions on the fixed CTU-UHB held-out test split.

The analysis uses existing held-out predictions from:

`results/tables/manuscript_test_predictions.csv`

No model was retrained for this analysis.

## Models

1. `signal_features`
2. `signal_plus_ssl`
3. `signal_plus_ssl_plus_signal_phenotype`

## Calibration Results

| Model | Brier (95% CI) | Calibration intercept (95% CI) | Calibration slope (95% CI) | Expected calibration error (95% CI) |
|---|---:|---:|---:|---:|
| Signal features | 0.2525 (0.2234-0.2834) | -1.3838 (-1.8290 to -1.0136) | 0.4876 (0.1338-0.8750) | 0.2745 (0.2102-0.3376) |
| Signal + SSL | 0.2554 (0.2199-0.2901) | -1.4050 (-1.8754 to -1.0263) | 0.4520 (0.1745-0.7691) | 0.2662 (0.2080-0.3333) |
| Signal + SSL + phenotype | 0.2516 (0.2150-0.2863) | -1.4045 (-1.8738 to -1.0199) | 0.4732 (0.1993-0.7838) | 0.2629 (0.2044-0.3313) |

## Interpretation

All three models are poorly calibrated as absolute risk probability models:

- Calibration slopes are far below 1.
- Calibration intercepts are strongly negative.
- Mean prediction error and expected calibration error are high.

This likely reflects use of class-balanced logistic models on an imbalanced outcome. The current models are better interpreted as risk-ranking/risk-enrichment tools rather than deployable absolute-probability calculators.

The key manuscript implication is:

"The SSL-enhanced models improved risk enrichment, but all candidate models showed suboptimal calibration on the held-out test set. Therefore, the present work should be framed as dynamic risk stratification and representation validation, not as a ready-to-deploy absolute risk calculator."

## Decision Curve Results

Mean net benefit over clinically plausible threshold ranges:

| Threshold range | Strategy | Mean net benefit | Mean delta vs signal features |
|---|---|---:|---:|
| 0.05-0.20 | Signal features | 0.0944 | NA |
| 0.05-0.20 | Signal + SSL | 0.0953 | +0.0010 |
| 0.05-0.20 | Signal + SSL + phenotype | 0.0996 | +0.0052 |
| 0.10-0.30 | Signal features | 0.0262 | NA |
| 0.10-0.30 | Signal + SSL | 0.0307 | +0.0045 |
| 0.10-0.30 | Signal + SSL + phenotype | 0.0320 | +0.0058 |
| 0.05-0.50 | Signal features | -0.0397 | NA |
| 0.05-0.50 | Signal + SSL | -0.0298 | +0.0099 |
| 0.05-0.50 | Signal + SSL + phenotype | -0.0265 | +0.0132 |

The decision-curve advantage of SSL-enhanced models is small and threshold-dependent. The most favorable model is usually `signal_plus_ssl_plus_signal_phenotype`, but the absolute gain over signal features is modest.

## Practical Conclusion

Current evidence supports:

- risk enrichment,
- phenotype-assisted stratification,
- ranking-oriented model comparison,
- exploratory decision-curve benefit in selected threshold ranges.

Current evidence does not yet support:

- direct clinical deployment,
- stable individualized absolute risk probabilities,
- strong net-benefit superiority across all thresholds.

## Generated Outputs

- `results/tables/calibration_metrics_ci.csv`
- `results/tables/calibration_curve_bins.csv`
- `results/tables/decision_curve_net_benefit.csv`
- `results/tables/decision_curve_summary.csv`
- `results/figures/manuscript_calibration_curves.png`
- `results/figures/decision_curve_analysis.png`

## Next Statistical Step

If this project moves toward a clinical prediction manuscript, the next statistical addition should be calibration improvement:

1. Split the current training set into training and calibration folds, or use cross-fitting.
2. Apply Platt scaling or isotonic calibration to the risk scores.
3. Recompute calibration curves, Brier score, and decision curves.
4. Keep the current uncalibrated results as the transparent baseline.
