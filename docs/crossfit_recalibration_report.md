# Cross-Fitted Recalibration Report

Date: 2026-05-21

## Purpose

The previous calibration analysis showed that all manuscript candidate models were poorly calibrated as absolute risk predictors. This report evaluates cross-fitted recalibration without using the held-out test set for calibration-model fitting.

## Design

Calibration was fitted only within the original training set:

1. Generate 5-fold out-of-fold predictions inside the training records.
2. Fit Platt scaling and isotonic calibration using the out-of-fold training predictions.
3. Train the base model on the full training set.
4. Predict the held-out test set.
5. Apply the training-derived calibrators to the held-out test predictions.
6. Recompute discrimination, Brier score, calibration metrics, and decision curves.

Models recalibrated:

- `signal_features`
- `signal_plus_ssl`
- `signal_plus_ssl_plus_signal_phenotype`

## Main Finding

Platt scaling is the best current recalibration option. It improves calibration and Brier score while preserving AUROC and AUPRC because it is monotonic with respect to the raw score.

Isotonic calibration improves Brier score and expected calibration error but reduces AUPRC and sometimes AUROC, likely because the calibration mapping creates tied or compressed risk scores in a modest-sized calibration set. Therefore, isotonic calibration should be treated as exploratory, not the main manuscript option.

## Recalibrated Performance

### Signal + SSL

| Method | AUROC | AUPRC | Brier | Calibration slope | ECE |
|---|---:|---:|---:|---:|---:|
| Raw | 0.6482 | 0.4432 | 0.2554 | 0.4520 | 0.2662 |
| Platt | 0.6482 | 0.4432 | 0.1530 | 1.2360 | 0.0266 |
| Isotonic | 0.6248 | 0.2758 | 0.1615 | 0.5722 | 0.0616 |

### Signal + SSL + Signal Phenotype

| Method | AUROC | AUPRC | Brier | Calibration slope | ECE |
|---|---:|---:|---:|---:|---:|
| Raw | 0.6560 | 0.4324 | 0.2516 | 0.4732 | 0.2629 |
| Platt | 0.6560 | 0.4324 | 0.1524 | 1.3744 | 0.0248 |
| Isotonic | 0.6389 | 0.2866 | 0.1608 | 0.5816 | 0.0539 |

## Bootstrap Differences Versus Raw

For `signal_plus_ssl`:

- Platt minus raw Brier: -0.1022 (95% CI -0.1432 to -0.0598).
- Platt minus raw expected calibration error: -0.2090 (95% CI -0.2549 to -0.1352).
- Platt leaves AUROC and AUPRC unchanged.

For `signal_plus_ssl_plus_signal_phenotype`:

- Platt minus raw Brier: -0.0991 (95% CI -0.1410 to -0.0566).
- Platt minus raw expected calibration error: -0.2052 (95% CI -0.2554 to -0.1294).
- Platt leaves AUROC and AUPRC unchanged.

## Decision Curve Impact

In the 0.10-0.30 threshold range, mean net benefit improved after Platt recalibration:

- Raw `signal_plus_ssl_plus_signal_phenotype`: 0.0320.
- Platt `signal_plus_ssl_plus_signal_phenotype`: 0.0513.
- Mean delta versus raw: +0.0193.

Across 0.05-0.50:

- Raw `signal_plus_ssl_plus_signal_phenotype`: -0.0265.
- Platt `signal_plus_ssl_plus_signal_phenotype`: 0.0472.
- Mean delta versus raw: +0.0736.

This indicates that recalibration is not merely cosmetic; it materially changes clinical-threshold behavior.

## Manuscript Recommendation

Use this hierarchy in the paper:

1. Raw model results for transparent discrimination and phenotype/risk-enrichment evidence.
2. Cross-fitted Platt recalibration as the preferred absolute-probability calibration method.
3. Isotonic calibration only as a sensitivity result, or omit from the main text if space is limited.

Suggested wording:

"Because class-balanced model fitting produced poorly calibrated absolute probabilities, we fitted recalibration models using out-of-fold predictions within the training set and evaluated them on the held-out test set. Cross-fitted Platt scaling substantially improved Brier score and expected calibration error while preserving AUROC and AUPRC, supporting its use when absolute risk probabilities are required. Isotonic calibration improved calibration metrics but reduced precision-recall performance, and was therefore treated as exploratory."

## Generated Outputs

- `results/tables/recalibrated_test_predictions.csv`
- `results/tables/recalibration_metrics_ci.csv`
- `results/tables/recalibration_metric_differences.csv`
- `results/tables/recalibration_curve_bins.csv`
- `results/tables/recalibrated_decision_curve_net_benefit.csv`
- `results/tables/recalibrated_decision_curve_summary.csv`
- `results/figures/recalibration_curves.png`
- `results/figures/recalibrated_decision_curve_analysis.png`
