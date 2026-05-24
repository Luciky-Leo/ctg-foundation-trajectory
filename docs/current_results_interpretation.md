# Current Results Interpretation

Date: 2026-05-21.

## Completed Analyses

1. Installed a working PyTorch CPU environment using uv-managed Python 3.12.13.
2. Created a fixed record-level split: 386 train records and 166 test records.
3. Built full CTU-UHB windows: 7602 windows, 2 channels, 2400 samples per 10-minute window.
4. Trained the initial convolutional masked reconstruction encoder as an engineering prototype.
5. Replaced it with a PatchTST-style masked transformer encoder.
6. Added mixed/block/channel masking and a compact contrastive objective.
7. Ran a compact transformer sweep and selected `patch_len=80`, `d_model=64`, `layers=3`, channel masking, and `contrastive_weight=0.2` as the current manuscript candidate.
8. Extracted record-level transformer embeddings, SSL phenotypes, and signal-feature phenotypes.
9. Ran fixed train/test validation, bootstrap confidence intervals, model-difference bootstrap tests, CTGDL-FHRMA representation validation, and phenotype prototype curve generation.
10. Added permutation-based prediction explanation for the manuscript candidate models.
11. Added missingness sensitivity analysis after excluding `fhr_missing_fraction` and `uc_missing_fraction`.
12. Added calibration and decision curve analysis for the manuscript candidate models.
13. Added cross-fitted Platt and isotonic recalibration using training-set out-of-fold predictions.
14. Assembled manuscript-ready multi-panel figures with matching source-data tables.

## Manuscript Candidate Validation

The current manuscript candidate should use the best sweep artifacts, not the later longer-epoch experimental run. Record-level validation uses 386 training records and 166 held-out test records. Bootstrap intervals use 2000 test-set resamples.

| Model | AUROC (95% CI) | AUPRC (95% CI) | Brier (95% CI) |
|---|---:|---:|---:|
| Signal features | 0.6303 (0.5254-0.7284) | 0.3366 (0.2173-0.4941) | 0.2525 (0.2234-0.2834) |
| Transformer SSL embeddings | 0.5512 (0.4377-0.6611) | 0.2495 (0.1704-0.3882) | 0.2671 (0.2428-0.2911) |
| Signal + transformer SSL | 0.6482 (0.5386-0.7491) | 0.4432 (0.2974-0.5891) | 0.2554 (0.2199-0.2901) |
| Signal phenotype only | 0.6003 (0.4951-0.6970) | 0.2488 (0.1680-0.3519) | 0.2523 (0.2298-0.2751) |
| SSL phenotype only | 0.5153 (0.3952-0.6240) | 0.2170 (0.1444-0.3089) | 0.2503 (0.2485-0.2522) |
| Signal + signal phenotype | 0.6328 (0.5243-0.7306) | 0.3495 (0.2210-0.5007) | 0.2503 (0.2216-0.2807) |
| Signal + SSL phenotype | 0.6281 (0.5214-0.7267) | 0.3377 (0.2207-0.4932) | 0.2537 (0.2252-0.2852) |
| Signal + SSL + signal phenotype | 0.6560 (0.5490-0.7577) | 0.4324 (0.2867-0.5877) | 0.2516 (0.2150-0.2863) |

## Stability Interpretation

The strongest statistically supported improvement is in AUPRC, not AUROC or calibration.

- Signal + SSL versus signal features: AUPRC difference 0.0978 (95% CI 0.0148-0.1864).
- Signal + SSL + signal phenotype versus signal features: AUPRC difference 0.0891 (95% CI 0.0073-0.1766).
- AUROC differences cross zero for both comparisons, so the current evidence should not claim a stable AUROC gain.
- Brier differences also cross zero, so the current evidence should not claim improved calibration.

This supports a defensible manuscript claim that SSL representations improve risk enrichment under class imbalance, while classical signal features remain the clinical anchor.

## Phenotype Interpretation

Signal-derived phenotypes remain more clinically interpretable than SSL-only phenotypes.

Signal phenotype 2 is the strongest current high-risk dynamic phenotype: neonatal risk 34.39%, lower mean cord pH, higher FHR variability, lower FHR 5th percentile, and higher deceleration-proxy burden. This is the best clinical interpretation anchor for the paper.

The current best SSL phenotypes are balanced in size but show only a modest risk gradient. They should be framed as foundation-style representation phenotypes and linked to prototype curves, not as final clinical classes.

Prototype records and CTG curves were generated for every signal and SSL phenotype:

- Signal prototypes: records 1122, 1016, 1353, and 1271.
- SSL prototypes: records 1229, 1219, 2010, and 1344.
- Figures: `results/figures/phenotype_prototypes/*.png`.
- Table: `results/tables/phenotype_prototype_records.csv`.

## Prediction Explanation

Permutation importance was added for `signal_features`, `signal_plus_ssl`, and `signal_plus_ssl_plus_signal_phenotype`.

The current explanation supports a balanced interpretation:

- Clinical signal features and SSL dimensions both contribute to AUPRC enrichment.
- In `signal_plus_ssl`, positive AUPRC importance sums are larger for SSL embeddings because there are 64 embedding dimensions, but mean positive importance per feature is similar for clinical signal features and SSL embeddings.
- Top recurring features include `fhr_missing_fraction`, SSL dimensions such as `emb_55` and `emb_32`, and `fhr_mean`.

The important caution is that `fhr_missing_fraction` may reflect signal quality, difficult monitoring, clinical deterioration, or artifact. It should be treated as a risk marker and sensitivity-analysis target, not as a direct physiological mechanism.

Prediction-explanation outputs:

- `results/tables/permutation_feature_importance.csv`.
- `results/tables/permutation_feature_family_importance.csv`.
- `results/tables/explanation_representative_records.csv`.
- `results/figures/permutation_importance_top_features.png`.
- `results/figures/permutation_importance_feature_families.png`.

## Missingness Sensitivity

Because `fhr_missing_fraction` was an influential predictor, the main manuscript candidate models were rerun after excluding both FHR and UC missingness features.

The key result is mixed but useful:

- `signal_plus_ssl` AUPRC changed from 0.4432 to 0.4284 after excluding missingness.
- `signal_plus_ssl_plus_signal_phenotype` AUPRC changed from 0.4324 to 0.4319.
- The AUPRC advantage over signal-only models remained directionally positive after excluding missingness, but the bootstrap confidence intervals crossed zero.

Therefore, the appropriate conclusion is that SSL-enhanced models are not wholly dependent on missingness features, but missingness-independent superiority is not statistically secure in the current sample.

Missingness sensitivity outputs:

- `results/tables/missingness_sensitivity_metrics.csv`.
- `results/tables/missingness_sensitivity_differences.csv`.
- `results/figures/missingness_sensitivity_performance.png`.

## Calibration and Clinical Utility

Calibration and decision curve analysis were added using the fixed held-out test predictions.

Calibration is a weakness of the current candidate models:

- `signal_features` calibration slope: 0.4876.
- `signal_plus_ssl` calibration slope: 0.4520.
- `signal_plus_ssl_plus_signal_phenotype` calibration slope: 0.4732.
- Calibration intercepts are strongly negative for all three models.

This means the current models should be framed as risk-ranking and risk-enrichment models, not deployable absolute risk calculators.

Decision curve analysis shows small threshold-dependent net-benefit gains:

- In the 0.10-0.30 threshold range, mean net benefit is 0.0262 for `signal_features`, 0.0307 for `signal_plus_ssl`, and 0.0320 for `signal_plus_ssl_plus_signal_phenotype`.
- The gain over signal features is modest: about 0.0045 to 0.0058 mean net benefit in the 0.10-0.30 threshold range.

Calibration and DCA outputs:

- `results/tables/calibration_metrics_ci.csv`.
- `results/tables/calibration_curve_bins.csv`.
- `results/tables/decision_curve_net_benefit.csv`.
- `results/tables/decision_curve_summary.csv`.
- `results/figures/manuscript_calibration_curves.png`.
- `results/figures/decision_curve_analysis.png`.

## Cross-Fitted Recalibration

Cross-fitted recalibration was added because raw calibration slopes were substantially below 1. Recalibrators were fitted only on training-set out-of-fold predictions and evaluated on the fixed held-out test set.

Platt scaling is the preferred recalibration strategy:

- For `signal_plus_ssl`, Brier improved from 0.2554 to 0.1530 and ECE improved from 0.2662 to 0.0266, while AUROC and AUPRC were unchanged.
- For `signal_plus_ssl_plus_signal_phenotype`, Brier improved from 0.2516 to 0.1524 and ECE improved from 0.2629 to 0.0248, while AUROC and AUPRC were unchanged.
- In the 0.10-0.30 threshold range, Platt-calibrated `signal_plus_ssl_plus_signal_phenotype` had mean net benefit 0.0513, compared with 0.0320 for the raw version.

Isotonic calibration should remain exploratory because it improved calibration metrics but reduced AUPRC substantially.

Recalibration outputs:

- `results/tables/recalibrated_test_predictions.csv`.
- `results/tables/recalibration_metrics_ci.csv`.
- `results/tables/recalibration_metric_differences.csv`.
- `results/tables/recalibrated_decision_curve_summary.csv`.
- `results/figures/recalibration_curves.png`.
- `results/figures/recalibrated_decision_curve_analysis.png`.

## Manuscript Figure Package

The current manuscript figure package is in `results/manuscript`.

Main figures:

- Figure 1: study design, data scale, and evidence boundary.
- Figure 2: PR/ROC curves, bootstrap validation, AUPRC enrichment, missingness sensitivity, and permutation explanation.
- Figure 3: signal-phenotype risk gradient, clinical outcome profile, phenotype heatmap, signal burden, and prototype CTG records.
- Figure 4: cross-fitted recalibration, risk-score compression, bin-level calibration error, and recalibrated decision curve analysis.

Supplementary figures:

- Figure S1: CTGDL-FHRMA external embedding/domain-shift validation, assigned SSL phenotypes, and CTU SSL risk gradient.
- Figure S2: transformer sweep, enrichment-calibration frontier, SSL phenotype gradient, and feature-family contribution.

Each figure is exported as PNG, PDF, SVG, and TIFF. Each figure has a corresponding CSV source-data table plus `manuscript_figure_manifest.csv`.

## External Validation

CTGDL-FHRMA is connected as an external representation validation source. It shows substantial domain shift from CTU-UHB, with CTU-vs-FHRMA domain classifier AUROC 0.9088. This should be framed as external representation/domain-shift validation, not external neonatal outcome validation.

## Manuscript Implication

The current data support this framing:

"Open intrapartum CTG time-series can support reproducible dynamic phenotyping and neonatal risk enrichment. A PatchTST-style self-supervised encoder adds precision-recall value beyond classical signal features, while clinically engineered signal phenotypes provide interpretable dynamic risk prototypes. Raw risk scores require recalibration for absolute probability use; cross-fitted Platt scaling substantially improves calibration without changing discrimination. External CTGDL validation currently demonstrates representation transport and domain shift rather than outcome-level generalization."

The next work should not be bigger-model training. It should be figure legends, Results text, Methods text, and a clear STARD/TRIPOD-style validation narrative around the existing manuscript-ready figures.
