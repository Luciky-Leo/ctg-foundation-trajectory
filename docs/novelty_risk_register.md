# Novelty and Risk Register

## Main Novelty

The strongest novelty is not "deep learning for CTG". That already exists.

The intended novelty is:

1. Self-supervised CTG representation learning.
2. Dynamic phenotype discovery rather than only binary classification.
3. FHR and UC joint modeling.
4. Cross-dataset validation or harmonized replication.
5. Clinical interpretability and subgroup robustness.

## Major Risks

| Risk | Impact | Mitigation |
|---|---:|---|
| CTU-UHB has only 552 records | High | Use self-supervised design, careful validation, and avoid overclaiming foundation-model scale |
| External datasets have incompatible outcomes | High | Separate representation validation, feature validation, and outcome validation |
| CTGDL subset terms differ | Medium | Only use subsets with compatible terms and document excluded subsets |
| pH is an imperfect neonatal outcome | Medium | Analyze multiple thresholds and composite outcomes |
| Deep model underperforms features | Medium | Reframe around dynamic phenotypes and interpretability if predictive gain is modest |
| Public-only paper ceiling | Medium | Prepare optional local validation extension |

## Claims to Avoid

1. Do not claim clinical deployment readiness from public data alone.
2. Do not call the model a general medical foundation model.
3. Do not overstate causality from prediction.
4. Do not claim fairness for unavailable demographic axes.
5. Do not present Shiny visualization as the primary innovation.

