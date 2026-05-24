# Skills Integration Plan for CTG Foundation Trajectory

Date: 2026-05-21

## Path Check

`E:\Reserch\SkillsE` does not exist on this machine. The available local skill root is:

`E:\Reserch\Skills`

I inspected the relevant local skill folders and the CNS method database. The conclusion is that no local skill provides a ready-made obstetric CTG foundation-model workflow. The useful parts are validation design, prediction explanation, figure export, Nature-style writing, data availability, and high-impact-method framing.

## High-Value Integrations

| Local skill/source | Fit for CTG project | How to use it |
|---|---|---|
| `bioSkills_learning/source_repo/machine-learning/model-validation` | High | Add repeated record-level stratified validation, keep all windows from one CTG record in the same fold, and report optimism-resistant performance. |
| `bioSkills_learning/source_repo/machine-learning/prediction-explanation` | High | Add SHAP for signal-feature and signal+SSL models; combine with time-window saliency for transformer embeddings. |
| `bioSkills_learning/source_repo/reporting/figure-export` | High | Export manuscript figures as PDF/SVG/PNG/TIFF with journal-sized dimensions and consistent typography. |
| `nature-skills_learning/source_repo/skills/nature-writing` | High | Rebuild abstract, introduction, results narrative, and claim-evidence map from the actual bootstrap and prototype evidence. |
| `nature-skills_learning/source_repo/skills/nature-data` | High | Prepare Data Availability and Code Availability statements for CTU-UHB, CTGDL, UCI CTG, processed tables, model weights, and scripts. |
| `nature-skills_learning/source_repo/skills/nature-academic-search` | Medium-high | Run a formal novelty and prior-art search around self-supervised CTG, fetal monitoring AI, dynamic phenotyping, foundation models, and external validation. |
| `paper-framework-figure-studio-pro_learning` | Medium-high | Build a publication framework figure: open CTG sources -> masked time-series encoder -> dynamic phenotypes -> neonatal risk validation -> external domain shift. |
| `cns_bioinfo_methods_database` | Medium | Use its high-impact method audit logic, benchmark framing, and code-first reproducibility discipline. Direct method reuse is limited because the database is mostly single-cell/spatial/multi-omics. |
| `ovarian_adc_visual_bioinfo_skillpack` | Low-medium | Do not reuse ovarian cancer biology. Reuse only the visual QC habits: panel contracts, claim-to-figure mapping, and CNS-style figure review. |

## CNS Method Database Query Result

The local CNS method database was queried for `foundation`, `transformer`, `trajectory`, `benchmark`, `representation`, `model`, `atlas`, and `clinical`.

Direct matches for CTG foundation modeling were not found. Transferable hits were limited to broad method categories:

- `benchmark`: benchmarking and reproducibility logic.
- `atlas`: resource/atlas construction logic.

This means the CTG project should not claim that it is based on a copied CNS method. The better strategy is to borrow the manuscript architecture:

1. Open multi-source data.
2. Self-supervised representation learning.
3. Fixed train/test and bootstrap validation.
4. Interpretable phenotype prototypes.
5. External representation/domain-shift validation.
6. Code-first reproducibility package.

## Recommended Additions to the CTG Project

### 1. Stability Validation Module

Add a script such as:

`scripts/04_models/repeated_group_validation.py`

Purpose:

- Repeated record-level stratified splits.
- Optional nested model selection for logistic/XGBoost heads.
- Keep window aggregation and phenotype generation training-only inside each fold when feasible.
- Report AUROC, AUPRC, Brier, calibration slope/intercept, and CI across repeats.

This uses the `model-validation` skill logic.

### 2. Prediction Explanation Module

Add a script such as:

`scripts/04_models/explain_predictions.py`

Purpose:

- SHAP for signal-feature and signal+SSL logistic models.
- Global feature importance table.
- Per-record explanation for representative high-risk and low-risk cases.
- Link tabular importance to prototype CTG curves.

This uses the `prediction-explanation` skill logic. If SHAP is not installed, fallback to permutation importance.

### 3. Manuscript Figure Export Module

Add a script such as:

`scripts/05_reports/export_manuscript_figures.py`

Purpose:

- Convert current figures into journal-size PDF/PNG/TIFF outputs.
- Build multi-panel figures:
  - Figure 1: study workflow and model architecture.
  - Figure 2: bootstrap model performance.
  - Figure 3: signal and SSL phenotype structure.
  - Figure 4: phenotype prototype CTG curves.
  - Figure 5: CTGDL external representation/domain shift.

This uses the `figure-export` and figure-framework logic.

### 4. Data and Code Availability Package

Add:

`docs/data_code_availability.md`

Purpose:

- Public datasets and access URLs.
- Raw data used.
- Processed data generated.
- Model weights generated.
- Code/scripts.
- Restricted-data caveats for unavailable CTGDL SPaM/DUA material.

This uses the `nature-data` logic.

### 5. Prior-Art Search Package

Add:

`docs/prior_art_search_strategy.md`

Purpose:

- PubMed query blocks.
- Key concepts:
  - cardiotocography
  - fetal heart rate
  - self-supervised learning
  - foundation model
  - dynamic phenotyping
  - neonatal acidemia
  - external validation
- Inclusion/exclusion rules.
- Prior-art risk table.

This uses the `nature-academic-search` logic.

## What Not To Force

- Do not force single-cell/spatial algorithms into CTG signal analysis.
- Do not use ovarian ADC biological modules for this project.
- Do not call the CTG model a true clinical foundation model unless pretraining scale, external datasets, and downstream task breadth are substantially expanded.
- Do not frame CTGDL as external outcome validation unless outcome labels are available and harmonized.

## Practical Priority

The next best implementation order is:

1. `data_code_availability.md`
2. `explain_predictions.py`
3. `export_manuscript_figures.py`
4. `repeated_group_validation.py`
5. `prior_art_search_strategy.md`

This sequence helps the project become manuscript-ready faster than another round of model scaling.
