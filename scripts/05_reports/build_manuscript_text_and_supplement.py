from pathlib import Path

import pandas as pd


TEXT_DIR = Path("results/manuscript/text")
SUPP_DIR = Path("results/manuscript/supplementary_tables")

MODEL_LABELS = {
    "signal_features": "Signal features",
    "ssl_embeddings": "SSL embeddings",
    "signal_plus_ssl": "Signal + SSL",
    "signal_phenotype_only": "Signal phenotype only",
    "ssl_phenotype_only": "SSL phenotype only",
    "signal_plus_signal_phenotype": "Signal + signal phenotype",
    "signal_plus_ssl_phenotype": "Signal + SSL phenotype",
    "signal_plus_ssl_plus_signal_phenotype": "Signal + SSL + signal phenotype",
}


def read_csv(path):
    return pd.read_csv(path)


def fmt(value, digits=4):
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def metric_text(df, model, metric):
    row = df[(df["model"] == model) & (df["metric"] == metric)].iloc[0]
    return f"{fmt(row['point'])} (95% CI {fmt(row['ci_2_5'])}-{fmt(row['ci_97_5'])})"


def diff_text(df, comparison, metric):
    row = df[(df["comparison"] == comparison) & (df["metric"] == metric)].iloc[0]
    return f"{fmt(row['bootstrap_mean_difference'])} (95% CI {fmt(row['ci_2_5'])}-{fmt(row['ci_97_5'])})"


def calibration_text(df, model, method, metric):
    row = df[
        (df["model"] == model)
        & (df["calibration_method"] == method)
        & (df["metric"] == metric)
    ].iloc[0]
    return f"{fmt(row['point'])} (95% CI {fmt(row['ci_2_5'])}-{fmt(row['ci_97_5'])})"


def build_supplementary_tables():
    SUPP_DIR.mkdir(parents=True, exist_ok=True)

    registry = read_csv("config/dataset_registry.csv")
    split = read_csv("data/processed/ctu_uhb_record_split.csv")
    ctu_windows = read_csv("data/processed/ctu_uhb_window_index.csv")
    ctu_windows_with_split = ctu_windows.merge(split[["record_id", "split"]], on="record_id", how="left")
    ctgdl_windows = read_csv("data/processed/ctgdl_fhrma_window_index.csv")
    features = read_csv("data/processed/ctu_uhb_record_features.csv")
    features_with_split = features.merge(split[["record_id", "split"]], on="record_id", how="left")
    external = read_csv("results/tables/ctgdl_external_embedding_validation.csv")

    ctu_rows = pd.DataFrame(
        [
            {
                "dataset_or_subset": "CTU-UHB complete cohort",
                "role": "primary raw CTG cohort",
                "records": len(split),
                "windows": len(ctu_windows),
                "neonatal_risk_rate": features["neonatal_risk"].mean(),
                "notes": "Open intrapartum FHR and uterine contraction signals with clinical metadata.",
            },
            {
                "dataset_or_subset": "CTU-UHB training split",
                "role": "model development",
                "records": int((split["split"] == "train").sum()),
                "windows": int(ctu_windows_with_split[ctu_windows_with_split["split"] == "train"].shape[0]),
                "neonatal_risk_rate": features_with_split.query("split == 'train'")["neonatal_risk"].mean(),
                "notes": "Record-level split to avoid window leakage.",
            },
            {
                "dataset_or_subset": "CTU-UHB held-out test split",
                "role": "internal validation",
                "records": int((split["split"] == "test").sum()),
                "windows": int(ctu_windows_with_split[ctu_windows_with_split["split"] == "test"].shape[0]),
                "neonatal_risk_rate": features_with_split.query("split == 'test'")["neonatal_risk"].mean(),
                "notes": "Held-out record-level test set for manuscript metrics.",
            },
            {
                "dataset_or_subset": "CTGDL-FHRMA",
                "role": "external representation/domain-shift validation",
                "records": int(ctgdl_windows["record_id"].nunique()),
                "windows": len(ctgdl_windows),
                "neonatal_risk_rate": "",
                "notes": "Used for embedding transport and phenotype assignment, not external outcome validation.",
            },
        ]
    )
    registry_export = registry.rename(columns={"dataset": "dataset_or_subset"})
    table_s1 = pd.concat([ctu_rows, registry_export], ignore_index=True, sort=False)
    table_s1.to_csv(SUPP_DIR / "Supplementary_Table_1_datasets_and_cohort_scale.csv", index=False)

    boot = read_csv("results/tables/bootstrap_validation_ci.csv")
    table_s2 = boot.copy()
    table_s2.insert(1, "model_label", table_s2["model"].map(MODEL_LABELS).fillna(table_s2["model"]))
    table_s2.to_csv(SUPP_DIR / "Supplementary_Table_2_model_performance_bootstrap.csv", index=False)

    diffs = read_csv("results/tables/bootstrap_model_differences.csv")
    table_s3 = diffs.copy()
    table_s3["ci_excludes_zero"] = (table_s3["ci_2_5"] > 0) | (table_s3["ci_97_5"] < 0)
    table_s3.to_csv(SUPP_DIR / "Supplementary_Table_3_bootstrap_model_differences.csv", index=False)

    signal = read_csv("results/tables/ctu_uhb_signal_phenotype_summary.csv")
    signal.insert(0, "phenotype_type", "signal_phenotype")
    signal = signal.rename(columns={"signal_phenotype": "phenotype"})
    ssl = read_csv("results/tables/ctu_uhb_ssl_phenotype_summary.csv")
    ssl.insert(0, "phenotype_type", "ssl_phenotype")
    ssl = ssl.rename(columns={"ssl_phenotype": "phenotype"})
    proto = read_csv("results/tables/phenotype_prototype_records.csv")
    pheno = pd.concat([signal, ssl], ignore_index=True, sort=False)
    table_s4 = pheno.merge(proto, on=["phenotype_type", "phenotype"], how="left", suffixes=("", "_prototype"))
    table_s4.to_csv(SUPP_DIR / "Supplementary_Table_4_dynamic_phenotype_summary_and_prototypes.csv", index=False)

    family = read_csv("results/tables/permutation_feature_family_importance.csv")
    features_importance = read_csv("results/tables/permutation_feature_importance.csv")
    top_features = (
        features_importance.sort_values(["model", "importance_mean_auprc"], ascending=[True, False])
        .groupby("model")
        .head(15)
        .copy()
    )
    family["table_block"] = "feature_family_importance"
    top_features["table_block"] = "top_feature_importance"
    table_s5 = pd.concat([family, top_features], ignore_index=True, sort=False)
    table_s5.to_csv(SUPP_DIR / "Supplementary_Table_5_prediction_explainability.csv", index=False)

    miss_metrics = read_csv("results/tables/missingness_sensitivity_metrics.csv")
    miss_diffs = read_csv("results/tables/missingness_sensitivity_differences.csv")
    miss_metrics["table_block"] = "scenario_metrics"
    miss_diffs["table_block"] = "scenario_differences"
    table_s6 = pd.concat([miss_metrics, miss_diffs], ignore_index=True, sort=False)
    table_s6.to_csv(SUPP_DIR / "Supplementary_Table_6_missingness_sensitivity.csv", index=False)

    recal = read_csv("results/tables/recalibration_metrics_ci.csv")
    recal_diffs = read_csv("results/tables/recalibration_metric_differences.csv")
    dca_summary = read_csv("results/tables/recalibrated_decision_curve_summary.csv")
    recal["table_block"] = "recalibration_metrics"
    recal_diffs["table_block"] = "recalibration_differences"
    dca_summary["table_block"] = "decision_curve_summary"
    table_s7 = pd.concat([recal, recal_diffs, dca_summary], ignore_index=True, sort=False)
    table_s7.to_csv(SUPP_DIR / "Supplementary_Table_7_recalibration_and_decision_curve.csv", index=False)

    assigned = read_csv("results/tables/ctgdl_fhrma_assigned_ssl_phenotypes.csv")
    assigned_counts = assigned["assigned_ctu_ssl_phenotype"].value_counts().sort_index().reset_index()
    assigned_counts.columns = ["assigned_ctu_ssl_phenotype", "n_records"]
    external["table_block"] = "external_embedding_metrics"
    assigned_counts["table_block"] = "assigned_ssl_phenotype_counts"
    table_s8 = pd.concat([external, assigned_counts], ignore_index=True, sort=False)
    table_s8.to_csv(SUPP_DIR / "Supplementary_Table_8_external_representation_validation.csv", index=False)

    index = pd.DataFrame(
        [
            {
                "table": "Supplementary Table 1",
                "file": "Supplementary_Table_1_datasets_and_cohort_scale.csv",
                "description": "Dataset sources, cohort scale, record-level split, window counts, and external validation role.",
            },
            {
                "table": "Supplementary Table 2",
                "file": "Supplementary_Table_2_model_performance_bootstrap.csv",
                "description": "Held-out model performance with 2000-bootstrap confidence intervals.",
            },
            {
                "table": "Supplementary Table 3",
                "file": "Supplementary_Table_3_bootstrap_model_differences.csv",
                "description": "Bootstrap pairwise model differences and zero-crossing flags.",
            },
            {
                "table": "Supplementary Table 4",
                "file": "Supplementary_Table_4_dynamic_phenotype_summary_and_prototypes.csv",
                "description": "Signal and SSL phenotype summaries with representative prototype records.",
            },
            {
                "table": "Supplementary Table 5",
                "file": "Supplementary_Table_5_prediction_explainability.csv",
                "description": "Permutation importance by feature family and top individual predictors.",
            },
            {
                "table": "Supplementary Table 6",
                "file": "Supplementary_Table_6_missingness_sensitivity.csv",
                "description": "Performance and model-difference sensitivity after removing missingness features.",
            },
            {
                "table": "Supplementary Table 7",
                "file": "Supplementary_Table_7_recalibration_and_decision_curve.csv",
                "description": "Cross-fitted recalibration metrics, recalibration differences, and decision curve summaries.",
            },
            {
                "table": "Supplementary Table 8",
                "file": "Supplementary_Table_8_external_representation_validation.csv",
                "description": "CTGDL-FHRMA domain-shift and phenotype-assignment validation.",
            },
        ]
    )
    index.to_csv(SUPP_DIR / "Supplementary_Tables_Index.csv", index=False)


def build_manuscript_text():
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    boot = read_csv("results/tables/bootstrap_validation_ci.csv")
    diffs = read_csv("results/tables/bootstrap_model_differences.csv")
    missing = read_csv("results/tables/missingness_sensitivity_metrics.csv")
    missing_diffs = read_csv("results/tables/missingness_sensitivity_differences.csv")
    signal = read_csv("results/tables/ctu_uhb_signal_phenotype_summary.csv")
    external = read_csv("results/tables/ctgdl_external_embedding_validation.csv")
    recal = read_csv("results/tables/recalibration_metrics_ci.csv")
    recal_diffs = read_csv("results/tables/recalibration_metric_differences.csv")
    dca_summary = read_csv("results/tables/recalibrated_decision_curve_summary.csv")
    family = read_csv("results/tables/permutation_feature_family_importance.csv")

    signal_plus_ssl_auprc = metric_text(boot, "signal_plus_ssl", "auprc")
    signal_only_auprc = metric_text(boot, "signal_features", "auprc")
    final_auroc = metric_text(boot, "signal_plus_ssl_plus_signal_phenotype", "auroc")
    final_auprc = metric_text(boot, "signal_plus_ssl_plus_signal_phenotype", "auprc")
    auprc_diff_ssl = diff_text(diffs, "signal_plus_ssl minus signal_features", "auprc")
    auprc_diff_final = diff_text(diffs, "signal_plus_ssl_plus_signal_phenotype minus signal_features", "auprc")

    no_miss_ssl = missing[
        (missing["scenario"] == "no_missingness_features")
        & (missing["model"] == "signal_plus_ssl")
        & (missing["metric"] == "auprc")
    ].iloc[0]
    no_miss_final = missing[
        (missing["scenario"] == "no_missingness_features")
        & (missing["model"] == "signal_plus_ssl_plus_signal_phenotype")
        & (missing["metric"] == "auprc")
    ].iloc[0]
    no_miss_diff = missing_diffs[
        (missing_diffs["comparison_type"] == "model_difference_within_scenario")
        & (missing_diffs["scenario"] == "no_missingness_features")
        & (missing_diffs["comparison"] == "signal_plus_ssl_plus_signal_phenotype minus signal_features")
        & (missing_diffs["metric"] == "auprc")
    ].iloc[0]

    high_pheno = signal.sort_values("neonatal_risk_mean", ascending=False).iloc[0]
    low_pheno = signal.sort_values("neonatal_risk_mean", ascending=True).iloc[0]

    raw_brier = calibration_text(recal, "signal_plus_ssl_plus_signal_phenotype", "raw", "brier")
    platt_brier = calibration_text(recal, "signal_plus_ssl_plus_signal_phenotype", "platt", "brier")
    raw_ece = calibration_text(recal, "signal_plus_ssl_plus_signal_phenotype", "raw", "expected_calibration_error")
    platt_ece = calibration_text(recal, "signal_plus_ssl_plus_signal_phenotype", "platt", "expected_calibration_error")
    platt_brier_diff = recal_diffs[
        (recal_diffs["model"] == "signal_plus_ssl_plus_signal_phenotype")
        & (recal_diffs["comparison"] == "platt minus raw")
        & (recal_diffs["metric"] == "brier")
    ].iloc[0]
    platt_ece_diff = recal_diffs[
        (recal_diffs["model"] == "signal_plus_ssl_plus_signal_phenotype")
        & (recal_diffs["comparison"] == "platt minus raw")
        & (recal_diffs["metric"] == "expected_calibration_error")
    ].iloc[0]
    dca_raw = dca_summary[
        (dca_summary["threshold_range"] == "0.10-0.30")
        & (dca_summary["strategy"] == "signal_plus_ssl_plus_signal_phenotype_raw")
    ].iloc[0]
    dca_platt = dca_summary[
        (dca_summary["threshold_range"] == "0.10-0.30")
        & (dca_summary["strategy"] == "signal_plus_ssl_plus_signal_phenotype_platt")
    ].iloc[0]

    external_map = dict(zip(external["metric"], external["value"]))
    clinical_imp = family[
        (family["model"] == "signal_plus_ssl") & (family["feature_family"] == "clinical_signal")
    ].iloc[0]
    ssl_imp = family[
        (family["model"] == "signal_plus_ssl") & (family["feature_family"] == "ssl_embedding")
    ].iloc[0]

    text = f"""# Manuscript Draft: Figure Legends, Results, Methods, and Supplementary Tables

Date: 2026-05-23.

This draft is organized in figure order. It uses the current public-data CTG foundation-style trajectory analysis and should be checked against journal word limits before submission.

## Figure Legends

### Figure 1. Study design and evidence chain for open-data CTG dynamic phenotyping

Overview of the open-data intrapartum CTG analysis workflow. (A) CTU-UHB/CTU-CHB was used as the primary raw-signal cohort for fetal heart rate (FHR) and uterine contraction (UC) modelling. CTGDL-FHRMA and UCI Cardiotocography were retained as external representation and feature-level resources. Records were split at the record level into 386 training records and 166 held-out test records. Ten-minute FHR and UC windows were used to train a masked time-series encoder, derive dynamic phenotypes, and evaluate neonatal risk enrichment, calibration, and clinical utility. (B) Cohort scale for CTU-UHB training and test records and CTGDL-FHRMA external records. (C) Outcome balance in the record-level training and held-out test split. (D) Evidence map showing which modules support outcome prediction, interpretation, robustness, and external-domain claims.

### Figure 2. Self-supervised CTG representations improve risk enrichment and remain interpretable

Held-out validation and explanation of neonatal risk prediction models. (A) Precision-recall curves in the CTU-UHB held-out test set. (B) Receiver-operating-characteristic curves in the held-out test set. (C) Bootstrap validation of AUROC and AUPRC; points show observed test-set estimates and whiskers show 95% bootstrap confidence intervals from 2,000 resamples. (D) Bootstrap AUPRC differences versus signal-only features. (E) Missingness sensitivity after removing FHR and UC missingness fractions from the feature set. (F) Top permutation-importance features for the signal-plus-SSL model, ranked by mean AUPRC decrease. SSL, self-supervised learning.

### Figure 3. Dynamic signal phenotypes provide clinically interpretable CTG prototypes

Signal-derived phenotype structure and representative CTG trajectories. (A) Neonatal risk rates across four signal phenotypes. (B) Mean cord pH and 5-minute Apgar profile across phenotypes. (C) Standardized phenotype profile heatmap summarizing outcome and signal differences. (D) Mean FHR standard deviation and deceleration-proxy burden across signal phenotypes. (E-H) Representative prototype records closest to the phenotype centroids. Phenotype 2 showed the highest neonatal risk rate and the strongest adverse signal profile, including higher FHR variability and deceleration-proxy burden.

### Figure 4. Cross-fitted Platt recalibration improves absolute risk calibration and decision-curve behavior

Calibration and clinical utility of the final signal-plus-SSL-plus-signal-phenotype model. (A) Calibration curves for raw, Platt-scaled, and isotonic predictions. Recalibrators were fitted using training-set out-of-fold predictions and evaluated in the held-out test set. (B) Brier score and expected calibration error by calibration method. (C) Risk-score distributions before and after Platt recalibration, stratified by observed outcome. (D) Bin-level absolute calibration error. (E) Decision curve analysis across risk thresholds. (F) Mean net benefit in the 0.10-0.30 threshold range. Platt scaling improved calibration while preserving discrimination and AUPRC.

### Figure S1. CTGDL-FHRMA supports external representation and domain-shift validation

External representation analysis using CTGDL-FHRMA records. (A) Principal-component visualization of CTU-UHB and CTGDL-FHRMA transformer embeddings. (B) Domain separability quantified by a CTU-versus-CTGDL classifier. (C) CTGDL-FHRMA records assigned to CTU-derived SSL phenotypes. (D) CTU-UHB neonatal risk gradient across SSL phenotypes. This analysis evaluates representation transport and domain shift, not external neonatal outcome prediction.

### Figure S2. Model-selection and sensitivity panels for the transformer phenotyping workflow

Supplementary model-selection and sensitivity analyses. (A) Top transformer configuration and feature-set combinations ranked by held-out AUPRC. (B) Enrichment-calibration frontier comparing AUPRC with Brier score across model families. (C) CTU-UHB neonatal risk gradient across SSL phenotypes. (D) Feature-family contribution to AUPRC-based permutation importance for signal-plus-SSL and final phenotype-augmented models.

## Results

### Open CTG data supported a reproducible dynamic phenotyping workflow

The analysis used CTU-UHB/CTU-CHB as the primary raw-signal cohort and retained CTGDL-FHRMA for external representation validation. CTU-UHB records were split at the record level into 386 training records and 166 held-out test records. This design avoided leakage between training and validation windows and supported a reproducible pipeline from raw FHR/UC signals to dynamic representations, phenotypes, risk prediction, and calibration analysis (Fig. 1).

### Self-supervised representations improved precision-recall enrichment beyond classical signal summaries

In the held-out CTU-UHB test set, signal-only features achieved an AUPRC of {signal_only_auprc}. Adding transformer SSL embeddings increased AUPRC to {signal_plus_ssl_auprc}. The bootstrap AUPRC difference for signal plus SSL versus signal-only features was {auprc_diff_ssl}. The final signal-plus-SSL-plus-signal-phenotype model achieved an AUROC of {final_auroc} and an AUPRC of {final_auprc}. Its AUPRC difference versus signal-only features was {auprc_diff_final}. In contrast, AUROC and Brier-score differences crossed zero, indicating that the main stable gain was risk enrichment under class imbalance rather than global discrimination or raw calibration (Fig. 2A-D; Fig. S2A,B).

Permutation analysis supported a mixed clinical and representation-based signal. In the signal-plus-SSL model, the positive AUPRC importance sum was {fmt(clinical_imp['positive_importance_sum_auprc'])} for clinical signal features and {fmt(ssl_imp['positive_importance_sum_auprc'])} for SSL embedding dimensions. Mean positive AUPRC importance per feature was similar across the two families ({fmt(clinical_imp['positive_importance_mean_auprc'])} and {fmt(ssl_imp['positive_importance_mean_auprc'])}, respectively), suggesting that the gain reflected multiple embedding dimensions rather than a single dominant latent variable (Fig. 2F; Fig. S2D).

Because FHR missingness contributed to prediction, models were rerun after excluding FHR and UC missingness fractions. Without missingness features, signal plus SSL achieved an AUPRC of {fmt(no_miss_ssl['point'])} (95% CI {fmt(no_miss_ssl['ci_2_5'])}-{fmt(no_miss_ssl['ci_97_5'])}), and the final model achieved an AUPRC of {fmt(no_miss_final['point'])} (95% CI {fmt(no_miss_final['ci_2_5'])}-{fmt(no_miss_final['ci_97_5'])}). The no-missingness AUPRC difference for the final model versus signal-only features was {fmt(no_miss_diff['bootstrap_mean_difference'])} (95% CI {fmt(no_miss_diff['ci_2_5'])}-{fmt(no_miss_diff['ci_97_5'])}). These results show directionally retained enrichment but do not prove missingness-independent superiority (Fig. 2E).

### Signal phenotypes provided the clearest clinical interpretation anchor

Four signal-derived phenotypes showed clinically interpretable differences in neonatal risk and CTG signal profiles. Signal phenotype {int(high_pheno['signal_phenotype'])} had the highest neonatal risk rate ({fmt(high_pheno['neonatal_risk_mean'])}), lower mean cord pH ({fmt(high_pheno['cord_ph_mean'])}), higher FHR standard deviation ({fmt(high_pheno['fhr_sd_mean'])}), and higher deceleration-proxy burden ({fmt(high_pheno['deceleration_proxy_count_mean'])}). Signal phenotype {int(low_pheno['signal_phenotype'])} had the lowest neonatal risk rate ({fmt(low_pheno['neonatal_risk_mean'])}). Prototype records illustrated the dynamic FHR and UC patterns underlying each signal phenotype (Fig. 3). These findings support using signal phenotypes as the clinical interpretation layer for the SSL-enhanced model.

### Platt recalibration improved absolute risk estimates and threshold behavior

Raw predictions from the final signal-plus-SSL-plus-signal-phenotype model were poorly calibrated. The raw Brier score was {raw_brier}, and the raw expected calibration error was {raw_ece}. Cross-fitted Platt recalibration improved the Brier score to {platt_brier} and expected calibration error to {platt_ece}. Bootstrap differences for Platt minus raw predictions were {fmt(platt_brier_diff['bootstrap_mean_difference'])} for Brier score (95% CI {fmt(platt_brier_diff['ci_2_5'])}-{fmt(platt_brier_diff['ci_97_5'])}) and {fmt(platt_ece_diff['bootstrap_mean_difference'])} for expected calibration error (95% CI {fmt(platt_ece_diff['ci_2_5'])}-{fmt(platt_ece_diff['ci_97_5'])}). In the 0.10-0.30 threshold range, mean net benefit increased from {fmt(dca_raw['mean_net_benefit'])} for raw predictions to {fmt(dca_platt['mean_net_benefit'])} after Platt recalibration. Isotonic recalibration improved calibration metrics but reduced AUPRC, so it should remain exploratory (Fig. 4).

### CTGDL-FHRMA showed representation transport with substantial domain shift

External CTGDL-FHRMA analysis included {int(float(external_map['ctgdl_fhrma_records']))} records and was compared with {int(float(external_map['ctu_records']))} CTU-UHB records in the transformer embedding space. The first two principal components explained {fmt(external_map['pca_explained_variance_pc1_pc2'])} of embedding variance. A CTU-versus-CTGDL domain classifier achieved an AUROC of {fmt(external_map['domain_classifier_auroc_ctgdl_vs_ctu'])}, indicating substantial domain shift. CTGDL-FHRMA records were assigned across CTU-derived SSL phenotypes, with most assigned to phenotypes 1 and 3. Because neonatal outcome labels were not used for CTGDL-FHRMA, these analyses support external representation validation rather than external clinical outcome validation (Fig. S1). Transformer model-selection and SSL phenotype sensitivity panels are provided as supplementary evidence for the final figure package (Fig. S2).

## Methods

### Study design and data sources

This was a retrospective open-data modelling study of intrapartum cardiotocography. CTU-UHB/CTU-CHB was used as the primary raw-signal cohort because it provides FHR, UC, and clinical metadata for intrapartum records. CTGDL-FHRMA was used for external representation and domain-shift analysis. UCI Cardiotocography was retained as a feature-level reference resource but was not used as raw-signal outcome validation. Dataset URLs, access notes, and roles are summarized in Supplementary Table 1.

### Record split and signal windowing

Records were split at the record level into training and held-out test sets. This split was used before downstream validation to prevent leakage between overlapping windows from the same CTG record. FHR and UC signals were converted into fixed-length windows using a target sampling rate of 4 Hz and 10-minute windows. Record-level summaries and embeddings were later aggregated from window-level representations.

### Self-supervised time-series representation learning

A PatchTST-style masked time-series transformer was trained on FHR and UC windows. The selected manuscript-candidate configuration used patch length 80, model dimension 64, three transformer layers, four attention heads, channel masking, and a contrastive objective weight of 0.2. The encoder was trained with masked reconstruction and compact contrastive learning. After training, window embeddings were aggregated to record-level SSL representations.

### Dynamic phenotype derivation

Two complementary phenotype layers were derived. First, classical CTG signal summaries were clustered to produce clinically interpretable signal phenotypes. These summaries included FHR level, FHR variability, low-percentile FHR, UC summaries, FHR-UC correlation, and deceleration-proxy burden. Second, record-level SSL embeddings were clustered to produce representation phenotypes. Prototype records were selected as records closest to phenotype centroids and plotted as paired FHR/UC trajectories.

### Outcome and prediction models

The primary endpoint was a composite neonatal risk outcome derived from available CTU-UHB clinical metadata. Models compared classical signal features, SSL embeddings, signal phenotypes, SSL phenotypes, and combined feature sets. Logistic prediction models were evaluated in the fixed held-out test set. Model comparisons focused on AUROC, AUPRC, and Brier score.

### Bootstrap validation and model comparison

Held-out test-set performance was summarized using observed test-set metrics and 95% confidence intervals from 2,000 bootstrap resamples. Pairwise model differences were also estimated by bootstrap resampling of the same held-out records. The primary model-comparison emphasis was AUPRC because the neonatal risk outcome was class-imbalanced.

### Prediction explanation and missingness sensitivity

Permutation importance was computed for manuscript-candidate models using AUPRC and AUROC decreases after feature shuffling. Importance values were summarized by feature family and by individual predictor. Because FHR and UC missingness fractions could reflect both signal quality and clinical difficulty of monitoring, sensitivity models were refitted after excluding missingness fractions.

### Recalibration and decision curve analysis

Raw held-out predictions were assessed for calibration. Platt and isotonic recalibrators were fitted using training-set out-of-fold predictions and then applied to the fixed held-out test set. Calibration was evaluated using calibration curves, Brier score, calibration intercept, calibration slope, and expected calibration error. Decision curve analysis was used to summarize net benefit across clinically plausible risk thresholds, with emphasis on the 0.10-0.30 threshold range.

### External representation analysis

The trained transformer encoder was applied to CTGDL-FHRMA windows to obtain external record-level embeddings. CTU-UHB and CTGDL-FHRMA embeddings were compared using principal-component visualization and a domain classifier. CTGDL-FHRMA records were assigned to CTU-derived SSL phenotypes. Because external neonatal outcome labels were not used, this analysis was interpreted as representation and domain-shift validation rather than external outcome validation.

### Reproducibility

All figure and supplementary-table outputs were generated from project CSV artifacts using local scripts. Manuscript figures were generated by `scripts/05_reports/build_manuscript_figures.py`. Figure legends, Results text, Methods text, and supplementary tables were generated by `scripts/05_reports/build_manuscript_text_and_supplement.py`.

## Supplementary Tables

Supplementary tables are stored in `results/manuscript/supplementary_tables`.

- Supplementary Table 1: datasets and cohort scale.
- Supplementary Table 2: bootstrap model performance.
- Supplementary Table 3: bootstrap model differences.
- Supplementary Table 4: dynamic phenotype summaries and prototype records.
- Supplementary Table 5: prediction explainability.
- Supplementary Table 6: missingness sensitivity.
- Supplementary Table 7: recalibration and decision curve analysis.
- Supplementary Table 8: external representation validation.
"""

    (TEXT_DIR / "figure_legends_results_methods.md").write_text(text, encoding="utf-8")


def main():
    build_supplementary_tables()
    build_manuscript_text()
    print(f"Wrote manuscript text to {TEXT_DIR}")
    print(f"Wrote supplementary CSV tables to {SUPP_DIR}")


if __name__ == "__main__":
    main()
