import csv
from pathlib import Path


def count_files(path, pattern="*"):
    root = Path(path)
    if not root.exists():
        return 0
    return len([p for p in root.rglob(pattern) if p.is_file()])


def read_metrics(path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt_float(value, digits=4):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def metric_label(metric):
    return {
        "auroc": "AUROC",
        "auprc": "AUPRC",
        "brier": "Brier",
    }.get(metric, metric)


def fmt_ci(low, high):
    return f"{fmt_float(low)} to {fmt_float(high)}"


def count_csv_rows(path):
    p = Path(path)
    if not p.exists():
        return 0
    with p.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    return max(0, len(rows) - 1)


def main():
    metrics = read_metrics("results/tables/baseline_smoke_metrics.csv")
    real_subset_metrics = read_metrics("results/tables/ctu_uhb_subset_baseline_metrics.csv")
    real_full_metrics = read_metrics("results/tables/ctu_uhb_full_baseline_metrics.csv")
    downstream_metrics = read_metrics("results/tables/downstream_validation_metrics.csv")
    external_metrics = read_metrics("results/tables/ctgdl_external_embedding_validation.csv")
    bootstrap_metrics = read_metrics("results/tables/bootstrap_validation_ci.csv")
    bootstrap_diffs = read_metrics("results/tables/bootstrap_model_differences.csv")
    prototype_records = read_metrics("results/tables/phenotype_prototype_records.csv")
    family_importance = read_metrics("results/tables/permutation_feature_family_importance.csv")
    representative_records = read_metrics("results/tables/explanation_representative_records.csv")
    missingness_metrics = read_metrics("results/tables/missingness_sensitivity_metrics.csv")
    missingness_diffs = read_metrics("results/tables/missingness_sensitivity_differences.csv")
    calibration_metrics = read_metrics("results/tables/calibration_metrics_ci.csv")
    decision_summary_rows = read_metrics("results/tables/decision_curve_summary.csv")
    recalibration_metrics = read_metrics("results/tables/recalibration_metrics_ci.csv")
    recalibration_diffs = read_metrics("results/tables/recalibration_metric_differences.csv")
    recalibrated_dca_summary = read_metrics("results/tables/recalibrated_decision_curve_summary.csv")
    lines = [
        "# Project Status",
        "",
        "## Data Files",
        "",
        f"- CTU-UHB raw files: {count_files('data/raw/ctu_uhb')}",
        f"- CTGDL raw files: {count_files('data/raw/ctgdl')}",
        f"- UCI raw files: {count_files('data/raw/uci_ctg')}",
        f"- Example files: {count_files('data/example')}",
        f"- Processed files: {count_files('data/processed')}",
        f"- CTU-UHB parsed header rows: {count_csv_rows('data/interim/ctu_uhb_header_index.csv')}",
        f"- CTU-UHB long-format rows: {count_csv_rows('data/processed/ctu_uhb_subset_long.csv')}",
        f"- CTU-UHB full record-feature rows: {count_csv_rows('data/processed/ctu_uhb_record_features.csv')}",
        f"- CTU-UHB window-index rows: {count_csv_rows('data/processed/ctu_uhb_window_index.csv')}",
        f"- CTGDL FHRMA window-index rows: {count_csv_rows('data/processed/ctgdl_fhrma_window_index.csv')}",
        f"- SSL record-embedding rows: {count_csv_rows('results/tables/ssl_record_embeddings.csv')}",
        f"- CTGDL FHRMA record-embedding rows: {count_csv_rows('results/tables/ctgdl_fhrma_record_embeddings.csv')}",
        f"- SSL phenotype rows: {count_csv_rows('results/tables/ctu_uhb_ssl_phenotypes.csv')}",
        f"- Signal phenotype rows: {count_csv_rows('results/tables/ctu_uhb_signal_phenotypes.csv')}",
        f"- Manuscript test-prediction rows: {count_csv_rows('results/tables/manuscript_test_predictions.csv')}",
        f"- Bootstrap CI rows: {count_csv_rows('results/tables/bootstrap_validation_ci.csv')}",
        f"- Phenotype prototype records: {count_csv_rows('results/tables/phenotype_prototype_records.csv')}",
        f"- Phenotype prototype figures: {count_files('results/figures/phenotype_prototypes', '*.png')}",
        f"- Permutation feature-importance rows: {count_csv_rows('results/tables/permutation_feature_importance.csv')}",
        f"- Explanation representative-record rows: {count_csv_rows('results/tables/explanation_representative_records.csv')}",
        f"- Missingness sensitivity metric rows: {count_csv_rows('results/tables/missingness_sensitivity_metrics.csv')}",
        f"- Calibration metric rows: {count_csv_rows('results/tables/calibration_metrics_ci.csv')}",
        f"- Decision curve rows: {count_csv_rows('results/tables/decision_curve_net_benefit.csv')}",
        f"- Recalibration metric rows: {count_csv_rows('results/tables/recalibration_metrics_ci.csv')}",
        f"- Recalibrated prediction rows: {count_csv_rows('results/tables/recalibrated_test_predictions.csv')}",
        f"- Manuscript figure files: {count_files('results/manuscript/figures')}",
        f"- Manuscript source-data tables: {count_files('results/manuscript/source_data', '*.csv')}",
        f"- Manuscript text draft files: {count_files('results/manuscript/text')}",
        f"- Supplementary table CSV files: {count_files('results/manuscript/supplementary_tables', '*.csv')}",
        f"- Supplementary table workbook files: {count_files('results/manuscript/supplementary_tables', '*.xlsx')}",
        f"- LaTeX manuscript files: {count_files('results/manuscript/latex')}",
        "",
        "## Smoke-Test Metrics",
        "",
    ]
    if metrics:
        for row in metrics:
            lines.append(f"- {row['metric']}: {row['value']}")
    else:
        lines.append("- No smoke-test metrics found.")
    lines.extend([
        "",
        "## Real CTU-UHB Subset Metrics",
        "",
    ])
    if real_subset_metrics:
        for row in real_subset_metrics:
            lines.append(f"- {row['metric']}: {row['value']}")
    else:
        lines.append("- No real-subset metrics found.")
    lines.extend([
        "",
        "## Real CTU-UHB Full Metrics",
        "",
    ])
    if real_full_metrics:
        for row in real_full_metrics:
            lines.append(f"- {row['metric']}: {row['value']}")
    else:
        lines.append("- No full-dataset metrics found.")
    lines.extend([
        "",
        "## Previous Downstream Validation Artifact",
        "",
        "These values come from `results/tables/downstream_validation_metrics.csv` and are retained for traceability. Use the manuscript candidate bootstrap section below for the current preferred evidence summary.",
        "",
    ])
    if downstream_metrics:
        for row in downstream_metrics:
            lines.append(f"- {row['model']}: AUROC {row['auroc']}, AUPRC {row['auprc']}, Brier {row['brier']}")
    else:
        lines.append("- No downstream validation metrics found.")
    lines.extend([
        "",
        "## Manuscript Candidate Bootstrap Validation",
        "",
    ])
    if bootstrap_metrics:
        preferred_models = [
            "signal_features",
            "signal_plus_ssl",
            "signal_plus_ssl_plus_signal_phenotype",
        ]
        for model in preferred_models:
            rows = [row for row in bootstrap_metrics if row["model"] == model and row["metric"] in {"auroc", "auprc", "brier"}]
            metric_text = []
            for row in rows:
                metric_text.append(
                    f"{metric_label(row['metric'])} {fmt_float(row['point'])} "
                    f"({fmt_float(row['ci_2_5'])}-{fmt_float(row['ci_97_5'])})"
                )
            if metric_text:
                lines.append(f"- {model}: " + "; ".join(metric_text))
    else:
        lines.append("- No bootstrap validation metrics found.")
    if bootstrap_diffs:
        lines.extend([
            "",
            "Key bootstrap model differences:",
        ])
        for row in bootstrap_diffs:
            if row["metric"] == "auprc":
                lines.append(
                    f"- {row['comparison']} AUPRC difference: {fmt_float(row['bootstrap_mean_difference'])} "
                    f"({fmt_float(row['ci_2_5'])}-{fmt_float(row['ci_97_5'])})"
                )
    if prototype_records:
        lines.extend([
            "",
            "Phenotype prototype records:",
        ])
        for row in prototype_records:
            lines.append(
                f"- {row['phenotype_type']} {row['phenotype']}: record {row['record_id']}, "
                f"n={row['n_records_in_phenotype']}, prototype_record_neonatal_risk={row['neonatal_risk']}, "
                f"cord_pH={row['cord_ph']}"
            )
    lines.extend([
        "",
        "## Prediction Explanation",
        "",
    ])
    if family_importance:
        for row in family_importance:
            lines.append(
                f"- {row['model']} / {row['feature_family']}: "
                f"AUPRC positive sum {fmt_float(row['positive_importance_sum_auprc'])}, "
                f"mean per feature {fmt_float(row['positive_importance_mean_auprc'])}, "
                f"n={row['n_features']}"
            )
    else:
        lines.append("- No permutation feature-family importance found.")
    if representative_records:
        top = representative_records[0]
        lines.append(
            f"- Top representative high-risk record for signal_plus_ssl: record {top['record_id']}, "
            f"predicted risk {fmt_float(top['probability'])}, neonatal_risk={top['neonatal_risk']}, "
            f"cord_pH={top['cord_ph']}"
        )
    lines.extend([
        "",
        "## Missingness Sensitivity",
        "",
    ])
    if missingness_metrics:
        for scenario in ["full_features", "no_missingness_features"]:
            for model in ["signal_features", "signal_plus_ssl", "signal_plus_ssl_plus_signal_phenotype"]:
                rows = [
                    row for row in missingness_metrics
                    if row["scenario"] == scenario and row["model"] == model and row["metric"] == "auprc"
                ]
                if rows:
                    row = rows[0]
                    lines.append(
                        f"- {scenario} / {model}: AUPRC {fmt_float(row['point'])} "
                        f"({fmt_float(row['ci_2_5'])}-{fmt_float(row['ci_97_5'])})"
                    )
    else:
        lines.append("- No missingness sensitivity metrics found.")
    if missingness_diffs:
        for row in missingness_diffs:
            if (
                row["comparison_type"] == "model_difference_within_scenario"
                and row["scenario"] == "no_missingness_features"
                and row["metric"] == "auprc"
            ):
                lines.append(
                    f"- No-missingness {row['comparison']} AUPRC difference: "
                    f"{fmt_float(row['bootstrap_mean_difference'])} "
                    f"({fmt_ci(row['ci_2_5'], row['ci_97_5'])})"
                )
    lines.extend([
        "",
        "## Calibration and Decision Curves",
        "",
    ])
    if calibration_metrics:
        for model in ["signal_features", "signal_plus_ssl", "signal_plus_ssl_plus_signal_phenotype"]:
            slope = next(
                (
                    row for row in calibration_metrics
                    if row["model"] == model and row["metric"] == "calibration_slope"
                ),
                None,
            )
            intercept = next(
                (
                    row for row in calibration_metrics
                    if row["model"] == model and row["metric"] == "calibration_intercept"
                ),
                None,
            )
            ece = next(
                (
                    row for row in calibration_metrics
                    if row["model"] == model and row["metric"] == "expected_calibration_error"
                ),
                None,
            )
            if slope and intercept and ece:
                lines.append(
                    f"- {model}: calibration slope {fmt_float(slope['point'])} "
                    f"({fmt_ci(slope['ci_2_5'], slope['ci_97_5'])}), "
                    f"intercept {fmt_float(intercept['point'])}, "
                    f"ECE {fmt_float(ece['point'])}"
                )
    else:
        lines.append("- No calibration metrics found.")
    if decision_summary_rows:
        lines.append("")
        lines.append("Decision curve mean net benefit, threshold range 0.10-0.30:")
        for row in decision_summary_rows:
            if row["threshold_range"] == "0.10-0.30" and row["strategy"] in [
                "signal_features",
                "signal_plus_ssl",
                "signal_plus_ssl_plus_signal_phenotype",
            ]:
                delta = row.get("mean_delta_vs_signal_features", "")
                suffix = f", delta vs signal {fmt_float(delta)}" if delta not in ("", None) else ""
                lines.append(
                    f"- {row['strategy']}: mean net benefit {fmt_float(row['mean_net_benefit'])}{suffix}"
                )
    lines.extend([
        "",
        "## Cross-Fitted Recalibration",
        "",
    ])
    if recalibration_metrics:
        for model in ["signal_plus_ssl", "signal_plus_ssl_plus_signal_phenotype"]:
            for method in ["raw", "platt", "isotonic"]:
                brier = next(
                    (
                        row for row in recalibration_metrics
                        if row["model"] == model and row["calibration_method"] == method and row["metric"] == "brier"
                    ),
                    None,
                )
                slope = next(
                    (
                        row for row in recalibration_metrics
                        if row["model"] == model and row["calibration_method"] == method and row["metric"] == "calibration_slope"
                    ),
                    None,
                )
                ece = next(
                    (
                        row for row in recalibration_metrics
                        if row["model"] == model and row["calibration_method"] == method and row["metric"] == "expected_calibration_error"
                    ),
                    None,
                )
                auprc = next(
                    (
                        row for row in recalibration_metrics
                        if row["model"] == model and row["calibration_method"] == method and row["metric"] == "auprc"
                    ),
                    None,
                )
                if brier and slope and ece and auprc:
                    lines.append(
                        f"- {model} / {method}: AUPRC {fmt_float(auprc['point'])}, "
                        f"Brier {fmt_float(brier['point'])}, slope {fmt_float(slope['point'])}, "
                        f"ECE {fmt_float(ece['point'])}"
                    )
    else:
        lines.append("- No cross-fitted recalibration metrics found.")
    if recalibration_diffs:
        lines.append("")
        lines.append("Platt recalibration differences versus raw:")
        for row in recalibration_diffs:
            if (
                row["comparison"] == "platt minus raw"
                and row["model"] in ["signal_plus_ssl", "signal_plus_ssl_plus_signal_phenotype"]
                and row["metric"] in ["brier", "expected_calibration_error"]
            ):
                lines.append(
                    f"- {row['model']} {row['metric']}: {fmt_float(row['bootstrap_mean_difference'])} "
                    f"({fmt_ci(row['ci_2_5'], row['ci_97_5'])})"
                )
    if recalibrated_dca_summary:
        lines.append("")
        lines.append("Recalibrated decision curve mean net benefit, threshold range 0.10-0.30:")
        for row in recalibrated_dca_summary:
            if row["threshold_range"] == "0.10-0.30" and row["strategy"] in [
                "signal_plus_ssl_plus_signal_phenotype_raw",
                "signal_plus_ssl_plus_signal_phenotype_platt",
                "signal_plus_ssl_plus_signal_phenotype_isotonic",
            ]:
                delta = row.get("mean_delta_vs_raw_signal_plus_ssl_plus_signal_phenotype", "")
                suffix = f", delta vs raw {fmt_float(delta)}" if delta not in ("", None) else ""
                lines.append(
                    f"- {row['strategy']}: mean net benefit {fmt_float(row['mean_net_benefit'])}{suffix}"
                )
    lines.extend([
        "",
        "## CTGDL External Representation Validation",
        "",
    ])
    if external_metrics:
        for row in external_metrics:
            lines.append(f"- {row['metric']}: {row['value']}")
    else:
        lines.append("- No CTGDL external validation metrics found.")
    lines.extend([
        "",
        "## Manuscript-Ready Figure Package",
        "",
        "A reproducible figure package has been assembled in `results/manuscript`:",
        "",
        "- Main figures: Figure 1 study design/evidence map, Figure 2 model validation/explanation, Figure 3 phenotype prototypes, Figure 4 recalibration/decision curves.",
        "- Supplementary figures: Figure S1 CTGDL external representation/domain-shift validation, Figure S2 model-selection/sensitivity analyses.",
        "- Export formats: PNG, PDF, SVG, and TIFF for each figure.",
        "- Source-data tables: one CSV per figure plus `manuscript_figure_manifest.csv`.",
        "",
        "## Manuscript Draft and Supplementary Tables",
        "",
        "The figure-ordered manuscript text and supplementary tables are assembled in `results/manuscript`:",
        "",
        "- Text draft: `results/manuscript/text/figure_legends_results_methods.md`.",
        "- Supplementary CSV tables: `results/manuscript/supplementary_tables/Supplementary_Table_*.csv`.",
        "- Supplementary workbook: `results/manuscript/supplementary_tables/Supplementary_Tables_CTGFHT.xlsx`.",
        "",
        "## LaTeX Manuscript",
        "",
        "A journal-neutral LaTeX manuscript draft has been assembled in `results/manuscript/latex`:",
        "",
        "- Source: `results/manuscript/latex/main.tex`.",
        "- Compiled PDF: `results/manuscript/latex/main.pdf`.",
        "- Dataset reference stubs: `results/manuscript/latex/references.bib`.",
        "- Notes: author details, acknowledgements, author contributions, competing interests, and final target-journal citations remain placeholders.",
        "",
        "## Next Required Step",
        "",
        "Move from manuscript-ready outputs to manuscript text:",
        "",
        "- Polish the draft into target-journal style and add citations.",
        "- Convert the draft into a Word manuscript or journal template.",
        "- Review supplementary tables for journal-specific naming and upload limits.",
        "- Use CTGDL primarily as external representation/domain-shift validation unless outcome labels become available.",
        "",
    ])
    out_path = Path("results/reports/project_status.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
