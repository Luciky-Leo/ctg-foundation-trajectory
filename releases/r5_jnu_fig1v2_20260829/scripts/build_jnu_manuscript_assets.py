#!/usr/bin/env python3
"""Create manuscript-ready JNU sensitivity summaries from locked CSV outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "04_tables"
OUT_JSON = ROOT / "03_analysis" / "JNU_MANUSCRIPT_NUMBERS.json"
MANUSCRIPT_TABLES = ROOT / "06_manuscript" / "frontiers_r5_jnu_source" / "supplementary_tables"


def record(frame: pd.DataFrame, **filters: object) -> dict:
    selected = frame
    for column, value in filters.items():
        selected = selected[selected[column] == value]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one row for {filters}, found {len(selected)}")
    return selected.iloc[0].to_dict()


def main() -> None:
    dev_risk = pd.read_csv(TABLES / "development_risk_distribution_summary.csv")
    morphology = pd.read_csv(TABLES / "development_morphology_distribution_summary.csv")
    fixed = pd.read_csv(TABLES / "fixed_test_distribution_summary.csv")
    dev_boot = pd.read_csv(TABLES / "development_paired_hierarchical_bootstrap.csv")
    fixed_boot = pd.read_csv(TABLES / "fixed_test_paired_hierarchical_bootstrap.csv")

    routes = ["random", "ctu_only", "jnu_only", "jnu_to_ctu", "jnu_asymmetric"]
    summary_rows = []
    for route in routes:
        risk = record(dev_risk, route=route, input_variant="full", evaluation_set="development_oof")
        morph = record(morphology, route=route)
        row = {
            "route": route,
            "development_auprc_median": risk["auprc_median"],
            "development_auprc_q1": risk["auprc_q1"],
            "development_auprc_q3": risk["auprc_q3"],
            "development_auroc_median": risk["auroc_median"],
            "morphology_macro_auprc_median": morph["auprc_median"],
            "morphology_macro_auroc_median": morph["auroc_median"],
        }
        fixed_row = fixed[
            (fixed["route"] == route)
            & (fixed["input_variant"] == "full")
            & (fixed["evaluation_set"] == "fixed_test_exploratory")
        ]
        row["fixed_test_auprc_median"] = float(fixed_row.iloc[0]["auprc_median"]) if len(fixed_row) else None
        row["fixed_test_auroc_median"] = float(fixed_row.iloc[0]["auroc_median"]) if len(fixed_row) else None
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLES / "supplementary_jnu_encoder_summary.csv", index=False)
    MANUSCRIPT_TABLES.mkdir(parents=True, exist_ok=True)
    summary[
        [
            "route",
            "development_auprc_median",
            "development_auroc_median",
            "fixed_test_auprc_median",
            "fixed_test_auroc_median",
        ]
    ].to_csv(MANUSCRIPT_TABLES / "Supplementary_Table_S14_encoder_origin_risk_stability.csv", index=False)

    morphology_routes = ["classical_signal", *routes]
    morphology_rows = []
    for route in morphology_routes:
        morph = record(morphology, route=route)
        morphology_rows.append(
            {
                "route": route,
                "macro_auprc_median": morph["auprc_median"],
                "macro_auroc_median": morph["auroc_median"],
            }
        )
    pd.DataFrame(morphology_rows).to_csv(
        MANUSCRIPT_TABLES / "Supplementary_Table_S15_frozen_encoder_morphology_probes.csv",
        index=False,
    )

    comparison_specs = [
        ("development", "risk", "auprc", "jnu_to_ctu", "ctu_only"),
        ("development", "morphology_macro", "auroc", "jnu_to_ctu", "ctu_only"),
        ("development", "morphology_macro", "auroc", "jnu_to_ctu", "random"),
        ("development", "morphology_macro", "auroc", "jnu_to_ctu", "classical_signal"),
        ("development", "risk", "auprc", "jnu_asymmetric", "jnu_only"),
        ("development", "morphology_macro", "auroc", "jnu_asymmetric", "jnu_only"),
        ("fixed_test_exploratory", None, "auprc", "jnu_to_ctu", "ctu_only"),
        ("fixed_test_exploratory", None, "auprc", "jnu_only", "ctu_only"),
    ]
    bootstrap_rows = []
    for data_role, analysis, metric, candidate, baseline in comparison_specs:
        frame = dev_boot if data_role == "development" else fixed_boot
        selected = frame[(frame["metric"] == metric) & (frame["candidate"] == candidate) & (frame["baseline"] == baseline)]
        if analysis is not None:
            selected = selected[selected["analysis"] == analysis]
        if len(selected) != 1:
            raise RuntimeError(f"Bootstrap row mismatch: {(data_role, analysis, metric, candidate, baseline)}")
        source = selected.iloc[0]
        point_column = "point_mean_across_seeds_tasks" if analysis == "morphology_macro" else "point_mean_across_seeds"
        bootstrap_rows.append(
            {
                "data_role": data_role,
                "analysis": analysis or "risk",
                "metric": metric,
                "candidate": candidate,
                "baseline": baseline,
                "point_difference": source[point_column],
                "ci_low": source["ci_low"],
                "ci_high": source["ci_high"],
                "positive_seed_count": int(source["positive_seed_count"]),
                "seed_count": int(source["seed_count"]),
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(TABLES / "supplementary_jnu_bootstrap_summary.csv", index=False)
    bootstrap.to_csv(
        MANUSCRIPT_TABLES / "Supplementary_Table_S16_jnu_paired_hierarchical_bootstrap.csv",
        index=False,
    )

    sensitivity_rows = []
    for data_role, frame in [("development", dev_risk), ("fixed_test_exploratory", fixed)]:
        for route in ["ctu_only", "jnu_to_ctu"]:
            evaluation = "development_oof" if data_role == "development" else "fixed_test_exploratory"
            full = record(frame, route=route, input_variant="full", evaluation_set=evaluation)["auprc_median"]
            for variant in ["fhr_only", "uc_only", "one_hz"]:
                variant_evaluation = "development_sensitivity" if data_role == "development" else "fixed_test_sensitivity_exploratory"
                value = record(frame, route=route, input_variant=variant, evaluation_set=variant_evaluation)["auprc_median"]
                sensitivity_rows.append(
                    {
                        "data_role": data_role,
                        "route": route,
                        "input_variant": variant,
                        "auprc_median": value,
                        "delta_vs_full": value - full,
                    }
                )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(TABLES / "supplementary_jnu_input_sensitivity.csv", index=False)
    sensitivity.to_csv(
        MANUSCRIPT_TABLES / "Supplementary_Table_S17_channel_sampling_sensitivity.csv",
        index=False,
    )

    leakage_audit = pd.DataFrame(
        [
            ("archive_identifier", "Zenodo 21800730; CC BY 4.0"),
            ("archive_md5", "ac1cfcba2f1b3336596544211d776771; matched"),
            ("waveform_readability", "20,769/20,769 records readable (100%)"),
            ("patient_grouping", "12,606 unique patient groups retained"),
            ("ssl_windows", "62,307 non-overlapping 10-minute windows; FHR+UC; 4 Hz"),
            ("jnu_labels_used_in_ssl", "None; clinical labels withheld"),
            ("ctu_fold_isolation", "Fold-specific adaptation used development-training records only"),
            ("fixed_test_access", "Opened after development promotion decision; historical exposure disclosed"),
            ("promotion_decision", "Supplement-only because the risk-enrichment gate failed"),
            ("external_outcome_validation", "Not performed"),
        ],
        columns=["audit_item", "result"],
    )
    leakage_audit.to_csv(
        MANUSCRIPT_TABLES / "Supplementary_Table_S18_jnu_data_leakage_audit.csv",
        index=False,
    )

    payload = {
        "source_tables": [
            "development_risk_distribution_summary.csv",
            "development_morphology_distribution_summary.csv",
            "development_paired_hierarchical_bootstrap.csv",
            "fixed_test_distribution_summary.csv",
            "fixed_test_paired_hierarchical_bootstrap.csv",
        ],
        "encoder_summary": summary.astype(object).where(pd.notnull(summary), None).to_dict("records"),
        "paired_bootstrap": bootstrap.astype(object).where(pd.notnull(bootstrap), None).to_dict("records"),
        "input_sensitivity": sensitivity.astype(object).where(pd.notnull(sensitivity), None).to_dict("records"),
        "promotion": {
            "risk_gate": False,
            "morphology_representation_gate": True,
            "main_figure_promotion": False,
            "decision": "SUPPLEMENT_ONLY",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(OUT_JSON)


if __name__ == "__main__":
    main()
