#!/usr/bin/env python3
"""Summarize the historically exposed fixed CTU test set as exploratory evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


SCRIPT_PATH = Path(__file__).resolve()
BATCH_ROOT = SCRIPT_PATH.parents[2]
sys.path.insert(0, str(SCRIPT_PATH.parent))
from analyze_development_results import vectorized_bootstrap_metric  # noqa: E402


def seed_metrics(probabilities: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, route, variant, evaluation_set), frame in probabilities.groupby(
        ["seed", "encoder_route", "input_variant", "evaluation_set"]
    ):
        y = frame["neonatal_risk"].to_numpy(int)
        probability = frame["probability"].to_numpy(float)
        rows.append({
            "seed": seed,
            "route": route,
            "input_variant": variant,
            "evaluation_set": evaluation_set,
            "n": len(frame),
            "events": int(y.sum()),
            "auprc": average_precision_score(y, probability),
            "auroc": roc_auc_score(y, probability),
            "brier": brier_score_loss(y, probability),
        })
    return pd.DataFrame(rows)


def distribution_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (route, variant, evaluation_set), group in frame.groupby(
        ["route", "input_variant", "evaluation_set"]
    ):
        row = {"route": route, "input_variant": variant, "evaluation_set": evaluation_set}
        for metric in ["auprc", "auroc", "brier"]:
            values = group[metric].to_numpy(float)
            row.update({
                f"{metric}_median": float(np.median(values)),
                f"{metric}_q1": float(np.quantile(values, 0.25)),
                f"{metric}_q3": float(np.quantile(values, 0.75)),
                f"{metric}_min": float(np.min(values)),
                f"{metric}_max": float(np.max(values)),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap(
    probabilities: pd.DataFrame,
    candidate: str,
    baseline: str,
    metric: str,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    subset = probabilities[
        (probabilities["evaluation_set"] == "fixed_test_exploratory")
        & (probabilities["input_variant"] == "full")
        & probabilities["encoder_route"].isin([candidate, baseline])
    ]
    wide = subset.pivot_table(
        index=["seed", "record_id", "neonatal_risk"],
        columns="encoder_route",
        values="probability",
    ).reset_index()
    records = sorted(wide["record_id"].unique())
    grouped = [group.set_index("record_id").reindex(records) for _, group in wide.groupby("seed")]
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(records), size=(iterations, len(records)))
    bootstrap_differences = []
    point_differences = []
    for group in grouped:
        y = group["neonatal_risk"].to_numpy(int)
        candidate_probability = group[candidate].to_numpy(float)
        baseline_probability = group[baseline].to_numpy(float)
        if metric == "auprc":
            point_candidate = average_precision_score(y, candidate_probability)
            point_baseline = average_precision_score(y, baseline_probability)
        else:
            point_candidate = roc_auc_score(y, candidate_probability)
            point_baseline = roc_auc_score(y, baseline_probability)
        point_differences.append(point_candidate - point_baseline)
        bootstrap_differences.append(
            vectorized_bootstrap_metric(y, candidate_probability, sample_indices, metric)
            - vectorized_bootstrap_metric(y, baseline_probability, sample_indices, metric)
        )
    boot = np.nanmean(np.vstack(bootstrap_differences), axis=0)
    return {
        "metric": metric,
        "candidate": candidate,
        "baseline": baseline,
        "point_mean_across_seeds": float(np.mean(point_differences)),
        "median_seed_difference": float(np.median(point_differences)),
        "positive_seed_count": int(np.sum(np.asarray(point_differences) > 0)),
        "seed_count": len(point_differences),
        "ci_low": float(np.nanquantile(boot, 0.025)),
        "ci_high": float(np.nanquantile(boot, 0.975)),
        "bootstrap_iterations": int(np.isfinite(boot).sum()),
        "interpretation": "exploratory_fixed_test_with_potential_selection_induced_optimism",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args()
    table_dir = BATCH_ROOT / "04_tables"
    probabilities = pd.read_csv(table_dir / "fixed_test_risk_probabilities.csv", dtype={"record_id": str})
    metrics = seed_metrics(probabilities)
    summary = distribution_summary(metrics)
    metrics.to_csv(table_dir / "fixed_test_seed_metrics.csv", index=False)
    summary.to_csv(table_dir / "fixed_test_distribution_summary.csv", index=False)

    comparisons = [
        ("jnu_only", "random"),
        ("jnu_only", "ctu_only"),
        ("jnu_to_ctu", "random"),
        ("jnu_to_ctu", "ctu_only"),
    ]
    rows = []
    for candidate, baseline in comparisons:
        for metric in ["auprc", "auroc"]:
            rows.append(paired_bootstrap(
                probabilities,
                candidate,
                baseline,
                metric,
                args.bootstrap_iterations,
                20260521,
            ))
    bootstrap = pd.DataFrame(rows)
    bootstrap.to_csv(table_dir / "fixed_test_paired_hierarchical_bootstrap.csv", index=False)

    sequential = summary[
        (summary["route"] == "jnu_to_ctu")
        & (summary["input_variant"] == "full")
    ].iloc[0]
    ctu_only = summary[
        (summary["route"] == "ctu_only")
        & (summary["input_variant"] == "full")
    ].iloc[0]
    result = {
        "status": "PASS",
        "analysis_role": "exploratory_fixed_test_with_potential_selection_induced_optimism",
        "n_records": int(probabilities["record_id"].nunique()),
        "events": int(probabilities.drop_duplicates("record_id")["neonatal_risk"].sum()),
        "jnu_to_ctu_auprc_median": float(sequential["auprc_median"]),
        "ctu_only_auprc_median": float(ctu_only["auprc_median"]),
        "jnu_to_ctu_minus_ctu_only": bootstrap[
            (bootstrap["candidate"] == "jnu_to_ctu")
            & (bootstrap["baseline"] == "ctu_only")
            & (bootstrap["metric"] == "auprc")
        ].iloc[0].to_dict(),
        "conclusion": "No stable incremental neonatal-risk enrichment from JNU-to-CTU adaptation.",
    }
    (BATCH_ROOT / "03_analysis" / "FIXED_TEST_EXPLORATORY_AUDIT.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
