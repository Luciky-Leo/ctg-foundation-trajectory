#!/usr/bin/env python3
"""Audit development-only risk and morphology evidence before fixed-test access."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


SCRIPT_PATH = Path(__file__).resolve()
BATCH_ROOT = SCRIPT_PATH.parents[2]
sys.path.insert(0, str(SCRIPT_PATH.parent))
import run_cross_domain_ssl as cross  # noqa: E402


ROUTES = ["random", "ctu_only", "jnu_only", "jnu_to_ctu", "jnu_asymmetric"]


def fit_probe_probability(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    columns: list[str],
    outcome: str,
    seed: int,
) -> np.ndarray | None:
    y_train = cross.binary_values(train[outcome])
    y_validation = cross.binary_values(validation[outcome])
    if len(np.unique(y_train)) < 2 or len(np.unique(y_validation)) < 2:
        return None
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(train[columns]))
    x_validation = scaler.transform(imputer.transform(validation[columns]))
    model = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=seed,
    ).fit(x_train, y_train)
    return model.predict_proba(x_validation)[:, 1]


def embedding_path(seed: int, fold: int, route: str) -> Path:
    root = BATCH_ROOT / "03_analysis" / "embeddings" / f"seed_{seed}"
    if route == "random":
        return root / "random_development.csv"
    if route == "jnu_only":
        return root / "jnu_only_development.csv"
    if route == "jnu_asymmetric":
        return root / "jnu_asymmetric_development.csv"
    return root / f"fold_{fold}_{route}_development.csv"


def build_morphology_probabilities() -> pd.DataFrame:
    base = cross.r5.load_base_data()
    morphology = pd.read_csv(cross.CTU_MORPHOLOGY, dtype={"record_id": str})
    source = base.merge(
        morphology[["record_id", *cross.MORPHOLOGY_TASKS]],
        on="record_id",
        how="inner",
        validate="one_to_one",
    )
    folds = cross.make_folds(base)
    rows: list[pd.DataFrame] = []
    for seed in cross.SEEDS:
        for fold, (train_ids, validation_ids) in enumerate(folds, start=1):
            train_source = source[source["record_id"].isin(train_ids)].copy()
            validation_source = source[source["record_id"].isin(validation_ids)].copy()
            for task in cross.MORPHOLOGY_TASKS:
                probability = fit_probe_probability(
                    train_source,
                    validation_source,
                    cross.SIGNAL_FEATURES,
                    task,
                    seed,
                )
                if probability is not None:
                    frame = validation_source[["record_id", task]].rename(columns={task: "outcome"}).copy()
                    frame["outcome"] = cross.binary_values(frame["outcome"])
                    frame["probability"] = probability
                    frame["seed"] = seed
                    frame["fold"] = fold
                    frame["task"] = task
                    frame["route"] = "classical_signal"
                    rows.append(frame)

            for route in ROUTES:
                embeddings = pd.read_csv(embedding_path(seed, fold, route), dtype={"record_id": str})
                merged = source.merge(embeddings, on="record_id", how="inner", validate="one_to_one")
                train = merged[merged["record_id"].isin(train_ids)].copy()
                validation = merged[merged["record_id"].isin(validation_ids)].copy()
                embedding_columns = [column for column in merged.columns if column.startswith("emb_")]
                for task in cross.MORPHOLOGY_TASKS:
                    probability = fit_probe_probability(train, validation, embedding_columns, task, seed)
                    if probability is None:
                        continue
                    frame = validation[["record_id", task]].rename(columns={task: "outcome"}).copy()
                    frame["outcome"] = cross.binary_values(frame["outcome"])
                    frame["probability"] = probability
                    frame["seed"] = seed
                    frame["fold"] = fold
                    frame["task"] = task
                    frame["route"] = route
                    rows.append(frame)
    output = pd.concat(rows, ignore_index=True)
    output = output[["seed", "fold", "route", "task", "record_id", "outcome", "probability"]]
    output.to_csv(BATCH_ROOT / "04_tables" / "development_morphology_probabilities.csv", index=False)
    return output


def metric_value(y: np.ndarray, probability: np.ndarray, metric: str) -> float:
    if metric == "auprc":
        return float(average_precision_score(y, probability))
    if metric == "auroc":
        return float(roc_auc_score(y, probability))
    raise ValueError(metric)


def vectorized_bootstrap_metric(
    y: np.ndarray,
    probability: np.ndarray,
    sample_indices: np.ndarray,
    metric: str,
) -> np.ndarray:
    """Compute bootstrap AP/AUC for all resamples in one vectorized sort."""
    y_boot = np.asarray(y, dtype=np.int8)[sample_indices]
    probability_boot = np.asarray(probability, dtype=np.float64)[sample_indices]
    order = np.argsort(-probability_boot, axis=1, kind="stable")
    sorted_y = np.take_along_axis(y_boot, order, axis=1)
    positives = sorted_y.sum(axis=1)
    negatives = sorted_y.shape[1] - positives
    valid = (positives > 0) & (negatives > 0)
    output = np.full(len(sample_indices), np.nan, dtype=np.float64)
    if metric == "auprc":
        true_positives = np.cumsum(sorted_y, axis=1)
        ranks = np.arange(1, sorted_y.shape[1] + 1, dtype=np.float64)
        precision = true_positives / ranks[None, :]
        output[valid] = (precision * sorted_y).sum(axis=1)[valid] / positives[valid]
        return output
    if metric == "auroc":
        true_positives = np.cumsum(sorted_y, axis=1)
        false_positives = np.cumsum(1 - sorted_y, axis=1)
        tpr = true_positives / np.maximum(positives[:, None], 1)
        fpr = false_positives / np.maximum(negatives[:, None], 1)
        tpr = np.column_stack([np.zeros(len(tpr)), tpr])
        fpr = np.column_stack([np.zeros(len(fpr)), fpr])
        output[valid] = np.trapezoid(tpr, fpr, axis=1)[valid]
        return output
    raise ValueError(metric)


def risk_seed_metrics(probabilities: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (seed, route, variant, evaluation_set), frame in probabilities.groupby(
        ["seed", "encoder_route", "input_variant", "evaluation_set"]
    ):
        y = frame["neonatal_risk"].to_numpy(int)
        probability = frame["probability"].to_numpy(float)
        prediction = (probability >= 0.5).astype(int)
        rows.append({
            "seed": seed,
            "route": route,
            "input_variant": variant,
            "evaluation_set": evaluation_set,
            "n": len(frame),
            "events": int(y.sum()),
            "auprc": average_precision_score(y, probability),
            "auroc": roc_auc_score(y, probability),
            "balanced_accuracy": balanced_accuracy_score(y, prediction),
            "f1": f1_score(y, prediction, zero_division=0),
        })
    return pd.DataFrame(rows)


def morphology_seed_metrics(probabilities: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_rows: list[dict[str, object]] = []
    for (seed, route, task), frame in probabilities.groupby(["seed", "route", "task"]):
        y = frame["outcome"].to_numpy(int)
        probability = frame["probability"].to_numpy(float)
        prediction = (probability >= 0.5).astype(int)
        task_rows.append({
            "seed": seed,
            "route": route,
            "task": task,
            "n": len(frame),
            "positives": int(y.sum()),
            "auprc": average_precision_score(y, probability),
            "auroc": roc_auc_score(y, probability),
            "balanced_accuracy": balanced_accuracy_score(y, prediction),
            "f1": f1_score(y, prediction, zero_division=0),
        })
    task_metrics = pd.DataFrame(task_rows)
    macro = task_metrics.groupby(["seed", "route"], as_index=False)[
        ["auprc", "auroc", "balanced_accuracy", "f1"]
    ].mean()
    return task_metrics, macro


def hierarchical_risk_bootstrap(
    probabilities: pd.DataFrame,
    candidate: str,
    baseline: str,
    metric: str,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    frame = probabilities[
        (probabilities["evaluation_set"] == "development_oof")
        & (probabilities["input_variant"] == "full")
        & probabilities["encoder_route"].isin([candidate, baseline])
    ].copy()
    wide = frame.pivot_table(
        index=["seed", "record_id", "neonatal_risk"],
        columns="encoder_route",
        values="probability",
    ).reset_index()
    records = sorted(wide["record_id"].unique())
    by_seed = {seed_value: group.set_index("record_id") for seed_value, group in wide.groupby("seed")}
    point_values = []
    for group in by_seed.values():
        point_values.append(metric_value(group["neonatal_risk"].to_numpy(int), group[candidate], metric) - metric_value(
            group["neonatal_risk"].to_numpy(int), group[baseline], metric
        ))
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(records), size=(iterations, len(records)))
    bootstrap_differences = []
    for group in by_seed.values():
        aligned = group.reindex(records)
        y = aligned["neonatal_risk"].to_numpy(int)
        candidate_metric = vectorized_bootstrap_metric(y, aligned[candidate], sample_indices, metric)
        baseline_metric = vectorized_bootstrap_metric(y, aligned[baseline], sample_indices, metric)
        bootstrap_differences.append(candidate_metric - baseline_metric)
    boot = np.nanmean(np.vstack(bootstrap_differences), axis=0)
    return {
        "analysis": "risk",
        "metric": metric,
        "candidate": candidate,
        "baseline": baseline,
        "point_mean_across_seeds": float(np.mean(point_values)),
        "median_seed_difference": float(np.median(point_values)),
        "positive_seed_count": int(np.sum(np.asarray(point_values) > 0)),
        "seed_count": len(point_values),
        "ci_low": float(np.nanquantile(boot, 0.025)),
        "ci_high": float(np.nanquantile(boot, 0.975)),
        "bootstrap_iterations": int(np.isfinite(boot).sum()),
    }


def hierarchical_morphology_bootstrap(
    probabilities: pd.DataFrame,
    candidate: str,
    baseline: str,
    metric: str,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    subset = probabilities[probabilities["route"].isin([candidate, baseline])]
    wide = subset.pivot_table(
        index=["seed", "task", "record_id", "outcome"],
        columns="route",
        values="probability",
    ).reset_index()
    records = sorted(wide["record_id"].unique())
    grouped = {
        (seed_value, task): group.set_index("record_id")
        for (seed_value, task), group in wide.groupby(["seed", "task"])
    }

    seed_points: dict[int, list[float]] = {}
    for (seed_value, _task), group in grouped.items():
        y = group["outcome"].to_numpy(int)
        value = metric_value(y, group[candidate], metric) - metric_value(y, group[baseline], metric)
        seed_points.setdefault(int(seed_value), []).append(value)
    seed_means = [float(np.mean(values)) for values in seed_points.values()]

    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(records), size=(iterations, len(records)))
    bootstrap_differences = []
    for group in grouped.values():
        aligned = group.reindex(records)
        if aligned[["outcome", candidate, baseline]].isna().any().any():
            raise RuntimeError("morphology bootstrap alignment contains missing records")
        y = aligned["outcome"].to_numpy(int)
        candidate_metric = vectorized_bootstrap_metric(y, aligned[candidate], sample_indices, metric)
        baseline_metric = vectorized_bootstrap_metric(y, aligned[baseline], sample_indices, metric)
        bootstrap_differences.append(candidate_metric - baseline_metric)
    boot = np.nanmean(np.vstack(bootstrap_differences), axis=0)
    return {
        "analysis": "morphology_macro",
        "metric": metric,
        "candidate": candidate,
        "baseline": baseline,
        "point_mean_across_seeds_tasks": float(np.mean(seed_means)),
        "median_seed_difference": float(np.median(seed_means)),
        "positive_seed_count": int(np.sum(np.asarray(seed_means) > 0)),
        "seed_count": len(seed_means),
        "ci_low": float(np.nanquantile(boot, 0.025)),
        "ci_high": float(np.nanquantile(boot, 0.975)),
        "bootstrap_iterations": int(np.isfinite(boot).sum()),
    }


def distribution_summary(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    metrics = ["auprc", "auroc", "balanced_accuracy", "f1"]
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        for metric in metrics:
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


def json_clean_record(series: pd.Series) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in series.to_dict().items():
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            output[key] = None
        elif isinstance(value, np.integer):
            output[key] = int(value)
        elif isinstance(value, np.floating):
            output[key] = float(value)
        else:
            output[key] = value
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args()
    table_dir = BATCH_ROOT / "04_tables"
    risk_probabilities = pd.read_csv(table_dir / "development_risk_probabilities.csv", dtype={"record_id": str})
    morphology_probabilities = build_morphology_probabilities()

    risk_seed = risk_seed_metrics(risk_probabilities)
    morphology_task, morphology_macro = morphology_seed_metrics(morphology_probabilities)
    risk_seed.to_csv(table_dir / "development_risk_seed_metrics.csv", index=False)
    morphology_task.to_csv(table_dir / "development_morphology_task_oof_metrics.csv", index=False)
    morphology_macro.to_csv(table_dir / "development_morphology_macro_seed_metrics.csv", index=False)
    distribution_summary(
        risk_seed,
        ["route", "input_variant", "evaluation_set"],
    ).to_csv(table_dir / "development_risk_distribution_summary.csv", index=False)
    distribution_summary(morphology_macro, ["route"]).to_csv(
        table_dir / "development_morphology_distribution_summary.csv", index=False
    )

    comparisons = [
        ("jnu_only", "random"),
        ("jnu_to_ctu", "random"),
        ("jnu_to_ctu", "ctu_only"),
        ("jnu_asymmetric", "jnu_only"),
    ]
    bootstrap_rows: list[dict[str, object]] = []
    for candidate, baseline in comparisons:
        for metric in ["auprc", "auroc"]:
            bootstrap_rows.append(hierarchical_risk_bootstrap(
                risk_probabilities,
                candidate,
                baseline,
                metric,
                args.bootstrap_iterations,
                20260521,
            ))
    morphology_comparisons = comparisons + [("jnu_to_ctu", "classical_signal")]
    for candidate, baseline in morphology_comparisons:
        for metric in ["auprc", "auroc"]:
            bootstrap_rows.append(hierarchical_morphology_bootstrap(
                morphology_probabilities,
                candidate,
                baseline,
                metric,
                args.bootstrap_iterations,
                20260521,
            ))
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(table_dir / "development_paired_hierarchical_bootstrap.csv", index=False)

    risk_gate_row = bootstrap[
        (bootstrap["analysis"] == "risk")
        & (bootstrap["metric"] == "auprc")
        & (bootstrap["candidate"] == "jnu_to_ctu")
        & (bootstrap["baseline"] == "ctu_only")
    ].iloc[0]
    morphology_gate_row = bootstrap[
        (bootstrap["analysis"] == "morphology_macro")
        & (bootstrap["metric"] == "auroc")
        & (bootstrap["candidate"] == "jnu_to_ctu")
        & (bootstrap["baseline"] == "ctu_only")
    ].iloc[0]
    random_morphology_row = bootstrap[
        (bootstrap["analysis"] == "morphology_macro")
        & (bootstrap["metric"] == "auroc")
        & (bootstrap["candidate"] == "jnu_to_ctu")
        & (bootstrap["baseline"] == "random")
    ].iloc[0]
    risk_pass = bool(risk_gate_row["ci_low"] > 0 and risk_gate_row["positive_seed_count"] >= 4)
    morphology_pass = bool(
        morphology_gate_row["ci_low"] > 0
        and random_morphology_row["ci_low"] > 0
        and morphology_gate_row["positive_seed_count"] >= 4
        and random_morphology_row["positive_seed_count"] >= 4
    )
    decision = {
        "status": "PASS",
        "risk_promotion_gate": risk_pass,
        "morphology_representation_gate": morphology_pass,
        "main_figure_promotion": risk_pass and morphology_pass,
        "decision": "MAIN_FIGURE" if risk_pass and morphology_pass else "SUPPLEMENT_ONLY",
        "reason": (
            "JNU transfer must improve both development risk enrichment and morphology representation "
            "relative to CTU-only/random under paired intervals."
        ),
        "risk_jnu_to_ctu_minus_ctu_only": json_clean_record(risk_gate_row),
        "morphology_jnu_to_ctu_minus_ctu_only": json_clean_record(morphology_gate_row),
        "morphology_jnu_to_ctu_minus_random": json_clean_record(random_morphology_row),
        "fixed_test_accessed": False,
    }
    (BATCH_ROOT / "03_analysis" / "JNU_PROMOTION_DECISION.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
