import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SIGNAL_FEATURES = [
    "fhr_missing_fraction",
    "uc_missing_fraction",
    "fhr_mean",
    "fhr_sd",
    "fhr_p05",
    "fhr_p50",
    "fhr_p95",
    "uc_mean",
    "uc_sd",
    "uc_p95",
    "fhr_uc_corr",
    "deceleration_proxy_count",
]

MISSINGNESS_FEATURES = ["fhr_missing_fraction", "uc_missing_fraction"]
NO_MISSINGNESS_SIGNAL_FEATURES = [x for x in SIGNAL_FEATURES if x not in MISSINGNESS_FEATURES]

MODEL_NAMES = [
    "signal_features",
    "signal_plus_ssl",
    "signal_plus_ssl_plus_signal_phenotype",
]

SCENARIOS = {
    "full_features": SIGNAL_FEATURES,
    "no_missingness_features": NO_MISSINGNESS_SIGNAL_FEATURES,
}


def build_model(numeric_cols, categorical_cols):
    transformers = []
    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
            numeric_cols,
        ))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))
    return Pipeline([
        ("preprocess", ColumnTransformer(transformers=transformers)),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
    ])


def model_columns(model_name, signal_cols, emb_cols):
    if model_name == "signal_features":
        return signal_cols, []
    if model_name == "signal_plus_ssl":
        return signal_cols + emb_cols, []
    if model_name == "signal_plus_ssl_plus_signal_phenotype":
        return signal_cols + emb_cols, ["signal_phenotype"]
    raise ValueError(f"Unknown model: {model_name}")


def metric_values(y, p):
    return {
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
    }


def percentile_ci(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return "", "", ""
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def bootstrap_metrics(y_test, prob_by_key, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y_test))
    boot = {key: {"auroc": [], "auprc": [], "brier": []} for key in prob_by_key}

    within_scenario_diffs = {}
    for scenario in SCENARIOS:
        for left, right in [
            ("signal_plus_ssl", "signal_features"),
            ("signal_plus_ssl_plus_signal_phenotype", "signal_features"),
        ]:
            for metric in ("auroc", "auprc", "brier"):
                within_scenario_diffs[(scenario, left, right, metric)] = []

    no_missingness_diffs = {}
    for model_name in MODEL_NAMES:
        for metric in ("auroc", "auprc", "brier"):
            no_missingness_diffs[(model_name, metric)] = []

    for _ in range(n_bootstrap):
        sample = rng.choice(indices, size=len(indices), replace=True)
        y = y_test[sample]
        if len(np.unique(y)) < 2:
            continue

        sample_metrics = {}
        for key, prob in prob_by_key.items():
            values = metric_values(y, prob[sample])
            sample_metrics[key] = values
            for metric, value in values.items():
                boot[key][metric].append(value)

        for scenario in SCENARIOS:
            for left, right in [
                ("signal_plus_ssl", "signal_features"),
                ("signal_plus_ssl_plus_signal_phenotype", "signal_features"),
            ]:
                left_key = (scenario, left)
                right_key = (scenario, right)
                for metric in ("auroc", "auprc", "brier"):
                    within_scenario_diffs[(scenario, left, right, metric)].append(
                        sample_metrics[left_key][metric] - sample_metrics[right_key][metric]
                    )

        for model_name in MODEL_NAMES:
            full_key = ("full_features", model_name)
            reduced_key = ("no_missingness_features", model_name)
            for metric in ("auroc", "auprc", "brier"):
                no_missingness_diffs[(model_name, metric)].append(
                    sample_metrics[reduced_key][metric] - sample_metrics[full_key][metric]
                )

    return boot, within_scenario_diffs, no_missingness_diffs


def make_figure(metrics_df, out_path):
    plot_df = metrics_df[metrics_df["metric"].isin(["auroc", "auprc"])].copy()
    model_order = MODEL_NAMES
    scenario_order = list(SCENARIOS.keys())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    palette = {"full_features": "#2f6f9f", "no_missingness_features": "#c45a3d"}
    offsets = {"full_features": -0.13, "no_missingness_features": 0.13}
    y_positions = np.arange(len(model_order))
    for ax, metric in zip(axes, ["auroc", "auprc"]):
        for scenario in scenario_order:
            sub = (
                plot_df[(plot_df["scenario"] == scenario) & (plot_df["metric"] == metric)]
                .set_index("model")
                .loc[model_order]
                .reset_index()
            )
            y = y_positions + offsets[scenario]
            ax.errorbar(
                sub["point"],
                y,
                xerr=[sub["point"] - sub["ci_2_5"], sub["ci_97_5"] - sub["point"]],
                fmt="o",
                capsize=3,
                color=palette[scenario],
                label=scenario,
            )
        ax.set_yticks(y_positions)
        ax.set_yticklabels(model_order)
        ax.set_xlabel(metric.upper())
        ax.grid(axis="x", alpha=0.25)
        ax.set_title(f"Missingness sensitivity: {metric.upper()}")
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Sensitivity analysis excluding FHR/UC missingness features.")
    parser.add_argument("--features", default="data/processed/ctu_uhb_record_features.csv")
    parser.add_argument("--embeddings", default="results/tables/sweep/p80_d64_l3_channel_cw02_record_embeddings.csv")
    parser.add_argument("--ssl-phenotypes", default="results/tables/sweep/p80_d64_l3_channel_cw02_ssl_phenotypes.csv")
    parser.add_argument("--signal-phenotypes", default="results/tables/ctu_uhb_signal_phenotypes.csv")
    parser.add_argument("--record-split", default="data/processed/ctu_uhb_record_split.csv")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--predictions-out", default="results/tables/missingness_sensitivity_predictions.csv")
    parser.add_argument("--metrics-out", default="results/tables/missingness_sensitivity_metrics.csv")
    parser.add_argument("--diff-out", default="results/tables/missingness_sensitivity_differences.csv")
    parser.add_argument("--figure-out", default="results/figures/missingness_sensitivity_performance.png")
    args = parser.parse_args()

    features = pd.read_csv(args.features, dtype={"record_id": str})
    embeddings = pd.read_csv(args.embeddings, dtype={"record_id": str}).drop(columns=["neonatal_risk"], errors="ignore")
    ssl_pheno = pd.read_csv(args.ssl_phenotypes, dtype={"record_id": str})[["record_id", "ssl_phenotype"]]
    signal_pheno = pd.read_csv(args.signal_phenotypes, dtype={"record_id": str})[["record_id", "signal_phenotype"]]
    split = pd.read_csv(args.record_split, dtype={"record_id": str})[["record_id", "split"]]

    df = features.merge(embeddings, on="record_id", how="inner")
    df = df.merge(ssl_pheno, on="record_id", how="inner")
    df = df.merge(signal_pheno, on="record_id", how="inner")
    df = df.merge(split, on="record_id", how="inner")
    df["neonatal_risk"] = df["neonatal_risk"].astype(int)
    emb_cols = [col for col in df.columns if col.startswith("emb_")]

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
    y_test = test_df["neonatal_risk"].to_numpy()

    point_metrics = {}
    prob_by_key = {}
    prediction_rows = []

    for scenario, signal_cols in SCENARIOS.items():
        for model_name in MODEL_NAMES:
            numeric_cols, categorical_cols = model_columns(model_name, signal_cols, emb_cols)
            feature_cols = numeric_cols + categorical_cols
            model = build_model(numeric_cols, categorical_cols)
            model.fit(train_df[feature_cols], train_df["neonatal_risk"])
            prob = model.predict_proba(test_df[feature_cols])[:, 1]
            key = (scenario, model_name)
            prob_by_key[key] = prob
            point_metrics[key] = metric_values(y_test, prob)
            for record_id, y, p in zip(test_df["record_id"], y_test, prob):
                prediction_rows.append({
                    "scenario": scenario,
                    "model": model_name,
                    "record_id": record_id,
                    "y_true": int(y),
                    "probability": float(p),
                })

    boot, within_scenario_diffs, no_missingness_diffs = bootstrap_metrics(
        y_test,
        prob_by_key,
        args.n_bootstrap,
        args.seed,
    )

    metric_rows = []
    for scenario in SCENARIOS:
        for model_name in MODEL_NAMES:
            key = (scenario, model_name)
            for metric in ("auroc", "auprc", "brier"):
                boot_mean, low, high = percentile_ci(boot[key][metric])
                metric_rows.append({
                    "scenario": scenario,
                    "model": model_name,
                    "metric": metric,
                    "point": point_metrics[key][metric],
                    "bootstrap_mean": boot_mean,
                    "ci_2_5": low,
                    "ci_97_5": high,
                    "n_bootstrap_valid": len(boot[key][metric]),
                })
    metrics_df = pd.DataFrame(metric_rows)

    diff_rows = []
    for (scenario, left, right, metric), values in within_scenario_diffs.items():
        mean, low, high = percentile_ci(values)
        diff_rows.append({
            "comparison_type": "model_difference_within_scenario",
            "scenario": scenario,
            "comparison": f"{left} minus {right}",
            "metric": metric,
            "bootstrap_mean_difference": mean,
            "ci_2_5": low,
            "ci_97_5": high,
            "n_bootstrap_valid": len(values),
        })
    for (model_name, metric), values in no_missingness_diffs.items():
        mean, low, high = percentile_ci(values)
        diff_rows.append({
            "comparison_type": "no_missingness_minus_full",
            "scenario": "no_missingness_features vs full_features",
            "comparison": model_name,
            "metric": metric,
            "bootstrap_mean_difference": mean,
            "ci_2_5": low,
            "ci_97_5": high,
            "n_bootstrap_valid": len(values),
        })

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(args.predictions_out, index=False)
    metrics_df.to_csv(args.metrics_out, index=False)
    pd.DataFrame(diff_rows).to_csv(args.diff_out, index=False)
    make_figure(metrics_df, args.figure_out)

    print(f"Wrote missingness sensitivity predictions to {args.predictions_out}")
    print(f"Wrote missingness sensitivity metrics to {args.metrics_out}")
    print(f"Wrote missingness sensitivity differences to {args.diff_out}")
    print(f"Wrote missingness sensitivity figure to {args.figure_out}")


if __name__ == "__main__":
    main()
