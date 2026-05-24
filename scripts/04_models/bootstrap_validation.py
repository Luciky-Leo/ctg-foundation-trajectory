import argparse
import csv
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


MODEL_CONFIGS = [
    ("signal_features", SIGNAL_FEATURES, []),
    ("ssl_embeddings", None, []),
    ("signal_plus_ssl", None, []),
    ("signal_phenotype_only", [], ["signal_phenotype"]),
    ("ssl_phenotype_only", [], ["ssl_phenotype"]),
    ("signal_plus_signal_phenotype", SIGNAL_FEATURES, ["signal_phenotype"]),
    ("signal_plus_ssl_phenotype", SIGNAL_FEATURES, ["ssl_phenotype"]),
    ("signal_plus_ssl_plus_signal_phenotype", None, ["signal_phenotype"]),
]


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


def metric_values(y, p):
    return {
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
    }


def calibration_slope_intercept(y, p):
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(logits, y)
    return float(model.intercept_[0]), float(model.coef_[0][0])


def percentile_ci(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return "", "", ""
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def main():
    parser = argparse.ArgumentParser(description="Bootstrap CIs for manuscript-grade CTU-UHB validation.")
    parser.add_argument("--features", default="data/processed/ctu_uhb_record_features.csv")
    parser.add_argument("--embeddings", default="results/tables/sweep/p80_d64_l3_channel_cw02_record_embeddings.csv")
    parser.add_argument("--ssl-phenotypes", default="results/tables/sweep/p80_d64_l3_channel_cw02_ssl_phenotypes.csv")
    parser.add_argument("--signal-phenotypes", default="results/tables/ctu_uhb_signal_phenotypes.csv")
    parser.add_argument("--record-split", default="data/processed/ctu_uhb_record_split.csv")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--predictions-out", default="results/tables/manuscript_test_predictions.csv")
    parser.add_argument("--ci-out", default="results/tables/bootstrap_validation_ci.csv")
    parser.add_argument("--diff-out", default="results/tables/bootstrap_model_differences.csv")
    parser.add_argument("--figure-out", default="results/figures/bootstrap_model_performance.png")
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
    prediction_rows = []
    point_rows = []
    prob_by_model = {}

    for name, numeric_cols, categorical_cols in MODEL_CONFIGS:
        if name == "ssl_embeddings":
            numeric_cols = emb_cols
        elif name == "signal_plus_ssl":
            numeric_cols = SIGNAL_FEATURES + emb_cols
        elif name == "signal_plus_ssl_plus_signal_phenotype":
            numeric_cols = SIGNAL_FEATURES + emb_cols
        model = build_model(numeric_cols, categorical_cols)
        feature_cols = numeric_cols + categorical_cols
        model.fit(train_df[feature_cols], train_df["neonatal_risk"])
        prob = model.predict_proba(test_df[feature_cols])[:, 1]
        prob_by_model[name] = prob
        metrics = metric_values(y_test, prob)
        intercept, slope = calibration_slope_intercept(y_test, prob)
        point_rows.append({
            "model": name,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "event_rate_test": float(y_test.mean()),
            "auroc": metrics["auroc"],
            "auprc": metrics["auprc"],
            "brier": metrics["brier"],
            "calibration_intercept": intercept,
            "calibration_slope": slope,
        })
        for record_id, y, p in zip(test_df["record_id"], y_test, prob):
            prediction_rows.append({"record_id": record_id, "y_true": int(y), "model": name, "probability": float(p)})

    Path(args.predictions_out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(args.predictions_out, index=False)

    rng = np.random.default_rng(args.seed)
    boot = {name: {"auroc": [], "auprc": [], "brier": []} for name in prob_by_model}
    diffs = {}
    comparisons = [
        ("signal_plus_ssl", "signal_features"),
        ("signal_plus_ssl_plus_signal_phenotype", "signal_features"),
        ("signal_plus_ssl", "signal_plus_signal_phenotype"),
    ]
    for left, right in comparisons:
        for metric in ("auroc", "auprc", "brier"):
            diffs[(left, right, metric)] = []

    indices = np.arange(len(y_test))
    for _ in range(args.n_bootstrap):
        sample = rng.choice(indices, size=len(indices), replace=True)
        y = y_test[sample]
        if len(np.unique(y)) < 2:
            continue
        sample_metrics = {}
        for name, prob in prob_by_model.items():
            p = prob[sample]
            values = metric_values(y, p)
            sample_metrics[name] = values
            for metric, value in values.items():
                boot[name][metric].append(value)
        for left, right in comparisons:
            for metric in ("auroc", "auprc", "brier"):
                diffs[(left, right, metric)].append(sample_metrics[left][metric] - sample_metrics[right][metric])

    ci_rows = []
    point_df = pd.DataFrame(point_rows)
    for _, row in point_df.iterrows():
        name = row["model"]
        for metric in ("auroc", "auprc", "brier"):
            boot_mean, low, high = percentile_ci(boot[name][metric])
            ci_rows.append({
                "model": name,
                "metric": metric,
                "point": row[metric],
                "bootstrap_mean": boot_mean,
                "ci_2_5": low,
                "ci_97_5": high,
                "n_bootstrap_valid": len(boot[name][metric]),
            })
        ci_rows.append({
            "model": name,
            "metric": "calibration_intercept",
            "point": row["calibration_intercept"],
            "bootstrap_mean": "",
            "ci_2_5": "",
            "ci_97_5": "",
            "n_bootstrap_valid": "",
        })
        ci_rows.append({
            "model": name,
            "metric": "calibration_slope",
            "point": row["calibration_slope"],
            "bootstrap_mean": "",
            "ci_2_5": "",
            "ci_97_5": "",
            "n_bootstrap_valid": "",
        })

    pd.DataFrame(ci_rows).to_csv(args.ci_out, index=False)

    diff_rows = []
    for (left, right, metric), values in diffs.items():
        mean, low, high = percentile_ci(values)
        diff_rows.append({
            "comparison": f"{left} minus {right}",
            "metric": metric,
            "bootstrap_mean_difference": mean,
            "ci_2_5": low,
            "ci_97_5": high,
            "n_bootstrap_valid": len(values),
        })
    pd.DataFrame(diff_rows).to_csv(args.diff_out, index=False)

    plot_df = pd.DataFrame([row for row in ci_rows if row["metric"] in ("auroc", "auprc")])
    order = ["signal_features", "signal_plus_ssl", "signal_plus_ssl_plus_signal_phenotype", "signal_plus_signal_phenotype"]
    plot_df = plot_df[plot_df["model"].isin(order)].copy()
    Path(args.figure_out).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, metric in zip(axes, ("auroc", "auprc")):
        sub = plot_df[plot_df["metric"] == metric].set_index("model").loc[order].reset_index()
        x = np.arange(len(sub))
        ax.errorbar(
            sub["point"],
            x,
            xerr=[sub["point"] - sub["ci_2_5"], sub["ci_97_5"] - sub["point"]],
            fmt="o",
            capsize=3,
        )
        ax.set_yticks(x)
        ax.set_yticklabels(sub["model"])
        ax.set_xlabel(metric.upper())
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_title("Bootstrap AUROC")
    axes[1].set_title("Bootstrap AUPRC")
    fig.tight_layout()
    fig.savefig(args.figure_out, dpi=180)
    plt.close(fig)

    print(f"Wrote test predictions to {args.predictions_out}")
    print(f"Wrote bootstrap CIs to {args.ci_out}")
    print(f"Wrote model-difference CIs to {args.diff_out}")
    print(f"Wrote bootstrap figure to {args.figure_out}")


if __name__ == "__main__":
    main()

