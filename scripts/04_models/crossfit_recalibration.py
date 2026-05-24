import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
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
    ("signal_plus_ssl", None, []),
    ("signal_plus_ssl_plus_signal_phenotype", None, ["signal_phenotype"]),
]


MODEL_LABELS = {
    "signal_features": "Signal features",
    "signal_plus_ssl": "Signal + SSL",
    "signal_plus_ssl_plus_signal_phenotype": "Signal + SSL + phenotype",
}


COLORS = {
    "raw": "#777777",
    "platt": "#2f6f9f",
    "isotonic": "#c45a3d",
}


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


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


def model_feature_config(name, emb_cols):
    for model_name, numeric_cols, categorical_cols in MODEL_CONFIGS:
        if model_name != name:
            continue
        if name in {"signal_plus_ssl", "signal_plus_ssl_plus_signal_phenotype"}:
            numeric_cols = SIGNAL_FEATURES + emb_cols
        return numeric_cols, categorical_cols
    raise ValueError(f"Unknown model: {name}")


def metric_values(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    return {
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
    }


def calibration_slope_intercept(y, p):
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    try:
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        model.fit(safe_logit(p).reshape(-1, 1), y)
        return float(model.intercept_[0]), float(model.coef_[0][0])
    except Exception:
        return np.nan, np.nan


def calibration_bins(y, p, n_bins):
    df = pd.DataFrame({"y": np.asarray(y, dtype=int), "p": np.asarray(p, dtype=float)})
    try:
        df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    except ValueError:
        df["bin"] = pd.cut(df["p"], bins=n_bins, duplicates="drop")
    rows = []
    for idx, (_, group) in enumerate(df.groupby("bin", observed=True), start=1):
        rows.append({
            "bin": idx,
            "n": int(len(group)),
            "events": int(group["y"].sum()),
            "mean_predicted_risk": float(group["p"].mean()),
            "observed_event_rate": float(group["y"].mean()),
            "absolute_error": float(abs(group["y"].mean() - group["p"].mean())),
        })
    return pd.DataFrame(rows)


def calibration_metrics(y, p, n_bins):
    intercept, slope = calibration_slope_intercept(y, p)
    bins = calibration_bins(y, p, n_bins)
    ece = float((bins["absolute_error"] * bins["n"]).sum() / bins["n"].sum()) if len(bins) else np.nan
    out = metric_values(y, p)
    out.update({
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "expected_calibration_error": ece,
        "mean_prediction_error": float(np.mean(np.asarray(p, dtype=float) - np.asarray(y, dtype=int))),
    })
    return out


def fit_crossfit_calibrators(train_df, feature_cols, categorical_cols, n_splits, seed):
    y_train = train_df["neonatal_risk"].astype(int).to_numpy()
    oof = np.zeros(len(train_df), dtype=float)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (inner_idx, cal_idx) in enumerate(cv.split(train_df[feature_cols], y_train), start=1):
        inner = train_df.iloc[inner_idx]
        cal = train_df.iloc[cal_idx]
        model = build_model([c for c in feature_cols if c not in categorical_cols], categorical_cols)
        model.fit(inner[feature_cols], inner["neonatal_risk"].astype(int))
        oof[cal_idx] = model.predict_proba(cal[feature_cols])[:, 1]

    platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    platt.fit(safe_logit(oof).reshape(-1, 1), y_train)
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(oof, y_train)
    return oof, platt, isotonic


def apply_platt(calibrator, p):
    return calibrator.predict_proba(safe_logit(p).reshape(-1, 1))[:, 1]


def net_benefit(y, p, threshold):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = p >= threshold
    n = len(y)
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    return float(tp / n - fp / n * (threshold / (1 - threshold)))


def treat_all_net_benefit(y, threshold):
    prevalence = float(np.mean(y))
    return prevalence - (1 - prevalence) * (threshold / (1 - threshold))


def percentile_ci(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def bootstrap_metrics(y_test, prob_by_key, n_bins, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y_test))
    boot = {key: {} for key in prob_by_key}
    metric_names = [
        "auroc",
        "auprc",
        "brier",
        "calibration_intercept",
        "calibration_slope",
        "expected_calibration_error",
        "mean_prediction_error",
    ]
    for key in prob_by_key:
        for metric in metric_names:
            boot[key][metric] = []

    for _ in range(n_bootstrap):
        sample = rng.choice(indices, size=len(indices), replace=True)
        y = y_test[sample]
        if len(np.unique(y)) < 2:
            continue
        for key, prob in prob_by_key.items():
            values = calibration_metrics(y, prob[sample], n_bins)
            for metric in metric_names:
                boot[key][metric].append(values[metric])
    return boot


def bootstrap_differences(y_test, prob_by_key, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y_test))
    diffs = {}
    for model_name in {key[0] for key in prob_by_key}:
        for method in ("platt", "isotonic"):
            for metric in ("brier", "calibration_slope", "expected_calibration_error", "auprc", "auroc"):
                diffs[(model_name, method, metric)] = []

    for _ in range(n_bootstrap):
        sample = rng.choice(indices, size=len(indices), replace=True)
        y = y_test[sample]
        if len(np.unique(y)) < 2:
            continue
        metric_cache = {}
        for key, prob in prob_by_key.items():
            metric_cache[key] = calibration_metrics(y, prob[sample], n_bins=5)
        for model_name in {key[0] for key in prob_by_key}:
            raw_key = (model_name, "raw")
            for method in ("platt", "isotonic"):
                method_key = (model_name, method)
                for metric in ("brier", "calibration_slope", "expected_calibration_error", "auprc", "auroc"):
                    diffs[(model_name, method, metric)].append(
                        metric_cache[method_key][metric] - metric_cache[raw_key][metric]
                    )
    return diffs


def make_metric_rows(y_test, prob_by_key, boot, n_bins):
    rows = []
    for key, prob in prob_by_key.items():
        model_name, calibration_method = key
        point = calibration_metrics(y_test, prob, n_bins)
        for metric, point_value in point.items():
            mean, low, high = percentile_ci(boot[key][metric])
            rows.append({
                "model": model_name,
                "calibration_method": calibration_method,
                "metric": metric,
                "point": point_value,
                "bootstrap_mean": mean,
                "ci_2_5": low,
                "ci_97_5": high,
                "n_bootstrap_valid": len(boot[key][metric]),
            })
    return pd.DataFrame(rows)


def make_difference_rows(diffs):
    rows = []
    for (model_name, method, metric), values in diffs.items():
        mean, low, high = percentile_ci(values)
        rows.append({
            "model": model_name,
            "comparison": f"{method} minus raw",
            "metric": metric,
            "bootstrap_mean_difference": mean,
            "ci_2_5": low,
            "ci_97_5": high,
            "n_bootstrap_valid": len(values),
        })
    return pd.DataFrame(rows)


def decision_curve(y, prob_by_key, thresholds):
    rows = []
    for threshold in thresholds:
        rows.append({
            "strategy": "treat_all",
            "threshold": threshold,
            "net_benefit": treat_all_net_benefit(y, threshold),
        })
        rows.append({
            "strategy": "treat_none",
            "threshold": threshold,
            "net_benefit": 0.0,
        })
        for (model_name, method), prob in prob_by_key.items():
            rows.append({
                "strategy": f"{model_name}_{method}",
                "threshold": threshold,
                "net_benefit": net_benefit(y, prob, threshold),
            })
    return pd.DataFrame(rows)


def decision_summary(dca, threshold_ranges):
    rows = []
    for label, low, high in threshold_ranges:
        sub = dca[(dca["threshold"] >= low) & (dca["threshold"] <= high)]
        base = sub[sub["strategy"] == "signal_plus_ssl_plus_signal_phenotype_raw"].set_index("threshold")["net_benefit"]
        for strategy, group in sub.groupby("strategy"):
            values = group.set_index("threshold")["net_benefit"]
            delta = ""
            if strategy != "signal_plus_ssl_plus_signal_phenotype_raw" and len(base):
                aligned = values.reindex(base.index).dropna()
                aligned_base = base.reindex(aligned.index)
                if len(aligned):
                    delta = float((aligned - aligned_base).mean())
            rows.append({
                "threshold_range": label,
                "strategy": strategy,
                "mean_net_benefit": float(values.mean()),
                "mean_delta_vs_raw_signal_plus_ssl_plus_signal_phenotype": delta,
            })
    return pd.DataFrame(rows)


def plot_calibration(bins_df, out_path):
    models = ["signal_features", "signal_plus_ssl", "signal_plus_ssl_plus_signal_phenotype"]
    fig, axes = plt.subplots(1, len(models), figsize=(15, 4.6), sharex=True, sharey=True)
    for ax, model_name in zip(axes, models):
        for method, group in bins_df[bins_df["model"] == model_name].groupby("calibration_method"):
            ax.plot(
                group["mean_predicted_risk"],
                group["observed_event_rate"],
                marker="o",
                color=COLORS.get(method),
                label=method,
            )
        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
        ax.set_title(MODEL_LABELS.get(model_name, model_name))
        ax.set_xlabel("Mean predicted risk")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Observed event rate")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.suptitle("Cross-fitted recalibration curves", y=1.02)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_decision_curve(dca, out_path):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    keep = [
        "signal_plus_ssl_plus_signal_phenotype_raw",
        "signal_plus_ssl_plus_signal_phenotype_platt",
        "signal_plus_ssl_plus_signal_phenotype_isotonic",
        "treat_all",
        "treat_none",
    ]
    styles = {
        "signal_plus_ssl_plus_signal_phenotype_raw": ("Raw", "#777777", "-"),
        "signal_plus_ssl_plus_signal_phenotype_platt": ("Platt", "#2f6f9f", "-"),
        "signal_plus_ssl_plus_signal_phenotype_isotonic": ("Isotonic", "#c45a3d", "-"),
        "treat_all": ("Treat all", "gray", "--"),
        "treat_none": ("Treat none", "black", ":"),
    }
    for strategy in keep:
        group = dca[dca["strategy"] == strategy]
        label, color, linestyle = styles[strategy]
        ax.plot(group["threshold"], group["net_benefit"], label=label, color=color, linestyle=linestyle)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve after cross-fitted recalibration")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Cross-fitted Platt and isotonic recalibration for CTU-UHB risk models.")
    parser.add_argument("--features", default="data/processed/ctu_uhb_record_features.csv")
    parser.add_argument("--embeddings", default="results/tables/sweep/p80_d64_l3_channel_cw02_record_embeddings.csv")
    parser.add_argument("--ssl-phenotypes", default="results/tables/sweep/p80_d64_l3_channel_cw02_ssl_phenotypes.csv")
    parser.add_argument("--signal-phenotypes", default="results/tables/ctu_uhb_signal_phenotypes.csv")
    parser.add_argument("--record-split", default="data/processed/ctu_uhb_record_split.csv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.50)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--predictions-out", default="results/tables/recalibrated_test_predictions.csv")
    parser.add_argument("--metrics-out", default="results/tables/recalibration_metrics_ci.csv")
    parser.add_argument("--diff-out", default="results/tables/recalibration_metric_differences.csv")
    parser.add_argument("--bins-out", default="results/tables/recalibration_curve_bins.csv")
    parser.add_argument("--dca-out", default="results/tables/recalibrated_decision_curve_net_benefit.csv")
    parser.add_argument("--dca-summary-out", default="results/tables/recalibrated_decision_curve_summary.csv")
    parser.add_argument("--calibration-figure-out", default="results/figures/recalibration_curves.png")
    parser.add_argument("--dca-figure-out", default="results/figures/recalibrated_decision_curve_analysis.png")
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

    prob_by_key = {}
    prediction_rows = []
    bins = []

    for model_name, _, _ in MODEL_CONFIGS:
        numeric_cols, categorical_cols = model_feature_config(model_name, emb_cols)
        feature_cols = numeric_cols + categorical_cols
        oof, platt, isotonic = fit_crossfit_calibrators(
            train_df,
            feature_cols,
            categorical_cols,
            args.n_splits,
            args.seed,
        )

        final_model = build_model(numeric_cols, categorical_cols)
        final_model.fit(train_df[feature_cols], train_df["neonatal_risk"].astype(int))
        raw_prob = final_model.predict_proba(test_df[feature_cols])[:, 1]
        probs = {
            "raw": raw_prob,
            "platt": apply_platt(platt, raw_prob),
            "isotonic": isotonic.predict(raw_prob),
        }
        for method, prob in probs.items():
            prob_by_key[(model_name, method)] = prob
            model_bins = calibration_bins(y_test, prob, args.n_bins)
            model_bins.insert(0, "calibration_method", method)
            model_bins.insert(0, "model", model_name)
            bins.append(model_bins)
            for record_id, y, p in zip(test_df["record_id"], y_test, prob):
                prediction_rows.append({
                    "record_id": record_id,
                    "y_true": int(y),
                    "model": model_name,
                    "calibration_method": method,
                    "probability": float(p),
                })

    boot = bootstrap_metrics(y_test, prob_by_key, args.n_bins, args.n_bootstrap, args.seed)
    diffs = bootstrap_differences(y_test, prob_by_key, args.n_bootstrap, args.seed)
    metrics_df = make_metric_rows(y_test, prob_by_key, boot, args.n_bins)
    diff_df = make_difference_rows(diffs)
    bins_df = pd.concat(bins, ignore_index=True)

    thresholds = np.round(
        np.arange(args.threshold_min, args.threshold_max + args.threshold_step / 2, args.threshold_step),
        4,
    )
    dca = decision_curve(y_test, prob_by_key, thresholds)
    dca_summary = decision_summary(
        dca,
        [
            ("0.05-0.20", 0.05, 0.20),
            ("0.10-0.30", 0.10, 0.30),
            ("0.05-0.50", 0.05, 0.50),
        ],
    )

    Path(args.predictions_out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(args.predictions_out, index=False)
    metrics_df.to_csv(args.metrics_out, index=False)
    diff_df.to_csv(args.diff_out, index=False)
    bins_df.to_csv(args.bins_out, index=False)
    dca.to_csv(args.dca_out, index=False)
    dca_summary.to_csv(args.dca_summary_out, index=False)
    plot_calibration(bins_df, args.calibration_figure_out)
    plot_decision_curve(dca, args.dca_figure_out)

    print(f"Wrote recalibrated predictions to {args.predictions_out}")
    print(f"Wrote recalibration metrics to {args.metrics_out}")
    print(f"Wrote recalibration metric differences to {args.diff_out}")
    print(f"Wrote recalibration bins to {args.bins_out}")
    print(f"Wrote recalibrated decision curves to {args.dca_out}")
    print(f"Wrote recalibrated decision summary to {args.dca_summary_out}")
    print(f"Wrote recalibration figure to {args.calibration_figure_out}")
    print(f"Wrote recalibrated DCA figure to {args.dca_figure_out}")


if __name__ == "__main__":
    main()
