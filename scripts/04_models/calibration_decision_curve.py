import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


DEFAULT_MODELS = [
    "signal_features",
    "signal_plus_ssl",
    "signal_plus_ssl_plus_signal_phenotype",
]


MODEL_LABELS = {
    "signal_features": "Signal features",
    "signal_plus_ssl": "Signal + SSL",
    "signal_plus_ssl_plus_signal_phenotype": "Signal + SSL + phenotype",
}


COLORS = {
    "signal_features": "#2f6f9f",
    "signal_plus_ssl": "#8a6bb8",
    "signal_plus_ssl_plus_signal_phenotype": "#c45a3d",
}


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_slope_intercept(y, p):
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    logits = safe_logit(p).reshape(-1, 1)
    try:
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        model.fit(logits, y)
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
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    intercept, slope = calibration_slope_intercept(y, p)
    bins = calibration_bins(y, p, n_bins)
    ece = float((bins["absolute_error"] * bins["n"]).sum() / bins["n"].sum()) if len(bins) else np.nan
    mean_error = float(np.mean(p - y))
    return {
        "brier": float(brier_score_loss(y, p)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "expected_calibration_error": ece,
        "mean_prediction_error": mean_error,
    }


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


def load_prediction_matrix(path, models):
    pred = pd.read_csv(path, dtype={"record_id": str})
    pred = pred[pred["model"].isin(models)].copy()
    missing = sorted(set(models) - set(pred["model"].unique()))
    if missing:
        raise ValueError(f"Missing models in prediction table: {missing}")
    wide = pred.pivot(index=["record_id", "y_true"], columns="model", values="probability").reset_index()
    return wide


def bootstrap_calibration(wide, models, n_bins, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(wide))
    boot = {(model, metric): [] for model in models for metric in [
        "brier",
        "calibration_intercept",
        "calibration_slope",
        "expected_calibration_error",
        "mean_prediction_error",
    ]}
    for _ in range(n_bootstrap):
        sample = rng.choice(indices, size=len(indices), replace=True)
        y = wide["y_true"].to_numpy()[sample]
        if len(np.unique(y)) < 2:
            continue
        for model in models:
            p = wide[model].to_numpy()[sample]
            values = calibration_metrics(y, p, n_bins)
            for metric, value in values.items():
                boot[(model, metric)].append(value)
    rows = []
    y = wide["y_true"].to_numpy()
    for model in models:
        p = wide[model].to_numpy()
        point = calibration_metrics(y, p, n_bins)
        for metric, point_value in point.items():
            mean, low, high = percentile_ci(boot[(model, metric)])
            rows.append({
                "model": model,
                "metric": metric,
                "point": point_value,
                "bootstrap_mean": mean,
                "ci_2_5": low,
                "ci_97_5": high,
                "n_bootstrap_valid": len(boot[(model, metric)]),
            })
    return pd.DataFrame(rows)


def bootstrap_decision_curves(wide, models, thresholds, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(wide))
    y_full = wide["y_true"].to_numpy()
    rows = []
    boot_model = {(model, threshold): [] for model in models for threshold in thresholds}
    boot_all = {threshold: [] for threshold in thresholds}
    for _ in range(n_bootstrap):
        sample = rng.choice(indices, size=len(indices), replace=True)
        y = y_full[sample]
        if len(np.unique(y)) < 2:
            continue
        for threshold in thresholds:
            boot_all[threshold].append(treat_all_net_benefit(y, threshold))
            for model in models:
                p = wide[model].to_numpy()[sample]
                boot_model[(model, threshold)].append(net_benefit(y, p, threshold))

    for threshold in thresholds:
        point_all = treat_all_net_benefit(y_full, threshold)
        mean, low, high = percentile_ci(boot_all[threshold])
        rows.append({
            "strategy": "treat_all",
            "threshold": threshold,
            "net_benefit": point_all,
            "bootstrap_mean": mean,
            "ci_2_5": low,
            "ci_97_5": high,
            "n_bootstrap_valid": len(boot_all[threshold]),
        })
        rows.append({
            "strategy": "treat_none",
            "threshold": threshold,
            "net_benefit": 0.0,
            "bootstrap_mean": 0.0,
            "ci_2_5": 0.0,
            "ci_97_5": 0.0,
            "n_bootstrap_valid": n_bootstrap,
        })
        for model in models:
            p_full = wide[model].to_numpy()
            point = net_benefit(y_full, p_full, threshold)
            mean, low, high = percentile_ci(boot_model[(model, threshold)])
            rows.append({
                "strategy": model,
                "threshold": threshold,
                "net_benefit": point,
                "bootstrap_mean": mean,
                "ci_2_5": low,
                "ci_97_5": high,
                "n_bootstrap_valid": len(boot_model[(model, threshold)]),
            })
    return pd.DataFrame(rows)


def decision_summary(dca, models, threshold_ranges):
    rows = []
    for label, low, high in threshold_ranges:
        sub = dca[(dca["threshold"] >= low) & (dca["threshold"] <= high)].copy()
        base = sub[sub["strategy"] == "signal_features"].set_index("threshold")["net_benefit"]
        for strategy in ["treat_all", "treat_none"] + models:
            values = sub[sub["strategy"] == strategy].set_index("threshold")["net_benefit"]
            if len(values) == 0:
                continue
            mean_nb = float(values.mean())
            delta_vs_signal = ""
            if strategy != "signal_features" and len(base):
                aligned = values.reindex(base.index).dropna()
                aligned_base = base.reindex(aligned.index)
                if len(aligned):
                    delta_vs_signal = float((aligned - aligned_base).mean())
            rows.append({
                "threshold_range": label,
                "strategy": strategy,
                "mean_net_benefit": mean_nb,
                "mean_delta_vs_signal_features": delta_vs_signal,
            })
    return pd.DataFrame(rows)


def plot_calibration(bins_df, out_path):
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    for model, group in bins_df.groupby("model"):
        ax.plot(
            group["mean_predicted_risk"],
            group["observed_event_rate"],
            marker="o",
            color=COLORS.get(model),
            label=MODEL_LABELS.get(model, model),
        )
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed event rate")
    ax.set_title("Held-out calibration by risk quantile")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_decision_curve(dca, models, out_path):
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for model in models:
        group = dca[dca["strategy"] == model]
        ax.plot(
            group["threshold"],
            group["net_benefit"],
            color=COLORS.get(model),
            label=MODEL_LABELS.get(model, model),
        )
    for strategy, label, style in [
        ("treat_all", "Treat all", "--"),
        ("treat_none", "Treat none", ":"),
    ]:
        group = dca[dca["strategy"] == strategy]
        ax.plot(group["threshold"], group["net_benefit"], color="gray", linestyle=style, label=label)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve analysis")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Calibration and decision curve analysis for CTU-UHB held-out predictions.")
    parser.add_argument("--predictions", default="results/tables/manuscript_test_predictions.csv")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.50)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--calibration-metrics-out", default="results/tables/calibration_metrics_ci.csv")
    parser.add_argument("--calibration-bins-out", default="results/tables/calibration_curve_bins.csv")
    parser.add_argument("--decision-curve-out", default="results/tables/decision_curve_net_benefit.csv")
    parser.add_argument("--decision-summary-out", default="results/tables/decision_curve_summary.csv")
    parser.add_argument("--calibration-figure-out", default="results/figures/manuscript_calibration_curves.png")
    parser.add_argument("--decision-figure-out", default="results/figures/decision_curve_analysis.png")
    args = parser.parse_args()

    wide = load_prediction_matrix(args.predictions, args.models)
    y = wide["y_true"].to_numpy()
    thresholds = np.round(
        np.arange(args.threshold_min, args.threshold_max + args.threshold_step / 2, args.threshold_step),
        4,
    )

    calibration_metrics_df = bootstrap_calibration(wide, args.models, args.n_bins, args.n_bootstrap, args.seed)
    bins = []
    for model in args.models:
        model_bins = calibration_bins(y, wide[model].to_numpy(), args.n_bins)
        model_bins.insert(0, "model", model)
        bins.append(model_bins)
    bins_df = pd.concat(bins, ignore_index=True)

    dca_df = bootstrap_decision_curves(wide, args.models, thresholds, args.n_bootstrap, args.seed)
    summary_df = decision_summary(
        dca_df,
        args.models,
        [
            ("0.05-0.20", 0.05, 0.20),
            ("0.10-0.30", 0.10, 0.30),
            ("0.05-0.50", 0.05, 0.50),
        ],
    )

    Path(args.calibration_metrics_out).parent.mkdir(parents=True, exist_ok=True)
    calibration_metrics_df.to_csv(args.calibration_metrics_out, index=False)
    bins_df.to_csv(args.calibration_bins_out, index=False)
    dca_df.to_csv(args.decision_curve_out, index=False)
    summary_df.to_csv(args.decision_summary_out, index=False)

    plot_calibration(bins_df, args.calibration_figure_out)
    plot_decision_curve(dca_df, args.models, args.decision_figure_out)

    print(f"Wrote calibration metrics to {args.calibration_metrics_out}")
    print(f"Wrote calibration bins to {args.calibration_bins_out}")
    print(f"Wrote decision curve net benefit to {args.decision_curve_out}")
    print(f"Wrote decision curve summary to {args.decision_summary_out}")
    print(f"Wrote calibration figure to {args.calibration_figure_out}")
    print(f"Wrote decision curve figure to {args.decision_figure_out}")


if __name__ == "__main__":
    main()
