import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
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


SIGNAL_FEATURE_LABELS = {
    "fhr_missing_fraction": "FHR missing fraction",
    "uc_missing_fraction": "UC missing fraction",
    "fhr_mean": "Mean FHR",
    "fhr_sd": "FHR variability (SD)",
    "fhr_p05": "FHR 5th percentile",
    "fhr_p50": "Median FHR",
    "fhr_p95": "FHR 95th percentile",
    "uc_mean": "Mean uterine contraction",
    "uc_sd": "UC variability (SD)",
    "uc_p95": "UC 95th percentile",
    "fhr_uc_corr": "FHR-UC correlation",
    "deceleration_proxy_count": "Deceleration-proxy burden",
}


MODEL_CONFIGS = [
    ("signal_features", SIGNAL_FEATURES, []),
    ("signal_plus_ssl", None, []),
    ("signal_plus_ssl_plus_signal_phenotype", None, ["signal_phenotype"]),
]


def feature_family(feature):
    if feature in SIGNAL_FEATURES:
        return "clinical_signal"
    if feature.startswith("emb_"):
        return "ssl_embedding"
    if feature in {"signal_phenotype", "ssl_phenotype"}:
        return "phenotype"
    return "other"


def display_name(feature):
    if feature in SIGNAL_FEATURE_LABELS:
        return SIGNAL_FEATURE_LABELS[feature]
    if feature.startswith("emb_"):
        return f"SSL dimension {feature.replace('emb_', '')}"
    if feature == "signal_phenotype":
        return "Signal phenotype"
    if feature == "ssl_phenotype":
        return "SSL phenotype"
    return feature


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
        if name == "signal_plus_ssl":
            numeric_cols = SIGNAL_FEATURES + emb_cols
        elif name == "signal_plus_ssl_plus_signal_phenotype":
            numeric_cols = SIGNAL_FEATURES + emb_cols
        return numeric_cols, categorical_cols
    raise ValueError(f"Unknown model config: {name}")


def coefficient_table(model, model_name):
    preprocess = model.named_steps["preprocess"]
    classifier = model.named_steps["model"]
    try:
        names = preprocess.get_feature_names_out()
    except AttributeError:
        names = [f"feature_{i}" for i in range(classifier.coef_.shape[1])]
    clean_names = []
    for name in names:
        if "__" in name:
            clean_names.append(name.split("__", 1)[1])
        else:
            clean_names.append(name)
    return pd.DataFrame({
        "model": model_name,
        "processed_feature": clean_names,
        "coefficient": classifier.coef_[0],
        "abs_coefficient": np.abs(classifier.coef_[0]),
    }).sort_values("abs_coefficient", ascending=False)


def run_permutation(model, model_name, x_test, y_test, n_repeats, seed):
    rows = []
    for scoring in ("average_precision", "roc_auc"):
        result = permutation_importance(
            model,
            x_test,
            y_test,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=seed,
            n_jobs=1,
        )
        metric = "auprc" if scoring == "average_precision" else "auroc"
        metric_df = pd.DataFrame({
            "feature": x_test.columns,
            f"importance_mean_{metric}": result.importances_mean,
            f"importance_std_{metric}": result.importances_std,
        })
        rows.append(metric_df)
    merged = rows[0].merge(rows[1], on="feature", how="outer")
    merged.insert(0, "model", model_name)
    merged["feature_family"] = merged["feature"].map(feature_family)
    merged["feature_label"] = merged["feature"].map(display_name)
    merged["rank_auprc"] = merged["importance_mean_auprc"].rank(ascending=False, method="min").astype(int)
    merged["rank_auroc"] = merged["importance_mean_auroc"].rank(ascending=False, method="min").astype(int)
    return merged.sort_values(["model", "rank_auprc", "rank_auroc"])


def make_top_feature_plot(importance_df, out_path, top_n):
    models = list(importance_df["model"].drop_duplicates())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), sharex=False)
    if len(models) == 1:
        axes = [axes]
    for ax, model_name in zip(axes, models):
        sub = importance_df[importance_df["model"] == model_name].copy()
        sub = sub.sort_values("importance_mean_auprc", ascending=False).head(top_n)
        sub = sub.sort_values("importance_mean_auprc", ascending=True)
        colors = sub["feature_family"].map({
            "clinical_signal": "#2f6f9f",
            "ssl_embedding": "#8a6bb8",
            "phenotype": "#c45a3d",
        }).fillna("#777777")
        ax.barh(sub["feature_label"], sub["importance_mean_auprc"], xerr=sub["importance_std_auprc"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(model_name)
        ax.set_xlabel("Permutation importance for AUPRC")
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_family_plot(family_df, out_path):
    models = list(family_df["model"].drop_duplicates())
    families = ["clinical_signal", "ssl_embedding", "phenotype", "other"]
    x = np.arange(len(models))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8, 4.5))
    palette = {
        "clinical_signal": "#2f6f9f",
        "ssl_embedding": "#8a6bb8",
        "phenotype": "#c45a3d",
        "other": "#777777",
    }
    for idx, family in enumerate(families):
        values = []
        for model in models:
            row = family_df[(family_df["model"] == model) & (family_df["feature_family"] == family)]
            values.append(float(row["positive_importance_sum_auprc"].iloc[0]) if len(row) else 0.0)
        ax.bar(x + (idx - 1.5) * width, values, width=width, label=family, color=palette[family])
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Positive AUPRC permutation importance sum")
    ax.set_title("Feature-family contribution to neonatal-risk enrichment")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def representative_records(df, model_probs, model_name, n_records):
    rows = []
    test_df = df[df["split"] == "test"].copy()
    test_df["probability"] = model_probs[model_name]
    candidates = pd.concat([
        test_df.sort_values(["neonatal_risk", "probability"], ascending=[False, False]).head(n_records),
        test_df.sort_values("probability", ascending=False).head(n_records),
        test_df.sort_values("probability", ascending=True).head(n_records),
    ]).drop_duplicates("record_id")
    for _, row in candidates.iterrows():
        rows.append({
            "model": model_name,
            "record_id": row["record_id"],
            "neonatal_risk": int(row["neonatal_risk"]),
            "probability": float(row["probability"]),
            "cord_ph": row.get("cord_ph", ""),
            "apgar5": row.get("apgar5", ""),
            "signal_phenotype": row.get("signal_phenotype", ""),
            "ssl_phenotype": row.get("ssl_phenotype", ""),
            "fhr_mean": row.get("fhr_mean", ""),
            "fhr_sd": row.get("fhr_sd", ""),
            "fhr_p05": row.get("fhr_p05", ""),
            "uc_p95": row.get("uc_p95", ""),
            "fhr_uc_corr": row.get("fhr_uc_corr", ""),
            "deceleration_proxy_count": row.get("deceleration_proxy_count", ""),
        })
    return pd.DataFrame(rows).sort_values(["model", "probability"], ascending=[True, False])


def main():
    parser = argparse.ArgumentParser(description="Explain CTU-UHB neonatal-risk prediction models.")
    parser.add_argument("--features", default="data/processed/ctu_uhb_record_features.csv")
    parser.add_argument("--embeddings", default="results/tables/sweep/p80_d64_l3_channel_cw02_record_embeddings.csv")
    parser.add_argument("--ssl-phenotypes", default="results/tables/sweep/p80_d64_l3_channel_cw02_ssl_phenotypes.csv")
    parser.add_argument("--signal-phenotypes", default="results/tables/ctu_uhb_signal_phenotypes.csv")
    parser.add_argument("--record-split", default="data/processed/ctu_uhb_record_split.csv")
    parser.add_argument("--n-repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--importance-out", default="results/tables/permutation_feature_importance.csv")
    parser.add_argument("--family-out", default="results/tables/permutation_feature_family_importance.csv")
    parser.add_argument("--coef-out", default="results/tables/model_coefficients.csv")
    parser.add_argument("--representatives-out", default="results/tables/explanation_representative_records.csv")
    parser.add_argument("--top-figure-out", default="results/figures/permutation_importance_top_features.png")
    parser.add_argument("--family-figure-out", default="results/figures/permutation_importance_feature_families.png")
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--n-representative-records", type=int, default=5)
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
    y_train = train_df["neonatal_risk"].to_numpy()
    y_test = test_df["neonatal_risk"].to_numpy()

    importance_tables = []
    coefficient_tables = []
    model_probs = {}

    for model_name, _, _ in MODEL_CONFIGS:
        numeric_cols, categorical_cols = model_feature_config(model_name, emb_cols)
        feature_cols = numeric_cols + categorical_cols
        model = build_model(numeric_cols, categorical_cols)
        model.fit(train_df[feature_cols], y_train)
        model_probs[model_name] = model.predict_proba(test_df[feature_cols])[:, 1]
        importance_tables.append(
            run_permutation(model, model_name, test_df[feature_cols], y_test, args.n_repeats, args.seed)
        )
        coefficient_tables.append(coefficient_table(model, model_name))

    importance_df = pd.concat(importance_tables, ignore_index=True)
    Path(args.importance_out).parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(args.importance_out, index=False)

    coefficient_df = pd.concat(coefficient_tables, ignore_index=True)
    coefficient_df.to_csv(args.coef_out, index=False)

    family_df = (
        importance_df.assign(
            positive_importance_auprc=lambda x: x["importance_mean_auprc"].clip(lower=0),
            positive_importance_auroc=lambda x: x["importance_mean_auroc"].clip(lower=0),
        )
        .groupby(["model", "feature_family"], as_index=False)
        .agg(
            positive_importance_sum_auprc=("positive_importance_auprc", "sum"),
            positive_importance_mean_auprc=("positive_importance_auprc", "mean"),
            positive_importance_sum_auroc=("positive_importance_auroc", "sum"),
            positive_importance_mean_auroc=("positive_importance_auroc", "mean"),
            n_features=("feature", "count"),
        )
    )
    family_df.to_csv(args.family_out, index=False)

    rep_df = representative_records(
        df,
        model_probs,
        "signal_plus_ssl",
        args.n_representative_records,
    )
    rep_df.to_csv(args.representatives_out, index=False)

    make_top_feature_plot(importance_df, args.top_figure_out, args.top_n)
    make_family_plot(family_df, args.family_figure_out)

    print(f"Wrote feature importance to {args.importance_out}")
    print(f"Wrote family importance to {args.family_out}")
    print(f"Wrote model coefficients to {args.coef_out}")
    print(f"Wrote representative records to {args.representatives_out}")
    print(f"Wrote top-feature figure to {args.top_figure_out}")
    print(f"Wrote family figure to {args.family_figure_out}")


if __name__ == "__main__":
    main()
