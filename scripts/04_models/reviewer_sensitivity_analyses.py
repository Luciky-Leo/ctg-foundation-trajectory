import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
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


CONFIGS = [
    "p80_d64_l3_channel_cw02",
    "p60_d64_l2_mixed_cw02",
    "p80_d48_l2_mixed_cw01",
    "p120_d64_l2_block_cw02",
]


def build_model(numeric_cols, categorical_cols, penalty="l2", l1_ratio=None, c_value=1.0):
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))

    if penalty == "elasticnet":
        model = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="saga",
            penalty="elasticnet",
            l1_ratio=0.5 if l1_ratio is None else l1_ratio,
            C=c_value,
            random_state=20260521,
        )
    else:
        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            penalty="l2",
            C=c_value,
        )

    return Pipeline([("preprocess", ColumnTransformer(transformers=transformers)), ("model", model)])


def safe_metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    out = {
        "auroc": np.nan,
        "auprc": np.nan,
        "brier": np.nan,
        "event_prevalence": float(np.mean(y)) if len(y) else np.nan,
    }
    if len(np.unique(y)) >= 2:
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
        out["brier"] = float(brier_score_loss(y, p))
    return out


def load_selected_df(args, config="p80_d64_l3_channel_cw02"):
    features = pd.read_csv(args.features, dtype={"record_id": str})
    split = pd.read_csv(args.record_split, dtype={"record_id": str})[["record_id", "split"]]
    signal_pheno = pd.read_csv(args.signal_phenotypes, dtype={"record_id": str})[
        ["record_id", "signal_phenotype"]
    ]
    embeddings = pd.read_csv(
        Path(args.sweep_dir) / f"{config}_record_embeddings.csv",
        dtype={"record_id": str},
    ).drop(columns=["neonatal_risk"], errors="ignore")
    df = features.merge(embeddings, on="record_id", how="inner")
    df = df.merge(signal_pheno, on="record_id", how="inner")
    df = df.merge(split, on="record_id", how="inner")
    df["neonatal_risk"] = df["neonatal_risk"].astype(int)
    return df


def evaluate_fixed_split(df, label_col, numeric_cols, categorical_cols, penalty="l2", l1_ratio=None, c_value=1.0):
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
    feature_cols = numeric_cols + categorical_cols
    model = build_model(numeric_cols, categorical_cols, penalty=penalty, l1_ratio=l1_ratio, c_value=c_value)
    model.fit(train_df[feature_cols], train_df[label_col].astype(int))
    prob = model.predict_proba(test_df[feature_cols])[:, 1]
    metrics = safe_metrics(test_df[label_col].astype(int), prob)
    metrics.update(
        {
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_train_events": int(train_df[label_col].sum()),
            "n_test_events": int(test_df[label_col].sum()),
        }
    )
    return metrics


def development_selection(args):
    rows = []
    for config in CONFIGS:
        df = load_selected_df(args, config=config)
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        train_df = df[df["split"] == "train"].copy()
        dev_train_idx, dev_val_idx = train_test_split(
            train_df.index,
            test_size=0.25,
            random_state=args.seed,
            stratify=train_df["neonatal_risk"],
        )
        for model_name, categorical_cols in [
            ("signal_plus_ssl", []),
            ("signal_plus_ssl_plus_signal_phenotype", ["signal_phenotype"]),
        ]:
            numeric_cols = SIGNAL_FEATURES + emb_cols
            feature_cols = numeric_cols + categorical_cols
            model = build_model(numeric_cols, categorical_cols)
            model.fit(train_df.loc[dev_train_idx, feature_cols], train_df.loc[dev_train_idx, "neonatal_risk"])
            prob = model.predict_proba(train_df.loc[dev_val_idx, feature_cols])[:, 1]
            metrics = safe_metrics(train_df.loc[dev_val_idx, "neonatal_risk"], prob)
            rows.append(
                {
                    "analysis_type": "development_model_selection",
                    "config": config,
                    "model": model_name,
                    "selection_split": "75% original-training records fit / 25% original-training records validation",
                    "n_dev_train": int(len(dev_train_idx)),
                    "n_dev_val": int(len(dev_val_idx)),
                    "n_dev_train_events": int(train_df.loc[dev_train_idx, "neonatal_risk"].sum()),
                    "n_dev_val_events": int(train_df.loc[dev_val_idx, "neonatal_risk"].sum()),
                    **metrics,
                }
            )
    ranked = pd.DataFrame(rows).sort_values(["auprc", "auroc"], ascending=False)
    best = ranked.iloc[0].to_dict()
    best_df = load_selected_df(args, config=best["config"])
    emb_cols = [c for c in best_df.columns if c.startswith("emb_")]
    categorical_cols = ["signal_phenotype"] if best["model"] == "signal_plus_ssl_plus_signal_phenotype" else []
    test_metrics = evaluate_fixed_split(
        best_df,
        "neonatal_risk",
        SIGNAL_FEATURES + emb_cols,
        categorical_cols,
    )
    rows.append(
        {
            "analysis_type": "development_selected_test_evaluation",
            "config": best["config"],
            "model": best["model"],
            "selection_split": "selected by original-training internal validation; original held-out test evaluated once",
            **test_metrics,
        }
    )
    return rows


def pca_dimensionality_sensitivity(args):
    df = load_selected_df(args)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    train_mask = df["split"].eq("train")
    rows = []

    full_metrics = evaluate_fixed_split(
        df,
        "neonatal_risk",
        SIGNAL_FEATURES + emb_cols,
        ["signal_phenotype"],
    )
    rows.append(
        {
            "analysis_type": "dimensionality_regularization",
            "model": "signal_plus_ssl64_plus_signal_phenotype_l2",
            "ssl_dimensions": 64,
            "penalty": "l2",
            "n_predictors_before_one_hot": 77,
            "events_per_predictor_before_one_hot": full_metrics["n_train_events"] / 77,
            **full_metrics,
        }
    )

    for n_components in [5, 10, 20]:
        pca = PCA(n_components=n_components, random_state=args.seed)
        train_scores = pca.fit_transform(df.loc[train_mask, emb_cols])
        test_scores = pca.transform(df.loc[~train_mask, emb_cols])
        pca_cols = [f"ssl_pc_{i + 1:02d}" for i in range(n_components)]
        work = df.drop(columns=emb_cols).copy()
        for col in pca_cols:
            work[col] = np.nan
        work.loc[train_mask, pca_cols] = train_scores
        work.loc[~train_mask, pca_cols] = test_scores
        for penalty in ["l2", "elasticnet"]:
            metrics = evaluate_fixed_split(
                work,
                "neonatal_risk",
                SIGNAL_FEATURES + pca_cols,
                ["signal_phenotype"],
                penalty=penalty,
            )
            n_predictors = 12 + n_components + 1
            rows.append(
                {
                    "analysis_type": "dimensionality_regularization",
                    "model": f"signal_plus_ssl_pca{n_components}_plus_signal_phenotype_{penalty}",
                    "ssl_dimensions": n_components,
                    "penalty": penalty,
                    "pca_explained_variance_ratio": float(np.sum(pca.explained_variance_ratio_)),
                    "n_predictors_before_one_hot": n_predictors,
                    "events_per_predictor_before_one_hot": metrics["n_train_events"] / n_predictors,
                    **metrics,
                }
            )
    return rows


def outcome_sensitivity(args):
    df = load_selected_df(args)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    df["cord_ph_lt_7_10"] = (df["cord_ph"] < 7.10).astype(int)
    df["cord_ph_lt_7_05"] = (df["cord_ph"] < 7.05).astype(int)
    df["cord_ph_lt_7_15"] = (df["cord_ph"] < 7.15).astype(int)
    rows = []
    for label_col, label_definition in [
        ("neonatal_risk", "primary composite: cord pH <7.15 or 5-minute Apgar <7"),
        ("cord_ph_lt_7_15", "cord pH <7.15 only"),
        ("cord_ph_lt_7_10", "cord pH <7.10 only"),
        ("cord_ph_lt_7_05", "cord pH <7.05 only"),
    ]:
        for model_name, numeric_cols, categorical_cols in [
            ("signal_features", SIGNAL_FEATURES, []),
            ("signal_plus_ssl_plus_signal_phenotype", SIGNAL_FEATURES + emb_cols, ["signal_phenotype"]),
        ]:
            try:
                metrics = evaluate_fixed_split(df, label_col, numeric_cols, categorical_cols)
                prevalence = metrics["event_prevalence"]
                enrichment = metrics["auprc"] / prevalence if prevalence and not math.isnan(metrics["auprc"]) else np.nan
            except Exception as exc:
                metrics = {
                    "n_train": int((df["split"] == "train").sum()),
                    "n_test": int((df["split"] == "test").sum()),
                    "n_train_events": int(df.loc[df["split"] == "train", label_col].sum()),
                    "n_test_events": int(df.loc[df["split"] == "test", label_col].sum()),
                    "event_prevalence": float(df.loc[df["split"] == "test", label_col].mean()),
                    "auroc": np.nan,
                    "auprc": np.nan,
                    "brier": np.nan,
                }
                enrichment = np.nan
                metrics["error"] = str(exc)
            rows.append(
                {
                    "analysis_type": "outcome_sensitivity",
                    "outcome": label_col,
                    "outcome_definition": label_definition,
                    "model": model_name,
                    "auprc_enrichment_over_test_prevalence": enrichment,
                    **metrics,
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Reviewer-requested sensitivity analyses for CTG manuscript.")
    parser.add_argument("--features", default="data/processed/ctu_uhb_record_features.csv")
    parser.add_argument("--record-split", default="data/processed/ctu_uhb_record_split.csv")
    parser.add_argument("--signal-phenotypes", default="results/tables/ctu_uhb_signal_phenotypes.csv")
    parser.add_argument("--sweep-dir", default="results/tables/sweep")
    parser.add_argument("--out", default="results/tables/reviewer_sensitivity_analyses.csv")
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    rows = []
    rows.extend(development_selection(args))
    rows.extend(pca_dimensionality_sensitivity(args))
    rows.extend(outcome_sensitivity(args))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote reviewer sensitivity analyses to {out}")


if __name__ == "__main__":
    main()
