import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
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


def build_model(numeric_cols, categorical_cols=None):
    categorical_cols = categorical_cols or []
    transformers = []
    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
            numeric_cols,
        ))
    if categorical_cols:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols,
        ))
    pre = ColumnTransformer(transformers=transformers)
    return Pipeline([
        ("preprocess", pre),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
    ])


def evaluate_model(name, model, train_df, test_df, feature_cols, categorical_cols=None):
    categorical_cols = categorical_cols or []
    x_train = train_df[feature_cols + categorical_cols]
    x_test = test_df[feature_cols + categorical_cols]
    y_train = train_df["neonatal_risk"].astype(int)
    y_test = test_df["neonatal_risk"].astype(int)
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {
        "model": name,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "event_rate_test": round(float(y_test.mean()), 4),
        "auroc": round(float(roc_auc_score(y_test, prob)), 4),
        "auprc": round(float(average_precision_score(y_test, prob)), 4),
        "brier": round(float(brier_score_loss(y_test, prob)), 4),
        "sensitivity_at_0_5": round(float(tp / max(1, tp + fn)), 4),
        "specificity_at_0_5": round(float(tn / max(1, tn + fp)), 4),
        "probabilities": prob,
        "y_test": y_test.to_numpy(),
    }


def main():
    parser = argparse.ArgumentParser(description="Run record-level train/test validation for CTU-UHB SSL phenotypes.")
    parser.add_argument("--features", default="data/processed/ctu_uhb_record_features.csv")
    parser.add_argument("--embeddings", default="results/tables/ssl_record_embeddings.csv")
    parser.add_argument("--phenotypes", default="results/tables/ctu_uhb_ssl_phenotypes.csv")
    parser.add_argument("--signal-phenotypes", default="results/tables/ctu_uhb_signal_phenotypes.csv")
    parser.add_argument("--record-split", default="data/processed/ctu_uhb_record_split.csv")
    parser.add_argument("--out", default="results/tables/downstream_validation_metrics.csv")
    parser.add_argument("--roc-out", default="results/figures/downstream_roc.png")
    parser.add_argument("--calibration-out", default="results/figures/downstream_calibration.png")
    args = parser.parse_args()

    features = pd.read_csv(args.features, dtype={"record_id": str})
    embeddings = pd.read_csv(args.embeddings, dtype={"record_id": str})
    phenotypes = pd.read_csv(args.phenotypes, dtype={"record_id": str})[["record_id", "ssl_phenotype"]]
    signal_phenotypes = pd.read_csv(args.signal_phenotypes, dtype={"record_id": str})[["record_id", "signal_phenotype"]]
    df = features.merge(embeddings.drop(columns=["neonatal_risk"], errors="ignore"), on="record_id", how="inner")
    df = df.merge(phenotypes, on="record_id", how="inner")
    df = df.merge(signal_phenotypes, on="record_id", how="inner")
    df["neonatal_risk"] = df["neonatal_risk"].astype(int)
    emb_cols = [col for col in df.columns if col.startswith("emb_")]
    split_path = Path(args.record_split)
    if split_path.exists():
        split = pd.read_csv(split_path, dtype={"record_id": str})[["record_id", "split"]]
        df = df.merge(split, on="record_id", how="inner")
        train_df = df[df["split"] == "train"].copy()
        test_df = df[df["split"] == "test"].copy()
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=0.30,
            random_state=20260521,
            stratify=df["neonatal_risk"],
        )

    configs = [
        ("signal_features", SIGNAL_FEATURES, []),
        ("ssl_embeddings", emb_cols, []),
        ("signal_plus_ssl", SIGNAL_FEATURES + emb_cols, []),
        ("ssl_phenotype_only", [], ["ssl_phenotype"]),
        ("signal_phenotype_only", [], ["signal_phenotype"]),
        ("signal_plus_ssl_phenotype", SIGNAL_FEATURES, ["ssl_phenotype"]),
        ("signal_plus_signal_phenotype", SIGNAL_FEATURES, ["signal_phenotype"]),
        ("signal_plus_ssl_plus_signal_phenotype", SIGNAL_FEATURES + emb_cols, ["signal_phenotype"]),
    ]

    metrics = []
    curves = []
    for name, numeric_cols, categorical_cols in configs:
        model = build_model(numeric_cols, categorical_cols)
        result = evaluate_model(name, model, train_df, test_df, numeric_cols, categorical_cols)
        metrics.append({key: value for key, value in result.items() if key not in ("probabilities", "y_test")})
        curves.append((name, result["y_test"], result["probabilities"]))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(out_path, index=False)

    Path(args.roc_out).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.5, 5))
    for name, y_test, prob in curves:
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        plt.plot(fpr, tpr, label=f"{name} AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Record-level CTU-UHB validation ROC")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.roc_out, dpi=180)
    plt.close()

    plt.figure(figsize=(6.5, 5))
    for name, y_test, prob in curves:
        frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=5, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", label=name)
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Mean predicted risk")
    plt.ylabel("Observed event fraction")
    plt.title("Record-level CTU-UHB calibration")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.calibration_out, dpi=180)
    plt.close()

    print(f"Wrote validation metrics to {args.out}")
    print(f"Wrote ROC figure to {args.roc_out}")
    print(f"Wrote calibration figure to {args.calibration_out}")


if __name__ == "__main__":
    main()
