import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


def main():
    parser = argparse.ArgumentParser(description="Cluster clinically interpretable signal-derived CTG phenotypes.")
    parser.add_argument("--features", default="data/processed/ctu_uhb_record_features.csv")
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--record-split", default="data/processed/ctu_uhb_record_split.csv")
    parser.add_argument("--out", default="results/tables/ctu_uhb_signal_phenotypes.csv")
    parser.add_argument("--summary-out", default="results/tables/ctu_uhb_signal_phenotype_summary.csv")
    parser.add_argument("--figure-out", default="results/figures/signal_phenotype_pca.png")
    args = parser.parse_args()

    df = pd.read_csv(args.features, dtype={"record_id": str})
    pre = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    split_path = Path(args.record_split)
    if split_path.exists():
        split = pd.read_csv(split_path, dtype={"record_id": str})[["record_id", "split"]]
        df = df.merge(split, on="record_id", how="left")
        train_mask = df["split"].eq("train").to_numpy()
    else:
        train_mask = [True] * len(df)
    x_train = pre.fit_transform(df.loc[train_mask, SIGNAL_FEATURES])
    x = pre.transform(df[SIGNAL_FEATURES])
    km = KMeans(n_clusters=args.clusters, random_state=20260521, n_init=50)
    km.fit(x_train)
    df["signal_phenotype"] = km.predict(x) + 1

    summary_cols = [
        "neonatal_risk",
        "cord_ph",
        "apgar5",
        "fhr_mean",
        "fhr_sd",
        "fhr_p05",
        "uc_mean",
        "uc_p95",
        "fhr_uc_corr",
        "deceleration_proxy_count",
    ]
    summary = df.groupby("signal_phenotype")[summary_cols].agg(["count", "mean", "median"]).round(4)
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]
    summary = summary.reset_index()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    keep_cols = ["record_id", "signal_phenotype"] + summary_cols
    df[keep_cols].to_csv(args.out, index=False)
    summary.to_csv(args.summary_out, index=False)

    coords = PCA(n_components=2, random_state=20260521).fit_transform(x)
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=df["signal_phenotype"], s=28, cmap="tab10", alpha=0.85)
    plt.xlabel("Signal feature PC1")
    plt.ylabel("Signal feature PC2")
    plt.title("CTU-UHB signal-derived dynamic phenotypes")
    plt.colorbar(scatter, label="Phenotype")
    Path(args.figure_out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.figure_out, dpi=180)
    plt.close()

    print(f"Wrote signal phenotype assignments to {args.out}")
    print(f"Wrote signal phenotype summary to {args.summary_out}")
    print(f"Wrote signal phenotype PCA figure to {args.figure_out}")


if __name__ == "__main__":
    main()
