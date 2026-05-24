import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser(description="Cluster SSL record embeddings into dynamic CTG phenotypes.")
    parser.add_argument("--embeddings", default="results/tables/ssl_record_embeddings.csv")
    parser.add_argument("--features", default="data/processed/ctu_uhb_record_features.csv")
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--record-split", default="data/processed/ctu_uhb_record_split.csv")
    parser.add_argument("--out", default="results/tables/ctu_uhb_ssl_phenotypes.csv")
    parser.add_argument("--summary-out", default="results/tables/ctu_uhb_ssl_phenotype_summary.csv")
    parser.add_argument("--figure-out", default="results/figures/ssl_phenotype_pca.png")
    args = parser.parse_args()

    emb = pd.read_csv(args.embeddings)
    features = pd.read_csv(args.features, dtype={"record_id": str})
    emb["record_id"] = emb["record_id"].astype(str)
    df = features.merge(emb, on=["record_id", "neonatal_risk"], how="inner")
    emb_cols = [col for col in df.columns if col.startswith("emb_")]

    scaler = StandardScaler()
    split_path = Path(args.record_split)
    if split_path.exists():
        split = pd.read_csv(split_path, dtype={"record_id": str})[["record_id", "split"]]
        df = df.merge(split, on="record_id", how="left")
        train_mask = df["split"].eq("train").to_numpy()
    else:
        train_mask = [True] * len(df)
    x_train = scaler.fit_transform(df.loc[train_mask, emb_cols])
    x = scaler.transform(df[emb_cols])
    km = KMeans(n_clusters=args.clusters, random_state=20260521, n_init=50)
    km.fit(x_train)
    df["ssl_phenotype"] = km.predict(x) + 1

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
    summary = (
        df.groupby("ssl_phenotype")[summary_cols]
        .agg(["count", "mean", "median"])
        .round(4)
    )
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]
    summary = summary.reset_index()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    keep_cols = [
        "record_id",
        "ssl_phenotype",
        "neonatal_risk",
        "cord_ph",
        "apgar5",
        "fhr_mean",
        "fhr_sd",
        "uc_mean",
        "uc_p95",
        "fhr_uc_corr",
        "deceleration_proxy_count",
    ]
    df[keep_cols].to_csv(args.out, index=False)
    summary.to_csv(args.summary_out, index=False)

    pca = PCA(n_components=2, random_state=20260521)
    coords = pca.fit_transform(x)
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=df["ssl_phenotype"], s=28, cmap="tab10", alpha=0.85)
    plt.xlabel("SSL embedding PC1")
    plt.ylabel("SSL embedding PC2")
    plt.title("CTU-UHB SSL dynamic phenotypes")
    plt.colorbar(scatter, label="Phenotype")
    Path(args.figure_out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.figure_out, dpi=180)
    plt.close()

    print(f"Wrote phenotype assignments to {args.out}")
    print(f"Wrote phenotype summary to {args.summary_out}")
    print(f"Wrote phenotype PCA figure to {args.figure_out}")


if __name__ == "__main__":
    main()
