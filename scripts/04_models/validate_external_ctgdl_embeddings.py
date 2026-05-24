import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def embedding_cols(df):
    return [col for col in df.columns if col.startswith("emb_")]


def main():
    parser = argparse.ArgumentParser(description="External representation validation using CTGDL FHRMA embeddings.")
    parser.add_argument("--ctu-embeddings", default="results/tables/sweep/p80_d64_l3_channel_cw02_record_embeddings.csv")
    parser.add_argument("--ctu-phenotypes", default="results/tables/sweep/p80_d64_l3_channel_cw02_ssl_phenotypes.csv")
    parser.add_argument("--external-embeddings", default="results/tables/ctgdl_fhrma_record_embeddings.csv")
    parser.add_argument("--summary-out", default="results/tables/ctgdl_external_embedding_validation.csv")
    parser.add_argument("--assignment-out", default="results/tables/ctgdl_fhrma_assigned_ssl_phenotypes.csv")
    parser.add_argument("--figure-out", default="results/figures/ctgdl_external_embedding_pca.png")
    args = parser.parse_args()

    ctu = pd.read_csv(args.ctu_embeddings, dtype={"record_id": str})
    ctu_pheno = pd.read_csv(args.ctu_phenotypes, dtype={"record_id": str})[["record_id", "ssl_phenotype"]]
    ctu = ctu.merge(ctu_pheno, on="record_id", how="inner")
    external = pd.read_csv(args.external_embeddings, dtype={"record_id": str})
    cols = embedding_cols(ctu)

    scaler = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    ctu_x = scaler.fit_transform(ctu[cols])
    ext_x = scaler.transform(external[cols])

    centroids = {}
    for phenotype, group in ctu.groupby("ssl_phenotype"):
        centroids[int(phenotype)] = ctu_x[group.index.to_numpy()].mean(axis=0)
    centroid_ids = sorted(centroids)
    centroid_matrix = np.vstack([centroids[i] for i in centroid_ids])
    distances = ((ext_x[:, None, :] - centroid_matrix[None, :, :]) ** 2).sum(axis=2)
    assigned = [centroid_ids[i] for i in distances.argmin(axis=1)]
    external_assignments = external[["record_id", "n_windows"]].copy()
    external_assignments["assigned_ctu_ssl_phenotype"] = assigned
    Path(args.assignment_out).parent.mkdir(parents=True, exist_ok=True)
    external_assignments.to_csv(args.assignment_out, index=False)

    domain_x = np.vstack([ctu_x, ext_x])
    domain_y = np.array([0] * len(ctu_x) + [1] * len(ext_x))
    train_x, test_x, train_y, test_y = train_test_split(
        domain_x,
        domain_y,
        test_size=0.30,
        random_state=20260521,
        stratify=domain_y,
    )
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    clf.fit(train_x, train_y)
    domain_prob = clf.predict_proba(test_x)[:, 1]
    domain_auc = roc_auc_score(test_y, domain_prob)

    pca = PCA(n_components=2, random_state=20260521)
    coords = pca.fit_transform(domain_x)
    plot_df = pd.DataFrame({
        "pc1": coords[:, 0],
        "pc2": coords[:, 1],
        "dataset": ["CTU-UHB"] * len(ctu_x) + ["CTGDL-FHRMA"] * len(ext_x),
    })
    Path(args.figure_out).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    for dataset, group in plot_df.groupby("dataset"):
        plt.scatter(group["pc1"], group["pc2"], s=24, alpha=0.75, label=dataset)
    plt.xlabel("Embedding PC1")
    plt.ylabel("Embedding PC2")
    plt.title("CTU-UHB vs CTGDL-FHRMA transformer embedding space")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.figure_out, dpi=180)
    plt.close()

    phenotype_counts = external_assignments["assigned_ctu_ssl_phenotype"].value_counts().sort_index()
    rows = [
        {"metric": "ctu_records", "value": len(ctu)},
        {"metric": "ctgdl_fhrma_records", "value": len(external)},
        {"metric": "domain_classifier_auroc_ctgdl_vs_ctu", "value": round(float(domain_auc), 4)},
        {"metric": "pca_explained_variance_pc1_pc2", "value": round(float(pca.explained_variance_ratio_.sum()), 4)},
    ]
    for phenotype, count in phenotype_counts.items():
        rows.append({"metric": f"ctgdl_assigned_ssl_phenotype_{phenotype}", "value": int(count)})

    with Path(args.summary_out).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote CTGDL external validation summary to {args.summary_out}")
    print(f"Wrote CTGDL phenotype assignments to {args.assignment_out}")
    print(f"Wrote CTGDL PCA figure to {args.figure_out}")


if __name__ == "__main__":
    main()

