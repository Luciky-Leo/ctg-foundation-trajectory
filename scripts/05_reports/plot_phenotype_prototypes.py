import argparse
import csv
import re
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
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


def parse_gain(token):
    if "/" in token:
        left, _unit = token.split("/", 1)
    else:
        left = token
    if "(" in left:
        gain = float(left.split("(", 1)[0])
    else:
        gain = float(left)
    return gain


def parse_header(path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    first = lines[0].split()
    record = {"record_id": first[0], "n_signals": int(first[1]), "fs": float(first[2]), "n_samples": int(first[3])}
    signals = []
    metadata = {}
    for line in lines[1:]:
        if line.startswith("#"):
            body = line[1:].strip()
            match = re.match(r"^(.+?)\s{2,}(.+)$", body)
            if match:
                key, value = match.groups()
                key = re.sub(r"[^a-z0-9]+", "_", key.lower().replace(".", "")).strip("_")
                metadata[key] = value.strip()
        elif line.strip():
            parts = line.split()
            signals.append({"file": parts[0], "gain": parse_gain(parts[2]), "label": parts[-1]})
    return record, signals, metadata


def read_signal(indir, record_id):
    header = Path(indir) / f"{record_id}.hea"
    record, signals, metadata = parse_header(header)
    dat = (Path(indir) / signals[0]["file"]).read_bytes()
    expected = record["n_samples"] * record["n_signals"]
    values = np.array(struct.unpack("<" + "h" * expected, dat[: expected * 2]), dtype=np.float32)
    values = values.reshape(record["n_samples"], record["n_signals"])
    fhr = values[:, 0] / signals[0]["gain"]
    uc = values[:, 1] / signals[1]["gain"]
    fhr[(fhr < 50) | (fhr > 210)] = np.nan
    uc[(uc < 0) | (uc > 150)] = np.nan
    minutes = np.arange(len(fhr)) / record["fs"] / 60
    return minutes, fhr, uc, metadata


def choose_prototypes(df, phenotype_col, feature_cols):
    x = SimpleImputer(strategy="median").fit_transform(df[feature_cols])
    x = StandardScaler().fit_transform(x)
    working = df[["record_id", phenotype_col, "neonatal_risk", "cord_ph", "apgar5"]].copy()
    working["_row"] = np.arange(len(working))
    rows = []
    for phenotype, group in working.groupby(phenotype_col):
        idx = group["_row"].to_numpy()
        centroid = x[idx].mean(axis=0)
        distances = ((x[idx] - centroid) ** 2).sum(axis=1)
        best_pos = int(np.argmin(distances))
        chosen = group.iloc[best_pos]
        rows.append({
            "phenotype_type": phenotype_col,
            "phenotype": phenotype,
            "record_id": str(chosen["record_id"]),
            "distance_to_centroid": float(distances[best_pos]),
            "neonatal_risk": int(chosen["neonatal_risk"]),
            "cord_ph": chosen["cord_ph"],
            "apgar5": chosen["apgar5"],
            "n_records_in_phenotype": len(group),
        })
    return rows


def plot_record(indir, row, outdir):
    minutes, fhr, uc, metadata = read_signal(indir, row["record_id"])
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(minutes, fhr, color="#1f4e79", linewidth=0.9)
    axes[0].set_ylabel("FHR bpm")
    axes[0].set_ylim(50, 210)
    axes[0].grid(alpha=0.2)
    axes[1].plot(minutes, uc, color="#9a3412", linewidth=0.9)
    axes[1].set_ylabel("UC")
    axes[1].set_xlabel("Minutes from record start")
    axes[1].set_ylim(0, 110)
    axes[1].grid(alpha=0.2)
    fig.tight_layout(pad=0.8)
    safe_type = str(row["phenotype_type"]).replace("_", "-")
    out_path = Path(outdir) / f"{safe_type}_{row['phenotype']}_record_{row['record_id']}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Plot phenotype prototype FHR/UC curves from raw CTU-UHB records.")
    parser.add_argument("--indir", default="data/raw/ctu_uhb")
    parser.add_argument("--features", default="data/processed/ctu_uhb_record_features.csv")
    parser.add_argument("--signal-phenotypes", default="results/tables/ctu_uhb_signal_phenotypes.csv")
    parser.add_argument("--ssl-phenotypes", default="results/tables/sweep/p80_d64_l3_channel_cw02_ssl_phenotypes.csv")
    parser.add_argument("--embeddings", default="results/tables/sweep/p80_d64_l3_channel_cw02_record_embeddings.csv")
    parser.add_argument("--outdir", default="results/figures/phenotype_prototypes")
    parser.add_argument("--summary-out", default="results/tables/phenotype_prototype_records.csv")
    args = parser.parse_args()

    features = pd.read_csv(args.features, dtype={"record_id": str})
    signal_pheno = pd.read_csv(args.signal_phenotypes, dtype={"record_id": str})[["record_id", "signal_phenotype"]]
    ssl_pheno = pd.read_csv(args.ssl_phenotypes, dtype={"record_id": str})[["record_id", "ssl_phenotype"]]
    embeddings = pd.read_csv(args.embeddings, dtype={"record_id": str}).drop(columns=["neonatal_risk"], errors="ignore")
    emb_cols = [col for col in embeddings.columns if col.startswith("emb_")]

    signal_df = features.merge(signal_pheno, on="record_id", how="inner")
    ssl_df = features.merge(ssl_pheno, on="record_id", how="inner").merge(embeddings, on="record_id", how="inner")

    rows = []
    rows.extend(choose_prototypes(signal_df, "signal_phenotype", SIGNAL_FEATURES))
    rows.extend(choose_prototypes(ssl_df, "ssl_phenotype", emb_cols))

    for row in rows:
        row["figure"] = plot_record(args.indir, row, args.outdir)

    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.summary_out).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote phenotype prototype records to {args.summary_out}")
    print(f"Wrote prototype figures to {args.outdir}")


if __name__ == "__main__":
    main()
