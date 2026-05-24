import argparse
import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT_CONFIGS = [
    {
        "name": "p80_d48_l2_mixed_cw01",
        "patch_len": 80,
        "d_model": 48,
        "layers": 2,
        "nhead": 4,
        "mask_strategy": "mixed",
        "contrastive_weight": 0.10,
    },
    {
        "name": "p60_d64_l2_mixed_cw02",
        "patch_len": 60,
        "d_model": 64,
        "layers": 2,
        "nhead": 4,
        "mask_strategy": "mixed",
        "contrastive_weight": 0.20,
    },
    {
        "name": "p120_d64_l2_block_cw02",
        "patch_len": 120,
        "d_model": 64,
        "layers": 2,
        "nhead": 4,
        "mask_strategy": "block",
        "contrastive_weight": 0.20,
    },
    {
        "name": "p80_d64_l3_channel_cw02",
        "patch_len": 80,
        "d_model": 64,
        "layers": 3,
        "nhead": 4,
        "mask_strategy": "channel",
        "contrastive_weight": 0.20,
    },
]


def run(cmd):
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run a compact transformer SSL hyperparameter sweep.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out", default="results/tables/transformer_sweep_summary.csv")
    args = parser.parse_args()

    Path("results/models/sweep").mkdir(parents=True, exist_ok=True)
    Path("results/tables/sweep").mkdir(parents=True, exist_ok=True)
    Path("results/figures/sweep").mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for config in DEFAULT_CONFIGS:
        name = config["name"]
        model_out = f"results/models/sweep/{name}_metrics.txt"
        window_emb = f"results/tables/sweep/{name}_window_embeddings.csv"
        record_emb = f"results/tables/sweep/{name}_record_embeddings.csv"
        phenotype_out = f"results/tables/sweep/{name}_ssl_phenotypes.csv"
        phenotype_summary = f"results/tables/sweep/{name}_ssl_phenotype_summary.csv"
        phenotype_fig = f"results/figures/sweep/{name}_ssl_pca.png"
        validation_out = f"results/tables/sweep/{name}_validation_metrics.csv"
        roc_out = f"results/figures/sweep/{name}_roc.png"
        calibration_out = f"results/figures/sweep/{name}_calibration.png"

        train_cmd = [
            args.python,
            "scripts/04_models/train_transformer_ssl_encoder.py",
            "--epochs",
            str(args.epochs),
            "--out",
            model_out,
            "--patch-len",
            str(config["patch_len"]),
            "--d-model",
            str(config["d_model"]),
            "--layers",
            str(config["layers"]),
            "--nhead",
            str(config["nhead"]),
            "--mask-strategy",
            config["mask_strategy"],
            "--contrastive-weight",
            str(config["contrastive_weight"]),
        ]
        run(train_cmd)
        run([
            args.python,
            "scripts/04_models/extract_transformer_embeddings.py",
            "--weights",
            f"results/models/sweep/{name}_metrics.pt",
            "--config",
            f"results/models/sweep/{name}_metrics.json",
            "--window-out",
            window_emb,
            "--record-out",
            record_emb,
        ])
        run([
            args.python,
            "scripts/04_models/cluster_phenotypes.py",
            "--embeddings",
            record_emb,
            "--out",
            phenotype_out,
            "--summary-out",
            phenotype_summary,
            "--figure-out",
            phenotype_fig,
        ])
        run([
            args.python,
            "scripts/04_models/evaluate_downstream_tasks.py",
            "--embeddings",
            record_emb,
            "--phenotypes",
            phenotype_out,
            "--out",
            validation_out,
            "--roc-out",
            roc_out,
            "--calibration-out",
            calibration_out,
        ])

        metrics = pd.read_csv(validation_out)
        for _, row in metrics.iterrows():
            if row["model"] in ("ssl_embeddings", "signal_plus_ssl", "ssl_phenotype_only", "signal_plus_ssl_phenotype", "signal_plus_ssl_plus_signal_phenotype"):
                summary_rows.append({
                    "config": name,
                    "patch_len": config["patch_len"],
                    "d_model": config["d_model"],
                    "layers": config["layers"],
                    "nhead": config["nhead"],
                    "mask_strategy": config["mask_strategy"],
                    "contrastive_weight": config["contrastive_weight"],
                    "epochs": args.epochs,
                    "model": row["model"],
                    "auroc": row["auroc"],
                    "auprc": row["auprc"],
                    "brier": row["brier"],
                })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    ranked = pd.DataFrame(summary_rows).sort_values(["auprc", "auroc"], ascending=False)
    ranked.to_csv(out_path.with_name(out_path.stem + "_ranked.csv"), index=False)
    print(f"Wrote sweep summary to {out_path}")
    print(f"Wrote ranked sweep summary to {out_path.with_name(out_path.stem + '_ranked.csv')}")


if __name__ == "__main__":
    main()

