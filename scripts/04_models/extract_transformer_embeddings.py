import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_transformer_ssl_encoder import PatchTransformerAutoencoder


def main():
    parser = argparse.ArgumentParser(description="Extract PatchTST-style transformer SSL embeddings from CTU windows.")
    parser.add_argument("--windows", default="data/processed/ctu_uhb_windows.npz")
    parser.add_argument("--window-index", default="data/processed/ctu_uhb_window_index.csv")
    parser.add_argument("--weights", default="results/models/transformer_ssl_encoder_metrics.pt")
    parser.add_argument("--config", default="results/models/transformer_ssl_encoder_metrics.json")
    parser.add_argument("--window-out", default="results/tables/ssl_window_embeddings.csv")
    parser.add_argument("--record-out", default="results/tables/ssl_record_embeddings.csv")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    x = np.load(args.windows)["x"].astype(np.float32)
    index = pd.read_csv(args.window_index)
    model_keys = ["channels", "length", "patch_len", "d_model", "nhead", "layers", "dropout", "projection_dim"]
    model = PatchTransformerAutoencoder(**{key: config[key] for key in model_keys if key in config})
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model.eval()

    embeddings = []
    with torch.no_grad():
        for start in range(0, x.shape[0], 64):
            batch = torch.tensor(x[start:start + 64], dtype=torch.float32)
            pooled = model.encode_clean(batch)
            embeddings.append(pooled.numpy())
    emb = np.concatenate(embeddings, axis=0)

    window_df = index[["window_id", "record_id"]].copy()
    if "neonatal_risk" in index.columns:
        window_df["neonatal_risk"] = index["neonatal_risk"]
    else:
        window_df["neonatal_risk"] = ""
    if "dataset" in index.columns:
        window_df["dataset"] = index["dataset"]
    for i in range(emb.shape[1]):
        window_df[f"emb_{i:02d}"] = emb[:, i]
    Path(args.window_out).parent.mkdir(parents=True, exist_ok=True)
    window_df.to_csv(args.window_out, index=False)

    grouped = defaultdict(list)
    label_by_record = {}
    for row_idx, record_id in enumerate(window_df["record_id"].astype(str)):
        grouped[record_id].append(emb[row_idx])
        label_by_record[record_id] = window_df.loc[row_idx, "neonatal_risk"]

    record_rows = []
    for record_id, values in sorted(grouped.items()):
        vec = np.vstack(values).mean(axis=0)
        row = {"record_id": record_id, "n_windows": len(values), "neonatal_risk": label_by_record[record_id]}
        for i, value in enumerate(vec):
            row[f"emb_{i:02d}"] = float(value)
        record_rows.append(row)

    with Path(args.record_out).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record_rows[0].keys()))
        writer.writeheader()
        writer.writerows(record_rows)

    print(f"Wrote transformer window embeddings to {args.window_out}")
    print(f"Wrote transformer record embeddings to {args.record_out}")


if __name__ == "__main__":
    main()
