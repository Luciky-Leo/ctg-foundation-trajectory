import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(16, 2, kernel_size=9, padding=4),
        )

    def forward(self, batch):
        encoded = self.encoder(batch)
        return self.decoder(encoded)


def main():
    parser = argparse.ArgumentParser(description="Extract SSL encoder embeddings from CTU-UHB window tensors.")
    parser.add_argument("--windows", default="data/processed/ctu_uhb_windows.npz")
    parser.add_argument("--window-index", default="data/processed/ctu_uhb_window_index.csv")
    parser.add_argument("--weights", default="results/models/ssl_encoder_trainonly_metrics.pt")
    parser.add_argument("--window-out", default="results/tables/ssl_window_embeddings.csv")
    parser.add_argument("--record-out", default="results/tables/ssl_record_embeddings.csv")
    args = parser.parse_args()

    x = np.load(args.windows)["x"].astype(np.float32)
    index = pd.read_csv(args.window_index)
    model = ConvAutoencoder()
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model.eval()

    loader = DataLoader(TensorDataset(torch.tensor(x)), batch_size=64, shuffle=False)
    embeddings = []
    with torch.no_grad():
        for (batch,) in loader:
            encoded = model.encoder(batch)
            pooled = encoded.mean(dim=2)
            embeddings.append(pooled.numpy())
    emb = np.concatenate(embeddings, axis=0)

    window_df = index[["window_id", "record_id", "neonatal_risk"]].copy()
    for i in range(emb.shape[1]):
        window_df[f"emb_{i:02d}"] = emb[:, i]
    Path(args.window_out).parent.mkdir(parents=True, exist_ok=True)
    window_df.to_csv(args.window_out, index=False)

    grouped = defaultdict(list)
    label_by_record = {}
    for row_idx, record_id in enumerate(window_df["record_id"].astype(str)):
        grouped[record_id].append(emb[row_idx])
        label_by_record[record_id] = int(window_df.loc[row_idx, "neonatal_risk"])

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

    print(f"Wrote window embeddings to {args.window_out}")
    print(f"Wrote record embeddings to {args.record_out}")


if __name__ == "__main__":
    main()
