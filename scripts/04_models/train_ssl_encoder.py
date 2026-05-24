import argparse
from pathlib import Path

import numpy as np
import pandas as pd


try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:
    raise SystemExit(
        "PyTorch is required for this step. Install requirements first, then rerun. "
        f"Import error: {exc}"
    )


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
        return self.decoder(self.encoder(batch))


def main():
    parser = argparse.ArgumentParser(description="Train a minimal masked-reconstruction encoder on CTU windows.")
    parser.add_argument("--windows", default="data/processed/ctu_uhb_windows.npz")
    parser.add_argument("--out", default="results/models/ssl_encoder_trainonly_metrics.txt")
    parser.add_argument("--window-index", default="data/processed/ctu_uhb_window_index.csv")
    parser.add_argument("--record-split", default="data/processed/ctu_uhb_record_split.csv")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--mask-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    data = np.load(args.windows)["x"]
    split_path = Path(args.record_split)
    if split_path.exists():
        index = pd.read_csv(args.window_index, dtype={"record_id": str})
        split = pd.read_csv(split_path, dtype={"record_id": str})
        train_records = set(split.loc[split["split"] == args.train_split, "record_id"])
        keep = index["record_id"].astype(str).isin(train_records).to_numpy()
        data = data[keep]
        print(f"Using {data.shape[0]} windows from {len(train_records)} train records for SSL pretraining")
    else:
        print("No record split found; using all windows for SSL pretraining")
    x = torch.tensor(data, dtype=torch.float32)

    model = ConvAutoencoder()
    loader = DataLoader(TensorDataset(x), batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    losses = []
    model.train()
    for epoch in range(args.epochs):
        epoch_losses = []
        for (batch,) in loader:
            masked = batch.clone()
            mask = torch.rand_like(masked[:, :1, :]) < args.mask_fraction
            masked = masked.masked_fill(mask.repeat(1, 2, 1), 0.0)
            pred = model(masked)
            loss = loss_fn(pred[mask.repeat(1, 2, 1)], batch[mask.repeat(1, 2, 1)])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(sum(epoch_losses) / max(1, len(epoch_losses)))
        print(f"epoch {epoch + 1}/{args.epochs}: masked_mse={losses[-1]:.6f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path.with_suffix(".pt"))
    out_path.write_text("\n".join(f"epoch_{i + 1}_masked_mse={v:.6f}" for i, v in enumerate(losses)), encoding="utf-8")
    print(f"Wrote SSL metrics to {out_path}")
    print(f"Wrote SSL model weights to {out_path.with_suffix('.pt')}")


if __name__ == "__main__":
    main()
