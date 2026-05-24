import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class PatchTransformerAutoencoder(nn.Module):
    def __init__(self, channels=2, length=2400, patch_len=80, d_model=48, nhead=4, layers=2, dropout=0.1, projection_dim=64):
        super().__init__()
        if length % patch_len != 0:
            raise ValueError("length must be divisible by patch_len")
        self.channels = channels
        self.length = length
        self.patch_len = patch_len
        self.n_patches = length // patch_len
        self.n_tokens = channels * self.n_patches
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))
        self.channel_embed = nn.Parameter(torch.zeros(1, channels, 1, d_model))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.decoder = nn.Linear(d_model, patch_len)
        self.projector = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.channel_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def patchify(self, x):
        b, c, l = x.shape
        patches = x.reshape(b, c, self.n_patches, self.patch_len)
        return patches

    def embed_patches(self, patches):
        z = self.patch_embed(patches)
        z = z + self.channel_embed
        z = z.reshape(patches.shape[0], self.n_tokens, -1)
        z = z + self.pos_embed
        return z

    def forward(self, x, mask):
        patches = self.patchify(x)
        z = self.embed_patches(patches)
        z = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(z), z)
        encoded = self.encoder(z)
        pred = self.decoder(encoded).reshape(x.shape[0], self.channels, self.n_patches, self.patch_len)
        return pred, patches, encoded

    def pool_encoded(self, encoded):
        return encoded.mean(dim=1)

    def project(self, encoded):
        return self.projector(self.pool_encoded(encoded))

    def encode_clean(self, x):
        patches = self.patchify(x)
        tokens = self.embed_patches(patches)
        encoded = self.encoder(tokens)
        return self.pool_encoded(encoded)


def load_train_windows(args):
    data = np.load(args.windows)["x"].astype(np.float32)
    split_path = Path(args.record_split)
    if split_path.exists():
        index = pd.read_csv(args.window_index, dtype={"record_id": str})
        split = pd.read_csv(split_path, dtype={"record_id": str})
        train_records = set(split.loc[split["split"] == args.train_split, "record_id"])
        keep = index["record_id"].astype(str).isin(train_records).to_numpy()
        data = data[keep]
        print(f"Using {data.shape[0]} windows from {len(train_records)} train records for transformer SSL pretraining")
    else:
        print("No record split found; using all windows for transformer SSL pretraining")
    return data


def make_mask(batch_size, channels, n_patches, strategy, mask_fraction, device):
    mask = torch.zeros(batch_size, channels, n_patches, dtype=torch.bool, device=device)
    if strategy == "random":
        mask = torch.rand(batch_size, channels, n_patches, device=device) < mask_fraction
    elif strategy == "block":
        span = max(1, int(round(n_patches * mask_fraction)))
        for i in range(batch_size):
            start = torch.randint(0, max(1, n_patches - span + 1), (1,), device=device).item()
            mask[i, :, start:start + span] = True
    elif strategy == "channel":
        span = max(1, int(round(n_patches * mask_fraction)))
        for i in range(batch_size):
            channel = torch.randint(0, channels, (1,), device=device).item()
            start = torch.randint(0, max(1, n_patches - span + 1), (1,), device=device).item()
            mask[i, channel, start:start + span] = True
    elif strategy == "mixed":
        random_mask = torch.rand(batch_size, channels, n_patches, device=device) < (mask_fraction * 0.5)
        mask |= random_mask
        span = max(1, int(round(n_patches * mask_fraction * 0.75)))
        for i in range(batch_size):
            mode = torch.randint(0, 2, (1,), device=device).item()
            start = torch.randint(0, max(1, n_patches - span + 1), (1,), device=device).item()
            if mode == 0:
                mask[i, :, start:start + span] = True
            else:
                channel = torch.randint(0, channels, (1,), device=device).item()
                mask[i, channel, start:start + span] = True
    else:
        raise ValueError(f"Unknown mask strategy: {strategy}")

    empty = ~mask.reshape(batch_size, -1).any(dim=1)
    if empty.any():
        mask[empty, 0, 0] = True
    return mask.reshape(batch_size, channels * n_patches)


def masked_mse(pred, target, flat_mask):
    b, c, p, patch_len = pred.shape
    patch_mask = flat_mask.reshape(b, c, p)
    expanded = patch_mask.unsqueeze(-1).expand_as(pred)
    return ((pred - target) ** 2)[expanded].mean()


def nt_xent_loss(z1, z2, temperature):
    if z1.shape[0] < 2:
        return z1.new_tensor(0.0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    logits = z @ z.T / temperature
    logits.fill_diagonal_(-1e9)
    batch_size = z1.shape[0]
    labels = torch.arange(2 * batch_size, device=z.device)
    labels = (labels + batch_size) % (2 * batch_size)
    return F.cross_entropy(logits, labels)


def main():
    parser = argparse.ArgumentParser(description="Train a PatchTST-style masked transformer encoder on CTU windows.")
    parser.add_argument("--windows", default="data/processed/ctu_uhb_windows.npz")
    parser.add_argument("--window-index", default="data/processed/ctu_uhb_window_index.csv")
    parser.add_argument("--record-split", default="data/processed/ctu_uhb_record_split.csv")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--out", default="results/models/transformer_ssl_encoder_metrics.txt")
    parser.add_argument("--patch-len", type=int, default=80)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--mask-strategy", choices=["random", "block", "channel", "mixed"], default="mixed")
    parser.add_argument("--mask-fraction", type=float, default=0.25)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--jitter-std", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    data = load_train_windows(args)
    x = torch.tensor(data, dtype=torch.float32)
    model = PatchTransformerAutoencoder(
        channels=x.shape[1],
        length=x.shape[2],
        patch_len=args.patch_len,
        d_model=args.d_model,
        nhead=args.nhead,
        layers=args.layers,
        dropout=args.dropout,
        projection_dim=args.projection_dim,
    )
    loader = DataLoader(TensorDataset(x), batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    losses = []
    model.train()
    for epoch in range(args.epochs):
        total_losses = []
        recon_losses = []
        contrastive_losses = []
        for (batch,) in loader:
            b = batch.shape[0]
            mask1 = make_mask(b, model.channels, model.n_patches, args.mask_strategy, args.mask_fraction, batch.device)
            mask2 = make_mask(b, model.channels, model.n_patches, args.mask_strategy, args.mask_fraction, batch.device)
            view1 = batch + torch.randn_like(batch) * args.jitter_std
            view2 = batch + torch.randn_like(batch) * args.jitter_std
            pred1, target1, encoded1 = model(view1, mask1)
            pred2, target2, encoded2 = model(view2, mask2)
            recon_loss = 0.5 * (masked_mse(pred1, target1, mask1) + masked_mse(pred2, target2, mask2))
            contrastive_loss = nt_xent_loss(model.project(encoded1), model.project(encoded2), args.temperature)
            loss = recon_loss + args.contrastive_weight * contrastive_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_losses.append(float(loss.detach()))
            recon_losses.append(float(recon_loss.detach()))
            contrastive_losses.append(float(contrastive_loss.detach()))
        epoch_total = sum(total_losses) / max(1, len(total_losses))
        epoch_recon = sum(recon_losses) / max(1, len(recon_losses))
        epoch_contrastive = sum(contrastive_losses) / max(1, len(contrastive_losses))
        losses.append((epoch_total, epoch_recon, epoch_contrastive))
        print(
            f"epoch {epoch + 1}/{args.epochs}: "
            f"total={epoch_total:.6f} recon={epoch_recon:.6f} contrastive={epoch_contrastive:.6f}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path = out_path.with_suffix(".pt")
    config_path = out_path.with_suffix(".json")
    torch.save(model.state_dict(), weights_path)
    out_path.write_text(
        "\n".join(
            f"epoch_{i + 1}_total={total:.6f},recon={recon:.6f},contrastive={contrastive:.6f}"
            for i, (total, recon, contrastive) in enumerate(losses)
        ),
        encoding="utf-8",
    )
    config = {
        "channels": int(x.shape[1]),
        "length": int(x.shape[2]),
        "patch_len": args.patch_len,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "layers": args.layers,
        "dropout": args.dropout,
        "projection_dim": args.projection_dim,
        "mask_strategy": args.mask_strategy,
        "mask_fraction": args.mask_fraction,
        "contrastive_weight": args.contrastive_weight,
        "temperature": args.temperature,
        "jitter_std": args.jitter_std,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Wrote transformer SSL metrics to {out_path}")
    print(f"Wrote transformer SSL model weights to {weights_path}")
    print(f"Wrote transformer SSL config to {config_path}")


if __name__ == "__main__":
    main()
