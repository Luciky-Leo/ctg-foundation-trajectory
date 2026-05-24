import argparse
import csv
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd


def robust_fill_and_scale(window):
    out = window.copy().astype(np.float32)
    for channel in range(out.shape[1]):
        x = out[:, channel]
        finite = np.isfinite(x)
        if finite.sum() == 0:
            out[:, channel] = 0
            continue
        median = np.nanmedian(x)
        x[~finite] = median
        q25, q75 = np.percentile(x, [25, 75])
        scale = q75 - q25
        if scale <= 1e-6:
            scale = np.std(x) or 1.0
        out[:, channel] = (x - median) / scale
    return out


def main():
    parser = argparse.ArgumentParser(description="Build CTGDL FHRMA windows aligned to CTU-UHB transformer inputs.")
    parser.add_argument("--archive", default="data/raw/ctgdl/v5_open/CTGDL_FHRMA_proc_csv.tar.gz")
    parser.add_argument("--out", default="data/processed/ctgdl_fhrma_windows.npz")
    parser.add_argument("--index-out", default="data/processed/ctgdl_fhrma_window_index.csv")
    parser.add_argument("--window-points", type=int, default=2400)
    parser.add_argument("--stride-points", type=int, default=1200)
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()

    windows = []
    rows = []
    with tarfile.open(args.archive, "r:gz") as archive:
        members = [m for m in archive.getmembers() if m.isfile() and m.name.endswith(".csv")]
        members = sorted(members, key=lambda m: m.name)
        if args.max_records > 0:
            members = members[: args.max_records]
        for member in members:
            with archive.extractfile(member) as handle:
                df = pd.read_csv(handle)
            if not {"fhr", "uc"}.issubset(df.columns):
                continue
            signal = df[["fhr", "uc"]].to_numpy(dtype=np.float32)
            signal[:, 0] = np.where((signal[:, 0] >= 50) & (signal[:, 0] <= 210), signal[:, 0], np.nan)
            signal[:, 1] = np.where((signal[:, 1] >= 0) & (signal[:, 1] <= 150), signal[:, 1], np.nan)
            record_id = Path(member.name).stem
            for start in range(0, max(1, len(signal) - args.window_points + 1), args.stride_points):
                stop = start + args.window_points
                if stop > len(signal):
                    break
                window = robust_fill_and_scale(signal[start:stop, :])
                windows.append(window.T.astype(np.float32))
                rows.append({
                    "window_id": len(windows) - 1,
                    "record_id": record_id,
                    "start_sample": start,
                    "stop_sample": stop,
                    "dataset": "CTGDL_FHRMA",
                })

    if not windows:
        raise RuntimeError("No CTGDL FHRMA windows were built")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, x=np.stack(windows, axis=0))
    index_path = Path(args.index_out)
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CTGDL FHRMA windows {len(windows)} to {out_path}")
    print(f"Wrote CTGDL FHRMA index to {index_path}")


if __name__ == "__main__":
    main()

