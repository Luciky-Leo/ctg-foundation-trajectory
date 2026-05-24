import argparse
import csv
import math
import re
import struct
from pathlib import Path

import numpy as np


def normalize_key(key):
    key = key.strip().lower().replace(".", "")
    key = key.replace("(", "_").replace(")", "")
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return key.strip("_")


def parse_gain(token):
    if "/" in token:
        left, unit = token.split("/", 1)
    else:
        left, unit = token, ""
    if "(" in left:
        gain = float(left.split("(", 1)[0])
    else:
        gain = float(left)
    return gain, unit


def parse_header(path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    first = lines[0].split()
    record = {
        "record_id": first[0],
        "n_signals": int(first[1]),
        "sampling_frequency": float(first[2]),
        "n_samples": int(first[3]),
    }
    signals = []
    metadata = {}
    for line in lines[1:]:
        if line.startswith("#"):
            body = line[1:].strip()
            if not body or body.startswith("--"):
                continue
            match = re.match(r"^(.+?)\s{2,}(.+)$", body)
            if match:
                key, value = match.groups()
                metadata[normalize_key(key)] = value.strip()
        elif line.strip():
            parts = line.split()
            gain, unit = parse_gain(parts[2])
            signals.append({"file": parts[0], "gain": gain, "unit": unit, "label": parts[-1]})
    return record, signals, metadata


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_signal(indir, header_path):
    record, signals, metadata = parse_header(header_path)
    dat_path = indir / signals[0]["file"]
    raw = dat_path.read_bytes()
    expected_values = record["n_samples"] * record["n_signals"]
    values = np.array(struct.unpack("<" + "h" * expected_values, raw[: expected_values * 2]), dtype=np.float32)
    values = values.reshape(record["n_samples"], record["n_signals"])
    values[:, 0] = values[:, 0] / signals[0]["gain"]
    values[:, 1] = values[:, 1] / signals[1]["gain"]
    invalid_fhr = (values[:, 0] < 50) | (values[:, 0] > 210)
    invalid_uc = (values[:, 1] < 0) | (values[:, 1] > 150)
    values[invalid_fhr, 0] = np.nan
    values[invalid_uc, 1] = np.nan
    return record, metadata, values


def robust_fill_and_scale(window):
    out = window.copy()
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
    parser = argparse.ArgumentParser(description="Build CTU-UHB window tensors for self-supervised modeling.")
    parser.add_argument("--indir", default="data/raw/ctu_uhb")
    parser.add_argument("--out", default="data/processed/ctu_uhb_windows.npz")
    parser.add_argument("--index-out", default="data/processed/ctu_uhb_window_index.csv")
    parser.add_argument("--window-minutes", type=float, default=10)
    parser.add_argument("--stride-minutes", type=float, default=5)
    parser.add_argument("--max-windows", type=int, default=0, help="Use 0 for all windows.")
    args = parser.parse_args()

    indir = Path(args.indir)
    headers = sorted(indir.glob("*.hea"))
    windows = []
    index_rows = []
    for header in headers:
        record, metadata, signal = read_signal(indir, header)
        fs = record["sampling_frequency"]
        win = int(args.window_minutes * 60 * fs)
        stride = int(args.stride_minutes * 60 * fs)
        ph = safe_float(metadata.get("ph"))
        apgar5 = safe_float(metadata.get("apgar5"))
        neonatal_risk = int((ph is not None and ph < 7.15) or (apgar5 is not None and apgar5 < 7))
        for start in range(0, max(1, signal.shape[0] - win + 1), stride):
            stop = start + win
            if stop > signal.shape[0]:
                break
            window = robust_fill_and_scale(signal[start:stop, :])
            windows.append(window.T.astype(np.float32))
            index_rows.append({
                "window_id": len(windows) - 1,
                "record_id": record["record_id"],
                "start_sample": start,
                "stop_sample": stop,
                "start_minute": round(start / fs / 60, 3),
                "stop_minute": round(stop / fs / 60, 3),
                "cord_ph": metadata.get("ph", ""),
                "apgar5": metadata.get("apgar5", ""),
                "neonatal_risk": neonatal_risk,
            })
            if args.max_windows > 0 and len(windows) >= args.max_windows:
                break
        if args.max_windows > 0 and len(windows) >= args.max_windows:
            break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.stack(windows, axis=0)
    np.savez_compressed(out_path, x=x)

    index_path = Path(args.index_out)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0].keys()))
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"Wrote windows {x.shape} to {out_path}")
    print(f"Wrote window index with {len(index_rows)} rows to {index_path}")


if __name__ == "__main__":
    main()

