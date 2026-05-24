import argparse
import csv
import math
import re
import struct
from pathlib import Path


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


def mean(values):
    values = [x for x in values if x is not None]
    return sum(values) / len(values) if values else None


def sd(values):
    values = [x for x in values if x is not None]
    if len(values) < 2:
        return None
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def percentile(values, q):
    values = sorted(x for x in values if x is not None)
    if not values:
        return None
    idx = int(round((len(values) - 1) * q))
    return values[idx]


def pearson(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    mx = mean(xs)
    my = mean(ys)
    sx = sd(xs)
    sy = sd(ys)
    if not sx or not sy:
        return None
    return sum((a - mx) * (b - my) for a, b in pairs) / ((len(pairs) - 1) * sx * sy)


def deceleration_proxy_count(fhr_values):
    clean = [x for x in fhr_values if x is not None]
    if len(clean) < 20:
        return 0
    baseline = percentile(clean, 0.5)
    return sum(1 for x in clean if x < baseline - 15)


def read_record(indir, header_path):
    record, signals, metadata = parse_header(header_path)
    dat_path = indir / signals[0]["file"]
    raw = dat_path.read_bytes()
    expected_values = record["n_samples"] * record["n_signals"]
    values = struct.unpack("<" + "h" * expected_values, raw[: expected_values * 2])
    fhr = []
    uc = []
    for i in range(record["n_samples"]):
        fhr_value = values[i * 2] / signals[0]["gain"]
        uc_value = values[i * 2 + 1] / signals[1]["gain"]
        fhr.append(fhr_value if 50 <= fhr_value <= 210 else None)
        uc.append(uc_value if 0 <= uc_value <= 150 else None)
    return record, metadata, fhr, uc


def fmt(value, digits=4):
    return round(value, digits) if value is not None else ""


def main():
    parser = argparse.ArgumentParser(description="Extract record-level CTG features directly from CTU-UHB WFDB-16 files.")
    parser.add_argument("--indir", default="data/raw/ctu_uhb")
    parser.add_argument("--out", default="data/processed/ctu_uhb_record_features.csv")
    args = parser.parse_args()

    indir = Path(args.indir)
    headers = sorted(indir.glob("*.hea"))
    if not headers:
        raise FileNotFoundError(f"No .hea files found in {indir}")

    rows = []
    for header in headers:
        record, metadata, fhr, uc = read_record(indir, header)
        ph = safe_float(metadata.get("ph"))
        apgar5 = safe_float(metadata.get("apgar5"))
        neonatal_risk = int((ph is not None and ph < 7.15) or (apgar5 is not None and apgar5 < 7))
        valid_fhr = [x for x in fhr if x is not None]
        valid_uc = [x for x in uc if x is not None]
        rows.append({
            "record_id": record["record_id"],
            "n_samples": record["n_samples"],
            "duration_min": fmt(record["n_samples"] / record["sampling_frequency"] / 60),
            "fhr_missing_fraction": fmt(1 - len(valid_fhr) / max(1, len(fhr))),
            "uc_missing_fraction": fmt(1 - len(valid_uc) / max(1, len(uc))),
            "fhr_mean": fmt(mean(fhr)),
            "fhr_sd": fmt(sd(fhr)),
            "fhr_p05": fmt(percentile(fhr, 0.05)),
            "fhr_p50": fmt(percentile(fhr, 0.50)),
            "fhr_p95": fmt(percentile(fhr, 0.95)),
            "uc_mean": fmt(mean(uc)),
            "uc_sd": fmt(sd(uc)),
            "uc_p95": fmt(percentile(uc, 0.95)),
            "fhr_uc_corr": fmt(pearson(fhr, uc)),
            "deceleration_proxy_count": deceleration_proxy_count(fhr),
            "cord_ph": metadata.get("ph", ""),
            "bdecf": metadata.get("bdecf", ""),
            "apgar1": metadata.get("apgar1", ""),
            "apgar5": metadata.get("apgar5", ""),
            "gestational_age_weeks": metadata.get("gest_weeks", ""),
            "birth_weight_g": metadata.get("weight_g", ""),
            "sex": metadata.get("sex", ""),
            "maternal_age": metadata.get("age", ""),
            "diabetes": metadata.get("diabetes", ""),
            "hypertension": metadata.get("hypertension", ""),
            "preeclampsia": metadata.get("preeclampsia", ""),
            "meconium": metadata.get("meconium", ""),
            "delivery_type": metadata.get("deliv_type", ""),
            "neonatal_risk": neonatal_risk,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} CTU-UHB record-level feature rows to {out_path}")


if __name__ == "__main__":
    main()

