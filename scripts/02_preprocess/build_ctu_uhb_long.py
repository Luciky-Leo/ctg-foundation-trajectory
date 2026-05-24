import argparse
import csv
import re
import struct
from pathlib import Path


def normalize_key(key):
    key = key.strip().lower().replace(".", "")
    key = key.replace("(", "_").replace(")", "")
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return key.strip("_")


def parse_gain(token):
    # Example tokens:
    # 100(0)/bpm
    # 100/nd
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
            signals.append({
                "file": parts[0],
                "format": parts[1],
                "gain": gain,
                "unit": unit,
                "label": parts[-1],
            })
    return record, signals, metadata


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Build long-format FHR/UC table from downloaded CTU-UHB WFDB-16 records.")
    parser.add_argument("--indir", default="data/raw/ctu_uhb")
    parser.add_argument("--out", default="data/processed/ctu_uhb_subset_long.csv")
    parser.add_argument("--max-records", type=int, default=0, help="Use 0 for all downloaded records.")
    args = parser.parse_args()

    indir = Path(args.indir)
    headers = sorted(indir.glob("*.hea"))
    if args.max_records > 0:
        headers = headers[:args.max_records]
    if not headers:
        raise FileNotFoundError(f"No .hea files found in {indir}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_id",
        "sample_index",
        "time_sec",
        "fhr",
        "uc",
        "cord_ph",
        "apgar1",
        "apgar5",
        "gestational_age_weeks",
        "birth_weight_g",
        "sex",
        "maternal_age",
        "diabetes",
        "hypertension",
        "preeclampsia",
        "meconium",
        "delivery_type",
        "neonatal_risk",
    ]

    total_rows = 0
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for header in headers:
            record, signals, metadata = parse_header(header)
            if record["n_signals"] != 2:
                continue
            dat_path = indir / signals[0]["file"]
            raw = dat_path.read_bytes()
            expected_values = record["n_samples"] * record["n_signals"]
            values = struct.unpack("<" + "h" * expected_values, raw[: expected_values * 2])
            ph = safe_float(metadata.get("ph"))
            apgar5 = safe_float(metadata.get("apgar5"))
            neonatal_risk = int((ph is not None and ph < 7.15) or (apgar5 is not None and apgar5 < 7))
            for i in range(record["n_samples"]):
                fhr_raw = values[i * 2]
                uc_raw = values[i * 2 + 1]
                writer.writerow({
                    "record_id": record["record_id"],
                    "sample_index": i,
                    "time_sec": round(i / record["sampling_frequency"], 3),
                    "fhr": round(fhr_raw / signals[0]["gain"], 4),
                    "uc": round(uc_raw / signals[1]["gain"], 4),
                    "cord_ph": metadata.get("ph", ""),
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
                total_rows += 1
    print(f"Wrote {total_rows} CTU-UHB long-format rows to {out_path}")


if __name__ == "__main__":
    main()

