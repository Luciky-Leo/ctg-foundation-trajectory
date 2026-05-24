import argparse
import csv
import re
from pathlib import Path


def normalize_key(key):
    key = key.strip().lower()
    key = key.replace(".", "")
    key = key.replace("(", "_").replace(")", "")
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return key.strip("_")


def parse_header(path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    first = lines[0].split()
    record_id = first[0]
    n_signals = first[1] if len(first) > 1 else ""
    sampling_frequency = first[2] if len(first) > 2 else ""
    n_samples = first[3] if len(first) > 3 else ""

    comments = {}
    signal_labels = []
    for line in lines[1:]:
        if line.startswith("#"):
            body = line[1:].strip()
            if not body or body.startswith("--"):
                continue
            if ":" in body:
                key, value = body.split(":", 1)
                comments[normalize_key(key)] = value.strip()
            elif "=" in body:
                key, value = body.split("=", 1)
                comments[normalize_key(key)] = value.strip()
            else:
                match = re.match(r"^(.+?)\s{2,}(.+)$", body)
                if match:
                    key, value = match.groups()
                    comments[normalize_key(key)] = value.strip()
        elif line.strip():
            parts = line.split()
            if parts:
                signal_labels.append(parts[-1])

    row = {
        "record_id": record_id,
        "n_signals": n_signals,
        "sampling_frequency": sampling_frequency,
        "n_samples": n_samples,
        "signal_labels": "|".join(signal_labels),
        "header_file": str(path),
    }
    for key, value in comments.items():
        row[key] = value
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", default="data/raw/ctu_uhb")
    parser.add_argument("--out", default="data/interim/ctu_uhb_header_index.csv")
    args = parser.parse_args()

    paths = sorted(Path(args.indir).glob("*.hea"))
    rows = [parse_header(path) for path in paths]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        print(f"No header files found in {args.indir}")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} CTU-UHB header rows to {out_path}")


if __name__ == "__main__":
    main()
