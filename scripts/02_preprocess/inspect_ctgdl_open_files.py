import argparse
import csv
import tarfile
from pathlib import Path

import pandas as pd


def inspect_csv(path):
    try:
        df = pd.read_csv(path, nrows=5)
        return "|".join(df.columns.astype(str))
    except Exception as exc:
        return f"error: {exc}"


def main():
    parser = argparse.ArgumentParser(description="Inspect downloaded CTGDL open files without assuming schema.")
    parser.add_argument("--indir", default="data/raw/ctgdl/v5_open")
    parser.add_argument("--out", default="results/tables/ctgdl_open_file_inventory.csv")
    parser.add_argument("--max-members", type=int, default=25)
    args = parser.parse_args()

    indir = Path(args.indir)
    rows = []
    for path in sorted(indir.glob("*")):
        if path.name == "download_manifest.csv":
            continue
        row = {
            "file": path.name,
            "size_bytes": path.stat().st_size,
            "type": path.suffix.lower(),
            "member_count": "",
            "sample_members": "",
            "sample_columns": "",
        }
        if path.name.endswith(".tar.gz"):
            with tarfile.open(path, "r:gz") as archive:
                members = [m for m in archive.getmembers() if m.isfile()]
                row["member_count"] = len(members)
                row["sample_members"] = "|".join(m.name for m in members[: args.max_members])
        elif path.name.endswith(".csv"):
            row["sample_columns"] = inspect_csv(path)
        rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CTGDL inventory to {out_path}")


if __name__ == "__main__":
    main()

