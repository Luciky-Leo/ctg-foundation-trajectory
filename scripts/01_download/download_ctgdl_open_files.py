import argparse
import csv
import json
import urllib.request
from pathlib import Path


API_URL = "https://zenodo.org/api/records/19510407"
WANTED_FILES = [
    "CTGDL - Data Collection.pdf",
    "CTGDL_FHRMA_proc_csv.tar.gz",
    "CTGDL_FHEMA_metadata.csv",
    "CTGDL_norm_metadata.csv",
    "ctgl_classification_model.pth",
]


def download(url, out_path, overwrite=False):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return "exists"
    request = urllib.request.Request(url, headers={"User-Agent": "CTG_Foundation_Trajectory/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    out_path.write_bytes(data)
    return f"downloaded {len(data)} bytes"


def main():
    parser = argparse.ArgumentParser(description="Download open CTGDL v5 files for external validation planning.")
    parser.add_argument("--outdir", default="data/raw/ctgdl/v5_open")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    rows = []
    request = urllib.request.Request(API_URL, headers={"User-Agent": "CTG_Foundation_Trajectory/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        record = json.loads(response.read().decode("utf-8"))
    available = {item["key"]: item["links"]["self"] for item in record.get("files", [])}
    for filename in WANTED_FILES:
        if filename not in available:
            rows.append({"file": filename, "url": "", "status": "missing_from_zenodo_record"})
            print(f"{filename}: missing_from_zenodo_record")
            continue
        url = available[filename]
        status = download(url, outdir / filename, overwrite=args.overwrite)
        rows.append({"file": filename, "url": url, "status": status})
        print(f"{filename}: {status}")

    manifest = outdir / "download_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "url", "status"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
