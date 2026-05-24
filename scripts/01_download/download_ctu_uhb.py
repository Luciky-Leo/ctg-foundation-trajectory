import argparse
import sys
import urllib.request
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


BASE_URL = "https://physionet.org/files/ctu-uhb-ctgdb/1.0.0"


def download(url, out_path, overwrite=False, retries=3):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return "exists"
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            out_path.write_bytes(data)
            return f"downloaded {len(data)} bytes"
        except Exception as exc:
            last_error = exc
            time.sleep(min(8, attempt * 2))
    return f"error: {last_error}"


def main():
    parser = argparse.ArgumentParser(description="Download CTU-UHB / CTU-CHB CTG records from PhysioNet.")
    parser.add_argument("--outdir", default="data/raw/ctu_uhb")
    parser.add_argument("--limit", type=int, default=20, help="Number of records to download. Use 0 for all records.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    records_file = outdir / "RECORDS"
    print(f"Downloading RECORDS from {BASE_URL}/RECORDS")
    download(f"{BASE_URL}/RECORDS", records_file, overwrite=args.overwrite)
    record_ids = [line.strip() for line in records_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit > 0:
        record_ids = record_ids[:args.limit]

    def download_record(record_id):
        hea_status = download(f"{BASE_URL}/{record_id}.hea", outdir / f"{record_id}.hea", overwrite=args.overwrite, retries=args.retries)
        dat_status = download(f"{BASE_URL}/{record_id}.dat", outdir / f"{record_id}.dat", overwrite=args.overwrite, retries=args.retries)
        return record_id, hea_status, dat_status

    manifest = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_record = {executor.submit(download_record, record_id): record_id for record_id in record_ids}
        for idx, future in enumerate(as_completed(future_to_record), start=1):
            try:
                record_id, hea_status, dat_status = future.result()
            except Exception as exc:
                record_id = future_to_record[future]
                hea_status = f"error: {exc}"
                dat_status = f"error: {exc}"
            manifest[record_id] = (hea_status, dat_status)
            print(f"[{idx}/{len(record_ids)}] {record_id}: {hea_status}; {dat_status}")

    manifest_rows = ["record_id,hea_status,dat_status"]
    for record_id in record_ids:
        hea_status, dat_status = manifest[record_id]
        manifest_rows.append(f"{record_id},{hea_status},{dat_status}")

    (outdir / "download_manifest.csv").write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    print(f"Finished CTU-UHB download into {outdir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        sys.exit(1)
