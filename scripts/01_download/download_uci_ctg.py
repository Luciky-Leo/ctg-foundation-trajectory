import argparse
import urllib.request
from pathlib import Path


UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00193/CTG.xls"


def main():
    parser = argparse.ArgumentParser(description="Download UCI Cardiotocography feature dataset.")
    parser.add_argument("--outdir", default="data/raw/uci_ctg")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "CTG.xls"
    if out_path.exists() and not args.overwrite:
        print(f"Exists: {out_path}")
        return
    request = urllib.request.Request(
        UCI_URL,
        headers={"User-Agent": "Mozilla/5.0 CTG_Foundation_Trajectory/0.1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    out_path.write_bytes(data)
    print(f"Wrote {len(data)} bytes to {out_path}")


if __name__ == "__main__":
    main()
