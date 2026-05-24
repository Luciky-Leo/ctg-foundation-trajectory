import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description="Create a fixed record-level train/test split for CTU-UHB.")
    parser.add_argument("--features", default="data/processed/ctu_uhb_record_features.csv")
    parser.add_argument("--out", default="data/processed/ctu_uhb_record_split.csv")
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    df = pd.read_csv(args.features, dtype={"record_id": str})
    train, test = train_test_split(
        df[["record_id", "neonatal_risk"]],
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["neonatal_risk"].astype(int),
    )
    train = train.copy()
    test = test.copy()
    train["split"] = "train"
    test["split"] = "test"
    out = pd.concat([train, test], axis=0).sort_values("record_id")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote record split to {out_path}: train={len(train)}, test={len(test)}")


if __name__ == "__main__":
    main()

