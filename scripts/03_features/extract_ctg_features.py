import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def safe_float(value):
    if value in ("", None):
        return None
    try:
        return float(value)
    except ValueError:
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


def count_deceleration_proxy(fhr_values):
    clean = [x for x in fhr_values if x is not None]
    if len(clean) < 10:
        return 0
    baseline = sorted(clean)[len(clean) // 2]
    return sum(1 for x in clean if x < baseline - 15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/example/synthetic_ctg_long.csv")
    parser.add_argument("--out", default="data/processed/ctg_feature_table.csv")
    args = parser.parse_args()

    grouped = defaultdict(list)
    with Path(args.input).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            grouped[row["record_id"]].append(row)

    feature_rows = []
    for record_id, rows in sorted(grouped.items()):
        fhr = [safe_float(row.get("fhr")) for row in rows]
        uc = [safe_float(row.get("uc")) for row in rows]
        n = len(rows)
        missing_fhr = sum(1 for x in fhr if x is None)
        outcome = rows[0].get("neonatal_risk", "")
        cord_ph = rows[0].get("cord_ph", "")
        feature_rows.append({
            "record_id": record_id,
            "n_points": n,
            "fhr_missing_fraction": round(missing_fhr / n, 4) if n else "",
            "fhr_mean": round(mean(fhr), 4) if mean(fhr) is not None else "",
            "fhr_sd": round(sd(fhr), 4) if sd(fhr) is not None else "",
            "fhr_min": round(min(x for x in fhr if x is not None), 4),
            "fhr_max": round(max(x for x in fhr if x is not None), 4),
            "uc_mean": round(mean(uc), 4) if mean(uc) is not None else "",
            "uc_sd": round(sd(uc), 4) if sd(uc) is not None else "",
            "fhr_uc_corr": round(pearson(fhr, uc), 4) if pearson(fhr, uc) is not None else "",
            "deceleration_proxy_count": count_deceleration_proxy(fhr),
            "phenotype_true": rows[0].get("phenotype_true", ""),
            "cord_ph": cord_ph,
            "apgar5": rows[0].get("apgar5", ""),
            "gestational_age_weeks": rows[0].get("gestational_age_weeks", ""),
            "birth_weight_g": rows[0].get("birth_weight_g", ""),
            "delivery_mode": rows[0].get("delivery_mode", ""),
            "neonatal_risk": outcome,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0].keys()))
        writer.writeheader()
        writer.writerows(feature_rows)
    print(f"Wrote {len(feature_rows)} feature rows to {out_path}")


if __name__ == "__main__":
    main()

