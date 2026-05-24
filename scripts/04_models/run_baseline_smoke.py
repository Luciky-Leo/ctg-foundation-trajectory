import argparse
import csv
import math
from pathlib import Path


FEATURES = [
    "fhr_missing_fraction",
    "fhr_mean",
    "fhr_sd",
    "uc_mean",
    "uc_sd",
    "fhr_uc_corr",
    "deceleration_proxy_count",
]


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def auc_score(y_true, scores):
    pairs = sorted(zip(scores, y_true), key=lambda item: item[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum = 0
    for rank, (_, y) in enumerate(pairs, start=1):
        if y == 1:
            rank_sum += rank
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/ctg_feature_table.csv")
    parser.add_argument("--out", default="results/tables/baseline_smoke_metrics.csv")
    args = parser.parse_args()

    rows = []
    with Path(args.features).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    columns = {name: [safe_float(row.get(name)) for row in rows] for name in FEATURES}
    means = {}
    sds = {}
    for name, values in columns.items():
        clean = [x for x in values if x is not None]
        means[name] = sum(clean) / len(clean)
        sds[name] = math.sqrt(sum((x - means[name]) ** 2 for x in clean) / max(1, len(clean) - 1)) or 1.0

    scores = []
    y_true = []
    for row in rows:
        score = 0.0
        score += 0.7 * ((safe_float(row["fhr_mean"]) - means["fhr_mean"]) / sds["fhr_mean"])
        score += -0.8 * ((safe_float(row["fhr_sd"]) - means["fhr_sd"]) / sds["fhr_sd"])
        score += 0.4 * ((safe_float(row["uc_mean"]) - means["uc_mean"]) / sds["uc_mean"])
        score += 0.9 * ((safe_float(row["deceleration_proxy_count"]) - means["deceleration_proxy_count"]) / sds["deceleration_proxy_count"])
        score += 0.3 * ((safe_float(row["fhr_missing_fraction"]) - means["fhr_missing_fraction"]) / sds["fhr_missing_fraction"])
        scores.append(score)
        y_true.append(int(row["neonatal_risk"]))

    auc = auc_score(y_true, scores)
    threshold = sorted(scores)[len(scores) // 2]
    preds = [1 if score >= threshold else 0 for score in scores]
    tp = sum(1 for y, p in zip(y_true, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(y_true, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(y_true, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, preds) if y == 1 and p == 0)

    metrics = [
        {"metric": "n_records", "value": len(rows)},
        {"metric": "event_rate", "value": round(sum(y_true) / len(y_true), 4)},
        {"metric": "auc_smoke_score", "value": round(auc, 4) if auc is not None else ""},
        {"metric": "sensitivity_at_median_score", "value": round(tp / max(1, tp + fn), 4)},
        {"metric": "specificity_at_median_score", "value": round(tn / max(1, tn + fp), 4)},
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(metrics)
    print(f"Wrote baseline smoke metrics to {out_path}")


if __name__ == "__main__":
    main()

