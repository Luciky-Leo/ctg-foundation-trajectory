import argparse
import csv
import math
import random
from pathlib import Path


def make_trace(record_id, phenotype, n_points, rng):
    base = {
        "stable": 140,
        "reduced_variability": 145,
        "late_deceleration": 150,
        "tachy_contraction": 155,
    }[phenotype]
    variability = {
        "stable": 7.0,
        "reduced_variability": 2.5,
        "late_deceleration": 5.0,
        "tachy_contraction": 6.0,
    }[phenotype]
    pH = {
        "stable": rng.normalvariate(7.25, 0.04),
        "reduced_variability": rng.normalvariate(7.18, 0.06),
        "late_deceleration": rng.normalvariate(7.10, 0.07),
        "tachy_contraction": rng.normalvariate(7.14, 0.06),
    }[phenotype]
    apgar5 = max(1, min(10, int(round(9 - max(0, 7.18 - pH) * 20 + rng.normalvariate(0, 0.8)))))
    gest_age = rng.normalvariate(39.5, 1.2)
    birth_weight = rng.normalvariate(3350, 420)
    delivery_mode = "cesarean" if rng.random() < (0.08 if phenotype == "stable" else 0.18) else "vaginal"

    rows = []
    contraction_period = rng.randint(120, 240)
    for t in range(n_points):
        minutes = t / 60
        uc_wave = max(0.0, math.sin((2 * math.pi * t) / contraction_period))
        uc = 8 + 55 * (uc_wave ** 4) + rng.normalvariate(0, 3)
        fhr = base + variability * math.sin(2 * math.pi * t / rng.randint(70, 120)) + rng.normalvariate(0, variability)

        if phenotype in ("late_deceleration", "tachy_contraction") and uc_wave > 0.7:
            lag = 20 if phenotype == "late_deceleration" else 5
            delayed_wave = max(0.0, math.sin((2 * math.pi * max(t - lag, 0)) / contraction_period))
            if delayed_wave > 0.7:
                fhr -= 18 * (delayed_wave ** 2)

        if phenotype == "reduced_variability":
            fhr = base + (fhr - base) * 0.45

        if rng.random() < 0.015:
            fhr_value = ""
        else:
            fhr_value = round(max(60, min(210, fhr)), 2)

        rows.append({
            "record_id": record_id,
            "time_sec": t,
            "fhr": fhr_value,
            "uc": round(max(0, min(100, uc)), 2),
            "phenotype_true": phenotype,
            "cord_ph": round(pH, 3),
            "apgar5": apgar5,
            "gestational_age_weeks": round(gest_age, 2),
            "birth_weight_g": round(birth_weight, 1),
            "delivery_mode": delivery_mode,
            "neonatal_risk": int(pH < 7.15 or apgar5 < 7),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/example/synthetic_ctg_long.csv")
    parser.add_argument("--records", type=int, default=80)
    parser.add_argument("--points", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    phenotypes = ["stable", "reduced_variability", "late_deceleration", "tachy_contraction"]
    rows = []
    for i in range(args.records):
        phenotype = phenotypes[i % len(phenotypes)]
        rows.extend(make_trace(f"SYN{i + 1:04d}", phenotype, args.points, rng))

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

