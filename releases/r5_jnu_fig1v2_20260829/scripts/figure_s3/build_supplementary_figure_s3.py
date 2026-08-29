#!/usr/bin/env python3
"""Build Supplementary Figure S3 from frozen JNU/CTU source tables."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


SCRIPT = Path(__file__).resolve()
BATCH_ROOT = SCRIPT.parents[2]
TABLES = BATCH_ROOT / "04_tables"
OUT = BATCH_ROOT / "05_figures"
LOCAL = SCRIPT.parent
SOURCE = LOCAL / "Supplementary_Figure_S3_source_data.csv"
OVERLAP = LOCAL / "Supplementary_Figure_S3_text_overlap_audit.csv"

MM = 1 / 25.4
INK = "#2F3136"
GRID = "#D8D9DE"
BLUE = "#3977B4"
PURPLE = "#735DB1"
ORANGE = "#C9832E"
CORAL = "#D75862"
GRAY = "#74777C"
GREEN = "#4F8360"
ROUTE_COLORS = {
    "random": GRAY,
    "ctu_only": BLUE,
    "jnu_only": PURPLE,
    "jnu_to_ctu": ORANGE,
    "jnu_asymmetric": CORAL,
    "classical_signal": GREEN,
}
ROUTE_LABELS = {
    "random": "Random",
    "ctu_only": "CTU-only",
    "jnu_only": "JNU-only",
    "jnu_to_ctu": "JNU → CTU",
    "jnu_asymmetric": "JNU asym.",
    "classical_signal": "Classical signal",
}


def setup() -> None:
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.45, alpha=0.75)
    ax.set_axisbelow(True)


def panel_label(ax, label: str) -> None:
    ax.text(-0.16, 1.06, label, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="left", va="top", color=INK)


def seed_strip(ax, frame: pd.DataFrame, route_order: list[str], metric: str, x_label: str) -> None:
    offsets = np.linspace(-0.17, 0.17, 5)
    for row_index, route in enumerate(route_order):
        values = frame.loc[frame["route"] == route].sort_values("seed")[metric].to_numpy(float)
        for seed_index, value in enumerate(values):
            ax.scatter(value, row_index + offsets[seed_index], s=18, color=ROUTE_COLORS[route], alpha=0.82, edgecolor="white", linewidth=0.3, zorder=3)
        if len(values):
            ax.scatter(np.median(values), row_index, marker="D", s=31, color=INK, edgecolor="white", linewidth=0.4, zorder=4)
    ax.set_yticks(range(len(route_order)), [ROUTE_LABELS[route] for route in route_order])
    ax.invert_yaxis()
    ax.set_xlabel(x_label)
    clean_axis(ax)


def forest(ax, rows: list[dict[str, object]], x_label: str, limits: tuple[float, float]) -> None:
    ax.axvline(0, color=GRAY, lw=0.7, linestyle="--")
    for index, row in enumerate(rows):
        y = len(rows) - 1 - index
        color = row.get("color", PURPLE)
        ax.errorbar(
            float(row["point"]),
            y,
            xerr=[[float(row["point"]) - float(row["low"])], [float(row["high"]) - float(row["point"])]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.2,
            markersize=4.5,
        )
    ax.set_yticks(range(len(rows)), [str(row["label"]) for row in rows[::-1]])
    ax.set_xlim(*limits)
    ax.set_xlabel(x_label)
    clean_axis(ax)


def text_overlap_audit(fig) -> int:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    overlaps = []
    for axis_index, ax in enumerate(fig.axes):
        texts = [text for text in ax.texts if text.get_text().strip()]
        boxes = [text.get_window_extent(renderer=renderer).expanded(0.98, 0.90) for text in texts]
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if boxes[i].overlaps(boxes[j]):
                    width = max(0, min(boxes[i].x1, boxes[j].x1) - max(boxes[i].x0, boxes[j].x0))
                    height = max(0, min(boxes[i].y1, boxes[j].y1) - max(boxes[i].y0, boxes[j].y0))
                    area = width * height
                    smaller = min(boxes[i].width * boxes[i].height, boxes[j].width * boxes[j].height)
                    if smaller and area / smaller > 0.08:
                        overlaps.append((axis_index, texts[i].get_text(), texts[j].get_text(), area / smaller))
    with OVERLAP.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["axis", "text_1", "text_2", "overlap_fraction"])
        writer.writerows(overlaps)
    return len(overlaps)


def main() -> None:
    setup()
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    risk = pd.read_csv(TABLES / "development_risk_seed_metrics.csv")
    risk = risk[(risk["input_variant"] == "full") & (risk["evaluation_set"] == "development_oof")]
    morphology = pd.read_csv(TABLES / "development_morphology_macro_seed_metrics.csv")
    bootstrap = pd.read_csv(TABLES / "development_paired_hierarchical_bootstrap.csv")
    fixed = pd.read_csv(TABLES / "fixed_test_seed_metrics.csv")
    fixed_full = fixed[(fixed["input_variant"] == "full") & (fixed["evaluation_set"] == "fixed_test_exploratory")]
    dev_all = pd.read_csv(TABLES / "development_risk_seed_metrics.csv")

    source_rows = []
    for panel, frame in [("A", risk), ("B", morphology), ("D", fixed_full)]:
        for row in frame.to_dict("records"):
            source_rows.append({"panel": panel, **row})

    fig, axes = plt.subplots(2, 3, figsize=(180 * MM, 176 * MM))
    plt.subplots_adjust(left=0.105, right=0.985, bottom=0.105, top=0.975, wspace=0.63, hspace=0.62)

    route_order = ["random", "ctu_only", "jnu_only", "jnu_to_ctu", "jnu_asymmetric"]
    seed_strip(axes[0, 0], risk, route_order, "auprc", "Development OOF AUPRC")
    axes[0, 0].axvline(79 / 386, color=GRAY, lw=0.7, linestyle=":")
    panel_label(axes[0, 0], "A")

    morph_order = ["classical_signal", "random", "ctu_only", "jnu_only", "jnu_to_ctu", "jnu_asymmetric"]
    seed_strip(axes[0, 1], morphology, morph_order, "auroc", "Morphology macro-AUROC")
    panel_label(axes[0, 1], "B")

    def boot_row(analysis, metric, candidate, baseline):
        return bootstrap[
            (bootstrap["analysis"] == analysis)
            & (bootstrap["metric"] == metric)
            & (bootstrap["candidate"] == candidate)
            & (bootstrap["baseline"] == baseline)
        ].iloc[0]

    risk_pair = boot_row("risk", "auprc", "jnu_to_ctu", "ctu_only")
    morph_ctu = boot_row("morphology_macro", "auroc", "jnu_to_ctu", "ctu_only")
    morph_random = boot_row("morphology_macro", "auroc", "jnu_to_ctu", "random")
    forest_rows = [
        {"label": "Risk AUPRC\nvs CTU-only", "point": risk_pair["point_mean_across_seeds"], "low": risk_pair["ci_low"], "high": risk_pair["ci_high"], "color": ORANGE},
        {"label": "Morphology AUROC\nvs CTU-only", "point": morph_ctu["point_mean_across_seeds_tasks"], "low": morph_ctu["ci_low"], "high": morph_ctu["ci_high"], "color": PURPLE},
        {"label": "Morphology AUROC\nvs random", "point": morph_random["point_mean_across_seeds_tasks"], "low": morph_random["ci_low"], "high": morph_random["ci_high"], "color": PURPLE},
    ]
    forest(axes[0, 2], forest_rows, "JNU → CTU paired difference", (-0.055, 0.105))
    panel_label(axes[0, 2], "C")
    for row in forest_rows:
        source_rows.append({"panel": "C", **row})

    fixed_order = ["random", "ctu_only", "jnu_only", "jnu_to_ctu"]
    seed_strip(axes[1, 0], fixed_full, fixed_order, "auprc", "Fixed-test AUPRC (exploratory)")
    axes[1, 0].axvline(34 / 166, color=GRAY, lw=0.7, linestyle=":")
    panel_label(axes[1, 0], "D")

    sensitivity_rows = []
    for set_name, frame, label in [
        ("development", dev_all, "Dev"),
        ("fixed", fixed, "Test"),
    ]:
        for route in ["ctu_only", "jnu_to_ctu"]:
            route_frame = frame[frame["route"] == route]
            full_by_seed = route_frame[route_frame["input_variant"] == "full"].set_index("seed")["auprc"]
            route_short = {"ctu_only": "CTU", "jnu_to_ctu": "JNU→CTU"}[route]
            row = {"row": f"{label} · {route_short}"}
            for variant in ["fhr_only", "uc_only", "one_hz"]:
                variant_by_seed = route_frame[route_frame["input_variant"] == variant].set_index("seed")["auprc"]
                row[variant] = float((variant_by_seed - full_by_seed).median())
            sensitivity_rows.append(row)
            source_rows.append({"panel": "E", "set": set_name, "route": route, **row})
    sensitivity = pd.DataFrame(sensitivity_rows).set_index("row")
    matrix = sensitivity[["fhr_only", "uc_only", "one_hz"]].to_numpy()
    limit = max(0.04, float(np.max(np.abs(matrix))))
    image = axes[1, 1].imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axes[1, 1].set_xticks(range(3), ["FHR-only", "UC-only", "1 Hz"], rotation=25, ha="right")
    axes[1, 1].set_yticks(range(len(sensitivity)), sensitivity.index)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes[1, 1].text(column, row, f"{matrix[row, column]:+.3f}", ha="center", va="center", fontsize=5.2, color="white" if abs(matrix[row, column]) > limit * 0.55 else INK)
    axes[1, 1].set_xlabel("Median ΔAUPRC vs full FHR+UC")
    axes[1, 1].tick_params(length=0)
    panel_label(axes[1, 1], "E")

    asym_risk = boot_row("risk", "auprc", "jnu_asymmetric", "jnu_only")
    asym_morph = boot_row("morphology_macro", "auroc", "jnu_asymmetric", "jnu_only")
    asym_rows = [
        {"label": "Risk AUPRC", "point": asym_risk["point_mean_across_seeds"], "low": asym_risk["ci_low"], "high": asym_risk["ci_high"], "color": ORANGE},
        {"label": "Morphology AUROC", "point": asym_morph["point_mean_across_seeds_tasks"], "low": asym_morph["ci_low"], "high": asym_morph["ci_high"], "color": CORAL},
    ]
    forest(axes[1, 2], asym_rows, "Asymmetric minus mixed JNU", (-0.125, 0.075))
    panel_label(axes[1, 2], "F")
    for row in asym_rows:
        source_rows.append({"panel": "F", **row})

    legend = [
        Line2D([0], [0], marker="D", color="none", markerfacecolor=INK, markeredgecolor="white", markersize=5, label="median"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GRAY, markeredgecolor="white", markersize=5, label="individual seed"),
    ]
    axes[0, 0].legend(handles=legend, frameon=False, fontsize=5.2, loc="lower right")

    pd.DataFrame(source_rows).to_csv(SOURCE, index=False)
    overlap_count = text_overlap_audit(fig)
    if overlap_count:
        raise RuntimeError(f"text overlap audit failed: {overlap_count}; see {OVERLAP}")

    stem = OUT / "Supplementary_Figure_S3_JNU_cross_domain_sensitivity"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=300, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
