from __future__ import annotations

# SOURCE_CODE_FIRST: ported from local PERSIST HF114 lollipop/bubble grammar
# and HF132 horizontal stacked-composition grammar. Local source snapshots:
# source_code_snapshots/HF114_source_code_snapshot.py
# source_code_snapshots/HF132_source_code_snapshot.py

import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
REDRAW_ROOT = SCRIPT.parents[1]
REVISION_ROOT = SCRIPT.parents[3]
SOURCE = REVISION_ROOT / "03_analysis" / "Figure_1_final_source_data.csv"
OUT_DIR = REDRAW_ROOT / "outputs" / "C"
TABLE_DIR = REDRAW_ROOT / "intermediate_tables"
MM = 1.0 / 25.4

INK = "#2F3136"
GRID = "#D8D9DE"
JNU = "#7561B2"
CTU = "#3977B4"
CTGDL = "#C9832E"
NO_EVENT = "#AEB3BA"
EVENT = "#D75862"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 6.5,
            "axes.linewidth": 0.55,
            "lines.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def load_source() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = list(csv.DictReader(SOURCE.open(encoding="utf-8")))
    c = {row["item"]: row for row in rows if row["panel"] == "C"}
    counts = pd.DataFrame(
        [
            ("JNU-CTG records", int(c["jnu_records"]["value"].replace(",", "")), "JNU pretraining", JNU),
            ("JNU patient groups", int(c["jnu_groups"]["value"].replace(",", "")), "JNU pretraining", JNU),
            ("CTU-UHB records", int(c["ctu_all"]["value"]), "Outcome cohort", CTU),
            ("CTGDL records", int(c["ctgdl"]["value"]), "Domain context", CTGDL),
        ],
        columns=["label", "count", "role", "color"],
    )
    counts["log10_count"] = np.log10(counts["count"])

    event_rows = []
    for item, label in (("development", "Development"), ("fixed_test", "Fixed test")):
        total, events = (int(value) for value in c[item]["value"].split("/"))
        event_rows.append((label, total, events, total - events, events / total))
    events = pd.DataFrame(
        event_rows,
        columns=["cohort", "total", "events", "no_event", "event_fraction"],
    )
    return counts, events


def render() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    counts, events = load_source()
    counts.to_csv(TABLE_DIR / "Figure_1C_counts_input_mapped.tsv", sep="\t", index=False)
    events.to_csv(TABLE_DIR / "Figure_1C_events_input_mapped.tsv", sep="\t", index=False)

    width_mm, height_mm = 87.0, 68.0
    fig = plt.figure(figsize=(width_mm * MM, height_mm * MM), facecolor="white")
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.30, 0.78],
        left=0.255,
        right=0.96,
        bottom=0.13,
        top=0.80,
        hspace=0.58,
    )
    fig.text(0.01, 0.97, "C", ha="left", va="top", fontsize=8, fontweight="bold", color=INK)
    fig.text(0.075, 0.97, "Public CTG scale and CTU events", ha="left", va="top", fontsize=7.6, fontweight="bold", color=INK)

    ax = fig.add_subplot(gs[0])
    y = np.arange(len(counts))
    start = 2.0
    bubble_sizes = 30 + 58 * (counts["log10_count"] - counts["log10_count"].min()) / (
        counts["log10_count"].max() - counts["log10_count"].min()
    )
    for i, row in counts.iterrows():
        ax.hlines(i, start, row.log10_count, color=row.color, linestyle=(0, (3, 2)), lw=0.9, alpha=0.78, zorder=2)
        ax.scatter(row.log10_count, i, s=bubble_sizes.iloc[i], color=row.color, edgecolor="white", linewidth=0.7, zorder=3)
        ax.text(row.log10_count + 0.075, i, f"{row['count']:,}", ha="left", va="center", fontsize=6.2, fontweight="bold", color=INK)
    ax.set_yticks(y, counts["label"], fontsize=6.1)
    ax.invert_yaxis()
    ax.set_xlim(1.95, 4.62)
    ax.set_xticks([2, 3, 4], ["100", "1,000", "10,000"], fontsize=5.8)
    ax.set_xlabel("Record or patient-group count (log scale)", fontsize=6.0, labelpad=2)
    ax.grid(axis="x", color=GRID, linewidth=0.45, alpha=0.65)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", length=2, width=0.5)

    ax2 = fig.add_subplot(gs[1])
    yy = np.arange(len(events))
    no_frac = events["no_event"] / events["total"]
    ax2.barh(yy, no_frac, color=NO_EVENT, height=0.52, label="No event")
    ax2.barh(yy, events["event_fraction"], left=no_frac, color=EVENT, height=0.52, label="Neonatal risk")
    for i, row in events.iterrows():
        ax2.text(
            1.025,
            i,
            f"{row.events}/{row.total} ({100 * row.event_fraction:.1f}%)",
            ha="left",
            va="center",
            fontsize=6.0,
            fontweight="bold",
            color=INK,
            clip_on=False,
        )
    ax2.set_yticks(yy, events["cohort"], fontsize=6.1)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.32)
    ax2.set_xticks([0, 0.5, 1.0], ["0", "50", "100%"], fontsize=5.8)
    ax2.set_xlabel("Outcome composition", fontsize=6.0, labelpad=2)
    ax2.grid(axis="x", color=GRID, linewidth=0.45, alpha=0.65)
    ax2.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax2.spines[side].set_visible(False)
    ax2.spines["bottom"].set_color(INK)
    ax2.tick_params(axis="y", length=0, pad=3)
    ax2.tick_params(axis="x", length=2, width=0.5)
    ax2.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.04),
        frameon=False,
        ncol=2,
        fontsize=5.8,
        handlelength=1.1,
        columnspacing=1.0,
    )

    for ext in ("png", "pdf", "svg"):
        path = OUT_DIR / f"Figure_1C_persist.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None)
    plt.close(fig)


if __name__ == "__main__":
    render()
