from __future__ import annotations

# SOURCE_CODE_FIRST: ported from local PERSIST HF001 sparse bubble-matrix
# grammar. Local source snapshot:
# source_code_snapshots/HF001_source_code_snapshot.py

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
REDRAW_ROOT = SCRIPT.parents[1]
OUT_DIR = REDRAW_ROOT / "outputs" / "D"
TABLE_DIR = REDRAW_ROOT / "intermediate_tables"
MM = 1.0 / 25.4

INK = "#2F3136"
GRID = "#D8D9DE"
SUPPORT = "#3977B4"
EXPLORATORY = "#D75862"
CONTEXT = "#C9832E"
ABSENT = "#A23B45"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 6.4,
            "axes.linewidth": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def mapped_roles() -> pd.DataFrame:
    rows = [
        ("Representation learning", "Unlabeled public CTG", "supporting", "JNU + CTU development"),
        ("Representation learning", "CTU development", "supporting", "fold-specific SSL"),
        ("Morphology probe", "CTU expert labels", "supporting", "representation task"),
        ("Risk enrichment", "CTU development", "supporting", "model fitting"),
        ("Risk enrichment", "Fixed test", "exploratory", "historically exposed"),
        ("Domain context", "CTGDL", "context", "representation transport"),
        ("Outcome validation", "Harmonized external", "absent", "not available"),
    ]
    return pd.DataFrame(rows, columns=["analysis_task", "data_role", "status", "boundary"])


def render() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    roles = mapped_roles()
    roles.to_csv(TABLE_DIR / "Figure_1D_roles_input_mapped.tsv", sep="\t", index=False)

    row_order = [
        "Representation learning",
        "Morphology probe",
        "Risk enrichment",
        "Domain context",
        "Outcome validation",
    ]
    col_order = [
        "Unlabeled\nCTG",
        "Expert\nlabels",
        "CTU\ndev.",
        "Fixed\ntest",
        "CTGDL",
        "External\noutcome",
    ]
    key_map = {
        "Unlabeled public CTG": col_order[0],
        "CTU expert labels": col_order[1],
        "CTU development": col_order[2],
        "Fixed test": col_order[3],
        "CTGDL": col_order[4],
        "Harmonized external": col_order[5],
    }

    width_mm, height_mm = 87.0, 68.0
    fig = plt.figure(figsize=(width_mm * MM, height_mm * MM), facecolor="white")
    fig.text(0.01, 0.97, "D", ha="left", va="top", fontsize=8, fontweight="bold", color=INK)
    fig.text(0.075, 0.97, "Analysis roles and claim boundaries", ha="left", va="top", fontsize=7.4, fontweight="bold", color=INK)
    ax = fig.add_axes([0.31, 0.24, 0.66, 0.57])

    n_rows, n_cols = len(row_order), len(col_order)
    for i in range(n_rows):
        for j in range(n_cols):
            ax.add_patch(
                Rectangle(
                    (j - 0.46, i - 0.46),
                    0.92,
                    0.92,
                    facecolor="#F7F8FA" if (i + j) % 2 == 0 else "#FFFFFF",
                    edgecolor=GRID,
                    linewidth=0.45,
                    zorder=0,
                )
            )

    style = {
        "supporting": dict(marker="o", edgecolors=SUPPORT, s=78, linewidth=0.7, facecolors=SUPPORT),
        "exploratory": dict(marker="o", edgecolors=EXPLORATORY, s=86, linewidth=1.25, facecolors="white"),
        "context": dict(marker="D", edgecolors=CONTEXT, s=64, linewidth=0.7, facecolors=CONTEXT),
    }
    for row in roles.itertuples(index=False):
        i = row_order.index(row.analysis_task)
        j = col_order.index(key_map[row.data_role])
        if row.status == "absent":
            ax.scatter(j, i, marker="x", s=84, color=ABSENT, linewidth=1.4, zorder=3)
        else:
            ax.scatter(j, i, zorder=3, **style[row.status])

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xticks(range(n_cols), col_order, fontsize=5.3)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_yticks(range(n_rows), row_order, fontsize=5.9)
    ax.tick_params(axis="y", length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=SUPPORT, markeredgecolor=SUPPORT, markersize=4.5, label="Supporting role"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=EXPLORATORY, markeredgewidth=1.1, markersize=4.8, label="Exploratory test"),
        Line2D([], [], marker="D", linestyle="none", markerfacecolor=CONTEXT, markeredgecolor=CONTEXT, markersize=4.0, label="Representation context"),
        Line2D([], [], marker="x", linestyle="none", color=ABSENT, markersize=4.8, markeredgewidth=1.2, label="Not available"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.055),
        ncol=2,
        frameon=False,
        fontsize=5.5,
        handletextpad=0.35,
        columnspacing=0.9,
    )

    for ext in ("png", "pdf", "svg"):
        path = OUT_DIR / f"Figure_1D_persist.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None)
    plt.close(fig)


if __name__ == "__main__":
    render()
