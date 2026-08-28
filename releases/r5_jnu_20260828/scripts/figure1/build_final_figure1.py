"""Rebuild R5 Figure 1 using the approved R1 visual grammar.

The R1 figure is used only as a layout and illustration reference. All labels,
model dimensions, cohort roles, and evidence boundaries below are bound to the
R5 revision-locked analysis route.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REVISION_ROOT = PROJECT_ROOT / "R2_R5_JNU_Enhancement_20260827"
OUT_DIR = REVISION_ROOT / "05_figures"
REDRAW_DIR = Path(__file__).resolve().parent
SOURCE_DATA = REVISION_ROOT / "03_analysis" / "Figure_1_final_source_data.csv"
LOCAL_SOURCE_DATA = REDRAW_DIR / "Figure_1_final_source_data.csv"
OVERLAP_AUDIT = REDRAW_DIR / "Figure_1_text_overlap_audit.csv"

MM_TO_INCH = 1.0 / 25.4
FIGURE_WIDTH_MM = 180.0
FIGURE_HEIGHT_MM = 240.0

INK = "#2F3136"
GRID = "#D8D9DE"
WHITE = "#FFFFFF"
BLUE = "#9BBBE1"
BLUE_DARK = "#3977B4"
NEUTRAL = "#A4A5A9"
NEUTRAL_DARK = "#666A70"
LAVENDER = "#B7B7EB"
LAVENDER_DARK = "#735DB1"
OCHRE = "#EAB883"
OCHRE_DARK = "#C9832E"
CORAL = "#F09BA0"
CORAL_DARK = "#D75862"
GREEN = "#A8C9B2"
GREEN_DARK = "#4F8360"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.5,
            "font.weight": "normal",
            "axes.linewidth": 0.6,
            "lines.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": WHITE,
        }
    )


def arrow(ax, start, end, *, color=INK, lw=0.75, rad=0.0, dashed=False) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "lw": lw,
            "color": color,
            "shrinkA": 3,
            "shrinkB": 3,
            "linestyle": (0, (3, 2)) if dashed else "-",
            "connectionstyle": f"arc3,rad={rad}",
        },
    )


def panel_heading(ax, label: str, title: str, subtitle: str | None = None) -> None:
    ax.text(0.005, 0.985, label, ha="left", va="top", fontsize=8, fontweight="bold", color=INK)
    ax.text(0.042, 0.985, title, ha="left", va="top", fontsize=8.4, fontweight="bold", color=INK)
    if subtitle:
        ax.text(0.042, 0.925, subtitle, ha="left", va="top", fontsize=5.8, color=NEUTRAL_DARK)


def signal_trace(ax, x0, y0, width, height, color, *, uc=False, phase=0.0, lw=0.65) -> None:
    t = np.linspace(0, 1, 120)
    if uc:
        y = 0.18 + 0.54 * np.sin(2 * np.pi * (2.65 * t + phase)) ** 8
        y += 0.025 * np.sin(2 * np.pi * 13 * t)
    else:
        y = 0.50 + 0.13 * np.sin(2 * np.pi * (2.7 * t + phase))
        y += 0.055 * np.sin(2 * np.pi * (12.0 * t + phase))
        y += 0.020 * np.sin(2 * np.pi * 27.0 * t)
    ax.plot(x0 + width * t, y0 + height * y, color=color, lw=lw, solid_capstyle="round")


def card(ax, x, y, width, height, index, title, lines, color, icon_fn) -> tuple[float, float, float, float]:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.006,rounding_size=0.014",
            facecolor=WHITE,
            edgecolor=color,
            linewidth=0.75,
        )
    )
    header_h = 0.055
    ax.add_patch(Rectangle((x + 0.001, y + height - header_h), width - 0.002, header_h - 0.001, facecolor=color, edgecolor="none", alpha=0.52))
    ax.add_patch(plt.Circle((x + 0.026, y + height - header_h / 2), 0.014, facecolor=WHITE, edgecolor=color, lw=0.65))
    ax.text(x + 0.026, y + height - header_h / 2, str(index), ha="center", va="center", fontsize=5.6, fontweight="bold", color=INK)
    ax.text(x + 0.049, y + height - header_h / 2, title, ha="left", va="center", fontsize=5.6, fontweight="bold", color=INK)
    icon_fn(ax, x + 0.012, y + 0.070, width - 0.024, height - 0.142, color)
    ax.text(x + width / 2, y + 0.034, "\n".join(lines), ha="center", va="center", fontsize=4.85, color=INK, linespacing=1.02)
    return x, y, width, height


def icon_records(ax, x, y, width, height, color) -> None:
    for offset in (0.018, 0.009, 0.0):
        ax.add_patch(Rectangle((x + 0.018 + offset, y + 0.028 + offset), width * 0.60, height * 0.62, facecolor=WHITE, edgecolor=INK, lw=0.35))
    signal_trace(ax, x + 0.035, y + height * 0.53, width * 0.46, height * 0.20, BLUE_DARK, phase=0.1, lw=0.45)
    signal_trace(ax, x + 0.035, y + height * 0.28, width * 0.46, height * 0.19, OCHRE_DARK, uc=True, phase=0.15, lw=0.45)
    ax.add_patch(Rectangle((x + width * 0.72, y + height * 0.12), width * 0.17, height * 0.31, facecolor=color, edgecolor=INK, lw=0.35, alpha=0.42))


def icon_split(ax, x, y, width, height, color) -> None:
    for row, yy in enumerate((0.68, 0.43, 0.18)):
        fill = NEUTRAL_DARK if row < 2 else WHITE
        edge = NEUTRAL_DARK
        for col in range(3):
            ax.add_patch(plt.Circle((x + width * (0.20 + 0.17 * col), y + height * yy), width * 0.055, facecolor=fill, edgecolor=edge, lw=0.35))
        lock_x = x + width * 0.73
        ax.add_patch(FancyBboxPatch((lock_x, y + height * yy - 0.04), width * 0.16, height * 0.23, boxstyle="round,pad=0.002,rounding_size=0.004", facecolor=WHITE, edgecolor=color, lw=0.55))
        ax.plot([lock_x + width * 0.04, lock_x + width * 0.04, lock_x + width * 0.12, lock_x + width * 0.12], [y + height * yy + 0.03, y + height * yy + 0.10, y + height * yy + 0.10, y + height * yy + 0.03], color=color, lw=0.45)


def icon_cv(ax, x, y, width, height, color) -> None:
    fold_y = [0.70, 0.45, 0.20]
    for row, yy in enumerate(fold_y):
        for col in range(3):
            fill = GREEN if col != row else WHITE
            ax.add_patch(FancyBboxPatch((x + width * (0.12 + 0.20 * col), y + height * yy), width * 0.14, height * 0.15, boxstyle="round,pad=0.001,rounding_size=0.003", facecolor=fill, edgecolor=GREEN_DARK, lw=0.35, alpha=0.9))
        ax.text(x + width * 0.83, y + height * (yy + 0.07), f"F{row + 1}", fontsize=4.3, va="center", ha="center", color=GREEN_DARK, fontweight="bold")
    ax.plot([x + width * 0.08, x + width * 0.90], [y + height * 0.12, y + height * 0.12], color=GREEN_DARK, lw=0.5)


def icon_seeds(ax, x, y, width, height, color) -> None:
    for layer in range(3):
        xx = x + width * (0.16 + layer * 0.20)
        ax.add_patch(FancyBboxPatch((xx, y + height * 0.22), width * 0.14, height * 0.56, boxstyle="round,pad=0.002,rounding_size=0.004", facecolor=LAVENDER, edgecolor=LAVENDER_DARK, lw=0.40, alpha=0.55))
    for i in range(5):
        ax.add_patch(plt.Circle((x + width * (0.12 + i * 0.17), y + height * 0.08), width * 0.035, facecolor=LAVENDER_DARK, edgecolor=WHITE, lw=0.25, alpha=0.85))


def icon_model(ax, x, y, width, height, color) -> None:
    for i in range(5):
        ax.add_patch(plt.Circle((x + width * (0.15 + i * 0.13), y + height * 0.68), width * 0.030, facecolor=LAVENDER_DARK, edgecolor="none", alpha=0.85))
    for i in range(5):
        ax.add_patch(Rectangle((x + width * (0.10 + i * 0.14), y + height * 0.24), width * 0.065, height * (0.12 + 0.05 * i), facecolor=OCHRE, edgecolor=OCHRE_DARK, lw=0.25, alpha=0.9))
    ax.plot([x + width * 0.08, x + width * 0.82], [y + height * 0.18, y + height * 0.18], color=INK, lw=0.40)


def icon_test(ax, x, y, width, height, color) -> None:
    for row in range(3):
        x0 = x + width * 0.12
        y0 = y + height * (0.18 + row * 0.24)
        ax.plot([x0, x0, x0 + width * 0.42], [y0, y0 + height * 0.18, y0 + height * 0.18], color=INK, lw=0.35)
        ax.plot([x0 + width * 0.03, x0 + width * 0.15, x0 + width * 0.28, x0 + width * 0.40], [y0 + height * 0.04, y0 + height * (0.10 + row * 0.015), y0 + height * 0.13, y0 + height * (0.17 - row * 0.012)], color=CORAL_DARK, lw=0.55)
    ax.add_patch(Rectangle((x + width * 0.68, y + height * 0.22), width * 0.18, height * 0.50, facecolor=WHITE, edgecolor=INK, lw=0.40))
    for row in range(3):
        yy = y + height * (0.62 - row * 0.14)
        ax.plot([x + width * 0.71, x + width * 0.75, x + width * 0.81], [yy, yy - height * 0.05, yy + height * 0.03], color=CORAL_DARK, lw=0.45)


def draw_panel_a(ax) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel_heading(
        ax,
        "A",
        "Leakage-aware CTU evidence chain with bounded public pretraining sensitivity",
        "CTU roles are locked before window generation; JNU labels are excluded from SSL and the fixed-test exposure is disclosed.",
    )
    y, height, width = 0.49, 0.34, 0.143
    xs = [0.015, 0.181, 0.347, 0.513, 0.679, 0.845]
    cards = [
        card(ax, xs[0], y, width, height, 1, "Primary cohort", ["CTU-UHB", "552 labelled records"], BLUE, icon_records),
        card(ax, xs[1], y, width, height, 2, "Locked roles", ["386 development", "166 fixed test"], NEUTRAL, icon_split),
        card(ax, xs[2], y, width, height, 3, "Development", ["3-fold validation", "fold-specific SSL"], GREEN, icon_cv),
        card(ax, xs[3], y, width, height, 4, "Frozen route", ["p80-d48-l2", "5 fixed seeds"], LAVENDER, icon_seeds),
        card(ax, xs[4], y, width, height, 5, "Risk model", ["10 signal features", "+ 5 SSL PCs"], OCHRE, icon_model),
        card(ax, xs[5], y, width - 0.004, height, 6, "Fixed test", ["historically exposed", "exploratory only"], CORAL, icon_test),
    ]
    for left, right in zip(cards[:-1], cards[1:]):
        arrow(ax, (left[0] + left[2], y + height / 2), (right[0], y + height / 2))

    jnu_x, jnu_y, jnu_w, jnu_h = 0.055, 0.075, 0.425, 0.245
    ax.add_patch(FancyBboxPatch((jnu_x, jnu_y), jnu_w, jnu_h, boxstyle="round,pad=0.008,rounding_size=0.016", facecolor=WHITE, edgecolor=LAVENDER_DARK, lw=0.70))
    ax.add_patch(Rectangle((jnu_x + 0.001, jnu_y + jnu_h - 0.052), jnu_w - 0.002, 0.051, facecolor=LAVENDER, edgecolor="none", alpha=0.48))
    ax.text(jnu_x + 0.024, jnu_y + jnu_h - 0.026, "Cross-domain pretraining sensitivity", ha="left", va="center", fontsize=5.9, fontweight="bold", color=INK)
    icon_records(ax, jnu_x + 0.018, jnu_y + 0.032, 0.100, 0.132, LAVENDER)
    ax.text(jnu_x + 0.135, jnu_y + 0.148, "JNU-CTG  |  20,769 records  |  12,606 groups", ha="left", va="center", fontsize=5.45, fontweight="bold", color=INK)
    ax.text(jnu_x + 0.135, jnu_y + 0.094, "62,307 non-overlapping 10-min windows\nFHR + UC only; all clinical labels withheld", ha="left", va="center", fontsize=5.1, color=INK, linespacing=1.03)
    ax.text(jnu_x + 0.135, jnu_y + 0.035, "Sensitivity branch; not primary-model selection", ha="left", va="center", fontsize=5.15, fontweight="bold", color=CORAL_DARK)
    arrow(ax, (jnu_x + jnu_w * 0.88, jnu_y + jnu_h), (xs[3] + width * 0.48, y), color=LAVENDER_DARK, lw=0.65, rad=0.10, dashed=True)

    ext_x, ext_y, ext_w, ext_h = 0.505, 0.075, 0.215, 0.245
    ax.add_patch(FancyBboxPatch((ext_x, ext_y), ext_w, ext_h, boxstyle="round,pad=0.008,rounding_size=0.016", facecolor=WHITE, edgecolor=OCHRE_DARK, lw=0.70))
    ax.text(ext_x + 0.018, ext_y + ext_h - 0.040, "CTGDL domain context", ha="left", va="center", fontsize=5.8, fontweight="bold", color=INK)
    ax.text(ext_x + 0.018, ext_y + 0.145, "135 public records", ha="left", va="center", fontsize=5.45, fontweight="bold", color=INK)
    ax.text(ext_x + 0.018, ext_y + 0.092, "Frozen representation\ntransport only", ha="left", va="center", fontsize=5.1, color=INK, linespacing=1.03)
    ax.text(ext_x + 0.018, ext_y + 0.035, "No outcome validation", ha="left", va="center", fontsize=5.15, fontweight="bold", color=CORAL_DARK)
    arrow(ax, (xs[3] + width * 0.65, y), (ext_x + ext_w * 0.55, ext_y + ext_h), color=OCHRE_DARK, lw=0.65, rad=0.08, dashed=True)

    bound_x, bound_y, bound_w, bound_h = 0.745, 0.075, 0.230, 0.245
    ax.add_patch(FancyBboxPatch((bound_x, bound_y), bound_w, bound_h, boxstyle="round,pad=0.008,rounding_size=0.016", facecolor="#FFF9F9", edgecolor=CORAL_DARK, lw=0.70))
    ax.text(bound_x + 0.018, bound_y + bound_h - 0.040, "Interpretation boundary", ha="left", va="center", fontsize=5.8, fontweight="bold", color=INK)
    ax.text(bound_x + 0.018, bound_y + 0.150, "Historical test exposure may\nretain selection optimism.", ha="left", va="center", fontsize=5.05, color=INK, linespacing=1.03)
    ax.text(bound_x + 0.018, bound_y + 0.090, "Retrospective delivery-level\nrisk enrichment only.", ha="left", va="center", fontsize=5.05, color=INK, linespacing=1.03)
    ax.text(bound_x + 0.018, bound_y + 0.032, "No lead-time or deployment claim", ha="left", va="center", fontsize=5.10, fontweight="bold", color=CORAL_DARK)


def draw_token_row(ax, x, y, width, height, color, masked: set[int] | None = None, *, uc=False) -> None:
    masked = masked or set()
    n = 7
    gap = width * 0.025
    token_w = (width - gap * (n - 1)) / n
    for i in range(n):
        left = x + i * (token_w + gap)
        is_masked = i in masked
        face = NEUTRAL if is_masked else WHITE
        edge = NEUTRAL_DARK if is_masked else color
        ax.add_patch(FancyBboxPatch((left, y), token_w, height, boxstyle="round,pad=0.001,rounding_size=0.004", facecolor=face, edgecolor=edge, lw=0.45))
        if not is_masked:
            signal_trace(ax, left + token_w * 0.10, y + height * 0.10, token_w * 0.80, height * 0.75, color, uc=uc, phase=i * 0.07, lw=0.34)


def draw_panel_b(ax) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel_heading(
        ax,
        "B",
        "Frozen PatchTST-style encoder and prespecified representation comparisons",
        "Architecture and downstream dimensionality are unchanged; JNU pretraining is an additional sensitivity route.",
    )

    ax.text(0.028, 0.725, "FHR", ha="left", va="center", fontsize=5.8, fontweight="bold", color=BLUE_DARK)
    ax.text(0.028, 0.555, "UC", ha="left", va="center", fontsize=5.8, fontweight="bold", color=OCHRE_DARK)
    ax.add_patch(FancyBboxPatch((0.075, 0.655), 0.190, 0.115, boxstyle="round,pad=0.004,rounding_size=0.010", facecolor=WHITE, edgecolor=BLUE_DARK, lw=0.55, linestyle=(0, (3, 2))))
    ax.add_patch(FancyBboxPatch((0.075, 0.485), 0.190, 0.115, boxstyle="round,pad=0.004,rounding_size=0.010", facecolor=WHITE, edgecolor=OCHRE_DARK, lw=0.55, linestyle=(0, (3, 2))))
    signal_trace(ax, 0.086, 0.661, 0.168, 0.100, BLUE_DARK, lw=0.65)
    signal_trace(ax, 0.086, 0.491, 0.168, 0.100, OCHRE_DARK, uc=True, lw=0.65)
    ax.text(0.095, 0.435, "10-min window  |  4 Hz  |  2,400 points/channel", ha="left", va="center", fontsize=5.15, color=INK)

    draw_token_row(ax, 0.315, 0.650, 0.155, 0.078, BLUE_DARK)
    draw_token_row(ax, 0.315, 0.505, 0.155, 0.078, OCHRE_DARK, uc=True)
    ax.text(0.315, 0.435, "Patch length 80\n30 patches/channel  |  60 total tokens", ha="left", va="center", fontsize=5.05, color=INK, linespacing=1.02)

    draw_token_row(ax, 0.520, 0.650, 0.155, 0.078, BLUE_DARK, {1, 4})
    draw_token_row(ax, 0.520, 0.505, 0.155, 0.078, OCHRE_DARK, {2, 3}, uc=True)
    ax.text(0.520, 0.435, "Mixed masking  |  25% target", ha="left", va="center", fontsize=5.15, fontweight="bold", color=INK)
    ax.text(0.520, 0.386, "FHR-asymmetric masking: development-only ablation", ha="left", va="center", fontsize=4.65, color=CORAL_DARK)

    enc_x, enc_y, enc_w, enc_h = 0.725, 0.505, 0.135, 0.245
    ax.add_patch(FancyBboxPatch((enc_x, enc_y), enc_w, enc_h, boxstyle="round,pad=0.006,rounding_size=0.012", facecolor=LAVENDER, edgecolor=LAVENDER_DARK, lw=0.65, alpha=0.47))
    for layer in range(2):
        xx = enc_x + 0.021 + layer * 0.032
        ax.add_patch(FancyBboxPatch((xx, enc_y + 0.095), 0.020, 0.105, boxstyle="round,pad=0.002,rounding_size=0.004", facecolor=WHITE, edgecolor=LAVENDER_DARK, lw=0.42))
    ax.text(enc_x + enc_w / 2, enc_y + 0.058, "Transformer\n2 layers  |  4 heads\n$d_{model}=48$", ha="center", va="center", fontsize=5.0, fontweight="bold", color=INK, linespacing=1.0)

    emb_x, emb_y, emb_w, emb_h = 0.900, 0.535, 0.075, 0.185
    ax.add_patch(FancyBboxPatch((emb_x, emb_y), emb_w, emb_h, boxstyle="round,pad=0.005,rounding_size=0.010", facecolor=WHITE, edgecolor=LAVENDER_DARK, lw=0.60))
    for i in range(4):
        ax.add_patch(plt.Circle((emb_x + emb_w / 2, emb_y + 0.033 + i * 0.025), 0.006, facecolor=LAVENDER_DARK, edgecolor="none", alpha=0.86))
    ax.text(emb_x + emb_w / 2, emb_y + emb_h - 0.025, "48-D record", ha="center", va="center", fontsize=4.15, fontweight="bold", color=INK)

    arrow(ax, (0.268, 0.625), (0.311, 0.625))
    arrow(ax, (0.473, 0.625), (0.516, 0.625))
    arrow(ax, (0.678, 0.625), (enc_x, 0.625))
    arrow(ax, (enc_x + enc_w, 0.625), (emb_x, 0.625))

    recon_x, recon_y, recon_w, recon_h = 0.395, 0.120, 0.215, 0.190
    contrast_x, contrast_y, contrast_w, contrast_h = 0.650, 0.120, 0.215, 0.190
    for x, y, width, height, edge, title, weight in [
        (recon_x, recon_y, recon_w, recon_h, CORAL_DARK, "Masked reconstruction", "weight 1.0"),
        (contrast_x, contrast_y, contrast_w, contrast_h, OCHRE_DARK, "NT-Xent consistency", "weight 0.1"),
    ]:
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.006,rounding_size=0.012", facecolor=WHITE, edgecolor=edge, lw=0.60, linestyle=(0, (3, 2))))
        ax.text(x + width / 2, y + height - 0.035, title, ha="center", va="center", fontsize=5.25, fontweight="bold", color=INK)
        ax.text(x + width / 2, y + 0.030, weight, ha="center", va="center", fontsize=4.85, color=INK)

    for i in range(6):
        xx = recon_x + 0.025 + i * 0.026
        ax.add_patch(Rectangle((xx, recon_y + 0.072), 0.014, 0.026, facecolor=CORAL if i in {1, 4} else WHITE, edgecolor=CORAL_DARK, lw=0.35))
    for i in range(4):
        y1 = contrast_y + 0.072 + i * 0.020
        ax.add_patch(plt.Circle((contrast_x + 0.045, y1), 0.005, facecolor=LAVENDER_DARK, edgecolor="none"))
        ax.add_patch(plt.Circle((contrast_x + 0.165, y1), 0.005, facecolor=LAVENDER_DARK, edgecolor="none"))
        ax.plot([contrast_x + 0.052, contrast_x + 0.158], [y1, y1], color=OCHRE_DARK, lw=0.35, alpha=0.65)

    arrow(ax, (enc_x + 0.040, enc_y), (recon_x + recon_w * 0.70, recon_y + recon_h), color=CORAL_DARK, rad=0.08)
    arrow(ax, (emb_x + emb_w * 0.50, emb_y), (contrast_x + contrast_w * 0.64, contrast_y + contrast_h), color=OCHRE_DARK, rad=-0.10)

    route_x, route_y, route_w, route_h = 0.025, 0.105, 0.315, 0.205
    ax.add_patch(FancyBboxPatch((route_x, route_y), route_w, route_h, boxstyle="round,pad=0.006,rounding_size=0.012", facecolor=WHITE, edgecolor=GRID, lw=0.60))
    ax.text(route_x + 0.018, route_y + route_h - 0.034, "Frozen encoder origin comparison", ha="left", va="center", fontsize=5.25, fontweight="bold", color=INK)
    routes = [
        ("Random initialization", NEUTRAL_DARK),
        ("CTU development-only SSL", BLUE_DARK),
        ("JNU-only SSL", LAVENDER_DARK),
        ("JNU pretraining → CTU adaptation", OCHRE_DARK),
    ]
    for index, (label, color) in enumerate(routes):
        yy = route_y + route_h - 0.072 - index * 0.035
        ax.add_patch(plt.Circle((route_x + 0.024, yy), 0.006, facecolor=color, edgecolor="none"))
        ax.text(route_x + 0.040, yy, label, ha="left", va="center", fontsize=4.8, color=INK)


def draw_panel_c(ax) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel_heading(ax, "C", "Public CTG scale and CTU outcome balance")
    ax.text(0.075, 0.825, "Record/group scale", ha="left", va="center", fontsize=5.8, fontweight="bold", color=INK)
    ax.text(0.925, 0.825, "bar length: log10(count)", ha="right", va="center", fontsize=4.35, color=NEUTRAL_DARK)
    ax.plot([0.075, 0.925], [0.785, 0.785], color=GRID, lw=0.50)
    counts = [
        ("JNU records", 20769, LAVENDER),
        ("JNU patient groups", 12606, LAVENDER),
        ("CTU-UHB records", 552, BLUE),
        ("CTGDL records", 135, OCHRE),
    ]
    max_log = np.log10(max(value for _, value, _ in counts))
    for i, (label, value, color) in enumerate(counts):
        y = 0.690 - i * 0.105
        ax.text(0.075, y, label, ha="left", va="center", fontsize=4.9, color=INK)
        ax.add_patch(Rectangle((0.350, y - 0.018), 0.455, 0.036, facecolor=WHITE, edgecolor=GRID, lw=0.35))
        ax.add_patch(Rectangle((0.350, y - 0.018), 0.455 * np.log10(value) / max_log, 0.036, facecolor=color, edgecolor="none", alpha=0.96))
        ax.text(0.825, y, f"{value:,}", ha="left", va="center", fontsize=5.1, fontweight="bold", color=INK)

    ax.text(0.075, 0.255, "CTU neonatal-risk prevalence", ha="left", va="center", fontsize=5.6, fontweight="bold", color=INK)
    for y, label, total, events in [(0.150, "Development", 386, 79), (0.055, "Fixed test", 166, 34)]:
        bar_x, bar_w, bar_h = 0.350, 0.455, 0.050
        no_events = total - events
        no_w = bar_w * no_events / total
        ev_w = bar_w * events / total
        ax.text(0.075, y + bar_h / 2, f"{label}  n={total}", ha="left", va="center", fontsize=4.8, color=INK)
        ax.add_patch(FancyBboxPatch((bar_x, y), bar_w, bar_h, boxstyle="round,pad=0.002,rounding_size=0.007", facecolor=WHITE, edgecolor=GRID, lw=0.35))
        ax.add_patch(Rectangle((bar_x, y), no_w, bar_h, facecolor=NEUTRAL, edgecolor="none", alpha=0.96))
        ax.add_patch(Rectangle((bar_x + no_w, y), ev_w, bar_h, facecolor=CORAL, edgecolor="none", alpha=0.98))
        ax.text(0.825, y + bar_h / 2, f"{events}  ({events / total:.1%})", ha="left", va="center", fontsize=4.8, fontweight="bold", color=INK)

def draw_panel_d(ax) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel_heading(ax, "D", "Analysis roles and claim boundaries")
    rows = [
        ("Representation learning", "JNU + CTU development", "labels withheld", LAVENDER_DARK),
        ("Morphology probe", "CTU expert events", "representation task", BLUE_DARK),
        ("Neonatal-risk enrichment", "CTU development / fixed test", "exploratory", OCHRE_DARK),
        ("External outcome validation", "No harmonized outcome cohort", "not performed", CORAL_DARK),
    ]
    ax.text(0.055, 0.805, "Question", fontsize=4.8, fontweight="bold", color=NEUTRAL_DARK, va="center")
    ax.text(0.405, 0.805, "Data role", fontsize=4.8, fontweight="bold", color=NEUTRAL_DARK, va="center")
    ax.text(0.795, 0.805, "Allowed claim", fontsize=4.8, fontweight="bold", color=NEUTRAL_DARK, va="center")
    ax.plot([0.055, 0.945], [0.770, 0.770], color=GRID, lw=0.5)
    for index, (question, data_role, allowed, color) in enumerate(rows):
        y = 0.665 - index * 0.145
        ax.add_patch(plt.Circle((0.070, y), 0.010, facecolor=color, edgecolor="none"))
        ax.text(0.095, y, question, ha="left", va="center", fontsize=4.8, fontweight="bold", color=INK)
        ax.text(0.405, y, data_role, ha="left", va="center", fontsize=4.7, color=INK)
        face = "#F2F7F4" if allowed not in {"exploratory", "not performed"} else "#FFF5F3"
        edge = GREEN_DARK if allowed not in {"exploratory", "not performed"} else CORAL_DARK
        ax.add_patch(FancyBboxPatch((0.765, y - 0.035), 0.175, 0.070, boxstyle="round,pad=0.003,rounding_size=0.010", facecolor=face, edgecolor=edge, lw=0.45))
        ax.text(0.852, y, allowed, ha="center", va="center", fontsize=4.15, fontweight="bold", color=edge)
        if index < len(rows) - 1:
            ax.plot([0.055, 0.945], [y - 0.073, y - 0.073], color=GRID, lw=0.35)

    ax.add_patch(FancyBboxPatch((0.055, 0.035), 0.890, 0.105, boxstyle="round,pad=0.005,rounding_size=0.012", facecolor="#FFF9F9", edgecolor=CORAL_DARK, lw=0.55))
    ax.text(0.500, 0.087, "JNU sensitivity did not qualify for a primary risk-improvement claim; detailed results are supplementary.", ha="center", va="center", fontsize=4.75, fontweight="bold", color=CORAL_DARK)


def write_source_data() -> None:
    rows = [
        ("A", "primary_cohort", "CTU-UHB", "552 labelled records", "primary outcome cohort"),
        ("A", "fixed_split", "development/test", "386/166 records", "fixed before R5 reanalysis"),
        ("A", "architecture_selection", "3-fold development-only validation", "fold-specific SSL pretraining", "test not used for R5 configuration selection"),
        ("A", "seed_refit", "frozen p80-d48-l2 encoder", "5 fixed seeds", "architecture and downstream dimensions unchanged"),
        ("A", "primary_model", "10 signal features + 5 SSL PCs", "15 predictors", "PCA fitted in development records"),
        ("A", "test_boundary", "fixed test", "exploratory risk enrichment", "historical exposure may cause selection optimism"),
        ("A", "jnu_branch", "JNU-CTG", "20,769 records; 12,606 patient groups; 62,307 windows", "labels excluded; sensitivity branch; not primary-model selection"),
        ("A", "external_branch", "CTGDL-FHRMA", "135 records", "representation/domain context only; no outcome validation"),
        ("A", "prediction_horizon", "delivery-level analysis", "no prespecified lead-time", "not a deployment claim"),
        ("B", "input", "FHR + UC", "10 min at 4 Hz; 2400 points/channel", "SSL labels withheld"),
        ("B", "tokenization", "patch length 80", "30 patches/channel; 60 total tokens", "non-overlapping patches"),
        ("B", "masking", "mixed", "25% target", "selected development-only configuration"),
        ("B", "masking_sensitivity", "FHR-asymmetric", "development-only", "not used to replace the frozen route"),
        ("B", "encoder", "transformer", "d_model=48; 2 layers; 4 heads", "74,768 trainable parameters"),
        ("B", "embedding", "record embedding", "window mean; 48 dimensions", "record aggregation downstream"),
        ("B", "objective_reconstruction", "masked reconstruction", "weight 1.0", "development-only SSL objective"),
        ("B", "objective_contrastive", "NT-Xent consistency", "weight 0.1", "temperature 0.2"),
        ("B", "encoder_origins", "random; CTU-only; JNU-only; JNU-to-CTU", "prespecified comparison", "no post-test route selection"),
        ("C", "jnu_records", "records", "20,769", "antepartum public CTG"),
        ("C", "jnu_groups", "patient groups", "12,606", "group identifier retained"),
        ("C", "ctu_all", "records", "552", "intrapartum CTU-UHB"),
        ("C", "development", "records/events", "386/79", "20.47% primary outcome"),
        ("C", "fixed_test", "records/events", "166/34", "20.48% primary outcome"),
        ("C", "ctgdl", "records", "135", "external representation branch"),
        ("D", "representation", "JNU and CTU development", "labels withheld", "representation learning"),
        ("D", "morphology", "CTU expert events", "linear probe", "representation task only"),
        ("D", "risk", "CTU development and fixed test", "exploratory", "no stable incremental benefit"),
        ("D", "external_outcome", "none harmonized", "not performed", "no external outcome validation"),
    ]
    header = ["panel", "item", "role", "value", "claim_boundary"]
    for path in (SOURCE_DATA, LOCAL_SOURCE_DATA):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)


def audit_text_overlaps(fig) -> int:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    records = []
    overlaps = []
    for axis_index, ax in enumerate(fig.axes):
        texts = [item for item in ax.texts if item.get_text().strip()]
        bboxes = [item.get_window_extent(renderer=renderer).expanded(0.99, 0.92) for item in texts]
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if bboxes[i].overlaps(bboxes[j]):
                    intersection_width = max(0.0, min(bboxes[i].x1, bboxes[j].x1) - max(bboxes[i].x0, bboxes[j].x0))
                    intersection_height = max(0.0, min(bboxes[i].y1, bboxes[j].y1) - max(bboxes[i].y0, bboxes[j].y0))
                    intersection_area = intersection_width * intersection_height
                    smaller_area = min(bboxes[i].width * bboxes[i].height, bboxes[j].width * bboxes[j].height)
                    if smaller_area > 0 and intersection_area / smaller_area > 0.08:
                        overlaps.append((axis_index, texts[i].get_text().replace("\n", " / "), texts[j].get_text().replace("\n", " / "), intersection_area / smaller_area))
    with OVERLAP_AUDIT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["axis", "text_1", "text_2", "overlap_fraction_of_smaller_bbox"])
        writer.writerows(overlaps)
    return len(overlaps)


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_source_data()

    fig = plt.figure(figsize=(FIGURE_WIDTH_MM * MM_TO_INCH, FIGURE_HEIGHT_MM * MM_TO_INCH), facecolor=WHITE)
    ax_a = fig.add_axes([0.000, 0.665, 1.000, 0.325])
    ax_b = fig.add_axes([0.000, 0.345, 1.000, 0.295])
    ax_c = fig.add_axes([0.000, 0.015, 0.495, 0.300])
    ax_d = fig.add_axes([0.505, 0.015, 0.495, 0.300])
    draw_panel_a(ax_a)
    draw_panel_b(ax_b)
    draw_panel_c(ax_c)
    draw_panel_d(ax_d)

    overlap_count = audit_text_overlaps(fig)
    if overlap_count:
        raise RuntimeError(f"Text overlap audit failed with {overlap_count} intersections; see {OVERLAP_AUDIT}")

    output = OUT_DIR / "Figure_1_final_study_design"
    fig.savefig(output.with_suffix(".png"), dpi=300, facecolor=WHITE)
    fig.savefig(output.with_suffix(".pdf"), facecolor=WHITE)
    fig.savefig(output.with_suffix(".svg"), facecolor=WHITE)
    fig.savefig(output.with_suffix(".tiff"), dpi=300, facecolor=WHITE, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(output.with_suffix(".jpg"), dpi=300, facecolor=WHITE, pil_kwargs={"quality": 95, "subsampling": 0})
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
