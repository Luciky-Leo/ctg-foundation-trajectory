from __future__ import annotations

# SOURCE_CODE_FIRST evidence for A/B is the retained Image2 source artwork in
# image2_sources/ plus this deterministic crop, direction-correction, and exact
# metadata overlay script.

import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


SCRIPT = Path(__file__).resolve()
REDRAW_ROOT = SCRIPT.parents[1]
SOURCE_ROOT = REDRAW_ROOT / "image2_sources"
MM = 1.0 / 25.4


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def crop_white_margin(image: Image.Image, margin: int = 26) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    nonwhite = np.min(rgb, axis=2) < 247
    ys, xs = np.where(nonwhite)
    if len(xs) == 0:
        return image.copy()
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(image.width, int(xs.max()) + margin + 1)
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(image.height, int(ys.max()) + margin + 1)
    return image.crop((x0, y0, x1, y1))


def correct_panel_a_ctgdl_direction(image: Image.Image) -> Image.Image:
    """Reverse the generated CTGDL branch so the frozen encoder is applied outward."""
    fixed = image.convert("RGB").copy()
    draw = ImageDraw.Draw(fixed)
    x = int(round(fixed.width * 0.607))
    y0 = int(round(fixed.height * 0.515))
    y1 = int(round(fixed.height * 0.661))
    draw.rectangle((x - 11, y0 - 4, x + 11, y1 + 4), fill="white")
    color = "#C9832E"
    for y in range(y0, y1 - 16, 16):
        draw.line((x, y, x, min(y + 9, y1 - 16)), fill=color, width=3)
    draw.polygon(((x, y1), (x - 8, y1 - 13), (x + 8, y1 - 13)), fill=color)
    return fixed


def render_panel(
    panel: str,
    source_name: str,
    title: str,
    subtitle: str,
) -> dict[str, str | int | float]:
    output_dir = REDRAW_ROOT / "outputs" / panel
    output_dir.mkdir(parents=True, exist_ok=True)
    source = SOURCE_ROOT / source_name
    generated = Image.open(source)
    if panel == "A":
        generated = correct_panel_a_ctgdl_direction(generated)
    artwork = crop_white_margin(generated)
    cropped = output_dir / f"Figure_1{panel}_image2_artwork_cropped.png"
    artwork.save(cropped, dpi=(300, 300))

    width_mm = 180.0
    art_width_mm = 176.0
    title_height_mm = 10.0
    art_height_mm = art_width_mm * artwork.height / artwork.width
    height_mm = title_height_mm + art_height_mm + 1.5

    fig = plt.figure(figsize=(width_mm * MM, height_mm * MM), facecolor="white")
    fig.text(0.006, 0.975, panel, ha="left", va="top", fontsize=8, fontweight="bold", color="#2F3136")
    fig.text(0.038, 0.975, title, ha="left", va="top", fontsize=8, fontweight="bold", color="#2F3136")
    fig.text(0.038, 0.925, subtitle, ha="left", va="top", fontsize=6.2, color="#666A70")
    ax = fig.add_axes(
        [
            (width_mm - art_width_mm) / (2 * width_mm),
            1.5 / height_mm,
            art_width_mm / width_mm,
            art_height_mm / height_mm,
        ]
    )
    ax.imshow(artwork)
    ax.axis("off")

    stem = f"Figure_1{panel}_image2_{'workflow' if panel == 'A' else 'ssl'}"
    paths = {}
    for ext in ("png", "pdf", "svg"):
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None)
        paths[ext] = str(path)
    plt.close(fig)
    return {
        "panel": panel,
        "source": str(source),
        "cropped_source": str(cropped),
        "source_width_px": artwork.width,
        "source_height_px": artwork.height,
        "render_width_mm": width_mm,
        "render_height_mm": round(height_mm, 3),
        **paths,
    }


def main() -> None:
    setup_style()
    records = [
        render_panel(
            "A",
            "Figure_1A_image2_workflow_source.png",
            "Leakage-aware public CTG workflow",
            "Development-only selection; the historically exposed fixed test remains exploratory.",
        ),
        render_panel(
            "B",
            "Figure_1B_image2_ssl_source.png",
            "Frozen PatchTST-style CTG encoder",
            "10-min, 4-Hz windows | patch length 80 | d_model 48 | 2 layers | 4 heads",
        ),
    ]
    ledger = REDRAW_ROOT / "intermediate_tables" / "image2_panel_asset_ledger.tsv"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
