from __future__ import annotations

from pathlib import Path
import shutil

import fitz
from PIL import Image


SCRIPT = Path(__file__).resolve()
REDRAW_ROOT = SCRIPT.parents[1]
REVISION_ROOT = SCRIPT.parents[3]
FIGURE_DIR = REVISION_ROOT / "05_figures"
MM_TO_PT = 72.0 / 25.4


def mm(value: float) -> float:
    return value * MM_TO_PT


def page_size(pdf_path: Path) -> tuple[float, float]:
    doc = fitz.open(pdf_path)
    rect = doc[0].rect
    doc.close()
    return rect.width, rect.height


def place_pdf(page: fitz.Page, path: Path, x_mm: float, y_mm: float, width_mm: float) -> float:
    src = fitz.open(path)
    src_rect = src[0].rect
    target_width = mm(width_mm)
    target_height = target_width * src_rect.height / src_rect.width
    rect = fitz.Rect(mm(x_mm), mm(y_mm), mm(x_mm) + target_width, mm(y_mm) + target_height)
    page.show_pdf_page(rect, src, 0, keep_proportion=True, overlay=True)
    src.close()
    return target_height / MM_TO_PT


def main() -> None:
    a = REDRAW_ROOT / "outputs" / "A" / "Figure_1A_image2_workflow.pdf"
    b = REDRAW_ROOT / "outputs" / "B" / "Figure_1B_image2_ssl.pdf"
    c = REDRAW_ROOT / "outputs" / "C" / "Figure_1C_persist.pdf"
    d = REDRAW_ROOT / "outputs" / "D" / "Figure_1D_persist.pdf"
    for path in (a, b, c, d):
        if not path.exists():
            raise FileNotFoundError(path)

    _, a_h_pt = page_size(a)
    _, b_h_pt = page_size(b)
    _, c_h_pt = page_size(c)
    a_h = a_h_pt / MM_TO_PT
    b_h = b_h_pt / MM_TO_PT
    c_h = c_h_pt / MM_TO_PT
    vertical_gap = 2.0
    page_height_mm = a_h + b_h + c_h + 2 * vertical_gap
    if page_height_mm > 240.0:
        raise RuntimeError(f"Figure height {page_height_mm:.2f} mm exceeds 240 mm")

    local_pdf = REDRAW_ROOT / "Figure_1_R5_JNU_v2_20260829.pdf"
    doc = fitz.open()
    page = doc.new_page(width=mm(180.0), height=mm(page_height_mm))
    y = 0.0
    y += place_pdf(page, a, 0.0, y, 180.0) + vertical_gap
    y += place_pdf(page, b, 0.0, y, 180.0) + vertical_gap
    place_pdf(page, c, 0.0, y, 87.0)
    place_pdf(page, d, 93.0, y, 87.0)
    doc.save(local_pdf, garbage=4, deflate=True)
    doc.close()

    rendered = fitz.open(local_pdf)
    pix = rendered[0].get_pixmap(dpi=300, alpha=False)
    local_png = REDRAW_ROOT / "Figure_1_R5_JNU_v2_20260829.png"
    pix.save(local_png)
    rendered.close()

    image = Image.open(local_png).convert("RGB")
    local_tiff = REDRAW_ROOT / "Figure_1_R5_JNU_v2_20260829.tiff"
    local_jpg = REDRAW_ROOT / "Figure_1_R5_JNU_v2_20260829.jpg"
    image.save(local_tiff, dpi=(300, 300), compression="tiff_lzw")
    image.save(local_jpg, dpi=(300, 300), quality=95, subsampling=0)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for source in (local_pdf, local_png, local_tiff, local_jpg):
        shutil.copy2(source, FIGURE_DIR / source.name)

    with (REDRAW_ROOT / "final_assembly_dimensions.tsv").open("w", encoding="utf-8") as handle:
        handle.write("figure_width_mm\tfigure_height_mm\tpanel_A_height_mm\tpanel_B_height_mm\tpanel_C_D_height_mm\tassembly_scale\n")
        handle.write(f"180.000\t{page_height_mm:.3f}\t{a_h:.3f}\t{b_h:.3f}\t{c_h:.3f}\t100%\n")


if __name__ == "__main__":
    main()
