# PDF final visual QA: Figure 1 v2 revision

Status: `PASS`

Date: 2026-08-29

## Files reviewed

- `06_manuscript/frontiers_r5_jnu_source/frontiers_ctg_manuscript_r5.pdf` (14 pages)
- `06_manuscript/frontiers_r5_jnu_source/frontiers_ctg_supplementary_material_r5.pdf` (9 pages)
- `07_response/Response_to_Reviewer_5_R2_JNU_20260828.pdf` (4 pages)

## Visual checks

- Main-manuscript page 4 was rendered at 180 dpi and inspected at full resolution. Figure 1 and its complete legend fit within the page; no figure, caption, header, footer, or page-number collision was present.
- Figure 1 panels A-D remained readable at the embedded manuscript size. No panel title, numeric annotation, axis label, header, symbol, or legend overlapped another element.
- The CTGDL branch in Panel A points outward from the frozen representation and is described as representation-domain context, not external outcome validation.
- The fixed test is explicitly exploratory in Panels A and D. The figure contains no lead-time, deployment, or external neonatal-outcome validation claim.
- Supplement page 8 was rendered at 160 dpi after the Table S16 column-width adjustment. All cells and confidence intervals remain inside the table boundary.
- All four response-letter pages were rendered at 120 dpi and inspected. Smart-quote and long-dash missing-glyph warnings were removed; headings, reviewer text, responses, and page numbers are complete and unobstructed.

## Compile checks

- Main manuscript: no oversized-float warning after Figure 1 height was set to `0.68\textheight`.
- Reviewer response: no missing-character or overfull-box warning after conversion to standard LaTeX punctuation and sentence reflow.
- Supplement: no overfull-box warning after Table S16 column-width reallocation.

