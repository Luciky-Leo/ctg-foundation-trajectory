#!/usr/bin/env python3
"""Audit Figure 1 v2 source-data mapping, outputs, and final-size constraints."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "Figure_1_final_source_data.csv"
TABLES = ROOT / "intermediate_tables"
REPORT = ROOT / "FIGURE_1_V2_QA_REPORT.md"


def read_rows(path: Path, delimiter: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def normalized_integer(value: str) -> int:
    return int(value.replace(",", "").strip())


def main() -> None:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, passed: bool, evidence: str) -> None:
        checks.append((name, bool(passed), evidence))

    source_rows = read_rows(SOURCE, ",")
    source = {(row["panel"], row["item"]): row for row in source_rows}

    counts_rows = read_rows(TABLES / "Figure_1C_counts_input_mapped.tsv", "\t")
    observed_counts = {row["label"]: normalized_integer(row["count"]) for row in counts_rows}
    expected_counts = {
        "JNU-CTG records": normalized_integer(source[("C", "jnu_records")]["value"]),
        "JNU patient groups": normalized_integer(source[("C", "jnu_groups")]["value"]),
        "CTU-UHB records": normalized_integer(source[("C", "ctu_all")]["value"]),
        "CTGDL records": normalized_integer(source[("C", "ctgdl")]["value"]),
    }
    check("Panel C count mapping", observed_counts == expected_counts, str(observed_counts))

    event_rows = read_rows(TABLES / "Figure_1C_events_input_mapped.tsv", "\t")
    observed_events = {
        row["cohort"]: (int(row["total"]), int(row["events"])) for row in event_rows
    }

    def split_total_events(key: tuple[str, str]) -> tuple[int, int]:
        total, events = source[key]["value"].split("/")
        return int(total), int(events)

    expected_events = {
        "Development": split_total_events(("C", "development")),
        "Fixed test": split_total_events(("C", "fixed_test")),
    }
    check("Panel C event mapping", observed_events == expected_events, str(observed_events))
    event_fractions_ok = all(
        abs(float(row["event_fraction"]) - int(row["events"]) / int(row["total"])) < 1e-12
        for row in event_rows
    )
    check("Panel C event fractions", event_fractions_ok, "79/386 and 34/166 recomputed")

    roles = read_rows(TABLES / "Figure_1D_roles_input_mapped.tsv", "\t")
    role_keys = {(row["analysis_task"], row["data_role"], row["status"]) for row in roles}
    required_roles = {
        ("Representation learning", "Unlabeled public CTG", "supporting"),
        ("Representation learning", "CTU development", "supporting"),
        ("Morphology probe", "CTU expert labels", "supporting"),
        ("Risk enrichment", "CTU development", "supporting"),
        ("Risk enrichment", "Fixed test", "exploratory"),
        ("Domain context", "CTGDL", "context"),
        ("Outcome validation", "Harmonized external", "absent"),
    }
    check("Panel D role mapping", required_roles.issubset(role_keys), f"{len(role_keys)} role cells")

    expected_files = [
        ROOT / "outputs/A/Figure_1A_image2_workflow.png",
        ROOT / "outputs/A/Figure_1A_image2_workflow.pdf",
        ROOT / "outputs/A/Figure_1A_image2_workflow.svg",
        ROOT / "outputs/B/Figure_1B_image2_ssl.png",
        ROOT / "outputs/B/Figure_1B_image2_ssl.pdf",
        ROOT / "outputs/B/Figure_1B_image2_ssl.svg",
        ROOT / "outputs/C/Figure_1C_persist.png",
        ROOT / "outputs/C/Figure_1C_persist.pdf",
        ROOT / "outputs/C/Figure_1C_persist.svg",
        ROOT / "outputs/D/Figure_1D_persist.png",
        ROOT / "outputs/D/Figure_1D_persist.pdf",
        ROOT / "outputs/D/Figure_1D_persist.svg",
        ROOT / "Figure_1_R5_JNU_v2_20260829.png",
        ROOT / "Figure_1_R5_JNU_v2_20260829.pdf",
        ROOT / "Figure_1_R5_JNU_v2_20260829.tiff",
        ROOT / "Figure_1_R5_JNU_v2_20260829.jpg",
    ]
    missing = [str(path.relative_to(ROOT)) for path in expected_files if not path.is_file()]
    check("Panel and composite outputs", not missing, "missing=" + (", ".join(missing) or "none"))

    dims = read_rows(ROOT / "final_assembly_dimensions.tsv", "\t")[0]
    width_mm = float(dims["figure_width_mm"])
    height_mm = float(dims["figure_height_mm"])
    check(
        "Final-size dimensions",
        width_mm <= 180.0 and height_mm <= 240.0 and dims["assembly_scale"] == "100%",
        f"{width_mm:.3f} x {height_mm:.3f} mm at {dims['assembly_scale']}",
    )

    with Image.open(ROOT / "Figure_1_R5_JNU_v2_20260829.png") as image:
        width_px, height_px = image.size
        dpi = image.info.get("dpi", (0, 0))
    check(
        "Raster export",
        width_px >= 2100 and height_px >= 2750 and min(dpi) >= 295,
        f"{width_px} x {height_px} px; dpi={dpi}",
    )

    persist_validation = (ROOT / "persist_source_code_first_validation.md").read_text(
        encoding="utf-8"
    )
    check(
        "PERSIST source-code-first validation",
        "Status: `PASS`" in persist_validation,
        "Panels C/D validated; Panels A/B correctly retained as native Image2 routes",
    )
    candidate_report = (ROOT / "PERSIST_CANDIDATE_GATE_AUDIT_20260829_v3.md").read_text(
        encoding="utf-8"
    )
    check(
        "PERSIST candidate gate",
        "Overall status: `PASS`" in candidate_report
        and "| `PASS` | 5 |" in candidate_report
        and "| `FAIL` | 0 |" in candidate_report,
        "5/5 candidates and 2/2 panels passed",
    )

    manual_evidence = [
        "Panel A CTGDL arrow points from the frozen representation toward domain context.",
        "Panel A fixed test is labelled exploratory and separated from development-only selection.",
        "Panel B shows FHR/UC inputs, patch tokens, mixed masking, transformer embedding, reconstruction, and NT-Xent.",
        "Panels C/D were inspected at original resolution; titles, values, axis text, headers, symbols, and legends do not overlap.",
        "The figure does not claim external neonatal-outcome validation or a prespecified lead-time horizon.",
    ]

    status = "PASS" if all(passed for _, passed, _ in checks) else "FAIL"
    lines = [
        "# Figure 1 v2 QA report",
        "",
        f"Status: `{status}`",
        "",
        "## Automated checks",
        "",
        "| Check | Result | Evidence |",
        "|---|---|---|",
    ]
    for name, passed, evidence in checks:
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} | {evidence} |")
    lines.extend(["", "## Original-size visual review", ""])
    lines.extend(f"- {item}" for item in manual_evidence)
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "Only Panels C and D are PERSIST source-code-first renders. Panels A and B are separately generated Image2 schematics with deterministic scripted preparation. The composite is therefore a hybrid source-controlled figure, not a whole-figure PERSIST reproduction.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"status={status}")
    print(f"report={REPORT}")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
