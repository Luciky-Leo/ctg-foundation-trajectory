#!/usr/bin/env python3
"""Build verified Frontiers-upload and GitHub/Zenodo R5 JNU packages."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from PIL import Image


BATCH = Path(__file__).resolve().parents[2]
PROJECT = BATCH.parent
SOURCE = BATCH / "06_manuscript" / "frontiers_r5_jnu_source"
FIGURES = BATCH / "05_figures"
RESPONSE = BATCH / "07_response"
REPRO = BATCH / "09_reproducibility"
SUBMISSION_ROOT = BATCH / "08_submission"
PREVIOUS = PROJECT / "R2_R5_Targeted_Revision_20260826"

PACKAGE_NAME = "Frontiers_R5_JNU_Resubmission_FINAL2_20260828"
PAYLOAD_NAME = "CTG_R5_JNU_GitHub_Zenodo_FINAL2_20260828"
PACKAGE = SUBMISSION_ROOT / PACKAGE_NAME
PAYLOAD = SUBMISSION_ROOT / PAYLOAD_NAME


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def export_figure(source: Path, destination_stem: Path) -> None:
    destination_stem.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as raw:
        image = raw.convert("RGB")
        image.save(destination_stem.with_suffix(".jpg"), quality=95, subsampling=0, dpi=(300, 300))
        image.save(destination_stem.with_suffix(".tiff"), compression="tiff_lzw", dpi=(300, 300))


def manifest(root: Path, name: str = "CHECKSUMS_SHA256.csv") -> Path:
    target = root / name
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != target:
            rows.append({"relative_path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    return target


def zip_and_verify(root: Path) -> dict[str, object]:
    zip_path = root.parent / f"{root.name}.zip"
    if zip_path.exists():
        raise FileExistsError(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, Path(root.name) / path.relative_to(root))

    verify_root = BATCH / "99_temp" / f"verify_{root.name}"
    if verify_root.exists():
        raise FileExistsError(verify_root)
    verify_root.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as archive:
        bad_member = archive.testzip()
        archive.extractall(verify_root)
    extracted = verify_root / root.name
    original_files = {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob("*") if p.is_file()}
    extracted_files = {p.relative_to(extracted).as_posix(): sha256(p) for p in extracted.rglob("*") if p.is_file()}
    passed = bad_member is None and original_files == extracted_files
    return {
        "status": "PASS" if passed else "FAIL",
        "zip": str(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "zip_sha256": sha256(zip_path),
        "files": len(original_files),
        "bad_member": bad_member,
    }


for target in [PACKAGE, PAYLOAD]:
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)

# Frontiers upload package.
manuscript_dir = PACKAGE / "01_Manuscript_upload"
for name in ["frontiers_ctg_manuscript_r5.pdf", "frontiers_ctg_manuscript_r5.tex", "references.bib"]:
    copy(SOURCE / name, manuscript_dir / name)

figure_sources = {
    "Figure_1_R5_JNU_study_design": FIGURES / "Figure_1_final_study_design.png",
    "Figure_2_R5_model_validation": SOURCE / "figures" / "Figure_2_R5_revised.png",
    "Figure_3_dynamic_phenotypes": SOURCE / "figures" / "Figure_3_dynamic_phenotype_prototypes.jpg",
    "Figure_4_R5_calibration": SOURCE / "figures" / "Figure_4_R5_calibration.png",
}
for name, source in figure_sources.items():
    export_figure(source, PACKAGE / "02_Figures_optional_300dpi_JPG_TIFF" / name)

supp_dir = PACKAGE / "03_Supplementary_files_upload"
for name in ["frontiers_ctg_supplementary_material_r5.pdf", "frontiers_ctg_supplementary_material_r5.tex"]:
    copy(SOURCE / name, supp_dir / name)
for source in sorted((SOURCE / "supplementary_tables").glob("*")):
    if source.is_file():
        copy(source, supp_dir / "supplementary_tables" / source.name)
supplementary_figure_sources = {
    "Supplementary_Figure_S1_model_selection_ablation": SOURCE / "figures" / "Figure_S1_R5_model_selection_ablation.png",
    "Supplementary_Figure_S2_external_representation": SOURCE / "figures" / "Supplementary_Figure_S2_R5_external_representation.png",
    "Supplementary_Figure_S3_JNU_cross_domain_sensitivity": FIGURES / "Supplementary_Figure_S3_JNU_cross_domain_sensitivity.png",
}
for name, source in supplementary_figure_sources.items():
    export_figure(source, supp_dir / "supplementary_figures_300dpi" / name)

review_dir = PACKAGE / "04_Review_only_files_upload"
for name in ["Response_to_Reviewer_5_R2_JNU_20260828.pdf", "Response_to_Reviewer_5_R2_JNU_20260828.md"]:
    copy(RESPONSE / name, review_dir / name)

support_dir = PACKAGE / "05_LaTeX_support_if_requested"
for name in [
    "frontiers_ctg_manuscript_r5.bbl",
    "frontiersinFPHY_FAMS.cls",
    "Frontiers-Harvard.bst",
    "FrontiersinHarvard.cls",
    "logo1.eps",
    "logo2.eps",
    "logos.eps",
    "YM-logo.eps",
]:
    copy(SOURCE / name, support_dir / name)

audit_names = [
    "JNU_R5_COMPLETION_AUDIT.md",
    "JNU_R5_AUDIT_CHECKS.csv",
    "JNU_LEAKAGE_AUDIT.md",
    "NUMERIC_CROSS_AUDIT.md",
    "FIGURE_QUALITY_REVIEW.md",
    "REFERENCE_LOCAL_AUDIT.md",
    "REFERENCE_ONLINE_AUDIT.md",
    "REFERENCE_ONLINE_AUDIT.csv",
    "METHOD_FOUR_GATE_AUDIT_CTG_R5_JNU.md",
    "method_four_gate_audit_CTG_R5_JNU.csv",
    "REPRODUCIBILITY_MANIFEST.csv",
    "REPRODUCIBILITY_MANIFEST.json",
]
for name in audit_names:
    copy(REPRO / name, PACKAGE / "06_Reproducibility_review_only" / name)

upload_map = """# Frontiers R5 JNU individual upload map

Do not upload the ZIP to the Frontiers file boxes. Upload the individual files below.

## Manuscript

- `01_Manuscript_upload/frontiers_ctg_manuscript_r5.pdf`
- `01_Manuscript_upload/frontiers_ctg_manuscript_r5.tex`
- `01_Manuscript_upload/references.bib`

## Main figures

Upload the four JPG files from `02_Figures_optional_300dpi_JPG_TIFF`. TIFF alternatives are provided for production requests.

## Supplementary files

- `03_Supplementary_files_upload/frontiers_ctg_supplementary_material_r5.pdf`
- The 22 machine-readable CSV tables and their index under `supplementary_tables`
- Supplementary figure JPG files only if the portal requests them separately

## Review-only files

- `04_Review_only_files_upload/Response_to_Reviewer_5_R2_JNU_20260828.pdf`

The analysis is a retrospective delivery-level risk-enrichment study. JNU supplies cross-domain label-free pretraining and morphology-transfer evidence, not external neonatal-outcome validation.
"""
(PACKAGE / "UPLOAD_FILE_MAP.md").write_text(upload_map, encoding="utf-8")
manifest(PACKAGE)

# GitHub/Zenodo payload. It is intentionally source/result focused and excludes raw data and weights.
copy(PREVIOUS / "08_submission" / "CTG_R5_GitHub_Zenodo_Payload_20260826" / "LICENSE", PAYLOAD / "LICENSE")
copy(PREVIOUS / "08_submission" / "CTG_R5_GitHub_Zenodo_Payload_20260826" / "requirements.txt", PAYLOAD / "requirements.txt")

for name in [
    "frontiers_ctg_manuscript_r5.pdf",
    "frontiers_ctg_manuscript_r5.tex",
    "frontiers_ctg_supplementary_material_r5.pdf",
    "frontiers_ctg_supplementary_material_r5.tex",
    "references.bib",
    "frontiers_ctg_manuscript_r5.bbl",
    "frontiersinFPHY_FAMS.cls",
    "Frontiers-Harvard.bst",
    "FrontiersinHarvard.cls",
]:
    copy(SOURCE / name, PAYLOAD / "manuscript" / name)

for source in sorted((SOURCE / "figures").glob("*")):
    if source.is_file():
        copy(source, PAYLOAD / "figures" / source.name)
for source in sorted(FIGURES.glob("Figure_1_final_study_design.*")):
    copy(source, PAYLOAD / "figures" / source.name)
for source in sorted(FIGURES.glob("Supplementary_Figure_S3_JNU_cross_domain_sensitivity.*")):
    copy(source, PAYLOAD / "figures" / source.name)
for source in sorted((SOURCE / "supplementary_tables").glob("*")):
    if source.is_file():
        copy(source, PAYLOAD / "supplement" / source.name)

for source in sorted((BATCH / "03_analysis" / "scripts").glob("*")):
    if source.is_file():
        copy(source, PAYLOAD / "scripts" / source.name)
for source in sorted((BATCH / "03_analysis" / "figure1_redesign").glob("*.py")):
    copy(source, PAYLOAD / "scripts" / "figure1" / source.name)
for source in sorted((BATCH / "03_analysis" / "figure_s3").glob("*.py")):
    copy(source, PAYLOAD / "scripts" / "figure_s3" / source.name)
for source in sorted((BATCH / "02_protocol").glob("*")):
    if source.is_file():
        copy(source, PAYLOAD / "protocol" / source.name)

result_allow = {".csv", ".json", ".md", ".flag"}
for source in sorted((BATCH / "04_tables").glob("*")):
    if source.is_file() and source.suffix.lower() in result_allow and "probabilities" not in source.name:
        copy(source, PAYLOAD / "results" / source.name)
for source in sorted((BATCH / "03_analysis").glob("*")):
    if source.is_file() and source.suffix.lower() in result_allow:
        copy(source, PAYLOAD / "results" / source.name)
for name in audit_names + ["R5_JNU_METHOD_FOUR_GATE_INPUT.csv"]:
    copy(REPRO / name, PAYLOAD / "reproducibility" / name)
copy(RESPONSE / "Response_to_Reviewer_5_R2_JNU_20260828.md", PAYLOAD / "reproducibility" / "Response_to_Reviewer_5_R2_JNU_20260828.md")

readme = """# CTG R5 JNU cross-domain revision payload

This payload accompanies the 2026-08-28 Reviewer 5 revision of the public CTG representation-learning study.

## Revision-specific addition

JNU-CTG (Zenodo `10.5281/zenodo.21800730`) was used only for label-free antepartum pretraining. All 20,769 records were readable and mapped to 12,606 patient groups; 62,307 non-overlapping 10-minute windows were generated. Clinical labels were withheld during SSL, and grouped splits prevented patient overlap.

The prespecified promotion gate was not met for neonatal-risk enrichment. JNU-to-CTU adaptation transferred expert-labelled morphology information but did not stably outperform CTU-only SSL for risk enrichment. These results remain supplementary and are not presented as external outcome validation.

## Contents

- `manuscript`: final Frontiers source, bibliography, supplement, and PDFs.
- `figures`: manuscript figures and reproducible Figure 1/S3 vector exports.
- `supplement`: machine-readable supplementary tables S1-S18.
- `scripts`: JNU preparation, frozen cross-domain training, analysis, figure, and audit code.
- `protocol`: frozen JNU cross-domain addendum.
- `results`: compact aggregate tables and promotion decision; no patient-level raw signals.
- `reproducibility`: leakage, numeric, figure, reference, and four-gate audits.

## Deliberate exclusions

Raw CTG records, window arrays, model checkpoints, window-level embeddings, credentials, and temporary build files are excluded. Obtain public data from the original providers under their licenses. JNU-CTG is CC BY 4.0.

## Persistent links

- Repository: https://github.com/Luciky-Leo/ctg-foundation-trajectory
- Zenodo concept DOI: https://doi.org/10.5281/zenodo.20364141

The included source code is distributed under `LICENSE`. Dataset licenses remain with the original providers.
"""
(PAYLOAD / "README.md").write_text(readme, encoding="utf-8")
manifest(PAYLOAD)

package_result = zip_and_verify(PACKAGE)
payload_result = zip_and_verify(PAYLOAD)
status = "PASS" if package_result["status"] == payload_result["status"] == "PASS" else "FAIL"
report = {
    "status": status,
    "frontiers_package": package_result,
    "github_zenodo_payload": payload_result,
    "raw_data_included": False,
    "model_weights_included": False,
}
(SUBMISSION_ROOT / "PACKAGE_BUILD_AND_ZIP_VERIFY.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
raise SystemExit(0 if status == "PASS" else 1)
