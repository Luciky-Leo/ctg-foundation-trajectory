#!/usr/bin/env python3
"""Finalize and verify the canonical preflight-passing release archives."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


BATCH = Path(__file__).resolve().parents[2]
SUBMISSION = BATCH / "08_submission"
PACKAGE = SUBMISSION / "Frontiers_R5_JNU_Resubmission_UPLOAD_READY_20260828"
PAYLOAD = SUBMISSION / "CTG_R5_JNU_GitHub_Zenodo_PUBLIC_RELEASE_20260828"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(root: Path) -> None:
    target = root / "CHECKSUMS_SHA256.csv"
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != target:
            rows.append({"relative_path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)


def archive_and_verify(root: Path) -> dict[str, object]:
    archive_path = root.with_suffix(".zip")
    if archive_path.exists():
        raise FileExistsError(archive_path)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, Path(root.name) / path.relative_to(root))
    verify = BATCH / "99_temp" / f"verify_authoritative_{root.name}"
    if verify.exists():
        raise FileExistsError(verify)
    verify.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        archive.extractall(verify)
    extracted = verify / root.name
    expected = {path.relative_to(root).as_posix(): sha256(path) for path in root.rglob("*") if path.is_file()}
    observed = {path.relative_to(extracted).as_posix(): sha256(path) for path in extracted.rglob("*") if path.is_file()}
    passed = bad_member is None and expected == observed
    return {
        "status": "PASS" if passed else "FAIL",
        "archive": str(archive_path),
        "bytes": archive_path.stat().st_size,
        "sha256": sha256(archive_path),
        "files": len(expected),
    }


preflight = PACKAGE / "SUBMISSION_PRECHECK_REPORT.md"
if "Status: `PASS`" not in preflight.read_text(encoding="utf-8"):
    raise RuntimeError("Canonical submission preflight is not PASS")

payload_preflight = PAYLOAD / "reproducibility" / "SUBMISSION_PRECHECK_REPORT.md"
payload_preflight.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(preflight, payload_preflight)

for root in [PACKAGE, PAYLOAD]:
    forbidden = [path for path in root.rglob("*") if path.is_file() and (path.suffix.lower() in {".pt", ".pth", ".npy", ".npz"} or "raw_ctg" in path.name.lower())]
    if forbidden:
        raise RuntimeError(f"Forbidden raw/model artifact in public release: {forbidden}")
    write_manifest(root)

package_result = archive_and_verify(PACKAGE)
payload_result = archive_and_verify(PAYLOAD)
status = "PASS" if package_result["status"] == payload_result["status"] == "PASS" else "FAIL"
report = {
    "status": status,
    "canonical_preflight": "PASS",
    "frontiers_upload_ready": package_result,
    "github_zenodo_public_release": payload_result,
    "raw_data_included": False,
    "model_weights_included": False,
}
(SUBMISSION / "AUTHORITATIVE_RELEASE_VERIFY.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
raise SystemExit(0 if status == "PASS" else 1)
