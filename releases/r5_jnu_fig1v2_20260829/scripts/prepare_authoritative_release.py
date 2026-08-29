#!/usr/bin/env python3
"""Prepare append-only authoritative folders before the canonical preflight."""

from pathlib import Path
import shutil


BATCH = Path(__file__).resolve().parents[2]
SUBMISSION = BATCH / "08_submission"
SOURCE_PACKAGE = SUBMISSION / "Frontiers_R5_JNU_Resubmission_FINAL2_20260828"
SOURCE_PAYLOAD = SUBMISSION / "CTG_R5_JNU_GitHub_Zenodo_FINAL2_20260828"
PACKAGE = SUBMISSION / "Frontiers_R5_JNU_Resubmission_UPLOAD_READY_20260828"
PAYLOAD = SUBMISSION / "CTG_R5_JNU_GitHub_Zenodo_PUBLIC_RELEASE_20260828"

for source, target in [(SOURCE_PACKAGE, PACKAGE), (SOURCE_PAYLOAD, PAYLOAD)]:
    if not source.is_dir():
        raise FileNotFoundError(source)
    if target.exists():
        raise FileExistsError(target)
    shutil.copytree(source, target)

status = """# Authoritative release status

The only authoritative local release folders are:

- `Frontiers_R5_JNU_Resubmission_UPLOAD_READY_20260828`
- `CTG_R5_JNU_GitHub_Zenodo_PUBLIC_RELEASE_20260828`

Earlier package builds in this directory are append-only construction records and are superseded. Do not upload an earlier ZIP.

The upload-ready folder must contain a canonical `SUBMISSION_PRECHECK_REPORT.md` with `Status: PASS` before final archiving.
"""
(SUBMISSION / "AUTHORITATIVE_RELEASE.md").write_text(status, encoding="utf-8")
print(PACKAGE)
print(PAYLOAD)
