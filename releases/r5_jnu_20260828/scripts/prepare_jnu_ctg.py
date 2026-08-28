#!/usr/bin/env python3
"""Validate JNU-CTG and build label-free 10-minute SSL windows from the ZIP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_MD5 = "ac1cfcba2f1b3336596544211d776771"
EXPECTED_SHA256 = "6efe9ce43f5ef803eb4ef65c87892e1a768899ff3bb82de56c18c834d5e09a98"
EXPECTED_RECORDS = 20_769
EXPECTED_PATIENTS = 12_606
SAMPLES_PER_RECORD = 7_200
SAMPLES_PER_WINDOW = 2_400
WINDOWS_PER_RECORD = 3


def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def robust_fill_and_scale(window: np.ndarray) -> np.ndarray:
    out = window.astype(np.float32, copy=True)
    for channel in range(out.shape[1]):
        values = out[:, channel]
        finite = np.isfinite(values)
        if not finite.any():
            out[:, channel] = 0.0
            continue
        median = float(np.nanmedian(values))
        values[~finite] = median
        q25, q75 = np.percentile(values, [25, 75])
        scale = float(q75 - q25)
        if scale <= 1e-6:
            scale = float(np.std(values))
        if scale <= 1e-6:
            scale = 1.0
        out[:, channel] = (values - median) / scale
    return out


def load_waveform(archive: zipfile.ZipFile, member: str) -> tuple[np.ndarray, dict[str, float]]:
    with archive.open(member) as handle:
        frame = pd.read_csv(handle, usecols=["FHR", "UC"], dtype=np.float32)
    values = frame[["FHR", "UC"]].to_numpy(dtype=np.float32, copy=True)
    if values.shape != (SAMPLES_PER_RECORD, 2):
        raise ValueError(f"unexpected waveform shape {values.shape}")

    fhr_invalid = (~np.isfinite(values[:, 0])) | (values[:, 0] < 50) | (values[:, 0] > 210)
    uc_invalid = (~np.isfinite(values[:, 1])) | (values[:, 1] < 0) | (values[:, 1] > 150)
    raw_fhr_zero = float(np.mean(values[:, 0] == 0))
    raw_uc_zero = float(np.mean(values[:, 1] == 0))
    values[fhr_invalid, 0] = np.nan
    values[uc_invalid, 1] = np.nan
    qc = {
        "fhr_invalid_fraction": float(np.mean(fhr_invalid)),
        "uc_invalid_fraction": float(np.mean(uc_invalid)),
        "fhr_zero_fraction_raw": raw_fhr_zero,
        "uc_zero_fraction_raw": raw_uc_zero,
    }
    return values, qc


def write_window_index(metadata: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "window_id", "record_id", "patient_id", "segment_index",
                "start_sample", "stop_sample", "start_minute", "stop_minute",
            ],
        )
        writer.writeheader()
        window_id = 0
        for row in metadata.itertuples(index=False):
            for segment in range(WINDOWS_PER_RECORD):
                start = segment * SAMPLES_PER_WINDOW
                writer.writerow({
                    "window_id": window_id,
                    "record_id": row.record_id,
                    "patient_id": row.patient_id,
                    "segment_index": segment,
                    "start_sample": start,
                    "stop_sample": start + SAMPLES_PER_WINDOW,
                    "start_minute": segment * 10,
                    "stop_minute": (segment + 1) * 10,
                })
                window_id += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--max-records", type=int, default=0, help="Smoke-test limit; 0 means all records.")
    args = parser.parse_args()

    archive_path = Path(args.archive)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    state_path = outdir / "prepare_state.json"
    matrix_path = outdir / "jnu_ctg_windows_f16.npy"
    index_path = outdir / "jnu_window_index.csv"
    qc_partial = outdir / "jnu_record_qc.partial.csv"
    qc_final = outdir / "jnu_record_qc.csv"
    failures_path = outdir / "jnu_waveform_failures.csv"

    started = time.time()
    md5 = file_hash(archive_path, "md5")
    sha256 = file_hash(archive_path, "sha256")
    if md5 != EXPECTED_MD5 or sha256 != EXPECTED_SHA256:
        raise SystemExit(f"archive checksum mismatch: md5={md5}, sha256={sha256}")

    with zipfile.ZipFile(archive_path) as archive:
        bad_members = archive.testzip()
        if bad_members is not None:
            raise SystemExit(f"ZIP integrity failure at {bad_members}")
        names = archive.namelist()
        waveform_members = {
            Path(name).stem: name
            for name in names
            if name.startswith("JNU-CTG/waveforms/csv_raw_7200/") and name.endswith(".csv")
        }
        metadata = pd.read_csv(
            archive.open("JNU-CTG/metadata.csv"),
            usecols=["record_id", "patient_id"],
            dtype={"record_id": "string", "patient_id": "string"},
        )
        license_text = archive.read("JNU-CTG/LICENSE.txt").decode("utf-8", errors="replace")

        if len(metadata) != EXPECTED_RECORDS:
            raise SystemExit(f"expected {EXPECTED_RECORDS} metadata rows, found {len(metadata)}")
        if metadata["patient_id"].nunique(dropna=False) != EXPECTED_PATIENTS:
            raise SystemExit(
                f"expected {EXPECTED_PATIENTS} patient groups, "
                f"found {metadata['patient_id'].nunique(dropna=False)}"
            )
        if len(waveform_members) != EXPECTED_RECORDS:
            raise SystemExit(f"expected {EXPECTED_RECORDS} waveform members, found {len(waveform_members)}")
        license_lower = license_text.lower()
        license_markers = (
            "creativecommons.org/licenses/by/4.0",
            "creative commons attribution 4.0 international",
        )
        if not any(marker in license_lower for marker in license_markers):
            raise SystemExit("CC BY 4.0 license marker not found")
        missing_members = sorted(set(metadata["record_id"].astype(str)) - set(waveform_members))
        if missing_members:
            raise SystemExit(f"metadata records without waveform files: {missing_members[:10]}")

        if args.max_records:
            metadata = metadata.iloc[: args.max_records].copy()
        n_records = len(metadata)
        n_windows = n_records * WINDOWS_PER_RECORD

        start_record = 0
        if state_path.exists() and matrix_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("archive_sha256") != sha256 or state.get("n_records") != n_records:
                raise SystemExit("existing checkpoint does not match this archive/run")
            start_record = int(state.get("next_record_index", 0))
            matrix = np.lib.format.open_memmap(matrix_path, mode="r+")
            qc_mode = "a"
        else:
            matrix = np.lib.format.open_memmap(
                matrix_path,
                mode="w+",
                dtype=np.float16,
                shape=(n_windows, 2, SAMPLES_PER_WINDOW),
            )
            qc_mode = "w"

        qc_fields = [
            "record_index", "record_id", "patient_id", "readable", "error",
            "fhr_invalid_fraction", "uc_invalid_fraction",
            "fhr_zero_fraction_raw", "uc_zero_fraction_raw",
        ]
        failures: list[dict[str, str]] = []
        if failures_path.exists() and start_record:
            failures = pd.read_csv(failures_path, dtype=str).fillna("").to_dict("records")

        with qc_partial.open(qc_mode, newline="", encoding="utf-8") as qc_handle:
            writer = csv.DictWriter(qc_handle, fieldnames=qc_fields)
            if qc_mode == "w":
                writer.writeheader()
            for record_index in range(start_record, n_records):
                row = metadata.iloc[record_index]
                record_id = str(row["record_id"])
                patient_id = str(row["patient_id"])
                qc_row = {
                    "record_index": record_index,
                    "record_id": record_id,
                    "patient_id": patient_id,
                    "readable": 1,
                    "error": "",
                }
                try:
                    waveform, qc = load_waveform(archive, waveform_members[record_id])
                    qc_row.update(qc)
                    for segment in range(WINDOWS_PER_RECORD):
                        start = segment * SAMPLES_PER_WINDOW
                        window = robust_fill_and_scale(waveform[start : start + SAMPLES_PER_WINDOW])
                        matrix[record_index * WINDOWS_PER_RECORD + segment] = window.T.astype(np.float16)
                except Exception as error:  # retain complete failure audit
                    qc_row.update({
                        "readable": 0,
                        "error": f"{type(error).__name__}: {error}",
                        "fhr_invalid_fraction": "",
                        "uc_invalid_fraction": "",
                        "fhr_zero_fraction_raw": "",
                        "uc_zero_fraction_raw": "",
                    })
                    failures.append({
                        "record_index": str(record_index),
                        "record_id": record_id,
                        "patient_id": patient_id,
                        "error": qc_row["error"],
                    })
                writer.writerow(qc_row)

                if (record_index + 1) % args.checkpoint_every == 0 or record_index + 1 == n_records:
                    matrix.flush()
                    qc_handle.flush()
                    os.fsync(qc_handle.fileno())
                    pd.DataFrame(
                        failures,
                        columns=["record_index", "record_id", "patient_id", "error"],
                    ).to_csv(failures_path, index=False)
                    state_path.write_text(json.dumps({
                        "status": "RUNNING" if record_index + 1 < n_records else "COMPLETE",
                        "archive_sha256": sha256,
                        "n_records": n_records,
                        "n_windows": n_windows,
                        "next_record_index": record_index + 1,
                        "failure_count": len(failures),
                    }, indent=2) + "\n", encoding="utf-8")
                    print(
                        f"processed {record_index + 1}/{n_records}; failures={len(failures)}; "
                        f"elapsed_min={(time.time() - started) / 60:.1f}",
                        flush=True,
                    )

    os.replace(qc_partial, qc_final)
    write_window_index(metadata, index_path)
    qc = pd.read_csv(qc_final)
    readable = int(qc["readable"].sum())
    readable_fraction = readable / len(qc)
    if readable_fraction < 0.95:
        raise SystemExit(f"waveform readability {readable_fraction:.3%} is below 95%")

    summary = {
        "status": "PASS",
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "md5": md5,
        "sha256": sha256,
        "license": "CC BY 4.0",
        "metadata_records_total": EXPECTED_RECORDS,
        "patient_groups_total": EXPECTED_PATIENTS,
        "processed_records": len(metadata),
        "processed_windows": len(metadata) * WINDOWS_PER_RECORD,
        "readable_records": readable,
        "readable_fraction": readable_fraction,
        "failed_records": len(metadata) - readable,
        "matrix_shape": [len(metadata) * WINDOWS_PER_RECORD, 2, SAMPLES_PER_WINDOW],
        "matrix_dtype": "float16",
        "labels_used_for_ssl": [],
        "elapsed_minutes": (time.time() - started) / 60,
    }
    (outdir / "JNU_PREPROCESS_QC.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
