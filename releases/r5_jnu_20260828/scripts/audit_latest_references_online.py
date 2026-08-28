#!/usr/bin/env python3
"""Verify metadata for the references added in the R5 JNU revision."""

from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "09_reproducibility"
HEADERS = {"User-Agent": "CTG-R5-reference-audit/1.0 mailto:luff94@163.com"}


def get_json(url: str) -> dict:
    response = requests.get(url, timeout=60, headers=HEADERS)
    response.raise_for_status()
    return response.json()


def normalize(value: str) -> str:
    return " ".join(value.lower().replace("{", "").replace("}", "").split())


rows: list[dict[str, str]] = []

zenodo = get_json("https://zenodo.org/api/records/21800730")
zenodo_title = zenodo["metadata"]["title"]
rows.append(
    {
        "key": "bai2026jnuctg",
        "source": "Zenodo API",
        "identifier": "10.5281/zenodo.21800730",
        "observed_title": zenodo_title,
        "status": "PASS" if "jnu-ctg" in normalize(zenodo_title) else "FAIL",
        "evidence_url": "https://zenodo.org/api/records/21800730",
    }
)

github = get_json("https://api.github.com/repos/sfwon17/Prism-CTG")
rows.append(
    {
        "key": "prismctg2026-code",
        "source": "GitHub API",
        "identifier": github["full_name"],
        "observed_title": github.get("description") or github["name"],
        "status": "PASS" if github["full_name"].lower() == "sfwon17/prism-ctg" else "FAIL",
        "evidence_url": github["html_url"],
    }
)

arxiv_response = requests.get(
    "https://export.arxiv.org/api/query?id_list=2605.02917,2601.06149",
    timeout=60,
    headers=HEADERS,
)
arxiv_response.raise_for_status()
namespace = {"atom": "http://www.w3.org/2005/Atom"}
entries = {}
for entry in ET.fromstring(arxiv_response.text).findall("atom:entry", namespace):
    arxiv_id = entry.findtext("atom:id", default="", namespaces=namespace).rstrip("/").split("/")[-1].split("v")[0]
    entries[arxiv_id] = " ".join(entry.findtext("atom:title", default="", namespaces=namespace).split())

for key, arxiv_id, expected in [
    ("prismctg2026", "2605.02917", "PRISM-CTG"),
    ("fridman2026ctgfoundation", "2601.06149", "foundation model"),
]:
    observed = entries.get(arxiv_id, "")
    rows.append(
        {
            "key": key,
            "source": "arXiv API",
            "identifier": arxiv_id,
            "observed_title": observed,
            "status": "PASS" if normalize(expected) in normalize(observed) else "FAIL",
            "evidence_url": f"https://arxiv.org/abs/{arxiv_id}",
        }
    )

for key, doi, expected in [
    ("mendis2025crossdatabase", "10.1109/JTEHM.2025.3548401", "Cross-database evaluation"),
    ("benmbarek2026humanai", "10.1038/s41746-026-02556-y", "Randomised study of human machine collaboration"),
]:
    payload = get_json(f"https://api.crossref.org/works/{doi}")["message"]
    observed = payload["title"][0]
    rows.append(
        {
            "key": key,
            "source": "Crossref API",
            "identifier": doi,
            "observed_title": observed,
            "status": "PASS" if normalize(expected) in normalize(observed) else "FAIL",
            "evidence_url": f"https://doi.org/{doi}",
        }
    )

OUT.mkdir(parents=True, exist_ok=True)
csv_path = OUT / "REFERENCE_ONLINE_AUDIT.csv"
with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

status = "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"
md_path = OUT / "REFERENCE_ONLINE_AUDIT.md"
lines = [
    "# Latest-reference online audit",
    "",
    f"Status: {status}",
    "",
    "Official metadata endpoints were queried on 2026-08-28.",
    "",
    "| Key | Source | Identifier | Status |",
    "|---|---|---|---|",
]
lines.extend(f"| {row['key']} | {row['source']} | {row['identifier']} | {row['status']} |" for row in rows)
lines.extend(["", "Machine-readable titles and evidence URLs are retained in `REFERENCE_ONLINE_AUDIT.csv`."])
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(json.dumps({"status": status, "records": len(rows), "csv": str(csv_path)}, indent=2))
raise SystemExit(0 if status == "PASS" else 1)
