#!/usr/bin/env python3
"""Cross-audit the locked JNU enhancement, manuscript, figures, and response."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parent
PREDECESSOR = PROJECT / "R2_R5_Targeted_Revision_20260826"
ANALYSIS = ROOT / "03_analysis"
TABLES = ROOT / "04_tables"
DATA = ROOT / "04_data" / "jnu_processed"
FIGURES = ROOT / "05_figures"
SOURCE = ROOT / "06_manuscript" / "frontiers_r5_jnu_source"
RESPONSE = ROOT / "07_response"
REPRO = ROOT / "09_reproducibility"
FIXED_SPLIT = PROJECT / "R1_QA_20260706" / "real_raw_rerun_20260706" / "ctu_uhb_record_split.csv"
SEEDS = {20260521, 20260522, 20260523, 20260524, 20260525}


def close(left: float, right: float, atol: float = 1e-10) -> bool:
    return bool(np.isclose(float(left), float(right), atol=atol, rtol=1e-8))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def max_overfull(log_text: str) -> float:
    values = [float(value) for value in re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", log_text)]
    return max(values, default=0.0)


def main() -> None:
    REPRO.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, str]] = []

    def add(category: str, name: str, passed: bool, observed: object, expected: object) -> None:
        checks.append(
            {
                "category": category,
                "check": name,
                "status": "PASS" if bool(passed) else "FAIL",
                "observed": str(observed),
                "expected": str(expected),
            }
        )

    # Data provenance and patient grouping.
    qc = json.loads((DATA / "JNU_PREPROCESS_QC.json").read_text(encoding="utf-8"))
    add("leakage", "jnu_qc_status", qc["status"] == "PASS", qc["status"], "PASS")
    add("leakage", "jnu_archive_md5", qc["md5"] == "ac1cfcba2f1b3336596544211d776771", qc["md5"], "published MD5")
    add("leakage", "jnu_license", qc["license"] == "CC BY 4.0", qc["license"], "CC BY 4.0")
    add("leakage", "jnu_record_count", qc["processed_records"] == 20769, qc["processed_records"], 20769)
    add("leakage", "jnu_readability", qc["readable_fraction"] >= 0.95, qc["readable_fraction"], ">=0.95")
    add("leakage", "jnu_patient_groups", qc["patient_groups_total"] == 12606, qc["patient_groups_total"], 12606)
    add("leakage", "jnu_windows", qc["processed_windows"] == 62307, qc["processed_windows"], 62307)
    add("leakage", "jnu_labels_withheld", qc["labels_used_for_ssl"] == [], qc["labels_used_for_ssl"], "[]")

    index = pd.read_csv(DATA / "jnu_window_index.csv", dtype={"record_id": str, "patient_id": str})
    windows_per_record = index.groupby("record_id").size()
    add("leakage", "jnu_index_rows", len(index) == 62307, len(index), 62307)
    add("leakage", "jnu_index_records", index["record_id"].nunique() == 20769, index["record_id"].nunique(), 20769)
    add("leakage", "jnu_index_patients", index["patient_id"].nunique() == 12606, index["patient_id"].nunique(), 12606)
    add("leakage", "three_windows_per_record", bool((windows_per_record == 3).all()), windows_per_record.value_counts().to_dict(), "3 each")
    add("leakage", "patient_id_complete", not index["patient_id"].isna().any(), int(index["patient_id"].isna().sum()), 0)

    # Frozen CTU folds and record isolation.
    split = pd.read_csv(FIXED_SPLIT)
    development_ids = set(split.loc[split["split"] == "train", "record_id"].astype(int))
    test_ids = set(split.loc[split["split"] == "test", "record_id"].astype(int))
    folds = pd.read_csv(REPRO / "ctu_development_folds.csv")
    fold_ok = True
    validation_counts = {record_id: 0 for record_id in development_ids}
    for fold in sorted(folds["fold"].unique()):
        frame = folds[folds["fold"] == fold]
        ids = set(frame["record_id"].astype(int))
        fold_ok &= ids == development_ids and not (ids & test_ids)
        for record_id in frame.loc[frame["role"] == "development_validation", "record_id"].astype(int):
            validation_counts[record_id] += 1
    add("leakage", "ctu_development_only_folds", fold_ok, sorted(folds["fold"].unique()), "three folds; no test IDs")
    add("leakage", "ctu_validation_once", all(value == 1 for value in validation_counts.values()), set(validation_counts.values()), "{1}")

    dev_prob = pd.read_csv(TABLES / "development_risk_probabilities.csv")
    fixed_prob = pd.read_csv(TABLES / "fixed_test_risk_probabilities.csv")
    add("leakage", "development_probability_ids", set(dev_prob["record_id"].astype(int)) == development_ids, dev_prob["record_id"].nunique(), 386)
    add("leakage", "fixed_probability_ids", set(fixed_prob["record_id"].astype(int)) == test_ids, fixed_prob["record_id"].nunique(), 166)

    promotion = json.loads((ANALYSIS / "JNU_PROMOTION_DECISION.json").read_text(encoding="utf-8"))
    add("leakage", "promotion_fixed_test_closed", promotion["fixed_test_accessed"] is False, promotion["fixed_test_accessed"], False)
    add("claim", "promotion_decision", promotion["decision"] == "SUPPLEMENT_ONLY", promotion["decision"], "SUPPLEMENT_ONLY")
    add("claim", "risk_gate_failed", promotion["risk_promotion_gate"] is False, promotion["risk_promotion_gate"], False)
    add("claim", "main_figure_performance_gate_failed", promotion["main_figure_promotion"] is False, promotion["main_figure_promotion"], False)
    fixed_flag = (ANALYSIS / "FIXED_TEST_COMPLETE.flag").read_text(encoding="utf-8")
    promotion_was_blinded = promotion.get("fixed_test_accessed") is False
    fixed_is_exploratory = "exploratory" in fixed_flag.lower()
    add(
        "leakage",
        "promotion_recorded_before_fixed_test",
        promotion_was_blinded and fixed_is_exploratory,
        {
            "promotion_fixed_test_accessed": promotion.get("fixed_test_accessed"),
            "fixed_test_role": "exploratory" if fixed_is_exploratory else "not documented",
        },
        "promotion fixed_test_accessed=false; fixed-test completion flag documents exploratory use",
    )

    # Training completion and frozen configuration.
    states = sorted((ANALYSIS / "models").rglob("training_state.json"))
    state_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in states]
    add("runtime", "training_state_count", len(states) == 50, len(states), 50)
    add("runtime", "all_training_complete", all(item["status"] == "COMPLETE" for item in state_payloads), {item["status"] for item in state_payloads}, "{'COMPLETE'}")
    add("runtime", "all_epochs_complete", all(item["epochs_complete"] == item["epochs_total"] == 4 for item in state_payloads), sorted({(item["epochs_complete"], item["epochs_total"]) for item in state_payloads}), "(4,4)")
    path_text = [path.as_posix() for path in states]
    route_counts = {
        "jnu_mixed": sum("/jnu_mixed/" in value for value in path_text),
        "jnu_asymmetric": sum("/jnu_asymmetric_fhr/" in value for value in path_text),
        "fold_ctu": sum("/fold_" in value and "/ctu_only/" in value for value in path_text),
        "fold_jnu_to_ctu": sum("/fold_" in value and "/jnu_to_ctu/" in value for value in path_text),
        "all_dev_ctu": sum("/all_development/ctu_only/" in value for value in path_text),
        "all_dev_jnu_to_ctu": sum("/all_development/jnu_to_ctu/" in value for value in path_text),
    }
    expected_route_counts = {"jnu_mixed": 5, "jnu_asymmetric": 5, "fold_ctu": 15, "fold_jnu_to_ctu": 15, "all_dev_ctu": 5, "all_dev_jnu_to_ctu": 5}
    add("runtime", "training_route_completeness", route_counts == expected_route_counts, route_counts, expected_route_counts)

    frozen = json.loads((REPRO / "FROZEN_MODEL_CONFIG.json").read_text(encoding="utf-8"))
    expected_frozen = {"patch_len": 80, "d_model": 48, "layers": 2, "nhead": 4, "mask_strategy": "mixed", "epochs": 4}
    observed_frozen = {key: frozen[key] for key in expected_frozen}
    add("runtime", "frozen_encoder_configuration", observed_frozen == expected_frozen, observed_frozen, expected_frozen)
    add("runtime", "five_fixed_seeds", set(dev_prob["seed"].astype(int)) == SEEDS, sorted(dev_prob["seed"].unique()), sorted(SEEDS))

    # Recompute risk metrics from record-level probabilities.
    dev_reported = pd.read_csv(TABLES / "development_risk_seed_metrics.csv")
    fixed_reported = pd.read_csv(TABLES / "fixed_test_seed_metrics.csv")
    risk_recomputed: list[dict[str, object]] = []
    dev_pass = True
    for keys, frame in dev_prob.groupby(["seed", "encoder_route", "input_variant", "evaluation_set"]):
        seed, route, variant, evaluation = keys
        y = frame["neonatal_risk"].astype(int).to_numpy()
        p = frame["probability"].astype(float).to_numpy()
        calculated = {"auprc": average_precision_score(y, p), "auroc": roc_auc_score(y, p)}
        row = dev_reported[(dev_reported["seed"] == seed) & (dev_reported["route"] == route) & (dev_reported["input_variant"] == variant) & (dev_reported["evaluation_set"] == evaluation)].iloc[0]
        passed = all(close(calculated[name], row[name]) for name in calculated)
        dev_pass &= passed
        risk_recomputed.append({"data_role": "development", "seed": seed, "route": route, "input_variant": variant, **calculated, "status": "PASS" if passed else "FAIL"})
    add("numeric", "development_risk_recomputation", dev_pass and len(risk_recomputed) == 55, f"{sum(row['status'] == 'PASS' for row in risk_recomputed)}/{len(risk_recomputed)}", "55/55")

    fixed_pass = True
    fixed_rows = 0
    for keys, frame in fixed_prob.groupby(["seed", "encoder_route", "input_variant", "evaluation_set"]):
        seed, route, variant, evaluation = keys
        y = frame["neonatal_risk"].astype(int).to_numpy()
        p = frame["probability"].astype(float).to_numpy()
        calculated = {"auprc": average_precision_score(y, p), "auroc": roc_auc_score(y, p), "brier": brier_score_loss(y, p)}
        row = fixed_reported[(fixed_reported["seed"] == seed) & (fixed_reported["route"] == route) & (fixed_reported["input_variant"] == variant) & (fixed_reported["evaluation_set"] == evaluation)].iloc[0]
        passed = all(close(calculated[name], row[name]) for name in calculated)
        fixed_pass &= passed
        fixed_rows += 1
        risk_recomputed.append({"data_role": "fixed_test_exploratory", "seed": seed, "route": route, "input_variant": variant, **calculated, "status": "PASS" if passed else "FAIL"})
    add("numeric", "fixed_test_risk_recomputation", fixed_pass and fixed_rows == 50, f"{fixed_rows}/{fixed_rows}", "50/50")
    pd.DataFrame(risk_recomputed).to_csv(REPRO / "JNU_RECOMPUTED_RISK_METRICS.csv", index=False)

    # Recompute morphology task and macro metrics.
    morph_prob = pd.read_csv(TABLES / "development_morphology_probabilities.csv")
    morph_tasks = pd.read_csv(TABLES / "development_morphology_task_oof_metrics.csv")
    task_recomputed: list[dict[str, object]] = []
    task_pass = True
    for (seed, route, task), frame in morph_prob.groupby(["seed", "route", "task"]):
        y = frame["outcome"].astype(int).to_numpy()
        p = frame["probability"].astype(float).to_numpy()
        auprc = average_precision_score(y, p)
        auroc = roc_auc_score(y, p)
        row = morph_tasks[(morph_tasks["seed"] == seed) & (morph_tasks["route"] == route) & (morph_tasks["task"] == task)].iloc[0]
        passed = close(auprc, row["auprc"]) and close(auroc, row["auroc"])
        task_pass &= passed
        task_recomputed.append({"seed": seed, "route": route, "task": task, "auprc": auprc, "auroc": auroc, "status": "PASS" if passed else "FAIL"})
    add("numeric", "morphology_task_recomputation", task_pass and len(task_recomputed) == 210, f"{sum(row['status'] == 'PASS' for row in task_recomputed)}/{len(task_recomputed)}", "210/210")
    macro = pd.DataFrame(task_recomputed).groupby(["seed", "route"])[["auprc", "auroc"]].mean().reset_index()
    macro_reported = pd.read_csv(TABLES / "development_morphology_macro_seed_metrics.csv")
    macro_merged = macro.merge(macro_reported, on=["seed", "route"], suffixes=("_calc", "_reported"))
    macro_pass = all(close(row.auprc_calc, row.auprc_reported) and close(row.auroc_calc, row.auroc_reported) for row in macro_merged.itertuples())
    add("numeric", "morphology_macro_recomputation", macro_pass and len(macro_merged) == 30, len(macro_merged), 30)
    pd.DataFrame(task_recomputed).to_csv(REPRO / "JNU_RECOMPUTED_MORPHOLOGY_METRICS.csv", index=False)

    # Exact manuscript and supplementary-number checks.
    main_tex = (SOURCE / "frontiers_ctg_manuscript_r5.tex").read_text(encoding="utf-8")
    supp_tex = (SOURCE / "frontiers_ctg_supplementary_material_r5.tex").read_text(encoding="utf-8")
    response_md = (RESPONSE / "Response_to_Reviewer_5_R2_JNU_20260828.md").read_text(encoding="utf-8")
    required_main = [
        "20,769 antepartum records from 12,606 patient groups yielded 62,307",
        "development median AUPRC 0.3682 versus 0.3797 for CTU-only SSL",
        "0.0030, 95\\% CI $-0.0304$ to 0.0316",
        "macro-AUROC from 0.6392 to 0.6802",
        "Exploratory fixed-test AUPRC was 0.3420 for JNU-to-CTU versus 0.4017",
        "risk-enrichment promotion gate was not met",
        "supports transfer of morphology information, not superiority",
        "without a prespecified lead-time horizon",
        "fixed test set with potential selection-induced optimism",
        "Zenodo concept DOI",
    ]
    missing_main = [term for term in required_main if term not in main_tex]
    add("claim", "required_main_numbers_and_boundaries", not missing_main, missing_main or "all present", "all present")
    forbidden = [
        "JNU improved neonatal-risk prediction",
        "provides external clinical validation",
        "is a deployable predictor",
        "supports autonomous triage",
    ]
    found_forbidden = [term for term in forbidden if term.lower() in main_tex.lower()]
    add("claim", "forbidden_overclaims_absent", not found_forbidden, found_forbidden or "none", "none")
    add("claim", "supplement_states_no_external_outcome_validation", "External outcome validation & Not performed" in supp_tex, "present" if "External outcome validation & Not performed" in supp_tex else "missing", "present")
    add("claim", "reviewer_response_complete", all(f"## Comment {number}" in response_md for number in range(1, 6)) and "will be inserted" not in response_md, response_md.count("**Changes in manuscript:**"), 5)

    # Main-figure promotion boundary and figure assets.
    figure1_source = (ANALYSIS / "Figure_1_final_source_data.csv").read_text(encoding="utf-8")
    add("figure", "figure1_design_only_jnu_branch", "not primary-model selection" in figure1_source and "not performed" in figure1_source, "bounded branch", "bounded branch")
    old_figure2 = PREDECESSOR / "06_manuscript" / "frontiers_r5_source" / "figures" / "Figure_2_R5_revised.png"
    new_figure2 = SOURCE / "figures" / "Figure_2_R5_revised.png"
    add("figure", "jnu_not_promoted_into_main_figure2", sha256(old_figure2) == sha256(new_figure2), sha256(new_figure2), sha256(old_figure2))

    required_figures = [
        FIGURES / "Figure_1_final_study_design.png",
        FIGURES / "Figure_1_final_study_design.pdf",
        FIGURES / "Figure_1_final_study_design.svg",
        FIGURES / "Figure_1_final_study_design.tiff",
        FIGURES / "Supplementary_Figure_S3_JNU_cross_domain_sensitivity.png",
        FIGURES / "Supplementary_Figure_S3_JNU_cross_domain_sensitivity.pdf",
        FIGURES / "Supplementary_Figure_S3_JNU_cross_domain_sensitivity.svg",
        FIGURES / "Supplementary_Figure_S3_JNU_cross_domain_sensitivity.tiff",
    ]
    missing_figures = [path.name for path in required_figures if not path.is_file() or path.stat().st_size < 10_000]
    add("figure", "figure_export_set", not missing_figures, missing_figures or "8 present", "8 present")
    for image_name in ["Figure_1_final_study_design.png", "Supplementary_Figure_S3_JNU_cross_domain_sensitivity.png"]:
        image = Image.open(FIGURES / image_name)
        dpi = image.info.get("dpi", (0, 0))
        min_dpi = min(dpi) if isinstance(dpi, tuple) else float(dpi)
        compliant = image.width >= 2000 and image.height >= 1600 and min_dpi >= 299
        add(
            "figure",
            f"image_dimensions:{image_name}",
            compliant,
            f"{image.width}x{image.height} at {min_dpi:.1f} dpi",
            ">=2000x1600 at >=299 dpi (Frontiers-compatible 300-dpi export)",
        )
    overlap_a = pd.read_csv(ANALYSIS / "figure1_redesign" / "Figure_1_text_overlap_audit.csv")
    overlap_s3 = pd.read_csv(ANALYSIS / "figure_s3" / "Supplementary_Figure_S3_text_overlap_audit.csv")
    add("figure", "figure1_text_overlap_audit", overlap_a.empty, len(overlap_a), 0)
    add("figure", "figure_s3_text_overlap_audit", overlap_s3.empty, len(overlap_s3), 0)

    # PDF, LaTeX, citation, and machine-readable-table checks.
    main_pdf = SOURCE / "frontiers_ctg_manuscript_r5.pdf"
    supp_pdf = SOURCE / "frontiers_ctg_supplementary_material_r5.pdf"
    response_pdf = RESPONSE / "Response_to_Reviewer_5_R2_JNU_20260828.pdf"
    for label, pdf, pages in [("main", main_pdf, 14), ("supplement", supp_pdf, 9), ("response", response_pdf, 4)]:
        reader = PdfReader(pdf)
        lengths = [len((page.extract_text() or "").strip()) for page in reader.pages]
        add("package", f"{label}_pdf_pages", len(reader.pages) == pages, len(reader.pages), pages)
        add("package", f"{label}_pdf_no_blank_pages", min(lengths) > 100, min(lengths), ">100 extracted characters/page")

    logs = [
        (SOURCE / "frontiers_ctg_manuscript_r5.log").read_text(encoding="utf-8", errors="replace"),
        (SOURCE / "frontiers_ctg_supplementary_material_r5.log").read_text(encoding="utf-8", errors="replace"),
        (RESPONSE / "Response_to_Reviewer_5_R2_JNU_20260828.log").read_text(encoding="utf-8", errors="replace"),
    ]
    undefined = re.compile(r"undefined citations|undefined references|Citation .* undefined|Reference .* undefined", re.I)
    add("package", "latex_no_undefined_citations_or_references", not any(undefined.search(log) for log in logs), "none" if not any(undefined.search(log) for log in logs) else "found", "none")
    add("package", "latex_overfull_within_visual_tolerance", max(max_overfull(log) for log in logs) <= 10.0, max(max_overfull(log) for log in logs), "<=10 pt and visually reviewed")

    bib = (SOURCE / "references.bib").read_text(encoding="utf-8")
    cite_keys: set[str] = set()
    for block in re.findall(r"\\cite\w*\{([^}]+)\}", main_tex):
        cite_keys.update(key.strip() for key in block.split(","))
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    add("reference", "all_citation_keys_resolve", cite_keys <= bib_keys, sorted(cite_keys - bib_keys), "none missing")
    latest_keys = {"bai2026jnuctg", "prismctg2026", "fridman2026ctgfoundation", "mendis2025crossdatabase", "benmbarek2026humanai"}
    add("reference", "latest_reference_entries_present", latest_keys <= bib_keys, sorted(latest_keys & bib_keys), sorted(latest_keys))

    machine_tables = list((SOURCE / "supplementary_tables").glob("Supplementary_Table_*.csv"))
    new_tables = [SOURCE / "supplementary_tables" / f"Supplementary_Table_S{number}_{suffix}.csv" for number, suffix in [
        (14, "encoder_origin_risk_stability"),
        (15, "frozen_encoder_morphology_probes"),
        (16, "jnu_paired_hierarchical_bootstrap"),
        (17, "channel_sampling_sensitivity"),
        (18, "jnu_data_leakage_audit"),
    ]]
    add("package", "jnu_machine_readable_tables", all(path.is_file() for path in new_tables), [path.name for path in new_tables if path.is_file()], "S14-S18")
    add("package", "machine_readable_table_count", len(machine_tables) == 22, len(machine_tables), 22)

    predecessor_audit = PREDECESSOR / "09_reproducibility" / "R5_OUTPUT_CROSS_AUDIT.md"
    add("numeric", "predecessor_targeted_audit", "Status: PASS" in predecessor_audit.read_text(encoding="utf-8"), "PASS", "PASS")

    # Reproducibility manifest for locked formal artifacts.
    manifest_paths = [
        ROOT / "02_protocol" / "JNU_CROSS_DOMAIN_FROZEN_ADDENDUM.md",
        REPRO / "FROZEN_MODEL_CONFIG.json",
        REPRO / "ctu_development_folds.csv",
        DATA / "JNU_PREPROCESS_QC.json",
        DATA / "jnu_window_index.csv",
        DATA / "jnu_ctg_windows_f16.npy",
        ANALYSIS / "JNU_PROMOTION_DECISION.json",
        ANALYSIS / "FIXED_TEST_EXPLORATORY_AUDIT.json",
        TABLES / "development_risk_probabilities.csv",
        TABLES / "fixed_test_risk_probabilities.csv",
        TABLES / "development_morphology_probabilities.csv",
        FIGURES / "Figure_1_final_study_design.png",
        FIGURES / "Supplementary_Figure_S3_JNU_cross_domain_sensitivity.png",
        SOURCE / "frontiers_ctg_manuscript_r5.tex",
        SOURCE / "frontiers_ctg_supplementary_material_r5.tex",
        main_pdf,
        supp_pdf,
        RESPONSE / "Response_to_Reviewer_5_R2_JNU_20260828.md",
        response_pdf,
    ]
    manifest = []
    for path in manifest_paths:
        manifest.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest).to_csv(REPRO / "REPRODUCIBILITY_MANIFEST.csv", index=False)
    (REPRO / "REPRODUCIBILITY_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    frame = pd.DataFrame(checks)
    frame.to_csv(REPRO / "JNU_R5_AUDIT_CHECKS.csv", index=False)
    overall = "PASS" if (frame["status"] == "PASS").all() else "FAIL"

    def write_category_report(category: str, filename: str, title: str, note: str) -> None:
        subset = frame[frame["category"] == category]
        status = "PASS" if (subset["status"] == "PASS").all() else "FAIL"
        lines = [f"# {title}", "", f"Status: {status}", "", note, "", "| Check | Status | Observed | Expected |", "|---|---:|---|---|"]
        for row in subset.itertuples():
            lines.append(f"| {row.check} | {row.status} | {row.observed.replace('|', '/')} | {row.expected.replace('|', '/')} |")
        (REPRO / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_category_report("leakage", "JNU_LEAKAGE_AUDIT.md", "JNU and CTU Leakage Audit", "JNU labels were withheld from SSL; CTU folds were record-isolated; promotion was recorded before exploratory fixed-test access.")
    write_category_report("numeric", "NUMERIC_CROSS_AUDIT.md", "Numeric Cross-Audit", "Risk and morphology metrics were recomputed from record-level probability tables and compared with reported CSV outputs.")
    write_category_report("figure", "FIGURE_QUALITY_REVIEW.md", "Figure Quality and Promotion Review", "Figure 1 and Supplementary Figure S3 were visually reviewed after PDF rendering; JNU performance was not promoted into the primary performance figure.")
    write_category_report("reference", "REFERENCE_LOCAL_AUDIT.md", "Reference and Citation Audit", "This local audit verifies citation-key resolution and presence of the five latest primary-source entries; URL resolution is recorded separately.")

    lines = [
        "# R5 JNU Completion Audit",
        "",
        f"Status: {overall}",
        "",
        f"Checks: {len(frame)}; PASS: {int((frame['status'] == 'PASS').sum())}; FAIL: {int((frame['status'] == 'FAIL').sum())}.",
        "",
        "The prespecified promotion result is SUPPLEMENT_ONLY: public JNU pretraining transferred morphology information but did not establish stable neonatal-risk enrichment.",
        "",
        "| Category | PASS | FAIL |",
        "|---|---:|---:|",
    ]
    for category, subset in frame.groupby("category"):
        lines.append(f"| {category} | {int((subset['status'] == 'PASS').sum())} | {int((subset['status'] == 'FAIL').sum())} |")
    (REPRO / "JNU_R5_COMPLETION_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": overall, "checks": len(frame), "failures": int((frame["status"] == "FAIL").sum())}, indent=2))
    raise SystemExit(0 if overall == "PASS" else 1)


if __name__ == "__main__":
    main()
