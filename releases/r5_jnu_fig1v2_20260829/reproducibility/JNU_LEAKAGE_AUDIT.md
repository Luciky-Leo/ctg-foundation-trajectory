# JNU and CTU Leakage Audit

Status: PASS

JNU labels were withheld from SSL; CTU folds were record-isolated; promotion was recorded before exploratory fixed-test access.

| Check | Status | Observed | Expected |
|---|---:|---|---|
| jnu_qc_status | PASS | PASS | PASS |
| jnu_archive_md5 | PASS | ac1cfcba2f1b3336596544211d776771 | published MD5 |
| jnu_license | PASS | CC BY 4.0 | CC BY 4.0 |
| jnu_record_count | PASS | 20769 | 20769 |
| jnu_readability | PASS | 1.0 | >=0.95 |
| jnu_patient_groups | PASS | 12606 | 12606 |
| jnu_windows | PASS | 62307 | 62307 |
| jnu_labels_withheld | PASS | [] | [] |
| jnu_index_rows | PASS | 62307 | 62307 |
| jnu_index_records | PASS | 20769 | 20769 |
| jnu_index_patients | PASS | 12606 | 12606 |
| three_windows_per_record | PASS | {3: 20769} | 3 each |
| patient_id_complete | PASS | 0 | 0 |
| ctu_development_only_folds | PASS | [np.int64(1), np.int64(2), np.int64(3)] | three folds; no test IDs |
| ctu_validation_once | PASS | {1} | {1} |
| development_probability_ids | PASS | 386 | 386 |
| fixed_probability_ids | PASS | 166 | 166 |
| promotion_fixed_test_closed | PASS | False | False |
| promotion_recorded_before_fixed_test | PASS | {'promotion_fixed_test_accessed': False, 'fixed_test_role': 'exploratory'} | promotion fixed_test_accessed=false; fixed-test completion flag documents exploratory use |
