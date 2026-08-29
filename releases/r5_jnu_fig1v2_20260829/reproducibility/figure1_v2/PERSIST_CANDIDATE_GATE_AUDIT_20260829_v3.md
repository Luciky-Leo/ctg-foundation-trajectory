# PERSIST Template Candidate Gate Audit

Generated: 2026-08-29 12:22:41

Overall status: `PASS`

Rule: PERSIST/HF candidate choice must be visual-semantic, source-code-first, and evidence-bound. A generic template can be useful, but it is not an HF capsule. A rejected HF reference is visual inspiration only, not a renderable candidate.

## Candidate Row Counts

| Status | Count |
|---|---:|
| `PASS` | 5 |
| `WARN` | 0 |
| `FAIL` | 0 |

## Panel Gate Summary

| Panel | Status | Candidates | Renderable | HF/PERSIST | Generic | Native | Findings | Action |
|---|---|---:|---:|---:|---:|---:|---|---|
| C | `PASS` | 3 | 2 | 2 | 0 | 0 | none | candidate_gate_can_proceed_to_user_choice_or_rendering |
| D | `PASS` | 2 | 1 | 1 | 0 | 0 | none | candidate_gate_can_proceed_to_user_choice_or_rendering |

## Candidate Findings

| Panel | Option | Candidate | Kind | Status | Renderable | Findings | Allowed label | Blocked label |
|---|---|---|---|---|---|---|---|---|
| C | C1 | `HF114_2025-11-17_a4a424a4` | `hf_capsule` | `PASS` | yes | none | HF/PERSIST source-code candidate only after source inspection | manuscript-ready HF/PERSIST until rendered, validated, quality-reviewed, and final-size selected |
| C | C2 | `HF132_2025-12-24_4aa08c0c` | `hf_capsule` | `PASS` | yes | none | HF/PERSIST source-code candidate only after source inspection | manuscript-ready HF/PERSIST until rendered, validated, quality-reviewed, and final-size selected |
| C | C3 | `HF145_2026-01-20_c1120b32` | `hf_capsule` | `PASS` | no | none | HF/PERSIST source-code candidate only after source inspection | manuscript-ready HF/PERSIST until rendered, validated, quality-reviewed, and final-size selected |
| D | D1 | `HF001_2025-04-30_886b26b9` | `hf_capsule` | `PASS` | yes | none | HF/PERSIST source-code candidate only after source inspection | manuscript-ready HF/PERSIST until rendered, validated, quality-reviewed, and final-size selected |
| D | D2 | `HF187_2026-04-09_31eb17f4` | `hf_capsule` | `PASS` | no | none | HF/PERSIST source-code candidate only after source inspection | manuscript-ready HF/PERSIST until rendered, validated, quality-reviewed, and final-size selected |

## Use

- `FAIL`: repair the candidate gate before rendering or presenting options.
- `WARN`: usable only with label downgrades, completed scoring, or explicit user choice.
- Generic candidates must be reported as generic PERSIST/template candidates, not HF.
- Native candidates must be reported as native workflows, not PERSIST/HF.
