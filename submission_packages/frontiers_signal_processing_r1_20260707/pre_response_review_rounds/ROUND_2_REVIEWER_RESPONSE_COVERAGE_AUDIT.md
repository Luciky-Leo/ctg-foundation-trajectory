# Round 2 - Reviewer Response Coverage Audit

This pass checks that each reviewer item has a response section and a posting-matrix status in the locked response file.

| request | topic | response section | matrix status | expectation |
|---|---|---:|---|---|
| R1-1 | Abstract completion | True | READY_TO_POST | complete result chain with SSL, recalibration, domain-shift boundary and conclusion |
| R1-2 | Figure 1 workflow labels | True | READY_TO_POST | 10-minute windows and channel masking |
| R1-3 | Supplementary figure naming | True | READY_TO_POST | S1/S2/S3 naming separated from main Figures 1-4 |
| R1-4 | Final-model rationale | True | READY_TO_POST | interpretability rationale and SSL-alone ablation boundary |
| R1-5 | pH endpoint rationale | True | READY_TO_POST | pH <7.15 rationale plus stricter sensitivity analyses |
| R1-6 | Reconstruction-loss instability | True | READY_TO_POST | training trace is inspection-only, not representation-quality proof |
| R1-7 | CTU SSL phenotype risk gradient | True | READY_TO_POST | S2D gradient included with representation-only boundary |
| R1-8 | AI text artifacts | True | READY_TO_POST | targeted text audit and repetition cleanup |
| R2-1 | Sample size and generalization | True | READY_TO_POST | 552-record limit and generalization boundary |
| R2-2 | Practical significance | True | READY_TO_POST | risk-enrichment, calibration and interpretability boundary |
| R2-3 | Recent CTG comparison | True | READY_WITH_CURRENT_CONTEXT_ONLY | bounded context, not direct head-to-head benchmark |
| R2-4 | Discussion length | True | READY_TO_POST | condensed discussion |
| R2-5 | Abbreviations | True | READY_TO_POST | CTG, FHR, UC, SSL, AUPRC, AUROC and ECE first-use definitions |
| R2-6 | Transformer hyperparameters | True | READY_TO_POST | Supplementary Table 11 |
| R2-7 | PatchTST rationale | True | READY_TO_POST | PatchTST selected pragmatically for long two-channel CTG windows |
