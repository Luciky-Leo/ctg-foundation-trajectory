# Response to Reviewers - R1 Interactive Forum Pack

Manuscript: Self-supervised biomedical signal representation learning from intrapartum cardiotocography for neonatal risk enrichment

Journal: Frontiers in Signal Processing

Revision pack date: 07 July 2026
Generated: 2026-07-07T00:08:22.704350+00:00

## Use Note

This is the finalized locked-replay interactive-response pack. The SSL/final numeric replies use the seeded locked replay route and no longer retain the unused alternative route.

## Current Verification Snapshot

- Readiness status: `READY_FOR_FRONTIERS_RESUBMISSION_AFTER_LOCKED_REPLAY`
- Gate counts: `{"PASS": 53, "WARN": 3, "FAIL": 1}`
- Upload blockers: none remaining after locked-replay synchronization.
- CNS blockers: G04_external_outcome_validation_boundary (CNS-tier only; not a Frontiers R1 blocker)
- Text compliance: `R1_TEXT_COMPLIANCE_AUDIT_PASS`; abbreviation_fail_rows=0; ai_artifact_fail_rows=0; reviewer_request_rows=8
- Response rows: 15 total; 15 ready-to-post rows after locked-replay finalization.

## Executive Summary For Editor

We thank both reviewers for their constructive minor-revision requests. We completed the abstract, redesigned Figure 1 as a workflow-first schematic, corrected the 10-minute/channel-masking labels, separated Supplementary Figures S1/S2, added endpoint and sample-size/generalization rationale, clarified the practical risk-enrichment boundary, added bounded recent CTG model context, defined abbreviations at first use, added Supplementary Table 11 for transformer hyperparameters and audited the manuscript for stock AI-like phrasing. We also preserve a clear boundary that external CTG datasets are representation/domain-shift evidence only, not external neonatal-outcome validation.

## Posting Readiness Matrix

| Comment | Status | Route dependency | Final action |
|---|---|---|---|
| R1-1 | `READY_TO_POST` | LOCKED_REPLAY_FINALIZED | Abstract and upload text now use seeded locked-replay values and bounded SSL wording. |
| R1-2 | `READY_TO_POST` | NONE | No additional action unless Figure 1 is altered again. |
| R1-3 | `READY_TO_POST` | NONE | No additional action. |
| R1-4 | `READY_TO_POST` | LOCKED_REPLAY_FINALIZED | Final-model rationale now states that SSL-containing AUPRC-difference intervals cross zero and that the final model is retained for interpretation/recalibration. |
| R1-5 | `READY_TO_POST` | NONE | No additional action. |
| R1-6 | `READY_TO_POST` | NONE | No additional action. |
| R1-7 | `READY_TO_POST` | LOCKED_REPLAY_FINALIZED | CTU-UHB source-cohort gradient wording is retained with explicit representation-only and no-external-outcome-validation boundaries. |
| R1-8 | `READY_TO_POST` | NONE | Repeat the audit after any route propagation edit. |
| R2-1 | `READY_TO_POST` | NONE | No additional action. |
| R2-2 | `READY_TO_POST` | LOCKED_REPLAY_FINALIZED | Practical significance wording is bounded to retrospective enrichment, calibration and interpretability rather than deployable screening or stable AUPRC gain. |
| R2-3 | `READY_WITH_CURRENT_CONTEXT_ONLY` | CTGDL_WORDING_PENDING | Use current context-only wording unless Claude approves CTGDL response-only wording. |
| R2-4 | `READY_TO_POST` | NONE | No additional action. |
| R2-5 | `READY_TO_POST` | NONE | Repeat abbreviation audit after route propagation. |
| R2-6 | `READY_TO_POST` | NONE | No additional action. |
| R2-7 | `READY_TO_POST` | NONE | No additional action. |

## Reviewer 1

### R1-1. Abstract completion

**Reviewer request.** Complete the abstract/result chain that ended mid-sentence.

**Response.** Corrected. The abstract and upload text now contain a complete result chain: signal-only baseline, seeded locked-replay SSL-containing model result, phenotype-augmented final model rationale, recalibration, external-representation boundary and conclusion. We adopted the seeded locked replay as the reproducibility basis: signal-only AUPRC was 0.3366, signal+SSL AUPRC was 0.3765, and the signal-plus-SSL-plus-signal-phenotype model reached AUPRC 0.4016. Because both SSL-containing AUPRC-difference confidence intervals versus signal-only features crossed zero, the revised abstract frames SSL as an interpretable representation, phenotype-enrichment and recalibration layer rather than as a statistically resolved AUPRC-improvement claim.

**Verification.** Status `READY_TO_POST`; evidence status `PASS`; evidence: Frontiers_Signal_Processing_Submission_Package_20260601/01_manuscript_frontiers_full/Frontiers_Original_Research_Manuscript.docx.

### R1-2. Figure 1 workflow labels

**Reviewer request.** Correct 60-minute/mixed-masking labels and improve workflow figure.

**Response.** Corrected and redesigned. Figure 1 is now workflow-first: Panel A foregrounds open-data roles, the leakage-controlled 386/166 record split, 10-minute FHR/UC windows, downstream risk/phenotype/recalibration analyses and the external representation-only boundary. Panel B is a smaller SSL method inset and now states 10-minute windows and channel masking for the final model; Panel C combines cohort scale and outcome balance. The previous evidence-map bookkeeping panel was removed from the main figure.

**Verification.** Status `READY_TO_POST`; evidence status `PASS`; evidence: CTG_Foundation_Trajectory/results/manuscript/source_data/Figure_1_source_data.csv; Frontiers_Signal_Processing_Submission_Package_20260601/04_figures_images/Figure_1_study_design_pipeline.jpg.

### R1-3. Supplementary figure naming

**Reviewer request.** Correct legacy main-figure numbering for the two supplementary figures.

**Response.** Corrected. The active package separates main Figures 1-4 from Supplementary Figures S1/S2. The supplementary figure files and source-data references now use Supplementary Figure S1 and Supplementary Figure S2 naming.

**Verification.** Status `READY_TO_POST`; evidence status `PASS`; evidence: Frontiers_Signal_Processing_Submission_Package_20260601/05_supplementary_files/supplementary_figures/Supplementary_Figure_S1_model_selection_and_sensitivity.jpg; Frontiers_Signal_Processing_Submission_Package_20260601/05_supplementary_files/supplementary_figures/Supplementary_Figure_S2_external_domain_shift.jpg.

### R1-4. Final model rationale and SSL ablation

**Reviewer request.** Justify final model despite signal+SSL being marginally higher; note SSL-alone underperforms.

**Response.** Revised. We now treat SSL-alone as an ablation and explicitly state that the phenotype-augmented final model is retained for interpretability, phenotype inspection and recalibration rather than because it establishes a statistically resolved held-out AUPRC improvement. The originally submitted values noted by the reviewer (signal+SSL AUPRC 0.4432 and final AUPRC 0.4324) came from an unseeded run that we could not exactly reproduce; to guarantee reproducibility, we re-ran the full pipeline with a fixed seed and all values in this revision correspond to that locked replay. Under the locked replay, SSL embeddings alone were insufficient as a stand-alone predictor (AUPRC 0.2125), signal+SSL reached AUPRC 0.3765, and the final signal-plus-SSL-plus-signal-phenotype model reached AUPRC 0.4016. The AUPRC-difference confidence intervals versus signal-only features crossed zero, so SSL is presented as an interpretive and phenotype-enrichment layer rather than as an independent discrimination-gain result.

**Verification.** Status `READY_TO_POST`; evidence status `PASS`; evidence: Frontiers_Signal_Processing_Submission_Package_20260601/01_manuscript_frontiers_full/Frontiers_Original_Research_Manuscript.docx.

### R1-5. pH <7.15 endpoint rationale

**Reviewer request.** Justify pH <7.15 and relate to stricter thresholds.

**Response.** Added. The Methods now explain that pH <7.15 was selected as a broader at-risk acidemia endpoint to preserve event count for an open-data risk-enrichment study, while stricter pH-only thresholds of <7.10 and <7.05 are reported as sensitivity analyses. The text also states that this threshold does not replace severe metabolic-acidosis definitions.

**Verification.** Status `READY_TO_POST`; evidence status `PASS`; evidence: Frontiers_Signal_Processing_Submission_Package_20260601/01_manuscript_frontiers_full/Frontiers_Original_Research_Manuscript.docx.

### R1-6. Reconstruction-loss instability

**Reviewer request.** Explain whether unstable reconstruction-loss trace affects representation quality.

**Response.** Added. The manuscript and Supplementary Figure S1 caption now clarify that the oscillating reconstruction-loss trace was used for training-dynamics inspection only. Model selection and interpretation rely on fixed held-out embeddings, phenotype structure and downstream validation, so the unstable reconstruction trace is not used as stand-alone representation-quality evidence.

**Verification.** Status `READY_TO_POST`; evidence status `PASS`; evidence: Frontiers_Signal_Processing_Submission_Package_20260601/06_upload_text/Response_to_Reviewers_R1_20260707_route_pending.md.

### R1-7. CTU SSL phenotype risk gradient

**Reviewer request.** Mention the rising CTU SSL-phenotype risk gradient shown in S2D.

**Response.** Added with boundary language. The external representation section now mentions the CTU-UHB SSL-phenotype risk gradient shown in Supplementary Figure S2D while preserving the boundary that CTGDL/FHRMA are representation-transfer/domain-shift evidence only and not external outcome validation. This statement is used as source-cohort phenotype-ordering evidence only and is not presented as an SSL discrimination-gain or external outcome-validation claim.

**Verification.** Status `READY_TO_POST`; evidence status `PASS`; evidence: Frontiers_Signal_Processing_Submission_Package_20260601/06_upload_text/Response_to_Reviewers_R1_20260707_route_pending.md.

### R1-8. AI text artifacts and repetitions

**Reviewer request.** Remove AI-generated text artifacts and repetitions.

**Response.** Revised and audited. We performed a targeted pass for stock AI-like phrases and repetitive prose using the same restricted phrase list as the R1 text-compliance audit. The audit reports zero critical AI-artifact hits and confirms that the active manuscript, official LaTeX and upload text remain synchronized for the corrected abbreviation definitions.

**Verification.** Status `READY_TO_POST`; evidence status `PASS`; evidence: R1_QA_20260707/r1_text_compliance_audit_20260707/R1_TEXT_COMPLIANCE_AUDIT_20260707.md.


## Reviewer 2

### R2-1. Limited sample size and generalization

**Reviewer request.** Discuss the impact of 552 records on model generalization.

**Response.** Added and strengthened. The manuscript now states the 552-record cohort size, limited event count and resulting generalization boundary. The revised framing describes the study as open-data retrospective risk enrichment and representation analysis rather than a deployable clinical prediction system.

**Verification.** Status `READY_TO_POST`; evidence status `PASS`; evidence: Frontiers_Signal_Processing_Submission_Package_20260601/01_manuscript_frontiers_full/Frontiers_Original_Research_Manuscript.docx.

### R2-2. Practical clinical significance

**Reviewer request.** Clarify practical significance and novelty of AUPRC gain.

**Response.** Revised and bounded. The manuscript now interprets AUPRC in terms of retrospective risk enrichment over event prevalence and explicitly states that local recalibration and external outcome validation would be required before clinical use. Under the locked replay, the SSL-containing AUPRC-difference confidence intervals cross zero, so the practical contribution is framed as an open-data signal-processing workflow for interpretable representation learning, phenotype enrichment, calibration and domain-shift assessment, not as a deployable screening tool or a stable AUPRC-gain claim.

**Verification.** Status `READY_TO_POST`; evidence status `PASS`; evidence: Frontiers_Signal_Processing_Submission_Package_20260601/01_manuscript_frontiers_full/Frontiers_Original_Research_Manuscript.docx.

### R2-3. Recent CTG deep-learning/foundation-model comparison

**Reviewer request.** Add quantitative comparison with recent CTG models.

**Response.** Added as bounded context. The revision includes quantitative source-reported context from recent CTG deep-learning and foundation/pretraining work, including Park et al.'s peer-reviewed supervised CTG interpretation study and emerging CTG pretraining resources. We do not present these as direct head-to-head benchmarks because cohorts, labels, windows and validation designs differ. A same-CTU CTGDL technical pilot was run after revision QA, but it is negative/boundary evidence only and is not promoted to active head-to-head SOTA wording unless Claude/user explicitly approves that scientific language.

**Verification.** Status `READY_WITH_CURRENT_CONTEXT_ONLY`; evidence status `PASS`; evidence: Frontiers_Signal_Processing_Submission_Package_20260601/06_upload_text/Response_to_Reviewers_R1_20260707_route_pending.md.

### R2-4. Discussion length

**Reviewer request.** Condense the Discussion.

**Response.** Condensed. We tightened the Discussion and removed repeated framing while preserving the reviewer-relevant limitations, PatchTST rationale, clinical boundary and external validation boundary.

**Verification.** Status `READY_TO_POST`; evidence status `PASS`; evidence: Frontiers_Signal_Processing_Submission_Package_20260601/06_upload_text/Response_to_Reviewers_R1_20260707_route_pending.md.

### R2-5. Abbreviations

**Reviewer request.** Define ECE, SSL, CTG, FHR, UC, AUPRC and AUROC at first appearance.

**Response.** Corrected and audited. The active manuscript, official LaTeX and upload text now define CTG, FHR, UC and SSL at first use, and the Methods/metric sections define AUPRC, AUROC and ECE at first use. The text-compliance audit reports zero abbreviation definition failures.

**Verification.** Status `READY_TO_POST`; evidence status `PASS`; evidence: Frontiers_Signal_Processing_Submission_Package_20260601/06_upload_text/Response_to_Reviewers_R1_20260707_route_pending.md.

### R2-6. Transformer hyperparameter table

**Reviewer request.** Add a summary table of transformer hyperparameters.

**Response.** Added. Supplementary Table 11 summarizes the transformer and SSL configuration, including input contract, tokenization, patch length, embedding size, encoder depth, masking mode, objectives, optimizer and training settings. The hyperparameter table contains 28 rows and is included in the supplementary file set.

**Verification.** Status `READY_TO_POST`; evidence status `PASS`; evidence: Frontiers_Signal_Processing_Submission_Package_20260601/05_supplementary_files/Supplementary_Table_11_transformer_hyperparameters.csv.

### R2-7. PatchTST rationale

**Reviewer request.** Explain why PatchTST was selected over alternative transformers.

**Response.** Added. The revised text links PatchTST to the signal structure: two-channel FHR/UC windows are long and non-stationary, patch tokenization reduces sequence length while preserving local morphology, and channel masking plus masked reconstruction/contrastive consistency match common CTG degradation patterns. We therefore present PatchTST as a pragmatic representation-learning architecture for this open CTG setting rather than as an intrinsically superior transformer.

**Verification.** Status `READY_TO_POST`; evidence status `PASS`; evidence: Frontiers_Signal_Processing_Submission_Package_20260601/01_manuscript_frontiers_full/Frontiers_Original_Research_Manuscript.tex.

## Final Pre-Posting Checklist

1. Confirm the locked-replay manuscript/DOCX and response files render correctly.
2. Keep CTGDL as bounded context only unless a future human decision approves manuscript-facing benchmark wording.
3. Rebuild active package, refresh ZIP, rerun submission preflight and rerun the R1 readiness gate.
