# Response to Reviewer 5

Manuscript title: *Self-supervised biomedical signal representation learning from intrapartum cardiotocography: stability, ablation, and exploratory neonatal risk enrichment*

We thank Reviewer 5 for identifying the central weaknesses in the original submission. We agree that the previous manuscript did not convincingly establish an incremental predictive benefit from self-supervised learning (SSL), that historical test-set ranking created selection risk, and that the labelled sample was small relative to the original predictor space. We did not attempt to defend the previous performance claim. We rebuilt the analysis around development-only architecture selection, a five-component SSL representation, random-encoder and objective/mask controls, five-seed stability, endpoint sensitivity, calibration of both conventional and SSL-containing models, and explicit limits on the prediction horizon and external analysis.

During the requested extension, we also added a prespecified public cross-domain sensitivity using the newly released JNU-CTG dataset. JNU-CTG contributed 62,307 non-overlapping 10-minute FHR/UC windows from 20,769 recordings and 12,606 patient groups. All clinical labels were withheld from SSL. This experiment was governed by a promotion rule fixed before the CTU fixed test set was opened. JNU pretraining transferred expert-labelled CTG morphology information but did not improve development neonatal-risk enrichment, and the fixed-test risk result was negative. The JNU results therefore remain supplementary. This additional analysis strengthens the evidence boundary rather than the predictive claim.

The revised manuscript is now framed as a transparent CTG representation-learning, stability and ablation study in which the incremental predictive value of SSL remains unresolved. The historical test-ranked AUPRC of 0.4016 is retained only in Supplementary Table S9 for transparency.

## Comment 1

> The most serious concern is that the claimed benefit of self-supervised learning is not convincingly demonstrated.

**Response:** We agree. The revised manuscript no longer claims that SSL improves neonatal outcome prediction. The primary comparison uses 10 classical signal features versus the same features plus five development-fitted SSL principal components. The signal baseline achieved fixed-test AUPRC 0.3765, AUROC 0.6428 and Brier score 0.2482. Across five selected-encoder seeds, the reduced SSL model achieved median AUPRC 0.3935, but median AUROC was lower (0.5954) and median Brier score was worse (0.2628). Every paired AUPRC confidence interval versus the signal baseline crossed zero.

Two negative controls now reinforce that conclusion. First, on the fixed development ablation split, a random encoder achieved AUPRC 0.3333, whereas the combined reconstruction-contrastive objective achieved 0.3106. Second, in the new cross-domain experiment, median development AUPRC was 0.3797 for CTU-only SSL and 0.3682 for JNU pretraining followed by CTU adaptation. The paired mean difference was 0.0030 (95% CI -0.0304 to 0.0316; positive in three of five seeds). The prespecified risk-enrichment promotion gate therefore failed.

The JNU route did improve a representation task: expert-morphology macro-AUROC increased from 0.6392 for CTU-only SSL to 0.6802 for JNU-to-CTU adaptation (paired difference 0.0401, 95% CI 0.0155 to 0.0647), and it exceeded a random encoder by 0.0611 (95% CI 0.0344 to 0.0884). However, classical signal features remained stronger (macro-AUROC 0.7357). We therefore state that public cross-domain pretraining transferred morphology information, not that it improved clinical prediction.

**Changes in manuscript:** Abstract (page 1, lines 3-24); Results, “Reduced SSL features did not provide a statistically resolved increment” (page 3, lines 92-107) and “Large public pretraining transferred morphology but not neonatal-risk enrichment” (page 3, lines 117-130; page 5, lines 131-142); Discussion (page 9, lines 184-231); Figure 2; Supplementary Figure S3; Supplementary Tables S3-S5 and S14-S16.

## Comment 2

> A second major concern is the potential contamination of the held-out test set through model/configuration selection.

**Response:** We agree that historical fixed-test ranking compromised any claim of complete test independence. Four compact configurations were therefore re-evaluated entirely within the 386 CTU development records using three-fold stratified validation. In every fold, SSL pretraining used only that fold's development-training windows; validation and fixed-test records were excluded. A one-standard-error rule selected the smallest eligible model (p80-d48-l2 mixed; 74,768 parameters).

The JNU experiment was also locked before fixed-test access. The architecture, folds, 10-signal-plus-five-PC downstream model, seeds and promotion rule were frozen. JNU labels were withheld. JNU-to-CTU pretraining used only JNU windows followed by the relevant CTU development-training windows. The development promotion decision was recorded as `SUPPLEMENT_ONLY` before the fixed test set was opened. Only after this decision did we run the fixed-test script.

Prior historical exposure cannot be undone. We therefore call this a “fixed test set with potential selection-induced optimism,” not an independent validation set. The historical 0.4016 AUPRC remains only in Supplementary Table S9. The new fixed-test JNU estimates are also explicitly exploratory: JNU-to-CTU median AUPRC was 0.3420 versus 0.4017 for CTU-only SSL, with a paired difference of -0.0403 (95% CI -0.0766 to 0.0016; positive in zero of five seeds). No fixed-test result was used to alter the frozen route.

**Changes in manuscript:** Abstract (page 1, lines 3-24); Figure 1A-D (page 4); Results (page 3, lines 82-85 and 117-130; page 5, lines 135-142); Methods, “Fixed record split and signal processing” and “Development-only architecture selection” (page 10, lines 265-285), and “Public cross-domain pretraining and morphology probes” (page 11, lines 302-328); Discussion (page 9, lines 206-213); Supplementary Figures S1 and S3; Supplementary Tables S2, S9, S16 and S18.

## Comment 3

> The sample size is inadequate for the complexity of the proposed supervised model. Only 79 positive events are available in training, whereas the SSL representation contains 64 dimensions and the combined model uses a substantially larger predictor space.

**Response:** We agree. The 64-dimensional embedding and phenotype-augmented model has been removed as the primary supervised model. The revised primary model contains 10 prespecified signal features plus five development-fitted SSL principal components, for 15 predictors and 79 development events (5.27 events per predictor). Phenotypes are excluded from the primary model and evaluated only as secondary interpretation variables.

We added PCA 5/10/20, elastic-net and missingness sensitivities and report predictor count and events per predictor. Higher-dimensional point estimates were not used to replace the prespecified five-component model because the fixed test set had already been viewed. The new JNU experiment does not increase the supervised event count: JNU clinical labels were deliberately withheld, and only CTU development outcomes trained the risk model. We explicitly state that dimensionality reduction mitigates but does not eliminate overfitting and that the JNU pretraining scale cannot compensate for limited outcome events.

**Changes in manuscript:** Abstract (page 1, lines 3-24); Results, reduced-model subsection (page 3, lines 92-107); Methods, “Classical features, reduced SSL model and regularization” (pages 11-12, lines 329-340) and “Public cross-domain pretraining and morphology probes” (page 11, lines 302-328); Discussion (page 9, lines 214-223); Supplementary Figure S1D; Supplementary Tables S5 and S18.

## Comment 4

> The proposed SSL framework contains several components-masked reconstruction, contrastive learning, masking and augmentation-but their individual contributions are not sufficiently demonstrated.

**Response:** We added prespecified development-only component sensitivities: random initialization, reconstruction-only, NT-Xent-only, combined reconstruction plus NT-Xent, and the combined objective without jitter. We also compared random, block, channel and mixed masking. All results are reported regardless of direction. The combined objective did not outperform the random encoder, and mixed masking did not outperform the other masks on the fixed development split. These results are not used to claim component superiority or to reselect the final model.

We additionally evaluated FHR-only, UC-only, 1-Hz and channel-asymmetric masking with the JNU route. Direction changed between development and exploratory fixed-test analyses. Channel-asymmetric masking had an unresolved development risk-AUPRC difference versus mixed JNU masking (0.0098, 95% CI -0.0290 to 0.0444) and materially lower morphology macro-AUROC (-0.0787, 95% CI -0.1006 to -0.0552). It was therefore retained as a negative development sensitivity, not adopted as the primary mask.

**Changes in manuscript:** Results, objective-ablation and JNU subsections (page 3, lines 108-130; page 5, lines 135-142); Methods, “Self-supervised encoder and ablations” and “Public cross-domain pretraining and morphology probes” (page 11, lines 286-328); Discussion (page 9, lines 184-189 and 224-231); Figure 2E; Supplementary Figures S1B-C and S3E-F; Supplementary Tables S3, S16 and S17.

## Comment 5

> The authors need to address the test-set selection issue, demonstrate model stability, establish a clinically meaningful prediction horizon, strengthen the baseline comparisons, perform SSL ablation studies, and substantially moderate the interpretation of the predictive results.

**Response:** We addressed each requested element:

1. **Test-set selection:** configuration selection is now development-only with fold-specific pretraining. JNU promotion was also decided from development results before fixed-test access. Historical exposure is disclosed and cannot be erased.
2. **Stability:** the frozen encoder routes were run with seeds 20260521-20260525. We report medians, IQRs, ranges and paired hierarchical bootstrap confidence intervals.
3. **Prediction horizon:** the data use complete available intrapartum records and cannot establish a fixed clinical lead time. The manuscript now states: “This is a retrospective delivery-level risk-enrichment analysis without a prespecified lead-time horizon.” Real-time warning, deployment and autonomous triage implications were removed.
4. **Baselines:** the primary comparator is a prespecified 10-feature signal model; missingness is sensitivity-only. Random embeddings are a negative control. Classical signal features are also included in the morphology probes, where they outperformed every SSL representation.
5. **Ablations:** reconstruction, NT-Xent, combined objective, jitter, four masks, FHR-only, UC-only, 1-Hz and channel-asymmetric masking are reported without selecting the most favourable fixed-test result.
6. **Moderated interpretation:** the title foregrounds stability, ablation and exploratory enrichment. The manuscript states that risk improvement was unresolved, that the JNU result supports morphology transfer only, that fixed-test results may be optimistic, and that CTGDL is not external outcome validation.

Endpoint sensitivities remain explicitly exploratory. For pH <7.15 alone, signal and reduced SSL AUPRCs were 0.4161 and 0.4044; for pH <7.10 they were 0.4507 and 0.4588; for pH <7.05 they were 0.5721 and 0.5382. The Apgar-only test analysis contained four events and is labelled unstable.

Finally, we added recent context without converting it into a superiority claim. A cross-database study motivated the 1-Hz/channel sensitivity; contemporary CTG SSL preprints motivated, but did not select, the multi-objective and channel-asymmetric ablations; and a randomized human-machine CTG study supports evaluating future systems as clinical assistance rather than autonomous deployment.

**Changes in manuscript:** Abstract (page 1, lines 3-24); Results (pages 3 and 5, lines 82-179); Discussion (pages 7 and 9-10, lines 182-247); Methods (pages 10-12, lines 248-374); Figures 1, 2 and 4; Supplementary Figures S1-S3; Supplementary Tables S2-S18.

## Revised claim

The revised evidence supports a reproducible workflow for CTG representation analysis, development-only model selection, seed stability, negative controls, interpretable phenotypes, calibration and bounded public cross-domain pretraining. It does not establish that SSL improves neonatal outcome prediction, does not provide a prespecified clinical prediction horizon, and does not constitute external outcome validation or clinical deployment evidence. The JNU experiment adds a specific positive result—transfer of expert-morphology information—alongside a prespecified negative result—no stable neonatal-risk enrichment.
