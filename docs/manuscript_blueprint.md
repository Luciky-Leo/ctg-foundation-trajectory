# Manuscript Blueprint

## Title

Self-supervised dynamic phenotyping of intrapartum fetal heart rate and uterine contraction trajectories for neonatal risk prediction: a multi-dataset open-data study

## Abstract Structure

Background:

- CTG interpretation remains subjective and conventional supervised AI models often fail to generalize.

Methods:

- Use open intrapartum CTG time-series data.
- Pretrain a self-supervised model on FHR and UC traces.
- Derive dynamic fetal risk phenotypes from learned embeddings.
- Evaluate neonatal risk prediction and compare with conventional CTG feature models.
- Validate across available open datasets or harmonized external subsets.

Results:

- Report cohort size, outcome prevalence, phenotype number, discrimination, calibration, decision-curve utility, and subgroup stability.

Conclusions:

- Self-supervised CTG representations may provide clinically interpretable dynamic fetal risk phenotypes and improve reproducible neonatal risk stratification.

## Key Claims That Need Evidence

1. The model learns meaningful dynamic CTG patterns without requiring dense expert labels.
2. The learned phenotypes correspond to clinically interpretable FHR/UC behavior.
3. The model improves prediction or calibration compared with conventional baselines.
4. The model is robust across preprocessing choices and validation sources.
5. Subgroup performance does not reveal obvious unsafe degradation.

## Likely Target Journals

Realistic public-data target:

- npj Digital Medicine, stretch if external validation and model release are strong.
- Computer Methods and Programs in Biomedicine.
- Biomedical Signal Processing and Control.
- Artificial Intelligence in Medicine.
- Current retargeting: Methods of Information in Medicine as a health-informatics method study; prior BMC options are no longer the active target.

Higher target usually requires local hospital validation or prospective expert review.
