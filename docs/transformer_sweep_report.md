# Transformer SSL Sweep Report

Date: 2026-05-21.

## Implemented Upgrades

The transformer self-supervised learning pipeline now includes:

1. Mask strategies: random, block, channel, and mixed.
2. Masked patch reconstruction.
3. SimCLR-style contrastive loss between two augmented masked views of the same CTG window.
4. Tunable patch length, model width, layer count, mask strategy, and contrastive weight.
5. Fixed record-level split discipline.

## Sweep Design

Four compact CPU-feasible configurations were tested for 4 epochs each:

| Config | Patch | Width | Layers | Mask | Contrastive weight |
|---|---:|---:|---:|---|---:|
| p80_d48_l2_mixed_cw01 | 80 | 48 | 2 | mixed | 0.1 |
| p60_d64_l2_mixed_cw02 | 60 | 64 | 2 | mixed | 0.2 |
| p120_d64_l2_block_cw02 | 120 | 64 | 2 | block | 0.2 |
| p80_d64_l3_channel_cw02 | 80 | 64 | 3 | channel | 0.2 |

## Best Current Candidate

The best first-pass configuration is:

```text
patch_len=80
d_model=64
layers=3
nhead=4
mask_strategy=channel
contrastive_weight=0.2
epochs=4
```

Best validation result:

| Model | AUROC | AUPRC | Brier |
|---|---:|---:|---:|
| signal + SSL embeddings | 0.6482 | 0.4432 | 0.2554 |
| signal + SSL + signal phenotype | 0.6560 | 0.4324 | 0.2516 |

This improves AUPRC over signal features alone, but the result remains exploratory because the dataset is small and the SSL training curve is noisy.

## Longer Training Check

The same configuration trained for 12 epochs did not improve the main validation metrics:

| Model | AUROC | AUPRC | Brier |
|---|---:|---:|---:|
| signal + SSL embeddings | 0.6070 | 0.3987 | 0.2611 |
| signal + SSL + signal phenotype | 0.6154 | 0.4227 | 0.2588 |

Interpretation: longer training is not automatically better. Model selection should use validation performance and stability, not epoch count.

## External CTGDL-FHRMA Representation Check

CTGDL FHRMA processed data were downloaded and embedded using the current best CTU-UHB-trained transformer.

External data:

- CTGDL FHRMA records: 135
- FHRMA windows: 2604

External representation findings:

| Metric | Value |
|---|---:|
| CTU-UHB records | 552 |
| CTGDL-FHRMA records | 135 |
| CTU vs CTGDL domain classifier AUROC | 0.9088 |
| PCA variance explained by PC1+PC2 | 0.5092 |

FHRMA nearest CTU SSL phenotype assignments:

| CTU SSL phenotype | FHRMA records assigned |
|---:|---:|
| 1 | 56 |
| 2 | 1 |
| 3 | 62 |
| 4 | 16 |

Interpretation: CTGDL-FHRMA is useful as an external representation/domain-shift validation source, but it should not be described as external neonatal outcome validation unless outcome labels equivalent to CTU-UHB pH/Apgar are verified.

