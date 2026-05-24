# Source Manifest

Last checked: 2026-05-21.

## CTU-UHB / CTU-CHB Intrapartum CTG Database

- Source: https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/
- DOI: https://doi.org/10.13026/C22013
- Access: open access on PhysioNet.
- License: Open Data Commons Attribution License v1.0.
- Current role: primary raw signal cohort.
- Key contents: 552 intrapartum CTG recordings, FHR and UC signals, WFDB-style header/data files, and clinical metadata.
- Relevance: best open source for foundation-style dynamic modeling because it has raw time-series data plus neonatal outcome fields.

## UCI Cardiotocography

- Source: https://archive-beta.ics.uci.edu/dataset/193/cardiotocography
- Access: open repository download.
- Current role: feature-level baseline.
- Key contents: 2126 CTG feature rows and expert labels.
- Limitation: no raw signal, so it cannot support self-supervised waveform pretraining.

## CTGDL

- Source: https://zenodo.org/records/18023488
- Current role: external or harmonized validation candidate.
- Key caveat: subset licenses differ. The Zenodo page notes that CTU-UHB uses Open Data Commons Attribution License v1.0, while the SPAM subset is governed by a Data Use Agreement.
- Project rule: use only subsets whose terms allow the intended analysis, and document the subset actually used.

## CTU-CHB Annotation Dataset

- Article: Annotation dataset of the cardiotocographic recordings constituting the CTU-CHB intra-partum CTG database.
- DOI: https://doi.org/10.1016/j.dib.2020.105690
- Current role: optional interpretability and event-annotation support.
- Project rule: verify downloadable files before making this a required analysis dependency.

