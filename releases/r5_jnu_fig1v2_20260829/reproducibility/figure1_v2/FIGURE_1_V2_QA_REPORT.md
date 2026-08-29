# Figure 1 v2 QA report

Status: `PASS`

## Automated checks

| Check | Result | Evidence |
|---|---|---|
| Panel C count mapping | PASS | {'JNU-CTG records': 20769, 'JNU patient groups': 12606, 'CTU-UHB records': 552, 'CTGDL records': 135} |
| Panel C event mapping | PASS | {'Development': (386, 79), 'Fixed test': (166, 34)} |
| Panel C event fractions | PASS | 79/386 and 34/166 recomputed |
| Panel D role mapping | PASS | 7 role cells |
| Panel and composite outputs | PASS | missing=none |
| Final-size dimensions | PASS | 180.000 x 236.430 mm at 100% |
| Raster export | PASS | 2126 x 2793 px; dpi=(299.9994, 299.9994) |
| PERSIST source-code-first validation | PASS | Panels C/D validated; Panels A/B correctly retained as native Image2 routes |
| PERSIST candidate gate | PASS | 5/5 candidates and 2/2 panels passed |

## Original-size visual review

- Panel A CTGDL arrow points from the frozen representation toward domain context.
- Panel A fixed test is labelled exploratory and separated from development-only selection.
- Panel B shows FHR/UC inputs, patch tokens, mixed masking, transformer embedding, reconstruction, and NT-Xent.
- Panels C/D were inspected at original resolution; titles, values, axis text, headers, symbols, and legends do not overlap.
- The figure does not claim external neonatal-outcome validation or a prespecified lead-time horizon.

## Claim boundary

Only Panels C and D are PERSIST source-code-first renders. Panels A and B are separately generated Image2 schematics with deterministic scripted preparation. The composite is therefore a hybrid source-controlled figure, not a whole-figure PERSIST reproduction.
