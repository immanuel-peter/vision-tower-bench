# Deterministic matched geometry rerun, September 13 2026

Headline-only rerun of the capacity-matched geometry arm with the seeded
reducer (`vtb.probe_run.fit_reducer`, seed 0, readout RNG untouched). The
original matched matrices used randomized PCA before its seed was fixed, so
their exact draws cannot be reconstructed; the values here supersede the old
matched magnitudes. Directions are unchanged: `projected` beats `tower` in
all eight model-task cells.

Scope: deepest Tower layer only, Stages `tower`/`merged`/`projected` for the
four Projectors (3 cells each), deepest `tower` for the two controls (1 cell
each). Same DIODE val split (541/115/115), six-rate grid
`[1e-4 ... 3e-2]`, 10 epochs, three seeds, multiscale head. Lane log:
`lane0.log`; `alerts.log` is empty. `expected_cells.tsv` records per-file
counts.

| model | task | `tower` | `merged` | `projected` | `tower`→`merged` | `merged`→`projected` |
|---|---|---|---|---|---|---|
| moonvit_v2 | depth `d1` | 0.5276 ± 0.0105 | 0.5934 ± 0.0141 | 0.6133 ± 0.0007 | +0.0658 | +0.0199 |
| moonvit_v2 | normal err | 32.4368 ± 1.4534 | 29.3888 ± 0.8293 | 29.1349 ± 0.5149 | -3.0480 | -0.2539 |
| kimi_k26 | depth `d1` | 0.5646 ± 0.0060 | 0.5431 ± 0.0022 | 0.6245 ± 0.0033 | -0.0215 | +0.0814 |
| kimi_k26 | normal err | 31.6822 ± 0.1595 | 31.8762 ± 0.3533 | 29.7342 ± 0.3404 | +0.1940 | -2.1420 |
| muse_glimmer | depth `d1` | 0.5649 ± 0.0031 | 0.5228 ± 0.0026 | 0.5866 ± 0.0005 | -0.0421 | +0.0638 |
| muse_glimmer | normal err | 31.4564 ± 0.4341 | 32.2395 ± 0.3573 | 30.6516 ± 0.3346 | +0.7831 | -1.5879 |
| qwen3_5 | depth `d1` | 0.5270 ± 0.0123 | 0.5359 ± 0.0064 | 0.5982 ± 0.0032 | +0.0089 | +0.0623 |
| qwen3_5 | normal err | 34.0508 ± 0.2700 | 36.2110 ± 0.0616 | 29.9001 ± 0.3608 | +2.1602 | -6.3109 |

Normals report angular error in degrees (lower is better). Spreads are
population standard deviations over three seeds. MoonViT-V2 remains the only
Projector whose learned step sits below its own lossless yardstick on both
tasks; the other three move the metric several times further than the
lossless regrouping does.
