# Pooling attention diagnostic, August 31 2026

These files preserve the Job 2 diagnostic that followed the failed attention validation. They are not the full
pooling re-check. The diagnostic probes the three collapsed full-token attention cells
named by that validation on both pooled and full tokens under four configurations:

- 1,500 images, 20 epochs
- 1,500 images, 100 epochs
- 5,000 images, 20 epochs
- 5,000 images, 100 epochs

Every cell searches `[1e-5, 3e-5, 1e-4, 3e-4, 1e-3]`, selects on validation, and reports
three test seeds. The selected cells are SigLIP2 attention raw Tower Relative Depth 0.370,
SigLIP2 attention matched Tower Relative Depth 0.630, and Muse Glimmer attention matched
Tower Relative Depth 0.760. The 24 primary JSONs are the complete `3 cells x 2 grids x 4
configurations` matrix. `repeat-baseline/` contains an independent repeat of all six
1,500-image/20-epoch cells.

## Decision

No tested configuration met the diagnostic target across all three cells.

| configuration | decisive full-token result |
|---|---|
| 1,500 / 20 | Muse matched collapses, 0.2193 +/- 0.2692 |
| 5,000 / 20, Lever A | SigLIP2 matched collapses, 0.6796 +/- 0.1551 |
| 1,500 / 100, Lever B | SigLIP2 matched remains above the spread target, 0.8133 +/- 0.0251 |
| 5,000 / 100 | SigLIP2 raw remains above the spread target, 0.7227 +/- 0.0275 |

Lever A therefore failed. More epochs helped, especially for Muse Glimmer, but neither
Lever B alone nor the combined setting brought every seed spread under about 0.02 while
also producing a usable validation curve. The full Job 3 re-check was not launched.

## Reproducibility finding

The raw 1,500-image/20-epoch cells reproduce the original run exactly. The original matched cells
do not: `torch.svd_lowrank` randomized the capacity-matching PCA before any seed was set,
so a single-cell process did not reproduce the matrix process's RNG history. Commit
`8c82a96` seeds the semantic reducer without changing the readout RNG. The six corrected
baseline cells repeat independently in `repeat-baseline/` with identical selected rates,
test means, and seed spreads. Five validation curves are identical; Muse Glimmer full
matched differs at the nonselected `1e-3` point by one validation image (0.1200 against
0.1156), without changing its selected rate or reported test result.

Both primary alert logs are empty. `run-baseline.log` and `run-rest.log` record all lane
starts and successful exits; the repeat is in `run-baseline-repeat.log`.
