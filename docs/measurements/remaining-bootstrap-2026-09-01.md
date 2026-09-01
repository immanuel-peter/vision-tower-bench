# Remaining targeted bootstraps and deterministic matched semantics, September 1 2026

## Outcome

Four pieces of statistics and reproducibility debt were addressed while the DIODE and pooled
feature caches were still resident on the Brev box.

1. All eight matched `projected`-against-final-`tower` geometry comparisons resolve in the
   Projector's favour. This covers four Projectors on depth and surface normals.
2. Qwen3.5's matched-mean Tower difference from Relative Depth 0.889 to 1.000 does not
   resolve: +0.00154 top-1 for the earlier cell, paired 95% interval
   [-0.00564, +0.00855]. The old claim that this arm genuinely declines is withdrawn.
3. The complete 112-cell semantic matched arm was regenerated with deterministic PCA. The
   twelve matched result JSONs in `results/` now supersede the randomized-PCA files from the
   August 31 matrix.
4. The eight raw Projector comparisons resolve in seven cases. Kimi K2.6 raw depth remains
   positive but crosses zero. Of Qwen3.5's four nominal late semantic declines, raw attention
   and raw mean resolve while matched attention and matched mean do not.

## Box and caches

Brev, 4x NVIDIA L40S 48 GB, 46 CPU cores, $4.22/hr. `/ephemeral` had 1.8 TB free at the end.
The final cache roots occupy 40 GiB for six 13,000-image pooled semantic caches and 96 GiB
for six 771-image full-token DIODE caches.

Three semantic caches and two geometry caches were missing at the start of this phase. All
five were rebuilt from the released Towers. The three semantic extractions ran together:
Qwen3.5 took 213s, Kimi K2.6 610s, and MoonViT-V2 852s, for 14m12s wall. The two geometry
extractions completed together in 109s. Every slice in all six semantic caches contains the
same 13,000 image ids, and every slice in all six geometry caches contains the same 771 ids.

## Projector intervals

Each comparison uses full patch tokens, the deterministic 512-wide matched reducer, the
Probe3D multiscale head, the full six-rate grid, 10 epochs, three seeds, and 10,000 paired
resamples of the seed-mean per-image metric over the 115-image DIODE test split. Positive
values favour `projected`; normal advantage is `tower` error minus `projected` error.

| Projector | task | `projected` | `tower` | advantage | paired 95% interval | wall |
|---|---|---:|---:|---:|---:|---:|
| Kimi K2.6 | depth `d1` | 0.62232 | 0.60495 | +0.01737 | [+0.00596, +0.02892] | 645s |
| Kimi K2.6 | normal error | 28.5296 | 30.7276 | +2.19797 deg | [+1.65383, +2.76855] | 599s |
| MoonViT-V2 | depth `d1` | 0.60866 | 0.53029 | +0.07836 | [+0.05956, +0.09838] | 645s |
| MoonViT-V2 | normal error | 29.1440 | 31.6408 | +2.49678 deg | [+1.96158, +3.04919] | 605s |
| Qwen3.5 | depth `d1` | 0.59177 | 0.53342 | +0.05835 | [+0.03738, +0.08088] | 503s |
| Qwen3.5 | normal error | 29.8926 | 34.2056 | +4.31301 deg | [+3.39556, +5.30180] | 506s |
| Muse Glimmer | depth `d1` | 0.58936 | 0.56325 | +0.02611 | [+0.01287, +0.03955] | 645s |
| Muse Glimmer | normal error | 30.6672 | 31.4596 | +0.79244 deg | [+0.37697, +1.20646] | 610s |

All eight overall lower bounds are positive. Fifteen of sixteen scene-specific intervals
also resolve; Kimi K2.6 depth outdoors is +0.00529 with interval [-0.00849, +0.01977]. The
overall Kimi interval and its indoor interval are positive.

The eight comparisons occupied 30m09s of wall time when scheduled across the four GPUs.
Their individual timings overlap and some overlap the tail of semantic extraction, so treat
them as operational costs, not isolated per-cell benchmarks.

## Raw Projector intervals

The raw continuation removes the 512-wide reducer and otherwise holds the geometry protocol
fixed: full patch tokens, Probe3D multiscale head, full six-rate grid, 10 epochs, three
seeds, and 10,000 paired resamples over 115 images.

| Projector | task | `projected` | `tower` | advantage | paired 95% interval | wall |
|---|---|---:|---:|---:|---:|---:|
| Kimi K2.6 | depth `d1` | 0.48741 | 0.46781 | +0.01960 | [-0.00032, +0.04021] | 749s |
| Kimi K2.6 | normal error | 29.0061 | 32.1187 | +3.11261 deg | [+2.42151, +3.82282] | 725s |
| MoonViT-V2 | depth `d1` | 0.48626 | 0.45926 | +0.02700 | [+0.01129, +0.04268] | 747s |
| MoonViT-V2 | normal error | 30.4614 | 33.2883 | +2.82685 deg | [+2.15059, +3.54422] | 711s |
| Qwen3.5 | depth `d1` | 0.54075 | 0.38069 | +0.16007 | [+0.12108, +0.19774] | 532s |
| Qwen3.5 | normal error | 29.3152 | 33.7584 | +4.44317 deg | [+3.50759, +5.44265] | 542s |
| Muse Glimmer | depth `d1` | 0.50063 | 0.40792 | +0.09271 | [+0.06235, +0.12374] | 773s |
| Muse Glimmer | normal error | 30.9422 | 33.8354 | +2.89322 deg | [+2.37672, +3.44321] | 723s |

Seven overall intervals resolve. Kimi K2.6 depth crosses zero by 0.00032 `d1`; its indoor
interval resolves and its outdoor interval does not. MoonViT-V2 depth also crosses zero
outdoors despite resolving overall. The remaining fourteen raw scene-specific intervals
resolve.

The fresh raw runs reproduce every original selected rate except Qwen3.5 depth `tower`,
which selects 1e-3 instead of 3e-3. That changes the fresh Qwen3.5 point gap from the matrix's
+0.1962 to +0.16007 without changing its direction or interval verdict.

## Qwen3.5 Relative Depth interval

The two matched-mean semantic cells were re-searched on the full eleven-rate grid and
trained for 20 epochs under three seeds. Ten thousand paired resamples operate on
seed-averaged correctness over the shared 1,950-image test split.

| earlier cell | final cell | difference | paired 95% interval | wall |
|---:|---:|---:|---:|---:|
| 0.87077 at Relative Depth 0.889 | 0.86923 at 1.000 | +0.00154 | [-0.00564, +0.00855] | 42s |

The point peak remains earlier, but the interval crosses zero. This is failure to resolve a
difference, not evidence that the two Relative Depth cells are equivalent.

The other three Qwen3.5 non-rising arms were then tested under the same semantic protocol.
The first cell is the nominal point peak from the deterministic eleven-rate results; the
second is the final Tower layer.

| readout | arm | earlier Relative Depth | earlier | final | difference | paired 95% interval | wall |
|---|---|---:|---:|---:|---:|---:|---:|
| attention | matched | 0.741 | 0.88256 | 0.88085 | +0.00171 | [-0.00632, +0.00957] | 64s |
| attention | raw | 0.889 | 0.88701 | 0.87419 | +0.01282 | [+0.00530, +0.02034] | 90s |
| mean | raw | 0.889 | 0.85983 | 0.84427 | +0.01556 | [+0.00530, +0.02632] | 60s |

Together with the earlier matched-mean result, both raw declines resolve and both matched
declines cross zero. These within-Tower tests use the same pooled cache at both Relative
Depth cells. They quantify test-image uncertainty for those curves; they do not validate
the unresolved full-token attention pooling control in ADR-0019.

## Deterministic semantic matched arm

Both readouts, six Towers, 112 cells total, the full eleven-rate grid, 20 epochs, and three
seeds. Attention and mean ran as separate fresh four-lane phases. Each phase produced six
JSONs, four `.DONE` markers, its exact `expected_cells.tsv`, and an empty `alerts.log`.

| phase | cells | lane-hours | wall | seconds/cell |
|---|---:|---:|---:|---:|
| attention matched | 56 | 0.513 | 598s | 33.0 |
| mean matched | 56 | 0.336 | 391s | 21.6 |
| total | 112 | 0.849 | 16m29s | 27.3 |

The earlier identical-box measurement was 0.88 lane-hours, 16 minutes, and 28.4s/cell for
the complete matched arm. It predicted this rerun accurately.

Deterministic PCA changes the August 31 randomized matched results rather than merely making
their JSON formatting reproducible. Mean absolute top-1 movement is 0.00270 for attention
and 0.00310 for mean. Eight attention cells and ten mean cells select a different rate.
The largest attention change is -0.0110 at a late DINOv2 Tower cell; the largest mean change
is +0.0156 at SigLIP2 Relative Depth 0.370. The cross-model semantic bootstrap files already
used deterministic PCA and agree with the corrected deepest cells.

The corrected matched-arm nominal winners are Muse Glimmer 0.9203 over SigLIP2 0.9162 for
attention, and SigLIP2 0.9126 over Muse Glimmer 0.9087 for mean. The existing paired
intervals for both comparisons cross zero, so these remain nominal orderings rather than
resolved winners.

## Cost and estimate audit

The cache rebuild, Qwen interval, eight geometry comparisons, and both matched semantic
phases ran from 00:45:19 to 01:35:18 UTC, 49m59s of wall time or about $3.52 at the box rate.
That is the bounded compute window, not the provider's final charge: verification,
documentation, and any idle rental time continue to bill by wall clock.

The prior matched semantic estimate was accurate. The geometry comparisons each took 8.4
to 10.8 minutes, while four-GPU scheduling reduced eight of them to 30 minutes wall. The
main thing the estimate missed was scientific rather than computational: deterministic PCA
moves enough matched point estimates and selected rates that the full matched semantic arm,
not only the bootstrap cells, had to be regenerated and preserved.

The raw continuation ran from 05:06:18 to 05:31:14 UTC, 24m56s wall, while the three Qwen3.5
semantic jobs overlapped the tail. The eight geometry jobs consumed 1.528 lane-hours and the
semantic jobs 0.059 lane-hours. At $4.22/hr the incremental rental cost was about $1.75. Raw
geometry took 8.9 to 12.9 minutes per comparison, slightly slower than matched geometry as
expected; four lanes kept the wall time under 25 minutes.

## Evidence

The twenty new JSONs from these phases in `results/bootstrap/` contain full validation
curves, selected rates, seed metrics, per-image predictions or metrics, and bootstrap
metadata. The twelve replaced
matched semantic JSONs in `results/` contain all 112 corrected cells. All long-running
commands ran in tmux with `UV_NO_SYNC=1`. Every job exited zero, all matrix alert logs are
empty, and independent checks reproduced every stored interval from its per-image data.
