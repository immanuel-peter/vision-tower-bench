# The geometry learning-rate grid is four points, not eight

PLAN.md specifies an eight-point learning-rate grid, validation-selected, three seeds,
identical everywhere. The semantic pillar runs that grid over `[3e-4 ... 1.0]`, which
suits its small attention pool. The geometry pillar trains a Probe3D multiscale
convolutional decoder instead, and the grid has to move down to straddle the rate the
first geometry run fixed.

Costed at eight points the geometry matrix is 11 runs per cell over 72 cells, which
measured out to roughly 17.6 hours of rented GPU. That is the reason to cut, but the
evidence for where to cut came from measuring the grid rather than from the budget.

Three cells were run at the full eight points before the matrix launched. Depth `d1` at
16 by 16, and both tasks at 32 by 32:

| rate | `merged` d1 | `projected` d1 | `tower` L15 d1 | `tower` L15 mean_deg |
|---|---|---|---|---|
| 1e-5 | 0.0445 | 0.0549 | 0.0533 | 28.5388 |
| 3e-5 | 0.1466 | 0.1492 | 0.1756 | 24.9513 |
| 1e-4 | 0.1981 | 0.2015 | 0.2749 | 22.8189 |
| 3e-4 | 0.3953 | 0.3277 | 0.4727 | 21.0032 |
| 1e-3 | 0.4966 | 0.4678 | 0.5983 | 20.5721 |
| 3e-3 | 0.4820 | 0.3644 | 0.5995 | 22.4693 |
| 1e-2 | 0.1245 | 0.1246 | 0.6029 | 28.5285 |
| 3e-2 | 0.1245 | 0.1246 | 0.1245 | 45.4092 |

Everything at 1e-4 and below is far enough from the optimum that no cell selects it. So
the grid runs `[3e-4, 1e-3, 3e-3, 1e-2]`, four points, in every cell of both arms, both
tasks and both models. Seven runs per cell instead of eleven.

The 32 by 32 column is why the cut is four points and not three. On the 16 by 16 Stages
1e-2 diverges to a degenerate 0.1245, and stopping the grid at 3e-3 would have looked
safe. At full spatial resolution depth is nearly flat from 1e-3 to 1e-2 and selects
1e-2, so a three-point grid would have truncated at the selected value. Normals at the
same cell select 1e-3 and degrade steadily above it, so the two tasks want different
ends of the range and the grid has to cover both.

The upper edge is a selected value in one measured cell, which is the cost of the cut.
The optimum for `tower` depth lies somewhere between 1e-2 and 3e-2, and 3e-2 collapses
to the degenerate value everywhere. Across 1e-3 to 1e-2 that curve moves 0.0046, well
under the seed spread, so truncating there changes no conclusion.

This departs from PLAN.md on point count while keeping what the plan is actually for:
one grid, chosen once, applied identically to every cell, so cells stay comparable. A
per-Stage or per-model grid would buy the same time and destroy that.

## What the full matrix said, and what it corrects

The four points stand. The reason recorded above for keeping the top one does not.

That reason read the split as 32 by 32 against 16 by 16, from three cells at one seed. All
three were unmatched. Over the full 72 cells at three seeds the split is by token width and
the grid size does not enter it. Every capacity-matched depth cell selects 1e-2, at 32 by 32
and at 16 by 16 alike, and every cell at 1024 wide and above collapses to the degenerate
0.1245 there. The 32-by-32 cell that carried the argument, unmatched DINOv2 `tower` at
Relative Depth 0.625, now selects 3e-3: it reads 0.6012 there against 0.5903 at 1e-2, where
the single-seed eight-point run had them 0.0034 apart the other way. That was noise.

So 1e-2 earns its place because the 512-wide reduction needs it, not because full spatial
resolution does. Sixteen cells select it, which is a stronger case than the one cell above,
and it arrives for a different reason.

The claim that nothing selects the bottom of the range is also wrong. 13 cells select 3e-4,
12 of them unmatched normals, and their validation curves fall monotonically toward it, so
their optimum sits at or below the grid floor. Both edges truncate, on different
(task, arm) combinations: 29 of 72 cells select an edge.

The grid stays identical in every cell, which is the property PLAN.md is actually asking
for and the only one that keeps cells comparable. Truncation costs absolute level in the
cells that hit an edge and moves no comparison between cells. Extending the range in both
directions is a change to make once, for every cell, and it needs a costed re-run rather
than a per-arm patch.

## The roster run uses six points

The roster run extends the grid to `[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]`. With six
searched rates and three seeds at the selected rate, each cell takes nine runs instead of
seven. On four GPUs, the estimate rises from 5.6 to 7.2 hours.

The runner and lane script had separate defaults. The runner kept the original eight
points while the lane script passed four, and only the lane value reached a cell. A test
now requires both defaults to match.

## The roster run measured the six points, and they hold

9 of 224 cells select an edge, against 29 of 72 on the four points. Four percent against
forty. Selected rates over the matrix:

| rate | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 | 3e-2 |
|---|---|---|---|---|---|---|
| cells | 5 | 63 | 92 | 20 | 40 | 4 |

Both extensions earned their place. Adding 1e-4 brackets the unmatched normals cells that
used to pin at the 3e-4 floor with a curve still falling: MoonViT-V2 normals `tower` at
Relative Depth 0.111 now reads 36.14 at 1e-4, 35.27 at 3e-4 and 38.24 at 1e-3, so the
optimum sits inside the grid. Adding 3e-2 brackets most of the matched depth cells that
used to pin at the 1e-2 ceiling; they still select 1e-2 with 3e-2 available and losing.

Six cells still bound their own level, and they cluster:

| model | task | arm | stage | picked | edge over neighbour | seed spread |
|---|---|---|---|---|---|---|
| Muse Glimmer | normal | unmatched | projected | 1e-4 | 0.5453 | 0.1447 |
| Qwen3.5 | normal | unmatched | tower | 1e-4 | 0.2828 | 0.2487 |
| Qwen3.5 | depth | unmatched | projected | 3e-2 | 0.0918 | 0.0032 |
| Qwen3.5 | depth | matched | projected | 3e-2 | 0.0122 | 0.0059 |
| Kimi K2.6 | depth | matched | projected | 3e-2 | 0.0066 | 0.0049 |
| SigLIP2 | depth | unmatched | tower | 3e-2 | 0.0183 | 0.0070 |

Four of the six sit at the 3e-2 ceiling, three of those on a `projected` Stage. The
remaining truncation is a depth-and-`projected` phenomenon, not a general one. The other
three edge cells, all at the 1e-4 floor, are plateaus where the edge and its neighbour differ
by less than the seed spread, so nothing is lost there.

Two top-edge curves are not monotone in rate. SigLIP2 depth `tower` at Relative Depth 0.519
reads 0.6127 at 1e-3, dips to 0.5590 at 3e-3, recovers to 0.6078 at 1e-2 and peaks at
0.6261 at 3e-2.
Qwen3.5 depth `projected` does the same: 0.4736, 0.4077, 0.4293, 0.5211. A search that
stops at the first turn picks the wrong side of these.

Keep the six-point grid. Extending it to 1e-1 would test the four ceiling cells for one
additional run per cell. Add that point to the next run.

## The semantic grid is now the one that truncates

The same check on the semantic pillar is worse than anything this ADR recorded for
geometry. 107 of the 112 attention cells select 3e-4, the floor of the eight-point grid
PLAN.md specifies, and none of the 112 mean cells do. The split is by readout, not by arm:
53 of 56 attention matched, 54 of 56 attention raw, 0 of 56 in each mean arm.

PLAN.md chose `[3e-4 ... 1.0]` to suit the small attention pool. It suits mean pooling
instead, and the attention readout is the headline one. Its absolute levels are reported
below their optimum almost everywhere.

How far below could not be measured from that run, because `probe_run` recorded only the
selected rate while `geometry_run` records a `learning_rate_search` dict per cell. That gap
is now closed. Cost the extension after the next semantic run has curves to read.
