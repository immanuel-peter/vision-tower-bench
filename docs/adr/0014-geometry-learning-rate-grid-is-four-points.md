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
