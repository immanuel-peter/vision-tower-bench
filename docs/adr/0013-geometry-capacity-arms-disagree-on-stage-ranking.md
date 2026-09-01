# The geometry capacity arms disagree on Stage ranking

ADR-0008 accepts capacity matching only when model rankings and Relative Depth curves
agree between the matched and unmatched runs, and requires any disagreement to be
reported rather than smoothed over. The geometry matrix produces one, and the second run
did not dissolve it.

On MoonViT-V2 depth `d1` at the deepest point, over three seeds at a validation-selected
rate, the unmatched arm reads `merged` 0.5205, `projected` 0.4775 and `tower` 0.4537. The
matched arm reads `projected` 0.6145, `merged` 0.6062 and `tower` 0.5378. `merged` and
`projected` swap places. The unmatched gap of -0.0430 is 8.3 seed standard deviations; the
matched gap of +0.0083 is 1.3, and is noise.

The disagreement is 0.0513 wide. On the semantic side the same two arms agreed to within
0.0283 and changed no conclusion, which is why that check passed there and fails here. The
swap is also the whole question: it separates a Projector that appears to destroy spatial
information from one that does not.

## What the second run tested

The first run pinned every cell at 1e-3 and this ADR blamed an unstable 4.85M-parameter
head on 541 images. Per-cell rate selection was the obvious way to break that: a wide
unstable head can pick a lower rate and settle down. It did not break. The unmatched
ordering reproduced at a higher confidence than before, and one of the three arguments
this ADR rested on is now dead.

Seed instability does not track head width. The unmatched `projected` cell at 4,852,480
parameters spreads 0.0062 across seeds, against 0.0039 at `merged` and 0.0127 at the
narrow 1024-wide `tower`. The widest cell in the matrix is the second most stable of the
three, and the narrowest is the least. On normals the same collapse holds: 0.4486 at
unmatched `projected` against 1.5599 at `merged`. The first run's 4.4931-degree swing at
`projected` was a fixed-rate artifact, not a property of the head.

What replaced it is visible in the validation search, and it is a sharper mechanism. Rate
headroom shrinks as the head gets wider. On depth `d1` over the val split:

| rate | unmatched `merged` 4096 | unmatched `projected` 7168 | matched `merged` 512 | matched `projected` 512 |
|---|---|---|---|---|
| 3e-4 | 0.3946 | 0.3297 | 0.3967 | 0.3818 |
| 1e-3 | 0.5000 | 0.4608 | 0.5499 | 0.5532 |
| 3e-3 | 0.4399 | 0.3385 | 0.5811 | 0.5746 |
| 1e-2 | 0.1244 | 0.1247 | 0.6088 | 0.5980 |

Both unmatched cells select 1e-3 because 3e-3 already costs them and 1e-2 collapses to the
degenerate 0.1245. Both matched cells select 1e-2 and are still climbing at the edge of the
grid. So the unmatched arm compares two cells at a rate that is optimal for neither, and
the 7168-wide one is hurt more: it trails `merged` by 0.0392 at the rate both select and by
0.1014 one step above it. Reducing every cell to 512 dimensions gives both cells back the
rate they want and the gap goes to 0.0083.

That is a measurement rather than an argument, and it says the unmatched depth gap tracks
how well a head of that width can be trained on 541 images, not what the Stage carries.

## What did not change

Neither arm shows `projected` below `tower` on either task, so the headline claim survives
either way. The matched arm is the one ADR-0010 nominates for the headline.

The `tower` against `merged` control still bounds how much either arm can be trusted. That
pair is a lossless regrouping, verified bit-exact against the cache on disk over all 771
images, and the matched arm still scores it 0.0684 apart on depth, 47 seed standard
deviations. Any Stage effect smaller than that is below the noise floor of this readout,
and the disputed `merged` against `projected` gap is smaller than it in both arms. In the
unmatched arm the margin is thin: 0.0430 against a 0.0668 yardstick, a factor of 1.6.

What no arm can settle at 541 training images is the smaller question of whether the
Projector costs anything at all relative to `merged`. Rate selection was the cheap way to
find out and it did not answer it. Answering it needs the DIODE training split, which is
the decision ADR-0011 parked pending this run.

## The roster run: MoonViT-V2 is not special

This ADR asked whether the swap is a MoonViT-V2 property. It is not. Over four Projectors
on two tasks, six of eight combinations swap the two middle Stages between arms:

| model | task | unmatched | matched | |
|---|---|---|---|---|
| MoonViT-V2 | depth | merged > projected > tower | projected > merged > tower | swap |
| MoonViT-V2 | normal | projected > merged > tower | merged > projected > tower | swap |
| Kimi K2.6 | normal | projected > merged > tower | projected > tower > merged | swap |
| Muse Glimmer | depth | projected > merged > tower | projected > tower > merged | swap |
| Muse Glimmer | normal | projected > merged > tower | projected > tower > merged | swap |
| Qwen3.5 | depth | projected > merged > tower | projected > tower > merged | swap |
| Kimi K2.6 | depth | projected > merged > tower | projected > merged > tower | agree |
| Qwen3.5 | normal | projected > tower > merged | projected > tower > merged | agree |

The MoonViT-V2 depth swap recorded above reproduces exactly. Capacity matching reorders the
middle two Stages across most of the roster. The ordering of `merged` against `projected`
therefore depends on the readout capacity, not only on the Stages.

MoonViT-V2 is the only model where the arms disagree about the top-ranked Stage, and they
disagree on both tasks. `merged` leads unmatched depth while `projected` leads matched depth;
the reverse holds for normals. In the other three Projectors, `projected` ranks first in both
arms every time. Their swaps occur between `tower` and `merged` below the top rank.

For Kimi K2.6, Muse Glimmer and Qwen3.5, the Projector ranks first in both arms. Their Stage
conclusion does not depend on capacity matching. For MoonViT-V2, the choice of arm decides
whether the Projector or the lossless regrouping ranks first. Capacity matching reorders
Stages across the roster, but only the MoonViT-V2 reorder changes the top-ranked Stage.

The mechanism generalises too. In the matched arm `merged` drops below `tower` in four of
the eight combinations, on a pair where the merge provably loses nothing. A 512-dimensional
reduction of a 4096 to 6144 wide `merged` Stage keeps less of what the head needs than the
same reduction of a 1024 to 1536 wide `tower` Stage. The reduction sets the ordering.

What the roster run does change is the conclusion this ADR draws from the yardstick. On
MoonViT-V2 the lossless step is 8 to 23 times the Projector step, so the Projector effect
sits under the noise floor. On the other three Projectors it does not: the ratios are 0.0
to 0.4, meaning the Projector moves the metric further than a step that loses nothing.
Muse Glimmer on depth moves +0.0691 at 34.4 seed deviations against a lossless step of
-0.0151 at 0.9. The noise-floor argument holds for one model out of four. See
`results/README.md` section 1.

## Amendment: paired image interval, August 31 2026

The original matched matrix fitted randomized PCA before seeding it, so its exact reducers
were dependent on lane history and cannot be reconstructed. The corrected geometry path
uses the deterministic, train-split-only reducer shared with the semantic pillar. This is a
reproducibility correction, not a new capacity arm.

Muse Glimmer's matched depth cells retain the same selected rate, 3e-3. The corrected
`projected` cell reads 0.58920 `d1` and `merged` reads 0.52298, moving the Stage gap from
+0.0691 to +0.06622. Ten thousand paired resamples of the seed-mean per-image metric over
the 115-image test split give a 95% interval of [+0.04992, +0.08338]. The interval excludes
zero overall, indoors, and outdoors. The Projector improvement therefore survives both the
deterministic reduction correction and test-image uncertainty.

This interval resolves the strongest Projector step; it does not put intervals on every
Stage comparison in the matrix. The full per-image dataset and learning-rate curves are in
`results/bootstrap/geometry-depth-matched-muse-projected-vs-merged.json`.

## Amendment: all matched Projector intervals, September 1 2026

The remaining headline comparison is `projected` against the final `tower`, not
`projected` against `merged`. It was rerun for both geometry tasks in every Projector with
the deterministic reducer, the full six-rate grid, three seeds, and ten thousand paired
image resamples over the 115-image test split. Positive depth differences and positive
normal advantages both mean that `projected` performs better.

| Projector | task | corrected `projected` | corrected `tower` | advantage | paired 95% interval |
|---|---|---:|---:|---:|---:|
| Kimi K2.6 | depth `d1` | 0.62232 | 0.60495 | +0.01737 | [+0.00596, +0.02892] |
| Kimi K2.6 | normal error | 28.5296 | 30.7276 | +2.19797 deg | [+1.65383, +2.76855] |
| MoonViT-V2 | depth `d1` | 0.60866 | 0.53029 | +0.07836 | [+0.05956, +0.09838] |
| MoonViT-V2 | normal error | 29.1440 | 31.6408 | +2.49678 deg | [+1.96158, +3.04919] |
| Qwen3.5 | depth `d1` | 0.59177 | 0.53342 | +0.05835 | [+0.03738, +0.08088] |
| Qwen3.5 | normal error | 29.8926 | 34.2056 | +4.31301 deg | [+3.39556, +5.30180] |
| Muse Glimmer | depth `d1` | 0.58936 | 0.56325 | +0.02611 | [+0.01287, +0.03955] |
| Muse Glimmer | normal error | 30.6672 | 31.4596 | +0.79244 deg | [+0.37697, +1.20646] |

All eight overall intervals exclude zero in the Projector's favour. The matched-arm
headline is therefore supported against test-image variation: on both measured geometry
tasks, every Projector beats its own final Tower Stage. This is a targeted test of the eight
headline comparisons, not a blanket interval over all 224 geometry cells.

Scene-specific intervals resolve in the same direction in fifteen of sixteen cases. The
exception is Kimi K2.6 depth outdoors, +0.00529 with interval [-0.00849, +0.01977]; its
overall and indoor effects resolve. The complete per-image predictions and scene-stratified
intervals are in the eight `results/bootstrap/geometry-*-projected-vs-tower.json` files.

The correction matters numerically. For example, Kimi K2.6 depth moves from the original
randomized-PCA gap of +0.0505 to +0.01737, and Muse Glimmer depth moves from +0.0540 to
+0.02611. Qwen3.5 depth also changes its selected rates. The direction and the headline
conclusion survive, but the original matched point estimates must not be presented as
exactly reproducible values.

## Amendment: raw-arm Projector intervals, September 1 2026

The same targeted comparison was run without the capacity reducer: `projected` against the
final `tower` Stage for both tasks and all four Projectors, using full patch tokens, the
six-rate grid, ten epochs, three seeds, and ten thousand paired resamples over the same 115
DIODE test images.

| Projector | task | `projected` | `tower` | advantage | paired 95% interval |
|---|---|---:|---:|---:|---:|
| Kimi K2.6 | depth `d1` | 0.48741 | 0.46781 | +0.01960 | [-0.00032, +0.04021] |
| Kimi K2.6 | normal error | 29.0061 | 32.1187 | +3.11261 deg | [+2.42151, +3.82282] |
| MoonViT-V2 | depth `d1` | 0.48626 | 0.45926 | +0.02700 | [+0.01129, +0.04268] |
| MoonViT-V2 | normal error | 30.4614 | 33.2883 | +2.82685 deg | [+2.15059, +3.54422] |
| Qwen3.5 | depth `d1` | 0.54075 | 0.38069 | +0.16007 | [+0.12108, +0.19774] |
| Qwen3.5 | normal error | 29.3152 | 33.7584 | +4.44317 deg | [+3.50759, +5.44265] |
| Muse Glimmer | depth `d1` | 0.50063 | 0.40792 | +0.09271 | [+0.06235, +0.12374] |
| Muse Glimmer | normal error | 30.9422 | 33.8354 | +2.89322 deg | [+2.37672, +3.44321] |

Seven of eight raw-arm intervals exclude zero in the Projector's favour. Kimi K2.6 depth is
the exception: its positive point estimate survives, but its lower bound misses zero by
0.00032 `d1`. This is not evidence of a loss, but it does prevent the universal claim that
both capacity arms resolve for every Projector. Combining both amendments, fifteen of the
sixteen Projector-versus-final-Tower comparisons resolve against test-image variation; all
sixteen point estimates favour the Projector.

Fourteen of sixteen raw scene-specific intervals resolve. Kimi K2.6 depth outdoors and
MoonViT-V2 depth outdoors cross zero. The raw outputs also reproduce most original selected
rates, but Qwen3.5 depth selects 1e-3 for `tower` rather than the matrix's 3e-3 and its fresh
gap is +0.16007 rather than +0.1962. The conclusion survives; the fresh bootstrap output is
the evidence for the interval claim.
