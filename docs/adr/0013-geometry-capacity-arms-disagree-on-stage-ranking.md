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
