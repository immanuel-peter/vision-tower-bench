# Roster run: six Towers, four Projectors

Every number here is new. The previous file reported two Towers, one of them a control with
no Projector, so every claim about the Projector rested on MoonViT-V2 alone. This run has
six Towers and four Projectors, 224 geometry cells and 224 semantic cells, and it does not
reproduce the headline claim.

The short version: on MoonViT-V2 the Projector moves depth eight times less than a step
that provably loses nothing, and that was read as evidence the Projector is harmless. Three
of the other Projectors do the opposite. They move the metric further than the lossless step
does, and they move it in the direction of better geometry. The case study was not
representative of the roster.

## Protocol

Both pillars run every cell on identical data with an identical head and an identical rate
grid, so cell-to-cell comparisons hold even where absolute levels do not.

Sections below name each Tower by the repository its adapter read at the time. Three of
those have since been republished and the adapters now load the releases, so a later run
will name `immanuelpeter/...` for the same weights (ADR-0018).

Geometry uses DIODE validation, 771 images, 325 indoors and 446 outdoor, prepared at native
768 by 1024 and scored on the centre 768 square that `square_crop` feeds each Stage
(ADR-0012). The split is 541 train, 115 val, 115 test, drawn once with a fixed seed. Depth
bins span 0 to 299.83 m, read from the prep manifest (ADR-0011). Heads are the Probe3D
multiscale decoder, 10 epochs, AdamW, batch 8. The grid is the six points ADR-0014 specifies,
`[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]`, searched on validation per cell, then retrained under
three seeds at the selected rate.

Semantics use ImageNet-100 validation, 13,000 images and 100 classes, cached at 448 square
and pooled to a 4x4 grid (ADR-0005). Each cell searches the eight points in
`vtb.probe_run.LEARNING_RATES`, `[3e-4 ... 1.0]`, over 20 epochs, and reports mean test
accuracy over three seeds. Attention and mean readouts both run. (Amended August 31: the
semantic cells were re-searched on the eleven points `[1e-5 ... 1.0]`; the re-run section at
the bottom of this file replaces the semantic results below in full.)

Both pillars run both capacity arms. `raw` trains the head at full token width, from
1,706,752 parameters at a 1024-wide geometry Stage up to 4,852,480 at 7168, and 2,152,036 up
to 8,443,492 on the semantic side. `matched` fits a frozen PCA reduction to 512 on each
cell's training split and trains 1,444,608 parameters everywhere for depth, 1,314,819 for
normals, and 1,627,748 for semantics (ADR-0008).

The original matched matrices used randomized PCA before fixing its seed, so their exact
PCA draws cannot be reconstructed. The semantic matched arm has since been regenerated in
full with the deterministic train-split-only reducer, and the matched JSONs and semantic
tables below are the corrected results. The geometry bootstrap reruns use the same reducer;
their corrected point estimates are reported alongside their intervals.

Cells: eight Relative Depth points each for DINOv2 and SigLIP2, which have no Projector, and
ten for the four Towers that do, adding `merged` and `projected` at the deepest point. That
is 56 cells per task-arm pair and 224 per pillar.

### What it cost

| pillar | invocations | lane-hours | wall on four lanes |
|---|---|---|---|
| semantic | 24 | 4.41 | 1.63 h |
| geometry, `raw` arm | 12 | 13.97 | 3.93 h |
| geometry, `matched` arm | 12 | 13.21 | 3.60 h |
| geometry total | 24 | 27.18 | 7.53 h |

The estimate in the brief was 28 to 33 lane-hours and 7 to 8 hours wall for geometry. It
holds, coming in just under the bottom of the range. There was no prior number for the
semantic pillar; 4.41 lane-hours is the first one.

Geometry ran as two four-lane phases, `raw` then `matched`, rather than one phase over both
arms. Both lane scripts weigh a job by cell count alone, and a `raw` cell costs about three
times a `matched` one, so a single phase puts one arm on each lane and the matched lanes
finish early and idle. Splitting by arm makes each phase uniform. The semantic pillar ran
before that was understood and shows the cost: its lanes came in at 0.62, 0.79, 1.37 and
1.63 lane-hours, so the run took 1.63 hours of wall time to do 4.41 lane-hours of work.
Weighting a job by cells times an arm factor would fix it.

Every cache landed within 1 percent of the predicted size, so no Stage is the wrong shape.
Geometry totalled 102.6 GB against 102.7 GB predicted, semantics 42.4 GB against 42.3 GB.

## 1. The Stage conclusion changes with the larger roster

The four-Projector result differs from the earlier MoonViT-V2 result.

The argument runs through a control. Merging four Tower tokens into one is a lossless
regrouping, verified bit-exact against the cache on disk with `torch.equal`, so any movement
in the metric from `tower` to `merged` is readout artifact rather than information change.
That movement is the yardstick. The question is whether the learned `merged` to `projected`
step stays under it.

On MoonViT-V2 it does, and this run reproduces that. In the capacity-matched arm `tower` to
`merged` moves depth `d1` by 0.0710 at 17.1 seed deviations while `merged` to `projected`
moves it by 0.0090 at 1.4, a ratio of 7.9 against the 8.2 reported from the two-model run.
Matched normals give 22.6.

No other Projector behaves that way.

| model | task | arm | `tower` to `merged` | `merged` to `projected` | ratio |
|---|---|---|---|---|---|
| moonvit_v2 | depth | matched | +0.0710 (17.1 sd) | +0.0090 (1.4 sd) | 7.9 |
| moonvit_v2 | normal | matched | -2.5689 (2.3 sd) | +0.1135 (0.2 sd) | 22.6 |
| kimi_k26 | depth | matched | +0.0113 (2.2 sd) | +0.0392 (6.6 sd) | 0.3 |
| kimi_k26 | normal | matched | +0.1730 (0.4 sd) | -2.3443 (4.4 sd) | 0.1 |
| muse_glimmer | depth | matched | -0.0151 (0.9 sd) | +0.0691 (34.4 sd) | 0.2 |
| muse_glimmer | normal | matched | +0.7109 (1.8 sd) | -1.6363 (3.9 sd) | 0.4 |
| qwen3_5 | depth | matched | -0.0017 (0.2 sd) | +0.0737 (13.5 sd) | 0.0 |
| qwen3_5 | normal | matched | +1.7134 (4.3 sd) | -6.2445 (22.2 sd) | 0.3 |

ADR-0010 nominates the matched arm for the headline. In it, one Projector of four sits below
its own lossless yardstick. The other three move the metric three to twenty times further
than the step that provably changes no information. The unmatched arm agrees on the shape
and disagrees on which models: there kimi_k26 depth reads 14.0 and MoonViT-V2 reads 1.6.

A later deterministic matched rerun preserves the strongest step: Muse Glimmer reads
0.58920 at `projected` against 0.52298 at `merged`, a +0.06622 `d1` difference with paired
95% image-bootstrap interval [+0.04992, +0.08338]. The original randomized-PCA point gap was
+0.0691. The corrected effect excludes zero overall and within both DIODE scene types.

A September 1 targeted rerun covers the headline comparison directly: `projected` against
the final `tower` Stage for all four Projectors and both measured geometry tasks. It uses
the deterministic matched reducer, the full six-rate grid, three seeds, and ten thousand
paired image resamples. Positive values below always favour `projected` (for normals the
advantage is `tower` error minus `projected` error).

| Projector | depth advantage (`d1`) | paired 95% interval | normal advantage (deg) | paired 95% interval |
|---|---:|---:|---:|---:|
| Kimi K2.6 | +0.01737 | [+0.00596, +0.02892] | +2.19797 | [+1.65383, +2.76855] |
| MoonViT-V2 | +0.07836 | [+0.05956, +0.09838] | +2.49678 | [+1.96158, +3.04919] |
| Qwen3.5 | +0.05835 | [+0.03738, +0.08088] | +4.31301 | [+3.39556, +5.30180] |
| Muse Glimmer | +0.02611 | [+0.01287, +0.03955] | +0.79244 | [+0.37697, +1.20646] |

All eight overall intervals exclude zero in the Projector's favour. This supports the
matched-arm headline against test-image variation; it does not put intervals on all 224
geometry cells. Fifteen of sixteen scene-specific intervals also resolve. Kimi K2.6 depth
outdoors is the exception, +0.00529 with interval [-0.00849, +0.01977], while its overall
and indoor effects resolve.

The deterministic correction changes some point gaps materially without changing their
direction. Kimi K2.6 depth falls from the original randomized-PCA +0.0505 to +0.01737, Muse
Glimmer depth from +0.0540 to +0.02611, and Qwen3.5 depth selects different rates. Use the
bootstrap files for corrected matched headline values rather than treating the old matched
matrix as exactly reproducible.

The corresponding raw-arm rerun uses full token width with the same six-rate grid, seeds,
test images, and resampling protocol:

| Projector | depth advantage (`d1`) | paired 95% interval | normal advantage (deg) | paired 95% interval |
|---|---:|---:|---:|---:|
| Kimi K2.6 | +0.01960 | [-0.00032, +0.04021] | +3.11261 | [+2.42151, +3.82282] |
| MoonViT-V2 | +0.02700 | [+0.01129, +0.04268] | +2.82685 | [+2.15059, +3.54422] |
| Qwen3.5 | +0.16007 | [+0.12108, +0.19774] | +4.44317 | [+3.50759, +5.44265] |
| Muse Glimmer | +0.09271 | [+0.06235, +0.12374] | +2.89322 | [+2.37672, +3.44321] |

Seven raw-arm intervals resolve. Kimi K2.6 raw depth is the sole exception: the point
estimate favours the Projector, but the interval crosses zero. Across both capacity arms,
fifteen of sixteen headline comparisons therefore resolve in the Projector's favour; all
sixteen point estimates do. The universal descriptive result survives, while the universal
inferential wording does not.

The direction matters as much as the size. Across all sixteen model-task-arm combinations
the `merged` to `projected` step improves the metric in fourteen. Both exceptions are
MoonViT-V2: unmatched depth at -0.0413, and matched normals at +0.1135, which is 0.2 seed
deviations and is not resolved by this run. The largest single move in the matrix is qwen3_5
unmatched depth, where `projected` beats `merged` by 0.1945 `d1` at 21.2 deviations.

The results contradict PLAN.md's prediction on both measured geometry tasks. The probes
usually read more task-relevant information after the Projector. The MoonViT-V2 control does
not generalise: in the other three Projectors, the learned step moves the metric farther than
the lossless regrouping does.

No Projector in the roster scores below its own `tower` on depth or surface normals in either
capacity arm. This run therefore finds no loss detectable by those probes. It does not test
all spatial information or the still-unscoped correspondence task.

## 2. Geometry peaks early, semantics climb to the end

Supporting hypothesis 1 holds, and it holds better than the Stage result did.

Geometry declines from a peak before the final layer in 23 of the 24 model-task-arm
combinations. The single exception is DINOv2 depth in the unmatched arm, which peaks at
Relative Depth 1.000 and therefore has nothing to fall to; its lead over the runner-up is
0.9 seed deviations, so that curve is flat over its last quarter rather than rising. The
capacity-matched arm of the same cell does peak early, at 0.875, and falls 0.0256 at 3.8
deviations.

Capacity-matched peaks, with the fall from peak to final layer:

| model | depth peak | falls | normal peak | falls |
|---|---|---|---|---|
| dinov2 | 0.875 | 0.0256 (3.8 sd) | 0.750 | 4.3426 (18.8 sd) |
| siglip2 | 0.630 | 0.0764 (19.2 sd) | 0.370 | 7.1108 (20.1 sd) |
| moonvit_v2 | 0.630 | 0.1167 (53.8 sd) | 0.370 | 4.7317 (3.7 sd) |
| kimi_k26 | 0.519 | 0.0593 (13.5 sd) | 0.370 | 5.1786 (7.8 sd) |
| qwen3_5 | 0.630 | 0.1422 (14.6 sd) | 0.370 | 10.2153 (13.6 sd) |
| muse_glimmer | 0.760 | 0.1227 (7.5 sd) | 0.620 | 7.2328 (16.1 sd) |

Where the spread swallows the peak, it swallows its position and not its existence. Six
combinations have a peak within one seed deviation of the runner-up: MoonViT-V2 matched
normals leads by 0.1 deviations, qwen3_5 matched normals by 0.3, qwen3_5 unmatched normals by
0.6, and both kimi_k26 unmatched cells by 0.0. In every one of those the fall to the final
layer is still 3.0 deviations or more, so the curve is measurably falling even though the
argmax is not resolved. Read those rows as "peaks somewhere in this region" rather than as a
located layer.

Semantics run the other way. Tower accuracy rises to the final layer in 19 of 24
point-estimate curves. Four of the five exceptions are Qwen3.5; the fifth is Kimi K2.6 raw
attention, whose 0.5-seed-deviation fall is noise. Under the pooled semantic protocol,
paired tests show that Qwen3.5's raw attention and raw mean declines resolve: the earlier
cells lead by +0.01282, interval
[+0.00530, +0.02034], and +0.01556, interval [+0.00530, +0.02632]. Its matched-attention
and matched-mean differences do not resolve: +0.00171, interval [-0.00632, +0.00957], and
+0.00154, interval [-0.00564, +0.00855]. Qwen3.5 is therefore the only Tower with a
resolved within-cache semantic decline into the last layer, but the evidence is specific to
both raw readouts rather than all four arms. The mean pooling control passed; ADR-0019 still
leaves the attention Relative Depth shape unvalidated against full patch tokens.

That is the result a reader can act on. It says which layer to tap for a spatial task, and
it says the answer is not the last one.

## 3. Semantics across Stages

MoonViT-V2 held flat across `tower`, `merged` and `projected`, and the roster mostly agrees,
but the steps are no longer all inside the noise.

| model | readout | arm | `tower` | `merged` | `projected` | `merged` to `projected` |
|---|---|---|---|---|---|---|
| moonvit_v2 | attention | matched | 0.8338 | 0.8395 | 0.8422 | +0.0027 (1.3 sd) |
| kimi_k26 | attention | matched | 0.8815 | 0.8756 | 0.8745 | -0.0011 (0.4 sd) |
| qwen3_5 | attention | matched | 0.8750 | 0.8769 | 0.8783 | +0.0014 (1.3 sd) |
| muse_glimmer | attention | matched | 0.9178 | 0.9154 | 0.9079 | -0.0075 (6.6 sd) |

On the headline attention-matched arm three of four Projectors hold flat, moving the metric
by 0.0027 or less against their own seed spread. Muse Glimmer does not: it loses 0.0075 at
6.6 deviations, and 0.0099 in total from `tower` to `projected`. That is a small number in
absolute terms and a solid one statistically.

(Amended September 1: the eleven-point deterministic matched rerun dissolves the Muse
Glimmer loss. Its `merged`-to-`projected` step reads +0.0042, and all four Projectors move by
at most 0.0051 in magnitude and 2.4 of the larger cell spread. The old 0.0075 loss was a
truncated-grid and randomized-reducer artifact. See the re-run section below.)

The mean-pooling arm is noisier and moves further. qwen3_5 matched mean drops 0.0194 from
`tower` to `merged` at 11.5 deviations then recovers 0.0174 at `projected`, and muse_glimmer
unmatched mean loses 0.0147 at 17.2 deviations across the Projector. The `tower` to `merged`
step is the lossless one, so a move of 0.0194 there is the same readout artifact the geometry
pillar measures, and it bounds how much of the neighbouring Projector step to believe.

(Amended September 1: deterministic matched mean reads 0.8692 at Qwen3.5 `tower`, 0.8542
at `merged`, and 0.8684 at `projected`. The same lossless-readout-artifact shape remains,
with a 0.0150 drop and 0.0142 recovery. The unmatched Muse Glimmer result is unchanged.)

Semantics survive the Connector in every model. The strong form of the original reading,
that the numbers are identical, holds for three Projectors out of four.

(Amended September 1: at the extended grid all four are nearly flat on the headline
attention-matched arm; no `merged`-to-`projected` difference exceeds 0.0051.)

## 4. Rankings change by task, and no Tower wins everywhere

The nominal best-Tower ranking changes by task and arm. These point estimates preceded the
paired-bootstrap analysis below, which does not resolve the semantic winners:

| task | arm | ranking |
|---|---|---|
| depth | matched | dinov2 0.7015 > qwen3_5 0.6736 > siglip2 0.6702 > kimi_k26 0.6595 > muse_glimmer 0.6588 > moonvit_v2 0.6546 |
| depth | unmatched | dinov2 0.6746 > qwen3_5 0.6439 > siglip2 0.6366 > muse_glimmer 0.6035 > moonvit_v2 0.5440 > kimi_k26 0.5089 |
| normal | matched | dinov2 18.85 > siglip2 23.90 > qwen3_5 24.24 > muse_glimmer 24.30 > kimi_k26 25.51 > moonvit_v2 27.03 |
| normal | unmatched | dinov2 19.46 > muse_glimmer 22.78 > qwen3_5 24.68 > siglip2 25.16 > moonvit_v2 27.97 > kimi_k26 30.34 |
| semantic, attention | matched | muse_glimmer 0.9178 > siglip2 0.9091 > dinov2 0.8988 > kimi_k26 0.8815 > qwen3_5 0.8776 > moonvit_v2 0.8338 |
| semantic, attention | unmatched | muse_glimmer 0.9145 > siglip2 0.9101 > dinov2 0.9055 > qwen3_5 0.8853 > kimi_k26 0.8786 > moonvit_v2 0.8373 |
| semantic, mean | matched | siglip2 0.9135 > muse_glimmer 0.9104 > dinov2 0.8916 > kimi_k26 0.8802 > qwen3_5 0.8709 > moonvit_v2 0.8415 |
| semantic, mean | unmatched | siglip2 0.9121 > muse_glimmer 0.9106 > dinov2 0.8916 > kimi_k26 0.8682 > qwen3_5 0.8593 > moonvit_v2 0.8147 |

(Amended September 1: on the eleven-point grid the attention rankings re-draw. The
deterministic matched arm keeps Muse Glimmer first, 0.9203 over SigLIP2's 0.9162. Unmatched
attention flips to SigLIP2 0.9171 over Muse Glimmer 0.9154 at 2.4 seed deviations - the roster's Muse-first
unmatched-attention column was a truncation artifact. Mean holds: SigLIP2 first in both
arms. A later paired bootstrap supersedes the winner claim: all four Muse-versus-SigLIP2
intervals cross zero. See the re-run and pooling-validation sections below.)

Three Towers take a first place: DINOv2 four times, Muse Glimmer twice, SigLIP2 twice. No
Tower wins everywhere, so nothing here undercuts the Capability Profile framing.

The split is cleaner than "no single winner" suggests. DINOv2 wins every geometry column and
no semantic column. Muse Glimmer and SigLIP2 split the semantic columns between them and win
no geometry column. The self-supervised control is the best geometry Tower in the roster and
third-best at classification; the two contrastive-trained Towers invert that exactly.

The margins are wide. DINOv2 leads matched normals by 5.05 degrees over SigLIP2 and unmatched
normals by 3.32 over Muse Glimmer, and it is third on both semantic readouts. kimi_k26 is
last on unmatched depth at 0.5089 and fourth on matched depth at 0.6595, which says more
about the capacity arms than about the Tower and is covered in section 5. MoonViT-V2 is last
on six of the eight columns and second to last on the other two, which is worth knowing given
that it carried every previous conclusion in this repo.

One caveat: DINOv2 trained on ImageNet, so its semantic column is not a clean comparison. It
still does not win there.

## 5. ADR-0013: the arms disagree in six pairs of eight, and MoonViT-V2 is still special

The capacity arms disagree about Stage ranking in six of the eight model-task pairs that
have a Projector, so the disagreement ADR-0013 recorded is general rather than a MoonViT-V2
quirk. That is not the whole answer though, and the detail is what the ADR needs.

| model | task | unmatched | matched | |
|---|---|---|---|---|
| kimi_k26 | depth | `projected` > `merged` > `tower` | `projected` > `merged` > `tower` | agree |
| kimi_k26 | normal | `projected` > `merged` > `tower` | `projected` > `tower` > `merged` | swap below the top |
| moonvit_v2 | depth | `merged` > `projected` > `tower` | `projected` > `merged` > `tower` | top changes |
| moonvit_v2 | normal | `projected` > `merged` > `tower` | `merged` > `projected` > `tower` | top changes |
| muse_glimmer | depth | `projected` > `merged` > `tower` | `projected` > `tower` > `merged` | swap below the top |
| muse_glimmer | normal | `projected` > `merged` > `tower` | `projected` > `tower` > `merged` | swap below the top |
| qwen3_5 | depth | `projected` > `merged` > `tower` | `projected` > `tower` > `merged` | swap below the top |
| qwen3_5 | normal | `projected` > `tower` > `merged` | `projected` > `tower` > `merged` | agree |

MoonViT-V2 is the only model where the arms disagree about which Stage ranks first, and it
does so on both tasks. In the other three Projectors `projected` ranks first in both arms
every time, and the swap is between `tower` and `merged` underneath it. That distinction is
the whole point, because the top of the ranking is what decides whether the Projector looks
harmful. Read against the roster, the disputed MoonViT-V2 gap is a property of MoonViT-V2
and not of the readout in general.

Rate selection still does not dissolve it, and the mechanism ADR-0013 proposed still shows.
Both unmatched MoonViT-V2 depth cells select 1e-3 while both matched ones select 1e-2, which
reproduces the earlier finding that rate headroom shrinks as the head widens. The wider
roster sharpens it: unmatched qwen3_5 `projected` selects 3e-2, the top of the widened grid,
and beats its own `merged` cell by 0.1945 there, while unmatched qwen3_5 `merged` selects
3e-4. Two Stages of one model, searched over the same grid, landing two orders of magnitude
apart. Comparing them at any single rate would be meaningless, which is the argument for
per-cell selection and against reading the unmatched arm as a like-for-like Stage comparison.

ADR-0013 needs updating in two places. The disagreement is not unique to MoonViT-V2, so the
ADR should stop implying it is. But the specific swap that changes the top-ranked Stage, and
therefore changes the Projector's verdict, is unique to MoonViT-V2 across four Projectors,
and the ADR should say so and stop treating that swap as the general case.

## 6. Grid edges: the six-point grid brackets nearly every optimum

9 of the 224 geometry cells select an edge, against 29 of 72 on the four-point grid. As a
share that is 4 percent against 40 percent. Widening the grid worked.

Selected rates across the matrix: 1e-4 five times, 3e-4 sixty-three, 1e-3 ninety-two, 3e-3
twenty, 1e-2 forty, 3e-2 four.

The nine edge cells, with whether the validation curve is still moving at the edge:

| model | task | arm | Stage | edge | still moving |
|---|---|---|---|---|---|
| kimi_k26 | normal | unmatched | `merged` | 1e-4 | no, plateau, 0.1013 over its neighbour |
| muse_glimmer | normal | unmatched | `projected` | 1e-4 | yes, by 0.5453 against spread 0.1447 |
| qwen3_5 | normal | unmatched | `merged` | 1e-4 | no, plateau, 0.3995 over its neighbour |
| qwen3_5 | normal | unmatched | `tower` | 1e-4 | yes, by 0.2828 against spread 0.2487 |
| qwen3_5 | normal | matched | `tower` | 1e-4 | no, plateau, 0.1071 over its neighbour |
| qwen3_5 | depth | unmatched | `projected` | 3e-2 | yes, by 0.0918 against spread 0.0032 |
| siglip2 | depth | unmatched | `tower` | 3e-2 | yes, by 0.0183 against spread 0.0070 |
| kimi_k26 | depth | matched | `projected` | 3e-2 | yes, by 0.0066 against spread 0.0049 |
| qwen3_5 | depth | matched | `projected` | 3e-2 | yes, by 0.0122 against spread 0.0059 |

Six of the nine are still climbing at the edge, so their optimum sits past it and their level
is reported below it. Three are plateaus where the edge beats its neighbour by less than the
run-to-run noise, and nothing is lost there.

The six live ones cluster. Five are `projected` or `tower` at the deepest point of qwen3_5,
kimi_k26 and siglip2, and four of those sit at the 3e-2 ceiling on depth. qwen3_5 `projected`
is the worst case: its validation curve reads 0.4736 at 1e-3, dips to 0.4077 at 3e-3, and
then climbs to 0.5211 at 3e-2, so that curve is not even unimodal and the grid stops while it
is rising. The 1e-4 floor is selected five times and only twice with a live curve, so the
bottom of the range is close to adequate and the top is not.

ADR-0014 should record that six points fixed the problem it was written about, and that the
remaining truncation is one-sided. Extending the ceiling to 1e-1 for the geometry pillar is
the cheap next change; extending the floor is not worth the cells.

### The semantic grid has the same problem, worse, and untested

107 of the 224 semantic cells select 3e-4, the floor of the eight-point semantic grid. The
split is entirely by readout and not by capacity arm:

| readout | arm | cells at the 3e-4 floor |
|---|---|---|
| attention | matched | 53 of 56 |
| attention | raw | 54 of 56 |
| mean | matched | 0 of 56 |
| mean | raw | 0 of 56 |

PLAN.md chose `[3e-4 ... 1.0]` to suit the small attention pool. It suits mean pooling
instead. Attention is the headline readout, so its absolute levels are reported below their
optimum throughout this file.

How far below is not knowable from these JSONs. `geometry_run` records a
`learning_rate_search` dict per cell and `probe_run` did not when this matrix ran, so there is
no way to check whether an attention cell's validation curve was still climbing at the floor.
`probe_run` now records it. Any re-run of this pillar should extend the grid downward and will
then be able to answer the question this one cannot.

(Amended August 31: the re-run at the bottom of this file answers it. The extended grid
reduced edge selections from 107 of 224 cells to 4, and the attention arm gained 0.002 to
0.011 top-1 at the deepest cells. Muse Glimmer's matched Projector loss in section 3 above
was one of those truncated cells.)

## What this run does not establish

541 training images for a head of 1.3 to 4.9 million parameters is thin. Absolute geometry
levels are noisy and should not be compared against published Probe3D numbers, and nobody has
published Probe3D numbers on DIODE anyway (ADR-0011). Every comparison above is between cells
that saw identical data, an identical head and an identical grid.

The lossless yardstick is verified bit-exact for MoonViT-V2 and Kimi K2.6 only. The
`tower` to `merged` step for qwen3_5 and Muse Glimmer is assumed lossless from the same
architectural argument but is not verified against the cache, and section 1 leans on it for
those two models. Writing that test is cheap and should happen before this result is
published.

The attention arm's absolute levels sit below their optimum everywhere, per section 6, and
the amount is unmeasured. (Amended August 31: measured by the eleven-point re-run at the
bottom of this file - 0.002 to 0.011 top-1 at the deepest cells, and the number of edge
selections fell from 107 of 224 cells to 4.) Rankings within a task are unaffected because every cell searched
the same grid.

Three seeds size the seed spread; they do not shrink it. Six of 24 geometry peak positions
are unresolved within that spread, and normals carry most of it.

Correspondence is still unscoped, the multilayer consistency run in ADR-0010 has not been
done, and the pooling validation ADR-0005 requires is still outstanding. The DIODE training
split was not downloaded. At 222 GB it waited on this run's answer, and the answer is that
the Stage question is now more interesting than it was, not less: three Projectors move
geometry further than a lossless step does, and 541 training images cannot say why.

## Depth

`d1`, `d2` and `d3` are the fraction of pixels within 1.25, 1.25^2 and 1.25^3 of the target
depth ratio, so higher is better. The seed spread on the selection metric is the population
standard deviation over three seeds.

### Unmatched arm

### facebook/dinov2-large, depth, unmatched

| Stage | Rel. Depth | Width | Grid | Params | LR | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.125 | 1024 | 32x32 | 1,706,752 | 0.001 | 0.3212 ± 0.0054 | 0.5850 | 0.7545 | 6.0997 |
| `tower` | 0.250 | 1024 | 32x32 | 1,706,752 | 0.001 | 0.3474 ± 0.0058 | 0.6272 | 0.7984 | 5.7509 |
| `tower` | 0.375 | 1024 | 32x32 | 1,706,752 | 0.001 | 0.3741 ± 0.0088 | 0.6625 | 0.8294 | 5.3665 |
| `tower` | 0.500 | 1024 | 32x32 | 1,706,752 | 0.001 | 0.5148 ± 0.0058 | 0.7943 | 0.9066 | 4.3680 |
| `tower` | 0.625 | 1024 | 32x32 | 1,706,752 | 0.01 | 0.6341 ± 0.0078 | 0.8608 | 0.9406 | 3.7988 |
| `tower` | 0.750 | 1024 | 32x32 | 1,706,752 | 0.001 | 0.6504 ± 0.0108 | 0.8713 | 0.9418 | 3.5472 |
| `tower` | 0.875 | 1024 | 32x32 | 1,706,752 | 0.001 | 0.6674 ± 0.0095 | 0.8754 | 0.9429 | 3.6059 |
| `tower` | 1.000 | 1024 | 32x32 | 1,706,752 | 0.003 | 0.6746 ± 0.0063 | 0.8801 | 0.9461 | 3.7051 |

### exolabs/Kimi-K2.6-vision, depth, unmatched

| Stage | Rel. Depth | Width | Grid | Params | LR | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.111 | 1152 | 32x32 | 1,772,288 | 0.001 | 0.1619 ± 0.0122 | 0.3049 | 0.4523 | 8.1958 |
| `tower` | 0.259 | 1152 | 32x32 | 1,772,288 | 0.001 | 0.1844 ± 0.0148 | 0.3644 | 0.5402 | 7.5768 |
| `tower` | 0.370 | 1152 | 32x32 | 1,772,288 | 0.001 | 0.2254 ± 0.0331 | 0.4586 | 0.6582 | 6.5966 |
| `tower` | 0.519 | 1152 | 32x32 | 1,772,288 | 0.001 | 0.4316 ± 0.0230 | 0.7131 | 0.8509 | 4.8488 |
| `tower` | 0.630 | 1152 | 32x32 | 1,772,288 | 0.001 | 0.4704 ± 0.0082 | 0.7408 | 0.8692 | 4.6780 |
| `tower` | 0.741 | 1152 | 32x32 | 1,772,288 | 0.001 | 0.5089 ± 0.0043 | 0.7781 | 0.8926 | 4.4644 |
| `tower` | 0.889 | 1152 | 32x32 | 1,772,288 | 0.001 | 0.5087 ± 0.0060 | 0.7749 | 0.8906 | 4.5195 |
| `tower` | 1.000 | 1152 | 32x32 | 1,772,288 | 0.001 | 0.4643 ± 0.0118 | 0.7399 | 0.8681 | 4.7129 |
| `merged` | 1.000 | 4608 | 16x16 | 3,541,760 | 0.003 | 0.4965 ± 0.0246 | 0.7646 | 0.8864 | 4.6680 |
| `projected` | 1.000 | 7168 | 16x16 | 4,852,480 | 0.003 | 0.4988 ± 0.0512 | 0.7667 | 0.8900 | 4.6819 |

### immanuelpeter/MoonViT-V2, depth, unmatched

| Stage | Rel. Depth | Width | Grid | Params | LR | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.111 | 1024 | 32x32 | 1,706,752 | 0.001 | 0.2899 ± 0.0046 | 0.5562 | 0.7355 | 6.2861 |
| `tower` | 0.259 | 1024 | 32x32 | 1,706,752 | 0.001 | 0.3295 ± 0.0081 | 0.6017 | 0.7712 | 5.8914 |
| `tower` | 0.370 | 1024 | 32x32 | 1,706,752 | 0.001 | 0.3281 ± 0.0129 | 0.6099 | 0.7894 | 5.5898 |
| `tower` | 0.519 | 1024 | 32x32 | 1,706,752 | 0.001 | 0.4404 ± 0.0098 | 0.7176 | 0.8566 | 4.8904 |
| `tower` | 0.630 | 1024 | 32x32 | 1,706,752 | 0.001 | 0.5440 ± 0.0047 | 0.7896 | 0.8938 | 4.2970 |
| `tower` | 0.741 | 1024 | 32x32 | 1,706,752 | 0.001 | 0.5071 ± 0.0067 | 0.7761 | 0.8879 | 4.3734 |
| `tower` | 0.889 | 1024 | 32x32 | 1,706,752 | 0.001 | 0.4437 ± 0.0100 | 0.7274 | 0.8592 | 4.7450 |
| `tower` | 1.000 | 1024 | 32x32 | 1,706,752 | 0.001 | 0.4562 ± 0.0154 | 0.7354 | 0.8613 | 4.7223 |
| `merged` | 1.000 | 4096 | 16x16 | 3,279,616 | 0.001 | 0.5217 ± 0.0060 | 0.7813 | 0.8901 | 4.4920 |
| `projected` | 1.000 | 7168 | 16x16 | 4,852,480 | 0.001 | 0.4804 ± 0.0076 | 0.7488 | 0.8687 | 4.7125 |

### meta-models/Muse-Glimmer-30B, depth, unmatched

| Stage | Rel. Depth | Width | Grid | Params | LR | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.120 | 1536 | 32x32 | 1,968,896 | 0.001 | 0.3485 ± 0.0065 | 0.6289 | 0.8040 | 5.6721 |
| `tower` | 0.240 | 1536 | 32x32 | 1,968,896 | 0.001 | 0.3787 ± 0.0045 | 0.6638 | 0.8254 | 5.4088 |
| `tower` | 0.380 | 1536 | 32x32 | 1,968,896 | 0.001 | 0.4462 ± 0.0085 | 0.7402 | 0.8778 | 4.8086 |
| `tower` | 0.500 | 1536 | 32x32 | 1,968,896 | 0.001 | 0.5870 ± 0.0046 | 0.8429 | 0.9279 | 4.1067 |
| `tower` | 0.620 | 1536 | 32x32 | 1,968,896 | 0.001 | 0.6035 ± 0.0076 | 0.8360 | 0.9283 | 3.8798 |
| `tower` | 0.760 | 1536 | 32x32 | 1,968,896 | 0.001 | 0.5925 ± 0.0024 | 0.8327 | 0.9237 | 3.9777 |
| `tower` | 0.880 | 1536 | 32x32 | 1,968,896 | 0.0003 | 0.4909 ± 0.0064 | 0.7683 | 0.8835 | 4.4339 |
| `tower` | 1.000 | 1536 | 32x32 | 1,968,896 | 0.001 | 0.4089 ± 0.0057 | 0.7022 | 0.8465 | 4.9569 |
| `merged` | 1.000 | 6144 | 16x16 | 4,328,192 | 0.001 | 0.4462 ± 0.0075 | 0.7291 | 0.8619 | 4.8987 |
| `projected` | 1.000 | 6656 | 16x16 | 4,590,336 | 0.001 | 0.4973 ± 0.0015 | 0.7660 | 0.8870 | 4.6870 |

### Qwen/Qwen3.8-27B, depth, unmatched

| Stage | Rel. Depth | Width | Grid | Params | LR | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.111 | 1152 | 28x28 | 1,772,288 | 0.001 | 0.3202 ± 0.0086 | 0.6063 | 0.7796 | 5.7524 |
| `tower` | 0.259 | 1152 | 28x28 | 1,772,288 | 0.001 | 0.4758 ± 0.0122 | 0.7532 | 0.8747 | 4.7118 |
| `tower` | 0.370 | 1152 | 28x28 | 1,772,288 | 0.001 | 0.5795 ± 0.0052 | 0.8137 | 0.9089 | 4.1574 |
| `tower` | 0.519 | 1152 | 28x28 | 1,772,288 | 0.001 | 0.6274 ± 0.0004 | 0.8507 | 0.9370 | 3.8631 |
| `tower` | 0.630 | 1152 | 28x28 | 1,772,288 | 0.001 | 0.6439 ± 0.0031 | 0.8558 | 0.9341 | 3.7988 |
| `tower` | 0.741 | 1152 | 28x28 | 1,772,288 | 0.001 | 0.5790 ± 0.0052 | 0.8248 | 0.9172 | 4.1324 |
| `tower` | 0.889 | 1152 | 28x28 | 1,772,288 | 0.001 | 0.5343 ± 0.0109 | 0.7920 | 0.8968 | 4.3964 |
| `tower` | 1.000 | 1152 | 28x28 | 1,772,288 | 0.003 | 0.3438 ± 0.0287 | 0.6196 | 0.7991 | 5.5974 |
| `merged` | 1.000 | 4608 | 14x14 | 3,541,760 | 0.0003 | 0.3455 ± 0.0126 | 0.6222 | 0.8025 | 5.4337 |
| `projected` | 1.000 | 5120 | 14x14 | 3,803,904 | 0.03 | 0.5400 ± 0.0032 | 0.7954 | 0.9022 | 4.5444 |

### google/siglip2-so400m-patch14-384, depth, unmatched

| Stage | Rel. Depth | Width | Grid | Params | LR | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.111 | 1152 | 32x32 | 1,772,288 | 0.001 | 0.3783 ± 0.0094 | 0.6630 | 0.8151 | 5.4106 |
| `tower` | 0.259 | 1152 | 32x32 | 1,772,288 | 0.001 | 0.5326 ± 0.0056 | 0.7952 | 0.9029 | 4.3421 |
| `tower` | 0.370 | 1152 | 32x32 | 1,772,288 | 0.001 | 0.5955 ± 0.0026 | 0.8271 | 0.9197 | 4.0041 |
| `tower` | 0.519 | 1152 | 32x32 | 1,772,288 | 0.03 | 0.6366 ± 0.0070 | 0.8516 | 0.9341 | 3.8029 |
| `tower` | 0.630 | 1152 | 32x32 | 1,772,288 | 0.001 | 0.6226 ± 0.0061 | 0.8374 | 0.9238 | 3.8483 |
| `tower` | 0.741 | 1152 | 32x32 | 1,772,288 | 0.001 | 0.6181 ± 0.0028 | 0.8392 | 0.9251 | 3.9004 |
| `tower` | 0.889 | 1152 | 32x32 | 1,772,288 | 0.001 | 0.5926 ± 0.0079 | 0.8303 | 0.9181 | 4.0557 |
| `tower` | 1.000 | 1152 | 32x32 | 1,772,288 | 0.001 | 0.5820 ± 0.0011 | 0.8263 | 0.9185 | 4.1100 |

### Capacity-matched arm

### facebook/dinov2-large, depth, capacity-matched

| Stage | Rel. Depth | Width | Grid | Params | LR | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.125 | 512 | 32x32 | 1,444,608 | 0.01 | 0.4566 ± 0.0100 | 0.7203 | 0.8430 | 5.3309 |
| `tower` | 0.250 | 512 | 32x32 | 1,444,608 | 0.003 | 0.4798 ± 0.0080 | 0.7576 | 0.8856 | 4.8621 |
| `tower` | 0.375 | 512 | 32x32 | 1,444,608 | 0.01 | 0.5371 ± 0.0306 | 0.8059 | 0.9074 | 4.4959 |
| `tower` | 0.500 | 512 | 32x32 | 1,444,608 | 0.01 | 0.6410 ± 0.0040 | 0.8639 | 0.9372 | 3.9311 |
| `tower` | 0.625 | 512 | 32x32 | 1,444,608 | 0.01 | 0.6691 ± 0.0073 | 0.8743 | 0.9442 | 3.5981 |
| `tower` | 0.750 | 512 | 32x32 | 1,444,608 | 0.01 | 0.6957 ± 0.0025 | 0.8898 | 0.9499 | 3.4116 |
| `tower` | 0.875 | 512 | 32x32 | 1,444,608 | 0.01 | 0.7015 ± 0.0061 | 0.8942 | 0.9503 | 3.4973 |
| `tower` | 1.000 | 512 | 32x32 | 1,444,608 | 0.01 | 0.6759 ± 0.0072 | 0.8777 | 0.9465 | 3.6944 |

### exolabs/Kimi-K2.6-vision, depth, capacity-matched

| Stage | Rel. Depth | Width | Grid | Params | LR | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.111 | 512 | 32x32 | 1,444,608 | 0.003 | 0.3840 ± 0.0073 | 0.6518 | 0.8215 | 5.4866 |
| `tower` | 0.259 | 512 | 32x32 | 1,444,608 | 0.01 | 0.5553 ± 0.0144 | 0.8172 | 0.9154 | 4.2740 |
| `tower` | 0.370 | 512 | 32x32 | 1,444,608 | 0.01 | 0.6157 ± 0.0093 | 0.8432 | 0.9301 | 3.9731 |
| `tower` | 0.519 | 512 | 32x32 | 1,444,608 | 0.01 | 0.6595 ± 0.0057 | 0.8645 | 0.9392 | 3.7666 |
| `tower` | 0.630 | 512 | 32x32 | 1,444,608 | 0.01 | 0.6470 ± 0.0022 | 0.8557 | 0.9346 | 3.7515 |
| `tower` | 0.741 | 512 | 32x32 | 1,444,608 | 0.01 | 0.6489 ± 0.0002 | 0.8579 | 0.9338 | 3.7279 |
| `tower` | 0.889 | 512 | 32x32 | 1,444,608 | 0.01 | 0.6482 ± 0.0032 | 0.8578 | 0.9360 | 3.7896 |
| `tower` | 1.000 | 512 | 32x32 | 1,444,608 | 0.01 | 0.6002 ± 0.0025 | 0.8326 | 0.9238 | 3.9968 |
| `merged` | 1.000 | 512 | 16x16 | 1,444,608 | 0.01 | 0.6115 ± 0.0068 | 0.8404 | 0.9280 | 4.1214 |
| `projected` | 1.000 | 512 | 16x16 | 1,444,608 | 0.03 | 0.6507 ± 0.0049 | 0.8557 | 0.9341 | 4.0188 |

### immanuelpeter/MoonViT-V2, depth, capacity-matched

| Stage | Rel. Depth | Width | Grid | Params | LR | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.111 | 512 | 32x32 | 1,444,608 | 0.01 | 0.4106 ± 0.0105 | 0.6935 | 0.8479 | 5.3333 |
| `tower` | 0.259 | 512 | 32x32 | 1,444,608 | 0.01 | 0.5253 ± 0.0070 | 0.7896 | 0.8978 | 4.5011 |
| `tower` | 0.370 | 512 | 32x32 | 1,444,608 | 0.01 | 0.5817 ± 0.0040 | 0.8216 | 0.9144 | 4.2111 |
| `tower` | 0.519 | 512 | 32x32 | 1,444,608 | 0.01 | 0.6188 ± 0.0031 | 0.8420 | 0.9293 | 3.9695 |
| `tower` | 0.630 | 512 | 32x32 | 1,444,608 | 0.01 | 0.6546 ± 0.0029 | 0.8574 | 0.9370 | 3.7831 |
| `tower` | 0.741 | 512 | 32x32 | 1,444,608 | 0.01 | 0.6307 ± 0.0004 | 0.8565 | 0.9366 | 3.8183 |
| `tower` | 0.889 | 512 | 32x32 | 1,444,608 | 0.01 | 0.6003 ± 0.0029 | 0.8340 | 0.9271 | 3.9775 |
| `tower` | 1.000 | 512 | 32x32 | 1,444,608 | 0.003 | 0.5379 ± 0.0010 | 0.7927 | 0.8968 | 4.3841 |
| `merged` | 1.000 | 512 | 16x16 | 1,444,608 | 0.01 | 0.6089 ± 0.0058 | 0.8358 | 0.9213 | 4.2521 |
| `projected` | 1.000 | 512 | 16x16 | 1,444,608 | 0.01 | 0.6179 ± 0.0066 | 0.8399 | 0.9227 | 4.2287 |

### meta-models/Muse-Glimmer-30B, depth, capacity-matched

| Stage | Rel. Depth | Width | Grid | Params | LR | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.120 | 512 | 32x32 | 1,444,608 | 0.003 | 0.4239 ± 0.0025 | 0.7200 | 0.8682 | 5.0048 |
| `tower` | 0.240 | 512 | 32x32 | 1,444,608 | 0.003 | 0.4525 ± 0.0067 | 0.7420 | 0.8797 | 4.7852 |
| `tower` | 0.380 | 512 | 32x32 | 1,444,608 | 0.003 | 0.5224 ± 0.0036 | 0.8018 | 0.9112 | 4.3362 |
| `tower` | 0.500 | 512 | 32x32 | 1,444,608 | 0.003 | 0.5972 ± 0.0005 | 0.8489 | 0.9330 | 3.9369 |
| `tower` | 0.620 | 512 | 32x32 | 1,444,608 | 0.003 | 0.6440 ± 0.0023 | 0.8629 | 0.9402 | 3.6510 |
| `tower` | 0.760 | 512 | 32x32 | 1,444,608 | 0.003 | 0.6588 ± 0.0010 | 0.8666 | 0.9407 | 3.6696 |
| `tower` | 0.880 | 512 | 32x32 | 1,444,608 | 0.003 | 0.6344 ± 0.0011 | 0.8542 | 0.9348 | 3.7381 |
| `tower` | 1.000 | 512 | 32x32 | 1,444,608 | 0.01 | 0.5361 ± 0.0232 | 0.7957 | 0.9032 | 4.3441 |
| `merged` | 1.000 | 512 | 16x16 | 1,444,608 | 0.003 | 0.5210 ± 0.0022 | 0.7883 | 0.9013 | 4.5225 |
| `projected` | 1.000 | 512 | 16x16 | 1,444,608 | 0.003 | 0.5901 ± 0.0018 | 0.8247 | 0.9173 | 4.1964 |

### Qwen/Qwen3.8-27B, depth, capacity-matched

| Stage | Rel. Depth | Width | Grid | Params | LR | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.111 | 512 | 28x28 | 1,444,608 | 0.01 | 0.4913 ± 0.0056 | 0.7639 | 0.8824 | 4.7828 |
| `tower` | 0.259 | 512 | 28x28 | 1,444,608 | 0.01 | 0.6042 ± 0.0026 | 0.8396 | 0.9279 | 4.0964 |
| `tower` | 0.370 | 512 | 28x28 | 1,444,608 | 0.01 | 0.6404 ± 0.0039 | 0.8622 | 0.9388 | 3.8311 |
| `tower` | 0.519 | 512 | 28x28 | 1,444,608 | 0.01 | 0.6597 ± 0.0057 | 0.8733 | 0.9436 | 3.7109 |
| `tower` | 0.630 | 512 | 28x28 | 1,444,608 | 0.01 | 0.6736 ± 0.0094 | 0.8753 | 0.9450 | 3.6788 |
| `tower` | 0.741 | 512 | 28x28 | 1,444,608 | 0.01 | 0.6644 ± 0.0050 | 0.8721 | 0.9433 | 3.7642 |
| `tower` | 0.889 | 512 | 28x28 | 1,444,608 | 0.01 | 0.6383 ± 0.0077 | 0.8601 | 0.9387 | 3.8930 |
| `tower` | 1.000 | 512 | 28x28 | 1,444,608 | 0.001 | 0.5314 ± 0.0101 | 0.7879 | 0.8949 | 4.4340 |
| `merged` | 1.000 | 512 | 14x14 | 1,444,608 | 0.001 | 0.5297 ± 0.0050 | 0.7886 | 0.8933 | 4.5699 |
| `projected` | 1.000 | 512 | 14x14 | 1,444,608 | 0.03 | 0.6034 ± 0.0059 | 0.8363 | 0.9226 | 4.2719 |

### google/siglip2-so400m-patch14-384, depth, capacity-matched

| Stage | Rel. Depth | Width | Grid | Params | LR | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.111 | 512 | 32x32 | 1,444,608 | 0.01 | 0.4819 ± 0.0126 | 0.7609 | 0.8826 | 4.7930 |
| `tower` | 0.259 | 512 | 32x32 | 1,444,608 | 0.01 | 0.6094 ± 0.0033 | 0.8498 | 0.9317 | 3.9696 |
| `tower` | 0.370 | 512 | 32x32 | 1,444,608 | 0.01 | 0.6489 ± 0.0022 | 0.8601 | 0.9353 | 3.9627 |
| `tower` | 0.519 | 512 | 32x32 | 1,444,608 | 0.01 | 0.6671 ± 0.0035 | 0.8673 | 0.9400 | 3.8251 |
| `tower` | 0.630 | 512 | 32x32 | 1,444,608 | 0.01 | 0.6702 ± 0.0026 | 0.8669 | 0.9391 | 3.7654 |
| `tower` | 0.741 | 512 | 32x32 | 1,444,608 | 0.01 | 0.6589 ± 0.0035 | 0.8651 | 0.9375 | 3.8168 |
| `tower` | 0.889 | 512 | 32x32 | 1,444,608 | 0.003 | 0.6346 ± 0.0032 | 0.8476 | 0.9315 | 3.9103 |
| `tower` | 1.000 | 512 | 32x32 | 1,444,608 | 0.01 | 0.5938 ± 0.0050 | 0.8359 | 0.9251 | 4.0305 |

## Surface normals

`mean_deg` and `rmse` are angular error in degrees, so lower is better. `d1`, `d2` and `d3`
are the fraction of annotated pixels within 11.25, 22.5 and 30 degrees.

### Unmatched arm

### facebook/dinov2-large, normal, unmatched

| Stage | Rel. Depth | Width | Grid | Params | LR | mean_deg | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.125 | 1024 | 32x32 | 1,576,963 | 0.001 | 33.2976 ± 0.0178 | 0.1892 | 0.3802 | 0.5070 | 39.6261 |
| `tower` | 0.250 | 1024 | 32x32 | 1,576,963 | 0.001 | 31.1591 ± 0.1627 | 0.2204 | 0.4263 | 0.5526 | 37.7804 |
| `tower` | 0.375 | 1024 | 32x32 | 1,576,963 | 0.001 | 28.1655 ± 0.0870 | 0.2683 | 0.5002 | 0.6195 | 35.1521 |
| `tower` | 0.500 | 1024 | 32x32 | 1,576,963 | 0.001 | 23.5468 ± 0.2720 | 0.3719 | 0.6150 | 0.7144 | 30.9337 |
| `tower` | 0.625 | 1024 | 32x32 | 1,576,963 | 0.001 | 20.2889 ± 0.4092 | 0.4690 | 0.6940 | 0.7733 | 28.0600 |
| `tower` | 0.750 | 1024 | 32x32 | 1,576,963 | 0.0003 | 19.4649 ± 0.0948 | 0.5114 | 0.7153 | 0.7832 | 27.4754 |
| `tower` | 0.875 | 1024 | 32x32 | 1,576,963 | 0.0003 | 20.2905 ± 0.1650 | 0.4875 | 0.6987 | 0.7711 | 28.3623 |
| `tower` | 1.000 | 1024 | 32x32 | 1,576,963 | 0.001 | 22.2159 ± 0.2773 | 0.4203 | 0.6570 | 0.7418 | 30.1264 |

### exolabs/Kimi-K2.6-vision, normal, unmatched

| Stage | Rel. Depth | Width | Grid | Params | LR | mean_deg | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.111 | 1152 | 32x32 | 1,642,499 | 0.0003 | 36.1312 ± 0.5243 | 0.1410 | 0.3179 | 0.4431 | 41.9055 |
| `tower` | 0.259 | 1152 | 32x32 | 1,642,499 | 0.0003 | 32.5895 ± 1.4177 | 0.1916 | 0.3856 | 0.5127 | 38.7008 |
| `tower` | 0.370 | 1152 | 32x32 | 1,642,499 | 0.0003 | 30.7247 ± 0.7791 | 0.2253 | 0.4348 | 0.5572 | 37.1985 |
| `tower` | 0.519 | 1152 | 32x32 | 1,642,499 | 0.0003 | 30.7084 ± 0.7363 | 0.2407 | 0.4386 | 0.5565 | 37.4285 |
| `tower` | 0.630 | 1152 | 32x32 | 1,642,499 | 0.0003 | 30.4836 ± 0.7852 | 0.2414 | 0.4485 | 0.5653 | 37.2977 |
| `tower` | 0.741 | 1152 | 32x32 | 1,642,499 | 0.0003 | 30.3659 ± 0.6948 | 0.2416 | 0.4543 | 0.5691 | 37.2275 |
| `tower` | 0.889 | 1152 | 32x32 | 1,642,499 | 0.0003 | 30.3440 ± 0.7691 | 0.2400 | 0.4545 | 0.5712 | 37.1964 |
| `tower` | 1.000 | 1152 | 32x32 | 1,642,499 | 0.0003 | 32.1145 ± 0.2873 | 0.2114 | 0.4119 | 0.5325 | 38.6795 |
| `merged` | 1.000 | 4608 | 16x16 | 3,411,971 | 0.0001 | 31.7011 ± 0.1206 | 0.2158 | 0.4176 | 0.5393 | 38.3063 |
| `projected` | 1.000 | 7168 | 16x16 | 4,722,691 | 0.0003 | 28.9968 ± 0.4593 | 0.2754 | 0.4985 | 0.6119 | 36.5440 |

### immanuelpeter/MoonViT-V2, normal, unmatched

| Stage | Rel. Depth | Width | Grid | Params | LR | mean_deg | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.111 | 1024 | 32x32 | 1,576,963 | 0.0003 | 35.2611 ± 0.1525 | 0.1671 | 0.3503 | 0.4749 | 41.5687 |
| `tower` | 0.259 | 1024 | 32x32 | 1,576,963 | 0.0003 | 31.9093 ± 0.2385 | 0.2063 | 0.4146 | 0.5400 | 38.4130 |
| `tower` | 0.370 | 1024 | 32x32 | 1,576,963 | 0.0003 | 30.5854 ± 0.2562 | 0.2272 | 0.4417 | 0.5659 | 37.2110 |
| `tower` | 0.519 | 1024 | 32x32 | 1,576,963 | 0.0003 | 29.4626 ± 0.5316 | 0.2538 | 0.4707 | 0.5901 | 36.3112 |
| `tower` | 0.630 | 1024 | 32x32 | 1,576,963 | 0.0003 | 27.9684 ± 0.3681 | 0.2817 | 0.5096 | 0.6237 | 35.0491 |
| `tower` | 0.741 | 1024 | 32x32 | 1,576,963 | 0.0003 | 30.0627 ± 0.5840 | 0.2451 | 0.4538 | 0.5721 | 36.8059 |
| `tower` | 0.889 | 1024 | 32x32 | 1,576,963 | 0.0003 | 31.0717 ± 0.2614 | 0.2262 | 0.4305 | 0.5489 | 37.6195 |
| `tower` | 1.000 | 1024 | 32x32 | 1,576,963 | 0.0003 | 33.3010 ± 0.3408 | 0.2043 | 0.3903 | 0.5079 | 39.8396 |
| `merged` | 1.000 | 4096 | 16x16 | 3,149,827 | 0.0003 | 31.4798 ± 1.5624 | 0.2309 | 0.4359 | 0.5512 | 38.4114 |
| `projected` | 1.000 | 7168 | 16x16 | 4,722,691 | 0.0003 | 30.4365 ± 0.4786 | 0.2509 | 0.4665 | 0.5791 | 37.8141 |

### meta-models/Muse-Glimmer-30B, normal, unmatched

| Stage | Rel. Depth | Width | Grid | Params | LR | mean_deg | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.120 | 1536 | 32x32 | 1,839,107 | 0.0003 | 30.7638 ± 0.1683 | 0.2259 | 0.4413 | 0.5653 | 37.5004 |
| `tower` | 0.240 | 1536 | 32x32 | 1,839,107 | 0.0003 | 30.2688 ± 0.0780 | 0.2339 | 0.4540 | 0.5749 | 37.0959 |
| `tower` | 0.380 | 1536 | 32x32 | 1,839,107 | 0.0003 | 28.5679 ± 0.1209 | 0.2636 | 0.4924 | 0.6100 | 35.5533 |
| `tower` | 0.500 | 1536 | 32x32 | 1,839,107 | 0.0003 | 25.3331 ± 0.1494 | 0.3312 | 0.5728 | 0.6804 | 32.7601 |
| `tower` | 0.620 | 1536 | 32x32 | 1,839,107 | 0.0003 | 22.7825 ± 0.1977 | 0.3992 | 0.6428 | 0.7327 | 30.5146 |
| `tower` | 0.760 | 1536 | 32x32 | 1,839,107 | 0.0003 | 29.1317 ± 0.2392 | 0.2692 | 0.4832 | 0.5972 | 36.3489 |
| `tower` | 0.880 | 1536 | 32x32 | 1,839,107 | 0.0003 | 32.5051 ± 0.4479 | 0.2222 | 0.4138 | 0.5301 | 39.3761 |
| `tower` | 1.000 | 1536 | 32x32 | 1,839,107 | 0.0003 | 33.8248 ± 0.3720 | 0.1836 | 0.3705 | 0.4926 | 40.1752 |
| `merged` | 1.000 | 6144 | 16x16 | 4,198,403 | 0.0003 | 32.4033 ± 0.7558 | 0.2096 | 0.4090 | 0.5285 | 39.1869 |
| `projected` | 1.000 | 6656 | 16x16 | 4,460,547 | 0.0001 | 30.9194 ± 0.1447 | 0.2348 | 0.4442 | 0.5622 | 38.0412 |

### Qwen/Qwen3.8-27B, normal, unmatched

| Stage | Rel. Depth | Width | Grid | Params | LR | mean_deg | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.111 | 1152 | 28x28 | 1,642,499 | 0.0003 | 32.1732 ± 1.1640 | 0.2066 | 0.4116 | 0.5348 | 38.7828 |
| `tower` | 0.259 | 1152 | 28x28 | 1,642,499 | 0.0003 | 26.6246 ± 0.7241 | 0.2962 | 0.5436 | 0.6576 | 33.7496 |
| `tower` | 0.370 | 1152 | 28x28 | 1,642,499 | 0.0003 | 25.2558 ± 1.0400 | 0.3312 | 0.5778 | 0.6847 | 32.6247 |
| `tower` | 0.519 | 1152 | 28x28 | 1,642,499 | 0.0003 | 24.6806 ± 0.8037 | 0.3473 | 0.5950 | 0.6977 | 32.3075 |
| `tower` | 0.630 | 1152 | 28x28 | 1,642,499 | 0.0003 | 25.6999 ± 0.7823 | 0.3240 | 0.5700 | 0.6780 | 33.2389 |
| `tower` | 0.741 | 1152 | 28x28 | 1,642,499 | 0.0003 | 27.5391 ± 0.7395 | 0.2933 | 0.5247 | 0.6360 | 34.9947 |
| `tower` | 0.889 | 1152 | 28x28 | 1,642,499 | 0.0003 | 29.4470 ± 0.4981 | 0.2605 | 0.4787 | 0.5928 | 36.5895 |
| `tower` | 1.000 | 1152 | 28x28 | 1,642,499 | 0.0001 | 33.7285 ± 0.2487 | 0.1927 | 0.3839 | 0.5001 | 40.1323 |
| `merged` | 1.000 | 4608 | 14x14 | 3,411,971 | 0.0001 | 36.0520 ± 0.5543 | 0.1635 | 0.3381 | 0.4521 | 42.1390 |
| `projected` | 1.000 | 5120 | 14x14 | 3,674,115 | 0.0003 | 29.4337 ± 0.4774 | 0.2630 | 0.4858 | 0.6006 | 36.8478 |

### google/siglip2-so400m-patch14-384, normal, unmatched

| Stage | Rel. Depth | Width | Grid | Params | LR | mean_deg | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.111 | 1152 | 32x32 | 1,642,499 | 0.0003 | 31.1498 ± 0.5694 | 0.2140 | 0.4292 | 0.5550 | 37.7250 |
| `tower` | 0.259 | 1152 | 32x32 | 1,642,499 | 0.0003 | 26.6409 ± 0.5423 | 0.2909 | 0.5352 | 0.6539 | 33.6841 |
| `tower` | 0.370 | 1152 | 32x32 | 1,642,499 | 0.0003 | 25.1602 ± 0.6379 | 0.3319 | 0.5771 | 0.6834 | 32.5233 |
| `tower` | 0.519 | 1152 | 32x32 | 1,642,499 | 0.0003 | 26.3867 ± 0.7395 | 0.3089 | 0.5492 | 0.6604 | 33.8426 |
| `tower` | 0.630 | 1152 | 32x32 | 1,642,499 | 0.0003 | 28.2179 ± 0.7949 | 0.2769 | 0.5037 | 0.6194 | 35.4383 |
| `tower` | 0.741 | 1152 | 32x32 | 1,642,499 | 0.0003 | 29.0876 ± 0.6317 | 0.2663 | 0.4857 | 0.5988 | 36.2216 |
| `tower` | 0.889 | 1152 | 32x32 | 1,642,499 | 0.0003 | 30.1578 ± 0.5761 | 0.2495 | 0.4617 | 0.5758 | 37.1682 |
| `tower` | 1.000 | 1152 | 32x32 | 1,642,499 | 0.0003 | 30.6009 ± 0.3867 | 0.2433 | 0.4533 | 0.5659 | 37.5917 |

### Capacity-matched arm

### facebook/dinov2-large, normal, capacity-matched

| Stage | Rel. Depth | Width | Grid | Params | LR | mean_deg | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.125 | 512 | 32x32 | 1,314,819 | 0.001 | 30.8478 ± 0.1671 | 0.2287 | 0.4425 | 0.5677 | 37.6856 |
| `tower` | 0.250 | 512 | 32x32 | 1,314,819 | 0.003 | 29.8682 ± 2.1282 | 0.2466 | 0.4653 | 0.5879 | 36.8131 |
| `tower` | 0.375 | 512 | 32x32 | 1,314,819 | 0.003 | 25.4657 ± 0.6502 | 0.3267 | 0.5790 | 0.6819 | 32.9074 |
| `tower` | 0.500 | 512 | 32x32 | 1,314,819 | 0.003 | 22.7496 ± 0.6949 | 0.3952 | 0.6408 | 0.7334 | 30.2997 |
| `tower` | 0.625 | 512 | 32x32 | 1,314,819 | 0.001 | 20.2639 ± 0.1659 | 0.4744 | 0.6952 | 0.7730 | 28.1088 |
| `tower` | 0.750 | 512 | 32x32 | 1,314,819 | 0.001 | 18.8508 ± 0.2614 | 0.5336 | 0.7271 | 0.7915 | 26.9678 |
| `tower` | 0.875 | 512 | 32x32 | 1,314,819 | 0.001 | 20.3814 ± 0.3886 | 0.4803 | 0.6992 | 0.7714 | 28.3337 |
| `tower` | 1.000 | 512 | 32x32 | 1,314,819 | 0.0003 | 23.1934 ± 0.1973 | 0.4029 | 0.6315 | 0.7212 | 31.1996 |

### exolabs/Kimi-K2.6-vision, normal, capacity-matched

| Stage | Rel. Depth | Width | Grid | Params | LR | mean_deg | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.111 | 512 | 32x32 | 1,314,819 | 0.001 | 31.7331 ± 0.3865 | 0.2121 | 0.4185 | 0.5455 | 38.4190 |
| `tower` | 0.259 | 512 | 32x32 | 1,314,819 | 0.001 | 26.6648 ± 0.6773 | 0.3035 | 0.5474 | 0.6574 | 33.9686 |
| `tower` | 0.370 | 512 | 32x32 | 1,314,819 | 0.001 | 25.5109 ± 0.8880 | 0.3308 | 0.5758 | 0.6811 | 32.9608 |
| `tower` | 0.519 | 512 | 32x32 | 1,314,819 | 0.001 | 27.2178 ± 1.3838 | 0.3041 | 0.5347 | 0.6435 | 34.6541 |
| `tower` | 0.630 | 512 | 32x32 | 1,314,819 | 0.001 | 27.9056 ± 1.2596 | 0.2881 | 0.5187 | 0.6291 | 35.2715 |
| `tower` | 0.741 | 512 | 32x32 | 1,314,819 | 0.001 | 28.9012 ± 1.1858 | 0.2719 | 0.4946 | 0.6064 | 36.1333 |
| `tower` | 0.889 | 512 | 32x32 | 1,314,819 | 0.0003 | 28.9423 ± 0.4194 | 0.2660 | 0.4947 | 0.6100 | 36.2404 |
| `tower` | 1.000 | 512 | 32x32 | 1,314,819 | 0.001 | 30.6895 ± 0.3123 | 0.2395 | 0.4554 | 0.5734 | 37.8288 |
| `merged` | 1.000 | 512 | 16x16 | 1,314,819 | 0.001 | 30.8625 ± 0.4531 | 0.2367 | 0.4505 | 0.5682 | 37.9534 |
| `projected` | 1.000 | 512 | 16x16 | 1,314,819 | 0.001 | 28.5182 ± 0.6087 | 0.2875 | 0.5171 | 0.6260 | 36.2813 |

### immanuelpeter/MoonViT-V2, normal, capacity-matched

| Stage | Rel. Depth | Width | Grid | Params | LR | mean_deg | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.111 | 512 | 32x32 | 1,314,819 | 0.001 | 31.5431 ± 0.8915 | 0.2273 | 0.4331 | 0.5526 | 38.4678 |
| `tower` | 0.259 | 512 | 32x32 | 1,314,819 | 0.001 | 27.9846 ± 1.0336 | 0.2790 | 0.5170 | 0.6303 | 35.1923 |
| `tower` | 0.370 | 512 | 32x32 | 1,314,819 | 0.001 | 27.0277 ± 1.1372 | 0.2975 | 0.5345 | 0.6463 | 34.2448 |
| `tower` | 0.519 | 512 | 32x32 | 1,314,819 | 0.001 | 27.4822 ± 3.1567 | 0.3057 | 0.5372 | 0.6370 | 34.8134 |
| `tower` | 0.630 | 512 | 32x32 | 1,314,819 | 0.001 | 27.1783 ± 2.7715 | 0.3072 | 0.5351 | 0.6386 | 34.4793 |
| `tower` | 0.741 | 512 | 32x32 | 1,314,819 | 0.001 | 28.3899 ± 2.9250 | 0.2869 | 0.5020 | 0.6162 | 35.5862 |
| `tower` | 0.889 | 512 | 32x32 | 1,314,819 | 0.001 | 29.6773 ± 2.3439 | 0.2639 | 0.4789 | 0.5898 | 36.7852 |
| `tower` | 1.000 | 512 | 32x32 | 1,314,819 | 0.001 | 31.7594 ± 1.4007 | 0.2374 | 0.4379 | 0.5488 | 38.9399 |
| `merged` | 1.000 | 512 | 16x16 | 1,314,819 | 0.001 | 29.1905 ± 0.6670 | 0.2767 | 0.5002 | 0.6099 | 36.8601 |
| `projected` | 1.000 | 512 | 16x16 | 1,314,819 | 0.001 | 29.3040 ± 0.6480 | 0.2730 | 0.4972 | 0.6077 | 36.9290 |

### meta-models/Muse-Glimmer-30B, normal, capacity-matched

| Stage | Rel. Depth | Width | Grid | Params | LR | mean_deg | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.120 | 512 | 32x32 | 1,314,819 | 0.0003 | 30.1532 ± 0.3697 | 0.2392 | 0.4589 | 0.5785 | 37.0316 |
| `tower` | 0.240 | 512 | 32x32 | 1,314,819 | 0.0003 | 29.5711 ± 0.4013 | 0.2496 | 0.4714 | 0.5886 | 36.5678 |
| `tower` | 0.380 | 512 | 32x32 | 1,314,819 | 0.0003 | 28.0283 ± 0.4184 | 0.2767 | 0.5079 | 0.6247 | 35.1999 |
| `tower` | 0.500 | 512 | 32x32 | 1,314,819 | 0.0003 | 26.1587 ± 0.6811 | 0.3169 | 0.5549 | 0.6638 | 33.5529 |
| `tower` | 0.620 | 512 | 32x32 | 1,314,819 | 0.0003 | 24.3014 ± 0.4858 | 0.3583 | 0.6059 | 0.7049 | 31.9281 |
| `tower` | 0.760 | 512 | 32x32 | 1,314,819 | 0.0003 | 28.6054 ± 0.8560 | 0.2763 | 0.5022 | 0.6153 | 36.0714 |
| `tower` | 0.880 | 512 | 32x32 | 1,314,819 | 0.0003 | 30.7152 ± 0.7427 | 0.2456 | 0.4569 | 0.5713 | 37.9954 |
| `tower` | 1.000 | 512 | 32x32 | 1,314,819 | 0.001 | 31.5342 ± 0.4085 | 0.2240 | 0.4368 | 0.5559 | 38.6510 |
| `merged` | 1.000 | 512 | 16x16 | 1,314,819 | 0.001 | 32.2451 ± 0.4025 | 0.2102 | 0.4179 | 0.5380 | 39.2153 |
| `projected` | 1.000 | 512 | 16x16 | 1,314,819 | 0.0003 | 30.6088 ± 0.4363 | 0.2413 | 0.4567 | 0.5738 | 37.9774 |

### Qwen/Qwen3.8-27B, normal, capacity-matched

| Stage | Rel. Depth | Width | Grid | Params | LR | mean_deg | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.111 | 512 | 28x28 | 1,314,819 | 0.001 | 29.3251 ± 0.7471 | 0.2571 | 0.4830 | 0.6023 | 36.4647 |
| `tower` | 0.259 | 512 | 28x28 | 1,314,819 | 0.001 | 25.1282 ± 0.5807 | 0.3390 | 0.5904 | 0.6912 | 32.6162 |
| `tower` | 0.370 | 512 | 28x28 | 1,314,819 | 0.001 | 24.2386 ± 0.9138 | 0.3597 | 0.6041 | 0.7042 | 31.7430 |
| `tower` | 0.519 | 512 | 28x28 | 1,314,819 | 0.001 | 24.5956 ± 1.1498 | 0.3501 | 0.5979 | 0.6982 | 32.1925 |
| `tower` | 0.630 | 512 | 28x28 | 1,314,819 | 0.001 | 27.0879 ± 2.8561 | 0.3059 | 0.5316 | 0.6410 | 34.4314 |
| `tower` | 0.741 | 512 | 28x28 | 1,314,819 | 0.0003 | 26.8129 ± 0.3620 | 0.3080 | 0.5467 | 0.6548 | 34.4681 |
| `tower` | 0.889 | 512 | 28x28 | 1,314,819 | 0.0003 | 27.8236 ± 0.3253 | 0.2895 | 0.5241 | 0.6347 | 35.3875 |
| `tower` | 1.000 | 512 | 28x28 | 1,314,819 | 0.0001 | 34.4539 ± 0.5360 | 0.1709 | 0.3534 | 0.4769 | 40.4072 |
| `merged` | 1.000 | 512 | 14x14 | 1,314,819 | 0.0003 | 36.1673 ± 0.1886 | 0.1582 | 0.3330 | 0.4477 | 42.0939 |
| `projected` | 1.000 | 512 | 14x14 | 1,314,819 | 0.0003 | 29.9228 ± 0.3512 | 0.2525 | 0.4773 | 0.5921 | 37.3421 |

### google/siglip2-so400m-patch14-384, normal, capacity-matched

| Stage | Rel. Depth | Width | Grid | Params | LR | mean_deg | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.111 | 512 | 32x32 | 1,314,819 | 0.001 | 29.5101 ± 0.5238 | 0.2525 | 0.4773 | 0.5961 | 36.6022 |
| `tower` | 0.259 | 512 | 32x32 | 1,314,819 | 0.001 | 25.2319 ± 0.5406 | 0.3343 | 0.5795 | 0.6847 | 32.6717 |
| `tower` | 0.370 | 512 | 32x32 | 1,314,819 | 0.001 | 23.9039 ± 0.4021 | 0.3687 | 0.6109 | 0.7097 | 31.5259 |
| `tower` | 0.519 | 512 | 32x32 | 1,314,819 | 0.001 | 25.3960 ± 0.9168 | 0.3346 | 0.5766 | 0.6822 | 33.0879 |
| `tower` | 0.630 | 512 | 32x32 | 1,314,819 | 0.001 | 28.1563 ± 2.0337 | 0.2873 | 0.5077 | 0.6170 | 35.3782 |
| `tower` | 0.741 | 512 | 32x32 | 1,314,819 | 0.001 | 29.2403 ± 1.7726 | 0.2684 | 0.4881 | 0.5979 | 36.4236 |
| `tower` | 0.889 | 512 | 32x32 | 1,314,819 | 0.001 | 30.6236 ± 1.6060 | 0.2506 | 0.4619 | 0.5694 | 37.7459 |
| `tower` | 1.000 | 512 | 32x32 | 1,314,819 | 0.0003 | 31.0147 ± 0.2987 | 0.2360 | 0.4463 | 0.5595 | 38.0554 |

## Semantics, eleven-point grid and deterministic matched correction

The roster run found 107 of the 112 attention cells selecting 3e-4, the floor of the
eight-point semantic grid, and none of the 112 mean cells doing so. This re-run re-searched
all 224 semantic cells on the eleven-point grid `vtb.probe_run.LEARNING_RATES` now specifies,
`[1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1 1]`, and replaced the semantic JSONs in
`results/`. Same data (ImageNet-100 validation, 13,000 images, 4x4 pooled grid, ADR-0005),
same heads, same three seeds, same two capacity arms. The geometry pillar is untouched.

The August 31 matched arm used randomized PCA before the reducer seed was fixed. On
September 1 all 112 matched cells, both readouts over six Towers, were regenerated from the
same caches with the deterministic reducer. The matched JSONs and tables in this section
now contain that correction; raw results are unchanged. Earlier matched point estimates in
the git history are not exactly reproducible and are superseded here.

### The extended grid cleared the truncation

Selected rates over the 224 cells:

| rate | 1e-5 | 3e-5 | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 | 3e-2 | 1e-1 | 3e-1 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| attention | 0 | 1 | 48 | 58 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| mean | 0 | 0 | 0 | 0 | 11 | 27 | 20 | 23 | 15 | 13 | 3 |

Zero cells select the new 1e-5 floor and one selects 3e-5 (Muse Glimmer attention raw
`merged`, where the curve rises to 0.9200 at 3e-5, holds at 1e-4, and falls after, so the
optimum is interior). The old 3e-4 floor is now an interior value selected 58 times. Against
the roster's 107 of 112 attention cells pinned at the floor, the low-rate truncation is
gone: **3 of 224 cells select an endpoint of the grid**, all mean-raw DINOv2 cells at the
1.0 ceiling. The one 3e-5 attention cell is an interior peak, not an endpoint.

The attention cells moved onto the two points just above the old floor. The split stayed by
readout and not by arm, exactly as ADR-0014 predicted:

| readout | arm | 1e-4 | 3e-4 | 1e-3 or above |
|---|---|---|---|---|
| attention | matched | 24 of 56 | 29 of 56 | 3 |
| attention | raw | 24 of 56 (+1 at 3e-5) | 29 of 56 | 2 |
| mean | matched | 0 | 0 | 56 |
| mean | raw | 0 | 0 | 56 |

So the optimum for the attention readout sits at 1e-4 to 3e-4, one grid step below where the
roster could see. Mean pooling still wants the upper half of the range, 1e-3 and above in
every cell.

### What moved, in seed deviations

Per Tower, per Stage, test accuracy against the roster JSONs in units of the roster cell's
seed standard deviation (`test_std`). The raw rows isolate the wider grid. The matched rows
combine that grid correction with deterministic PCA, because the randomized bases used by
the roster cannot be reconstructed.

| readout | arm | cells moved | mean move | median move | largest single move |
|---|---|---|---|---|---|
| attention | raw | 31 of 56 | 1.48 sd | 0.30 sd | muse_glimmer `merged` +0.0061 (+12.2 sd) |
| attention | matched | 54 of 56 | 2.37 sd | 0.87 sd | muse_glimmer `projected` +0.0112 (+18.7 sd) |
| mean | raw | 17 of 56 | 0.15 sd | 0.00 sd | qwen3_5 `tower` -0.0007 (-1.8 sd) |
| mean | matched | 56 of 56 | 3.47 sd | 1.76 sd | qwen3_5 `tower` d0.111 -0.0152 (-25.3 sd) |

The deterministic correction is not a uniformly favourable shift. Relative to the August
31 randomized matched JSONs, mean absolute accuracy changes by 0.00270 for attention and
0.00310 for mean; the largest changes are DINOv2 attention at one late Tower cell (-0.0110)
and SigLIP2 mean at Relative Depth 0.370 (+0.0156). Eighteen of 112 matched cells also select a
different rate. These are reproducibility corrections to the readout, not information
changes in a Tower or Stage.

The roster's attention readout was under-reported by 0.002 to 0.011 top-1 at the deepest
cells. The largest single correction is siglip2 attention raw `tower` d1.0: 0.9171 against
0.9101, +0.0070 at 7.8 seed deviations.

### The grid preserves nominal winners; the bootstrap does not resolve them

Winner of each semantic column, best Tower cell, re-run against roster:

| readout | arm | re-run | roster |
|---|---|---|---|
| attention | matched | **muse_glimmer 0.9203** > siglip2 0.9162 > dinov2 0.9079 > kimi_k26 0.8891 > qwen3_5 0.8824 > moonvit_v2 0.8366 | muse_glimmer 0.9178 > siglip2 0.9091 > ... |
| attention | raw | siglip2 0.9171 > muse_glimmer 0.9154 > dinov2 0.9079 > qwen3_5 0.8870 > kimi_k26 0.8781 > moonvit_v2 0.8373 | muse_glimmer 0.9145 > siglip2 0.9101 > ... |
| mean | matched | siglip2 0.9126 > muse_glimmer 0.9087 > dinov2 0.8933 > kimi_k26 0.8745 > qwen3_5 0.8708 > moonvit_v2 0.8397 | siglip2 0.9135 > muse_glimmer 0.9104 > ... |
| mean | raw | siglip2 0.9121 > muse_glimmer 0.9111 > dinov2 0.8916 > kimi_k26 0.8682 > qwen3_5 0.8598 > moonvit_v2 0.8147 | siglip2 0.9121 > muse_glimmer 0.9106 > ... |

The eleven-point grid gives Muse Glimmer the capacity-matched attention column and SigLIP2
both mean columns and raw attention. Those are nominal orderings, not resolved winners. A
later paired image bootstrap over the same 1,950 test images finds all four
Muse-versus-SigLIP2 intervals crossing zero. The claim that changing the readout changes the
semantic winner is therefore unsupported. No Tower wins everywhere remains a descriptive
summary of the measured Capability Profiles, not evidence for a resolved semantic winner.

### The Stage conclusion strengthens

With every cell at its own optimum, the roster's one statistical blemish on the semantic
pillar dissolves. Muse Glimmer's attention `merged` to `projected` step, which the roster
measured as a 0.0075 loss, becomes a +0.0042 point difference under deterministic PCA:

| model | arm | `tower` | `merged` | `projected` | `merged` to `projected` |
|---|---|---|---|---|---|
| moonvit_v2 | matched | 0.8366 | 0.8405 | 0.8456 | +0.0051 |
| kimi_k26 | matched | 0.8891 | 0.8858 | 0.8827 | -0.0031 |
| qwen3_5 | matched | 0.8790 | 0.8768 | 0.8812 | +0.0044 |
| muse_glimmer | matched | 0.9203 | 0.9149 | 0.9191 | +0.0042 |

The roster's semantic Projector loss was a rate-truncation and randomized-reducer artifact.
At the extended grid all four Projectors hold the attention readout nearly flat across the
Connector: every `merged`-to-`projected` difference is at most 0.0051 in magnitude and at
most 2.4 of the two cells' larger seed spreads. These descriptive seed comparisons do not
replace paired Stage intervals, but none supports the old loss claim.

The raw arm agrees in shape: every Projector's `merged` to `projected` step is inside 20
seed deviations, the largest being qwen3_5's +0.0121 at 20.2 sd in the Projector's favour.

### Relative Depth

Tower accuracy still rises to the final layer in 19 of the 24 semantic arms. The exceptions
are qwen3_5 in all four arms and kimi_k26 attention raw. Qwen3.5 falls from its point peak
by 0.0034 in matched attention, 0.0128 in raw attention, 0.0016 in matched mean, and 0.0155
in raw mean; the matched-attention point peak moves to Relative Depth 0.741 under the
deterministic reducer, while the other three sit at 0.889. Kimi K2.6 falls 0.0015 at 0.5
seed deviations and is noise. Paired intervals resolve both raw Qwen3.5 declines within the
pooled cache: attention
+0.01282 [+0.00530, +0.02034] and mean +0.01556 [+0.00530, +0.02632]. They do not resolve
either matched decline: attention +0.00171 [-0.00632, +0.00957] and mean +0.00154
[-0.00564, +0.00855]. The nominal shape is therefore a four-arm exception, but a genuine
late-layer decline is established only in the two raw arms. This does not repair the failed
full-token attention control, so only the raw-mean curve is pooling-validated.

### Tables

ImageNet-100 validation, 13,000 images, top-1 over three seeds. `val` is the accuracy the
rate search selected on.



### facebook/dinov2-large, attention readout, capacity-matched

| Stage | Rel. Depth | Width | Params | LR | top-1 | val |
|---|---|---|---|---|---|---|
| `tower` | 0.125 | 1024 | 1,627,748 | 0.0003 | 0.3677 ± 0.0004 | 0.3585 |
| `tower` | 0.250 | 1024 | 1,627,748 | 0.0003 | 0.4716 ± 0.0034 | 0.4554 |
| `tower` | 0.375 | 1024 | 1,627,748 | 0.0003 | 0.5612 ± 0.0047 | 0.5544 |
| `tower` | 0.500 | 1024 | 1,627,748 | 0.0003 | 0.6827 ± 0.0027 | 0.6938 |
| `tower` | 0.625 | 1024 | 1,627,748 | 0.001 | 0.7629 ± 0.0009 | 0.7785 |
| `tower` | 0.750 | 1024 | 1,627,748 | 0.0001 | 0.8209 ± 0.0019 | 0.8421 |
| `tower` | 0.875 | 1024 | 1,627,748 | 0.001 | 0.8716 ± 0.0021 | 0.8872 |
| `tower` | 1.000 | 1024 | 1,627,748 | 0.0001 | 0.9079 ± 0.0028 | 0.9092 |

### facebook/dinov2-large, attention readout, unmatched

| Stage | Rel. Depth | Width | Params | LR | top-1 | val |
|---|---|---|---|---|---|---|
| `tower` | 0.125 | 1024 | 2,152,036 | 0.001 | 0.3362 ± 0.0047 | 0.3349 |
| `tower` | 0.250 | 1024 | 2,152,036 | 0.0003 | 0.4429 ± 0.0025 | 0.4415 |
| `tower` | 0.375 | 1024 | 2,152,036 | 0.0003 | 0.5472 ± 0.0070 | 0.5508 |
| `tower` | 0.500 | 1024 | 2,152,036 | 0.001 | 0.6737 ± 0.0044 | 0.6913 |
| `tower` | 0.625 | 1024 | 2,152,036 | 0.0003 | 0.7827 ± 0.0002 | 0.7913 |
| `tower` | 0.750 | 1024 | 2,152,036 | 0.0003 | 0.8356 ± 0.0085 | 0.8482 |
| `tower` | 0.875 | 1024 | 2,152,036 | 0.0001 | 0.8844 ± 0.0009 | 0.8872 |
| `tower` | 1.000 | 1024 | 2,152,036 | 0.0001 | 0.9079 ± 0.0023 | 0.9118 |

### facebook/dinov2-large, mean readout, capacity-matched

| Stage | Rel. Depth | Width | Params | LR | top-1 | val |
|---|---|---|---|---|---|---|
| `tower` | 0.125 | 1024 | 51,300 | 0.3 | 0.2997 ± 0.0011 | 0.2887 |
| `tower` | 0.250 | 1024 | 51,300 | 0.3 | 0.3742 ± 0.0011 | 0.3713 |
| `tower` | 0.375 | 1024 | 51,300 | 0.3 | 0.4460 ± 0.0017 | 0.4415 |
| `tower` | 0.500 | 1024 | 51,300 | 0.1 | 0.5754 ± 0.0015 | 0.5641 |
| `tower` | 0.625 | 1024 | 51,300 | 0.1 | 0.6480 ± 0.0024 | 0.6497 |
| `tower` | 0.750 | 1024 | 51,300 | 0.03 | 0.7496 ± 0.0011 | 0.7585 |
| `tower` | 0.875 | 1024 | 51,300 | 0.01 | 0.8296 ± 0.0002 | 0.8236 |
| `tower` | 1.000 | 1024 | 51,300 | 0.003 | 0.8933 ± 0.0013 | 0.8979 |

### facebook/dinov2-large, mean readout, unmatched

| Stage | Rel. Depth | Width | Params | LR | top-1 | val |
|---|---|---|---|---|---|---|
| `tower` | 0.125 | 1024 | 102,500 | 1 | 0.2627 ± 0.0012 | 0.2554 |
| `tower` | 0.250 | 1024 | 102,500 | 1 | 0.3458 ± 0.0032 | 0.3462 |
| `tower` | 0.375 | 1024 | 102,500 | 1 | 0.4277 ± 0.0044 | 0.4067 |
| `tower` | 0.500 | 1024 | 102,500 | 0.3 | 0.5564 ± 0.0015 | 0.5364 |
| `tower` | 0.625 | 1024 | 102,500 | 0.1 | 0.6467 ± 0.0023 | 0.6549 |
| `tower` | 0.750 | 1024 | 102,500 | 0.03 | 0.7538 ± 0.0011 | 0.7518 |
| `tower` | 0.875 | 1024 | 102,500 | 0.01 | 0.8291 ± 0.0002 | 0.8164 |
| `tower` | 1.000 | 1024 | 102,500 | 0.001 | 0.8916 ± 0.0010 | 0.8923 |

### immanuelpeter/MoonViT-K2.6, attention readout, capacity-matched

| Stage | Rel. Depth | Width | Params | LR | top-1 | val |
|---|---|---|---|---|---|---|
| `tower` | 0.111 | 1152 | 1,627,748 | 0.0003 | 0.4309 ± 0.0032 | 0.4067 |
| `tower` | 0.259 | 1152 | 1,627,748 | 0.0003 | 0.6118 ± 0.0090 | 0.6236 |
| `tower` | 0.370 | 1152 | 1,627,748 | 0.0003 | 0.6928 ± 0.0029 | 0.7118 |
| `tower` | 0.518 | 1152 | 1,627,748 | 0.0003 | 0.8101 ± 0.0017 | 0.8149 |
| `tower` | 0.630 | 1152 | 1,627,748 | 0.0001 | 0.8593 ± 0.0030 | 0.8687 |
| `tower` | 0.741 | 1152 | 1,627,748 | 0.0001 | 0.8769 ± 0.0008 | 0.8846 |
| `tower` | 0.889 | 1152 | 1,627,748 | 0.0001 | 0.8810 ± 0.0037 | 0.8928 |
| `tower` | 1.000 | 1152 | 1,627,748 | 0.0001 | 0.8891 ± 0.0025 | 0.8928 |
| `merged` | 1.000 | 4608 | 1,627,748 | 0.0001 | 0.8858 ± 0.0013 | 0.8892 |
| `projected` | 1.000 | 7168 | 1,627,748 | 0.0001 | 0.8827 ± 0.0017 | 0.8923 |

### immanuelpeter/MoonViT-K2.6, attention readout, unmatched

| Stage | Rel. Depth | Width | Params | LR | top-1 | val |
|---|---|---|---|---|---|---|
| `tower` | 0.111 | 1152 | 2,283,108 | 0.0003 | 0.3152 ± 0.0043 | 0.3123 |
| `tower` | 0.259 | 1152 | 2,283,108 | 0.0003 | 0.5526 ± 0.0002 | 0.5477 |
| `tower` | 0.370 | 1152 | 2,283,108 | 0.0003 | 0.6715 ± 0.0057 | 0.6800 |
| `tower` | 0.518 | 1152 | 2,283,108 | 0.0003 | 0.8060 ± 0.0025 | 0.8297 |
| `tower` | 0.630 | 1152 | 2,283,108 | 0.0003 | 0.8545 ± 0.0017 | 0.8687 |
| `tower` | 0.741 | 1152 | 2,283,108 | 0.0001 | 0.8701 ± 0.0017 | 0.8821 |
| `tower` | 0.889 | 1152 | 2,283,108 | 0.0001 | 0.8781 ± 0.0030 | 0.8954 |
| `tower` | 1.000 | 1152 | 2,283,108 | 0.0001 | 0.8766 ± 0.0028 | 0.8918 |
| `merged` | 1.000 | 4608 | 5,822,052 | 0.0001 | 0.8853 ± 0.0009 | 0.8897 |
| `projected` | 1.000 | 7168 | 8,443,492 | 0.0001 | 0.8827 ± 0.0006 | 0.8923 |

### immanuelpeter/MoonViT-K2.6, mean readout, capacity-matched

| Stage | Rel. Depth | Width | Params | LR | top-1 | val |
|---|---|---|---|---|---|---|
| `tower` | 0.111 | 1152 | 51,300 | 0.1 | 0.3552 ± 0.0013 | 0.3364 |
| `tower` | 0.259 | 1152 | 51,300 | 0.03 | 0.5179 ± 0.0011 | 0.5144 |
| `tower` | 0.370 | 1152 | 51,300 | 0.03 | 0.6214 ± 0.0013 | 0.6179 |
| `tower` | 0.518 | 1152 | 51,300 | 0.01 | 0.7503 ± 0.0022 | 0.7574 |
| `tower` | 0.630 | 1152 | 51,300 | 0.01 | 0.8260 ± 0.0013 | 0.8374 |
| `tower` | 0.741 | 1152 | 51,300 | 0.003 | 0.8547 ± 0.0023 | 0.8585 |
| `tower` | 0.889 | 1152 | 51,300 | 0.003 | 0.8672 ± 0.0013 | 0.8790 |
| `tower` | 1.000 | 1152 | 51,300 | 0.003 | 0.8745 ± 0.0012 | 0.8846 |
| `merged` | 1.000 | 4608 | 51,300 | 0.003 | 0.8745 ± 0.0009 | 0.8826 |
| `projected` | 1.000 | 7168 | 51,300 | 0.01 | 0.8786 ± 0.0013 | 0.8841 |

### immanuelpeter/MoonViT-K2.6, mean readout, unmatched

| Stage | Rel. Depth | Width | Params | LR | top-1 | val |
|---|---|---|---|---|---|---|
| `tower` | 0.111 | 1152 | 115,300 | 0.3 | 0.2291 ± 0.0043 | 0.2405 |
| `tower` | 0.259 | 1152 | 115,300 | 0.3 | 0.4036 ± 0.0025 | 0.3979 |
| `tower` | 0.370 | 1152 | 115,300 | 0.3 | 0.5279 ± 0.0038 | 0.5205 |
| `tower` | 0.518 | 1152 | 115,300 | 0.1 | 0.7094 ± 0.0038 | 0.7205 |
| `tower` | 0.630 | 1152 | 115,300 | 0.03 | 0.8050 ± 0.0006 | 0.8241 |
| `tower` | 0.741 | 1152 | 115,300 | 0.01 | 0.8421 ± 0.0019 | 0.8554 |
| `tower` | 0.889 | 1152 | 115,300 | 0.01 | 0.8603 ± 0.0019 | 0.8754 |
| `tower` | 1.000 | 1152 | 115,300 | 0.01 | 0.8682 ± 0.0017 | 0.8718 |
| `merged` | 1.000 | 4608 | 460,900 | 0.003 | 0.8665 ± 0.0012 | 0.8764 |
| `projected` | 1.000 | 7168 | 716,900 | 0.01 | 0.8658 ± 0.0009 | 0.8733 |

### immanuelpeter/MoonViT-V2, attention readout, capacity-matched

| Stage | Rel. Depth | Width | Params | LR | top-1 | val |
|---|---|---|---|---|---|---|
| `tower` | 0.111 | 1024 | 1,627,748 | 0.0003 | 0.3855 ± 0.0046 | 0.3774 |
| `tower` | 0.259 | 1024 | 1,627,748 | 0.0003 | 0.5299 ± 0.0059 | 0.5087 |
| `tower` | 0.370 | 1024 | 1,627,748 | 0.0003 | 0.6026 ± 0.0022 | 0.5974 |
| `tower` | 0.518 | 1024 | 1,627,748 | 0.0003 | 0.7120 ± 0.0038 | 0.7118 |
| `tower` | 0.630 | 1024 | 1,627,748 | 0.0003 | 0.7607 ± 0.0045 | 0.7718 |
| `tower` | 0.741 | 1024 | 1,627,748 | 0.0003 | 0.8021 ± 0.0037 | 0.8097 |
| `tower` | 0.889 | 1024 | 1,627,748 | 0.0003 | 0.8253 ± 0.0025 | 0.8282 |
| `tower` | 1.000 | 1024 | 1,627,748 | 0.0001 | 0.8366 ± 0.0016 | 0.8533 |
| `merged` | 1.000 | 4096 | 1,627,748 | 0.0003 | 0.8405 ± 0.0034 | 0.8569 |
| `projected` | 1.000 | 7168 | 1,627,748 | 0.0001 | 0.8456 ± 0.0011 | 0.8518 |

### immanuelpeter/MoonViT-V2, attention readout, unmatched

| Stage | Rel. Depth | Width | Params | LR | top-1 | val |
|---|---|---|---|---|---|---|
| `tower` | 0.111 | 1024 | 2,152,036 | 0.0003 | 0.3773 ± 0.0075 | 0.3759 |
| `tower` | 0.259 | 1024 | 2,152,036 | 0.0003 | 0.5021 ± 0.0026 | 0.4933 |
| `tower` | 0.370 | 1024 | 2,152,036 | 0.0003 | 0.5884 ± 0.0062 | 0.5738 |
| `tower` | 0.518 | 1024 | 2,152,036 | 0.0003 | 0.7036 ± 0.0030 | 0.7082 |
| `tower` | 0.630 | 1024 | 2,152,036 | 0.0003 | 0.7699 ± 0.0033 | 0.7733 |
| `tower` | 0.741 | 1024 | 2,152,036 | 0.0003 | 0.8063 ± 0.0046 | 0.8092 |
| `tower` | 0.889 | 1024 | 2,152,036 | 0.0003 | 0.8137 ± 0.0035 | 0.8226 |
| `tower` | 1.000 | 1024 | 2,152,036 | 0.0003 | 0.8373 ± 0.0016 | 0.8482 |
| `merged` | 1.000 | 4096 | 5,297,764 | 0.0001 | 0.8451 ± 0.0025 | 0.8590 |
| `projected` | 1.000 | 7168 | 8,443,492 | 0.0001 | 0.8468 ± 0.0030 | 0.8544 |

### immanuelpeter/MoonViT-V2, mean readout, capacity-matched

| Stage | Rel. Depth | Width | Params | LR | top-1 | val |
|---|---|---|---|---|---|---|
| `tower` | 0.111 | 1024 | 51,300 | 0.03 | 0.2993 ± 0.0016 | 0.2928 |
| `tower` | 0.259 | 1024 | 51,300 | 0.03 | 0.4109 ± 0.0013 | 0.3918 |
| `tower` | 0.370 | 1024 | 51,300 | 0.03 | 0.5036 ± 0.0015 | 0.4615 |
| `tower` | 0.518 | 1024 | 51,300 | 0.03 | 0.6142 ± 0.0006 | 0.6128 |
| `tower` | 0.630 | 1024 | 51,300 | 0.03 | 0.6969 ± 0.0007 | 0.6918 |
| `tower` | 0.741 | 1024 | 51,300 | 0.03 | 0.7467 ± 0.0007 | 0.7564 |
| `tower` | 0.889 | 1024 | 51,300 | 0.03 | 0.7761 ± 0.0002 | 0.7836 |
| `tower` | 1.000 | 1024 | 51,300 | 0.1 | 0.8397 ± 0.0002 | 0.8400 |
| `merged` | 1.000 | 4096 | 51,300 | 0.03 | 0.8470 ± 0.0013 | 0.8441 |
| `projected` | 1.000 | 7168 | 51,300 | 0.03 | 0.8421 ± 0.0017 | 0.8369 |

### immanuelpeter/MoonViT-V2, mean readout, unmatched

| Stage | Rel. Depth | Width | Params | LR | top-1 | val |
|---|---|---|---|---|---|---|
| `tower` | 0.111 | 1024 | 102,500 | 0.3 | 0.2696 ± 0.0052 | 0.2697 |
| `tower` | 0.259 | 1024 | 102,500 | 0.3 | 0.3728 ± 0.0029 | 0.3626 |
| `tower` | 0.370 | 1024 | 102,500 | 0.3 | 0.4385 ± 0.0017 | 0.4251 |
| `tower` | 0.518 | 1024 | 102,500 | 0.1 | 0.5896 ± 0.0023 | 0.5708 |
| `tower` | 0.630 | 1024 | 102,500 | 0.1 | 0.6663 ± 0.0016 | 0.6641 |
| `tower` | 0.741 | 1024 | 102,500 | 0.1 | 0.7097 ± 0.0025 | 0.7144 |
| `tower` | 0.889 | 1024 | 102,500 | 0.3 | 0.7256 ± 0.0037 | 0.7333 |
| `tower` | 1.000 | 1024 | 102,500 | 0.1 | 0.8147 ± 0.0013 | 0.8277 |
| `merged` | 1.000 | 4096 | 409,700 | 0.03 | 0.8159 ± 0.0000 | 0.8297 |
| `projected` | 1.000 | 7168 | 716,900 | 0.03 | 0.8154 ± 0.0004 | 0.8246 |

### immanuelpeter/Muse-Glimmer-Vision, attention readout, capacity-matched

| Stage | Rel. Depth | Width | Params | LR | top-1 | val |
|---|---|---|---|---|---|---|
| `tower` | 0.120 | 1536 | 1,627,748 | 0.0003 | 0.4053 ± 0.0020 | 0.3964 |
| `tower` | 0.240 | 1536 | 1,627,748 | 0.0003 | 0.4961 ± 0.0050 | 0.4903 |
| `tower` | 0.380 | 1536 | 1,627,748 | 0.0003 | 0.5561 ± 0.0033 | 0.5559 |
| `tower` | 0.500 | 1536 | 1,627,748 | 0.001 | 0.6568 ± 0.0061 | 0.6610 |
| `tower` | 0.620 | 1536 | 1,627,748 | 0.0003 | 0.7581 ± 0.0101 | 0.7733 |
| `tower` | 0.760 | 1536 | 1,627,748 | 0.0001 | 0.8591 ± 0.0026 | 0.8887 |
| `tower` | 0.880 | 1536 | 1,627,748 | 0.0001 | 0.9029 ± 0.0017 | 0.9097 |
| `tower` | 1.000 | 1536 | 1,627,748 | 0.0001 | 0.9203 ± 0.0024 | 0.9256 |
| `merged` | 1.000 | 6144 | 1,627,748 | 0.0001 | 0.9149 ± 0.0018 | 0.9236 |
| `projected` | 1.000 | 6656 | 1,627,748 | 0.0001 | 0.9191 ± 0.0006 | 0.9251 |

### immanuelpeter/Muse-Glimmer-Vision, attention readout, unmatched

| Stage | Rel. Depth | Width | Params | LR | top-1 | val |
|---|---|---|---|---|---|---|
| `tower` | 0.120 | 1536 | 2,676,324 | 0.0003 | 0.3932 ± 0.0034 | 0.3923 |
| `tower` | 0.240 | 1536 | 2,676,324 | 0.0003 | 0.4733 ± 0.0017 | 0.4862 |
| `tower` | 0.380 | 1536 | 2,676,324 | 0.0003 | 0.5515 ± 0.0057 | 0.5477 |
| `tower` | 0.500 | 1536 | 2,676,324 | 0.0003 | 0.6742 ± 0.0056 | 0.6677 |
| `tower` | 0.620 | 1536 | 2,676,324 | 0.0001 | 0.7610 ± 0.0069 | 0.7759 |
| `tower` | 0.760 | 1536 | 2,676,324 | 0.0001 | 0.8651 ± 0.0051 | 0.8862 |
| `tower` | 0.880 | 1536 | 2,676,324 | 0.0001 | 0.9024 ± 0.0021 | 0.9113 |
| `tower` | 1.000 | 1536 | 2,676,324 | 0.0001 | 0.9154 ± 0.0007 | 0.9221 |
| `merged` | 1.000 | 6144 | 7,394,916 | 3e-05 | 0.9157 ± 0.0025 | 0.9200 |
| `projected` | 1.000 | 6656 | 7,919,204 | 0.0001 | 0.9135 ± 0.0017 | 0.9185 |

### immanuelpeter/Muse-Glimmer-Vision, mean readout, capacity-matched

| Stage | Rel. Depth | Width | Params | LR | top-1 | val |
|---|---|---|---|---|---|---|
| `tower` | 0.120 | 1536 | 51,300 | 0.003 | 0.3179 ± 0.0029 | 0.3226 |
| `tower` | 0.240 | 1536 | 51,300 | 0.003 | 0.3819 ± 0.0015 | 0.3815 |
| `tower` | 0.380 | 1536 | 51,300 | 0.003 | 0.4373 ± 0.0036 | 0.4200 |
| `tower` | 0.500 | 1536 | 51,300 | 0.003 | 0.5297 ± 0.0029 | 0.5179 |
| `tower` | 0.620 | 1536 | 51,300 | 0.003 | 0.6540 ± 0.0021 | 0.6554 |
| `tower` | 0.760 | 1536 | 51,300 | 0.003 | 0.8179 ± 0.0018 | 0.8374 |
| `tower` | 0.880 | 1536 | 51,300 | 0.001 | 0.8843 ± 0.0016 | 0.8985 |
| `tower` | 1.000 | 1536 | 51,300 | 0.01 | 0.9087 ± 0.0008 | 0.9231 |
| `merged` | 1.000 | 6144 | 51,300 | 0.003 | 0.9050 ± 0.0002 | 0.9179 |
| `projected` | 1.000 | 6656 | 51,300 | 0.003 | 0.9108 ± 0.0008 | 0.9185 |

### immanuelpeter/Muse-Glimmer-Vision, mean readout, unmatched

| Stage | Rel. Depth | Width | Params | LR | top-1 | val |
|---|---|---|---|---|---|---|
| `tower` | 0.120 | 1536 | 153,700 | 0.01 | 0.2969 ± 0.0040 | 0.3149 |
| `tower` | 0.240 | 1536 | 153,700 | 0.01 | 0.3632 ± 0.0028 | 0.3600 |
| `tower` | 0.380 | 1536 | 153,700 | 0.01 | 0.4214 ± 0.0037 | 0.4051 |
| `tower` | 0.500 | 1536 | 153,700 | 0.003 | 0.5185 ± 0.0054 | 0.5103 |
| `tower` | 0.620 | 1536 | 153,700 | 0.01 | 0.6549 ± 0.0011 | 0.6503 |
| `tower` | 0.760 | 1536 | 153,700 | 0.003 | 0.8193 ± 0.0032 | 0.8385 |
| `tower` | 0.880 | 1536 | 153,700 | 0.001 | 0.8834 ± 0.0032 | 0.8913 |
| `tower` | 1.000 | 1536 | 153,700 | 0.01 | 0.9111 ± 0.0006 | 0.9185 |
| `merged` | 1.000 | 6144 | 614,500 | 0.003 | 0.9132 ± 0.0005 | 0.9154 |
| `projected` | 1.000 | 6656 | 665,700 | 0.001 | 0.8986 ± 0.0017 | 0.9144 |

### immanuelpeter/Qwen3.8-27B-Vision, attention readout, capacity-matched

| Stage | Rel. Depth | Width | Params | LR | top-1 | val |
|---|---|---|---|---|---|---|
| `tower` | 0.111 | 1152 | 1,627,748 | 0.0003 | 0.4665 ± 0.0005 | 0.4395 |
| `tower` | 0.259 | 1152 | 1,627,748 | 0.0003 | 0.6390 ± 0.0042 | 0.6328 |
| `tower` | 0.370 | 1152 | 1,627,748 | 0.0003 | 0.7303 ± 0.0036 | 0.7385 |
| `tower` | 0.518 | 1152 | 1,627,748 | 0.0003 | 0.8130 ± 0.0009 | 0.8344 |
| `tower` | 0.630 | 1152 | 1,627,748 | 0.0001 | 0.8598 ± 0.0025 | 0.8738 |
| `tower` | 0.741 | 1152 | 1,627,748 | 0.0001 | 0.8824 ± 0.0038 | 0.8867 |
| `tower` | 0.889 | 1152 | 1,627,748 | 0.0003 | 0.8800 ± 0.0033 | 0.8938 |
| `tower` | 1.000 | 1152 | 1,627,748 | 0.0001 | 0.8790 ± 0.0008 | 0.8949 |
| `merged` | 1.000 | 4608 | 1,627,748 | 0.0001 | 0.8768 ± 0.0026 | 0.8851 |
| `projected` | 1.000 | 5120 | 1,627,748 | 0.0001 | 0.8812 ± 0.0029 | 0.8851 |

### immanuelpeter/Qwen3.8-27B-Vision, attention readout, unmatched

| Stage | Rel. Depth | Width | Params | LR | top-1 | val |
|---|---|---|---|---|---|---|
| `tower` | 0.111 | 1152 | 2,283,108 | 0.0003 | 0.4574 ± 0.0067 | 0.4354 |
| `tower` | 0.259 | 1152 | 2,283,108 | 0.0003 | 0.6450 ± 0.0034 | 0.6364 |
| `tower` | 0.370 | 1152 | 2,283,108 | 0.0003 | 0.7431 ± 0.0019 | 0.7426 |
| `tower` | 0.518 | 1152 | 2,283,108 | 0.0001 | 0.8145 ± 0.0030 | 0.8333 |
| `tower` | 0.630 | 1152 | 2,283,108 | 0.0001 | 0.8658 ± 0.0021 | 0.8774 |
| `tower` | 0.741 | 1152 | 2,283,108 | 0.0001 | 0.8826 ± 0.0042 | 0.8933 |
| `tower` | 0.889 | 1152 | 2,283,108 | 0.0001 | 0.8870 ± 0.0011 | 0.8954 |
| `tower` | 1.000 | 1152 | 2,283,108 | 0.0003 | 0.8742 ± 0.0036 | 0.8918 |
| `merged` | 1.000 | 4608 | 5,822,052 | 0.0003 | 0.8682 ± 0.0015 | 0.8856 |
| `projected` | 1.000 | 5120 | 6,346,340 | 0.0001 | 0.8803 ± 0.0006 | 0.8841 |

### immanuelpeter/Qwen3.8-27B-Vision, mean readout, capacity-matched

| Stage | Rel. Depth | Width | Params | LR | top-1 | val |
|---|---|---|---|---|---|---|
| `tower` | 0.111 | 1152 | 51,300 | 0.1 | 0.3641 ± 0.0029 | 0.3636 |
| `tower` | 0.259 | 1152 | 51,300 | 0.03 | 0.5426 ± 0.0013 | 0.5236 |
| `tower` | 0.370 | 1152 | 51,300 | 0.01 | 0.6337 ± 0.0015 | 0.6508 |
| `tower` | 0.518 | 1152 | 51,300 | 0.01 | 0.7427 ± 0.0017 | 0.7579 |
| `tower` | 0.630 | 1152 | 51,300 | 0.003 | 0.8214 ± 0.0013 | 0.8415 |
| `tower` | 0.741 | 1152 | 51,300 | 0.003 | 0.8556 ± 0.0017 | 0.8662 |
| `tower` | 0.889 | 1152 | 51,300 | 0.001 | 0.8708 ± 0.0008 | 0.8749 |
| `tower` | 1.000 | 1152 | 51,300 | 0.001 | 0.8692 ± 0.0022 | 0.8800 |
| `merged` | 1.000 | 4608 | 51,300 | 0.001 | 0.8542 ± 0.0050 | 0.8651 |
| `projected` | 1.000 | 5120 | 51,300 | 0.003 | 0.8684 ± 0.0017 | 0.8764 |

### immanuelpeter/Qwen3.8-27B-Vision, mean readout, unmatched

| Stage | Rel. Depth | Width | Params | LR | top-1 | val |
|---|---|---|---|---|---|---|
| `tower` | 0.111 | 1152 | 115,300 | 0.1 | 0.3523 ± 0.0026 | 0.3344 |
| `tower` | 0.259 | 1152 | 115,300 | 0.1 | 0.5038 ± 0.0013 | 0.5082 |
| `tower` | 0.370 | 1152 | 115,300 | 0.03 | 0.6275 ± 0.0009 | 0.6441 |
| `tower` | 0.518 | 1152 | 115,300 | 0.01 | 0.7236 ± 0.0004 | 0.7472 |
| `tower` | 0.630 | 1152 | 115,300 | 0.003 | 0.8017 ± 0.0013 | 0.8359 |
| `tower` | 0.741 | 1152 | 115,300 | 0.003 | 0.8521 ± 0.0013 | 0.8667 |
| `tower` | 0.889 | 1152 | 115,300 | 0.003 | 0.8598 ± 0.0017 | 0.8800 |
| `tower` | 1.000 | 1152 | 115,300 | 0.003 | 0.8443 ± 0.0010 | 0.8487 |
| `merged` | 1.000 | 4608 | 460,900 | 0.3 | 0.8289 ± 0.0036 | 0.8482 |
| `projected` | 1.000 | 5120 | 512,100 | 0.001 | 0.8629 ± 0.0006 | 0.8708 |

### google/siglip2-so400m-patch14-384, attention readout, capacity-matched

| Stage | Rel. Depth | Width | Params | LR | top-1 | val |
|---|---|---|---|---|---|---|
| `tower` | 0.111 | 1152 | 1,627,748 | 0.0003 | 0.4636 ± 0.0051 | 0.4390 |
| `tower` | 0.259 | 1152 | 1,627,748 | 0.0003 | 0.6525 ± 0.0030 | 0.6579 |
| `tower` | 0.370 | 1152 | 1,627,748 | 0.0003 | 0.7303 ± 0.0007 | 0.7441 |
| `tower` | 0.518 | 1152 | 1,627,748 | 0.0003 | 0.8323 ± 0.0023 | 0.8421 |
| `tower` | 0.630 | 1152 | 1,627,748 | 0.0001 | 0.8756 ± 0.0038 | 0.8933 |
| `tower` | 0.741 | 1152 | 1,627,748 | 0.0001 | 0.8983 ± 0.0015 | 0.9144 |
| `tower` | 0.889 | 1152 | 1,627,748 | 0.0001 | 0.9128 ± 0.0027 | 0.9195 |
| `tower` | 1.000 | 1152 | 1,627,748 | 0.0001 | 0.9162 ± 0.0030 | 0.9226 |

### google/siglip2-so400m-patch14-384, attention readout, unmatched

| Stage | Rel. Depth | Width | Params | LR | top-1 | val |
|---|---|---|---|---|---|---|
| `tower` | 0.111 | 1152 | 2,283,108 | 0.0003 | 0.4744 ± 0.0042 | 0.4600 |
| `tower` | 0.259 | 1152 | 2,283,108 | 0.0003 | 0.6682 ± 0.0023 | 0.6738 |
| `tower` | 0.370 | 1152 | 2,283,108 | 0.0003 | 0.7511 ± 0.0035 | 0.7728 |
| `tower` | 0.518 | 1152 | 2,283,108 | 0.0001 | 0.8294 ± 0.0036 | 0.8590 |
| `tower` | 0.630 | 1152 | 2,283,108 | 0.0001 | 0.8814 ± 0.0024 | 0.8892 |
| `tower` | 0.741 | 1152 | 2,283,108 | 0.0001 | 0.8997 ± 0.0023 | 0.9144 |
| `tower` | 0.889 | 1152 | 2,283,108 | 0.0001 | 0.9106 ± 0.0015 | 0.9149 |
| `tower` | 1.000 | 1152 | 2,283,108 | 0.0001 | 0.9171 ± 0.0005 | 0.9215 |

### google/siglip2-so400m-patch14-384, mean readout, capacity-matched

| Stage | Rel. Depth | Width | Params | LR | top-1 | val |
|---|---|---|---|---|---|---|
| `tower` | 0.111 | 1152 | 51,300 | 0.03 | 0.3513 ± 0.0008 | 0.3533 |
| `tower` | 0.259 | 1152 | 51,300 | 0.03 | 0.5403 ± 0.0020 | 0.5323 |
| `tower` | 0.370 | 1152 | 51,300 | 0.03 | 0.6660 ± 0.0020 | 0.6595 |
| `tower` | 0.518 | 1152 | 51,300 | 0.1 | 0.7644 ± 0.0005 | 0.7928 |
| `tower` | 0.630 | 1152 | 51,300 | 0.01 | 0.8566 ± 0.0011 | 0.8723 |
| `tower` | 0.741 | 1152 | 51,300 | 0.003 | 0.8867 ± 0.0008 | 0.9103 |
| `tower` | 0.889 | 1152 | 51,300 | 0.003 | 0.9070 ± 0.0020 | 0.9169 |
| `tower` | 1.000 | 1152 | 51,300 | 0.001 | 0.9126 ± 0.0013 | 0.9185 |

### google/siglip2-so400m-patch14-384, mean readout, unmatched

| Stage | Rel. Depth | Width | Params | LR | top-1 | val |
|---|---|---|---|---|---|---|
| `tower` | 0.111 | 1152 | 115,300 | 0.3 | 0.3207 ± 0.0021 | 0.3462 |
| `tower` | 0.259 | 1152 | 115,300 | 0.1 | 0.5195 ± 0.0044 | 0.5087 |
| `tower` | 0.370 | 1152 | 115,300 | 0.03 | 0.6550 ± 0.0017 | 0.6610 |
| `tower` | 0.518 | 1152 | 115,300 | 0.03 | 0.7706 ± 0.0005 | 0.7933 |
| `tower` | 0.630 | 1152 | 115,300 | 0.01 | 0.8578 ± 0.0021 | 0.8692 |
| `tower` | 0.741 | 1152 | 115,300 | 0.003 | 0.8856 ± 0.0011 | 0.9041 |
| `tower` | 0.889 | 1152 | 115,300 | 0.001 | 0.9065 ± 0.0006 | 0.9159 |
| `tower` | 1.000 | 1152 | 115,300 | 0.001 | 0.9121 ± 0.0005 | 0.9241 |

## Pooling validation (ADR-0005), August 31 2026

SigLIP2 and Muse Glimmer were probed on both the 4x4 pooled grid and full patch tokens over
the same 1,500 ImageNet-100 validation images - the first 1,500 in sorted order for both
grids, image ids verified identical between the caches. 72 cells, both readouts, both arms,
eleven-point grid. Full token JSONs live in `results/pooling/`; the verdict is ADR-0019.

Split outcome. The mean readout validates: rankings agree (SigLIP2 first on both Towers in
every arm) and Relative Depth curves agree, with the raw cells identical to four decimals
because averaging a 4x4 pooled grid and averaging 1,024 tokens are the same operation. The
attention readout fails the check: the full-token attention heads do not train at mid
Relative Depth on 1,050 training images - six of sixteen cells collapse to near-random where
the pooled curves rise smoothly - so their curves cannot confirm the pooled ones, and the
raw-arm ranking flips by one test image. Where the full-token head trains (the deepest
cells), it matches or beats pooled and the rankings agree in the matched arm.

The semantic pillar's attention readout therefore carries a caveat: its cross-model
rankings are measured on pooled features whose control did not validate. Mean-readout
comparisons are unaffected. Within-model comparisons remain controlled because every cell
uses the same pooled cache, but that does not establish that pooling preserves the absolute
shape of an attention Relative Depth curve. Details in ADR-0019.

### Paired bootstrap on the semantic readout rankings

The promised paired image bootstrap was added after the pooling diagnostic. The deepest
Tower cells for Muse Glimmer and SigLIP2 were re-searched on the full eleven-point grid,
then trained for three seeds at the selected rate. Ten thousand paired resamples of
seed-averaged correctness over the same 1,950 test images produce these intervals; every
difference is Muse minus SigLIP2.

| readout | arm | Muse | SigLIP2 | difference | paired 95% interval |
|---|---|---:|---:|---:|---:|
| attention | matched | 0.92034 | 0.91624 | +0.00410 | [-0.00359, +0.01214] |
| attention | raw | 0.91538 | 0.91709 | -0.00171 | [-0.00974, +0.00633] |
| mean | matched | 0.90872 | 0.91265 | -0.00393 | [-0.01299, +0.00530] |
| mean | raw | 0.91111 | 0.91214 | -0.00103 | [-0.01043, +0.00855] |

All four intervals cross zero. The pooled test set resolves neither a Muse-first attention
ranking nor a SigLIP2-first mean ranking. Hypothesis 2's claim that changing the readout
changes the semantic winner is therefore unsupported; winner identity is unresolved in
every semantic readout column.

The same procedure was applied within Qwen3.5 to all four non-rising semantic arms. Raw
attention and raw mean resolve earlier-layer advantages of +0.01282
[+0.00530, +0.02034] and +0.01556 [+0.00530, +0.02632]. Matched attention and matched mean
cross zero at +0.00171 [-0.00632, +0.00957] and +0.00154
[-0.00564, +0.00855]. Thus only the two raw declines resolve within the pooled cache. The
matched-mean result uses the deterministic reducer; its earlier cell reads 0.87077 against
0.86923 at the final layer.

The cross-task semantic comparison resolves against only one of those Towers:

| comparison, matched attention | difference | paired 95% interval |
|---|---:|---:|
| Muse Glimmer - DINOv2 | +0.01248 | [+0.00342, +0.02154] |
| SigLIP2 - DINOv2 | +0.00838 | [-0.00085, +0.01778] |

Muse Glimmer measurably exceeds DINOv2; SigLIP2 does not. Because Muse Glimmer and SigLIP2
are themselves unresolved, the supported statement is "DINOv2 is below Muse Glimmer," not
"DINOv2 is below the top semantic row." The geometry intervals below complete the
cross-task Capability Profile claim against Muse Glimmer.

The two raw reruns reproduce the committed headline accuracies at their reported precision.
The matched cells use the deterministic reducer from commit `8c82a96`; their original
randomized PCA draws and trained heads were not saved and cannot be reconstructed exactly.
On matched attention the corrected run raises both Towers by about 0.0008 and preserves the
committed +0.0041 margin.

This statistical result does not validate pooling. It neither compares pooled against full
tokens nor answers whether pooling changes the attention Relative Depth curve. The
ten per-image prediction datasets and their bootstrap metadata live in
`results/bootstrap/`.

### Paired bootstrap on the geometry headline comparisons

The first geometry pass targeted three predeclared comparisons rather than rerunning all
448 cells: the strongest Projector Stage step, and DINOv2 against the next-best Tower on
each geometry task. Each selected cell is re-searched on the full six-point grid under the
deterministic matched reducer, trained for three seeds, and scored per image. Ten thousand
paired resamples operate on the seed-mean metric over the same 115 test images. Positive
differences mean the first cell is better; for surface normals that is the second cell's
mean angular error minus DINOv2's because lower is better.

| comparison | corrected cells | first advantage | paired 95% interval |
|---|---:|---:|---:|
| Muse Glimmer `projected` - `merged`, depth `d1` | 0.58920 vs 0.52298 | +0.06622 | [+0.04992, +0.08338] |
| DINOv2 - Qwen3.5, depth `d1` | 0.69881 vs 0.66965 | +0.02916 | [+0.00989, +0.04835] |
| DINOv2 advantage over SigLIP2, normal `mean_deg` | 18.8204 vs 23.8575 | +5.0370 degrees | [+4.3038, +5.8164] |

All three overall intervals exclude zero. The Muse Projector step and DINOv2 normal lead
also resolve separately indoors and outdoors. DINOv2's depth lead resolves outdoors; the
51-image indoor interval [-0.00696, +0.07074] crosses zero, while the combined paired
interval remains positive.

This completes the uncertainty check for hypothesis 2's geometry half. Together with the
resolved Muse Glimmer-over-DINOv2 semantic interval, it supports the cross-task statement:
DINOv2 wins both measured geometry tasks but loses matched-attention semantics to Muse
Glimmer. It does not restore an ordering among the statistically unresolved top semantic
Towers, and it is conditional on the Relative Depth cells selected by the original matrix.
The September 1 follow-up then tested the eight matched Projector headline comparisons:
`projected` against final `tower` for four Projectors and two tasks. All eight overall
intervals exclude zero in the Projector's favour. The corresponding raw pass resolves seven
of eight; Kimi K2.6 depth crosses zero. The complete tables are in section 1. Together,
these are nineteen targeted geometry comparisons, not a blanket bootstrap over the matrix.
Their per-image metric datasets, validation curves, seed metrics, and bootstrap metadata
live beside the semantic datasets in `results/bootstrap/`. Cache reconstruction, timings,
and the deterministic semantic rerun are recorded in
`docs/measurements/remaining-bootstrap-2026-09-01.md`.

## Indoors against outdoor

Coverage is the fraction of each target the metric can score, averaged over the test split.
Depth is annotated on 0.9890 of an indoor image and 0.7771 of an outdoor one; normals reach
0.9803 indoors and 0.6116 outdoors. DIODE ships no validity mask for normals, so an
unannotated pixel is a zero vector and validity comes from normal magnitude. Read every
outdoor normals number as an average over about three fifths of the frame.

Both scene types give the same answers as the pooled numbers above. The capacity-matched arm
is shown.

### Depth

### facebook/dinov2-large, depth, capacity-matched

| Stage | Rel. Depth | indoors d1 | outdoor d1 | indoors coverage | outdoor coverage |
|---|---|---|---|---|---|
| `tower` | 0.125 | 0.4892 ± 0.0116 | 0.4307 ± 0.0088 | 0.9890 | 0.7771 |
| `tower` | 0.250 | 0.4641 ± 0.0153 | 0.4924 ± 0.0034 | 0.9890 | 0.7771 |
| `tower` | 0.375 | 0.5391 ± 0.0367 | 0.5355 ± 0.0257 | 0.9890 | 0.7771 |
| `tower` | 0.500 | 0.6917 ± 0.0032 | 0.6005 ± 0.0067 | 0.9890 | 0.7771 |
| `tower` | 0.625 | 0.6771 ± 0.0132 | 0.6628 ± 0.0025 | 0.9890 | 0.7771 |
| `tower` | 0.750 | 0.6964 ± 0.0018 | 0.6951 ± 0.0033 | 0.9890 | 0.7771 |
| `tower` | 0.875 | 0.7026 ± 0.0114 | 0.7007 ± 0.0020 | 0.9890 | 0.7771 |
| `tower` | 1.000 | 0.6938 ± 0.0082 | 0.6616 ± 0.0071 | 0.9890 | 0.7771 |

### exolabs/Kimi-K2.6-vision, depth, capacity-matched

| Stage | Rel. Depth | indoors d1 | outdoor d1 | indoors coverage | outdoor coverage |
|---|---|---|---|---|---|
| `tower` | 0.111 | 0.3243 ± 0.0141 | 0.4315 ± 0.0019 | 0.9890 | 0.7771 |
| `tower` | 0.259 | 0.5375 ± 0.0180 | 0.5696 ± 0.0126 | 0.9890 | 0.7771 |
| `tower` | 0.370 | 0.6197 ± 0.0101 | 0.6125 ± 0.0088 | 0.9890 | 0.7771 |
| `tower` | 0.519 | 0.6617 ± 0.0077 | 0.6577 ± 0.0057 | 0.9890 | 0.7771 |
| `tower` | 0.630 | 0.6308 ± 0.0051 | 0.6599 ± 0.0069 | 0.9890 | 0.7771 |
| `tower` | 0.741 | 0.6386 ± 0.0012 | 0.6572 ± 0.0007 | 0.9890 | 0.7771 |
| `tower` | 0.889 | 0.6425 ± 0.0024 | 0.6529 ± 0.0049 | 0.9890 | 0.7771 |
| `tower` | 1.000 | 0.5953 ± 0.0108 | 0.6041 ± 0.0106 | 0.9890 | 0.7771 |
| `merged` | 1.000 | 0.6277 ± 0.0091 | 0.5987 ± 0.0062 | 0.9890 | 0.7771 |
| `projected` | 1.000 | 0.6646 ± 0.0107 | 0.6396 ± 0.0011 | 0.9890 | 0.7771 |

### immanuelpeter/MoonViT-V2, depth, capacity-matched

| Stage | Rel. Depth | indoors d1 | outdoor d1 | indoors coverage | outdoor coverage |
|---|---|---|---|---|---|
| `tower` | 0.111 | 0.3905 ± 0.0172 | 0.4266 ± 0.0055 | 0.9890 | 0.7771 |
| `tower` | 0.259 | 0.5163 ± 0.0083 | 0.5325 ± 0.0077 | 0.9890 | 0.7771 |
| `tower` | 0.370 | 0.5805 ± 0.0060 | 0.5826 ± 0.0035 | 0.9890 | 0.7771 |
| `tower` | 0.519 | 0.6302 ± 0.0018 | 0.6097 ± 0.0064 | 0.9890 | 0.7771 |
| `tower` | 0.630 | 0.6644 ± 0.0051 | 0.6469 ± 0.0012 | 0.9890 | 0.7771 |
| `tower` | 0.741 | 0.6392 ± 0.0027 | 0.6240 ± 0.0014 | 0.9890 | 0.7771 |
| `tower` | 0.889 | 0.6085 ± 0.0056 | 0.5937 ± 0.0007 | 0.9890 | 0.7771 |
| `tower` | 1.000 | 0.5460 ± 0.0041 | 0.5314 ± 0.0018 | 0.9890 | 0.7771 |
| `merged` | 1.000 | 0.6447 ± 0.0102 | 0.5805 ± 0.0023 | 0.9890 | 0.7771 |
| `projected` | 1.000 | 0.6607 ± 0.0093 | 0.5838 ± 0.0050 | 0.9890 | 0.7771 |

### meta-models/Muse-Glimmer-30B, depth, capacity-matched

| Stage | Rel. Depth | indoors d1 | outdoor d1 | indoors coverage | outdoor coverage |
|---|---|---|---|---|---|
| `tower` | 0.120 | 0.3875 ± 0.0003 | 0.4530 ± 0.0043 | 0.9890 | 0.7771 |
| `tower` | 0.240 | 0.4193 ± 0.0101 | 0.4790 ± 0.0039 | 0.9890 | 0.7771 |
| `tower` | 0.380 | 0.4873 ± 0.0130 | 0.5504 ± 0.0057 | 0.9890 | 0.7771 |
| `tower` | 0.500 | 0.5846 ± 0.0037 | 0.6073 ± 0.0039 | 0.9890 | 0.7771 |
| `tower` | 0.620 | 0.6317 ± 0.0053 | 0.6539 ± 0.0008 | 0.9890 | 0.7771 |
| `tower` | 0.760 | 0.6709 ± 0.0038 | 0.6492 ± 0.0018 | 0.9890 | 0.7771 |
| `tower` | 0.880 | 0.6346 ± 0.0007 | 0.6342 ± 0.0015 | 0.9890 | 0.7771 |
| `tower` | 1.000 | 0.5586 ± 0.0133 | 0.5181 ± 0.0313 | 0.9890 | 0.7771 |
| `merged` | 1.000 | 0.5540 ± 0.0023 | 0.4948 ± 0.0021 | 0.9890 | 0.7771 |
| `projected` | 1.000 | 0.6131 ± 0.0033 | 0.5718 ± 0.0011 | 0.9890 | 0.7771 |

### Qwen/Qwen3.8-27B, depth, capacity-matched

| Stage | Rel. Depth | indoors d1 | outdoor d1 | indoors coverage | outdoor coverage |
|---|---|---|---|---|---|
| `tower` | 0.111 | 0.4690 ± 0.0040 | 0.5090 ± 0.0069 | 0.9890 | 0.7771 |
| `tower` | 0.259 | 0.6027 ± 0.0046 | 0.6054 ± 0.0043 | 0.9890 | 0.7771 |
| `tower` | 0.370 | 0.6558 ± 0.0053 | 0.6281 ± 0.0035 | 0.9890 | 0.7771 |
| `tower` | 0.519 | 0.6606 ± 0.0125 | 0.6589 ± 0.0036 | 0.9890 | 0.7771 |
| `tower` | 0.630 | 0.6787 ± 0.0151 | 0.6695 ± 0.0050 | 0.9890 | 0.7771 |
| `tower` | 0.741 | 0.6648 ± 0.0084 | 0.6641 ± 0.0025 | 0.9890 | 0.7771 |
| `tower` | 0.889 | 0.6469 ± 0.0061 | 0.6314 ± 0.0137 | 0.9890 | 0.7771 |
| `tower` | 1.000 | 0.5474 ± 0.0091 | 0.5186 ± 0.0110 | 0.9890 | 0.7771 |
| `merged` | 1.000 | 0.5565 ± 0.0048 | 0.5083 ± 0.0057 | 0.9890 | 0.7771 |
| `projected` | 1.000 | 0.6347 ± 0.0054 | 0.5784 ± 0.0063 | 0.9890 | 0.7771 |

### google/siglip2-so400m-patch14-384, depth, capacity-matched

| Stage | Rel. Depth | indoors d1 | outdoor d1 | indoors coverage | outdoor coverage |
|---|---|---|---|---|---|
| `tower` | 0.111 | 0.4618 ± 0.0232 | 0.4980 ± 0.0051 | 0.9890 | 0.7771 |
| `tower` | 0.259 | 0.6201 ± 0.0050 | 0.6010 ± 0.0048 | 0.9890 | 0.7771 |
| `tower` | 0.370 | 0.6840 ± 0.0033 | 0.6209 ± 0.0015 | 0.9890 | 0.7771 |
| `tower` | 0.519 | 0.6959 ± 0.0054 | 0.6441 ± 0.0020 | 0.9890 | 0.7771 |
| `tower` | 0.630 | 0.6872 ± 0.0025 | 0.6567 ± 0.0028 | 0.9890 | 0.7771 |
| `tower` | 0.741 | 0.6707 ± 0.0045 | 0.6496 ± 0.0031 | 0.9890 | 0.7771 |
| `tower` | 0.889 | 0.6463 ± 0.0061 | 0.6253 ± 0.0008 | 0.9890 | 0.7771 |
| `tower` | 1.000 | 0.5964 ± 0.0090 | 0.5917 ± 0.0025 | 0.9890 | 0.7771 |

### Surface normals

### facebook/dinov2-large, normal, capacity-matched

| Stage | Rel. Depth | indoors mean_deg | outdoor mean_deg | indoors coverage | outdoor coverage |
|---|---|---|---|---|---|
| `tower` | 0.125 | 31.5517 ± 0.4030 | 30.2869 ± 0.0236 | 0.9803 | 0.6116 |
| `tower` | 0.250 | 30.1081 ± 2.5090 | 29.6770 ± 1.8333 | 0.9803 | 0.6116 |
| `tower` | 0.375 | 24.8067 ± 0.8240 | 25.9908 ± 0.5128 | 0.9803 | 0.6116 |
| `tower` | 0.500 | 21.3455 ± 0.9225 | 23.8685 ± 0.5166 | 0.9803 | 0.6116 |
| `tower` | 0.625 | 17.9393 ± 0.2919 | 22.1163 ± 0.0656 | 0.9803 | 0.6116 |
| `tower` | 0.750 | 15.5161 ± 0.2854 | 21.5080 ± 0.2427 | 0.9803 | 0.6116 |
| `tower` | 0.875 | 17.0385 ± 0.4951 | 23.0453 ± 0.3045 | 0.9803 | 0.6116 |
| `tower` | 1.000 | 20.5754 ± 0.1844 | 25.2796 ± 0.2308 | 0.9803 | 0.6116 |

### exolabs/Kimi-K2.6-vision, normal, capacity-matched

| Stage | Rel. Depth | indoors mean_deg | outdoor mean_deg | indoors coverage | outdoor coverage |
|---|---|---|---|---|---|
| `tower` | 0.111 | 33.3234 ± 0.5237 | 30.4659 ± 0.2882 | 0.9803 | 0.6116 |
| `tower` | 0.259 | 26.2951 ± 0.9247 | 26.9594 ± 0.4865 | 0.9803 | 0.6116 |
| `tower` | 0.370 | 24.8144 ± 1.1522 | 26.0659 ± 0.6797 | 0.9803 | 0.6116 |
| `tower` | 0.519 | 26.8214 ± 1.6830 | 27.5337 ± 1.1455 | 0.9803 | 0.6116 |
| `tower` | 0.630 | 27.8044 ± 1.6030 | 27.9862 ± 0.9865 | 0.9803 | 0.6116 |
| `tower` | 0.741 | 29.1103 ± 1.4521 | 28.7346 ± 0.9777 | 0.9803 | 0.6116 |
| `tower` | 0.889 | 29.3895 ± 0.5349 | 28.5858 ± 0.3295 | 0.9803 | 0.6116 |
| `tower` | 1.000 | 31.6976 ± 0.3729 | 29.8861 ± 0.2643 | 0.9803 | 0.6116 |
| `merged` | 1.000 | 31.5287 ± 0.5793 | 30.3316 ± 0.3530 | 0.9803 | 0.6116 |
| `projected` | 1.000 | 28.3783 ± 0.8417 | 28.6297 ± 0.4534 | 0.9803 | 0.6116 |

### immanuelpeter/MoonViT-V2, normal, capacity-matched

| Stage | Rel. Depth | indoors mean_deg | outdoor mean_deg | indoors coverage | outdoor coverage |
|---|---|---|---|---|---|
| `tower` | 0.111 | 33.1793 ± 1.3264 | 30.2392 ± 0.5465 | 0.9803 | 0.6116 |
| `tower` | 0.259 | 27.8205 ± 1.4053 | 28.1153 ± 0.7379 | 0.9803 | 0.6116 |
| `tower` | 0.370 | 27.1026 ± 1.6382 | 26.9681 ± 0.7385 | 0.9803 | 0.6116 |
| `tower` | 0.519 | 27.3764 ± 3.7206 | 27.5666 ± 2.7077 | 0.9803 | 0.6116 |
| `tower` | 0.630 | 26.8221 ± 3.3862 | 27.4622 ± 2.2816 | 0.9803 | 0.6116 |
| `tower` | 0.741 | 28.1536 ± 3.6574 | 28.5781 ± 2.3416 | 0.9803 | 0.6116 |
| `tower` | 0.889 | 30.1616 ± 2.9005 | 29.2913 ± 1.9019 | 0.9803 | 0.6116 |
| `tower` | 1.000 | 32.9197 ± 1.8179 | 30.8347 ± 1.0704 | 0.9803 | 0.6116 |
| `merged` | 1.000 | 29.2834 ± 0.9346 | 29.1165 ± 0.4595 | 0.9803 | 0.6116 |
| `projected` | 1.000 | 29.5762 ± 0.8415 | 29.0871 ± 0.4943 | 0.9803 | 0.6116 |

### meta-models/Muse-Glimmer-30B, normal, capacity-matched

| Stage | Rel. Depth | indoors mean_deg | outdoor mean_deg | indoors coverage | outdoor coverage |
|---|---|---|---|---|---|
| `tower` | 0.120 | 30.6567 ± 0.5668 | 29.7520 ± 0.2174 | 0.9803 | 0.6116 |
| `tower` | 0.240 | 30.2888 ± 0.5147 | 28.9993 ± 0.3116 | 0.9803 | 0.6116 |
| `tower` | 0.380 | 27.9959 ± 0.5433 | 28.0542 ± 0.3340 | 0.9803 | 0.6116 |
| `tower` | 0.500 | 25.7936 ± 0.8565 | 26.4496 ± 0.5414 | 0.9803 | 0.6116 |
| `tower` | 0.620 | 23.2179 ± 0.6207 | 25.1648 ± 0.3806 | 0.9803 | 0.6116 |
| `tower` | 0.760 | 29.2551 ± 1.1061 | 28.0877 ± 0.6570 | 0.9803 | 0.6116 |
| `tower` | 0.880 | 32.2114 ± 0.8936 | 29.5229 ± 0.6242 | 0.9803 | 0.6116 |
| `tower` | 1.000 | 32.4569 ± 0.5056 | 30.7989 ± 0.3334 | 0.9803 | 0.6116 |
| `merged` | 1.000 | 33.1057 ± 0.5533 | 31.5593 ± 0.2827 | 0.9803 | 0.6116 |
| `projected` | 1.000 | 31.2144 ± 0.4889 | 30.1262 ± 0.4034 | 0.9803 | 0.6116 |

### Qwen/Qwen3.8-27B, normal, capacity-matched

| Stage | Rel. Depth | indoors mean_deg | outdoor mean_deg | indoors coverage | outdoor coverage |
|---|---|---|---|---|---|
| `tower` | 0.111 | 29.7600 ± 1.0829 | 28.9785 ± 0.4812 | 0.9803 | 0.6116 |
| `tower` | 0.259 | 24.2661 ± 0.7613 | 25.8151 ± 0.4368 | 0.9803 | 0.6116 |
| `tower` | 0.370 | 22.8247 ± 1.2173 | 25.3652 ± 0.6719 | 0.9803 | 0.6116 |
| `tower` | 0.519 | 23.2956 ± 1.5939 | 25.6315 ± 0.7962 | 0.9803 | 0.6116 |
| `tower` | 0.630 | 26.3461 ± 3.6539 | 27.6789 ± 2.2232 | 0.9803 | 0.6116 |
| `tower` | 0.741 | 26.4572 ± 0.4082 | 27.0964 ± 0.3377 | 0.9803 | 0.6116 |
| `tower` | 0.889 | 27.6985 ± 0.4120 | 27.9232 ± 0.2566 | 0.9803 | 0.6116 |
| `tower` | 1.000 | 37.1364 ± 0.7564 | 32.3163 ± 0.3667 | 0.9803 | 0.6116 |
| `merged` | 1.000 | 39.1792 ± 0.2437 | 33.7673 ± 0.1447 | 0.9803 | 0.6116 |
| `projected` | 1.000 | 30.0894 ± 0.4852 | 29.7899 ± 0.2452 | 0.9803 | 0.6116 |

### google/siglip2-so400m-patch14-384, normal, capacity-matched

| Stage | Rel. Depth | indoors mean_deg | outdoor mean_deg | indoors coverage | outdoor coverage |
|---|---|---|---|---|---|
| `tower` | 0.111 | 30.4421 ± 0.6346 | 28.7674 ± 0.4382 | 0.9803 | 0.6116 |
| `tower` | 0.259 | 24.7360 ± 0.7296 | 25.6270 ± 0.3929 | 0.9803 | 0.6116 |
| `tower` | 0.370 | 23.1202 ± 0.5824 | 24.5285 ± 0.2585 | 0.9803 | 0.6116 |
| `tower` | 0.519 | 24.9286 ± 1.2404 | 25.7685 ± 0.6591 | 0.9803 | 0.6116 |
| `tower` | 0.630 | 28.3677 ± 2.4164 | 27.9878 ± 1.7287 | 0.9803 | 0.6116 |
| `tower` | 0.741 | 29.6583 ± 2.1893 | 28.9073 ± 1.4406 | 0.9803 | 0.6116 |
| `tower` | 0.889 | 31.5293 ± 1.8416 | 29.9019 ± 1.4215 | 0.9803 | 0.6116 |
| `tower` | 1.000 | 32.4316 ± 0.2859 | 29.8856 ± 0.3096 | 0.9803 | 0.6116 |

## Correspondence, September 2 2026

ADR-0020 confirms that Probe3D's ScanNet, NAVI, and SPair evaluations score frozen
features directly. No probe is trained. The bench uses full patch tokens at 448 square,
resizes each Stage map to a shared 64 by 64 matching grid, and retains 1,000
ratio-ranked matches for the geometric datasets. SPair samples at most 200 test pairs per
class and reports macro PCK at 0.1 of target bounding-box scale. This is a declared bench
adaptation, not a reproduction of Probe3D's published rows.

The continuation completed all 18 jobs. ScanNet was rerun after fixing a coordinate-frame
bug: its RGB frames are 1296 by 968, while depth and intrinsics use 640 by 480. RGB is now
aligned to that frame before the shared crop. The old floor-limited ScanNet files were
replaced.

| Tower | Stage | ScanNet recall@10px | NAVI recall@2cm | SPair macro PCK@0.1 |
| --- | --- | ---: | ---: | ---: |
| DINOv2 | `tower` | 0.08958 | 0.53891 | 0.55475 |
| SigLIP2 | `tower` | 0.05843 | 0.40133 | 0.39818 |
| MoonViT-V2 | `tower` | 0.02580 | 0.33342 | 0.27600 |
| MoonViT-V2 | `merged` | 0.04709 | 0.38818 | 0.25745 |
| MoonViT-V2 | `projected` | 0.04714 | 0.39244 | 0.26140 |
| Kimi K2.6 | `tower` | 0.05466 | 0.36458 | 0.17821 |
| Kimi K2.6 | `merged` | 0.01618 | 0.23138 | 0.06944 |
| Kimi K2.6 | `projected` | 0.06219 | 0.42111 | 0.29184 |
| Qwen3.8 | `tower` | 0.08337 | 0.37847 | 0.30806 |
| Qwen3.8 | `merged` | 0.00262 | 0.11297 | 0.02772 |
| Qwen3.8 | `projected` | 0.06396 | 0.39731 | 0.33091 |
| Muse Glimmer | `tower` | 0.02075 | 0.21392 | 0.09092 |
| Muse Glimmer | `merged` | 0.00898 | 0.18450 | 0.05641 |
| Muse Glimmer | `projected` | 0.05253 | 0.31970 | 0.22148 |

Paired intervals compare `projected` with the final `tower` Stage over image pairs. A
positive difference favours `projected`.

| Tower | column | difference | paired 95% interval | pairs |
| --- | --- | ---: | ---: | ---: |
| MoonViT-V2 | ScanNet recall@10px | +0.021341 | [+0.020162, +0.022543] | 1,500 |
| MoonViT-V2 | NAVI recall@2cm | +0.059020 | [+0.054193, +0.063709] | 555 |
| MoonViT-V2 | SPair PCK@0.1 | -0.014355 | [-0.021642, -0.007021] | 3,408 |
| Kimi K2.6 | ScanNet recall@10px | +0.007535 | [+0.006092, +0.008967] | 1,500 |
| Kimi K2.6 | NAVI recall@2cm | +0.056532 | [+0.052110, +0.060899] | 555 |
| Kimi K2.6 | SPair PCK@0.1 | +0.114964 | [+0.106328, +0.123475] | 3,408 |
| Qwen3.8 | ScanNet recall@10px | -0.019414 | [-0.020802, -0.018019] | 1,500 |
| Qwen3.8 | NAVI recall@2cm | +0.018836 | [+0.014590, +0.023038] | 555 |
| Qwen3.8 | SPair PCK@0.1 | +0.021649 | [+0.011830, +0.031645] | 3,408 |
| Muse Glimmer | ScanNet recall@10px | +0.031772 | [+0.029930, +0.033612] | 1,500 |
| Muse Glimmer | NAVI recall@2cm | +0.105779 | [+0.099457, +0.112103] | 555 |
| Muse Glimmer | SPair PCK@0.1 | +0.132510 | [+0.123934, +0.141108] | 3,408 |

NAVI is the spatial headline. All four Projectors improve its multiview 3D recall, and all
four paired intervals exclude zero. The corrected ScanNet column is no longer floor-limited.
MoonViT-V2, Kimi K2.6, and Muse Glimmer improve there; Qwen3.8 loses 0.01941 recall with an
interval wholly below zero. These ScanNet values remain a declared bench adaptation rather
than Probe3D reproduction numbers.

SPair measures semantic part matching across different object instances. MoonViT-V2 loses
0.01436 PCK there while gaining 0.05902 on NAVI. Its loss occurs mostly at the lossless
`tower` to `merged` step, and its Projector recovers only part of it. The other three
Projectors improve SPair. One Connector therefore loses semantic matching ability while
gaining geometric correspondence. That is not a refutation of the spatial claim because
SPair does not measure multiview 3D consistency.

The Kimi K2.6 raster-order audit found no bug. The published merge loop and adapter both
emit 2 by 2 blocks in raster order, and the existing weight-backed test checks that layout
bit for bit. Qwen3.8 and Muse Glimmer also collapse at `merged` on the three direct-cosine
columns, so the effect is not unique to Kimi. The training-free scorer is sensitive to
within-block phase after lossless concatenation.

Full per-pair results are under `results/correspondence/`. The twelve paired intervals are
under `results/bootstrap/` with the `correspondence-` prefix.
