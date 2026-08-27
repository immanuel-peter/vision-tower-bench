# Geometry pillar, second run

Every number here comes from the full 72-cell matrix run at the protocol in ADR-0014: a
four-point learning-rate grid searched on the validation split, selected per cell, and
reported as the mean over three seeds with the population standard deviation across those
seeds. The first geometry run trained one seed per cell at a fixed 1e-3 and searched
nothing. Nothing from it survives here unrestated. Levels moved a long way under rate
selection, so read this file rather than the first-run numbers quoted anywhere else.

This run answers the spatial half of the claim in PLAN.md. The semantic half was measured
first and held flat: MoonViT-V2 reads 0.8345 top-1 at `tower`, 0.8417 at `merged`, and
0.8412 at `projected` on ImageNet-100. PLAN.md predicted that the same Projector destroys
spatial information. It does not.

## Protocol

DIODE validation, 771 images, 325 indoors and 446 outdoor, prepared at native 768 by 1024
and evaluated on the centre 768 by 768 square that `square_crop` actually feeds each Stage
(ADR-0012). The split is 541 train, 115 val, 115 test, drawn once with a fixed seed and
reused in every cell. Each cell trains a head at every rate in the grid, picks the rate
that scores best on `val`, retrains at that rate under three seeds, and reports `test`.
Every cell sees identical data and searches an identical grid, so cells stay comparable.

Heads are the Probe3D multiscale decoder, one probe per Stage and Relative Depth point
(ADR-0010), trained for 10 epochs with AdamW and batch 8. The grid is
`[3e-4, 1e-3, 3e-3, 1e-2]` (ADR-0014). Depth bins span 0 to 299.83 m, read from the prep
manifest rather than assumed, because DIODE is metric and reaches 299.83 m outdoors
against NYU's 10 m indoors (ADR-0011).

Both capacity arms run. `raw` trains 1,706,752 parameters at a 1024-wide Stage and
4,852,480 at 7168, which is the `tower` against `projected` pair the claim compares.
`matched` fits a frozen PCA reduction to 512 on each cell's training split and trains
1,444,608 parameters everywhere for depth and 1,314,819 for normals (ADR-0008).

Two models times two tasks times two arms is 8 invocations over 72 cells, run as four
lanes on four L40S GPUs in 1 hour 58 minutes.

## What the run shows

Geometry does not collapse at the Projector. In both capacity arms and on both tasks the
`projected` Stage scores better than `tower`. The claim in PLAN.md, that the Projector
preserves language-useful semantics while destroying spatial information, is not
supported. The semantic half of the claim stands and the spatial half is dead as stated.

The run does produce a positive result, from the other axis. Both Towers reach their
geometry peak before their last layer while their semantic accuracy is still climbing,
so the two capabilities live at different Relative Depths in the same Tower. That result
is solid on depth in every arm and on DINOv2 normals, and weak on MoonViT-V2 normals,
where the seed spread swallows most of the curve.

### 1. Geometry does not drop from `tower` to `projected`

It improves, in all four model-arm combinations that have a Projector. Capacity-matched
MoonViT-V2 depth `d1` rises from 0.5378 at `tower` to 0.6145 at `projected`, a gain of
0.0767 against a pooled seed spread of 0.0065, so about twelve standard deviations.
Matched normals improve from 31.7420 to 29.3454 mean degrees, a gain of 2.3966 against a
pooled spread of 1.0416, which is 2.3. The unmatched arm agrees on both tasks, 0.4537 to
0.4775 on depth at 2.4 standard deviations and 33.2673 to 30.3852 on normals at 7.6.

Set that beside the semantic result on the same Stages, 0.8345 to 0.8412. Semantics held
flat and geometry improved. PLAN.md predicted semantics flat and geometry collapsing. The
first half is right and the second half is wrong, so this run publishes a null result on
the headline claim.

### 2. `merged` does not track `tower`, and that is the finding

This is the control the run turns on, and it fails in a way that reframes everything else.
The merge is a lossless regrouping of four Tower tokens into one, verified against the
cache on disk rather than only in a fresh forward pass: reshaping the cached `tower` L27
tokens into the `merged` layout reproduces the cached `merged` tensor exactly under
`torch.equal`, over all 771 images. The two Stages hold identical information. The probe
does not agree. Matched `merged` beats matched `tower` by 0.0684 `d1` on depth, 47 standard
deviations, and by 2.5188 degrees on normals.

That is the size of a readout artifact, measured on a pair known to carry the same
information, and it is the yardstick every other Stage comparison has to clear. The
Projector step does not come close. Matched `merged` to `projected` moves depth by 0.0083,
1.3 standard deviations, and normals by 0.1222 degrees, a fifth of one. A step that
provably loses nothing moves depth eight times further than the Projector does, and
normals twenty times further. No claim of spatial information loss survives that
comparison.

The unmatched arm is where a collapse story would come from. There `merged` to `projected`
reads -0.0430 on depth, 8.3 standard deviations, and -1.0608 degrees on normals, 0.9 and
not real. The depth figure is still smaller than the lossless yardstick in the same arm,
where `tower` to `merged` reads +0.0668, but only by a factor of 1.6. That is the tightest
this argument gets, and ADR-0013 covers what the unmatched depth gap is.

### 3. Geometry peaks earlier on Relative Depth than semantics

Supporting hypothesis 1 holds on depth everywhere and on normals in three of four arms.
MoonViT-V2 peaks at Relative Depth 0.630 on depth in both arms and on unmatched normals,
then declines to the final layer: matched depth falls from 0.6543 to 0.5378, a drop of
0.1165 at 75 standard deviations, and unmatched normals from 27.9623 to 33.2673 at 16.
DINOv2 peaks at 0.750 on normals in both arms, falling from 18.8500 to 23.1962 degrees on
the matched arm at 19 standard deviations, and at 0.875 on matched depth.

Two cells qualify that. Matched MoonViT-V2 normals nominally peak at 0.370 rather than
0.630, but 0.370 leads 0.630 by 0.1008 degrees against seed spreads of 1.1777 and 2.7421,
so the two are tied and neither is a measured peak; the fall from either to the last layer
is only 3.8 standard deviations. Unmatched DINOv2 depth nominally peaks at the last layer,
0.6778 against 0.6673 at 0.875, a lead of 1.1 standard deviations, so that curve is flat
over its last quarter rather than falling. Every other arm falls away from a peak by at
least 7.9.

The semantic probe on the same Towers rose monotonically to the last layer, 0.3644 at
Relative Depth 0.125 to 0.9026 at 1.0 for DINOv2 and 0.3793 to 0.8345 for MoonViT-V2. The
two capabilities peak in different places on the same Tower: its last quarter keeps adding
semantics while it sheds geometry. That is a sharper result than the Stage comparison
produced, and it is where the spatial information actually goes. It is also the one finding
here that a reviewer can act on, because it says which layer to read for a spatial task.

### 4. The answer is the same indoors and outdoors

Both scene types give the same three answers. Neither drops from `tower` to `projected`:
capacity-matched MoonViT-V2 depth gains 0.1179 `d1` indoors and 0.0438 outdoors, and
matched normals improve by 3.2816 degrees indoors and 1.6914 outdoors. Neither shows a
Projector step anywhere near the lossless `merged` step, indoors or out: `merged` to
`projected` moves 0.0153 and 0.0026 on depth against a `tower` to `merged` step of 0.1026
and 0.0412. And both peak early on depth, at Relative Depth 0.630 for either scene type on
MoonViT-V2 and 0.875 on DINOv2.

What differs is the level, not the conclusion. DINOv2 reads normals far better indoors
than outdoors at every depth point past 0.375, 15.5210 degrees against 21.5029 at the
shared peak of 0.750. MoonViT-V2 shows a smaller and noisier gap that changes sign along
the Tower: indoors trails at the first depth point and at the last, and leads in the
middle, with seed spreads above 3 degrees through the middle that cover the difference.
Outdoor depth runs slightly ahead of indoor depth in the shallow half of MoonViT-V2 and
behind it everywhere else, which is worth holding against the coverage difference below
rather than reading as a scene effect.

## The two capacity arms disagree on Stage ranking

ADR-0008 accepts capacity matching only if model rankings and Relative Depth curves agree
between the matched and unmatched runs, and requires any disagreement to be reported. The
Relative Depth curves agree on depth, where every model and arm peaks at the same point,
and disagree on normals only where the seed spread already covers the difference. The
Stage ranking on depth does not agree. The unmatched arm orders `merged` 0.5205,
`projected` 0.4775, `tower` 0.4537, while the matched arm orders `projected` 0.6145,
`merged` 0.6062, `tower` 0.5378. The middle two swap.

On the semantic side the arms agreed to within 0.0283 and changed no conclusion. Here the
swap is 0.0513 wide and is the difference between a Projector that appears to lose spatial
information and one that does not. ADR-0013 carries the mechanism and what this run
settled about it. Both arms are published below and neither is dropped.

## What the grid does not cover

29 of the 72 cells select a rate at the edge of the four-point grid, so their optimum lies
at or past that edge. All 16 capacity-matched depth cells select the 1e-2 top edge, and
their validation curves are still rising there. 13 normals cells select the 3e-4 bottom
edge, 12 of them unmatched, and their curves fall monotonically toward it.

That splits by token width, not by task or grid. A 512-wide cell tolerates 1e-2; every
cell at 1024 and above collapses to the degenerate 0.1245 there, so the unmatched arm
never reaches the top of the grid and the matched arm never leaves it. ADR-0014 chose the
four points from three cells and read the split as 32 by 32 against 16 by 16. The full
matrix says otherwise; that ADR is corrected.

The grid stays identical in every cell, which is what makes cells comparable, so the
truncation costs absolute level in the cells that hit an edge and does not move any
comparison between cells. It does bound the levels: matched depth in particular is
reported below its optimum.

## Depth

### AI4Industry/MoonViT-V2, depth, unmatched

| Stage | Rel. Depth | Width | Grid | Params | LR | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.111 | 1024 | 32x32 | 1,706,752 | 0.001 | 0.2861 ± 0.0050 | 0.5503 | 0.7310 | 6.3135 |
| `tower` | 0.259 | 1024 | 32x32 | 1,706,752 | 0.001 | 0.3388 ± 0.0107 | 0.6112 | 0.7798 | 5.7947 |
| `tower` | 0.370 | 1024 | 32x32 | 1,706,752 | 0.001 | 0.3317 ± 0.0177 | 0.6115 | 0.7897 | 5.5940 |
| `tower` | 0.519 | 1024 | 32x32 | 1,706,752 | 0.001 | 0.4342 ± 0.0114 | 0.7130 | 0.8532 | 4.9131 |
| `tower` | 0.630 | 1024 | 32x32 | 1,706,752 | 0.001 | 0.5441 ± 0.0098 | 0.7895 | 0.8935 | 4.2964 |
| `tower` | 0.741 | 1024 | 32x32 | 1,706,752 | 0.001 | 0.5058 ± 0.0064 | 0.7745 | 0.8870 | 4.3872 |
| `tower` | 0.889 | 1024 | 32x32 | 1,706,752 | 0.001 | 0.4453 ± 0.0067 | 0.7281 | 0.8592 | 4.7484 |
| `tower` | 1.000 | 1024 | 32x32 | 1,706,752 | 0.001 | 0.4537 ± 0.0127 | 0.7335 | 0.8601 | 4.7270 |
| `merged` | 1.000 | 4096 | 16x16 | 3,279,616 | 0.001 | 0.5205 ± 0.0039 | 0.7807 | 0.8894 | 4.4923 |
| `projected` | 1.000 | 7168 | 16x16 | 4,852,480 | 0.001 | 0.4775 ± 0.0062 | 0.7463 | 0.8674 | 4.7222 |

### AI4Industry/MoonViT-V2, depth, capacity-matched

| Stage | Rel. Depth | Width | Grid | Params | LR | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.111 | 512 | 32x32 | 1,444,608 | 0.01 | 0.4061 ± 0.0109 | 0.6851 | 0.8429 | 5.4080 |
| `tower` | 0.259 | 512 | 32x32 | 1,444,608 | 0.01 | 0.5273 ± 0.0019 | 0.7899 | 0.8970 | 4.5006 |
| `tower` | 0.370 | 512 | 32x32 | 1,444,608 | 0.01 | 0.5759 ± 0.0075 | 0.8206 | 0.9134 | 4.2161 |
| `tower` | 0.519 | 512 | 32x32 | 1,444,608 | 0.01 | 0.6217 ± 0.0062 | 0.8430 | 0.9304 | 3.9480 |
| `tower` | 0.630 | 512 | 32x32 | 1,444,608 | 0.01 | 0.6543 ± 0.0016 | 0.8575 | 0.9369 | 3.7900 |
| `tower` | 0.741 | 512 | 32x32 | 1,444,608 | 0.01 | 0.6314 ± 0.0029 | 0.8581 | 0.9371 | 3.8028 |
| `tower` | 0.889 | 512 | 32x32 | 1,444,608 | 0.01 | 0.5956 ± 0.0102 | 0.8319 | 0.9257 | 3.9928 |
| `tower` | 1.000 | 512 | 32x32 | 1,444,608 | 0.003 | 0.5378 ± 0.0015 | 0.7916 | 0.8956 | 4.3697 |
| `merged` | 1.000 | 512 | 16x16 | 1,444,608 | 0.01 | 0.6062 ± 0.0014 | 0.8375 | 0.9209 | 4.2212 |
| `projected` | 1.000 | 512 | 16x16 | 1,444,608 | 0.01 | 0.6145 ± 0.0090 | 0.8389 | 0.9221 | 4.2414 |

### facebook/dinov2-large, depth, unmatched

| Stage | Rel. Depth | Width | Grid | Params | LR | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.125 | 1024 | 32x32 | 1,706,752 | 0.001 | 0.3199 ± 0.0047 | 0.5841 | 0.7534 | 6.1063 |
| `tower` | 0.250 | 1024 | 32x32 | 1,706,752 | 0.001 | 0.3471 ± 0.0088 | 0.6258 | 0.7968 | 5.7611 |
| `tower` | 0.375 | 1024 | 32x32 | 1,706,752 | 0.001 | 0.3736 ± 0.0074 | 0.6615 | 0.8286 | 5.3638 |
| `tower` | 0.500 | 1024 | 32x32 | 1,706,752 | 0.001 | 0.5162 ± 0.0069 | 0.7952 | 0.9069 | 4.3569 |
| `tower` | 0.625 | 1024 | 32x32 | 1,706,752 | 0.003 | 0.6210 ± 0.0035 | 0.8521 | 0.9369 | 3.8066 |
| `tower` | 0.750 | 1024 | 32x32 | 1,706,752 | 0.001 | 0.6519 ± 0.0109 | 0.8717 | 0.9420 | 3.5438 |
| `tower` | 0.875 | 1024 | 32x32 | 1,706,752 | 0.001 | 0.6673 ± 0.0116 | 0.8755 | 0.9429 | 3.6080 |
| `tower` | 1.000 | 1024 | 32x32 | 1,706,752 | 0.003 | 0.6778 ± 0.0069 | 0.8808 | 0.9469 | 3.7088 |

### facebook/dinov2-large, depth, capacity-matched

| Stage | Rel. Depth | Width | Grid | Params | LR | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.125 | 512 | 32x32 | 1,444,608 | 0.003 | 0.4564 ± 0.0103 | 0.7128 | 0.8460 | 5.2648 |
| `tower` | 0.250 | 512 | 32x32 | 1,444,608 | 0.01 | 0.5185 ± 0.0114 | 0.7890 | 0.8970 | 4.7577 |
| `tower` | 0.375 | 512 | 32x32 | 1,444,608 | 0.01 | 0.5261 ± 0.0326 | 0.8000 | 0.9051 | 4.5067 |
| `tower` | 0.500 | 512 | 32x32 | 1,444,608 | 0.01 | 0.6413 ± 0.0056 | 0.8659 | 0.9376 | 3.8949 |
| `tower` | 0.625 | 512 | 32x32 | 1,444,608 | 0.01 | 0.6696 ± 0.0062 | 0.8763 | 0.9443 | 3.6127 |
| `tower` | 0.750 | 512 | 32x32 | 1,444,608 | 0.01 | 0.7009 ± 0.0048 | 0.8922 | 0.9510 | 3.4108 |
| `tower` | 0.875 | 512 | 32x32 | 1,444,608 | 0.01 | 0.7061 ± 0.0016 | 0.8969 | 0.9509 | 3.5011 |
| `tower` | 1.000 | 512 | 32x32 | 1,444,608 | 0.01 | 0.6774 ± 0.0049 | 0.8782 | 0.9466 | 3.6969 |


## Surface normals

`mean_deg` and `rmse` are angular error in degrees, so lower is better. `d1`,
`d2` and `d3` are the fraction of annotated pixels within 11.25, 22.5 and 30 degrees.

### AI4Industry/MoonViT-V2, normal, unmatched

| Stage | Rel. Depth | Width | Grid | Params | LR | mean_deg | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.111 | 1024 | 32x32 | 1,576,963 | 0.0003 | 35.2320 ± 0.1437 | 0.1677 | 0.3518 | 0.4762 | 41.5606 |
| `tower` | 0.259 | 1024 | 32x32 | 1,576,963 | 0.0003 | 31.9213 ± 0.2158 | 0.2058 | 0.4140 | 0.5397 | 38.4179 |
| `tower` | 0.370 | 1024 | 32x32 | 1,576,963 | 0.0003 | 30.5820 ± 0.2772 | 0.2269 | 0.4416 | 0.5659 | 37.2039 |
| `tower` | 0.519 | 1024 | 32x32 | 1,576,963 | 0.0003 | 29.4622 ± 0.4699 | 0.2535 | 0.4704 | 0.5901 | 36.3053 |
| `tower` | 0.630 | 1024 | 32x32 | 1,576,963 | 0.0003 | 27.9623 ± 0.3673 | 0.2811 | 0.5095 | 0.6238 | 35.0385 |
| `tower` | 0.741 | 1024 | 32x32 | 1,576,963 | 0.0003 | 30.0688 ± 0.5721 | 0.2449 | 0.4538 | 0.5721 | 36.8122 |
| `tower` | 0.889 | 1024 | 32x32 | 1,576,963 | 0.0003 | 31.0752 ± 0.2573 | 0.2262 | 0.4304 | 0.5487 | 37.6182 |
| `tower` | 1.000 | 1024 | 32x32 | 1,576,963 | 0.0003 | 33.2673 ± 0.2962 | 0.2045 | 0.3909 | 0.5085 | 39.8079 |
| `merged` | 1.000 | 4096 | 16x16 | 3,149,827 | 0.0003 | 31.4460 ± 1.5599 | 0.2320 | 0.4366 | 0.5519 | 38.3886 |
| `projected` | 1.000 | 7168 | 16x16 | 4,722,691 | 0.0003 | 30.3852 ± 0.4486 | 0.2515 | 0.4669 | 0.5798 | 37.7636 |

### AI4Industry/MoonViT-V2, normal, capacity-matched

| Stage | Rel. Depth | Width | Grid | Params | LR | mean_deg | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.111 | 512 | 32x32 | 1,314,819 | 0.001 | 31.4571 ± 0.7571 | 0.2302 | 0.4357 | 0.5540 | 38.4173 |
| `tower` | 0.259 | 512 | 32x32 | 1,314,819 | 0.001 | 28.0562 ± 1.1356 | 0.2788 | 0.5146 | 0.6278 | 35.2341 |
| `tower` | 0.370 | 512 | 32x32 | 1,314,819 | 0.001 | 27.0576 ± 1.1777 | 0.2971 | 0.5337 | 0.6458 | 34.2678 |
| `tower` | 0.519 | 512 | 32x32 | 1,314,819 | 0.001 | 27.4909 ± 3.1550 | 0.3052 | 0.5370 | 0.6369 | 34.8219 |
| `tower` | 0.630 | 512 | 32x32 | 1,314,819 | 0.001 | 27.1584 ± 2.7421 | 0.3070 | 0.5336 | 0.6394 | 34.4539 |
| `tower` | 0.741 | 512 | 32x32 | 1,314,819 | 0.001 | 28.3740 ± 2.9346 | 0.2879 | 0.5035 | 0.6166 | 35.5771 |
| `tower` | 0.889 | 512 | 32x32 | 1,314,819 | 0.001 | 29.6905 ± 2.3797 | 0.2633 | 0.4772 | 0.5898 | 36.7955 |
| `tower` | 1.000 | 512 | 32x32 | 1,314,819 | 0.001 | 31.7420 ± 1.3131 | 0.2373 | 0.4370 | 0.5481 | 38.8972 |
| `merged` | 1.000 | 512 | 16x16 | 1,314,819 | 0.001 | 29.2232 ± 0.6953 | 0.2764 | 0.5014 | 0.6103 | 36.9184 |
| `projected` | 1.000 | 512 | 16x16 | 1,314,819 | 0.001 | 29.3454 ± 0.6675 | 0.2721 | 0.4958 | 0.6067 | 36.9464 |

### facebook/dinov2-large, normal, unmatched

| Stage | Rel. Depth | Width | Grid | Params | LR | mean_deg | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.125 | 1024 | 32x32 | 1,576,963 | 0.001 | 33.5269 ± 0.1331 | 0.1878 | 0.3779 | 0.5026 | 39.8670 |
| `tower` | 0.250 | 1024 | 32x32 | 1,576,963 | 0.001 | 31.1396 ± 0.1478 | 0.2188 | 0.4264 | 0.5529 | 37.7243 |
| `tower` | 0.375 | 1024 | 32x32 | 1,576,963 | 0.001 | 28.1179 ± 0.0862 | 0.2690 | 0.5018 | 0.6206 | 35.1142 |
| `tower` | 0.500 | 1024 | 32x32 | 1,576,963 | 0.001 | 23.5504 ± 0.2720 | 0.3721 | 0.6146 | 0.7145 | 30.9373 |
| `tower` | 0.625 | 1024 | 32x32 | 1,576,963 | 0.001 | 20.2918 ± 0.4124 | 0.4689 | 0.6938 | 0.7731 | 28.0644 |
| `tower` | 0.750 | 1024 | 32x32 | 1,576,963 | 0.0003 | 19.4548 ± 0.0934 | 0.5119 | 0.7156 | 0.7833 | 27.4695 |
| `tower` | 0.875 | 1024 | 32x32 | 1,576,963 | 0.0003 | 20.2910 ± 0.1639 | 0.4876 | 0.6987 | 0.7711 | 28.3627 |
| `tower` | 1.000 | 1024 | 32x32 | 1,576,963 | 0.001 | 22.2100 ± 0.2557 | 0.4209 | 0.6574 | 0.7420 | 30.1235 |

### facebook/dinov2-large, normal, capacity-matched

| Stage | Rel. Depth | Width | Grid | Params | LR | mean_deg | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.125 | 512 | 32x32 | 1,314,819 | 0.001 | 30.8365 ± 0.2288 | 0.2288 | 0.4438 | 0.5682 | 37.6737 |
| `tower` | 0.250 | 512 | 32x32 | 1,314,819 | 0.003 | 29.3149 ± 1.0814 | 0.2465 | 0.4766 | 0.6054 | 36.3978 |
| `tower` | 0.375 | 512 | 32x32 | 1,314,819 | 0.003 | 25.4561 ± 0.8159 | 0.3251 | 0.5787 | 0.6829 | 32.8569 |
| `tower` | 0.500 | 512 | 32x32 | 1,314,819 | 0.003 | 22.7559 ± 0.6928 | 0.3945 | 0.6405 | 0.7332 | 30.2941 |
| `tower` | 0.625 | 512 | 32x32 | 1,314,819 | 0.001 | 20.2734 ± 0.1735 | 0.4740 | 0.6951 | 0.7730 | 28.1122 |
| `tower` | 0.750 | 512 | 32x32 | 1,314,819 | 0.001 | 18.8500 ± 0.2525 | 0.5341 | 0.7270 | 0.7915 | 26.9738 |
| `tower` | 0.875 | 512 | 32x32 | 1,314,819 | 0.001 | 20.3733 ± 0.3741 | 0.4806 | 0.6994 | 0.7716 | 28.3238 |
| `tower` | 1.000 | 512 | 32x32 | 1,314,819 | 0.0003 | 23.1962 ± 0.1988 | 0.4028 | 0.6315 | 0.7212 | 31.2007 |


## Indoors against outdoor

Coverage is the fraction of each target the metric can score, averaged over the test
split. It differs sharply by scene type and by task: depth is annotated on 0.9890 of an
indoor image and 0.7771 of an outdoor one, while normals reach 0.9803 indoors and only
0.6116 outdoors. DIODE ships no validity mask for normals, so an unannotated pixel is a
zero vector and validity comes from normal magnitude. Read every outdoor normals number as
an average over about three fifths of the frame.

### AI4Industry/MoonViT-V2, depth, capacity-matched

| Stage | Rel. Depth | indoors d1 | outdoor d1 | indoors coverage | outdoor coverage |
|---|---|---|---|---|---|
| `tower` | 0.111 | 0.3868 ± 0.0127 | 0.4215 ± 0.0104 | 0.9890 | 0.7771 |
| `tower` | 0.259 | 0.5196 ± 0.0019 | 0.5334 ± 0.0020 | 0.9890 | 0.7771 |
| `tower` | 0.370 | 0.5699 ± 0.0100 | 0.5806 ± 0.0059 | 0.9890 | 0.7771 |
| `tower` | 0.519 | 0.6313 ± 0.0061 | 0.6140 ± 0.0074 | 0.9890 | 0.7771 |
| `tower` | 0.630 | 0.6663 ± 0.0039 | 0.6447 ± 0.0002 | 0.9890 | 0.7771 |
| `tower` | 0.741 | 0.6408 ± 0.0036 | 0.6239 ± 0.0027 | 0.9890 | 0.7771 |
| `tower` | 0.889 | 0.6070 ± 0.0060 | 0.5865 ± 0.0139 | 0.9890 | 0.7771 |
| `tower` | 1.000 | 0.5420 ± 0.0071 | 0.5344 ± 0.0037 | 0.9890 | 0.7771 |
| `merged` | 1.000 | 0.6446 ± 0.0043 | 0.5756 ± 0.0013 | 0.9890 | 0.7771 |
| `projected` | 1.000 | 0.6599 ± 0.0132 | 0.5782 ± 0.0067 | 0.9890 | 0.7771 |

### AI4Industry/MoonViT-V2, normal, capacity-matched

| Stage | Rel. Depth | indoors mean_deg | outdoor mean_deg | indoors coverage | outdoor coverage |
|---|---|---|---|---|---|
| `tower` | 0.111 | 33.0059 ± 1.0477 | 30.2229 ± 0.5265 | 0.9803 | 0.6116 |
| `tower` | 0.259 | 27.9178 ± 1.5304 | 28.1666 ± 0.8210 | 0.9803 | 0.6116 |
| `tower` | 0.370 | 27.1065 ± 1.6564 | 27.0186 ± 0.7967 | 0.9803 | 0.6116 |
| `tower` | 0.519 | 27.3853 ± 3.7252 | 27.5751 ± 2.7010 | 0.9803 | 0.6116 |
| `tower` | 0.630 | 26.7959 ± 3.3259 | 27.4472 ± 2.2769 | 0.9803 | 0.6116 |
| `tower` | 0.741 | 28.1312 ± 3.6720 | 28.5676 ± 2.3470 | 0.9803 | 0.6116 |
| `tower` | 0.889 | 30.1737 ± 2.9582 | 29.3055 ± 1.9200 | 0.9803 | 0.6116 |
| `tower` | 1.000 | 32.9150 ± 1.7641 | 30.8073 ± 0.9561 | 0.9803 | 0.6116 |
| `merged` | 1.000 | 29.3405 ± 0.9091 | 29.1296 ± 0.5279 | 0.9803 | 0.6116 |
| `projected` | 1.000 | 29.6334 ± 0.8878 | 29.1159 ± 0.4927 | 0.9803 | 0.6116 |

### facebook/dinov2-large, depth, capacity-matched

| Stage | Rel. Depth | indoors d1 | outdoor d1 | indoors coverage | outdoor coverage |
|---|---|---|---|---|---|
| `tower` | 0.125 | 0.4806 ± 0.0144 | 0.4372 ± 0.0072 | 0.9890 | 0.7771 |
| `tower` | 0.250 | 0.5274 ± 0.0190 | 0.5115 ± 0.0084 | 0.9890 | 0.7771 |
| `tower` | 0.375 | 0.5240 ± 0.0493 | 0.5277 ± 0.0215 | 0.9890 | 0.7771 |
| `tower` | 0.500 | 0.6872 ± 0.0002 | 0.6047 ± 0.0101 | 0.9890 | 0.7771 |
| `tower` | 0.625 | 0.6817 ± 0.0119 | 0.6599 ± 0.0021 | 0.9890 | 0.7771 |
| `tower` | 0.750 | 0.7047 ± 0.0068 | 0.6979 ± 0.0045 | 0.9890 | 0.7771 |
| `tower` | 0.875 | 0.7129 ± 0.0032 | 0.7006 ± 0.0005 | 0.9890 | 0.7771 |
| `tower` | 1.000 | 0.6937 ± 0.0062 | 0.6643 ± 0.0076 | 0.9890 | 0.7771 |

### facebook/dinov2-large, normal, capacity-matched

| Stage | Rel. Depth | indoors mean_deg | outdoor mean_deg | indoors coverage | outdoor coverage |
|---|---|---|---|---|---|
| `tower` | 0.125 | 31.5905 ± 0.4123 | 30.2356 ± 0.0844 | 0.9803 | 0.6116 |
| `tower` | 0.250 | 29.5397 ± 1.1970 | 29.1358 ± 0.9957 | 0.9803 | 0.6116 |
| `tower` | 0.375 | 24.7116 ± 1.0924 | 26.0493 ± 0.5995 | 0.9803 | 0.6116 |
| `tower` | 0.500 | 21.3494 ± 0.9238 | 23.8767 ± 0.5108 | 0.9803 | 0.6116 |
| `tower` | 0.625 | 17.9439 ± 0.2944 | 22.1298 ± 0.0775 | 0.9803 | 0.6116 |
| `tower` | 0.750 | 15.5210 ± 0.2830 | 21.5029 ± 0.2286 | 0.9803 | 0.6116 |
| `tower` | 0.875 | 17.0204 ± 0.4787 | 23.0453 ± 0.2914 | 0.9803 | 0.6116 |
| `tower` | 1.000 | 20.5807 ± 0.1848 | 25.2805 ± 0.2325 | 0.9803 | 0.6116 |


## What this run does not establish

541 training images for a head of 1.3 to 4.9 million parameters is thin, so the absolute
numbers are noisy and should not be compared against published Probe3D results. Nobody has
published Probe3D numbers on DIODE in any case (ADR-0011). Every question above is a
comparison between cells that saw identical data, an identical head and an identical rate
grid, and those comparisons hold; the levels do not.

Seed spread is not uniform across the matrix, and the normals cells carry most of it.
Capacity-matched MoonViT-V2 normals run above 2.7 degrees of spread through the middle of
the Tower against 0.09 to 0.4 on unmatched DINOv2 normals, which is why the peak position
survives on one and not the other. Three seeds size that spread; they do not shrink it.

The depth head bins over 0 to 299.83 m because DIODE is metric and reaches that far
outdoors. That range is correct for the data and coarse for indoor scenes, which live in
the first nine of 256 bins. The bin floor also pulls an untrained prediction to about
150 m. Both effects are identical in every cell, so they cost absolute indoor depth
accuracy without touching any Stage or Relative Depth comparison.

Correspondence is still unscoped, the multilayer run in ADR-0010 has not been done, and
the pooling validation ADR-0005 requires is still outstanding. The DIODE training split
was not downloaded; at 222 GB it waits on this run's answer, and this run's answer is
that the Stage comparison is not where the effect lives.

# Semantic pillar, first run

This run uses the 13,000-image, 100-class ImageNet-100 validation split. An A100 80GB
extracted features at 448 square and pooled them to a 4x4 grid (ADR-0005). Each cell
searches eight learning rates, selects one on the validation subset, and reports mean
test accuracy over three seeds.

Each model has four arms. Attention and mean readouts both run with and without the PCA
capacity matching from ADR-0008. `matched` trains 1,627,748 parameters in every cell.
`raw` trains 2,152,036 at width 1024 and 8,443,492 at width 7168.

The geometry pillar has since run; see above. The pooling validation required by ADR-0005
is still outstanding.

## What the run shows

The Projector did not reduce semantic accuracy in this run. Under capacity matching,
MoonViT-V2 reads 0.8345 at `tower`, 0.8417 at `merged`, and 0.8412 at `projected`.
Without matching, the same three are 0.8373, 0.8390, and 0.8362. The claim in PLAN.md
pairs this with a spatial collapse. The geometry run above tested it and did not find one.

`merged` matching `tower` does not measure information loss from merging. For this Tower,
the merge is a lossless regrouping of four Tower tokens into one. `torch.equal` verifies
the regrouping in `tests/test_moonvit_v2.py`. The informative comparison is `projected`
against `merged`.

Capacity matching passes ADR-0008's acceptance test. Across the attention arms, the
largest gap between matched and raw is 0.0283 and the mean gap is 0.0113. Model ordering
is the same in both, and both curves rise monotonically with Relative Depth. The reduction
changes no conclusion.

The attention pool leads mean pooling by 8 to 12 points at shallow and middle depths. The
two converge at the last DINOv2 layer, where attention reaches 0.9026 and mean pooling
reaches 0.8911.

DINOv2 leads MoonViT-V2 at every comparable Relative Depth, 0.9026 against 0.8345 at the
last layer. DINOv2 trained on ImageNet, so read this as one task rather than a ranking.
