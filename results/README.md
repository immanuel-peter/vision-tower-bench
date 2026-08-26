# Geometry pillar, first run

> **Status, 2026-08-26.** Every number in this file comes from the first geometry run,
> which trained one seed per cell at a fixed 1e-3 and ran no learning-rate search. The
> runner has since been rewritten to search a validation-selected grid and report three
> seeds (ADR-0014). The re-run of all 72 cells was launched and stopped after 3 cells
> when the instance was torn down, so nothing here has been restated at the new
> protocol yet. Treat the levels below as provisional and the ranking claims as
> untested against seed noise outside the deepest Stage triple.

This run answers the spatial half of the claim in PLAN.md. The semantic half was measured
first and held flat: MoonViT-V2 reads 0.8345 top-1 at `tower`, 0.8417 at `merged`, and
0.8412 at `projected` on ImageNet-100. PLAN.md predicted that the same Projector destroys
spatial information. It does not.

## Protocol

DIODE validation, 771 images, 325 indoors and 446 outdoor, prepared at native 768 by 1024
and evaluated on the centre 768 by 768 square that `square_crop` actually feeds each Stage
(ADR-0012). The split is 541 train, 115 val, 115 test, drawn once with a fixed seed and
reused in every cell. The geometry runner trains on `train` and reports `test`. At the time
of this run it ran no learning-rate search, so the val subset went unused and every cell saw
identical data. The runner no longer works that way; see the status note above.

Heads are the Probe3D multiscale decoder, one probe per Stage and Relative Depth point
(ADR-0010), trained for 10 epochs with AdamW at 1e-3 and batch 8. Depth bins span 0 to
299.83 m, read from the prep manifest rather than assumed, because DIODE is metric and
reaches 299.83 m outdoors against NYU's 10 m indoors (ADR-0011).

Both capacity arms run. `raw` trains 1,706,752 parameters at a 1024-wide Stage and
4,852,480 at 7168, which is the `tower` against `projected` pair the claim compares.
`matched` fits a frozen PCA reduction to 512 on each cell's training split and trains
1,444,608 parameters everywhere for depth and 1,314,819 for normals (ADR-0008).

Two models times two tasks times two arms is 8 invocations over 72 cells. Because the
runner trains one seed per cell, the deepest Stage triple was rerun at two further seeds
in both arms to size every gap below against run-to-run noise.

## What the run shows

Geometry does not collapse at the Projector. In both capacity arms and on both tasks the
`projected` Stage scores at least as well as `tower`. The claim in PLAN.md, that the
Projector preserves language-useful semantics while destroying spatial information, is not
supported. The semantic half of the claim stands and the spatial half is dead as stated.

The run does produce a positive result, from the other axis. Both Towers reach their
geometry peak well before their last layer while their semantic accuracy is still climbing,
so the two capabilities live at different Relative Depths in the same Tower.

Every number below for the deepest Stage triple is a mean over three seeds, with the
spread quoted as a standard deviation over those seeds. Cells away from that triple ran
one seed, so their levels carry the same noise without a measurement of it.

### 1. Geometry does not drop from `tower` to `projected`

It improves. Capacity-matched MoonViT-V2 depth `d1` rises from 0.4971 at `tower` to
0.5579 at `projected`, a gain of 0.0608 against a pooled seed spread of 0.0063, so about
nine standard deviations. Matched normals improve from 31.7655 to 29.2700 mean degrees, a
gain of 2.4955 against a pooled spread of 1.5090. The unmatched arm agrees in direction on
both tasks, 0.4575 to 0.4860 on depth and 34.2426 to 30.8729 on normals, though at 1.9 and
1.6 standard deviations it establishes only that nothing collapsed.

Set that beside the semantic result on the same Stages, 0.8345 to 0.8412. Semantics held
flat and geometry improved. PLAN.md predicted semantics flat and geometry collapsing. The
first half is right and the second half is wrong, so this run publishes a null result on
the headline claim.

### 2. `merged` does not track `tower`, and that is the finding

This is the control the run turns on, and it fails in a way that reframes everything else.
The merge is a lossless regrouping of four Tower tokens into one, verified here against the
cache on disk rather than only in a fresh forward pass: reshaping the cached `tower` L27
tokens into the `merged` layout reproduces the cached `merged` tensor exactly under
`torch.equal`. The two Stages hold identical information. The probe does not agree. Matched
`merged` beats matched `tower` by 0.0536 `d1` on depth, about eight standard deviations,
and by 2.6877 degrees on normals.

That is the size of a readout artifact, measured on a pair known to carry the same
information, and it is the yardstick every other Stage comparison has to clear. The
Projector step does not come close. Matched `merged` to `projected` moves depth by 0.0071,
one standard deviation, and normals by 0.1922 degrees, a quarter of one. A step that
provably loses nothing moves the metric roughly seven times further on depth than the
Projector does. No claim of spatial information loss survives that comparison.

The unmatched arm is where a collapse story would come from, and it does not hold up. There
`merged` to `projected` reads -0.0386 on depth, which is 4.6 standard deviations and looks
real, and -1.8269 degrees on normals, which is 0.6 and does not. The depth figure is a
capacity effect rather than an information effect: the unmatched `projected` head carries
4,852,480 parameters against 3,279,616 at `merged` and trains on 541 images, making it the
widest and least stable cell in the matrix. Its normals counterpart swings by 4.4931
degrees across seeds, against 0.1547 at the narrow `tower` Stage. Reducing every cell to
512 dimensions removes both the instability and the gap, which is exactly the confound
ADR-0008 built the matched arm to catch.

### 3. Geometry peaks earlier on Relative Depth than semantics

Supporting hypothesis 1 holds, in every model, task and arm. MoonViT-V2 peaks at Relative
Depth 0.630 on both tasks in both arms, then declines to the final layer: matched depth
falls from 0.6074 to 0.4985 and matched normals from 25.0813 to 30.4182 degrees. DINOv2
peaks at 0.875 on depth and 0.750 on normals, again in both arms, falling from 18.6189 to
23.1905 degrees on matched normals.

The semantic probe on the same Towers rose monotonically to the last layer, 0.3644 at
Relative Depth 0.125 to 0.9026 at 1.0 for DINOv2 and 0.3793 to 0.8345 for MoonViT-V2. The
two capabilities peak in different places on the same Tower: its last quarter keeps adding
semantics while it sheds geometry. That is a sharper result than the Stage comparison
produced, and it is where the spatial information actually goes. It is also the one finding
here that a reviewer can act on, because it says which layer to read for a spatial task.

### 4. The answer is the same indoors and outdoors

Both scene types give the same three answers. Neither drops from `tower` to `projected`:
capacity-matched MoonViT-V2 depth gains 0.0830 `d1` indoors and 0.0431 outdoors, and
matched normals improve by 3.4525 degrees indoors and 1.7329 outdoors, all averaged over
three seeds. Neither shows a Projector step anywhere near the lossless `merged` step,
indoors or out: `merged` to `projected` moves 0.0136 and 0.0020 on depth against a
`tower` to `merged` step of 0.0694 and 0.0411. And both peak early, at Relative Depth
0.630 on depth for either scene type and at 0.630 indoors against 0.519 outdoors on
normals.

What differs is the level, not the conclusion. DINOv2 reads normals far better indoors
than outdoors at every depth point past 0.375, 15.2565 degrees against 21.2982 at the
shared peak of 0.750. MoonViT-V2 shows a smaller and partly reversed gap: indoors leads on
normals through the middle of the Tower and trails at the last layer. Outdoor depth runs
slightly ahead of indoor depth in both models, which is worth holding against the coverage
difference below rather than reading as an outdoor advantage. Scene-split figures away from
the deepest Stage triple, including every peak position above, come from one seed.

## The two capacity arms disagree on Stage ranking

ADR-0008 accepts capacity matching only if model rankings and Relative Depth curves agree
between the matched and unmatched runs, and requires any disagreement to be reported. The
Relative Depth curves agree: every model, task and arm peaks at the same point, and the
tables below show the same shape. The Stage ranking on depth does not. The unmatched arm
orders `merged` 0.5246, `projected` 0.4860, `tower` 0.4575, while the matched arm orders
`projected` 0.5579, `merged` 0.5507, `tower` 0.4971. The middle two swap.

On the semantic side the arms agreed to within 0.0283 and changed no conclusion. Here the
swap is 0.0457 wide and is the difference between a Projector that appears to lose spatial
information and one that does not. This is recorded in ADR-0013 rather than resolved: the
matched arm is the one ADR-0010 nominates for the headline, the unmatched gap tracks head
width rather than Stage, and 541 training images is too thin a base to settle it. Both arms
are published below and neither is dropped.

## Depth


### AI4Industry/MoonViT-V2, depth, unmatched

| Stage | Rel. Depth | Width | Grid | Params | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|
| `tower` | 0.111 | 1024 | 32x32 | 1,706,752 | 0.2943 | 0.5568 | 0.7348 | 6.3144 |
| `tower` | 0.259 | 1024 | 32x32 | 1,706,752 | 0.3386 | 0.6112 | 0.7797 | 5.7902 |
| `tower` | 0.370 | 1024 | 32x32 | 1,706,752 | 0.3318 | 0.6099 | 0.7870 | 5.6402 |
| `tower` | 0.519 | 1024 | 32x32 | 1,706,752 | 0.4447 | 0.7176 | 0.8588 | 4.9105 |
| `tower` | 0.630 | 1024 | 32x32 | 1,706,752 | 0.5573 | 0.7972 | 0.9012 | 4.2427 |
| `tower` | 0.741 | 1024 | 32x32 | 1,706,752 | 0.5142 | 0.7828 | 0.8912 | 4.3585 |
| `tower` | 0.889 | 1024 | 32x32 | 1,706,752 | 0.4549 | 0.7358 | 0.8637 | 4.7189 |
| `tower` | 1.000 | 1024 | 32x32 | 1,706,752 | 0.4660 | 0.7446 | 0.8662 | 4.7004 |
| `merged` | 1.000 | 4096 | 16x16 | 3,279,616 | 0.5197 | 0.7815 | 0.8894 | 4.4868 |
| `projected` | 1.000 | 7168 | 16x16 | 4,852,480 | 0.4941 | 0.7583 | 0.8747 | 4.6642 |

### AI4Industry/MoonViT-V2, depth, capacity-matched

| Stage | Rel. Depth | Width | Grid | Params | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|
| `tower` | 0.111 | 512 | 32x32 | 1,444,608 | 0.3740 | 0.6485 | 0.8125 | 5.4944 |
| `tower` | 0.259 | 512 | 32x32 | 1,444,608 | 0.4648 | 0.7311 | 0.8654 | 4.7647 |
| `tower` | 0.370 | 512 | 32x32 | 1,444,608 | 0.4955 | 0.7642 | 0.8815 | 4.4878 |
| `tower` | 0.519 | 512 | 32x32 | 1,444,608 | 0.5585 | 0.8074 | 0.9093 | 4.1864 |
| `tower` | 0.630 | 512 | 32x32 | 1,444,608 | 0.6074 | 0.8245 | 0.9177 | 3.9160 |
| `tower` | 0.741 | 512 | 32x32 | 1,444,608 | 0.5916 | 0.8293 | 0.9208 | 3.9227 |
| `tower` | 0.889 | 512 | 32x32 | 1,444,608 | 0.5579 | 0.8077 | 0.9113 | 4.0856 |
| `tower` | 1.000 | 512 | 32x32 | 1,444,608 | 0.4985 | 0.7667 | 0.8781 | 4.4954 |
| `merged` | 1.000 | 512 | 16x16 | 1,444,608 | 0.5548 | 0.8000 | 0.8991 | 4.3433 |
| `projected` | 1.000 | 512 | 16x16 | 1,444,608 | 0.5560 | 0.7992 | 0.8967 | 4.3933 |

### facebook/dinov2-large, depth, unmatched

| Stage | Rel. Depth | Width | Grid | Params | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|
| `tower` | 0.125 | 1024 | 32x32 | 1,706,752 | 0.3239 | 0.5826 | 0.7539 | 6.1289 |
| `tower` | 0.250 | 1024 | 32x32 | 1,706,752 | 0.3540 | 0.6304 | 0.8040 | 5.7307 |
| `tower` | 0.375 | 1024 | 32x32 | 1,706,752 | 0.3830 | 0.6728 | 0.8380 | 5.2865 |
| `tower` | 0.500 | 1024 | 32x32 | 1,706,752 | 0.5247 | 0.8023 | 0.9105 | 4.3312 |
| `tower` | 0.625 | 1024 | 32x32 | 1,706,752 | 0.6200 | 0.8517 | 0.9338 | 3.7982 |
| `tower` | 0.750 | 1024 | 32x32 | 1,706,752 | 0.6536 | 0.8751 | 0.9435 | 3.5478 |
| `tower` | 0.875 | 1024 | 32x32 | 1,706,752 | 0.6733 | 0.8796 | 0.9454 | 3.6166 |
| `tower` | 1.000 | 1024 | 32x32 | 1,706,752 | 0.6733 | 0.8786 | 0.9461 | 3.6766 |

### facebook/dinov2-large, depth, capacity-matched

| Stage | Rel. Depth | Width | Grid | Params | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|
| `tower` | 0.125 | 512 | 32x32 | 1,444,608 | 0.4244 | 0.6776 | 0.8088 | 5.4631 |
| `tower` | 0.250 | 512 | 32x32 | 1,444,608 | 0.4373 | 0.7145 | 0.8581 | 5.0623 |
| `tower` | 0.375 | 512 | 32x32 | 1,444,608 | 0.4550 | 0.7341 | 0.8760 | 4.7519 |
| `tower` | 0.500 | 512 | 32x32 | 1,444,608 | 0.5335 | 0.8043 | 0.9120 | 4.2021 |
| `tower` | 0.625 | 512 | 32x32 | 1,444,608 | 0.6112 | 0.8411 | 0.9293 | 3.7828 |
| `tower` | 0.750 | 512 | 32x32 | 1,444,608 | 0.6561 | 0.8753 | 0.9438 | 3.5040 |
| `tower` | 0.875 | 512 | 32x32 | 1,444,608 | 0.6769 | 0.8851 | 0.9473 | 3.5802 |
| `tower` | 1.000 | 512 | 32x32 | 1,444,608 | 0.6742 | 0.8778 | 0.9472 | 3.6674 |

## Surface normals

`mean_deg` and `rmse` are angular error in degrees, so lower is better. `d1`,
`d2` and `d3` are the fraction of annotated pixels within 11.25, 22.5 and 30 degrees.


### AI4Industry/MoonViT-V2, normal, unmatched

| Stage | Rel. Depth | Width | Grid | Params | mean_deg | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.111 | 1024 | 32x32 | 1,576,963 | 38.0106 | 0.1480 | 0.3115 | 0.4144 | 43.9852 |
| `tower` | 0.259 | 1024 | 32x32 | 1,576,963 | 35.0798 | 0.1956 | 0.3618 | 0.4667 | 41.5367 |
| `tower` | 0.370 | 1024 | 32x32 | 1,576,963 | 34.3798 | 0.2057 | 0.3666 | 0.4750 | 40.8289 |
| `tower` | 0.519 | 1024 | 32x32 | 1,576,963 | 33.0202 | 0.2217 | 0.4163 | 0.5181 | 39.7048 |
| `tower` | 0.630 | 1024 | 32x32 | 1,576,963 | 32.8734 | 0.2243 | 0.4167 | 0.5189 | 39.6277 |
| `tower` | 0.741 | 1024 | 32x32 | 1,576,963 | 33.3127 | 0.2178 | 0.4027 | 0.5114 | 40.0000 |
| `tower` | 0.889 | 1024 | 32x32 | 1,576,963 | 34.1726 | 0.2001 | 0.3801 | 0.4947 | 40.6337 |
| `tower` | 1.000 | 1024 | 32x32 | 1,576,963 | 34.3217 | 0.2082 | 0.3941 | 0.4983 | 41.1041 |
| `merged` | 1.000 | 4096 | 16x16 | 3,149,827 | 29.5946 | 0.2650 | 0.4834 | 0.5971 | 37.0148 |
| `projected` | 1.000 | 7168 | 16x16 | 4,722,691 | 33.8102 | 0.2203 | 0.4120 | 0.5112 | 40.8779 |

### AI4Industry/MoonViT-V2, normal, capacity-matched

| Stage | Rel. Depth | Width | Grid | Params | mean_deg | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.111 | 512 | 32x32 | 1,314,819 | 30.8304 | 0.2407 | 0.4510 | 0.5684 | 37.9152 |
| `tower` | 0.259 | 512 | 32x32 | 1,314,819 | 27.1252 | 0.2964 | 0.5416 | 0.6519 | 34.4917 |
| `tower` | 0.370 | 512 | 32x32 | 1,314,819 | 26.1924 | 0.3112 | 0.5598 | 0.6674 | 33.5292 |
| `tower` | 0.519 | 512 | 32x32 | 1,314,819 | 25.1291 | 0.3438 | 0.5902 | 0.6919 | 32.7575 |
| `tower` | 0.630 | 512 | 32x32 | 1,314,819 | 25.0813 | 0.3471 | 0.5899 | 0.6907 | 32.7550 |
| `tower` | 0.741 | 512 | 32x32 | 1,314,819 | 26.0634 | 0.3235 | 0.5624 | 0.6709 | 33.5946 |
| `tower` | 0.889 | 512 | 32x32 | 1,314,819 | 28.0066 | 0.2860 | 0.5133 | 0.6278 | 35.3275 |
| `tower` | 1.000 | 512 | 32x32 | 1,314,819 | 30.4182 | 0.2565 | 0.4678 | 0.5816 | 37.8969 |
| `merged` | 1.000 | 512 | 16x16 | 1,314,819 | 28.7317 | 0.2875 | 0.5141 | 0.6219 | 36.5381 |
| `projected` | 1.000 | 512 | 16x16 | 1,314,819 | 28.9913 | 0.2795 | 0.5060 | 0.6161 | 36.6983 |

### facebook/dinov2-large, normal, unmatched

| Stage | Rel. Depth | Width | Grid | Params | mean_deg | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.125 | 1024 | 32x32 | 1,576,963 | 33.3550 | 0.1834 | 0.3821 | 0.5079 | 39.6404 |
| `tower` | 0.250 | 1024 | 32x32 | 1,576,963 | 31.1905 | 0.2184 | 0.4262 | 0.5523 | 37.8009 |
| `tower` | 0.375 | 1024 | 32x32 | 1,576,963 | 28.3375 | 0.2614 | 0.4958 | 0.6180 | 35.2632 |
| `tower` | 0.500 | 1024 | 32x32 | 1,576,963 | 23.9288 | 0.3610 | 0.6061 | 0.7080 | 31.1962 |
| `tower` | 0.625 | 1024 | 32x32 | 1,576,963 | 20.8079 | 0.4480 | 0.6835 | 0.7664 | 28.3762 |
| `tower` | 0.750 | 1024 | 32x32 | 1,576,963 | 19.6204 | 0.4986 | 0.7179 | 0.7857 | 27.4705 |
| `tower` | 0.875 | 1024 | 32x32 | 1,576,963 | 21.3644 | 0.4427 | 0.6806 | 0.7589 | 29.1478 |
| `tower` | 1.000 | 1024 | 32x32 | 1,576,963 | 22.4811 | 0.4103 | 0.6514 | 0.7375 | 30.2730 |

### facebook/dinov2-large, normal, capacity-matched

| Stage | Rel. Depth | Width | Grid | Params | mean_deg | d1 | d2 | d3 | rmse |
|---|---|---|---|---|---|---|---|---|---|
| `tower` | 0.125 | 512 | 32x32 | 1,314,819 | 30.7975 | 0.2316 | 0.4472 | 0.5703 | 37.7121 |
| `tower` | 0.250 | 512 | 32x32 | 1,314,819 | 28.2250 | 0.2675 | 0.5058 | 0.6260 | 35.3126 |
| `tower` | 0.375 | 512 | 32x32 | 1,314,819 | 25.7934 | 0.3218 | 0.5667 | 0.6730 | 33.1464 |
| `tower` | 0.500 | 512 | 32x32 | 1,314,819 | 22.7168 | 0.3976 | 0.6382 | 0.7316 | 30.2719 |
| `tower` | 0.625 | 512 | 32x32 | 1,314,819 | 20.1392 | 0.4798 | 0.6971 | 0.7742 | 28.0153 |
| `tower` | 0.750 | 512 | 32x32 | 1,314,819 | 18.6189 | 0.5444 | 0.7316 | 0.7946 | 26.7982 |
| `tower` | 0.875 | 512 | 32x32 | 1,314,819 | 20.0570 | 0.4981 | 0.7054 | 0.7753 | 28.1160 |
| `tower` | 1.000 | 512 | 32x32 | 1,314,819 | 23.1905 | 0.3964 | 0.6369 | 0.7266 | 31.0744 |

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
| `tower` | 0.111 | 0.3433 | 0.3984 | 0.9890 | 0.7771 |
| `tower` | 0.259 | 0.4457 | 0.4801 | 0.9890 | 0.7771 |
| `tower` | 0.370 | 0.4600 | 0.5239 | 0.9890 | 0.7771 |
| `tower` | 0.519 | 0.5576 | 0.5591 | 0.9890 | 0.7771 |
| `tower` | 0.630 | 0.6026 | 0.6113 | 0.9890 | 0.7771 |
| `tower` | 0.741 | 0.5772 | 0.6031 | 0.9890 | 0.7771 |
| `tower` | 0.889 | 0.5570 | 0.5585 | 0.9890 | 0.7771 |
| `tower` | 1.000 | 0.4890 | 0.5060 | 0.9890 | 0.7771 |
| `merged` | 1.000 | 0.5602 | 0.5505 | 0.9890 | 0.7771 |
| `projected` | 1.000 | 0.5670 | 0.5472 | 0.9890 | 0.7771 |

### AI4Industry/MoonViT-V2, normal, capacity-matched

| Stage | Rel. Depth | indoors mean_deg | outdoor mean_deg | indoors coverage | outdoor coverage |
|---|---|---|---|---|---|
| `tower` | 0.111 | 32.1323 | 29.7929 | 0.9803 | 0.6116 |
| `tower` | 0.259 | 26.6973 | 27.4662 | 0.9803 | 0.6116 |
| `tower` | 0.370 | 25.8576 | 26.4593 | 0.9803 | 0.6116 |
| `tower` | 0.519 | 24.5965 | 25.5534 | 0.9803 | 0.6116 |
| `tower` | 0.630 | 24.3163 | 25.6909 | 0.9803 | 0.6116 |
| `tower` | 0.741 | 25.2372 | 26.7218 | 0.9803 | 0.6116 |
| `tower` | 0.889 | 27.9776 | 28.0297 | 0.9803 | 0.6116 |
| `tower` | 1.000 | 31.1542 | 29.8316 | 0.9803 | 0.6116 |
| `merged` | 1.000 | 28.6364 | 28.8077 | 0.9803 | 0.6116 |
| `projected` | 1.000 | 29.0160 | 28.9715 | 0.9803 | 0.6116 |

### facebook/dinov2-large, depth, capacity-matched

| Stage | Rel. Depth | indoors d1 | outdoor d1 | indoors coverage | outdoor coverage |
|---|---|---|---|---|---|
| `tower` | 0.125 | 0.4290 | 0.4208 | 0.9890 | 0.7771 |
| `tower` | 0.250 | 0.3996 | 0.4674 | 0.9890 | 0.7771 |
| `tower` | 0.375 | 0.4149 | 0.4869 | 0.9890 | 0.7771 |
| `tower` | 0.500 | 0.5286 | 0.5374 | 0.9890 | 0.7771 |
| `tower` | 0.625 | 0.6060 | 0.6154 | 0.9890 | 0.7771 |
| `tower` | 0.750 | 0.6344 | 0.6734 | 0.9890 | 0.7771 |
| `tower` | 0.875 | 0.6695 | 0.6829 | 0.9890 | 0.7771 |
| `tower` | 1.000 | 0.6805 | 0.6692 | 0.9890 | 0.7771 |

### facebook/dinov2-large, normal, capacity-matched

| Stage | Rel. Depth | indoors mean_deg | outdoor mean_deg | indoors coverage | outdoor coverage |
|---|---|---|---|---|---|
| `tower` | 0.125 | 31.5078 | 30.2315 | 0.9803 | 0.6116 |
| `tower` | 0.250 | 28.3129 | 28.1550 | 0.9803 | 0.6116 |
| `tower` | 0.375 | 25.1891 | 26.2750 | 0.9803 | 0.6116 |
| `tower` | 0.500 | 21.3158 | 23.8333 | 0.9803 | 0.6116 |
| `tower` | 0.625 | 17.8036 | 22.0004 | 0.9803 | 0.6116 |
| `tower` | 0.750 | 15.2565 | 21.2982 | 0.9803 | 0.6116 |
| `tower` | 0.875 | 16.7220 | 22.7145 | 0.9803 | 0.6116 |
| `tower` | 1.000 | 20.5467 | 25.2973 | 0.9803 | 0.6116 |

## What this run does not establish

About 540 training images for a head of 1.4 to 4.9 million parameters is thin, so the
absolute numbers are noisy and should not be compared against published Probe3D results.
Nobody has published Probe3D numbers on DIODE in any case (ADR-0011). Every question above
is a comparison between cells that saw identical data and an identical head, and those
comparisons hold; the levels do not.

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
