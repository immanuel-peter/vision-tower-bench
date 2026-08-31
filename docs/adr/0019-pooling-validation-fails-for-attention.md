# The pooling validation failed for the attention readout

ADR-0005 requires a pooling validation before the pooled semantic cache can be treated as a
stand-in for full patch tokens: run semantic probes both ways on at least two Towers and
accept pooling only if model rankings and Relative Depth curves agree. ADR-0003 forbids
cutting this check. This ADR records the first run of it, on August 31 2026, and its
outcome is a split: the check passes for the mean readout and **fails for the attention
readout**, which is the headline one.

## What ran

Two Towers, SigLIP2 and Muse Glimmer - the two whose readout wins carry the semantic
hypothesis, and therefore exactly where pooling could manufacture a result. Both grids were
extracted over the same 1,500 ImageNet-100 validation images (the first 1,500 in sorted
order; pooled and full caches verified to agree on image ids per model), one at the 4x4
pooled grid the semantic pillar uses and one at full patch tokens. 72 cells, both readouts,
both capacity arms, the eleven-point grid, three seeds, on the same box as the August 31
semantic re-run.

Caches: SigLIP2 full 27 GB (ADR-0005 predicted 28.3), Muse Glimmer full 31 GB (ADR
predicted 47.6; the estimate assumed the roster's 13,000-image Stage mix). Cost 7.7
lane-hours, 2.2 h wall on four lanes. Full-token cells measured 473 s/cell raw and 206
s/cell matched against the brief's derived 360 s/cell - 1.3x and 0.6x, no phase far enough
off to stop for.

## The failure: full-token attention heads do not train at mid Relative Depth

At 1,024 tokens the attention pool needs to learn where to look, and on 1,050 training
images it mostly cannot. The validation curves of the collapsed cells fall off a cliff:

| cell | val at 1e-4 | val at 3e-4 | val at 1e-3 | test at selected rate |
|---|---|---|---|---|
| siglip2 attention raw `tower` d0.370 | 0.22 | 0.02 | 0.02 | 0.2326 ± 0.0147 |
| siglip2 attention matched `tower` d0.630 | 0.51 | 0.66 | 0.03 | 0.3733 ± 0.2940 |
| muse_glimmer attention matched `tower` d0.760 | 0.31 | 0.67 | 0.72 | 0.2770 ± 0.3262 |

Above 3e-4 every rate diverges; even the selected rate is unstable across seeds at the
mid-Tower cells (seed spreads up to 0.39). Where the head does train, it matches or beats
pooled. The deepest cells - the ones semantic rankings are read from - are the stable ones.

So this is an optimization failure of the one-query attention pool over 1,024 tokens on
1,050 training images, not evidence that pooling discards information. But ADR-0005's
acceptance rule does not ask why the curves disagree, and they disagree.

## The check, read by its own rule

Rankings (best Tower cell), pooled against full:

| readout | arm | pooled | full | agree |
|---|---|---|---|---|
| attention | matched | muse_glimmer 0.8948 > siglip2 0.8830 | muse_glimmer 0.9156 > siglip2 0.8563 | yes |
| attention | raw | siglip2 0.8844 > muse_glimmer 0.8800 | muse_glimmer 0.9111 > siglip2 0.8993 | **no** |
| mean | raw | siglip2 0.8756 > muse_glimmer 0.8193 | same values | yes |
| mean | matched | siglip2 0.8859 > muse_glimmer 0.8607 | siglip2 0.8741 > muse_glimmer 0.8578 | yes |

The raw attention flip is one test image: its pooled margin is 0.0044 and the test split is
225 images. It is a ranking disagreement only in the strictest reading, but the ADR's rule
is the ADR's rule.

Curve shapes. The mean readout agrees everywhere - on the raw arm pooled and full are
identical to four decimals on most cells, as they must be: the mean of a 4x4 pooled grid
equals the mean of all 1,024 tokens up to float rounding, and the matched mean curves agree
within 0.012 at every point. The attention curves do not agree: six of the sixteen
full-token attention cells collapse to near-random mid-Tower (siglip2 raw d0.370-d0.630 read
0.23 to 0.56 against pooled 0.50-0.79), and one of the exceptions is a cell whose validation
curve selected 1e-3 at 0.716 and whose three seed runs then read 0.277 ± 0.3262. The deepest
attention cells agree in shape with pooled.

## Consequences

1. The mean-pooling semantic results are validated: rankings and Relative Depth curves agree
   pooled against full on both Towers, and the raw cells agree to four decimals. Cross-model
   mean-readout comparisons stand.
2. The attention readout's cross-model comparison is carried by pooled features whose
   control did not validate. The deepest-point rankings agree in the matched arm and the
   absolute levels reproduce on both grids where the head trains, but the check ADR-0003
   forbids cutting has not passed for the headline readout. That is the honest state of the
   semantic pillar until the attention side of this control passes.
3. The failure is in the full-token head, not obviously in pooling: where the full-token
   head trains stably it agrees with pooled, and the mean readout - which reads the same
   information by construction - validates perfectly. The likely confound is optimization:
   one learned query over 1,024 tokens, 1,050 training examples, 20 epochs. Any fix costs
   either images (full tokens over 13,000 images are the 2.18 TB ADR-0005 rules out) or
   protocol (more epochs, or a token-count-matched head), and either way it is a costed
   decision for a later run, not a re-run of this one.

## What this does not change

- The mean readout results, and every hypothesis that rests on them.
- The geometry pillar, which always used full tokens.
- Within-model comparisons on the pooled cache (Stage orderings, Relative Depth curves):
  every cell of a comparison saw the same cache, so pooling cannot manufacture those
  differences. What pooling could in principle manufacture is the cross-model attention
  comparison - which is exactly what failed to validate.

The semantic pillar's attention readout should be described as "rankings on pooled features;
the pooling control passed for mean pooling and is inconclusive for attention" until a
passing validation exists.
