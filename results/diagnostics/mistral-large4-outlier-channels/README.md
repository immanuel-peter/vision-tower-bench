# Mistral Large 4 Tower L30 outlier channels, November 8 2026

The Mistral Large 4 Tower has no final norm, so Tower L30 is the raw residual
stream. Three of its channels are orders of magnitude larger than the rest,
and the matched readouts fail there the way MiMo 2.6 fails at Tower L28
([MiMo diagnostic](../mimo26-tower-outlier-channel/README.md)). No published
file or benchmark arm changed; the standardized runs below are diagnostics.

## Channels

[channel_statistics.json](channel_statistics.json), on the 13,000-image
recognition cache (448 px, 4x4 pooled):

| Slice | Largest channels (RMS) | Median channel RMS | Largest / median | First PCA component, training variance |
|---|---|---|---|---|
| Tower L26 | 2021 (40.0), 1237 (29.3), 171 (21.5) | 1.32 | 30 | 8.3% |
| Tower L30 | 484 (39,289), 1657 (31,026), 70 (25,742) | 12.3 | 3,200 | 99.96% |
| merged | 1727 (1.0), 753 (0.8) | 0.095 | 10 | 18.9% |
| projected | 5010 (3.8), 2356 (1.7) | 0.12 | 31 | 12.1% |

`probe.Reducer` centers but does not scale, so at Tower L30 the matched
reduction's first component is these channels. The Projector's RMSNorm
removes the effect for `merged` and `projected`. The same channels make the
parent's own BF16 forward drift from FP32 at L30
([parity diagnostic](../mistral-large4-parity/README.md)).

## Effects in the published files

- **Perturbation.** Tower L30 clean accuracy is 0.040, chance for 100
  classes; `merged` and `projected` reach 0.840 and 0.830. The Tower rows of
  `perturbation/mistral_large4_perturbation.json` and of
  `bootstrap/perturbation-mistral_large4-projected-minus-tower.json` are a
  readout failure, not a robustness measurement.
- **Label budget.** The attention-matched cell at Tower L30 is 0.017, 0.026
  and 0.028 at 1%, 5% and 20% of labels, and 0.800 at 100%. The mean readout
  reaches 0.102, 0.279, 0.739 and 0.842.
- **Geometry.** Matched depth d1 falls from 0.656 at Tower L26 to 0.186 at
  L30 (raw: 0.663 to 0.399), and matched normal error rises from 26.5 to 41.5
  degrees (raw: 24.8 to 35.5). The DIODE projected-versus-Tower intervals
  compare against these L30 cells, so their large Projector advantages mostly
  measure the Tower readout.
- **Recognition.** With 9,100 labelled images the readout escapes: attention
  matched Tower L30 reaches 0.800, against 0.855 at L26.

## Channel standardization

The [recovered reducer patch](../../../vtb/probe.py) (`StandardizedReducer`,
SHA256 `f5a2bc1d...`) scales channels to unit variance on the training split
before the PCA; `--standardize-channels` enables it.

- [mistral_large4_perturbation_standardized.json](mistral_large4_perturbation_standardized.json):
  Tower L30 clean accuracy rises from 0.040 to 0.832; occlusion levels 1 to 4
  read 0.792, 0.690, 0.491 and 0.263. `merged` and `projected` clean accuracy
  moves by 0.9 and 1.0 points (0.840 to 0.831, 0.830 to 0.820).
- The four attention-matched label-budget cells at Tower L30 with
  standardization are 0.342, 0.713, 0.838 and 0.871 at 1%, 5%, 20% and 100%.
  They sit in [label_budget_standardized.json](../mimo26-tower-outlier-channel/label_budget_standardized.json)
  beside the other Towers, which the default label-budget figure reads.

## Reproduce

`scripts/perturbation_run.py --standardize-channels` on the perturbation
cache, and `vtb.probe_run --match-capacity --standardize-channels --stages
tower --depth-points 30 --label-fraction F` on the recognition cache, with
the label-budget settings of `scripts/label_budget_matrix.sh`.
