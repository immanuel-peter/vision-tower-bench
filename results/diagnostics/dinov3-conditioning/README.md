# DINOv3 feature conditioning, October 18 2026

Issue #4 notes the DINOv3 paper's warning that early-layer features can be
ill-conditioned. The bench keeps DINOv2's feature protocol for both controls:
raw residual streams at intermediate depths and the learned final norm only at
layer 24. This check measures how differently the two streams are scaled. It
changes no benchmark arm.

Inputs are the unpooled DIODE validation caches, DINOv3 at 512 px and DINOv2 at
448 px, both 32 by 32 tokens. Twelve images sit at indices
`linspace(0, 770, 12).round()`, `val_000000` through `val_000770` in steps of
70. Each depth pools their 12,288 tokens. For each depth and model,
[natural_conditioning.json](natural_conditioning.json) records:

- token L2 norm mean and quantiles (0, 5, 50, 95, 100%)
- per-channel standard deviation quantiles
- the share of uncentered energy in the channel mean direction
- the first and first-16 centered principal-component energy fractions, computed on 256 evenly spaced tokens
- the same statistics after an unlearned per-token LayerNorm (eps 1e-6), as a diagnostic only

| Layer | DINOv3 token L2 | DINOv2 token L2 | DINOv3 mean-direction energy | DINOv2 mean-direction energy | DINOv3 centered PC1 | DINOv2 centered PC1 |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 23.22 | 1.80 | 0.676 | 0.257 | 0.331 | 0.153 |
| 6 | 29.24 | 1.68 | 0.637 | 0.235 | 0.207 | 0.163 |
| 9 | 42.40 | 1.86 | 0.447 | 0.183 | 0.184 | 0.126 |
| 12 | 64.36 | 2.88 | 0.383 | 0.106 | 0.143 | 0.072 |
| 15 | 103.35 | 5.16 | 0.209 | 0.111 | 0.102 | 0.060 |
| 18 | 187.15 | 10.68 | 0.224 | 0.100 | 0.085 | 0.061 |
| 21 | 462.71 | 27.32 | 0.402 | 0.151 | 0.098 | 0.063 |
| 24 | 8.54 | 45.51 | 0.174 | 0.088 | 0.083 | 0.055 |

DINOv3's intermediate streams are 13 to 23 times larger than DINOv2's, and
they grow to 463 at layer 21 before the final norm brings them back to 8.5. At
layer 3, the channel mean holds 68% of DINOv3's energy against 26% for DINOv2,
and the first centered component holds 33% against 15%. Early DINOv3 tokens
are dominated by a shared direction.

These figures describe twelve images and a 256-token spectrum sample, not the
full benchmark covariance. The matched arm's PCA and the raw arm also differ in
basis and capacity, so these numbers alone do not explain a gap between arms.
Any early-layer normalization would be a separate, labeled experiment.
