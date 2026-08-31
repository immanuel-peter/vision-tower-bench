# Geometry paired bootstrap, August 31 2026

## Outcome

Three predeclared matched-arm comparisons were rerun with deterministic PCA and paired over
the same DIODE test images. All three overall 95% intervals exclude zero. Muse Glimmer's
strongest Projector Stage step resolves, and DINOv2's lead over the next-best Tower resolves
on both measured geometry tasks.

| comparison | first | second | first advantage | paired 95% interval |
|---|---:|---:|---:|---:|
| Muse Glimmer `projected` vs `merged`, depth `d1` | 0.58920 | 0.52298 | +0.06622 | [+0.04992, +0.08338] |
| DINOv2 vs Qwen3.5, depth `d1` | 0.69881 | 0.66965 | +0.02916 | [+0.00989, +0.04835] |
| DINOv2 vs SigLIP2, normal `mean_deg` | 18.8204 | 23.8575 | +5.0370 degrees lower | [+4.3038, +5.8164] |

For normals, positive advantage is SigLIP2's error minus DINOv2's because lower
`mean_deg` is better.

## Protocol

DIODE validation contributes 771 images: 541 train, 115 validation and 115 test under the
existing fixed split. The test set contains 51 indoor and 64 outdoor images. Each selected
cell uses full patch tokens, the matched 512-wide arm, the Probe3D multiscale head, the full
six-rate grid `[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]`, 10 epochs and three seeds.

The runner retains one headline metric per seed and test image. It averages across seeds per
image, subtracts paired measurements, and draws 10,000 bootstrap samples of the shared image
axis. The percentile interval is 95% with bootstrap seed zero. The interval is conditional
on the Stage and Relative Depth cells selected by the original matrix; it measures test-image
uncertainty for the headline comparisons, not uncertainty over selecting the best layer.

Scene-specific intervals are secondary checks:

| comparison | indoor 95% interval | outdoor 95% interval |
|---|---:|---:|
| Muse Glimmer `projected` vs `merged`, depth | [+0.02659, +0.08523] | [+0.05781, +0.09481] |
| DINOv2 vs Qwen3.5, depth | [-0.00696, +0.07074] | [+0.01461, +0.04041] |
| DINOv2 vs SigLIP2, normals | [+6.2910, +8.9279] | [+2.6079, +3.4389] |

Only the indoor DINOv2 depth interval crosses zero. The predeclared overall interval remains
positive.

## Reproducibility correction

`torch.svd_lowrank` is randomized. The original matched geometry matrix called the reducer
before setting a per-cell seed, making its PCA basis dependent on lane history. The original
bases and trained heads were not saved. The bootstrap runner and `vtb.geometry_run` now use
the deterministic, train-split-only reducer introduced for the semantic correction.

All six selected cells keep their original learning rates. Corrected point estimates move
slightly:

| cell | committed matrix | deterministic rerun |
|---|---:|---:|
| Muse Glimmer depth `projected` L50 | 0.5901 | 0.5892 |
| Muse Glimmer depth `merged` L50 | 0.5210 | 0.5230 |
| DINOv2 depth `tower` L21 | 0.7015 | 0.6988 |
| Qwen3.5 depth `tower` L17 | 0.6736 | 0.6697 |
| DINOv2 normal `tower` L18 | 18.8508 | 18.8204 |
| SigLIP2 normal `tower` L10 | 23.9039 | 23.8575 |

The correction changes none of the three conclusions.

## Box and measured cost

Brev, 4x NVIDIA L40S 48 GB, 46 CPU cores, $4.22/hr, with 1.9 TB free before this run. The
released-Tower parity suite passed first: 67 tests in 5m14s. Every long command ran in tmux
with `UV_NO_SYNC=1`; extraction lanes used ten workers and explicit two-thread Torch caps.

| phase | UTC | wall |
|---|---|---:|
| DIODE validation download, extraction and prep | 22:13:54-22:18:41 | 4m47s |
| four full-token Tower extractions | 22:19:51-22:21:01 | 1m10s |
| three paired bootstrap comparisons | 22:22:21-22:36:50 | 14m29s |

Extraction times were DINOv2 31s, SigLIP2 43s, Qwen3.5 31s and Muse Glimmer 70s. The four
caches total 66.03 decimal GB, reported as 62 GiB by `du`; every slice contains the same 771
image ids. The DIODE directory occupies 28 GiB including both raw archives and prepared
targets.

Comparison times were 301s for the two Muse Stage cells, 869s for DINOv2 against Qwen3.5,
and 836s for DINOv2 against SigLIP2. The measured data-to-result window was 22m56s, about
$1.61 of box wall time. This excludes implementation, tests, documentation and idle time;
the provider's wall-clock bill remains authoritative.

The 30-to-60-minute end-to-end estimate held. Extraction matched the previous identical-box
measurements almost exactly. The predicted 9-to-12-minute probing wall was optimistic: the
slowest comparison took 14m29s while three randomized-SVD reductions and dense heads shared
the 46 cores.

## Evidence

The three JSON files in `results/bootstrap/` contain the protocol, full validation curves,
selected rates, three seed metrics, all 115 paired per-image measurements, and overall plus
scene-specific bootstrap metadata. All three tmux jobs exited zero; their logs contain no
traceback, CUDA error, runtime error or killed process. A final post-result suite again
passed all 67 tests in 69.83 seconds.
