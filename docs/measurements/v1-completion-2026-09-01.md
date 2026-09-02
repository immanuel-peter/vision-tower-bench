# v1 completion run, September 2 2026

## Continuation checkpoint

The continuation began at 06:28:38 UTC on the same Brev box. The box was still warm:
`/ephemeral/data` held 65 GiB, including ImageNet-100 validation and all three
correspondence datasets, and `/ephemeral/cache` held 551 MiB. The repository was clean at
`96da245`, equal to `origin/main`. No dataset or model cache was rebuilt before the required
test gate.

The managed shell did not inherit `HF_HOME` and could not write its default `uv` cache.
Two gate attempts stopped before comparing weights, first during collection and then on
offline-only metadata requests. With `HF_HOME=/ephemeral/hf`, network metadata access, and
the warm weight cache, the required gate passed all 80 tests in 126.90 seconds. No
`VTB_SKIP_WEIGHTS` setting was used.

The ScanNet diagnosis found a coordinate-frame bug within the 30-minute timebox. Archive
RGB frames are 1296 by 968 pixels, while the depth maps and supplied intrinsics use a 640
by 480 frame. The loader transformed the 640 by 480 intrinsics with the RGB resize factor,
putting the principal point far from its true location. Aligning RGB to the depth frame
before the declared square crop raised DINOv2 recall at 10 pixels from 0.00748 on the old
full run to 0.13220 on a corrected 20-pair smoke test. The full six-Tower ScanNet column
was therefore scheduled for replacement rather than exclusion.

The Kimi K2.6 merge audit found no raster-order bug. Its published model reshapes the
raster sequence into 2 by 2 blocks, permutes the blocks into raster order, and returns one
item per block. The adapter preserves that order. The existing weight-backed test also
checks that `merged` equals the correct 2 by 2 regrouping bit for bit and differs from a
flat four-token reshape. The correspondence collapse remains a measured property of this
lossless but phase-sensitive representation, not evidence of scrambled tokens.

## Continuation workstream A: correspondence complete

The six missing Qwen3.8 and Muse Glimmer jobs ran from 06:36:56 through 06:51:00 UTC.
The corrected six-Tower ScanNet replacement overlapped them on the two freed GPUs and ran
from 06:42:32 through 06:55:24. This 18m28s measurement window cost $1.30 at $4.22 per
hour. Both matrix alert logs were empty. A payload check found all 18 dataset jobs, all 42
expected Stage cells, 1,500 ScanNet pairs, 555 NAVI pairs, and 3,600 SPair pairs.

| Tower | column | `projected` minus `tower` | paired 95% interval |
| --- | --- | ---: | ---: |
| MoonViT-V2 | corrected ScanNet | +0.021341 | [+0.020162, +0.022543] |
| MoonViT-V2 | NAVI | +0.059020 | [+0.054193, +0.063709] |
| MoonViT-V2 | SPair | -0.014355 | [-0.021642, -0.007021] |
| Kimi K2.6 | corrected ScanNet | +0.007535 | [+0.006092, +0.008967] |
| Kimi K2.6 | NAVI | +0.056532 | [+0.052110, +0.060899] |
| Kimi K2.6 | SPair | +0.114964 | [+0.106328, +0.123475] |
| Qwen3.8 | corrected ScanNet | -0.019414 | [-0.020802, -0.018019] |
| Qwen3.8 | NAVI | +0.018836 | [+0.014590, +0.023038] |
| Qwen3.8 | SPair | +0.021649 | [+0.011830, +0.031645] |
| Muse Glimmer | corrected ScanNet | +0.031772 | [+0.029930, +0.033612] |
| Muse Glimmer | NAVI | +0.105779 | [+0.099457, +0.112103] |
| Muse Glimmer | SPair | +0.132510 | [+0.123934, +0.141108] |

NAVI carries the spatial conclusion. All four Projectors improve multiview 3D recall and
all four intervals exclude zero. The corrected ScanNet adaptation now has usable absolute
scores, but Qwen3.8 loses there while the other three improve. SPair remains a separate
semantic matching column. MoonViT-V2 loses semantic PCK while improving NAVI, so one
Connector loses semantic matching ability while gaining geometric correspondence. The
other three Projectors improve both NAVI and SPair.

Workstream A passed 81 tests in 126.49 seconds, including all published-weight checks, and
was committed and pushed as `02ebbcc` at about 07:06 UTC. Continuation wall time through
that durable checkpoint was 37m27s, or $2.63.

## Continuation workstream B: label-budget extraction

The six 13,000-image pooled extractions started at 07:06 UTC in a fresh
`/ephemeral/label-budget-features` root. Each lane used two OpenMP and MKL threads and the
measured batch sizes. Qwen3.8 and Muse Glimmer finished first; Kimi K2.6, DINOv2, and
MoonViT-V2 followed. SigLIP2 remained in progress at the 07:28 UTC checkpoint. These
contended extraction rates are cache-production diagnostics only. The required latency
comparison will run over 768 images per Tower with every other process stopped.

The official 1.917 GiB KITTI selected-depth archive downloaded without credentials during
this phase. Its public validation selection contains 1,000 paired RGB, sparse raw LiDAR,
and accumulated ground-truth depth maps. No KITTI experiment had started at this
checkpoint.

The interactive session was interrupted after the extraction launch and resumed at
16:58:52 UTC. By then all six caches were complete: 13,000 images and eight or ten saved
slices per Tower, occupying 40 GiB in total. No benchmark process remained active. The
continuation had reached 10h30m14s and about $44.31, crossing the required $30 report
threshold with about $20.69 left under the $65 continuation ceiling. Most of that interval
was an unintended idle gap after extraction completed. To protect the KITTI transfer
column, ADR-0003's cut order is now active: the Perturbation Study will measure one factor
rather than three.

## Continuation workstream B: label budgets complete

The final cache completed at 07:36:15 UTC. Feature extraction therefore occupied about
30m15s. The resumed session measured all six Towers sequentially over 768 images on an
otherwise idle GPU from 16:59:28 through 17:02:35, then ran the 48 probe cells from about
17:03 through 17:18:08. Active B measurement totalled about 48 minutes, or $3.40. The
interruption left the completed box idle from 07:36 to 16:58, about 9h22m and $39.55.

All 48 expected JSON payloads exist, each contains one deepest-Tower cell, and `alerts.log`
is empty. Training counts are 100, 455, 1,820, and 9,100 for requested budgets one, five,
20, and 100 percent. Attention changes leader from SigLIP2 at one and five percent to Muse
Glimmer at 20 and 100 percent. Mean keeps SigLIP2 first, although DINOv2 and Muse Glimmer
swap second and third after one percent. The Tower ranking is not constant across label
budgets.

| Tower | tokens at 448 square | idle images/s | attention 1% | attention 100% | mean 1% | mean 100% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DINOv2 | 1,024 | 44.205 | 0.3689 | 0.9079 | 0.4497 | 0.8933 |
| SigLIP2 | 1,024 | 45.422 | 0.6559 | 0.9168 | 0.6824 | 0.9128 |
| MoonViT-V2 | 1,024 | 29.444 | 0.2191 | 0.8369 | 0.2287 | 0.8397 |
| Kimi K2.6 | 1,024 | 35.127 | 0.3448 | 0.8862 | 0.3590 | 0.8742 |
| Qwen3.8 | 784 | 64.918 | 0.2451 | 0.8791 | 0.2891 | 0.8679 |
| Muse Glimmer | 1,024 | 23.432 | 0.4330 | 0.9212 | 0.4361 | 0.9089 |

Muse Glimmer, the largest Tower, does not win at low label budgets and is the slowest in
the roster. At full labels its 0.0044 attention edge over SigLIP2 costs nearly half the
throughput with the same number of exposed tokens. Qwen3.8 is fastest and exposes 240 fewer
tokens but trails the accuracy leaders. Hypothesis 3 is supported as a tradeoff and remains
in `PLAN.md`; it is not evidence that one Tower dominates all efficiency axes.

## Outcome

The correspondence disagreement stop condition fired after 12 of 18 full jobs. Both
completed Projectors improve geometric correspondence on ScanNet and NAVI, but semantic
correspondence splits. Kimi K2.6 improves on SPair by +0.11496 PCK, paired 95% interval
[+0.10633, +0.12348]. MoonViT-V2 degrades by -0.01436, interval
[-0.02164, -0.00702]. The negative interval is resolved, so the matrix stopped and
workstreams B through D did not run.

This is not a published-weight failure. The first parity attempt found three missing local
release bundles before any parity assertion ran. After restoring the public bundles, the
required gate passed: 74 tests in 176.61 seconds with no `VTB_SKIP_WEIGHTS`.

## Box and cost

The box had four NVIDIA L40S GPUs with 46,068 MiB each, 46 logical CPU cores, and 2.3 TB
free on `/ephemeral`. Rate was $4.22 per hour. Through the approved push, total observed
wall time was 3.860 hours and total spend was $16.29.

| phase | start UTC | end UTC | hours | cost | cumulative |
| --- | --- | --- | ---: | ---: | ---: |
| setup, required reading, correspondence build, data fetch, first gate | 01:22:29 | 01:41:13 | 0.312 | $1.32 | $1.32 |
| first stop checkpoint and OS shutdown attempt | 01:41:13 | 01:45:00 | 0.063 | $0.27 | $1.59 |
| idle after OS shutdown failed to stop Brev billing | 01:45:00 | 03:19:23 | 1.573 | $6.64 | $8.23 |
| restore release bundles and pass parity | 03:19:23 | 03:23:52 | 0.075 | $0.32 | $8.55 |
| three 20-pair smoke tests | 03:23:52 | 03:29:03 | 0.086 | $0.36 | $8.91 |
| full correspondence matrix, halted by stop condition | 03:29:03 | 03:46:09 | 0.285 | $1.20 | $10.11 |
| stop report, final tests, commits, and termination discovery | 03:46:09 | 03:55:00 | 0.147 | $0.62 | $10.73 |
| idle awaiting explicit push approval, then push | 03:55:00 | 05:14:04 | 1.318 | $5.56 | $16.29 |

The $6.64 idle row is real spend. `sudo shutdown -h now` returned success but did not
terminate the Brev rental. Provider-level termination is required; guest shutdown is not a
billing control on this box.

The two workstream checkpoints are `a4bae13` and `336ebb0`; the first ledger checkpoint is
`739fe39`. The execution policy initially rejected pushing to the unverified GitHub
`origin/main` without a new explicit user approval. After approval, all three commits were
pushed successfully. Waiting for that approval added 1h19m04s and $5.56. The guest contains
no authenticated Brev CLI credentials, so the provider stop must be issued from the user's
authenticated Brev client with `brev stop brev-lqrojopw9`.

## Setup and data

All three correspondence datasets downloaded without credentials.

| item | measured size or count |
| --- | ---: |
| ImageNet-100 validation | 13,000 images, 1.3 GiB |
| COCO val2017 | 5,000 images, 788 MiB |
| NAVI | 555 evaluation pairs, 61 GiB archive plus extraction |
| ScanNet subset | 1,500 pairs, 2.1 GiB archive plus extraction |
| SPair-71k | 3,600 selected pairs, 723 MiB archive plus extraction |
| Qwen local release bundle | 879 MiB |
| Muse Glimmer local release bundle | 3.6 GiB |
| Kimi K2.6 local release bundle | 899 MiB |

The three release bundles total about 5.4 GiB, not the 12 GB listed in the earlier
measurement note. ImageNet-100 validation export took about nine minutes because the
dataset loader fetched all configured shards before materialising the requested split.

## Correspondence protocol and measured time

ADR-0020 records the scope. ScanNet and NAVI are geometric correspondence; SPair is
semantic correspondence and cannot establish 3D consistency by itself. All three score
frozen Stage features directly. The bench adaptation uses full patch tokens at 448 square,
a shared 64 by 64 matching grid, and 1,000 ratio-ranked matches for the geometric columns.

The required DINOv2 smoke tests completed on 20 pairs per dataset. Scoring time, excluding
adapter construction, was 2.5 seconds on ScanNet, 11.6 seconds on NAVI, and 1.4 seconds on
SPair. SPair exposed one implementation bug: bfloat16 Stage features produced a bfloat16
sampling grid against a float input. Casting the coordinate grid to float fixed it and a
regression test covers the path.

The valid partial matrix completed four Towers on every dataset. Times below come from each
result payload and exclude adapter construction.

| Tower | ScanNet | NAVI | SPair | total |
| --- | ---: | ---: | ---: | ---: |
| DINOv2 | 121.2 s | 367.2 s | 135.2 s | 10.4 min |
| SigLIP2 | 123.6 s | 389.4 s | 149.6 s | 11.0 min |
| MoonViT-V2 | 260.2 s | 375.9 s | 407.7 s | 17.4 min |
| Kimi K2.6 | 238.9 s | 369.0 s | 324.5 s | 15.5 min |

Four lanes completed those 12 jobs in 17.1 minutes of matrix wall time. `alerts.log`
remained empty. Qwen3.5 and Muse Glimmer jobs were in flight when the stop condition fired;
their partial files were discarded.

## Correspondence result

| Tower | Stage | ScanNet recall@10px | NAVI recall@2cm | SPair macro PCK@0.1 |
| --- | --- | ---: | ---: | ---: |
| DINOv2 | `tower` | 0.00748 | 0.53891 | 0.55475 |
| SigLIP2 | `tower` | 0.00449 | 0.40133 | 0.39818 |
| MoonViT-V2 | `tower` | 0.00396 | 0.33342 | 0.27600 |
| MoonViT-V2 | `merged` | 0.00506 | 0.38818 | 0.25745 |
| MoonViT-V2 | `projected` | 0.00510 | 0.39244 | 0.26140 |
| Kimi K2.6 | `tower` | 0.00483 | 0.36458 | 0.17821 |
| Kimi K2.6 | `merged` | 0.00333 | 0.23138 | 0.06944 |
| Kimi K2.6 | `projected` | 0.00535 | 0.42111 | 0.29184 |

| Tower | column | `projected` minus `tower` | paired 95% interval |
| --- | --- | ---: | ---: |
| MoonViT-V2 | ScanNet | +0.001139 | [+0.000889, +0.001398] |
| MoonViT-V2 | NAVI | +0.059020 | [+0.054193, +0.063709] |
| MoonViT-V2 | SPair | **-0.014355** | **[-0.021642, -0.007021]** |
| Kimi K2.6 | ScanNet | +0.000524 | [+0.000278, +0.000769] |
| Kimi K2.6 | NAVI | +0.056532 | [+0.052110, +0.060899] |
| Kimi K2.6 | SPair | +0.114964 | [+0.106328, +0.123475] |

NAVI supplies the clean geometric conclusion: absolute scores are substantial and every
viewpoint-bin interval favours `projected`. ScanNet points the same way but is floor-limited
below one percent absolute recall under the 64 by 64 square-crop adaptation. On
MoonViT-V2 SPair, most loss occurs from `tower` to `merged`; the Projector recovers a small
part but leaves `projected` significantly below `tower`.

## Workstreams stopped before measurement

The label-budget selector and 48-cell matrix driver were implemented while correspondence
used the GPUs. They preserve validation and test indices and retain at least one training
example per class at a one-percent budget. No label-budget cache, probe, latency, or token
count result was produced.

KITTI Transfer Probe and Perturbation Study work did not start. No dataset credentials
blocked them; the earlier correspondence disagreement did.

## What the brief's estimates got wrong

- The handoff said setup had run, but the repository had no usable synced `.venv`. One
  explicit `uv sync --frozen` was required before `UV_NO_SYNC=1` could be used safely.
- The brief omitted the release-bundle restore command even though the parity tests require
  those repo-local files. Their measured total was 5.4 GiB, not 12 GB.
- ImageNet-100 validation export took about nine minutes, not the prior five-minute
  measurement, because it downloaded unrelated configured shards first.
- Correspondence extraction and scoring did fit the claimed minutes-per-Tower scale. The
  four completed triplets took 10.4 to 17.4 scorer minutes each and 17.1 matrix minutes on
  four lanes.
- ScanNet did not provide a healthy absolute signal under the declared 64 by 64 adaptation;
  it landed below one percent recall for every completed Stage. The brief anticipated a
  cheap column, not a floor-limited one.
- Guest shutdown was not equivalent to terminating the rented box. That mistaken
  assumption added 1.573 billed hours and $6.64, the largest cost in the run.
- The remote-approval gate added another 1.318 billed hours and $5.56 before the push could
  proceed.

No timing or cache-size estimate for workstreams B through D can be checked against this
run because the required stop condition prevented those experiments.
