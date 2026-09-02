# v1 completion run, September 2 2026

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
free on `/ephemeral`. Rate was $4.22 per hour. The cost through the experiment halt was
$10.11; the final test, commit, push, and provider-termination checkpoint is added to the
ledger at close.

| phase | start UTC | end UTC | hours | cost | cumulative |
| --- | --- | --- | ---: | ---: | ---: |
| setup, required reading, correspondence build, data fetch, first gate | 01:22:29 | 01:41:13 | 0.312 | $1.32 | $1.32 |
| first stop checkpoint and OS shutdown attempt | 01:41:13 | 01:45:00 | 0.063 | $0.27 | $1.59 |
| idle after OS shutdown failed to stop Brev billing | 01:45:00 | 03:19:23 | 1.573 | $6.64 | $8.23 |
| restore release bundles and pass parity | 03:19:23 | 03:23:52 | 0.075 | $0.32 | $8.55 |
| three 20-pair smoke tests | 03:23:52 | 03:29:03 | 0.086 | $0.36 | $8.91 |
| full correspondence matrix, halted by stop condition | 03:29:03 | 03:46:09 | 0.285 | $1.20 | $10.11 |

The $6.64 idle row is real spend. `sudo shutdown -h now` returned success but did not
terminate the Brev rental. Provider-level termination is required; guest shutdown is not a
billing control on this box.

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

No timing or cache-size estimate for workstreams B through D can be checked against this
run because the required stop condition prevented those experiments.
