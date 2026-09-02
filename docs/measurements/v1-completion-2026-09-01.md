# v1 completion run, September 2 2026

## Outcome

The run stopped at the mandatory parity gate before any experiment smoke test or full
matrix. The suite reported 67 passed, 1 skipped, and 6 failed in 367.06 seconds. Every
failure came from `tests/test_release_parity.py`, and every one was caused by a missing
repo-local release bundle:

- `hf/Qwen3.8-27B-Vision/model.safetensors`
- `hf/Muse-Glimmer-Vision/model.safetensors`
- `hf/MoonViT-K2.6/model.safetensors`

The bit-exact and released-versus-parent forward comparisons did not reach their
assertions. This run therefore did not show that the published weights differ. It showed
that the box lacked the 12 GB local bundle prerequisite those tests assume. The brief says
to stop if a parity test fails, so no correspondence, label-budget, Transfer Probe, or
Perturbation Study measurement ran.

## Cost ledger

Box rate was $4.22 per hour on four NVIDIA L40S 48 GB GPUs and 46 logical CPU cores.

| phase | start UTC | end UTC | hours | cost | cumulative |
|---|---|---|---:|---:|---:|
| setup, reading, data fetch, correspondence scope/build, parity gate | 01:22:29 | 01:41:13 | 0.312 | $1.32 | $1.32 |
| stop checkpoint, push, and shutdown preparation | 01:41:13 | 01:45:00 | 0.063 | $0.27 | $1.59 |

The cost ledger starts at the first observable timestamp on the box. Provider startup time
before that timestamp, if any, is not included.

## Correspondence work preserved

The Probe3D paper and released repository confirm three training-free evaluations:

- ScanNet geometric correspondence on 1,500 scene pairs, scored by pixel recall.
- NAVI geometric correspondence on wild-set views of the same object, scored by metric
  and pixel recall.
- SPair-71k semantic keypoint correspondence, scored by PCK at 10 percent of the target
  bounding-box scale.

All three released subsets have public download paths and required no credentials. ADR-0020
records the distinction between the two geometric columns and the SPair-71k semantic
column. It also records the run's protocol adaptation: full patch-token Stage maps are
resized to a shared 64 by 64 matching grid and retain 1,000 ratio-ranked matches.

The checkpoint includes `vtb/correspondence.py`, dataset and matrix runners, a paired
bootstrap runner, a reproducible fetch script, and seven focused tests. Those seven tests
passed inside the full suite. The implementation has not had the required 20-pair smoke
test, has no results, and must not be described as complete.

## Data and setup measurements

| item | measured size or time |
|---|---:|
| ImageNet-100 validation export | 13,000 images, 1.3 GiB |
| COCO val2017 | 5,000 images, 788 MiB |
| NAVI archive | 31.10 GB |
| NAVI archive plus extracted files | 61 GiB |
| ScanNet 1,500-pair archive | 1.10 GB |
| ScanNet archive plus extracted files | 2.1 GiB |
| SPair-71k archive | 226.96 MB |
| SPair-71k archive plus extracted files | 723 MiB |
| full pytest gate | 367.06 seconds |

`/ephemeral` had 2.3 TB free at the start, so disk capacity was not the blocker.

## What the brief's estimates missed

- The handoff said `scripts/brev_setup.sh` had run, but the repository had no usable
  synced `.venv`. The first ImageNet export used the mandatory `UV_NO_SYNC=1`, created an
  empty environment, and failed on the missing `datasets` package. One explicit
  `uv sync --frozen` repaired it.
- Exporting only ImageNet-100 validation still fetched all 30 training shards, both test
  shards, and all four validation shards before materialising the requested split. The
  export took about nine minutes rather than the prior five-minute measurement.
- `scripts/brev_setup.sh` does not restore the three repo-local release bundles. The prior
  measurement document listed "release bundles restored (12 GB)" as a separate step, but
  the completion brief omitted its command. The parity tests still require those files.

No extraction throughput, correspondence cell cost, cache size, label-budget timing,
KITTI timing, or Perturbation Study timing was measured. Reusing prior numbers for this run
would be misleading.

## Resume action

Restore the public bundles from `immanuelpeter/Qwen3.8-27B-Vision`,
`immanuelpeter/Muse-Glimmer-Vision`, and `immanuelpeter/MoonViT-K2.6` into the exact
repo-local `hf/` directories expected by `tests/test_release_parity.py`. Do not set
`VTB_SKIP_WEIGHTS`. Rerun the full gate against the exported ImageNet-100 images and
continue only if the bit-exact and forward comparisons execute and pass. Then run the
three 20-pair DINOv2 correspondence smoke tests before starting the full matrix.
