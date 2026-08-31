# Pooling attention diagnostic, August 31 2026

## Outcome

The diagnostic completed 24 cells and an independent six-cell baseline repeat with no
alerts. More images alone did not fix the full-token attention head. More epochs helped but
did not meet the stability target across all three named cells, even when combined with
5,000 images. Job 3 was not launched. ADR-0019 remains a failed attention validation and a
passed mean validation.

## Box

Brev, 4x NVIDIA L40S 48 GB, 46 CPU cores, 2.3 TB free on `/ephemeral`, $4.22/hr. All
extraction lanes used ten workers and explicit `OMP_NUM_THREADS=2` and `MKL_NUM_THREADS=2`.
All probing ran with `UV_NO_SYNC=1`. The release-parity suite passed before extraction.

## Timeline (UTC)

| phase | start | end | wall |
|---|---|---|---:|
| four 5,000-image extractions | 08:47:17 | 08:53:26 | 6m09s |
| four 1,500-image extractions | 08:53:26 | 08:55:34 | 2m08s |
| corrected 1,500/20 baseline | 09:20:40 | 09:24:25 | 3m45s |
| independent baseline repeat | 09:25:03 | 09:28:50 | 3m47s |
| remaining 18 diagnostic cells | 09:29:56 | 10:33:47 | 1h03m51s |

The experiment phases above cost about $5.60 in box wall time: $0.58 extraction and $5.02
diagnostic including the reproducibility repeat. This excludes setup, tests, later
bootstraps, and idle conversational time. The rented box bills wall-clock rather than GPU
utilization, so the provider's final bill is the authoritative total.

## Extraction

| model | images | grid | throughput | cache bytes |
|---|---:|---|---:|---:|
| SigLIP2 | 5,000 | pooled 4x4 | 30.20 img/s | 1.48 GB |
| SigLIP2 | 5,000 | full 1024 | 23.01 img/s | 94.37 GB |
| Muse Glimmer | 5,000 | pooled 4x4 | 21.47 img/s | 4.02 GB |
| Muse Glimmer | 5,000 | full 1024 | 13.82 img/s | 158.60 GB |
| SigLIP2 | 1,500 | pooled 4x4 | 28.76 img/s | 0.44 GB |
| SigLIP2 | 1,500 | full 1024 | 20.82 img/s | 28.31 GB |
| Muse Glimmer | 1,500 | pooled 4x4 | 21.51 img/s | 1.20 GB |
| Muse Glimmer | 1,500 | full 1024 | 12.56 img/s | 47.58 GB |

The 5,000-image full caches total 252.97 GB, not the brief's 193 GB. With pooled caches the
5,000-image control occupies 258.46 GB. Image ids agree between pooled and full grids and
the 1,500-image set is an exact prefix of the 5,000-image set for both Towers.

## Measured 5,000-image cell costs

These are single diagnostic cells on the five-point rate grid. Each includes five
validation-search fits and three seeded test fits.

| Tower | grid | arm | epochs | wall |
|---|---|---|---:|---:|
| SigLIP2 | full | raw | 20 | 761 s |
| SigLIP2 | full | raw | 100 | 3,831 s |
| SigLIP2 | full | matched | 20 | 461 s |
| SigLIP2 | full | matched | 100 | 2,004 s |
| Muse Glimmer | full | matched | 20 | 549 s |
| Muse Glimmer | full | matched | 100 | 2,051 s |
| SigLIP2 | pooled | raw | 20 | 13 s |
| SigLIP2 | pooled | raw | 100 | 57 s |
| SigLIP2 | pooled | matched | 20 | 11 s |
| SigLIP2 | pooled | matched | 100 | 38 s |
| Muse Glimmer | pooled | matched | 20 | 11 s |
| Muse Glimmer | pooled | matched | 100 | 36 s |

The brief's 1,576 s raw and 687 s matched estimates assumed the full eleven-rate grid.
Normalizing the 20-epoch five-rate measurements by the training-count ratio `14 / 8` gives
about 1,332 s for SigLIP2 raw and 807-961 s for the matched cells. The raw estimate was 18%
high; the matched estimate was 17-40% low.

## What the brief got wrong

- Extraction took 8m17s wall, not about 25 minutes; the measured roster throughputs were
  conservative for three lanes, while Muse full landed slightly below its old 14.8 img/s.
- Full-cache storage was understated by about 60 GB. Muse Glimmer full alone is 158.60 GB
  at 5,000 images because all eight Tower slices plus `merged` and wide `projected` Stages
  are present.
- The 1,500-image Muse full cache is 47.58 GB, not the 31 GB recorded by the earlier run.
- Capacity-matched results were not reproducible cell-by-cell because randomized PCA ran
  before a seed. The deterministic reducer fix is commit `8c82a96`.
- Lever A does not fix the head. The brief's expected $12 Job 3 phase was correctly avoided.
- The diagnostic itself exceeded the expected $2 once the deterministic baseline repeat
  and the slow 100-epoch raw cell were included; measured active diagnostic cost was $5.02.
