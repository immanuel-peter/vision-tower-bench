# Reproduce and resume the corrected suite

The retained `run-corrected-all.py` controller and `run-suite.py` command inventory
use independently forwarded images for both pinned MiniMax checkpoints. Every
CUDA command requires an explicit GPU reservation and holds
`flock /tmp/vtb-delivery/locks/gpu0.lock` for its full lifetime. Run shells through
`bash -ic`, with a separate environment and writable output root.

The delivery configuration is physical GPU 0, logical `cuda:0`, `GPUS=0`,
`LANES=1`, `THREADS=8`, and at most two loader workers. `HF_HOME` points to the
shared download cache. Prepared inputs are read-only; commands verify component
READY markers. Validate actual selected IDs and splits independently of those
markers.

`run-suite.py --category CATEGORY --list` prints commands without CUDA execution.
`--only SUBSTRING` restricts execution to a named job. The controller sequences
extraction, correspondence, priority Projector comparisons, recognition,
standardized diagnostics, primary label budgets, geometry, KITTI, throughput,
perturbations, sibling comparisons, and remaining within-model comparisons.
Commands, UTC starts, elapsed times, exit codes, resource settings, and source
hashes are retained in `execution.jsonl`. Timings include lock waits.

Resume only outputs whose actual contents and source epoch have been validated.
Matrix DONE markers do not establish success. Inspect alerts and every expected
JSON payload. Pre-correction outputs in `invalid-packed-attention` are excluded.
The six corrected primary feature caches have full file hashes and split-ID
manifests; sampled real inputs also match the independent reference exactly.

`postprocess.py` constructs CPU correspondence and perturbation paired intervals
from their source measurements, records the finite intersection and exclusion
union, and combines standardized label-budget diagnostics. The visibility audit
must explain every SPair exclusion from source keypoints. Nonempty pairs require
finite metrics.

`audit-paired-source.py` checks paired arrays against primary source payloads.
`audit-measurements.py` checks the complete inventory for each model, including
learning-rate grids, seeds, checkpoint identity, finite metrics, selected IDs,
and recomputed intervals. `publish-measurements.py` requires both complete
reports and the paired-source audit before copying additive measurements.
Historical M3 files remain intact. Explicit source overrides select its fresh
compatible controls for figures.

Regenerate all default figures with `scripts/figures.py`, run the full regression
suite including figure tests, and refresh per-Stage comparison tables, checklist,
and handoff after validation. See `image-isolation.md` for source-reference limits.
