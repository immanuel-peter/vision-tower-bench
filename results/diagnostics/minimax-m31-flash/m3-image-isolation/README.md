# Compatible M3 control reruns

These measurements use the pinned M3 weights and the same corrected image-independent
inference as M3.1 Flash. Historical M3 files remain in their original directories.
The explicit figure-source overrides in results/index.json select this additive
scope for the regenerated figures and training-only comparison.

All primary recognition and geometry cells, both label-budget readouts, all
correspondence datasets, 13 perturbation conditions, KITTI transfer, throughput,
within-model comparisons, and four standardized attention label-budget cells are
rerun under the existing corrected LR and selected-LR seed protocol.
All required measurements are complete and content-validated in
[the full control inventory](../measurement-validation-minimax_m3.json).
The control throughput record also lives in this scope.
The paired comparison and all figure selections use these compatible reruns.
