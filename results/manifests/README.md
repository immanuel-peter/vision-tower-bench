# Matrix completion records

These TSVs preserve the expected cell counts for the original six-Tower runs.
They are historical completion records, not inventories of the v2 suite.
Each filename is relative to the corresponding result directory:

| Manifest | Result directory |
|---|---|
| [expected_cells_semantic_v2.tsv](expected_cells_semantic_v2.tsv) | [recognition/](../recognition/) |
| [expected_cells_geometry_raw.tsv](expected_cells_geometry_raw.tsv) | [geometry/](../geometry/) |
| [expected_cells_geometry_matched.tsv](expected_cells_geometry_matched.tsv) | [geometry/](../geometry/) |

The same mapping is recorded in `cell_manifests` in [index.json](../index.json).
Other completion TSVs remain beside the runs they describe. Filenames and counts
inside every TSV are unchanged.


The MiniMax addition registers fourteen per-model manifests. Flash manifests
resolve against their root task directories, and M3 control manifests against
`diagnostics/minimax-m31-flash/m3-image-isolation/`. Standardized manifests
resolve against the separate diagnostic directory for each model. The exact
mapping is in `cell_manifests` in the index. Perturbation TSVs count three Stage
cells, each containing thirteen conditions. Full inventory reports separately
verify those 39 measurements and the throughput record per checkpoint.
