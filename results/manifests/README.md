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
