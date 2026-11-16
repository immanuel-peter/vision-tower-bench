# Results

Measurements live in the private Dataset [`immanuelpeter/vision-tower-bench-results`](https://huggingface.co/datasets/immanuelpeter/vision-tower-bench-results), never in Git. Locally they sit in the gitignored `results/`, which mirrors the Dataset's `results/` directory.

## Reading

```bash
hf download immanuelpeter/vision-tower-bench-results --repo-type dataset --include 'results/*' --local-dir .
```

Start from `results/index.json`. Its keys are:

- `cohorts`: named sets of slugs that were run together, each with `models` and `description`.
- `experiments`: one entry per result family, with `path`, `protocol`, `status`, `cohort` and an optional `supersedes`. `status` tells you which family is current.
- `cell_manifests`: the expected-cell TSVs in `manifests/` that show whether a run is complete.

Payloads are grouped by axis: `recognition/`, `geometry/`, `correspondence/`, `label-budget/`, `perturbation/`, `kitti/`, `bootstrap/` and `throughput/`. `diagnostics/`, `evidence/` and `history/` are the archive's old working notes. Read them for context and add nothing to them.

## Writing

- **Payloads** are the JSON the `vtb` runners write, kept exactly as written, since they record their own method. Each goes into its axis directory and follows the file naming already used there, with roster slugs.
- **The index** changes in two ways only. A new Tower gets a new `cohorts` entry, and a new kind of measurement gets a new `experiments` entry.
- **Manifests** are the run's expected-cell TSV, placed in `manifests/`.
- **Everything else** (notes, logs, checklists, summaries) stays in `tmp/<task>/`.

Upload as a Hub pull request, so the measurements are reviewed separately from the code PR:

```bash
hf upload immanuelpeter/vision-tower-bench-results results/<axis> results/<axis> --repo-type dataset --create-pr --commit-message "<slug>: <axis>"
```

Link each Hub PR in the code PR body.
