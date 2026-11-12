# Issue 21 delivery checklist

Issue: https://github.com/immanuel-peter/vision-tower-bench/issues/21
Base: a2277c8dd9d9f81bbf29cc4a5c545f8fd6513902
PR: https://github.com/immanuel-peter/vision-tower-bench/pull/26

Evidence root is `/tmp/vtb-delivery/issues/21`. Committed evidence is under
`results/diagnostics/minimax-m31-flash/`. The archived authoritative issue is
`live-issue.json`; environment and full-delivery requirements are in the parent
`COORDINATION.md`. This checklist supersedes the earlier extraction-only handoff.
All measurements, audits, figures and tests are complete. The delivery was
committed, pushed, linked and verified against the remote PR head. GPU0 was released after all CUDA children exited;
`gpu-release.json` records exclusive lock reacquisition and no active processes.

- [x] Checkpoint adapter pattern, Flash registry and batch/resolution registration.
  `vtb/adapters/minimax_m3.py`, `vtb/extract.py`, `scripts/perturbation_matrix.sh`,
  `tests/test_minimax_m3.py`; `tests-corrected-methods.log` and
  `tests-corrected-weights.log`.
- [x] Discover every Tower, Projector and merge shard from the pinned index;
  parameterize export and enforce 515 Tower plus eight Projector tensors.
  `scripts/export_minimax_m3_vision.py`, both export manifests,
  `validation.json` source and identity sections. Every tensor matches its
  separately pinned parent; cross-checkpoint differences exclude wrong-repo export.
- [x] Resolve nominal image sizes 2016 and 672 before benchmarking.
  `nominal-image-size-audit.json`, `config-audit.json`, `validation.json`.
  Native modeling has no image-size field use; runtime grids determine positions.
- [x] Complete all-layer and Projector drift, including equal tensors.
  `validation.json`, `drift-summary.json`, `layer-drift.json`, `weight-drift.png`.
  All 515 Tower and eight Projector tensors are covered; 509 Tower and all eight
  Projector tensors differ. Concatenated layer drift is not monotonic. Per-tensor
  means and concatenated statistics are distinguished explicitly.
- [x] Independent pinned-parent forwards before substantial corrected benchmarks.
  `isolated-parent-forward.json`, `validate-isolated-forward.py`,
  `image-isolation.md`: both checkpoints, 56 and 448 pixels, batches one, two and
  eight, reversed order and changed neighbors, 200 exact slice comparisons in
  twenty cases. Max absolute error is zero in BF16. The reference independently
  loads pinned source tensors into native HF classes, rather than using a second
  modeling implementation or full language-model forward.
- [x] Correct native packed attention across unrelated images; quarantine own
  pre-correction features, results and paired caches.
  `batch-isolation-failure.json`, `image-isolation.md`, runtime
  `invalid-packed-attention/`. Every image now has a separate native forward.
  Historical published M3 results are preserved with their narrower provenance.
- [x] Verify real input/cache parity and all six corrected primary caches.
  `real-input-cache-parity.json` has 132 exact image-slice comparisons.
  `cache-audit-{recognition,diode,kitti}-{model}.json` record every file hash,
  finite tokens, dimensions, all selected IDs, split IDs and validated source epoch.
- [x] Publish licensed Flash bundle and independently reload its immutable files.
  `hf/MiniMax-M3.1-Flash-Vision/{LICENSE,NOTICE,README.md}`;
  `hf-publication.json`, `hf-current-release-reload.json`.
  Repo `immanuelpeter/MiniMax-M3.1-Flash-Vision`, immutable revision
  `55510a4d372b8ae502a04cc570c0fd06a4361771`. All source tensors are exact;
  CPU BF16 Tower and standalone Projector reload forwards are exact at 56 and 448.
- [x] Verify prepared inputs and actual correspondence selection.
  `imagenet-provenance.json`, `prepared-provenance.json`,
  `dense-correspondence-provenance.json`, `correspondence-selection.json`,
  `selection-manifest.json`, parent `prepared-data-audit.json` and `data-sha256.txt`.
  Actual counts: recognition13000, DIODE771, KITTI1000, perturbation2000 per
  condition, NAVI555, ScanNet1500, SPair3600. Exact correspondence roots and IDs
  are recorded. Label-budget subset IDs are deterministically reconstructed from
  audited cache order and exact runner hashes for all twelve semantic job invocations.
- [x] Preserve current primary protocol and declare full expected inventory.
  `expected-inventory.json`, `run-suite.py`, `run-corrected-all.py`.
  Semantic eleven-rate and geometry six-rate searches use seed zero; selected LR
  retraining uses three seeds. Eight Tower depths plus merged/projected are kept.
- [x] Complete and content-check all semantic cells for both models.
  `semantic-completion-audit.json`: eighty recognition, sixteen primary label-budget
  and eight opt-in standardized diagnostic cells, all on 13000 images.
- [x] Recover exact standardization diagnostic without changing primary Reducer.
  `recovered-channel-standardization.patch`, `standardization-method.md`,
  `vtb/channel_standardization.py`, method tests and standardized run payloads.
  Patch SHA256 `f5a2bc1dcda0e6be9a2d6abea9c1bcc9cb5817282ff6406489b5022dc65e330a`.
  Training-only FP32 mean, sample std correction1 clamped1e-6, exact randomized
  low-rank SVD formula, same inference transform and seed rules. Four fractions
  per checkpoint supplement the existing standardized figure sources.
- [x] Complete all eighteen correspondence cells and audit source visibility.
  `spair-visibility-audit.json`, `audit-corrected-spair.log`, measurements and
  `paired-source-validation-correspondence.json`. Retain all 3,600 SPair IDs;
  source decoding proves 192 zero-visible pairs and 3,408 finite nonempty pairs.
  Every nonempty pair and aggregate is finite. Paired intersections and every
  excluded ID/reason are recorded. Fifteen correspondence paired payloads are
  independently checked, with class-macro SPair supplements.
- [x] Complete both full forty-cell geometry matrices. All 80cells are content-validated
  in `audit-progress-flash-geometry40.log` and `audit-progress-m3-geometry40.log`.
- [x] Complete both KITTI transfer matrices, both throughput records and all
  seventy-eight perturbation Stage-condition cells.
- [x] Fully validate perturbation caches after extraction: each checkpoint must
  pass `audit-perturbation-features.py` and produce a complete
  `perturbation-cache-audit-{model}.json`, covering156shards, thirteen conditions,
  three Stages, exact2000IDs, finite BF16 tensors, hashes and source epoch.
- [x] Prepare/test lossless identity per-image evidence materialization.
  `identity-evidence.md`, `test-identity-evidence.py`, `tests-identity-evidence.log`.
  Original scores/protocol are preserved, raw backups are exact, repeated clean
  vectors must agree, reruns are idempotent and inconsistent vectors are rejected.
- [x] Apply identity materialization to actual completed perturbation outputs.
  `postprocess.py --section remaining` must retain original raw payloads and write
  `identity-evidence-materialization.json` with both original and augmented hashes.
- [x] Complete all 49 explicitly named paired payloads: ten within Flash, ten
  within fresh M3, twenty-nine sibling comparisons. Priority Flash matched depth
  and normals and all fifteen correspondence payloads are already complete.
  All points and intervals must come from the same paired run. Remaining output
  is checked by `audit-paired-source.py --section remaining`, merged into twenty
  source-validated correspondence/perturbation payloads, and both full inventory audits.
- [x] Pass `audit-measurements.py` and `--model minimax_m3` without `--partial`.
  Both reports must be complete with no missing files. Verify the union is the
  expected49 paired files, not only a successful process count.
- [x] Publish103 canonical measurement payloads additively, plus exact original
  perturbation evidence and validation/provenance records. `publish-measurements.py`
  requires all full audits, preflights every destination and refuses overwrites.
  Original M3 files stay intact; explicit figure-source overrides select new controls.
- [x] Generate65 same-run per-Stage comparison rows and the paper-ready note.
  `summarize-comparison.py`, `per-stage-deltas.json`, `comparison-writeup.md`.
  Include matched/raw DIODE, error-direction units, KITTI and all13 perturbations.
  No manuscript source exists; the results writeup is the paper-ready record.
- [x] Finish additive results index, roster, correction scope, writeups and all
  default figure generation; retain figure source hashes and rendered artifacts.
  Preserve main label `Nemotron 3 Omni`; the stale assertion is fixed.
- [x] Affected corrected method and real-weight checks pass.
  `tests-corrected-methods.log`:45passed,1weight-disabled skip.
  `tests-corrected-weights.log`:12 passed,8deselected.
  `tests-figure-source-selection.log`:5passed.
- [x] Run the final full `VTB_SKIP_WEIGHTS=1 pytest -q`, including every figure
  test, after publishing actual measured inputs. No figure exclusions or skips.
- [x] Refresh command/timing/source manifests, checklist and `handoff.json` with
  measured counts, immutable HF revision, parity limits, test evidence and final head.
- [x] Create and immediately link a separate non-draft PR targeting main with
  `Fixes #21`: PR26. The final body links the complete measured comparison, release and validation.
- [x] Commit/push complete delivery, immediately link PR26 again, update final
  body, and verify remote head, checks and review state. Do not merge or close.


## Final evidence summary

`delivery-validation.json` verifies103 canonical measurement payloads,49 unique
paired files,20 independent paired-source checks,312 perturbation shards,14 new
cell manifests and30 figure exports. Full regression:148 passed / 51 weight-disabled
skips, including every figure test. Actual MiniMax weight/isolation checks:12 passed.
Both full inventory reports have zero missing files. `gpu-release.json` proves
the reservation was released before final CPU work.

Validated code and measurement delivery commit: `0efce77f7716fd0401fe9a3f5b74658659c770d9`.
Remote PR is OPEN, non-draft and targets main. Prior bot change requests remain
on older commits; `review-resolution.json` maps all findings to delivered evidence.
The runtime handoff records the final metadata-only head and live check state.

## Queued HF package follow-up

- [x] Eight-heading house card, 64 to 68 lines and 402 to 297 words; original audit text retained.
- [x] Complete source/file manifest, both weight hashes unchanged, metadata/code-only upload.
- [x] Fresh-download real-image CUDA BF16 batch-one example passed all three shapes at revision b161cf1d04676db7a695762ac6208713fa352861.
- [x] Driver checked before Torch, GPU0 released with 0 MiB/no processes and lock reacquired.
- [x] CPU regression 152 passed / 51 skipped, including four new packaging cases.
- [x] Original 103 benchmark payload hashes remain unchanged.

Evidence: `hf-followup/report.json`, `cuda-smoke.txt`, `tests-full.txt` and `gpu-release.json`.
