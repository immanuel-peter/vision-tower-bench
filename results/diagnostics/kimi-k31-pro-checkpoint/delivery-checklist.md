# Issue 24 current delivery checklist

Source: https://github.com/immanuel-peter/vision-tower-bench/issues/24
Base: a2277c8dd9d9f81bbf29cc4a5c545f8fd6513902
PR: https://github.com/immanuel-peter/vision-tower-bench/pull/29, non-draft, main, Fixes #24.
Active delivery. Full phase history is preserved in execution.jsonl and checklist-historical-before-current-state.md.

- [x] Adapter/export/registry: separate Pro checkpoint, indexed pinned shards, original K3 key/results preserved, final norm and Stage widths1024/4096/7168. tests/test_kimi_checkpoints.py and Stage tests.
- [x] Identity:165Tower+3Projector exact BF16 tensors each; all168Pro tensors differ from K3. identity-*.json, immutable reload and checkpoint-selection regression.
- [x] Independent official Pro source: vLLM1ecc9cc4ce6626352df39e4364579ec376c10761, Pro config/weights, no LM. reference/provenance.json and code hashes.
- [x] Accepted mathematical conversion parity: original zero tolerance, CPU/CUDA BF16,30cases each. Production official TORCH_SDPA BF16 remains FAILED; FP32 agrees. gpu-parity2-summary.json. No loosened tolerance.
- [x] Full drift including learned positions, final norm and Projector: weight-drift.json and weight-drift-aggregated.json. No causal RL-only claim.
- [x] License/release/reload: custom pinned Pro license, separate code license, benchmark HF6ab95e2,168parent tensors,17file hashes,30exact reload cases.
- [x] Inventory before runs: expected-inventory.json,286primary/102paired. Seed0 LR search then selected-rate three-seed retraining, unchanged epochs/splits/grids/reducers.
- [x] Both sibling feature sets complete: ImageNet13000pool4/all10, DIODE771full/all10, KITTI1000full/final2, perturbation13x2000/all3. Actual IDs/shapes/raster/finiteness/allshard hashes checked. Separate CPU/GPU namespaces retained.
- [x] Full metric suite: all286primary cells independently content-validated across88payloadgroups.92/102paired comparisons registered;8correspondence intervals staged/replayed. GPU1finishes within-ProKITTI and raw-normal pairs; GPU0released with0MiB/noPIDs/freeflock proof.
- [x] Actual correspondence selections: NAVI555,ScanNet1500,SPair3600 exact roots/IDs. SPair192individually verified zero-visible-keypoint IDs and3408nonempty. spair-visibility-audit.json. Finite all nonempty rows and aggregates required.
- [x] Final independent audit: all286primary actual-content checks PASS. AfterGPU1release, register8existing correspondence comparisons and replayall102intervals; exact inventory/sourcehashaudit. audit_completed_primary_content.py, audit_all_paired_intervals.py.
- [x] Final figures/index/roster/writeups/tests: fill actual missing Projector panels, retain assertions. Six current semantic/geometry figure tests pass; full final regression suite pending.
- [x] Published functional inference gate: fresh immutable5311e0408a6107f78f8c70bb70f7aa205b94e724, exact published command, real500x334image processed448, CUDA BF16 batch1, all3shapes,exit0/7.606s. inference-smoke/parent-published-smoke.
- [x] Example numerical correction: exact GELU in bench example matches adapter bit-exact on real cached merged tokens. Twelve CPU tests pass. Owner one-line patch inference-smoke/example-gelu-owner.patch. Corrected publication/CUDA numerical proof pending owner coordination and free-GPU grant. Original functional smoke retained as historical pass.
- [x] Separate non-draft PR29 pushed/linked. Continue this same PR; no duplicate, merge, issue close or PR watch.

## Resource and ownership boundary

GPU0phase completed and explicitly released at05:22:38UTC,0MiB/noPIDs/flock reacquired/released. GPU1remaining jobs continue unchanged under exclusive gpu1.lock,8threads,<=2loaders,vision batch1. Both arms of each paired fit share physical GPU/settings. No extra CPUtraining. Originalassignment/source snapshot predates jobs.

HF cards/structure/manifests/licenses/layout/weights belong to other actors. Preserve immutable bundle evidence and owner Git edits verbatim. Only bench-side functional example correction and exact owner handoff retained here; no competing HF upload.

Final completion supersedes historical progress above: delivery-validation.json verifies286/102 exact cells,142public payload copies, all102exact interval replays,176tests/51skips,10defaultfigures/30files/236inputs and corrected published d32b4a05dc0f52259334b0c808514b2bb44c3130 CUDA BF16 all3Stages bit-exact. Both original GPUs and corrected smoke released. No pending experiment or inference gate. Production official BF16 SDPA failure remains documented; accepted math-reference conversion is distinct. HF cards/layout/manifests/licenses remain other actors owned; parent made only explicitly approved example correction.
