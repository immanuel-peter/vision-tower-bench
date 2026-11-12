# Issue 22 scientific delivery checklist

- [x] Checkpoint adapter, pinned index-discovered shards, shared preprocessing/raster, 2560-wide Projector, registry, exporter, and regression tests.
- [x] All 333 BF16 source tensors exact. Independent complete-shard reader and Hub SHA256 confirm both source identities.
- [x] Independent pinned Qwen4ExpVisionModel versus Qwen3_5VisionModel, 20 cases each in CPU FP32, CPU BF16, and GPU BF16, at zero absolute/relative tolerance.
- [x] Immutable Hub publication and independent download/reload at 120ed5d5baf11cb478099350e94218ee3cbed3df. All 333 tensors and 20 CPU BF16 cases exact. Native state has 333 persistent tensors and two documented nonpersistent rotary buffers.
- [x] Checkpoint-specific Qwen Community License 1.0 copied byte-for-byte. No inherited Apache claim.
- [x] Full 333-tensor sibling drift, all 27 layers, positional embeddings, and Projector. All 331 shared shapes differ; final weight/bias change width. Training causes unproven.
- [x] Full corrected primary suite at 448: recognition40, geometry40, KITTI2, label budgets8, correspondence9 Stage cells per checkpoint, perturbation39 conditions per checkpoint, throughput768 images.
- [x] All 33 primary paired records: recognition4, geometry10, KITTI3, correspondence12, perturbation4. Matched hardware/settings and own rerun point estimates retained.
- [x] Prescribed seed-0 LR search and selected-rate seeds0/1/2, all exact sample IDs and full native grids verified. DIODE771 split541/115/115 regression preserves numeric outputs and documents validator-only correction.
- [x] SPair all3600 selections per checkpoint, exactly192 source-verified zero-keypoint IDs,3408 paired finite intersection. Explicit per-ID reasons; all nonempty metrics and aggregates finite.
- [x] Isolated exact-R2 standardized diagnostics, training-split-only sample mean/std and PCA, primary outputs unchanged. Four label fractions,78 perturbation condition cells, six raw replays, six channel records and four diagnostic paired records.
- [x] Conditioning limitations stated separately from checkpoint results and unsupported training explanations. Primary weak Tower/merged clean readouts support no intrinsic robustness claims.
- [x] Additive index, roster, history, standard results and fresh sibling controls. All584 historical result files and64 previous standardized runs preserved.
- [x] All12 figures regenerated. Figure source tests check corrected deepest stages, original sweeps, standardized labels, own paired intervals, and SPair macro averaging.
- [x] Final tests156 passed/50 expected weight skips; Qwen weighted tests7 passed. Source/license byte-preserving diff review and secrets scan passed.

The [benchmark summary](benchmark-summary.md), [requirement audit](requirement-content-audit.json), and [final validation](final-validation.json) provide the evidence map. PR URL, final remote SHA, checks/reviews, and parent delivery acknowledgment are recorded in the external issue22 handoff after publication.
