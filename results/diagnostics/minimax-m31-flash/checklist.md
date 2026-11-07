# Issue 21 delivery checklist

Source: https://github.com/immanuel-peter/vision-tower-bench/issues/21
Base: a2277c8dd9d9f81bbf29cc4a5c545f8fd6513902

- [x] implementation: vtb/adapters/minimax_m3.py; scripts/export_minimax_m3_vision.py; tests-minimax.log; tests-weights.log.
- [x] source identity: results/diagnostics/minimax-m31-flash/validation.json identity and source sections, 523 tensors per checkpoint.
- [x] independent forward parity: results/diagnostics/minimax-m31-flash/validation.json forward section; independent source loader using native HF implementation, not a second modeling implementation.
- [x] weight drift: results/diagnostics/minimax-m31-flash/validation.json drift, all 515 Tower and eight Projector tensors; drift-summary.json per-layer arithmetic means.
- [x] license: hf/MiniMax-M3.1-Flash-Vision/LICENSE and NOTICE; immutable release file hashes.
- [x] experiment inventory: expected-inventory.json and results/diagnostics/minimax-m31-flash/expected-inventory.json; seed-0 selection, three-seed selected-LR retraining.
- [ ] full experiments: validate using standard results payloads, validated cell/sample/seed counts; evidence TBD by worker.
- [ ] writeups and figures: validate using additive index/roster/figure sources and generated figures; evidence TBD by worker.
- [x] HF release: hf-publication.json revision 75a5fc5bba7cfb01cfcf15072a9d823714e54121; hf-reload.json all 523 exact tensors; hf-release-forward.json exact CPU BF16 forward at 56 and 448.
- [ ] tests: validate using affected and regression logs; evidence TBD by worker.
- [ ] PR: validate using non-draft main-target URL, remote head verified; evidence TBD by worker.

## Authoritative requirements and open questions

## Why

[MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) gives us a **controlled training-only comparison** against `minimax_m3`. The Tower architecture and preprocessing are the same. The weights are not.

**Same as M3:**
- Tower config (CLIP-style ViT, 3D RoPE, 32 layers, hidden size 1280, 16 heads, MLP size 5120, patch size 14) and 2×2 patch merging. The only config difference is `vision_config.image_size` (2016 → 672). That field is nominal with RoPE, and both preprocessors cap input at 1008 px.
- `preprocessor_config.json`
- The 515 Tower tensor names, `projector_hidden_size` / `text_config.hidden_size` (6144), and LM depth
- Precision of the vision path. Flash quantizes the LM (MXFP8 + NVFP4 MoE), but `vision_tower`, `multi_modal_projector` and `patch_merge_mlp` are in `ignored_layers`, so they keep full precision.

**Different:** the weights. I fetched sampled tensors with HTTP range requests and compared them:

| Tensor | Cosine | Relative L2 difference |
|---|---|---|
| `embeddings.patch_embedding.weight` | 0.998 | 7% |
| `layers.0.layer_norm1.weight` | 1.000 | 0.2% |
| `layers.31.mlp.fc2.weight` | 0.993 | 12% |
| `multi_modal_projector.linear_1.weight` | 0.65 | 80% |

All 10 hashed tensors differ. The Tower drifts more with depth, which fits continued training from the M3 checkpoint. The Projector was heavily retrained. So we'd expect a moderate shift in `tower` Stage results and possibly a large one in `merged` / `projected`. Any delta between the two Capability Profiles comes from training, not architecture. This is the same kind of sibling comparison as `mimo26_pro` / `mimo26_flash`.

## How

1. **Adapter.** Refactor `vtb/adapters/minimax_m3.py` to use the `Checkpoint` pattern from `vtb/adapters/mimo26.py`, with a base adapter and subclasses `MiniMaxM3Adapter` / `MiniMaxM31FlashAdapter`. Per-checkpoint fields:
   - `repo`: `MiniMaxAI/MiniMax-M3.1-Flash`, `revision`: `54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e`
   - Shards: Flash uses a different layout from M3. Its Tower is in `model-00048-of-00048`, its Projector in `00013` + `00048`, and its `patch_merge_mlp` in `00013`. Discover shards from the pinned `model.safetensors.index.json` (as `mimo26.source_weights` does) instead of hardcoding `VISION_SHARD` / `PROJECTOR_SHARDS`.
   - Bundle: `hf/MiniMax-M3.1-Flash-Vision`, model ID `immanuelpeter/MiniMax-M3.1-Flash-Vision`
   - `remap_tower`, `remap_projector`, `Preprocess`, `raster` and `merge_2x2` stay shared unchanged.
2. **Export.** Parameterize `scripts/export_minimax_m3_vision.py` by checkpoint (keep the 515 Tower / 8 Projector tensor-count checks) and export the Flash bundle into `hf/`.
3. **Registry.** Register `minimax_m31_flash` in `vtb/extract.py`, then add it to `batch_for` in `scripts/perturbation_matrix.sh` (batch 8, resolution 448, same as M3) and to the model lists and labels in `scripts/figures.py`.
4. **Tests.** Extend `tests/test_minimax_m3.py` to cover both checkpoints. Add a check that the Flash bundle's Tower and Projector tensors are **not** equal to M3's, so the export can't silently pick up the wrong repo.
5. **Runs.** Run the full suite at 448. If compute is tight, run the `projected`-vs-`tower` bootstrap comparisons first (correspondence SPair/ScanNet, geometry depth/normals), since the Projector delta is the largest.
6. **Writeup.** Add an M3 vs. M3.1-Flash per-Stage delta to the results index and note it in the paper as a training-only comparison.

## Open questions

- Does `vision_config.image_size` feed into anything in the HF `minimax_m3_vl` modeling code? It shouldn't, given RoPE, but check before relying on it.
- 4 of 515 Tower tensors were compared numerically. Once the bundle is exported, run a per-layer cosine sweep and add it to the writeup.

## Current remaining work

- Full experiments: recognition extraction complete and content-validated, 13000 images x ten slices. No probe or correspondence measurement claimed complete.
- GPU 0 explicitly released after recognition extraction. Await explicit parent regrant before any CUDA command.
- Resumable full commands: /tmp/vtb-delivery/issues/21/run-suite.py. Use --category CATEGORY --list to inspect commands. GPU jobs are serial and hold flock for full process lifetime.
- Regression: tests-regression-scoped.log has 130 passed, 48 skipped. Legacy MoonViT weight tests ignore VTB_SKIP_WEIGHTS and are excluded to avoid unrelated shard downloads. Figure tests await complete new results. Seven offline MiniMax tests pass; ten actual-weight and provenance tests pass in tests-weights-final.log.
- Prepared inputs: imagenet-provenance.json verifies 13000 IDs and 100 labels. prepared-provenance.json verifies 2000 images x 13 conditions and 1000 KITTI images. Actual pair selection in correspondence-selection.json verifies ScanNet1500, SPair3600, NAVI555 unique IDs and selected image files.
- Writeup/index/roster/figure registration exists with explicit pending status; final per-Stage deltas and figure generation await actual measurements.
- No paper source exists in this worktree; results writeup is the available paper-ready record.
- Commit, push and separate non-draft PR remain pending. Never claim issue delivery complete before measurements and PR are ready.
