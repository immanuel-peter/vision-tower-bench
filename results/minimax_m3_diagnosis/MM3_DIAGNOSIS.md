# MiniMax-M3 recognition diagnosis

## Verdict

MiniMax-M3's ImageNet-100 recognition is low because the Tower features themselves carry little class information. Pool-4, 448 vs native 672, preprocessing, adapter plumbing, and a cosine-kNN readout were all tested on this box; none of them recovers roster-level accuracy. Confidence: medium-high. Weight parity vs the parent MoE was not re-run here.

## Hypothesis table

| ID | Status | Deciding number |
|---|---|---|
| H0 extraction bug (preprocess/grid/token order) | REJECTED | adapter.extract L32 vs direct `MiniMaxM3VLVisionModel` + `raster`: max_abs = 0 |
| H1 pool-4 x token-0-concentrated semantics | REJECTED | full-grid L32 attention 0.4043 vs pool-4 0.3323 (need ~0.7). Token-0 L2 / rest = 1.00. Token-0 cosine-kNN 0.031 |
| H2 resolution/RoPE failure (native 672, bench 448) | REJECTED | 672-pool4 L32 attention 0.3038, worse than 448-pool4 0.3323 |
| H3 preprocessing mismatch (temporal repeat, normalization) | REJECTED | square-cropped 448 ours vs HF processor hidden: max_abs = 0 |
| H4 nonlinear-only class info | REJECTED | cosine-kNN k=20 mean-pool L20 0.1297, L32 0.1041, both below linear 0.33. MLP skipped |
| H5 genuinely weak Tower | SUPPORTED | H0-H4 rejected. Next-worst roster Tower is 0.837 at the same cell |

## Evidence chain

- Vocabulary used here: Tower, Projector, Stage (`tower` / `merged` / `projected`), Capability Profile. Not backbone. Not leaderboard.
- Semantic protocol (from `results/README.md`): ImageNet-100 validation, 13,000 images, 448 square, adaptive-avg-pooled to 4x4. Probe: `vtb.probe_run` eleven-point LR grid, 20 epochs, 3 seeds, match-capacity PCA to 512. Attention and mean readouts.
- Roster MiniMax-M3 pool-4 L32: attention 0.3407, mean 0.2238. Files: `results/minimax_m3_probe_attention_matched.json`, `results/minimax_m3_probe_mean_matched.json`. Best attention cell on that Tower is L20 raw 0.4186.
- This box pool-4 L32: attention 0.3323 +- 0.0037 (lr 3e-4), mean 0.2009 +- 0.0015 (lr 1e-2). Files: `/tmp/mm3_pool4_attn.json`, `/tmp/mm3_pool4_mean.json`. Same anomaly. Offset vs roster is small (bf16 extract on L40S vs the original run). Not a broken pipeline: chance is 0.01.
- Full-grid extract: `cache/minimax_m3_448_full`, 13,000 images, 1024 tokens, 324 GB. Command: `python -m vtb.extract --model minimax_m3 --images data/imagenet100/validation --device cuda --workers 8 --out cache`.
- Full-grid L32 attention (subsampled-train PCA to 512, then `run_cell`): 0.4043 +- 0.0112. File: `/tmp/mm3_full_attn.json`. Protocol rule: >= ~0.7 supports H1. 0.40 is a real +0.07 over pool-4, not a recovery to Kimi/GLM levels.
- Full-grid L32 mean (mean-pool shards, then matched linear): 0.2077 +- 0.0015. File: `/tmp/mm3_full_mean_L32.json`. 32 is divisible by 4, so equal-area pool-4 mean must match full-grid mean. 0.2077 vs 0.2009 agrees.
- Token-0 training probe skipped (H1 gate). Feature stats replace it: L32 token-0 L2 / rest = 0.995; cosine-kNN on token 0 is 0.0308 (L32) and 0.0415 (L20). Token 0 is not a class slot. HF `pooler_output = hidden_states[:, 0]` is the first packed token of the batch, not a per-image CLS.
- Forward parity, 8 ImageNet images, patched script `scripts/minimax_forward_parity.py`. Square-crop 448 then ours vs `MiniMaxM3VLImageProcessor`: pixel max_abs 2.4e-7, hidden max_abs 0. adapter.extract vs direct forward+raster: max_abs 0. pooler_output vs packed `[:, 0]`: max_abs 0. File: `/tmp/mm3_forward_parity.json`.
- HF processor on raw PIL (no square crop) uses dynamic `smart_resize` and does not emit a 32x32 grid. That is a resize policy difference. The bench path for every Tower is `square_crop`. On that path MiniMax matches HF bit-exact.
- `grid_thw [[1,16,16]]` on 32x32 patches raises in `apply_rotary_pos_emb_vision` (8192 vs 2048). Wrong grid does not produce silent NaNs or rank collapse. The roster grid `[[1,32,32]]` is the one that runs.
- 672 pool-4, final `tower` only: 13,000 images, 16 tokens. Probe L32 attention 0.3038 +- 0.0005. File: `/tmp/mm3_672.json`. Native size does not help.
- Feature stats on 2048-image subsample of full-grid plus all-image mean/token-0 kNN. File: `/tmp/mm3_feature_stats.json`.
  - L20: token norms uniform (token0/rest 1.003), effective rank 76.7, kNN mean 0.1297.
  - L32: token norms uniform (token0/rest 0.995), effective rank 18.5, energy in top 8 singular 0.695, kNN mean 0.1041.
- MLP probe skipped: kNN never reached 0.45. Linear attention 0.33 already beats kNN 0.13, so the missing class info is not "present but nonlinear."
- PCA silhouette vs recognition: L32 is low-rank (effective rank 18.5). A sharp figure-ground map can live in a few spatial components while class-conditional directions are weak. That is consistent with last-on-recognition and first-on-silhouette.
- Weight bit-exact test vs MiniMaxAI/MiniMax-M3 @ f0e1c1e was not re-run (parent shards gated/offline). `tests/test_minimax_m3.py` passed, 2 tests. `tests/test_parity.py::test_minimax_m3_tower_and_projector_match_parent_checkpoint` skipped.
- In-repo full-vs-pooled validation remains Muse Glimmer and SigLIP2 only (`results/pooling/`). MiniMax now has a full-grid number: 0.4043 attention, 0.2077 mean, L32 matched.

## If H1 had confirmed

It did not. Do not re-extract MiniMax full-grid for the v2 semantic tables. Do not change pooling protocol roster-wide. Optional footnote: MiniMax full-grid L32 attention is 0.404 vs pool-4 0.332. The gap is real and small. It does not move MiniMax off the bottom of the Capability Profile.

The Projector-hurts-geometry claim is outside this diagnosis.

## Draft Limits replacement

- MiniMax-M3 ImageNet-100 pool-4 attention is 0.33 top-1 (this box 0.3323; roster 0.3407). Full-grid attention is 0.404. Native 672 pool-4 is 0.304.
- Cause is not pool-4 dilution of a class token, not 448 vs 672, not HF preprocessing, not adapter raster/RoPE. Square-448 ours vs HF hidden is bit-exact. Token-0 norms match the rest of the grid.
- Treat MiniMax-M3 as a genuinely weak recognition Tower on this probe. Keep the pool-4 number in the v2 tables. Drop "the cause is unknown."

Other Draft edits:
- Depth-curve sentence that MiniMax peaks at L20 (0.419 raw) and falls to 0.34-0.37 at L32 can stay. This run did not re-probe the depth curve.
- Projector-vs-Tower geometry gaps are untouched.
- PCA-paradox sentence can stay, with one clause: L32 features are low-rank (effective rank ~18), so a silhouette can be sharp while class kNN is 0.10.

## Appendix

New scripts (not committed):
- `scripts/minimax_full_probe.py` (subsampled-train PCA, then `run_cell`; `--token-view`, `--mean-pool-first`)
- `scripts/minimax_token0_probe.py`
- `scripts/minimax_forward_parity.py`
- `scripts/minimax_feature_stats.py`
- `scripts/minimax_mlp_probe.py`
- `scripts/minimax_continue.sh` (tmux driver)
- `scripts/minimax_write_report.py`

Result JSONs:
- `/tmp/mm3_pool4_attn.json`, `/tmp/mm3_pool4_mean.json`
- `/tmp/mm3_full_attn.json`, `/tmp/mm3_full_mean_L32.json`
- `/tmp/mm3_forward_parity.json`
- `/tmp/mm3_672.json`
- `/tmp/mm3_feature_stats.json`
- `/tmp/mm3_token0.json` (skipped), `/tmp/mm3_mlp.json` (skipped)

Commands and wall time (UTC 2026-09-16):

| Phase | Wall | Notes |
|---|---|---|
| Step 0 pytest + smoke | 05:57:29-05:57:54 (~25 s) | 2 passed; 32 images, depth points [4,8,12,16,20,24,28,32] |
| Step 1 pool-4 extract | 05:58:36-06:08:32 (~10 min) | 13,000 images, 22 img/s, 16 tokens |
| Step 1 probes | 06:08:43-06:09:40 (~1 min) | attn 0.3323, mean 0.2009 |
| Step 2 full-grid extract | 06:10:14-06:23:15 (~13 min) | 324 GB, 1024 tokens, 17 img/s |
| Step 2 full-grid L32 attention | 06:23:46-06:50:36 (~27 min) | 0.4043. Vanilla `probe_run` would OOM (~68 GB float32). Used subsampled PCA + `run_cell` |
| Step 2 full-grid L32 mean | ~23 s | 0.2077, mean-pooled in shards |
| Step 4 forward parity | ~13 s (patched re-run) | first run died in (c) after (a)(b) had already printed max_abs=0 |
| Step 5 672 extract + probe | 06:51:15-07:21:37 (~30 min) | extract 7.3 img/s; probe 0.3038 |
| Step 6 stats | 07:21:37-07:23:33 (~2 min) | MLP skipped |
| Report | 07:23:33 | this file |

Caches kept until you delete them: `cache/minimax_m3_448_pool4`, `cache/minimax_m3_448_full` (324 GB), `cache/minimax_m3_672_pool4`. Disk after runs: ~223 GB free of 614 GB.

Box: 1x NVIDIA L40S 48 GB, 70 GB RAM. tmux session `mm3` (remain-on-exit). Log: `/tmp/mm3_logs/continue.log`.
