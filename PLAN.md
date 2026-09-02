# vision-tower-bench plan

Written after the August 23 grilling session. CONTEXT.md defines project terms, and the
ADRs record why each decision was made. This file covers execution.

## Headline claim

The Projector preserves semantics useful to the language model but destroys spatial
information. If projected features match raw Tower features on depth and correspondence,
the claim fails. That null result is still publishable.

The roster run collected that null result over 448 cells, six Towers and four Projectors,
at a validation-selected rate and three seeds. `projected` beats `tower` on depth and
surface normals in every Projector and both capacity arms. The training-free
correspondence run does not permit the same roster-wide statement. Both completed
Projectors improve geometric correspondence on ScanNet and NAVI, with paired intervals
excluding zero. Semantic correspondence on SPair splits: Kimi K2.6 improves by +0.11496
PCK, interval [+0.10633, +0.12348], while MoonViT-V2 degrades by -0.01436, interval
[-0.02164, -0.00702]. Most of the MoonViT-V2 loss occurs at the `tower` to `merged` step,
and the Projector recovers only part of it. That resolved negative result fired the run's
stop condition after 12 of 18 correspondence jobs. The narrower geometric result agrees
with depth and surface normals; the headline cannot claim that every Connector preserves
every kind of correspondence.

The narrower claim, that the Projector's effect hides under the readout noise floor set by
the lossless `tower` to `merged` step, held on MoonViT-V2 and fails on the other three. The
lossless-over-Projector ratio in the matched arm is 7.9 and 22.6 for MoonViT-V2 against 0.0
to 0.4 for Kimi K2.6, Qwen3.5 and Muse Glimmer. Muse Glimmer's depth Projector moves
+0.0691 at 34.4 seed deviations while its lossless step moves -0.0151 at 0.9. Read the
earlier two-model conclusion as a MoonViT-V2 result. A later deterministic matched rerun
reads +0.06622 with paired 95% image-bootstrap interval [0.04992, 0.08338], so the strongest
Projector step resolves beyond test-image variation. A September 1 targeted pass puts the
headline comparison itself on firmer ground: all eight matched intervals comparing
`projected` with final `tower`, four Projectors by two geometry tasks, exclude zero in the
Projector's favour. The corresponding raw-arm pass resolves seven of eight; Kimi K2.6 raw
depth remains positive at +0.01960 `d1`, but its paired 95% interval
[-0.00032, +0.04021] crosses zero. Across both arms, fifteen of sixteen headline intervals
therefore resolve in the Projector's favour. See `results/README.md` and ADR-0013.

Supporting hypotheses:

1. Semantic decodability improves toward the last layers; geometry peaks earlier. Geometry
   declines before the last layer in 23 of 24 arms, the exception being unmatched DINOv2
   depth. Semantics rises to the last layer in 19 of 24 point-estimate curves. Four of the
   five exceptions are Qwen3.5; the other, kimi_k26 attention raw, falls 0.5 seed deviations
   and is noise. Paired tests separate Qwen3.5's four nominal declines under the pooled
   semantic protocol. Raw attention and raw mean decline into the last Tower layer: their
   earlier-cell advantages are
   +0.01282, interval [+0.00530, +0.02034], and +0.01556, interval
   [+0.00530, +0.02632]. Matched attention and matched mean do not resolve: +0.00171,
   interval [-0.00632, +0.00957], and +0.00154, interval [-0.00564, +0.00855]. Thus the
   late semantic decline is an unmatched-arm Qwen3.5 result, not a four-arm property. The
   raw-mean curve is covered by the passed mean pooling control; ADR-0019 still leaves the
   attention Relative Depth curve unvalidated against full tokens. In all six Towers the
   geometry peak is at or before the semantic one.
2. Rankings change by task. No Tower wins everywhere: DINOv2 takes both geometry tasks,
   while Muse Glimmer and SigLIP2 occupy the top semantic rows. Results ship as Capability
   Profiles instead of one score. The stronger readout-dependence claim is not supported:
   paired 95% image-bootstrap intervals for Muse Glimmer against SigLIP2 cross zero in all
   four semantic readout-arm columns, so neither the nominal Muse-first attention ranking
   nor the nominal SigLIP2-first mean ranking resolves on the test set. The cross-task
   semantic contrast resolves only against Muse Glimmer: Muse exceeds DINOv2 by 0.01248 on
   matched attention, with paired 95% interval [0.00342, 0.02154], while SigLIP2's 0.00838
   margin over DINOv2 has interval [-0.00085, 0.01778]. The evidence therefore supports
   "DINOv2 is below Muse Glimmer," not "DINOv2 is below the top semantic row." DINOv2's
   matched geometry wins do resolve: it leads Qwen3.5 on depth by 0.02916 `d1`, paired 95%
   interval [0.00989, 0.04835], and beats SigLIP2 on surface normals by 5.037 degrees lower
   mean angular error, interval [4.304, 5.816]. The cross-task contrast between DINOv2 and
   Muse Glimmer is therefore inferential rather than only descriptive, while no ordering is
   claimed among the unresolved semantic Towers.
3. Scale stops dominating once token count, latency, and label efficiency enter the comparison.

## Roster

| Model | Source | Role | Adapter effort |
|---|---|---|---|
| MoonViT (Kimi K2.6) | [immanuelpeter/MoonViT-K2.6](https://huggingface.co/immanuelpeter/MoonViT-K2.6), Tower, Projector and standalone modeling code republished from [moonshotai/Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6) (ADR-0018) | Multimodal Tower | Done |
| MoonViT-V2 (Kimi K3) | [immanuelpeter/MoonViT-V2](https://huggingface.co/immanuelpeter/MoonViT-V2), Tower and Projector extracted from Kimi K3 shards 95 and 96 and republished together (ADR-0015) | Multimodal Tower | Done |
| Qwen3.8-27B Tower | [immanuelpeter/Qwen3.8-27B-Vision](https://huggingface.co/immanuelpeter/Qwen3.8-27B-Vision), all 333 `model.visual.*` tensors republished from shard 1 of 18 of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (ADR-0018) | Multimodal Tower | Done |
| Muse Glimmer PE | [immanuelpeter/Muse-Glimmer-Vision](https://huggingface.co/immanuelpeter/Muse-Glimmer-Vision), 50 blocks and the Projector republished from both shards of [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) (ADR-0018) | Multimodal Tower | Done |
| DINOv2 ViT-L/14 | [facebook/dinov2-large](https://huggingface.co/facebook/dinov2-large) | Self-supervised control | Done |
| SigLIP2-SO400M | [google/siglip2-so400m-patch14-384](https://huggingface.co/google/siglip2-so400m-patch14-384) | Contrastive control | Done |

Notes:

- Inkling stays out. Its image path is a four-layer patch MLP, so there are no intermediate layers to sweep. It cannot participate in the core experiment by construction.
- SigLIP2 over PE-Core-L because Muse's Tower already represents the Perception Encoder family. The Muse-versus-PE-Core ablation is a v2 experiment. The [patch14-384](https://huggingface.co/google/siglip2-so400m-patch14-384) checkpoint is the highest-use SO400M variant; if its fixed 384² training resolution misbehaves at the canonical 448², fall back to [patch16-512](https://huggingface.co/google/siglip2-so400m-patch16-512).
- All four multimodal Towers are patch-token only, with no CLS token anywhere. That is why the shared readout is a small attention pool rather than mean pooling alone.

## Extraction matrix

- Eight evenly spaced Relative Depth points per model (layer index over total layers, from 0.125 to 1.0), plus the `merged` and `projected` Stages where they exist.
- Canonical resolution 448². Every model handles it natively; Muse PE caps at exactly 1024 patches there.
- 896² runs exist only for the geometry pillar and only for Towers with positional headroom. Muse PE is excluded from these runs.
- Stages are model-relative and every missing Stage prints N/A in every table. `merged` is the Projector input after spatial merging. It should preserve `tower` geometry until the learned mapping. Tests verify the lossless one-frame regrouping for MoonViT-V2 and Kimi K2.6.
- Adapters return every cached Stage in raster order. Qwen groups patches by 2x2 merge block, while Muse uses window order inside intermediate blocks. Both adapters undo those permutations before caching (ADR-0017).
- Full patch tokens for ImageNet-100 would cost 2.18 TB per model and 13.1 TB across the roster. That fits neither the M4 Max nor the budget in ADR-0002. Semantic probes therefore cache a 4x4 pooled grid, while geometry probes keep full tokens (ADR-0005).
- Pooled ImageNet-100 cost is measured per model, not extrapolated, because Stage count drives it more than token width. DINOv2 writes eight slices for 34.1 GB; MoonViT-V2 writes ten for 80.9 GB, since `merged` and `projected` add tokens four and seven times wider at the deepest point. Roster estimate is about 400 GB, replaced model by model as adapters land (ADR-0005).

## Probing protocol

- Global semantic tasks use the same frozen-feature attention pool with one to two million parameters. Mean pooling appears once as an ablation column.
- Head size scales with token width. Without capacity matching, a `projected` cell at 7168 trains a head four times larger than a `tower` cell at 1024. Fit and freeze a PCA reduction to a common width first, giving every cell 1.63M trainable parameters. Also run unmatched heads, and accept the matched result only if rankings and Relative Depth curves agree (ADR-0008).
- Semantic features are cached as a 4x4 spatially pooled grid, identical for every model and Stage, so the attention pool still reads spatial tokens and cross-model comparison holds (ADR-0005). Report the pooling in the protocol section of the writeup.
- Cache full patch tokens for a fixed 1,500-image semantic subset, about 25 GB per model (the earlier 5,000-image, 80 GB figure was wrong by the size of the roster; ADR-0005 corrects it). Run the semantic probe both ways on at least two Towers and confirm that model rankings and Relative Depth curves agree. This pooling control cannot be cut (ADR-0005).
- Dense tasks use the Probe3D decoder family unchanged across every cell. Geometry keeps full patch tokens, because that pillar carries the headline claim. DIODE's 25,458 training images would cost 407 GB per model at full tokens, so any training run draws a capped 4,000-image subset with a fixed seed. v1 starts on the 771-image validation split alone (ADR-0011).
- Geometry runs depth and surface normals first, one probe per Stage and Relative Depth point, plus one multilayer run per model as an internal consistency check. Depth is metric on DIODE and reaches 230 m outdoors, so the head's bin range comes from the prep manifest rather than NYU's 10 m default (ADR-0011). Capacity matching applies here too: a multiscale depth head reads 1.71M parameters at a 1024-wide Stage and 4.85M at 7168 (ADR-0010).
- Probe3D's three correspondence evaluations score features directly rather than train a probe, so they sit outside the cache-then-probe shape and get scoped once that is confirmed.
- Label budgets: 1%, 5%, 20%, 100%.
- Identical hyperparameter search everywhere: one learning-rate grid, validation-selected, three seeds, paired bootstrap intervals on headline numbers. The grid is eleven points for semantics, `[1e-5 ... 1]` after the roster run found 107 of 112 attention cells pinning at the old floor, and six points for geometry (ADR-0014).
- Do not compute cross-model cosine similarity on raw embeddings. Dimensions carry no shared meaning between models. Every cross-model claim goes through a probe.

## Datasets

- Run ImageNet-100 first. Add ImageNet-1K if probes run fast enough, Places365 for scenes, and Stanford Cars for fine-grained recognition. iNaturalist backs up Cars.
- ImageNet-100 source is [ilee0022/ImageNet100](https://huggingface.co/datasets/ilee0022/ImageNet100), 117k train / 13k validation / 5k test at native resolution, 17.4 GB. The more popular `clane9/imagenet-100` is unusable here because its images are pre-resized to 160 pixels on the short side and the canonical run is 448 square.
- Use Probe3D's decoder and protocol for geometry, on DIODE rather than NYU (ADR-0011). The validation split is 771 images for 7.7 GB over direct S3 links with no login; the training split is 25,458 images for 222 GB and gets capped at 4,000. Report indoor and outdoor separately, which DIODE labels for free. Its outdoor scenes are tripod scans, not driving footage, so the AV-free rule in ADR-0001 holds and the writeup says so. The Transfer Probe is one KITTI depth column and nothing more (ADR-0001).
- Apply programmatic transforms to a fixed subset of real photos for the Perturbation Study. Change object scale, occlusion, or motion blur one factor at a time, and record every transform value. Choose COCO or ADE20K during week 4 based on license and download size.

## Compute

Extraction runs on rented Brev GPUs in bounded bursts. Everything else runs locally on the M4 Max against cached features.

- Features get cached once per (model, Relative Depth point, Stage, resolution) into resumable safetensors shards. Downstream code never touches a GPU. Safetensors adds no measurable overhead, so shard size is the tensor size.
- Push extracted Tower checkpoints to Hugging Face as soon as parity passes. The cheapest burst instances cannot be stopped, so HF is what keeps the next burst from re-downloading the source shards. All four multimodal Towers are published and every adapter loads its release (ADR-0018).
- Budget ceiling is $150. If costs climb, drop 896² runs before dropping any model (ADR-0002).
- Prefer `hyperstack_A100_80G` at $1.62/hr, checked August 23. It has the same 80 GB as an H100 at half the price, and extraction only needs forward passes. The cheapest H100 is $3.00/hr and buys about twice the throughput, so the two are close on cost per image. Use `hyperstack_A6000` at $0.60/hr or `massedcompute_L40S` at $1.06/hr for the control models and pipeline work.
- Prefer instances with bundled disk over metered volumes. Brev meters storage near $0.10/GB/month, so a 1 TB volume for a month would cost more than half the budget.
- The M4 Max runs DINOv2 at 6.0 images per second and MoonViT-V2 at 3.0, so ImageNet-100 would take 6 to 12 hours per model locally. That is why extraction is a Brev job.
- Measured on an A100 80GB: DINOv2 at 45 images per second, MoonViT-V2 at 15.5. Set `--workers` near the vCPU count. JPEG decode limits extraction, not the Tower, and the default of 4 left the GPU at 4 percent utilization. MoonViT-V2 runs at batch size 1, because it packs sequences and without flash attention a larger batch lowers throughput and then exhausts memory (ADR-0009).
- Instance setup is `scripts/brev_setup.sh`. It pins the cu129 wheels, since PyPI serves cu130 and the 12.8 driver on these instances silently falls back to CPU, and it fails loudly if torch cannot see the GPU.

## Known risks

1. The shard-surgery risk is retired. Every Tower is now republished on its own, so extraction downloads about 10 GB from the release repositories instead of range-reading the 715 GB in the parents. The export scripts and parity tests still range-read Qwen's 0.92 GB Tower and Muse's 3.84 GB Tower from their 63.52 GB of source shards (ADR-0016, ADR-0018). MoonViT-V2 has bit-exact weight parity (ADR-0007).
2. The Kimi licenses permit republication. The Kimi K3 License grants publication and derivative works over model weights and configuration files; its only binding condition for research is shipping the copyright and permission notice, and its revenue gates start at 20 million dollars (ADR-0015). Qwen and Muse are Apache 2.0.
3. Parity tests gate everything. An extracted Tower ships only after its outputs match the Tower inside the full model within BF16 tolerance on fixed images.

## Schedule

Start August 25. Five focused weeks before autumn quarter, then a low-intensity tail during term.

- Week 1: repo scaffold, `FeatureBatch` interface, MoonViT-V2 adapter with all three Stages and a bit-exact parity test, Kimi-K3 Projector loaded (ADR-0007), adapter-owned preprocessing (ADR-0006), Qwen shard-surgery script written.
- Week 2: remaining adapters and parity tests, full extraction matrix onto Brev, feature cache complete. This is the long pole; it ends here or the whole schedule slips visibly.
- Week 3: semantic probes plus the DINOv2 control, geometry pillar starts, writeup skeleton exists and collects numbers as they land.
- Week 4: geometry pillar finishes, Perturbation Study gets built and run, KITTI Transfer Probe column lands.
- Week 5: statistics pass (seeds, bootstrap), Capability Profile tables and plots, HF uploads.
- Term tail: blog editing, Space demo if it survived cuts, preprint-upgrade decision.

If the schedule slips, cut 896² runs first, reduce the Perturbation Study to one factor next, and cut Transfer Probe variants last. Keep the controls, parity tests (ADR-0003), pooling validation (ADR-0005), and capacity-matching validation (ADR-0008).

## Shipping order

1. Bench repo with adapters, probes, parity tests, reproducible commands.
2. Extracted Tower checkpoints under your HF account with provenance model cards. Done for all four multimodal Towers (ADR-0015, ADR-0018).
3. Per-example prediction datasets from every probe run.
4. Perturbation Study dataset with transform metadata.
5. HF Space demo, if time remains.

Write the blog so it can become a preprint without changing the substance. Include related work, the full protocol, and honest limitations.
