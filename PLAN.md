# vision-tower-bench plan

Written after the August 23 grilling session. CONTEXT.md defines project terms, and the
ADRs record why each decision was made. This file covers execution.

## Headline claim

The Projector preserves semantics useful to the language model but destroys spatial
information. If projected features match raw Tower features on depth and correspondence,
the claim fails. That null result is still publishable.

Supporting hypotheses:

1. Semantic decodability improves toward the last layers; geometry peaks earlier.
2. Rankings change by task. No Tower wins everywhere, which is why results ship as Capability Profiles instead of one score.
3. Scale stops dominating once token count, latency, and label efficiency enter the comparison.

## Roster

| Model | Source | Role | Adapter effort |
|---|---|---|---|
| MoonViT (Kimi K2.6) | [exolabs/Kimi-K2.6-vision](https://huggingface.co/exolabs/Kimi-K2.6-vision) weights, architecture adapted from [moonshotai/Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6) vision code; Tower and Projector in 2 shards, 0.94 GB | Multimodal Tower | Medium |
| MoonViT-V2 (Kimi K3) | [AI4Industry/MoonViT-V2](https://huggingface.co/AI4Industry/MoonViT-V2), standalone modeling code; Projector from one 0.09 GB Kimi K3 shard (ADR-0007) | Multimodal Tower | Done |
| Qwen3.8-27B Tower | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), all 333 `model.visual.*` tensors in shard 1 of 18, 0.92 GB of weights | Multimodal Tower | Low-medium |
| Muse Glimmer PE | [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) via transformers `MuseGlimmerVisionModel`; 50 blocks, 3.84 GB of vision weights spread across both shards | Multimodal Tower | Low-medium |
| DINOv2 ViT-L/14 | [facebook/dinov2-large](https://huggingface.co/facebook/dinov2-large) | Self-supervised control | Trivial |
| SigLIP2-SO400M | [google/siglip2-so400m-patch14-384](https://huggingface.co/google/siglip2-so400m-patch14-384) | Contrastive control | Trivial |

Notes:

- Inkling stays out. Its image path is a four-layer patch MLP, so there are no intermediate layers to sweep. It cannot participate in the core experiment by construction.
- SigLIP2 over PE-Core-L because Muse's Tower already represents the Perception Encoder family. The Muse-versus-PE-Core ablation is a v2 experiment. The [patch14-384](https://huggingface.co/google/siglip2-so400m-patch14-384) checkpoint is the highest-use SO400M variant; if its fixed 384² training resolution misbehaves at the canonical 448², fall back to [patch16-512](https://huggingface.co/google/siglip2-so400m-patch16-512).
- All four multimodal Towers are patch-token only, with no CLS token anywhere. That is why the shared readout is a small attention pool rather than mean pooling alone.

## Extraction matrix

- Eight evenly spaced Relative Depth points per model (layer index over total layers, from 0.125 to 1.0), plus the `merged` and `projected` Stages where they exist.
- Canonical resolution 448². Every model handles it natively; Muse PE caps at exactly 1024 patches there.
- 896² runs exist only for the geometry pillar and only for Towers with positional headroom. Muse PE is excluded from these runs.
- Stages are model-relative and every missing Stage prints N/A in every table. `merged` means after spatial merging and before the Projector's learned mapping. It is the Projector's input tensor in all four models. The expected result is that `merged` tracks `tower`, which would attribute any geometry loss in `projected` to the learned mapping. MoonViT-V2's one-frame merge is a lossless regrouping of four Tower tokens, checked by `torch.equal` in `tests/test_moonvit_v2.py`.
- Full patch tokens for ImageNet-100 would cost 2.18 TB per model and 13.1 TB across the roster. That fits neither the M4 Max nor the budget in ADR-0002. Semantic probes therefore cache a 4x4 pooled grid, while geometry probes keep full tokens (ADR-0005).
- Pooled ImageNet-100 cost is measured per model, not extrapolated, because Stage count drives it more than token width. DINOv2 writes eight slices for 34.1 GB; MoonViT-V2 writes ten for 80.9 GB, since `merged` and `projected` add tokens four and seven times wider at the deepest point. Roster estimate is about 400 GB, replaced model by model as adapters land (ADR-0005).

## Probing protocol

- Global semantic tasks use the same frozen-feature attention pool with one to two million parameters. Mean pooling appears once as an ablation column.
- Head size scales with token width. Without capacity matching, a `projected` cell at 7168 trains a head four times larger than a `tower` cell at 1024. Fit and freeze a PCA reduction to a common width first, giving every cell 1.63M trainable parameters. Also run unmatched heads, and accept the matched result only if rankings and Relative Depth curves agree (ADR-0008).
- Semantic features are cached as a 4x4 spatially pooled grid, identical for every model and Stage, so the attention pool still reads spatial tokens and cross-model comparison holds (ADR-0005). Report the pooling in the protocol section of the writeup.
- Cache full patch tokens for a fixed 5000-image semantic subset, about 80 GB across the roster. Run the semantic probe both ways on at least two Towers and confirm that model rankings and Relative Depth curves agree. This pooling control cannot be cut (ADR-0005).
- Dense tasks use the Probe3D decoder family unchanged across every cell. Geometry keeps full patch tokens, because that pillar carries the headline claim. DIODE's 25,458 training images would cost 407 GB per model at full tokens, so any training run draws a capped 4,000-image subset with a fixed seed. v1 starts on the 771-image validation split alone (ADR-0011).
- Geometry runs depth and surface normals first, one probe per Stage and Relative Depth point, plus one multilayer run per model as an internal consistency check. Depth is metric on DIODE and reaches 230 m outdoors, so the head's bin range comes from the prep manifest rather than NYU's 10 m default (ADR-0011). Capacity matching applies here too: a multiscale depth head reads 1.71M parameters at a 1024-wide Stage and 4.85M at 7168 (ADR-0010).
- Probe3D's three correspondence evaluations score features directly rather than train a probe, so they sit outside the cache-then-probe shape and get scoped once that is confirmed.
- Label budgets: 1%, 5%, 20%, 100%.
- Identical hyperparameter search everywhere: eight-point learning-rate grid, validation-selected, three seeds, paired bootstrap intervals on headline numbers.
- Do not compute cross-model cosine similarity on raw embeddings. Dimensions carry no shared meaning between models. Every cross-model claim goes through a probe.

## Datasets

- Run ImageNet-100 first. Add ImageNet-1K if probes run fast enough, Places365 for scenes, and Stanford Cars for fine-grained recognition. iNaturalist backs up Cars.
- ImageNet-100 source is [ilee0022/ImageNet100](https://huggingface.co/datasets/ilee0022/ImageNet100), 117k train / 13k validation / 5k test at native resolution, 17.4 GB. The more popular `clane9/imagenet-100` is unusable here because its images are pre-resized to 160 pixels on the short side and the canonical run is 448 square.
- Use Probe3D's decoder and protocol for geometry, on DIODE rather than NYU (ADR-0011). The validation split is 771 images for 7.7 GB over direct S3 links with no login; the training split is 25,458 images for 222 GB and gets capped at 4,000. Report indoor and outdoor separately, which DIODE labels for free. Its outdoor scenes are tripod scans, not driving footage, so the AV-free rule in ADR-0001 holds and the writeup says so. The Transfer Probe is one KITTI depth column and nothing more (ADR-0001).
- Apply programmatic transforms to a fixed subset of real photos for the Perturbation Study. Change object scale, occlusion, or motion blur one factor at a time, and record every transform value. Choose COCO or ADE20K during week 4 based on license and download size.

## Compute

Extraction runs on rented Brev GPUs in bounded bursts. Everything else runs locally on the M4 Max against cached features.

- Features get cached once per (model, Relative Depth point, Stage, resolution) into resumable safetensors shards. Downstream code never touches a GPU. Safetensors adds no measurable overhead, so shard size is the tensor size.
- Push extracted Tower checkpoints to Hugging Face as soon as parity passes. The cheapest burst instances cannot be stopped, so HF is what keeps the next burst from re-downloading the source shards.
- Budget ceiling is $150. If costs climb, drop 896² runs before dropping any model (ADR-0002).
- Prefer `hyperstack_A100_80G` at $1.62/hr, checked August 23. It has the same 80 GB as an H100 at half the price, and extraction only needs forward passes. The cheapest H100 is $3.00/hr and buys about twice the throughput, so the two are close on cost per image. Use `hyperstack_A6000` at $0.60/hr or `massedcompute_L40S` at $1.06/hr for the control models and pipeline work.
- Prefer instances with bundled disk over metered volumes. Brev meters storage near $0.10/GB/month, so a 1 TB volume for a month would cost more than half the budget.
- The M4 Max runs DINOv2 at 6.0 images per second and MoonViT-V2 at 3.0, so ImageNet-100 would take 6 to 12 hours per model locally. That is why extraction is a Brev job.
- Measured on an A100 80GB: DINOv2 at 45 images per second, MoonViT-V2 at 15.5. Set `--workers` near the vCPU count. JPEG decode limits extraction, not the Tower, and the default of 4 left the GPU at 4 percent utilization. MoonViT-V2 runs at batch size 1, because it packs sequences and without flash attention a larger batch lowers throughput and then exhausts memory (ADR-0009).
- Instance setup is `scripts/brev_setup.sh`. It pins the cu129 wheels, since PyPI serves cu130 and the 12.8 driver on these instances silently falls back to CPU, and it fails loudly if torch cannot see the GPU.

## Known risks

1. The shard-surgery risk is retired. Shard indices and range-read headers located every Tower and Projector without downloading a shard (`docs/measurements/roster-shard-audit.json`). Reaching the five remaining Towers and Projectors costs about 10 GB, not the 715 GB in their parent repositories. MoonViT-V2 already has bit-exact weight parity (ADR-0007).
2. Kimi K2.6 uses a Modified MIT license. Read the modification clause before republishing extracted weights. Qwen and Muse are Apache 2.0, so those republications are safe.
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
2. Extracted Tower checkpoints under your HF account with provenance model cards.
3. Per-example prediction datasets from every probe run.
4. Perturbation Study dataset with transform metadata.
5. HF Space demo, if time remains.

Write the blog so it can become a preprint without changing the substance. Include related work, the full protocol, and honest limitations.
