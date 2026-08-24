# vision-tower-bench plan

Written after the grilling session on Aug 23. Language lives in CONTEXT.md, decision rationale lives in docs/adr/. This file covers execution only.

## Headline claim

The Projector preserves language-useful semantics but destroys spatial information. If projected features match raw Tower features on depth and correspondence, the claim dies and the project becomes a null result. Either outcome publishes.

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
- Stages are model-relative and every missing Stage prints N/A in every table. `merged` means after spatial merging and before the Projector's learned mapping, which in all four models is literally the Projector's input tensor. Expected result: `merged` tracks `tower` closely, so any `projected` drop on geometry attributes to the learned mapping rather than to merging itself. For MoonViT-V2 this is measured, not expected: at one frame the merge is a lossless regrouping of four Tower tokens into one, checked by `torch.equal` in `tests/test_moonvit_v2.py`.
- Cache cost: full patch tokens for ImageNet-100 would cost 2.18 TB per model and 13.1 TB across the roster, which fits neither the M4 Max nor ADR-0002. The semantic pillar therefore caches a 4x4 pooled grid and the geometry pillar keeps full tokens (ADR-0005).
- Pooled ImageNet-100 cost is measured per model, not extrapolated, because Stage count drives it more than token width. DINOv2 writes eight slices for 34.1 GB; MoonViT-V2 writes ten for 80.9 GB, since `merged` and `projected` add tokens four and seven times wider at the deepest point. Roster estimate is about 400 GB, replaced model by model as adapters land (ADR-0005).

## Probing protocol

- Global semantic tasks share one readout: a frozen-feature attention pool of one to two million parameters, identical across models and Stages. Mean pooling appears once as an ablation column.
- Semantic features are cached as a 4x4 spatially pooled grid, identical for every model and Stage, so the attention pool still reads spatial tokens and cross-model comparison holds (ADR-0005). Report the pooling in the protocol section of the writeup.
- Pooling validation: a fixed 5000-image semantic subset is also cached at full patch tokens, about 80 GB across the roster. Run the semantic probe both ways on at least two Towers and confirm that model rankings and Relative Depth curves agree. This is a control, so the cut order never reaches it (ADR-0005).
- Dense tasks use the Probe3D decoder family (depth, correspondence) unchanged across every cell. Geometry keeps full patch tokens, because that pillar carries the headline claim and Probe3D-scale data is small enough to afford them.
- Label budgets: 1%, 5%, 20%, 100%.
- Identical hyperparameter search everywhere: eight-point learning-rate grid, validation-selected, three seeds, paired bootstrap intervals on headline numbers.
- Hard rule: no cross-model cosine similarity on raw embeddings. Dimensions carry no shared meaning between models. Every cross-model claim goes through a probe.

## Datasets

- Semantics: ImageNet-100 first, ImageNet-1K if probes run fast enough, Places365 for scenes, Stanford Cars for fine-grained. iNaturalist backs up Cars.
- Geometry: Probe3D's protocol and data. Transfer Probe: KITTI depth, one column, nothing more (ADR-0001).
- Perturbation Study: programmatic transforms at graded levels applied to a fixed subset of real photos, one factor per image family. Factors: object scale, occlusion, motion blur. Record the transform parameters for every image. Source pool: COCO or ADE20K images; pick during week 4 based on license and download size.

## Compute

Extraction runs on rented Brev GPUs in bounded bursts. Everything else runs locally on the M4 Max against cached features.

- Features get cached once per (model, Relative Depth point, Stage, resolution) into resumable safetensors shards. Downstream code never touches a GPU. Safetensors adds no measurable overhead, so shard size is the tensor size.
- Push extracted Tower checkpoints to Hugging Face as soon as parity passes. The cheapest burst instances cannot be stopped, so HF is what keeps the next burst from re-downloading the source shards.
- Budget ceiling is $150. If costs climb, drop 896² runs before dropping any model (ADR-0002).
- Instance choice: `hyperstack_A100_80G` at $1.62/hr, checked Aug 23. It has the same 80 GB as an H100 at half the price, and extraction is forward-pass only. The cheapest H100 is $3.00/hr and buys about twice the throughput, so the two are close on cost per image. Use `hyperstack_A6000` at $0.60/hr or `massedcompute_L40S` at $1.06/hr for the control models and pipeline work.
- Prefer instances with bundled disk over metered volumes. Brev meters storage near $0.10/GB/month, so a 1 TB volume for a month would cost more than half the budget.
- The M4 Max runs DINOv2 at 6.0 images per second and MoonViT-V2 at 3.0, so ImageNet-100 would take 6 to 12 hours per model locally. That is why extraction is a Brev job.

## Known risks

1. Retired. No roster model needs shard surgery. Every Tower and Projector was located from shard indices and range-read headers alone, without downloading a shard (`docs/measurements/roster-shard-audit.json`). Reaching all five remaining Towers and Projectors costs about 10 GB, not the 715 GB their repositories total. MoonViT-V2 is already shipped with bit-exact parity (ADR-0007).
2. Kimi K2.6 uses a Modified MIT license. Read the modification clause before republishing extracted weights. Qwen and Muse are Apache 2.0, so those republications are safe.
3. Parity tests gate everything. An extracted Tower ships only after its outputs match the Tower inside the full model within BF16 tolerance on fixed images.

## Schedule

Start Monday, August 25. Five focused weeks before autumn quarter, then a low-intensity tail during term.

- Week 1: repo scaffold, `FeatureBatch` interface, MoonViT-V2 adapter with all three Stages and a bit-exact parity test, Kimi-K3 Projector loaded (ADR-0007), adapter-owned preprocessing (ADR-0006), Qwen shard-surgery script written.
- Week 2: remaining adapters and parity tests, full extraction matrix onto Brev, feature cache complete. This is the long pole; it ends here or the whole schedule slips visibly.
- Week 3: semantic probes plus the DINOv2 control, geometry pillar starts, writeup skeleton exists and collects numbers as they land.
- Week 4: geometry pillar finishes, Perturbation Study gets built and run, KITTI Transfer Probe column lands.
- Week 5: statistics pass (seeds, bootstrap), Capability Profile tables and plots, HF uploads.
- Term tail: blog editing, Space demo if it survived cuts, preprint-upgrade decision.

Cut order when slipping: 896² runs, then Perturbation factors down to one, then Transfer Probe variants. Never the controls, never parity tests (ADR-0003), never the pooling validation (ADR-0005).

## Shipping order

1. Bench repo with adapters, probes, parity tests, reproducible commands.
2. Extracted Tower checkpoints under your HF account with provenance model cards.
3. Per-example prediction datasets from every probe run.
4. Perturbation Study dataset with transform metadata.
5. HF Space demo. First thing cut.

Venue: a blog post written preprint-ready. Related-work section, protocol section, honest limitations. If it earns arXiv later, the upgrade is an afternoon of formatting.
