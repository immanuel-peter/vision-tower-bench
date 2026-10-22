# Vision Tower Bench

Adding vision to a text-only LLM means picking a Tower, a layer, and usually a Projector, which full-model VLM benchmarks never isolate. This bench probes sixteen frozen Towers.

**[Report](https://ipeter.dev/blog/vision-tower-bench)** · **[Towers](https://huggingface.co/collections/immanuelpeter/vision-heads)** · **Paper (coming soon)**

## What it measures

Four capability axes:

| Axis | Question | Data |
|---|---|---|
| Recognition | Can a probe identify the object, with many labels and with almost none? | ImageNet-100 |
| Geometry | Can it recover depth and surface orientation from one image? | DIODE |
| Cross-view | Can it find the same physical point in a second photograph? | NAVI, ScanNet, SPair-71k |
| Occlusion | How much recognition survives when half the image is hidden? | ImageNet-100, perturbed |

## Roster

| Tower | Role | Source |
|---|---|---|
| DINOv2 ViT-L/14 | self-supervised control | [facebook/dinov2-large](https://huggingface.co/facebook/dinov2-large) |
| DINOv3 ViT-L/16 | self-supervised control | [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) |
| SigLIP2-SO400M | contrastive control | [google/siglip2-so400m-patch14-384](https://huggingface.co/google/siglip2-so400m-patch14-384) |
| Muse Glimmer | multimodal | [immanuelpeter/Muse-Glimmer-Vision](https://huggingface.co/immanuelpeter/Muse-Glimmer-Vision) |
| Kimi K2.6 | multimodal | [immanuelpeter/MoonViT-K2.6](https://huggingface.co/immanuelpeter/MoonViT-K2.6) |
| Qwen3.8-27B | multimodal | [immanuelpeter/Qwen3.8-27B-Vision](https://huggingface.co/immanuelpeter/Qwen3.8-27B-Vision) |
| Kimi K3 | multimodal | [immanuelpeter/MoonViT-V2](https://huggingface.co/immanuelpeter/MoonViT-V2) |
| Gemma 4 31B | multimodal | [immanuelpeter/Gemma4-31B-Vision](https://huggingface.co/immanuelpeter/Gemma4-31B-Vision) |
| GLM-5.3 Flash | multimodal | [immanuelpeter/GLM-5.3-Flash-Vision](https://huggingface.co/immanuelpeter/GLM-5.3-Flash-Vision) |
| MiniMax M3 | multimodal | [immanuelpeter/MiniMax-M3-Vision](https://huggingface.co/immanuelpeter/MiniMax-M3-Vision) |
| Nemotron 3 Nano Omni | multimodal | [immanuelpeter/C-RADIOv4-H-3-Omni](https://huggingface.co/immanuelpeter/C-RADIOv4-H-3-Omni) |
| Nemotron 3.5 Super VL | multimodal | [immanuelpeter/C-RADIOv4-H-3.5-Super](https://huggingface.co/immanuelpeter/C-RADIOv4-H-3.5-Super) |
| Nemotron 3.5 Bolt | multimodal | [immanuelpeter/C-RADIOv4-H-3.5-Bolt](https://huggingface.co/immanuelpeter/C-RADIOv4-H-3.5-Bolt) |
| MiMo V2.6 Pro | multimodal | [immanuelpeter/MiMoViT-2.6-Pro](https://huggingface.co/immanuelpeter/MiMoViT-2.6-Pro) |
| MiMo V2.6 Flash | multimodal | [immanuelpeter/MiMoViT-2.6-Flash](https://huggingface.co/immanuelpeter/MiMoViT-2.6-Flash) |
| DeepSeek V4.1 Flash | multimodal | [immanuelpeter/DeepSeek-ViT](https://huggingface.co/immanuelpeter/DeepSeek-ViT) |

The multimodal Towers are standalone extracts of their parents. The model cards record counts and tolerances.

## Results

The [results index](results/README.md) links each experiment, its evidence, and
its correction status. Recognition and geometry matrices are grouped by task;
diagnostics and historical run notes are separate. All measurements remain in Git.

The [machine-readable index](results/index.json) records the v1 six-Tower,
v2 eleven-Tower and v3 five-Tower cohorts and selects corrected deepest-stage geometry sources.
Full layer sweeps and paired-bootstrap reruns remain separate records.

## Reproducing the figures

```bash
uv sync --group figures
uv run --group figures python scripts/figures.py
```

This renders all sixteen Towers. The label-budget figure and the Capability Profile's semantic
axes use channel-standardized readouts, and MiMo 2.6 is left out of the occlusion figure. See the
[source-selection notes](results/README.md#choose-the-right-source) for the
scope of the geometry corrections and the historical layer sweeps.

## Evaluating a new Tower

**1. Wrap it.** Return patch tokens at eight even relative depths, plus merged and projected if you have them. Include your own preprocessing.

**2. Meet three constraints.** Patch tokens only, no class token. Raster order, so undo any merge-block or window grouping before caching. Square token grid, the pooling and dense decoders assume one.

**3. Prove it matches.** A Tower pulled from a larger checkpoint must match that checkpoint within BF16 tolerance on fixed images.

**4. Extract once.** Cache features per model, Stage, depth and resolution. Probes read the cache after that.

**5. Run the matrices.** Same learning-rate grid, three seeds, validation picks the rate.

## Layout

```bash
vtb/          # adapters, feature cache, probes, geometry and correspondence scoring
scripts/      # export, extraction matrices, bootstraps, figures
tests/        # parity, preprocessing, probe and merge-layout tests
results/      # published measurements
hf/           # model cards and configs for the published towers
CONTEXT.md    # project vocabulary
```

## Tests

```bash
uv run pytest -q                      # full suite, downloads Tower weights
VTB_SKIP_WEIGHTS=1 uv run pytest -q   # 135 tests, no downloads
```

## License

MIT covers code and results here, not weights. Published Towers keep parent terms. Apache-2.0 for Qwen3.8, Muse Glimmer, and Gemma 4. MIT for GLM-5.3 Flash and DeepSeek-ViT. Kimi License for both Kimi Towers. NVIDIA Open Model Agreement for C-RADIOv4-H-3-Omni. OpenMDW-1.1 for C-RADIOv4-H-3.5 Super and Bolt. MIT for MiMoViT-2.6 Pro and Flash. MiniMax Community License for MiniMax M3. Each release ships its license.
