# Vision Tower Bench

Adding vision to a text-only LLM means picking a Tower, a layer, and usually a Projector, which full-model VLM benchmarks never isolate. This bench probes twenty-one frozen Towers.

**[Report](https://ipeter.dev/blog/vision-tower-bench)** · **[Towers](https://huggingface.co/collections/immanuelpeter/vision-heads)** · **Paper (coming soon)**

## Install

```bash
uv add git+https://github.com/immanuel-peter/vision-tower-bench
```

The data-prep scripts need the `bench` extra (`vision-tower-bench[bench]`). This repo's `uv sync` pulls Linux `torch` and `torchvision` from the PyTorch cu129 index. That pin lives in `[tool.uv.sources]`, which does not carry over to installs from other projects, so a downstream project gets the default PyPI wheels unless it points `torch` at the index that matches its CUDA version.

```python
import vtb

tower = vtb.load("dinov2")  # any slug in vtb.SLUGS; kwargs: resolution, dtype, device
inputs = tower.collate([tower.preprocess()(image)])
for batch in tower.extract(inputs, image_ids=["img0"]):
    print(batch.stage, batch.layer_index, batch.tokens.shape)
```

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
| PE-Lang L | language-aligned control | [timm/vit_pe_lang_large_patch14_448.fb](https://huggingface.co/timm/vit_pe_lang_large_patch14_448.fb) |
| Muse Glimmer | multimodal | [immanuelpeter/Muse-Glimmer-Vision](https://huggingface.co/immanuelpeter/Muse-Glimmer-Vision) |
| Kimi K2.6 | multimodal | [immanuelpeter/MoonViT-K2.6](https://huggingface.co/immanuelpeter/MoonViT-K2.6) |
| Qwen3.8-27B | multimodal | [immanuelpeter/Qwen3.8-27B-Vision](https://huggingface.co/immanuelpeter/Qwen3.8-27B-Vision) |
| Qwen3.8 Flash-Next | multimodal | [immanuelpeter/Qwen3.8-Flash-Next-Vision](https://huggingface.co/immanuelpeter/Qwen3.8-Flash-Next-Vision) |
| Kimi K3 | multimodal | [immanuelpeter/MoonViT-V2](https://huggingface.co/immanuelpeter/MoonViT-V2) |
| Kimi K3.1 Pro | multimodal | [immanuelpeter/MoonViT-K3.1-Pro](https://huggingface.co/immanuelpeter/MoonViT-K3.1-Pro) |
| Gemma 4 31B | multimodal | [immanuelpeter/Gemma4-31B-Vision](https://huggingface.co/immanuelpeter/Gemma4-31B-Vision) |
| GLM-5.3 Flash | multimodal | [immanuelpeter/GLM-5.3-Flash-Vision](https://huggingface.co/immanuelpeter/GLM-5.3-Flash-Vision) |
| MiniMax M3.1 Flash | multimodal | [immanuelpeter/MiniMax-M3.1-Flash-Vision](https://huggingface.co/immanuelpeter/MiniMax-M3.1-Flash-Vision) |
| MiniMax M3 | multimodal | [immanuelpeter/MiniMax-M3-Vision](https://huggingface.co/immanuelpeter/MiniMax-M3-Vision) |
| Nemotron 3 Nano Omni | multimodal | [immanuelpeter/C-RADIOv4-H-3-Omni](https://huggingface.co/immanuelpeter/C-RADIOv4-H-3-Omni) |
| Nemotron 3.5 Super VL | multimodal | [immanuelpeter/C-RADIOv4-H-3.5-Super](https://huggingface.co/immanuelpeter/C-RADIOv4-H-3.5-Super) |
| Nemotron 3.5 Bolt | multimodal | [immanuelpeter/C-RADIOv4-H-3.5-Bolt](https://huggingface.co/immanuelpeter/C-RADIOv4-H-3.5-Bolt) |
| MiMo V2.6 Pro | multimodal | [immanuelpeter/MiMoViT-2.6-Pro](https://huggingface.co/immanuelpeter/MiMoViT-2.6-Pro) |
| MiMo V2.6 Flash | multimodal | [immanuelpeter/MiMoViT-2.6-Flash](https://huggingface.co/immanuelpeter/MiMoViT-2.6-Flash) |
| DeepSeek V4.1 Flash | multimodal | [immanuelpeter/DeepSeek-ViT](https://huggingface.co/immanuelpeter/DeepSeek-ViT) |
| Mistral Large 4 | multimodal | [immanuelpeter/PixtralViT-Large-4](https://huggingface.co/immanuelpeter/PixtralViT-Large-4) |

The multimodal Towers are standalone extracts of their parents. The model cards record counts and tolerances.

## Results

Findings are in the [report](https://ipeter.dev/blog/vision-tower-bench) and the paper (coming soon). The measurements behind them are kept in a private Hugging Face Dataset.

## Adding a Tower

See [docs/new-tower.md](docs/new-tower.md). Contributors and agents start from [AGENTS.md](AGENTS.md).

## Layout

```bash
vtb/          # adapters, parity checks, feature cache, probes, scoring
scripts/      # data prep, extraction matrices, figures
tests/        # parity, preprocessing, probe and merge-layout tests
hf/           # model cards and configs for the published towers
docs/         # how to add a Tower and work with results
CONTEXT.md    # project vocabulary
```

## Tests

```bash
uv run pytest -q                      # full suite, downloads Tower weights
VTB_SKIP_WEIGHTS=1 uv run pytest -q   # skip weight-dependent tests
VTB_PARITY_DEVICE=cuda uv run pytest tests/test_parity.py   # every release against its parent
```

## License

MIT covers code here, not weights. Published Towers keep their parent's terms, and each release ships its license.

- Qwen3.8-27B, Muse Glimmer, Gemma 4: Apache-2.0
- Qwen3.8 Flash-Next: Qwen Community License 1.0
- GLM-5.3 Flash, DeepSeek-ViT, MiMoViT-2.6 Pro and Flash: MIT
- Kimi K2.6, Kimi K3: Kimi License
- Kimi K3.1 Pro: Kimi K3.1-Pro License
- C-RADIOv4-H-3-Omni: NVIDIA Open Model Agreement
- C-RADIOv4-H-3.5 Super and Bolt: OpenMDW-1.1
- MiniMax M3, MiniMax M3.1 Flash: MiniMax Community License
- Mistral Large 4: Mistral Ecosystem Open-Weight License
