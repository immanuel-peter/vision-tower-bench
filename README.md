# Vision Tower Bench

Adding vision to a text-only LLM means picking a Tower, a layer, and usually a Projector, which full-model VLM benchmarks never isolate. This bench probes six frozen Towers. Instead of the Projector tossing away spatial information, 15 of 16 paired intervals across two geometry tasks and two probe capacities went the other way.

**[Report](https://ipeter.dev/blog/vision-tower-bench)** · **[Towers](https://huggingface.co/collections/immanuelpeter/vision-towers)** · **Paper (coming soon)**

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
| SigLIP2-SO400M | contrastive control | [google/siglip2-so400m-patch14-384](https://huggingface.co/google/siglip2-so400m-patch14-384) |
| Muse Glimmer PE | multimodal | [immanuelpeter/Muse-Glimmer-Vision](https://huggingface.co/immanuelpeter/Muse-Glimmer-Vision) |
| Kimi K2.6 | multimodal | [immanuelpeter/MoonViT-K2.6](https://huggingface.co/immanuelpeter/MoonViT-K2.6) |
| Qwen3.8-27B | multimodal | [immanuelpeter/Qwen3.8-27B-Vision](https://huggingface.co/immanuelpeter/Qwen3.8-27B-Vision) |
| Kimi K3 | multimodal | [immanuelpeter/MoonViT-V2](https://huggingface.co/immanuelpeter/MoonViT-V2) |

The four multimodal Towers are standalone extracts of their parents. The model cards record counts and tolerances.

## Results

Source files live in `results/`:

| Path | Contents |
|---|---|
| `results/*_probe_*.json` | recognition cells: 2 readouts, 2 capacity arms |
| `results/*_geometry_*.json` | depth and surface normals on DIODE |
| `results/correspondence/` | NAVI, ScanNet and SPair scores per stage |
| `results/bootstrap/` | paired intervals, with the per-image records behind them |
| `results/label-budget/` | recognition at 1%, 5%, 20% and 100% of labels |
| `results/perturbation/` | occlusion, fixed readout evaluated on perturbed features |
| `results/kitti/` | the transfer column |
| `results/README.md` | protocol notes and run-by-run findings |

## Reproducing the figures

```bash
uv sync --group figures
uv run --group figures python scripts/figures.py
```

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
uv run pytest -q                      # full suite, downloads about 6 GB of weights
VTB_SKIP_WEIGHTS=1 uv run pytest -q   # 73 tests, no downloads
```

## License

MIT covers code and results here, not weights. Published Towers keep parent terms. Apache-2.0 for Qwen3.8 and Muse Glimmer, Kimi License for both Kimi Towers. Each release ships its license and third-party notices.
