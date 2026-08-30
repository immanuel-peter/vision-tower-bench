# Adapters load released Towers

Publish the remaining three multimodal Towers, and have every multimodal adapter load the
released repository instead of the parent checkpoint. ADR-0015 did this for MoonViT-V2;
this extends it to the rest of the roster.

| Release | Parent | Contents |
|---|---|---|
| `immanuelpeter/Qwen3.8-27B-Vision` | `Qwen/Qwen3.8-27B` | 333 Tower tensors |
| `immanuelpeter/Muse-Glimmer-Vision` | `meta-models/Muse-Glimmer-30B` | 806 Tower tensors, 3 Projector tensors |
| `immanuelpeter/MoonViT-K2.6` | `moonshotai/Kimi-K2.6` | 329 Tower tensors, 6 Projector tensors, standalone modeling code |

Qwen ships one weight file because its merger lives inside `Qwen3_5VisionModel`; the
`projected` Stage comes out of the same forward pass as `pooler_output`. The other two keep
`projector.safetensors` and a `projector_config.json` beside the Tower, for the reason
ADR-0015 gives: the vision config does not describe the Projector.

## Why adapters stop reading the parents

ADR-0016 made range requests cheap enough to probe from a 63.52 GB checkpoint. It did not
make them dependable. A range read needs the parent repository to serve arbitrary byte
spans, and doing that at roster scale produced 429s and reset sockets that had to be
answered with span coalescing, a 256 MB cap, and retries on three separate failure classes.
None of that is needed to download a 0.92 GB file that exists on its own.

The compute section of PLAN.md already assumed releases would keep each new burst instance
from downloading the source shards again. The releases existed, but the adapters still
loaded the parent repositories.

Kimi K2.6 gains the most. Its adapter previously loaded weights from `exolabs/Kimi-K2.6-vision`
and pulled architecture code out of `moonshotai/Kimi-K2.6` through
`get_class_from_dynamic_module`, and that code targets transformers 4.x and imports one
helper 5.x dropped. The adapter carried a monkeypatch for it. The released repository
vendors a trimmed `modeling_moonvit.py` with no such import, so the adapter now loads
through `AutoModel` and the monkeypatch is gone from the probing path.

## Provenance stays testable in two layers

Each adapter keeps the constants naming its parent for the exporter and the tests.

`tests/test_parity.py` compares what the adapter actually loads against the parent
checkpoint, tensor by tensor. Because the adapter now loads the release, this is a stronger
test than before: it ties the repository the bench probes back to the checkpoint it came
out of, on every run.

`tests/test_release_parity.py` compares the working-tree bundle under `hf/` against a
pinned parent revision, both as weights and as a fixed-image forward pass. Its source side
comes from the export scripts rather than the adapters, so changing the adapter load target
does not change both sides of the comparison. `scripts/export_moonvit_k26.py` is where the
transformers 4.x monkeypatch now lives, because that script is the only code that still
reads Moonshot's implementation.

## Results recorded before this change name the parents

Every cached Stage and every probe cell carries the adapter's `model_id`. The roster run in
`results/` therefore names `Qwen/Qwen3.8-27B`, `meta-models/Muse-Glimmer-30B` and
`exolabs/Kimi-K2.6-vision`, while anything extracted from now on names the release. The
weights behind both names are the same tensors, which is what `tests/test_parity.py`
asserts, so the two are comparable. Read a mixed table by the roster key rather than by the
repository string, and say which is which in the writeup.

## Licenses

Qwen3.8-27B and Muse Glimmer are Apache 2.0, so each release ships the upstream `LICENSE`
and Muse's `USAGE_POLICY.md`. MoonViT-K2.6 ships the Kimi K2.6 License and the upstream
third-party notices under the license interpretation recorded in ADR-0015 for Kimi K3.
