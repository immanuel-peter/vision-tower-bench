"""Qwen3.8 27B and Flash-Next: the published Towers against their pinned parents.

Flash-Next's parent declares ``qwen4_exp``, so its features are also checked
against that native class, independent of the ``qwen3_5_vision`` rewrite the
bench and the release use.
"""

from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel

from vtb.adapters.qwen3_5 import (
    FLASH_NEXT,
    MERGE_SIZE,
    PATCH_SIZE,
    QWEN27B,
    TOWER_TENSORS,
    Checkpoint,
    Qwen3_5Adapter,
    Qwen38FlashNextAdapter,
    load_source_tower,
    load_tower,
    source_json,
    source_weights,
)
from vtb.parity import (
    bit_exact,
    compare,
    fixed_images,
    reference_raster,
    release_against_source,
    weights_match,
)

ADAPTERS = {QWEN27B: Qwen3_5Adapter, FLASH_NEXT: Qwen38FlashNextAdapter}


def release_dir(checkpoint: Checkpoint) -> Path:
    if (checkpoint.bundle / "model.safetensors").is_file():
        return checkpoint.bundle
    return Path(snapshot_download(checkpoint.model_id))


def native_parent(
    checkpoint: Checkpoint,
    adapter,
    weights: dict,
    dtype: torch.dtype,
    resolutions=(96, 448),
) -> dict:
    config = Qwen4ExpVisionConfig(
        **source_json(checkpoint, "config.json")["vision_config"]
    )
    reference = Qwen4ExpVisionModel._from_config(config, dtype=dtype).eval()
    reference.load_state_dict(weights, strict=True)
    reference = reference.to(adapter.device)
    images, cells = fixed_images(), {}
    for resolution in resolutions:
        adapter.resolution = resolution
        inputs = adapter.collate([adapter.preprocess()(image) for image in images])
        captured, handles = {}, []
        for layer in adapter.depth_points():
            handles.append(
                reference.blocks[layer - 1].register_forward_hook(
                    lambda _m, _a, output, layer=layer: captured.update(
                        {layer: output.detach()}
                    )
                )
            )
        try:
            with torch.inference_mode():
                expected = reference(
                    inputs["hidden_states"].to(adapter.device, dtype),
                    inputs["grid_thw"].to(adapter.device),
                )
        finally:
            for handle in handles:
                handle.remove()
        side = resolution // PATCH_SIZE
        for batch in adapter.extract(inputs, [str(i) for i in range(len(images))]):
            if batch.stage == "tower":
                target = reference_raster(
                    captured[batch.layer_index],
                    torch.tensor([[1, side, side]] * len(images)),
                )
            elif batch.stage == "merged":
                target = expected.last_hidden_state.reshape(
                    len(images), -1, config.hidden_size * MERGE_SIZE**2
                )
            else:
                target = expected.pooler_output.reshape(
                    len(images), -1, checkpoint.out_hidden_size
                )
            cells[f"{resolution}:{batch.stage}:{batch.layer_index}"] = compare(
                batch.tokens, target
            )
    return cells


def validate(
    checkpoint: Checkpoint, device: str = "cpu", dtype: torch.dtype = torch.float32
) -> dict:
    weights = source_weights(checkpoint)
    release = load_file(release_dir(checkpoint) / "model.safetensors")
    report = {"weights": {"tower": weights_match(release, weights, TOWER_TENSORS)}}
    adapter = ADAPTERS[checkpoint](device=device, dtype=dtype)
    report["forward"] = release_against_source(
        adapter, {"model": load_source_tower(checkpoint, dtype)}
    )
    if checkpoint == FLASH_NEXT:
        adapter.model = load_tower(dtype, checkpoint).to(device)
        report["forward"].update(
            {
                f"native:{key}": cell
                for key, cell in native_parent(
                    checkpoint, adapter, weights, dtype
                ).items()
            }
        )
    return {**report, "passed": bit_exact(report)}
