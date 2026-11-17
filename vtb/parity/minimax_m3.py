"""MiniMax M3 and M3.1 Flash: the published Tower and Projector against native HF parent classes.

The parent repositories ship no modeling code, so the reference loads source tensors into
transformers' own MiniMax classes. Tensor renaming, patch packing and the raster scatter
are written out here instead of reusing the adapter's helpers.
"""

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers.models.minimax_m3_vl import modeling_minimax_m3_vl as parent_code
from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import (
    MiniMaxM3VLConfig,
)

from vtb.adapters import minimax_m3 as mm
from vtb.parity import bit_exact, compare, weights_match
from vtb.shards import load_prefixed

ADAPTERS = {mm.M3: mm.MiniMaxM3Adapter, mm.FLASH: mm.MiniMaxM31FlashAdapter}
TOWER_WIDTH, PROJECTED_WIDTH = 1280, 6144


def parent_weights(checkpoint) -> tuple[dict, dict]:
    index = mm.source_json(checkpoint, "model.safetensors.index.json")["weight_map"]
    raw = {}
    for prefix in (mm.TOWER_PREFIX, mm.PROJECTOR_PREFIX, mm.MERGE_MLP_PREFIX):
        shards = sorted(
            {shard for name, shard in index.items() if name.startswith(prefix)}
        )
        raw[prefix] = load_prefixed(
            checkpoint.repo, shards, prefix, revision=checkpoint.revision
        )
    tower = {
        name.replace("encoder.layers.", "layers.").replace(
            "embeddings.patch_embedding.", "embeddings.proj."
        ): tensor
        for name, tensor in raw[mm.TOWER_PREFIX].items()
    }
    projector = dict(raw[mm.PROJECTOR_PREFIX])
    projector.update(
        {
            name.replace("linear_", "merge_linear_"): tensor
            for name, tensor in raw[mm.MERGE_MLP_PREFIX].items()
        }
    )
    return tower, projector


def packed_patches(pixels: np.ndarray, side: int) -> torch.Tensor:
    array = torch.from_numpy(pixels.copy()).permute(2, 0, 1).float() / 255
    array = (array - torch.tensor(mm.CLIP_MEAN)[:, None, None]) / torch.tensor(
        mm.CLIP_STD
    )[:, None, None]
    patch = mm.PATCH_SIZE
    rows = []
    for y in range(0, side, 2):
        for x in range(0, side, 2):
            for dy in range(2):
                for dx in range(2):
                    block = array[
                        :,
                        (y + dy) * patch : (y + dy + 1) * patch,
                        (x + dx) * patch : (x + dx + 1) * patch,
                    ]
                    rows.append(block[:, None].expand(-1, 2, -1, -1).reshape(-1))
    return torch.stack(rows)


def raster(packed: torch.Tensor, side: int) -> torch.Tensor:
    grid = torch.empty(1, side, side, packed.shape[-1])
    at = 0
    for y in range(0, side, 2):
        for x in range(0, side, 2):
            for dy in range(2):
                for dx in range(2):
                    grid[0, y + dy, x + dx] = packed[at]
                    at += 1
    return grid


def validate(
    checkpoint,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    resolutions: tuple[int, ...] = (56, 448),
) -> dict:
    tower_weights, projector_weights = parent_weights(checkpoint)
    bundle = checkpoint.bundle
    report = {
        "weights": {
            "tower": weights_match(
                load_file(bundle / "model.safetensors"), tower_weights, mm.TOWER_TENSORS
            ),
            "projector": weights_match(
                load_file(bundle / "projector.safetensors"),
                projector_weights,
                mm.PROJECTOR_TENSORS,
            ),
        },
        "forward": {},
        "image_size_invariant": True,
    }

    config = MiniMaxM3VLConfig.from_pretrained(
        checkpoint.repo, revision=checkpoint.revision
    )
    reference = parent_code.MiniMaxM3VLVisionModel._from_config(
        config.vision_config, dtype=dtype
    )
    reference.load_state_dict(tower_weights, strict=True)
    reference = reference.to(device).eval()
    projector = parent_code.MiniMaxM3VLMultiModalProjector(config).to(dtype=dtype)
    projector.load_state_dict(projector_weights, strict=True)
    projector = projector.to(device).eval()
    adapter = ADAPTERS[checkpoint](device=device, dtype=dtype)

    for resolution in resolutions:
        pixels = np.random.default_rng(21).integers(
            0, 256, (resolution, resolution, 3), dtype=np.uint8
        )
        side = resolution // mm.PATCH_SIZE
        values = packed_patches(pixels, side)
        report["forward"][f"{resolution}:preprocess"] = compare(
            mm.Preprocess(resolution)(Image.fromarray(pixels)), values
        )
        grid = torch.tensor([[1, side, side]], device=device)
        captures = {}
        handles = [
            reference.layers[i - 1].register_forward_hook(
                lambda _m, _i, out, i=i: captures.__setitem__(i, out.detach())
            )
            for i in adapter.depth_points()
        ]
        try:
            with torch.inference_mode():
                output = reference(values.to(device, dtype), grid).last_hidden_state
                expected_projected = (
                    projector(output.reshape(-1, TOWER_WIDTH))
                    .reshape(1, -1, PROJECTED_WIDTH)
                    .cpu()
                )
                nominal = reference.config.image_size
                reference.config.image_size = 672 if nominal != 672 else 2016
                altered = reference(values.to(device, dtype), grid).last_hidden_state
                reference.config.image_size = nominal
        finally:
            for handle in handles:
                handle.remove()
        report["image_size_invariant"] &= torch.equal(output, altered)
        adapter.resolution = resolution
        for batch in adapter.extract(mm.collate([values]), ["fixed-21"]):
            if batch.stage == "projected":
                expected = expected_projected
            else:
                grid_tokens = raster(
                    captures[batch.layer_index].reshape(-1, TOWER_WIDTH).cpu(), side
                )
                if batch.stage == "tower":
                    expected = grid_tokens.reshape(1, -1, TOWER_WIDTH)
                else:
                    expected = torch.stack(
                        [
                            grid_tokens[:, y : y + 2, x : x + 2].reshape(
                                1, 4 * TOWER_WIDTH
                            )
                            for y in range(0, side, 2)
                            for x in range(0, side, 2)
                        ],
                        dim=1,
                    )
            report["forward"][f"{resolution}:{batch.stage}:{batch.layer_index}"] = (
                compare(batch.tokens, expected.to(batch.tokens.dtype))
            )
    return {**report, "passed": bit_exact(report) and report["image_size_invariant"]}
