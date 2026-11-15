import os
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from json import loads
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn
from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import (
    MiniMaxM3VLConfig,
    MiniMaxM3VLVisionConfig,
)
from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
    MiniMaxM3VLMultiModalProjector,
    MiniMaxM3VLVisionModel,
)

from vtb.adapters.base import Inputs
from vtb.adapters.common import CLIP_MEAN, CLIP_STD, PatchPreprocess, default_device, depth_points, make_batch, pack_patches, raster
from vtb.feature_batch import FeatureBatch
from vtb.provenance import open_release
from vtb.shards import load_prefixed

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Checkpoint:
    name: str
    model_id: str
    repo: str
    revision: str
    release_revision: str

    @property
    def release_dir(self) -> Path:
        return ROOT / "hf" / self.model_id.split("/")[-1]

    @property
    def override(self) -> Path | None:
        override = os.environ.get(f"VTB_{self.name.upper()}_BUNDLE")
        return Path(override) if override else None

    def open(self, bundle: Path | None = None) -> Path:
        return open_release(self.model_id, self.release_revision, self.repo, self.revision, bundle=bundle or self.override, required=self == FLASH)

    @property
    def bundle(self) -> Path:
        return self.open()


M3 = Checkpoint("minimax_m3", "immanuelpeter/MiniMax-M3-Vision",
               "MiniMaxAI/MiniMax-M3", "f0e1c1e04d40177e4673a22097036854f536e9c0",
               "949099e5411c33f604f1b3bc7ed7c4c500b4b5e7")
FLASH = Checkpoint("minimax_m31_flash", "immanuelpeter/MiniMax-M3.1-Flash-Vision",
                  "MiniMaxAI/MiniMax-M3.1-Flash", "54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e",
                  "8a97b39d8005c174d901cb51eba8b5185abf7a47")
CHECKPOINTS = {item.name: item for item in (M3, FLASH)}
MODEL_ID = M3.model_id
SOURCE_REPO, SOURCE_REVISION = M3.repo, M3.revision
TOWER_PREFIX = "vision_tower.vision_model."
PROJECTOR_PREFIX = "multi_modal_projector."
MERGE_MLP_PREFIX = "patch_merge_mlp."
PATCH_SIZE = 14
TEMPORAL_PATCH_SIZE = 2
MERGE_SIZE = 2


def remap_tower(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped = {}
    for name, tensor in weights.items():
        name = name.replace("encoder.layers.", "layers.")
        name = name.replace("embeddings.patch_embedding.", "embeddings.proj.")
        remapped[name] = tensor
    return remapped


def remap_projector(linear: dict[str, torch.Tensor], merge: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped = dict(linear)
    for name, tensor in merge.items():
        remapped[name.replace("linear_", "merge_linear_")] = tensor
    return remapped


Preprocess = partial(PatchPreprocess, mean=CLIP_MEAN, std=CLIP_STD, patch_size=PATCH_SIZE, merge_size=MERGE_SIZE, temporal_patch_size=TEMPORAL_PATCH_SIZE)
collate = pack_patches


def merge_2x2(tokens: torch.Tensor) -> torch.Tensor:
    rows, count, width = tokens.shape
    side = int(count ** 0.5)
    blocks = tokens.view(rows, side // MERGE_SIZE, MERGE_SIZE, side // MERGE_SIZE, MERGE_SIZE, width)
    return blocks.permute(0, 1, 3, 2, 4, 5).reshape(rows, -1, MERGE_SIZE * MERGE_SIZE * width)


def source_json(checkpoint: Checkpoint, filename: str) -> dict:
    return loads(Path(hf_hub_download(checkpoint.repo, filename, revision=checkpoint.revision)).read_text())


def source_weights(checkpoint: Checkpoint) -> tuple[dict, dict]:
    index = source_json(checkpoint, "model.safetensors.index.json")["weight_map"]

    def read(prefix):
        selected = {name: shard for name, shard in index.items() if name.startswith(prefix)}
        if not selected:
            raise ValueError(f"no {prefix} tensors in pinned index")
        weights = load_prefixed(checkpoint.repo, sorted(set(selected.values())), prefix, revision=checkpoint.revision)
        if {prefix + name for name in weights} != set(selected):
            raise ValueError(f"incomplete {checkpoint.name} {prefix} state")
        return weights

    tower = remap_tower(read(TOWER_PREFIX))
    projector = remap_projector(read(PROJECTOR_PREFIX), read(MERGE_MLP_PREFIX))
    if len(tower) != 515 or len(projector) != 8:
        raise ValueError(f"expected 515 Tower and 8 Projector tensors, got {len(tower)}, {len(projector)}")
    return tower, projector


def load_source_parts(dtype: torch.dtype = torch.bfloat16, checkpoint: Checkpoint = M3):
    config = MiniMaxM3VLConfig.from_pretrained(checkpoint.repo, revision=checkpoint.revision)
    tower_weights, projector_weights = source_weights(checkpoint)
    tower = MiniMaxM3VLVisionModel._from_config(config.vision_config, dtype=dtype)
    tower.load_state_dict({name: tensor.to(dtype) for name, tensor in tower_weights.items()}, strict=True)
    projector = MiniMaxM3VLMultiModalProjector(config).to(dtype)
    projector.load_state_dict({name: tensor.to(dtype) for name, tensor in projector_weights.items()}, strict=True)
    return tower.eval(), projector.eval()


def projector_from_settings(settings: dict, dtype: torch.dtype) -> nn.Module:
    config = MiniMaxM3VLConfig(
        vision_config=MiniMaxM3VLVisionConfig(
            hidden_size=settings["input_size"],
            spatial_merge_size=MERGE_SIZE,
        ),
        text_config={"hidden_size": settings["output_size"]},
        projector_hidden_size=settings["hidden_size"],
    )
    projector = MiniMaxM3VLMultiModalProjector(config)
    return projector.to(dtype).eval()


def load_parts(dtype: torch.dtype, checkpoint: Checkpoint = M3, bundle: Path | None = None) -> tuple[MiniMaxM3VLVisionModel, nn.Module]:
    bundle = checkpoint.open(bundle)
    config = MiniMaxM3VLVisionConfig.from_pretrained(bundle, local_files_only=True)
    tower = MiniMaxM3VLVisionModel._from_config(config, dtype=dtype)
    tower.load_state_dict({
        name: tensor.to(dtype) for name, tensor in load_file(bundle / "model.safetensors").items()
    })
    settings = loads((bundle / "projector_config.json").read_text())
    projector = projector_from_settings(settings, dtype)
    projector.load_state_dict({
        name: tensor.to(dtype) for name, tensor in load_file(bundle / "projector.safetensors").items()
    })
    return tower.eval(), projector.to(dtype).eval()


class MiniMaxAdapter:
    checkpoint = M3
    default_resolution = 448
    stages = ("tower", "merged", "projected")
    collate = staticmethod(collate)

    def __init__(self, resolution: int | None = None, dtype: torch.dtype = torch.bfloat16, device: str | None = None, *, bundle: Path | None = None):
        self.resolution = self.default_resolution if resolution is None else resolution
        self.device = default_device() if device is None else device
        self.dtype = dtype
        tower, projector = load_parts(dtype, self.checkpoint, bundle)
        self.model_id = self.checkpoint.model_id
        self.model = tower.to(self.device)
        self.projector = projector.to(self.device)
        self.num_layers = self.model.config.num_hidden_layers

    def preprocess(self) -> PatchPreprocess:
        return Preprocess(self.resolution)

    def depth_points(self, n: int = 8) -> list[int]:
        return depth_points(self.num_layers, n)

    @torch.inference_mode()
    def extract(self, inputs: Inputs, image_ids: list[str]) -> Iterator[FeatureBatch]:
        points = self.depth_points()
        captured: dict[int, list[torch.Tensor]] = {}
        handles = []

        def capture(layer: int):
            def hook(_module, _args, output):
                tokens = output[0] if isinstance(output, tuple) else output
                captured.setdefault(layer, []).append(tokens.detach())

            return hook

        grid = inputs["grid_thw"]
        if len(grid) != len(image_ids) or not len(image_ids):
            raise ValueError("grid/image ID count differs or batch is empty")
        if not torch.all(grid == grid[0]) or grid[0, 0] != 1:
            raise ValueError("MiniMax image batches require equal still-image grids")
        counts = grid.prod(dim=1).tolist()
        if sum(counts) != len(inputs["pixel_values"]):
            raise ValueError("grid does not describe the input patch count")
        for layer in points:
            handles.append(self.model.layers[layer - 1].register_forward_hook(capture(layer)))
        outputs = []
        try:
            # The native HF model packs all inputs into one attention sequence.
            # Isolate unrelated images so cached features cannot depend on peers.
            for pixels, image_grid in zip(inputs["pixel_values"].split(counts), grid):
                outputs.append(self.model(
                    pixels.to(self.device, self.dtype),
                    image_grid.unsqueeze(0).to(self.device),
                ).last_hidden_state)
        finally:
            for handle in handles:
                handle.remove()

        rows = len(image_ids)
        width = self.model.config.hidden_size

        def packed(tokens: torch.Tensor) -> torch.Tensor:
            if tokens.ndim == 3:
                tokens = tokens.reshape(-1, tokens.shape[-1])
            return tokens.view(rows, -1, width)

        for layer in points:
            tokens = torch.cat([value.reshape(1, -1, width) for value in captured[layer]], dim=1)
            yield make_batch(self, raster(packed(tokens), rows), image_ids, "tower", layer)

        last_packed = packed(torch.cat([value.reshape(1, -1, width) for value in outputs], dim=1))
        merged = merge_2x2(raster(last_packed, rows))
        yield make_batch(self, merged, image_ids, "merged", self.num_layers)
        projected = torch.stack([self.projector(tokens) for tokens in last_packed])
        yield make_batch(self, projected, image_ids, "projected", self.num_layers)


class MiniMaxM3Adapter(MiniMaxAdapter):
    checkpoint = M3


class MiniMaxM31FlashAdapter(MiniMaxAdapter):
    checkpoint = FLASH
