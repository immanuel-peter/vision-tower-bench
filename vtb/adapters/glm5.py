from collections.abc import Iterator
from functools import partial

import torch
from transformers import AutoConfig, AutoModel, Glm5NextVisionModel

from vtb.adapters.base import Inputs
from vtb.adapters.common import CLIP_MEAN, CLIP_STD, PatchPreprocess, default_device, depth_points, make_batch, pack_patches, raster
from vtb.feature_batch import FeatureBatch
from vtb.provenance import open_release
from vtb.shards import load_prefixed

MODEL_ID = "immanuelpeter/GLM-5.3-Flash-Vision"
REVISION = "44a87fd4e6e31f5722a375238b7aedce3e0ad2db"

SOURCE_REPO = "zai-org/GLM-5.3-Flash"
SOURCE_REVISION = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
VISION_SHARD = "model-00062-of-00062.safetensors"
VISION_PREFIX = "model.visual."
PATCH_SIZE = 14
TEMPORAL_PATCH_SIZE = 2
MERGE_SIZE = 2

Preprocess = partial(PatchPreprocess, mean=CLIP_MEAN, std=CLIP_STD, patch_size=PATCH_SIZE, merge_size=MERGE_SIZE, temporal_patch_size=TEMPORAL_PATCH_SIZE)
collate = partial(pack_patches, key="hidden_states")


def load_source_tower(dtype: torch.dtype = torch.bfloat16) -> Glm5NextVisionModel:
    config = AutoConfig.from_pretrained(SOURCE_REPO, revision=SOURCE_REVISION)
    tower = Glm5NextVisionModel._from_config(config.vision_config, dtype=dtype)
    weights = load_prefixed(
        SOURCE_REPO, [VISION_SHARD], VISION_PREFIX, revision=SOURCE_REVISION
    )
    tower.load_state_dict({name: tensor.to(dtype) for name, tensor in weights.items()})
    return tower.eval()


def load_tower(dtype: torch.dtype) -> Glm5NextVisionModel:
    bundle = open_release(MODEL_ID, REVISION, SOURCE_REPO, SOURCE_REVISION)
    return AutoModel.from_pretrained(bundle, dtype=dtype, local_files_only=True).eval()


class GLM5Adapter:
    model_id = MODEL_ID
    default_resolution = 448
    stages = ("tower", "merged", "projected")
    collate = staticmethod(collate)

    def __init__(self, resolution: int | None = None, dtype: torch.dtype = torch.bfloat16, device: str | None = None):
        self.resolution = self.default_resolution if resolution is None else resolution
        self.device = default_device() if device is None else device
        self.dtype = dtype
        self.model = load_tower(dtype).to(self.device)
        self.num_layers = self.model.config.depth

    def preprocess(self) -> PatchPreprocess:
        return Preprocess(self.resolution)

    def depth_points(self, n: int = 8) -> list[int]:
        return depth_points(self.num_layers, n)

    @torch.inference_mode()
    def extract(self, inputs: Inputs, image_ids: list[str]) -> Iterator[FeatureBatch]:
        points = self.depth_points()
        captured: dict[int, torch.Tensor] = {}
        handles = []

        def capture(layer: int):
            def hook(_module, _args, output):
                tokens = output[0] if isinstance(output, tuple) else output
                captured[layer] = tokens.detach()

            return hook

        for layer in points:
            handles.append(self.model.blocks[layer - 1].register_forward_hook(capture(layer)))

        try:
            out = self.model(
                inputs["hidden_states"].to(self.device, self.dtype),
                inputs["grid_thw"].to(self.device),
            )
        finally:
            for handle in handles:
                handle.remove()

        rows = len(image_ids)
        width = self.model.config.hidden_size
        for layer in points:
            tokens = captured[layer].view(rows, -1, width)
            yield make_batch(self, raster(tokens, rows), image_ids, "tower", layer)

        merged = out.last_hidden_state.view(rows, -1, self.model.config.out_hidden_size)
        yield make_batch(self, merged, image_ids, "merged", self.num_layers)
        yield make_batch(self, out.pooler_output.view(rows, merged.shape[1], -1), image_ids, "projected", self.num_layers)
