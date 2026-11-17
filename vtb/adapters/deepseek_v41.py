from collections.abc import Iterator
from functools import partial
from math import isqrt
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from vtb.adapters.base import Inputs
from vtb.adapters.common import (
    HALF,
    PatchPreprocess,
    default_device,
    depth_points,
    make_batch,
)
from vtb.bundle import expect_tensors, write_bundle
from vtb.feature_batch import FeatureBatch
from vtb.models.deepseek_v41 import ARGS, PATCH_SIZE, Aligner, ViT
from vtb.provenance import open_release
from vtb.shards import load_prefixed

MODEL_ID = "immanuelpeter/DeepSeek-ViT"

SOURCE_REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
SOURCE_REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
VISION_SHARD = "model-00001-of-00048.safetensors"
TOWER_PREFIX = "vision."
ALIGNER_PREFIX = "aligner."
TOWER_TENSORS = 259
PROJECTOR_TENSORS = 4

Preprocess = partial(
    PatchPreprocess, mean=HALF, std=HALF, patch_size=PATCH_SIZE, flatten=False
)


def collate(samples: list[torch.Tensor]) -> Inputs:
    side = isqrt(samples[0].shape[0])
    return {"patches": torch.stack(samples), "grid": torch.tensor([side, side])}


def load_source_parts(dtype: torch.dtype = torch.bfloat16) -> tuple[ViT, Aligner]:
    tower = ViT()
    weights = load_prefixed(
        SOURCE_REPO, [VISION_SHARD], TOWER_PREFIX, revision=SOURCE_REVISION
    )
    tower.load_state_dict({name: tensor.to(dtype) for name, tensor in weights.items()})
    aligner = Aligner()
    aligner.load_state_dict(
        {
            name: tensor.to(dtype)
            for name, tensor in load_prefixed(
                SOURCE_REPO, [VISION_SHARD], ALIGNER_PREFIX, revision=SOURCE_REVISION
            ).items()
        }
    )
    return tower.to(dtype).eval(), aligner.to(dtype).eval()


def export(out: Path) -> None:
    expect_tensors(
        load_prefixed(
            SOURCE_REPO, [VISION_SHARD], TOWER_PREFIX, revision=SOURCE_REVISION
        ),
        TOWER_TENSORS,
        "Tower",
    )
    expect_tensors(
        load_prefixed(
            SOURCE_REPO, [VISION_SHARD], ALIGNER_PREFIX, revision=SOURCE_REVISION
        ),
        PROJECTOR_TENSORS,
        "Aligner",
    )

    tower, aligner = load_source_parts()
    write_bundle(
        out,
        weights={
            "model.safetensors": tower.state_dict(),
            "projector.safetensors": aligner.state_dict(),
        },
        configs={
            "config.json": {
                "architectures": ["DeepSeekV41ViT"],
                "vision_n_layers": 32,
                "vision_dim": 1024,
                "vision_patch_size": 14,
                "vision_downsample_ratio": 3,
                "aligner_dim": 5120,
            }
        },
        license=Path(hf_hub_download(SOURCE_REPO, "LICENSE", revision=SOURCE_REVISION)),
    )


def load_parts(dtype: torch.dtype) -> tuple[ViT, Aligner]:
    bundle = open_release(MODEL_ID, SOURCE_REPO, SOURCE_REVISION)
    tower = ViT()
    tower.load_state_dict(
        {
            name: tensor.to(dtype)
            for name, tensor in load_file(bundle / "model.safetensors").items()
        }
    )
    aligner = Aligner()
    aligner.load_state_dict(
        {
            name: tensor.to(dtype)
            for name, tensor in load_file(bundle / "projector.safetensors").items()
        }
    )
    return tower.to(dtype).eval(), aligner.to(dtype).eval()


class DeepSeekV41Adapter:
    model_id = MODEL_ID
    default_resolution = 448
    stages = ("tower", "merged", "projected")
    collate = staticmethod(collate)

    def __init__(
        self,
        resolution: int | None = None,
        dtype: torch.dtype = torch.bfloat16,
        device: str | None = None,
    ):
        self.resolution = self.default_resolution if resolution is None else resolution
        self.device = default_device() if device is None else device
        self.dtype = dtype
        tower, aligner = load_parts(dtype)
        self.model = tower.to(self.device)
        self.aligner = aligner.to(self.device)
        self.num_layers = len(self.model.blocks)

    def preprocess(self) -> PatchPreprocess:
        return Preprocess(self.resolution)

    def depth_points(self, n: int = 8) -> list[int]:
        return depth_points(self.num_layers, n)

    @torch.inference_mode()
    def extract(self, inputs: Inputs, image_ids: list[str]) -> Iterator[FeatureBatch]:
        points = self.depth_points()
        n_h, n_w = int(inputs["grid"][0]), int(inputs["grid"][1])
        rows = len(image_ids)
        width = ARGS.vision_dim
        towers: dict[int, list[torch.Tensor]] = {layer: [] for layer in points}
        last: list[torch.Tensor] = []
        merged: list[torch.Tensor] = []
        projected: list[torch.Tensor] = []

        for patches in inputs["patches"]:
            captured: dict[int, torch.Tensor] = {}
            handles = []

            def capture(layer: int):
                def hook(_module, _args, output):
                    tokens = output[0] if isinstance(output, tuple) else output
                    captured[layer] = tokens.detach()

                return hook

            for layer in points:
                handles.append(
                    self.model.blocks[layer - 1].register_forward_hook(capture(layer))
                )
            try:
                tokens = self.model(patches.to(self.device, self.dtype), n_h, n_w)
            finally:
                for handle in handles:
                    handle.remove()
            for layer in points:
                towers[layer].append(captured[layer])
            last.append(tokens)
            merged.append(self.aligner.unfold(tokens, n_h, n_w))
            projected.append(self.aligner(tokens, n_h, n_w))

        for layer in points:
            yield make_batch(
                self,
                torch.stack(towers[layer]).view(rows, -1, width),
                image_ids,
                "tower",
                layer,
            )
        yield make_batch(
            self, torch.stack(merged), image_ids, "merged", self.num_layers
        )
        yield make_batch(
            self, torch.stack(projected), image_ids, "projected", self.num_layers
        )
