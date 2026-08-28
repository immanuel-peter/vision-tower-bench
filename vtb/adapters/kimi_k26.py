from collections.abc import Iterator
from dataclasses import dataclass, field
from json import loads
from types import SimpleNamespace

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torchvision import transforms
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.utils import import_utils

from vtb.feature_batch import FeatureBatch
from vtb.images import square_crop

MODEL_ID = "exolabs/Kimi-K2.6-vision"
WEIGHTS_FILE = "kimi_k26_vision.safetensors"
TOWER_PREFIX = "vision_tower."
PROJECTOR_PREFIX = "mm_projector."

# exolabs republishes the weights but not the architecture, which stays with the source model.
CODE_REPO = "moonshotai/Kimi-K2.6"
CODE_MODULE = "modeling_kimi_k25"
PATCH_SIZE = 14


@dataclass(frozen=True)
class Preprocess:
    resolution: int
    pipeline: transforms.Compose = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pipeline",
            transforms.Compose([
                square_crop(self.resolution),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
        )

    def __call__(self, image) -> torch.Tensor:
        pixels = self.pipeline(image)
        channels, height, width = pixels.shape
        rows, cols = height // PATCH_SIZE, width // PATCH_SIZE
        patches = pixels.view(channels, rows, PATCH_SIZE, cols, PATCH_SIZE)
        return patches.permute(1, 3, 0, 2, 4).reshape(-1, channels, PATCH_SIZE, PATCH_SIZE)


def collate(samples: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    side = int(samples[0].shape[0] ** 0.5)
    return {
        "pixel_values": torch.cat(samples),
        "grid_thws": torch.tensor([[1, side, side]] * len(samples)),
    }


def remote_class(name: str):
    # The K2.6 code targets transformers 4.x and imports one helper that 5.x dropped.
    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: False
    return get_class_from_dynamic_module(f"{CODE_MODULE}.{name}", CODE_REPO)


def load_parts(dtype: torch.dtype, attention: str = "eager"):
    settings = loads(open(hf_hub_download(MODEL_ID, "config.json")).read())["vision_config"]
    settings["_attn_implementation"] = attention
    source = SimpleNamespace(**settings)

    weights = load_file(hf_hub_download(MODEL_ID, WEIGHTS_FILE))
    tower = remote_class("MoonViT3dPretrainedModel")(remote_class("VisionTowerConfig")(source))
    tower.load_state_dict(split(weights, TOWER_PREFIX))
    projector = remote_class("PatchMergerMLP")(remote_class("ProjectorConfig")(source))
    projector.load_state_dict(split(weights, PROJECTOR_PREFIX))
    return tower.to(dtype).eval(), projector.to(dtype).eval()


def split(weights: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {k.removeprefix(prefix): v for k, v in weights.items() if k.startswith(prefix)}


class KimiK26Adapter:
    """Use batch size 1 without flash attention, as measured in ADR-0009."""

    model_id = MODEL_ID
    stages = ("tower", "merged", "projected")
    collate = staticmethod(collate)

    def __init__(self, resolution: int = 448, dtype=torch.bfloat16, device: str = "mps"):
        tower, projector = load_parts(dtype)
        self.model = tower.to(device)
        self.projector = projector.to(device)
        self.resolution = resolution
        self.device = device
        self.dtype = dtype
        self.num_layers = self.model.config.num_hidden_layers

    def preprocess(self) -> Preprocess:
        return Preprocess(self.resolution)

    def depth_points(self, n: int = 8) -> list[int]:
        return [round(self.num_layers * (k + 1) / n) for k in range(n)]

    @torch.inference_mode()
    def extract(self, inputs: dict[str, torch.Tensor], image_ids: list[str]) -> Iterator[FeatureBatch]:
        points = self.depth_points()
        captured: dict[int, torch.Tensor] = {}
        handles = []

        def capture(layer: int):
            def hook(_module, _args, output):
                captured[layer] = output.detach()

            return hook

        for layer in points:
            # Use the encoder output at the final layer to retain its norm.
            target = self.model.encoder if layer == self.num_layers else self.model.encoder.blocks[layer - 1]
            handles.append(target.register_forward_hook(capture(layer)))

        try:
            merged = self.model(
                inputs["pixel_values"].to(self.device, self.dtype),
                inputs["grid_thws"].to(self.device),
            )
        finally:
            for handle in handles:
                handle.remove()

        rows = len(image_ids)
        width = self.model.config.hidden_size
        for layer in points:
            yield self._batch(captured[layer].view(rows, -1, width), image_ids, "tower", layer)

        stacked = torch.stack(merged)
        yield self._batch(stacked.flatten(2), image_ids, "merged", self.num_layers)
        yield self._batch(self.projector(stacked), image_ids, "projected", self.num_layers)

    def _batch(self, tokens: torch.Tensor, image_ids: list[str], stage: str, layer: int) -> FeatureBatch:
        return FeatureBatch(
            tokens=tokens.cpu(),
            image_ids=image_ids,
            model_id=self.model_id,
            stage=stage,
            layer_index=layer,
            num_layers=self.num_layers,
            resolution=self.resolution,
        )
