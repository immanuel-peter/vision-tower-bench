from collections.abc import Iterator
from functools import partial

import torch
from transformers import AutoModel

from vtb.adapters.base import Inputs
from vtb.adapters.common import IMAGENET_MEAN, IMAGENET_STD, ImagePreprocess, default_device, depth_points, make_batch, stack_pixels
from vtb.feature_batch import FeatureBatch
from vtb.provenance import log_loaded

MODEL_ID = "facebook/dinov2-large"
REVISION = "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"

Preprocess = partial(ImagePreprocess, mean=IMAGENET_MEAN, std=IMAGENET_STD)
collate = stack_pixels


class DINOv2Adapter:
    """Expose DINOv2 as the self-supervised control tower."""

    model_id = MODEL_ID
    default_resolution = 448
    stages = ("tower",)
    collate = staticmethod(collate)

    def __init__(self, resolution: int | None = None, dtype: torch.dtype = torch.bfloat16, device: str | None = None):
        self.resolution = self.default_resolution if resolution is None else resolution
        self.device = default_device() if device is None else device
        log_loaded(self.model_id, f"{MODEL_ID}@{REVISION}")
        self.model = AutoModel.from_pretrained(self.model_id, revision=REVISION, dtype=dtype).to(self.device).eval()
        self.num_layers = self.model.config.num_hidden_layers

    def preprocess(self) -> ImagePreprocess:
        return Preprocess(self.resolution)

    def depth_points(self, n: int = 8) -> list[int]:
        return depth_points(self.num_layers, n)

    @torch.inference_mode()
    def extract(self, inputs: Inputs, image_ids: list[str]) -> Iterator[FeatureBatch]:
        out = self.model(
            inputs["pixel_values"].to(self.device, self.model.dtype),
            output_hidden_states=True,
            interpolate_pos_encoding=True,
        )
        for layer in self.depth_points():
            # Use the model output at the final layer to retain its norm.
            source = out.last_hidden_state if layer == self.num_layers else out.hidden_states[layer]
            # Drop the CLS token.
            yield make_batch(self, source[:, 1:, :], image_ids, "tower", layer)
