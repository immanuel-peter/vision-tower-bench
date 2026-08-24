from collections.abc import Iterator

import torch
from transformers import AutoModel

from vtb.feature_batch import FeatureBatch


class DINOv2Adapter:
    """Self-supervised control Tower. No Projector, so `tower` is its only Stage."""

    model_id = "facebook/dinov2-large"
    stages = ("tower",)

    def __init__(self, resolution: int = 448, dtype=torch.bfloat16, device: str = "mps"):
        self.model = AutoModel.from_pretrained(self.model_id, dtype=dtype).to(device).eval()
        self.resolution = resolution
        self.device = device
        self.num_layers = self.model.config.num_hidden_layers

    def depth_points(self, n: int = 8) -> list[int]:
        """Layer indices at Relative Depth 1/n .. 1.0."""
        return [round(self.num_layers * (k + 1) / n) for k in range(n)]

    @torch.inference_mode()
    def extract(self, pixel_values: torch.Tensor, image_ids: list[str]) -> Iterator[FeatureBatch]:
        out = self.model(
            pixel_values.to(self.device, self.model.dtype),
            output_hidden_states=True,
            interpolate_pos_encoding=True,
        )
        for layer in self.depth_points():
            # hidden_states[0] is the embedding output, so index i is after block i.
            # Column 0 is the CLS token, dropped to keep every model patch-only.
            yield FeatureBatch(
                tokens=out.hidden_states[layer][:, 1:, :].cpu(),
                image_ids=image_ids,
                model_id=self.model_id,
                stage="tower",
                layer_index=layer,
                num_layers=self.num_layers,
                resolution=self.resolution,
            )
