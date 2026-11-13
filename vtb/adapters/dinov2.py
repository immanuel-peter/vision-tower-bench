from collections.abc import Iterator
from dataclasses import dataclass, field

import torch
from torchvision import transforms
from transformers import AutoModel

from vtb.feature_batch import FeatureBatch
from vtb.images import square_crop
from vtb.provenance import log_loaded

MODEL_ID = "facebook/dinov2-large"
REVISION = "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]),
        )

    def __call__(self, image) -> torch.Tensor:
        return self.pipeline(image)


def collate(samples: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(samples)


class DINOv2Adapter:
    """Expose DINOv2 as the self-supervised control tower."""

    model_id = MODEL_ID
    stages = ("tower",)
    collate = staticmethod(collate)

    def __init__(self, resolution: int = 448, dtype=torch.bfloat16, device: str = "mps"):
        log_loaded(self.model_id, f"{MODEL_ID}@{REVISION}")
        self.model = AutoModel.from_pretrained(self.model_id, revision=REVISION, dtype=dtype).to(device).eval()
        self.resolution = resolution
        self.device = device
        self.num_layers = self.model.config.num_hidden_layers

    def preprocess(self) -> Preprocess:
        return Preprocess(self.resolution)

    def depth_points(self, n: int = 8) -> list[int]:
        return [round(self.num_layers * (k + 1) / n) for k in range(n)]

    @torch.inference_mode()
    def extract(self, pixel_values: torch.Tensor, image_ids: list[str]) -> Iterator[FeatureBatch]:
        out = self.model(
            pixel_values.to(self.device, self.model.dtype),
            output_hidden_states=True,
            interpolate_pos_encoding=True,
        )
        for layer in self.depth_points():
            # Use the model output at the final layer to retain its norm.
            source = out.last_hidden_state if layer == self.num_layers else out.hidden_states[layer]
            # Drop the CLS token.
            yield FeatureBatch(
                tokens=source[:, 1:, :].cpu(),
                image_ids=image_ids,
                model_id=self.model_id,
                stage="tower",
                layer_index=layer,
                num_layers=self.num_layers,
                resolution=self.resolution,
            )
