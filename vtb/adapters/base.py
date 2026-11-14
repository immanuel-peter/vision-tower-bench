from collections.abc import Callable, Iterator
from typing import ClassVar, Protocol, runtime_checkable

import torch
from PIL import Image

from vtb.feature_batch import FeatureBatch

Inputs = dict[str, torch.Tensor]


@runtime_checkable
class TowerAdapter(Protocol):
    """The contract every Tower adapter meets.

    Constructors take ``resolution=None, dtype=torch.bfloat16, device=None``.
    ``None`` resolves to ``default_resolution`` and ``common.default_device()``.
    ``extract`` yields one FeatureBatch per Tower depth point, then the merged
    and projected Stages the adapter declares, all as raster-order patch tokens.
    """

    default_resolution: ClassVar[int]
    model_id: str
    resolution: int
    device: str
    num_layers: int

    @property
    def stages(self) -> tuple[str, ...]: ...

    def preprocess(self) -> Callable[[Image.Image], torch.Tensor]: ...

    def collate(self, samples: list[torch.Tensor]) -> Inputs: ...

    def depth_points(self, n: int = 8) -> list[int]: ...

    def extract(self, inputs: Inputs, image_ids: list[str]) -> Iterator[FeatureBatch]: ...
