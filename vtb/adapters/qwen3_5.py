import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from collections.abc import Iterator
from dataclasses import dataclass, field

import torch
from torchvision import transforms
from transformers import AutoModel, Qwen3_5VisionConfig, Qwen3_5VisionModel

from vtb.shards import load_prefixed

from vtb.feature_batch import FeatureBatch
from vtb.images import square_crop
from vtb.provenance import check_provenance, log_loaded

ROOT = Path(__file__).resolve().parents[2]
VISION_PREFIX = "model.visual."
TOWER_TENSORS = 333


@dataclass(frozen=True)
class Checkpoint:
    name: str
    model_id: str
    repo: str
    revision: str
    out_hidden_size: int
    license: str
    release_revision: str

    @property
    def bundle(self) -> Path:
        override = os.environ.get(f"VTB_{self.name.upper()}_BUNDLE")
        return Path(override) if override else ROOT / "hf" / self.model_id.split("/")[-1]


QWEN27B = Checkpoint(
    "qwen3_5", "immanuelpeter/Qwen3.8-27B-Vision", "Qwen/Qwen3.8-27B",
    "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", 5120, "apache-2.0", "60a1bf9336197d19458290fd43f3d527f6cf3078",
)
FLASH_NEXT = Checkpoint(
    "qwen38_flash_next", "immanuelpeter/Qwen3.8-Flash-Next-Vision", "Qwen/Qwen3.8-Flash-Next",
    "de4b8e4d43b917e7706784d8bb445c9af86a3540", 2560, "qwen-community-1.0", "120ed5d5baf11cb478099350e94218ee3cbed3df",
)
CHECKPOINTS = {c.name: c for c in (QWEN27B, FLASH_NEXT)}
MODEL_ID, SOURCE_REPO = QWEN27B.model_id, QWEN27B.repo


def source_json(checkpoint: Checkpoint, filename: str) -> dict:
    return json.loads(Path(hf_hub_download(checkpoint.repo, filename, revision=checkpoint.revision)).read_text())


def source_weights(checkpoint: Checkpoint) -> dict[str, torch.Tensor]:
    index = source_json(checkpoint, "model.safetensors.index.json")["weight_map"]
    selected = {k: v for k, v in index.items() if k.startswith(VISION_PREFIX)}
    weights = load_prefixed(checkpoint.repo, sorted(set(selected.values())), VISION_PREFIX, revision=checkpoint.revision)
    if len(weights) != TOWER_TENSORS or {VISION_PREFIX + k for k in weights} != set(selected):
        raise ValueError(f"incomplete {checkpoint.name} vision state: {len(weights)} tensors")
    return weights


def load_source_tower(checkpoint: Checkpoint, dtype=torch.bfloat16) -> Qwen3_5VisionModel:
    config = source_json(checkpoint, "config.json")["vision_config"]
    config.update(model_type="qwen3_5_vision", out_hidden_size=checkpoint.out_hidden_size)
    model = Qwen3_5VisionModel._from_config(Qwen3_5VisionConfig(**config), dtype=dtype)
    model.load_state_dict(source_weights(checkpoint), strict=True)
    return model.eval()


PATCH_SIZE = 16
TEMPORAL_PATCH_SIZE = 2
MERGE_SIZE = 2


@dataclass(frozen=True)
class Preprocess:
    """Flatten an image into merge-block patch rows. ``raster`` undoes that grouping."""

    resolution: int
    pipeline: transforms.Compose = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.resolution <= 0 or self.resolution % (PATCH_SIZE * MERGE_SIZE):
            raise ValueError("Qwen resolution must be a positive multiple of 32")
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
        # A still image repeats to fill the temporal patch.
        frames = pixels.unsqueeze(0).expand(TEMPORAL_PATCH_SIZE, -1, -1, -1)
        patches = frames.reshape(
            1, TEMPORAL_PATCH_SIZE, channels,
            rows // MERGE_SIZE, MERGE_SIZE, PATCH_SIZE,
            cols // MERGE_SIZE, MERGE_SIZE, PATCH_SIZE,
        )
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        return patches.reshape(rows * cols, channels * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE)


def collate(samples: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    side = int((samples[0].shape[0]) ** 0.5)
    return {
        "hidden_states": torch.cat(samples),
        "grid_thw": torch.tensor([[1, side, side]] * len(samples)),
    }


def raster(tokens: torch.Tensor, rows: int) -> torch.Tensor:
    width = tokens.shape[-1]
    side = int((tokens.numel() // (rows * width)) ** 0.5)
    blocks = side // MERGE_SIZE
    grid = tokens.view(rows, blocks, blocks, MERGE_SIZE, MERGE_SIZE, width)
    return grid.permute(0, 1, 3, 2, 4, 5).reshape(rows, side * side, width)


def load_tower(dtype: torch.dtype, checkpoint: Checkpoint = QWEN27B) -> Qwen3_5VisionModel:
    if (checkpoint.bundle / "model.safetensors").is_file():
        check_provenance(checkpoint.bundle, checkpoint.repo, checkpoint.revision)
        log_loaded(checkpoint.model_id, str(checkpoint.bundle))
        model = AutoModel.from_pretrained(checkpoint.bundle, dtype=dtype, local_files_only=True).eval()
    else:
        log_loaded(checkpoint.model_id, f"{checkpoint.model_id}@{checkpoint.release_revision}")
        model = AutoModel.from_pretrained(
            checkpoint.model_id, revision=checkpoint.release_revision, dtype=dtype,
        ).eval()
    if model.config.out_hidden_size != checkpoint.out_hidden_size:
        raise ValueError(f"wrong Projector width for {checkpoint.name}")
    return model


class QwenAdapter:
    checkpoint = QWEN27B
    stages = ("tower", "merged", "projected")
    collate = staticmethod(collate)

    def __init__(self, resolution: int = 448, dtype=torch.bfloat16, device: str = "mps"):
        Preprocess(resolution)
        self.model = load_tower(dtype, self.checkpoint).to(device)
        self.model_id = self.checkpoint.model_id
        self.resolution = resolution
        self.device = device
        self.dtype = dtype
        self.num_layers = self.model.config.depth

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
            yield self._batch(raster(tokens, rows), image_ids, "tower", layer)

        deepest = out.last_hidden_state.view(rows, -1, width * MERGE_SIZE**2)
        yield self._batch(deepest, image_ids, "merged", self.num_layers)
        yield self._batch(out.pooler_output.view(rows, deepest.shape[1], -1), image_ids, "projected", self.num_layers)

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


class Qwen3_5Adapter(QwenAdapter):
    checkpoint = QWEN27B
    model_id = QWEN27B.model_id


class Qwen38FlashNextAdapter(QwenAdapter):
    checkpoint = FLASH_NEXT
    model_id = FLASH_NEXT.model_id
