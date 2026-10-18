"""MiMo 2.6 Pro and Flash vision Towers, with every sampled depth restored to raster order."""

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from math import isqrt
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torchvision import transforms

from vtb.feature_batch import FeatureBatch
from vtb.images import square_crop
from vtb.models.mimo26 import MiMoVisionConfig, MiMoVisionModel
from vtb.shards import load_prefixed

ROOT = Path(__file__).resolve().parents[2]
PATCH_SIZE, TEMPORAL_PATCH_SIZE, MERGE_SIZE = 16, 2, 2
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
VISUAL_TENSORS = 364


@dataclass(frozen=True)
class Checkpoint:
    name: str
    model_id: str
    repo: str
    revision: str

    @property
    def release_dir(self) -> Path:
        return ROOT / "hf" / self.model_id.split("/")[-1]

    @property
    def bundle(self) -> Path:
        override = os.environ.get(f"VTB_{self.name.upper()}_BUNDLE")
        return Path(override) if override else self.release_dir


PRO = Checkpoint(
    "mimo26_pro",
    "immanuelpeter/MiMo-V2.6-Pro-Vision",
    "XiaomiMiMo/MiMo-V2.6-Pro-RL",
    "b61f21852680ee02a29e946e18cf8697f4844061",
)
FLASH = Checkpoint(
    "mimo26_flash",
    "immanuelpeter/MiMo-V2.6-Flash-Vision",
    "XiaomiMiMo/MiMo-V2.6-Flash-RL",
    "dcd3fd83fb181622fbc606fbb1709901982a62c2",
)
CHECKPOINTS = {item.name: item for item in (PRO, FLASH)}


def source_json(checkpoint: Checkpoint, filename: str) -> dict:
    return json.loads(Path(hf_hub_download(checkpoint.repo, filename, revision=checkpoint.revision)).read_text())


def source_weights(checkpoint: Checkpoint) -> dict[str, torch.Tensor]:
    """Discover shards from the pinned index and fetch only the visual tensors."""
    index = source_json(checkpoint, "model.safetensors.index.json")["weight_map"]
    selected = {name: shard for name, shard in index.items() if name.startswith("visual.")}
    shards = sorted(set(selected.values()))
    weights = load_prefixed(checkpoint.repo, shards, "visual.", revision=checkpoint.revision)
    if {"visual." + name for name in weights} != set(selected) or len(weights) != VISUAL_TENSORS:
        raise ValueError(f"incomplete {checkpoint.name} visual state: {len(weights)} tensors")
    return weights


@dataclass(frozen=True)
class Preprocess:
    resolution: int
    pipeline: transforms.Compose = field(init=False, repr=False)

    def __post_init__(self):
        if self.resolution <= 0 or self.resolution % (PATCH_SIZE * MERGE_SIZE):
            raise ValueError("MiMo resolution must be a positive multiple of 32")
        object.__setattr__(self, "pipeline", transforms.Compose([
            square_crop(self.resolution),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]))

    def __call__(self, image) -> torch.Tensor:
        pixels = self.pipeline(image.convert("RGB"))
        c, h, w = pixels.shape
        frames = pixels.unsqueeze(0).expand(TEMPORAL_PATCH_SIZE, -1, -1, -1)
        # Qwen2-VL processor order: merge block, within-block row/col, C/T/H/W.
        patches = frames.reshape(1, 2, c, h // 32, 2, 16, w // 32, 2, 16)
        return patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(-1, c * 2 * 16 * 16)


def collate(samples: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("empty MiMo batch")
    side = isqrt(samples[0].shape[0])
    if side % 2 or side**2 != samples[0].shape[0] or any(s.shape != samples[0].shape for s in samples):
        raise ValueError("MiMo batches require equal, even-sided square patch grids")
    return {"pixel_values": torch.cat(samples), "grid_thw": torch.tensor([[1, side, side]] * len(samples))}


def raster(tokens: torch.Tensor, grid_thw: torch.Tensor, column_major: bool) -> list[torch.Tensor]:
    """Undo the current block permutation, then undo each 2x2 merge block.

    Handle each image independently. Transposing a whole packed batch would mix
    images; transposing individual tokens would corrupt the within-block order.
    """
    restored = []
    offset = 0
    for t, h, w in grid_thw.cpu().tolist():
        if t != 1 or h % 2 or w % 2:
            raise ValueError("the image adapter requires t=1 and even spatial grids")
        count, width = h * w, tokens.shape[-1]
        sample = tokens[offset:offset + count]
        if column_major:
            sample = sample.reshape(w // 2, h // 2, 4, width).transpose(0, 1)
        blocks = sample.reshape(h // 2, w // 2, 2, 2, width)
        restored.append(blocks.permute(0, 2, 1, 3, 4).reshape(count, width))
        offset += count
    if offset != tokens.shape[0]:
        raise ValueError("grid_thw does not describe the token count")
    return restored


def load_model(checkpoint: Checkpoint, dtype: torch.dtype, bundle: Path | None = None) -> MiMoVisionModel:
    bundle = checkpoint.bundle if bundle is None else bundle
    if (bundle / "model.safetensors").is_file():
        manifest = bundle / "export_manifest.json"
        if manifest.is_file():
            provenance = json.loads(manifest.read_text())
            if (provenance["source_repo"], provenance["source_revision"]) != (checkpoint.repo, checkpoint.revision):
                raise ValueError(f"bundle provenance does not match {checkpoint.name}")
        config = MiMoVisionConfig.from_pretrained(bundle, local_files_only=True)
        weights = load_file(bundle / "model.safetensors")
    else:
        config = MiMoVisionConfig(**source_json(checkpoint, "config.json")["vision_config"])
        weights = {"visual." + name: tensor for name, tensor in source_weights(checkpoint).items()}
    with torch.device("meta"):
        model = MiMoVisionModel(config)
    model.load_state_dict(weights, strict=True, assign=True)
    # HF from_pretrained initializes this nonpersistent buffer in float32 even
    # when parameters are BF16. Preserve that policy; rounding the frequencies
    # to BF16 before building positions changes the features appreciably.
    head_dim = getattr(config, "qk_channels", 64)
    frequencies = 1.0 / (10000.0 ** (torch.arange(0, head_dim // 2, 2).float() / (head_dim // 2)))
    model = model.to(dtype=dtype)
    model.visual.rotary_pos_emb.inv_freq = frequencies
    return model.eval()


class MiMo26Adapter:
    checkpoint = PRO
    stages = ("tower", "merged", "projected")
    collate = staticmethod(collate)

    def __init__(self, resolution: int = 448, dtype=torch.bfloat16, device: str = "mps", *, bundle: Path | None = None):
        Preprocess(resolution)  # Reject unsupported grids before loading weights.
        self.model = load_model(self.checkpoint, dtype, bundle).to(device)
        self.model_id = self.checkpoint.model_id
        self.resolution, self.dtype, self.device = resolution, dtype, device
        self.num_layers = self.model.config.depth

    def preprocess(self):
        return Preprocess(self.resolution)

    def depth_points(self, n: int = 8) -> list[int]:
        return sorted({round(self.num_layers * (k + 1) / n) for k in range(n)} - {0})

    @torch.inference_mode()
    def extract(self, inputs: dict[str, torch.Tensor], image_ids: list[str]) -> Iterator[FeatureBatch]:
        grid = inputs["grid_thw"]
        if len(grid) != len(image_ids):
            raise ValueError("grid/image ID count differs")
        visual = self.model.visual
        captured = {}
        handles = []

        def capture(key):
            def hook(_module, _args, output):
                captured[key] = output.detach()
            return hook

        for layer in self.depth_points():
            handles.append(visual.blocks[layer - 1].register_forward_hook(capture(layer)))
        # merged includes the parent's LayerNorm, immediately before merger MLP.
        handles.append(visual.merger.ln_q.register_forward_hook(capture("merged")))
        try:
            projected = self.model(inputs["pixel_values"].to(self.device, self.dtype), grid)
        finally:
            for handle in handles:
                handle.remove()

        for layer in self.depth_points():
            column_major = visual.vit_window_attn_types[layer - 1] == 1
            tokens = torch.stack(raster(captured[layer], grid, column_major))
            yield self._batch(tokens, image_ids, "tower", layer)

        # Both pinned configs end in a full-attention row-order block. The parent
        # merger consumes that order, so the 2x2 groups already form a raster grid.
        if visual.vit_window_attn_types[-1] == 1:
            raise ValueError("pinned MiMo image exports must finish in row order")
        rows = len(image_ids)
        merged = captured["merged"].reshape(rows, -1, self.model.config.hidden_size * 4)
        yield self._batch(merged, image_ids, "merged", self.num_layers)
        yield self._batch(projected.reshape(rows, merged.shape[1], -1), image_ids, "projected", self.num_layers)

    def _batch(self, tokens, image_ids, stage, layer):
        return FeatureBatch(tokens.cpu(), image_ids, self.model_id, stage, layer, self.num_layers, self.resolution)


class MiMo26ProAdapter(MiMo26Adapter):
    checkpoint = PRO


class MiMo26FlashAdapter(MiMo26Adapter):
    checkpoint = FLASH
