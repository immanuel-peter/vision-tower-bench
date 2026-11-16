from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from json import loads
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModel

from vtb.adapters.base import Inputs
from vtb.adapters.common import HALF, PatchPreprocess, default_device, depth_points, make_batch, pack_patches
from vtb.bundle import ROOT, expect_tensors, write_bundle
from vtb.bundle_manifest import copy_assets, write_manifest
from vtb.feature_batch import FeatureBatch
from vtb.provenance import log_loaded
from vtb.shards import load_prefixed

PATCH_SIZE = 14

MODEL_ID = "immanuelpeter/MoonViT-V2"
PROJECTOR_FILE = "projector.safetensors"

# Kimi K3 is where export reads both halves from.
SOURCE_REPO = "moonshotai/Kimi-K3"
PROJECTOR_SHARD = "model-00095-of-000096.safetensors"
PROJECTOR_PREFIX = "mm_projector."
TOWER_SHARD = "model-00096-of-000096.safetensors"
TOWER_PREFIX = "vision_tower."


@dataclass(frozen=True)
class Checkpoint:
    repo: str
    revision: str
    model_id: str
    bundle: str


K3 = Checkpoint(SOURCE_REPO, "f831ab66814297da540d832a5235f8e904f29d06", MODEL_ID, "MoonViT-V2")
K31_PRO = Checkpoint("moonshotai/Kimi-K3.1-Pro", "6d0a0202956796543364d666ab19c6ad1a494ee1",
                     "immanuelpeter/MoonViT-K3.1-Pro", "MoonViT-K3.1-Pro")
CHECKPOINTS = {"moonvit_v2": K3, "kimi_k31_pro": K31_PRO}


Preprocess = partial(PatchPreprocess, mean=HALF, std=HALF, patch_size=PATCH_SIZE, flatten=False)
collate = partial(pack_patches, grid_key="grid_thws")


def load_projector_weights(dtype: torch.dtype, model_id: str = MODEL_ID) -> nn.Module:
    """Build Kimi K3's patch merger from its published weights."""
    local = Path(model_id) / PROJECTOR_FILE
    weights = load_file(local if local.is_file() else hf_hub_download(model_id, PROJECTOR_FILE))
    width, merged_width = weights["proj.2.weight"].shape

    projector = nn.Module()
    projector.proj = nn.Sequential(
        nn.Linear(merged_width, merged_width, bias=False),
        nn.GELU(),
        nn.Linear(merged_width, width, bias=False),
    )
    projector.post_norm = nn.RMSNorm(width, eps=1e-5)
    projector.load_state_dict(weights)
    return projector.to(dtype).eval()


class MoonViTV2Adapter:
    """Kimi K3 adapter; use batch size 1 without flash attention, as measured."""

    checkpoint = K3
    model_id = MODEL_ID
    default_resolution = 448
    stages = ("tower", "merged", "projected")
    collate = staticmethod(collate)

    def __init__(self, resolution: int | None = None, dtype: torch.dtype = torch.bfloat16, device: str | None = None):
        self.resolution = self.default_resolution if resolution is None else resolution
        self.device = default_device() if device is None else device
        self.dtype = dtype
        log_loaded(self.model_id, self.model_id)
        self.model = (
            AutoModel.from_pretrained(self.model_id, dtype=dtype, trust_remote_code=True)
            .to(self.device)
            .eval()
        )
        self.num_layers = self.model.config.num_hidden_layers
        self.projector = load_projector_weights(dtype, self.model_id).to(self.device)

    def preprocess(self) -> PatchPreprocess:
        return Preprocess(self.resolution)

    def depth_points(self, n: int = 8) -> list[int]:
        return depth_points(self.num_layers, n)

    @torch.inference_mode()
    def extract(self, inputs: Inputs, image_ids: list[str]) -> Iterator[FeatureBatch]:
        if len(image_ids) != 1:
            raise ValueError("Kimi extraction requires batch size 1 for reproducible attention")
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
        for layer in points:
            yield make_batch(self, captured[layer].view(rows, -1, self.model.config.hidden_size), image_ids, "tower", layer)

        # Merge each 2x2 patch group.
        stacked = torch.stack(merged).flatten(2)
        yield make_batch(self, stacked, image_ids, "merged", self.num_layers)
        yield make_batch(self, self.projector.post_norm(self.projector.proj(stacked)), image_ids, "projected", self.num_layers)


class KimiK31ProAdapter(MoonViTV2Adapter):
    checkpoint = K31_PRO
    model_id = K31_PRO.model_id


TOWER_REPO = "AI4Industry/MoonViT-V2"
CODE_REVISION = "c5acb88baf02a94e68ca8a225f7eebee152c1fc6"
TOWER_TENSORS = 165
PROJECTOR_TENSORS = 3

# Loading the Tower needs AI4Industry's remote code, so the published repository carries it.
VENDORED = (
    "config.json",
    "configuration_moonvit_v2.py",
    "modeling_moonvit_v2.py",
    "image_processing_moonvit_v2.py",
    "preprocessor_config.json",
)


def discover_shards(checkpoint) -> dict[str, list[str]]:
    path = hf_hub_download(checkpoint.repo, "model.safetensors.index.json", revision=checkpoint.revision)
    mapping = loads(Path(path).read_text())["weight_map"]
    result = {prefix: sorted({shard for name, shard in mapping.items() if name.startswith(prefix)})
              for prefix in (TOWER_PREFIX, PROJECTOR_PREFIX)}
    if not all(result.values()):
        raise ValueError("source index lacks Tower or Projector tensors")
    selected = {shard for group in result.values() for shard in group}
    if any(shard in selected and not name.startswith((TOWER_PREFIX, PROJECTOR_PREFIX))
           for name, shard in mapping.items()):
        raise ValueError("vision shards contain non-vision tensors; refusing a full LM download")
    return result


def read_parent(checkpoint, shards: list[str], prefix: str) -> dict[str, torch.Tensor]:
    combined = {}
    for shard in shards:
        weights = load_prefixed(checkpoint.repo, [shard], prefix, revision=checkpoint.revision)
        if combined.keys() & weights.keys():
            raise ValueError("duplicate source tensor")
        combined.update(weights)
    return combined


def check_against_standalone(tower: dict[str, torch.Tensor]) -> None:
    standalone = load_file(hf_hub_download(TOWER_REPO, "model.safetensors", revision=CODE_REVISION))
    if set(standalone) != set(tower):
        missing = set(standalone) ^ set(tower)
        raise SystemExit(f"tensor names differ from {TOWER_REPO}: {sorted(missing)[:5]}")
    for name, weight in standalone.items():
        if not torch.equal(weight, tower[name]):
            raise SystemExit(f"{name} differs from {TOWER_REPO}")


def projector_config(projector: dict[str, torch.Tensor]) -> dict:
    width, merged_width = projector["proj.2.weight"].shape
    return {
        "merge_kernel_size": [2, 2],
        "merge_type": "sd2_tpool",
        "input_size": merged_width,
        "hidden_size": merged_width,
        "output_size": width,
        "linear_bias": False,
        "activation_func": "gelu",
        "norm_type": "rmsnorm",
        "norm_eps": 1e-5,
        "torch_dtype": str(projector["proj.2.weight"].dtype).removeprefix("torch."),
    }


def export(out: Path, checkpoint: Checkpoint) -> None:
    shards = discover_shards(checkpoint)
    tower = expect_tensors(read_parent(checkpoint, shards[TOWER_PREFIX], TOWER_PREFIX), TOWER_TENSORS, "Tower")
    projector = expect_tensors(read_parent(checkpoint, shards[PROJECTOR_PREFIX], PROJECTOR_PREFIX), PROJECTOR_TENSORS, "Projector")

    if any(t.dtype != torch.bfloat16 for t in (*tower.values(), *projector.values())):
        raise ValueError("expected BF16 source tensors")
    if checkpoint == K3:
        check_against_standalone(tower)

    write_bundle(
        out,
        weights={"model.safetensors": tower, "projector.safetensors": projector},
        configs={"projector_config.json": projector_config(projector)},
        license=Path(hf_hub_download(checkpoint.repo, "LICENSE", revision=checkpoint.revision)),
        files={
            **{name: Path(hf_hub_download(TOWER_REPO, name, revision=CODE_REVISION)) for name in VENDORED},
            "LICENSE-CODE": Path(hf_hub_download(TOWER_REPO, "LICENSE", revision=CODE_REVISION)),
        },
    )
    for filename, prefix in (("model.safetensors", TOWER_PREFIX), ("projector.safetensors", PROJECTOR_PREFIX)):
        exported = load_file(out / filename)
        parent = read_parent(checkpoint, shards[prefix], prefix)
        if exported.keys() != parent.keys() or any(not torch.equal(exported[k], parent[k]) for k in parent):
            raise ValueError(f"export differs from pinned parent: {filename}")
    copy_assets(ROOT / "hf" / checkpoint.bundle, out)
    write_manifest(out, checkpoint.repo, checkpoint.revision,
                   [name for group in shards.values() for name in group],
                   extra={"exact_parent_identity": True, "code_repo": TOWER_REPO, "code_revision": CODE_REVISION,
                          "forward_parity": "Independent math-reference BF16 exact; production official SDPA BF16 disagreement retained."})
    print(f"Bundle: {checkpoint.model_id}. Publish only after independent forward parity.")
