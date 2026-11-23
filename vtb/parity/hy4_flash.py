"""Hy4 flash against the hy_v4_flash vision classes of transformers PR #50903, pinned at a commit.

The PR is not merged, so the reference compiles its vision classes from the SHA-256-checked
sources at that commit, with transformers' own attention and helper functions. It loads the
parent's tensors through the PR's vision renames, written out here, and runs the packed
per-image forward that ``HYV4FlashVisionTransformer`` runs. The pixels go through the PR image
processor's ``patchify``; its ``smart_resize`` must leave the bench's 512 px square unchanged.
The merged reference unfolds the RMSNorm output with ``F.unfold`` instead of the adapter's reshape.
"""

import ast
import math
import re
from collections.abc import Callable
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModel
from transformers.activations import ACT2FN
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.generic import get_max_seqlen, is_flash_attention_requested
from transformers.utils.import_utils import torch_compilable_check
from transformers.vision_utils import get_vision_attention_seqlens

from vtb.adapters import hy4_flash as hy
from vtb.parity import compare, fixed_images, weights_match
from vtb.parity.pe import pinned_sources

REFERENCE = (
    "huggingface/transformers",
    50903,
    "c2ec850510a37978ea83df226589623d5f2be776",
)
MODELING = "src/transformers/models/hy_v4_flash/modeling_hy_v4_flash.py"
IMAGE_PROCESSING = "src/transformers/models/hy_v4_vl/image_processing_hy_v4_vl.py"
SOURCES = {
    MODELING: "43341929e4bec215c224fc837b24b534b4de41954992673a0655a65060e38b9b",
    IMAGE_PROCESSING: "3e2063c2f2d197ad55b01adc4058844e6cbaa72a39a516d01982fe44207cb954",
}
CLASSES = (
    "repeat_kv",
    "eager_attention_forward",
    "HYV4FlashRMSNorm",
    "HYV4FlashVisionPatchMerger",
    "HYV4FlashVisionPatchEmbed",
    "HYV4FlashVisionAttention",
    "HYV4FlashVisionMLP",
    "HYV4FlashVisionBlock",
)
# The PR's vision rules in conversion_mapping.py, applied after the `vit.` prefix is removed.
RENAMES = (
    (r"mlp\.dense_h_to_4h", "mlp.fc1"),
    (r"mlp\.dense_4h_to_h", "mlp.fc2"),
    (r"^layers\.(\d+)\.input_layernorm\.", r"layers.\1.layer_norm1."),
    (r"^layers\.(\d+)\.post_attention_layernorm\.", r"layers.\1.layer_norm2."),
    (r"^perceive\.norm\.", "perceive.before_rms."),
    (r"^perceive\.proj\.0\.", "perceive.proj_conv."),
    (r"^perceive\.proj\.2\.", "perceive.proj_out."),
    (r"^perceive\.", "patch_merger."),
)
ATTENTION = "sdpa"
# Relative L2 per cell. FP32 on CPU is bit-exact through every Tower depth; on CUDA it is bit-exact
# through merged, and the batched Projector conv reaches 3.4e-5. BF16 on CUDA reaches 0.012 at L27,
# where each implementation is already 0.017 from its own FP32 output. Max abs is not used: a few
# tokens carry activations near 1000, so one BF16 ulp there is 4 to 8.
TOLERANCE = {torch.float32: 1e-4, torch.bfloat16: 0.03}


def identity(*args, **_kwargs):
    if len(args) == 1 and callable(args[0]):
        return args[0]
    return identity


def compile_nodes(
    path, wanted: set[str], namespace: dict, within: str | None = None
) -> None:
    body = ast.parse(path.read_text()).body
    if within is not None:
        body = next(n for n in body if getattr(n, "name", None) == within).body
    nodes = [node for node in body if getattr(node, "name", None) in wanted]
    missing = wanted - {node.name for node in nodes}
    if missing:
        raise ValueError(f"{sorted(missing)} not found in {path}")
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def reference_code() -> dict:
    """The PR's vision classes and image-processor helpers, compiled without importing the PR."""
    _, _, head = REFERENCE
    root = pinned_sources(REFERENCE[0], head, SOURCES)
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "math": math,
        "Callable": Callable,
        "ACT2FN": ACT2FN,
        "ALL_ATTENTION_FUNCTIONS": ALL_ATTENTION_FUNCTIONS,
        "GradientCheckpointingLayer": GradientCheckpointingLayer,
        "Unpack": Unpack,
        "TransformersKwargs": TransformersKwargs,
        "auto_docstring": identity,
        "get_max_seqlen": get_max_seqlen,
        "is_flash_attention_requested": is_flash_attention_requested,
        "torch_compilable_check": torch_compilable_check,
        "HYV4FlashVisionConfig": SimpleNamespace,
    }
    compile_nodes(root / MODELING, set(CLASSES), namespace)
    compile_nodes(root / IMAGE_PROCESSING, {"smart_resize"}, namespace)
    compile_nodes(
        root / IMAGE_PROCESSING, {"patchify"}, namespace, within="HYV4VLImageProcessor"
    )
    return namespace


def reference_config() -> SimpleNamespace:
    """The parent's own vision_config, with the PR's attribute_map aliases."""
    vision = hy.source_json("config.json")["vision_config"]
    return SimpleNamespace(
        **vision,
        attention_heads=vision["num_attention_heads"],
        layer_norm_eps=vision["rms_norm_eps"],
        _attn_implementation=ATTENTION,
    )


def rename(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    renamed = {}
    for name, tensor in weights.items():
        for pattern, target in RENAMES:
            name = re.sub(pattern, target, name)
        renamed[name] = tensor
    return renamed


class ReferenceTower(nn.Module):
    """HYV4FlashVisionTransformer's modules and packed forward, written out without its PreTrainedModel base."""

    def __init__(self, code: dict, config: SimpleNamespace):
        super().__init__()
        self.config = config
        self.embeddings = code["HYV4FlashVisionPatchEmbed"](config)
        self.layers = nn.ModuleList(
            code["HYV4FlashVisionBlock"](config)
            for _ in range(config.num_hidden_layers)
        )
        self.patch_merger = code["HYV4FlashVisionPatchMerger"](config)

    def forward(self, pixel_values, grid_thw, depths):
        hidden = self.embeddings(pixel_values, grid_thw)
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(
            grid_thw, self.config, kwargs={}
        )
        captured = {}
        for index, layer in enumerate(self.layers, start=1):
            hidden = layer(
                hidden,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                attention_mask=None,
            )
            if index in depths:
                captured[index] = hidden
        sizes = grid_thw.prod(dim=-1).tolist()
        merged = []
        handle = self.patch_merger.before_rms.register_forward_hook(
            lambda _m, _a, output: merged.append(output)
        )
        try:
            projected = [
                self.patch_merger(item, size=(h, w))
                for (_, h, w), item in zip(
                    grid_thw.tolist(), hidden.split(sizes, dim=1)
                )
            ]
        finally:
            handle.remove()
        return captured, merged, projected


def reference_tower(code, config, weights, device, dtype) -> ReferenceTower:
    with torch.device("meta"):
        model = ReferenceTower(code, config)
    model.load_state_dict(rename(weights), strict=True, assign=True)
    return model.to(device, dtype).eval()


def validate(
    device: str = "cpu", dtype: torch.dtype = torch.float32, bundle=None
) -> dict:
    bundle = hy.open_bundle(bundle)
    parent = hy.source_weights()
    release = load_file(bundle / "model.safetensors")
    report = {
        "source": f"{hy.SOURCE_REPO}@{hy.SOURCE_REVISION}",
        "reference": f"hy_v4_flash vision classes from https://github.com/{REFERENCE[0]}/pull/{REFERENCE[1]} at {REFERENCE[2]}, {ATTENTION} attention, on the parent's tensors",
        "weights": {
            "tower": weights_match(
                release, parent, hy.TOWER_TENSORS + hy.PROJECTOR_TENSORS
            )
        },
    }
    code = reference_code()
    config = reference_config()
    processor = hy.source_json("processor_config.json")["image_processor"]
    factor = processor["patch_size"] * processor["merge_size"]
    resized = code["smart_resize"](
        hy.RESOLUTION,
        hy.RESOLUTION,
        factor,
        processor["min_pixels"],
        processor["max_pixels"],
    )
    if resized != (hy.RESOLUTION, hy.RESOLUTION):
        raise ValueError(
            f"the parent processor would resize the bench square to {resized}"
        )

    adapter = hy.Hy4FlashAdapter(device=device, dtype=dtype, bundle=bundle)
    images = fixed_images()
    pixels = adapter.collate([adapter.preprocess()(image) for image in images])[
        "pixel_values"
    ]
    ids = [str(i) for i in range(len(images))]
    actual = {
        f"{b.stage}:{b.layer_index}": b.tokens
        for b in adapter.extract({"pixel_values": pixels}, ids)
    }

    patches, grid_h, grid_w = code["patchify"](
        None,
        pixels,
        processor["patch_size"],
        processor["merge_size"],
        processor["temporal_patch_size"],
    )
    grid_thw = torch.tensor([[1, grid_h, grid_w]] * len(images))
    reference = reference_tower(code, config, parent, device, dtype)
    with torch.inference_mode():
        captured, merged, projected = reference(
            patches.flatten(0, 1).to(device, dtype),
            grid_thw.to(device),
            set(adapter.depth_points()),
        )
    width, (rows, cols) = config.hidden_size, (grid_h // 2, grid_w // 2)
    expected = {
        f"tower:{layer}": tokens.reshape(len(images), grid_h * grid_w, width)
        for layer, tokens in captured.items()
    }
    expected[f"merged:{adapter.num_layers}"] = torch.cat(
        [
            F.unfold(
                m.reshape(grid_h, grid_w, width).permute(2, 0, 1)[None], 2, stride=2
            ).transpose(1, 2)
            for m in merged
        ]
    )
    expected[f"projected:{adapter.num_layers}"] = torch.cat(
        [
            p.reshape(1, rows, cols + 1, -1)[:, :, :cols].reshape(1, rows * cols, -1)
            for p in projected
        ]
    )
    report["forward"] = {
        key: compare(actual[key], expected[key].cpu()) for key in expected
    }

    released = AutoModel.from_pretrained(
        bundle, local_files_only=True, trust_remote_code=True, dtype=dtype
    )
    with torch.inference_mode():
        pooled = released.to(device).eval()(pixels.to(device, dtype)).pooler_output
    report["auto_model_projected"] = compare(pooled.cpu(), torch.cat(projected).cpu())

    tolerance = TOLERANCE[dtype]
    cells = [*report["forward"].values(), report["auto_model_projected"]]
    passed = report["weights"]["tower"]["bit_exact"] and all(
        cell["finite"] and cell["relative_l2"] <= tolerance for cell in cells
    )
    return {**report, "tolerance": tolerance, "passed": passed}
