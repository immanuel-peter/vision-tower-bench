"""Hy4 and Hy4 flash against the vision classes of the Transformers sources they were written from.

Hy4: the reference is ``HYV4VLVisionModel`` from the Transformers source tree at REFERENCE_REVISION,
which must be on PYTHONPATH.

Hy4 flash: the PR is not merged, so the reference compiles its vision classes from the SHA-256-checked
sources of transformers PR #50903 at a pinned commit, with transformers' own attention and helper
functions. It loads the parent's tensors through the PR's vision renames, written out here, and runs
the packed per-image forward that ``HYV4FlashVisionTransformer`` runs. The pixels go through the PR image
processor's ``patchify``; its ``smart_resize`` must leave the bench's 512 px square unchanged.
The merged reference unfolds the RMSNorm output with ``F.unfold`` instead of the adapter's reshape.

Both also load the release through ``AutoModel`` and compare its projected tokens.
"""

import math
import re
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModel

from vtb.adapters.hy4 import (
    HY4,
    HY4_FLASH,
    RESOLUTION,
    Hy4Adapter,
    Hy4FlashAdapter,
    source_json,
    source_weights,
)
from vtb.parity import (
    compare,
    compile_nodes,
    features,
    file_sha256,
    fixed_images,
    pinned_sources,
    weights_match,
)

REFERENCE_REVISION = "1031277cef60200fda98628e9adc8fb15f0e1549"
REFERENCE_SHA256 = "b93fad35ea419bd02eee77fde903564abd5a058d6f072e03a1be8f7cf645418c"
HY4_RENAMES = {
    ".input_layernorm.": ".layer_norm1.",
    ".post_attention_layernorm.": ".layer_norm2.",
    ".mlp.dense_h_to_4h.": ".mlp.fc1.",
    ".mlp.dense_4h_to_h.": ".mlp.fc2.",
    "perceive.norm.": "patch_merger.before_rms.",
    "perceive.proj.0.": "patch_merger.proj_conv.",
    "perceive.proj.2.": "patch_merger.proj_out.",
    "perceive.image_newline": "patch_merger.image_newline",
}

FLASH_REFERENCE = (
    "huggingface/transformers",
    50903,
    "c2ec850510a37978ea83df226589623d5f2be776",
)
FLASH_MODELING = "src/transformers/models/hy_v4_flash/modeling_hy_v4_flash.py"
FLASH_IMAGE_PROCESSING = "src/transformers/models/hy_v4_vl/image_processing_hy_v4_vl.py"
FLASH_SOURCES = {
    FLASH_MODELING: "43341929e4bec215c224fc837b24b534b4de41954992673a0655a65060e38b9b",
    FLASH_IMAGE_PROCESSING: "3e2063c2f2d197ad55b01adc4058844e6cbaa72a39a516d01982fe44207cb954",
}
FLASH_CLASSES = {
    "repeat_kv",
    "eager_attention_forward",
    "HYV4FlashRMSNorm",
    "HYV4FlashVisionPatchMerger",
    "HYV4FlashVisionPatchEmbed",
    "HYV4FlashVisionAttention",
    "HYV4FlashVisionMLP",
    "HYV4FlashVisionBlock",
}
# The PR's vision rules in conversion_mapping.py, applied after the `vit.` prefix is removed.
FLASH_RENAMES = (
    (r"mlp\.dense_h_to_4h", "mlp.fc1"),
    (r"mlp\.dense_4h_to_h", "mlp.fc2"),
    (r"^layers\.(\d+)\.input_layernorm\.", r"layers.\1.layer_norm1."),
    (r"^layers\.(\d+)\.post_attention_layernorm\.", r"layers.\1.layer_norm2."),
    (r"^perceive\.norm\.", "perceive.before_rms."),
    (r"^perceive\.proj\.0\.", "perceive.proj_conv."),
    (r"^perceive\.proj\.2\.", "perceive.proj_out."),
    (r"^perceive\.", "patch_merger."),
)
FLASH_ATTENTION = "sdpa"
# Hy4 compares max abs. Hy4 flash compares relative L2: a few of its tokens carry activations near 1000,
# so one BF16 ulp there is 4 to 8.
TOLERANCE = {
    HY4.name: {torch.float32: 2e-4, torch.bfloat16: 0.125},
    HY4_FLASH.name: {torch.float32: 1e-4, torch.bfloat16: 0.03},
}


def hy4_reference_name(name):
    for old, new in HY4_RENAMES.items():
        name = name.replace(old, new)
    return name


def load_hy4_reference(weights, config, device, dtype):
    try:
        from transformers.models.hy_v4_vl import modeling_hy_v4_vl as reference
    except ImportError as error:
        raise RuntimeError(
            f"Hy4 parity requires Transformers {REFERENCE_REVISION}"
        ) from error
    if file_sha256(Path(reference.__file__)) != REFERENCE_SHA256:
        raise RuntimeError(f"Hy4 parity requires Transformers {REFERENCE_REVISION}")
    values = {
        name: getattr(config, name)
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "out_hidden_size",
            "patch_size",
            "max_image_size",
            "rms_norm_eps",
        )
    }
    config = reference.HYV4VLVisionConfig(
        **values, interpolate_mode="bilinear", attn_implementation="sdpa"
    )
    with torch.device("meta"):
        model = reference.HYV4VLVisionModel(config)
    model.load_state_dict(
        {hy4_reference_name(name): tensor for name, tensor in weights.items()},
        strict=True,
        assign=True,
    )
    return model.to(device=device, dtype=dtype).eval()


@torch.inference_mode()
def hy4_reference_features(parent, adapter, images):
    pixels = adapter.collate([adapter.preprocess()(image) for image in images])[
        "pixel_values"
    ]
    patches = parent.config.patch_size
    side = adapter.resolution // patches
    count = len(images)
    packed = (
        pixels.reshape(count, 3, side, patches, side, patches)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(-1, 3 * patches**2)
    )
    grid = torch.tensor([[1, side, side]] * count, device=adapter.device)
    expected = {}
    width = parent.config.hidden_size
    layers = adapter.depth_points()
    handles = []
    normalized = []

    def capture(layer):
        def hook(_module, _args, output):
            expected[f"tower:{layer}"] = output.reshape(count, side**2, width)

        return hook

    for layer in layers:
        handles.append(parent.layers[layer - 1].register_forward_hook(capture(layer)))
    handles.append(
        parent.patch_merger.before_rms.register_forward_hook(
            lambda _module, _args, output: normalized.append(output)
        )
    )
    try:
        output = parent(packed.to(adapter.device, adapter.dtype), grid)
    finally:
        for handle in handles:
            handle.remove()
    merge_side = side // 2
    norm_grid = torch.cat(normalized, dim=0).reshape(count, side, side, width)
    expected[f"merged:{layers[-1]}"] = torch.stack(
        [
            norm_grid[:, row : row + 2, col : col + 2].reshape(count, 4 * width)
            for row in range(0, side, 2)
            for col in range(0, side, 2)
        ],
        dim=1,
    )
    delimited = output.pooler_output.reshape(count, merge_side, merge_side + 1, -1)
    expected[f"projected:{layers[-1]}"] = delimited[:, :, :-1].reshape(
        count, merge_side**2, -1
    )
    return expected


def validate_hy4(device, dtype, bundle):
    bundle = HY4.open(bundle)
    weights = source_weights(HY4)
    exact = weights_match(
        load_file(bundle / "model.safetensors"),
        weights,
        HY4.tower_tensors + HY4.projector_tensors,
    )
    adapter = Hy4Adapter(device=device, dtype=dtype, bundle=bundle)
    parent = load_hy4_reference(weights, adapter.model.config, device, dtype)
    images = fixed_images()
    actual = features(adapter, images)
    expected = hy4_reference_features(parent, adapter, images)
    if actual.keys() != expected.keys():
        raise ValueError(
            "Hy4 release and reference returned different Stages or depths"
        )
    forward = {key: compare(actual[key], expected[key]) for key in expected}
    tolerance = TOLERANCE[HY4.name][dtype]
    pixels = adapter.collate([adapter.preprocess()(image) for image in images])[
        "pixel_values"
    ].to(device, dtype)
    released = AutoModel.from_pretrained(
        bundle, local_files_only=True, trust_remote_code=True, dtype=dtype
    )
    with torch.inference_mode():
        last = adapter.num_layers
        _, _, projected = released.to(device).eval()(pixels, depths=(last,))
    auto_model = compare(projected.cpu(), expected[f"projected:{last}"].cpu())
    return {
        "weights": {"vision": exact},
        "forward": forward,
        "auto_model_projected": auto_model,
        "reference_revision": REFERENCE_REVISION,
        "tolerance": tolerance,
        "passed": exact["bit_exact"]
        and all(
            cell["finite"] and cell["max_abs"] <= tolerance
            for cell in (*forward.values(), auto_model)
        ),
    }


def identity(*args, **_kwargs):
    if len(args) == 1 and callable(args[0]):
        return args[0]
    return identity


def flash_reference_code() -> dict:
    """The PR's vision classes and image-processor helpers, compiled without importing the PR."""
    from transformers.activations import ACT2FN
    from transformers.modeling_layers import GradientCheckpointingLayer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.processing_utils import Unpack
    from transformers.utils import TransformersKwargs
    from transformers.utils.generic import get_max_seqlen, is_flash_attention_requested
    from transformers.utils.import_utils import torch_compilable_check
    from transformers.vision_utils import get_vision_attention_seqlens

    repo, _, head = FLASH_REFERENCE
    root = pinned_sources(repo, head, FLASH_SOURCES)
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
        "get_vision_attention_seqlens": get_vision_attention_seqlens,
        "HYV4FlashVisionConfig": SimpleNamespace,
    }
    compile_nodes(root / FLASH_MODELING, FLASH_CLASSES, namespace)
    compile_nodes(root / FLASH_IMAGE_PROCESSING, {"smart_resize"}, namespace)
    compile_nodes(
        root / FLASH_IMAGE_PROCESSING,
        {"patchify"},
        namespace,
        within="HYV4VLImageProcessor",
    )
    return namespace


def flash_reference_config() -> SimpleNamespace:
    """The parent's own vision_config, with the PR's attribute_map aliases."""
    vision = source_json(HY4_FLASH, "config.json")["vision_config"]
    return SimpleNamespace(
        **vision,
        attention_heads=vision["num_attention_heads"],
        layer_norm_eps=vision["rms_norm_eps"],
        _attn_implementation=FLASH_ATTENTION,
    )


def flash_rename(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    renamed = {}
    for name, tensor in weights.items():
        for pattern, target in FLASH_RENAMES:
            name = re.sub(pattern, target, name)
        renamed[name] = tensor
    return renamed


class FlashReferenceTower(nn.Module):
    """HYV4FlashVisionTransformer's modules and packed forward, written out without its PreTrainedModel base."""

    def __init__(self, code: dict, config: SimpleNamespace):
        super().__init__()
        self.config = config
        self.seqlens = code["get_vision_attention_seqlens"]
        self.embeddings = code["HYV4FlashVisionPatchEmbed"](config)
        self.layers = nn.ModuleList(
            code["HYV4FlashVisionBlock"](config)
            for _ in range(config.num_hidden_layers)
        )
        self.patch_merger = code["HYV4FlashVisionPatchMerger"](config)

    def forward(self, pixel_values, grid_thw, depths):
        hidden = self.embeddings(pixel_values, grid_thw)
        cu_seqlens, max_seqlen = self.seqlens(grid_thw, self.config, kwargs={})
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


def flash_reference_tower(code, config, weights, device, dtype) -> FlashReferenceTower:
    with torch.device("meta"):
        model = FlashReferenceTower(code, config)
    model.load_state_dict(flash_rename(weights), strict=True, assign=True)
    return model.to(device, dtype).eval()


def validate_flash(device, dtype, bundle) -> dict:
    bundle = HY4_FLASH.open(bundle)
    parent = source_weights(HY4_FLASH)
    release = load_file(bundle / "model.safetensors")
    report = {
        "source": f"{HY4_FLASH.repo}@{HY4_FLASH.revision}",
        "reference": f"hy_v4_flash vision classes from https://github.com/{FLASH_REFERENCE[0]}/pull/{FLASH_REFERENCE[1]} at {FLASH_REFERENCE[2]}, {FLASH_ATTENTION} attention, on the parent's tensors",
        "weights": {
            "tower": weights_match(
                release, parent, HY4_FLASH.tower_tensors + HY4_FLASH.projector_tensors
            )
        },
    }
    code = flash_reference_code()
    config = flash_reference_config()
    processor = source_json(HY4_FLASH, "processor_config.json")["image_processor"]
    factor = processor["patch_size"] * processor["merge_size"]
    resized = code["smart_resize"](
        RESOLUTION,
        RESOLUTION,
        factor,
        processor["min_pixels"],
        processor["max_pixels"],
    )
    if resized != (RESOLUTION, RESOLUTION):
        raise ValueError(
            f"the parent processor would resize the bench square to {resized}"
        )

    adapter = Hy4FlashAdapter(device=device, dtype=dtype, bundle=bundle)
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
    reference = flash_reference_tower(code, config, parent, device, dtype)
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

    tolerance = TOLERANCE[HY4_FLASH.name][dtype]
    cells = [*report["forward"].values(), report["auto_model_projected"]]
    passed = report["weights"]["tower"]["bit_exact"] and all(
        cell["finite"] and cell["relative_l2"] <= tolerance for cell in cells
    )
    return {**report, "tolerance": tolerance, "passed": passed}


def validate(
    checkpoint=HY4,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    *,
    bundle=None,
) -> dict:
    run = {HY4.name: validate_hy4, HY4_FLASH.name: validate_flash}[checkpoint.name]
    return run(device, dtype, bundle)
