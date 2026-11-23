"""Compare Hy4 with the vision-only reference from pinned Transformers.

Use the source tree at REFERENCE_REVISION on PYTHONPATH for this check.
"""

from pathlib import Path

import torch
from safetensors.torch import load_file

from vtb.adapters.hy4 import (
    MODEL_ID,
    SOURCE_REPO,
    SOURCE_REVISION,
    VISUAL_TENSORS,
    Hy4Adapter,
    source_weights,
)
from vtb.parity import compare, features, file_sha256, fixed_images, weights_match
from vtb.provenance import open_release

REFERENCE_REVISION = "1031277cef60200fda98628e9adc8fb15f0e1549"
REFERENCE_SHA256 = "b93fad35ea419bd02eee77fde903564abd5a058d6f072e03a1be8f7cf645418c"
TOLERANCE = {torch.float32: 2e-4, torch.bfloat16: 0.125}
RENAMES = {
    ".input_layernorm.": ".layer_norm1.",
    ".post_attention_layernorm.": ".layer_norm2.",
    ".mlp.dense_h_to_4h.": ".mlp.fc1.",
    ".mlp.dense_4h_to_h.": ".mlp.fc2.",
    "perceive.norm.": "patch_merger.before_rms.",
    "perceive.proj.0.": "patch_merger.proj_conv.",
    "perceive.proj.2.": "patch_merger.proj_out.",
    "perceive.image_newline": "patch_merger.image_newline",
}


def reference_name(name):
    for old, new in RENAMES.items():
        name = name.replace(old, new)
    return name


def load_reference(weights, config, device, dtype):
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
        {reference_name(name): tensor for name, tensor in weights.items()},
        strict=True,
        assign=True,
    )
    return model.to(device=device, dtype=dtype).eval()


@torch.inference_mode()
def reference_features(parent, adapter, images):
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


def validate(device="cpu", dtype=torch.float32):
    bundle = open_release(MODEL_ID, SOURCE_REPO, SOURCE_REVISION)
    weights = source_weights()
    exact = weights_match(
        load_file(bundle / "model.safetensors"), weights, VISUAL_TENSORS
    )
    adapter = Hy4Adapter(device=device, dtype=dtype, bundle=bundle)
    parent = load_reference(weights, adapter.model.config, device, dtype)
    images = fixed_images()
    actual = features(adapter, images)
    expected = reference_features(parent, adapter, images)
    if actual.keys() != expected.keys():
        raise ValueError(
            "Hy4 release and reference returned different Stages or depths"
        )
    forward = {key: compare(actual[key], expected[key]) for key in expected}
    return {
        "weights": {"vision": exact},
        "forward": forward,
        "reference_revision": REFERENCE_REVISION,
        "tolerance": TOLERANCE[dtype],
        "passed": exact["bit_exact"]
        and all(
            cell["finite"] and cell["max_abs"] <= TOLERANCE[dtype]
            for cell in forward.values()
        ),
    }
