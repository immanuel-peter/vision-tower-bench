"""Kimi K3 and K3.1 Pro: the published MoonViT against pinned official vLLM vision source.

The reference fetches vLLM's own vision, merge and Projector code at a pinned commit and
executes it with only engine dependencies replaced, so it never imports the bench's
modeling code. Single-rank linear and conv layers use PyTorch. The official packed Triton
rotary and TORCH_SDPA run directly when selected; ``sdpa_math`` keeps the bench's masked SDPA.
"""

import ast
import json
import logging
import os
import urllib.request
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import einops
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from vtb.adapters.moonvit_v2 import (
    K3,
    K31_PRO,
    PROJECTOR_PREFIX,
    PROJECTOR_TENSORS,
    TOWER_PREFIX,
    TOWER_TENSORS,
    Checkpoint,
    KimiK31ProAdapter,
    MoonViTV2Adapter,
    collate,
    discover_shards,
    read_parent,
)
from vtb.parity import file_sha256, weights_match
from vtb.shards import cache_root

VLLM_REVISION = "1ecc9cc4ce6626352df39e4364579ec376c10761"
VLLM_FILES = (
    "vllm/model_executor/models/registry.py",
    "vllm/models/kimi_k3/__init__.py",
    "vllm/models/kimi_k3/nvidia/model.py",
    "vllm/model_executor/models/kimi_k25_vit.py",
    "vllm/model_executor/layers/attention/mm_encoder_attention.py",
    "vllm/model_executor/layers/rotary_embedding/common.py",
    "vllm/model_executor/layers/rotary_embedding/packed_qk_rope.py",
    "vllm/v1/attention/ops/vit_attn_wrappers.py",
)
# The FP32 run uses vLLM's own TORCH_SDPA; the BF16 run uses masked SDPA math, which the
# bench matches bit for bit. Production BF16 TORCH_SDPA disagrees, as the model cards say.
CONFIGS = (("official_sdpa", torch.float32), ("sdpa_math", torch.bfloat16))
ADAPTERS = {K3: MoonViTV2Adapter, K31_PRO: KimiK31ProAdapter}


class Linear(nn.Linear):
    def __init__(self, input_size, output_size, bias=True, **kwargs):
        super().__init__(input_size, output_size, bias=bias)

    def forward(self, x):
        return super().forward(x), None


class QKVLinear(Linear):
    def __init__(self, hidden_size, head_size, total_num_heads, **kwargs):
        super().__init__(hidden_size, 3 * head_size * total_num_heads, **kwargs)


class Conv(nn.Conv2d):
    forward_native = nn.Conv2d.forward


class InputNorm(nn.Module):
    def forward(self, x, dtype):
        return x.to(dtype)


class Rotary(nn.Module):
    official_native = None

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, cos, sin):
        if self.official_native is not None:
            return self.official_native(
                x, cos, sin, is_neox_style=False, enable_fp32_compute=True
            )
        original = x.dtype
        paired = x.float().reshape(*x.shape[:-1], -1, 2)
        real, imag = paired.unbind(-1)
        cos, sin = cos[:, None], sin[:, None]
        return (
            torch.stack((real * cos - imag * sin, real * sin + imag * cos), -1)
            .flatten(-2)
            .to(original)
        )


class Attention(nn.Module):
    implementation = "sdpa_math"
    official_sdpa = None

    def __init__(self, scale, **kwargs):
        super().__init__()
        self.scale = scale
        self.attn_backend = None

    @staticmethod
    def maybe_compute_seq_lens(backend, cumulative, device):
        return None

    @staticmethod
    def compute_max_seqlen(backend, cumulative):
        return int(np.diff(cumulative).max())

    @staticmethod
    def maybe_recompute_cu_seqlens(backend, cumulative, hidden, tp, device):
        return torch.as_tensor(cumulative, dtype=torch.int32, device=device)

    def forward(self, q, k, v, cu_seqlens, **kwargs):
        if self.implementation == "official_sdpa":
            return self.official_sdpa(q, k, v, scale=self.scale, cu_seqlens=cu_seqlens)
        result = []
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            chunk = [a[:, start:end].transpose(1, 2) for a in (q, k, v)]
            if self.implementation == "sdpa":
                output = F.scaled_dot_product_attention(*chunk, scale=self.scale)
            elif self.implementation == "sdpa_math":
                queries, keys, values = [x.squeeze(0) for x in chunk]
                mask = torch.ones(
                    (1, end - start, end - start), dtype=torch.bool, device=q.device
                )
                output = F.scaled_dot_product_attention(
                    queries, keys, values, mask, scale=self.scale
                ).unsqueeze(0)
            else:
                queries, keys, values = chunk
                scores = (queries @ keys.transpose(-2, -1)) / (queries.shape[-1] ** 0.5)
                output = (
                    scores.softmax(-1, dtype=torch.float32).to(queries.dtype) @ values
                )
            result.append(output.transpose(1, 2))
        return torch.cat(result, 1)


def load_reference(
    source: Path, attention: str = "official_sdpa", rotary: str = "native"
):
    """Run upstream definitions, replacing only external engine dependencies."""
    Attention.implementation = attention
    official_root = source.parents[2]
    common = official_root / "model_executor/layers/rotary_embedding/common.py"
    if common.exists():
        parsed = ast.parse(common.read_text())
        cls = next(
            node
            for node in parsed.body
            if isinstance(node, ast.ClassDef) and node.name == "ApplyRotaryEmb"
        )
        native = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward_static"
        )
        native.decorator_list = []
        scope = {"torch": torch}
        exec(
            compile(ast.Module(body=[native], type_ignores=[]), str(common), "exec"),
            scope,
        )
        Rotary.official_native = staticmethod(scope["forward_static"])
    if attention == "official_sdpa":
        wrappers = official_root / "v1/attention/ops/vit_attn_wrappers.py"
        functions = ast.parse(wrappers.read_text())
        functions.body = [
            node
            for node in functions.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"apply_sdpa", "torch_sdpa_wrapper"}
        ]
        scope = {
            "torch": torch,
            "F": F,
            "einops": einops,
            "current_platform": SimpleNamespace(
                is_zen_cpu=lambda: False, is_rocm=lambda: False
            ),
            "gpu_sync_allowed": nullcontext,
        }
        exec(compile(functions, str(wrappers), "exec"), scope)
        Attention.official_sdpa = staticmethod(scope["torch_sdpa_wrapper"])
    tree = ast.parse(source.read_text())
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("vllm")
        )
    ]
    namespace = {
        "__name__": "independent_pinned_vllm_vision",
        "divide": lambda n, d: n // d,
        "get_tensor_model_parallel_world_size": lambda: 1,
        "init_logger": logging.getLogger,
        "get_act_fn": lambda name: nn.GELU(
            approximate="tanh" if name == "gelu_pytorch_tanh" else "none"
        ),
        "MMEncoderAttention": Attention,
        "Conv2dLayer": Conv,
        "IdentityInputNorm": InputNorm,
        "ColumnParallelLinear": Linear,
        "RowParallelLinear": Linear,
        "ReplicatedLinear": Linear,
        "QKVParallelLinear": QKVLinear,
        "QuantizationConfig": type(None),
        "ApplyRotaryEmb": Rotary,
        "maybe_prefix": lambda prefix, name: f"{prefix}.{name}" if prefix else name,
        "is_vit_use_data_parallel": lambda *args: False,
        "KimiK3VisionConfig": SimpleNamespace,
        "HAS_TRITON": False,
    }
    if rotary == "official_packed":
        import triton
        import triton.language as tl

        path = (
            official_root / "model_executor/layers/rotary_embedding/packed_qk_rope.py"
        )
        kernel = ast.parse(path.read_text())
        kernel.body = [
            node for node in kernel.body if not isinstance(node, ast.ImportFrom)
        ]
        scope = {
            "__name__": "independent_official_packed_rope",
            "torch": torch,
            "triton": triton,
            "tl": tl,
        }
        exec(compile(kernel, str(path), "exec"), scope)
        namespace.update(HAS_TRITON=True, packed_qk_rope_=scope["packed_qk_rope_"])
    exec(compile(tree, str(source), "exec"), namespace)
    return SimpleNamespace(**namespace)


def fetch_reference() -> Path:
    """Download only the pinned vLLM files the reference executes."""
    root = (
        Path(
            os.environ.get(
                "VTB_KIMI_REFERENCE", cache_root().parent / "vtb-kimi-reference"
            )
        )
        / VLLM_REVISION
    )
    for filename in VLLM_FILES:
        path = root / filename
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(
                urllib.request.urlopen(
                    f"https://raw.githubusercontent.com/vllm-project/vllm/{VLLM_REVISION}/{filename}"
                ).read()
            )
    if '"KimiK31ForConditionalGeneration"' not in (root / VLLM_FILES[0]).read_text():
        raise ValueError("pinned official registry does not register K3.1 Pro")
    return root / "vllm/model_executor/models/kimi_k25_vit.py"


def forward(
    checkpoint: Checkpoint,
    parent: dict,
    device: str,
    attention: str,
    dtype: torch.dtype,
) -> dict:
    source = fetch_reference()
    rotary = "official_packed" if str(device).startswith("cuda") else "native"
    reference = load_reference(source, attention, rotary)
    cfg = dict(
        json.loads(
            Path(
                hf_hub_download(
                    checkpoint.repo, "config.json", revision=checkpoint.revision
                )
            ).read_text()
        )["vision_config"]
    )
    for key in (
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_hidden_layers",
    ):
        cfg[key] = cfg["vt_" + key]
    cfg["hidden_act"] = cfg["activation_func"]
    config = SimpleNamespace(**cfg)
    tower = (
        reference.MoonViT3dPretrainedModel(config).to(dtype=dtype, device=device).eval()
    )
    tower.load_state_dict(parent["tower"], strict=True)
    projector = (
        reference.KimiK25MultiModalProjector(config, cfg["text_hidden_size"])
        .to(dtype=dtype, device=device)
        .eval()
    )
    mapping = {"proj.0.weight": "linear_1.weight", "proj.2.weight": "linear_2.weight"}
    projector.load_state_dict(
        {mapping.get(k, k): v for k, v in parent["projector"].items()}, strict=True
    )
    adapter = ADAPTERS[checkpoint](device=device, dtype=dtype)
    captured, handles = {}, []
    for depth in adapter.depth_points():
        module = (
            tower.encoder
            if depth == adapter.num_layers
            else tower.encoder.blocks[depth - 1]
        )
        handles.append(
            module.register_forward_hook(
                lambda _m, _a, y, depth=depth: captured.update({depth: y.detach()})
            )
        )
    atol, rtol = (0.125, 0.03) if dtype == torch.bfloat16 else (0.003, 0.001)
    cases = {}
    try:
        for seed in range(3):
            image = Image.fromarray(
                np.random.default_rng(seed).integers(
                    0, 256, (480, 640, 3), dtype=np.uint8
                )
            )
            inputs = collate([adapter.preprocess()(image)])
            with torch.inference_mode():
                merged = tower(
                    inputs["pixel_values"].to(device, dtype),
                    inputs["grid_thws"].to(device),
                )
                expected = {
                    ("tower", depth): value[None] for depth, value in captured.items()
                }
                expected["merged", adapter.num_layers] = torch.stack(merged).flatten(2)
                expected["projected", adapter.num_layers] = projector(merged[0])[None]
                for batch in adapter.extract(inputs, [str(seed)]):
                    target = expected[batch.stage, batch.layer_index].cpu()
                    delta = (batch.tokens.float() - target.float()).abs()
                    close = torch.isclose(batch.tokens, target, atol=atol, rtol=rtol)
                    cases[f"{seed}:{batch.stage}:{batch.layer_index}"] = {
                        "shape": list(target.shape),
                        "max_abs": float(delta.max()),
                        "mismatched": int((~close).sum()),
                        "passed": bool(
                            close.all() and torch.isfinite(batch.tokens).all()
                        ),
                    }
    finally:
        for handle in handles:
            handle.remove()
    exact = dtype != torch.bfloat16 or all(
        case["max_abs"] == 0 for case in cases.values()
    )
    return {
        "attention": attention,
        "rotary": rotary,
        "dtype": str(dtype),
        "reference_sha256": file_sha256(source),
        "cases": cases,
        "passed": exact and all(case["passed"] for case in cases.values()),
    }


def validate(
    checkpoint: Checkpoint, device: str = "cpu", dtype: torch.dtype | None = None
) -> dict:
    shards = discover_shards(checkpoint)
    parent = {
        part: read_parent(checkpoint, shards[prefix], prefix)
        for part, prefix in (("tower", TOWER_PREFIX), ("projector", PROJECTOR_PREFIX))
    }
    bundle = Path(snapshot_download(checkpoint.model_id))
    report = {
        "weights": {
            "tower": weights_match(
                load_file(bundle / "model.safetensors"), parent["tower"], TOWER_TENSORS
            ),
            "projector": weights_match(
                load_file(bundle / "projector.safetensors"),
                parent["projector"],
                PROJECTOR_TENSORS,
            ),
        }
    }
    report["forward"] = {
        attention: forward(checkpoint, parent, device, attention, config_dtype)
        for attention, config_dtype in CONFIGS
    }
    passed = all(cell["bit_exact"] for cell in report["weights"].values()) and all(
        run["passed"] for run in report["forward"].values()
    )
    return {**report, "passed": passed}
