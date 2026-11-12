"""Execute pinned official vLLM vision source without an engine or language model.

Single-rank linear/conv dependencies use PyTorch. Official packed Triton rotary
and TORCH_SDPA source execute directly when selected. The explicit math control
retains the bench's masked SDPA arithmetic. Upstream vision, positional
initialization, merge and Projector code executes unchanged. This reference
never imports the bench adapter's modeling code.
"""

import ast
import logging
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import einops
import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import GELUActivation


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
            return self.official_native(x, cos, sin, is_neox_style=False, enable_fp32_compute=True)
        original = x.dtype
        paired = x.float().reshape(*x.shape[:-1], -1, 2)
        real, imag = paired.unbind(-1)
        cos, sin = cos[:, None], sin[:, None]
        return torch.stack((real * cos - imag * sin, real * sin + imag * cos), -1).flatten(-2).to(original)


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
                mask = torch.ones((1, end - start, end - start), dtype=torch.bool, device=q.device)
                output = F.scaled_dot_product_attention(queries, keys, values, mask, scale=self.scale).unsqueeze(0)
            else:
                queries, keys, values = chunk
                scores = (queries @ keys.transpose(-2, -1)) / (queries.shape[-1] ** 0.5)
                output = scores.softmax(-1, dtype=torch.float32).to(queries.dtype) @ values
            result.append(output.transpose(1, 2))
        return torch.cat(result, 1)


def load_reference(source: Path, attention: str = "official_sdpa", rotary: str = "native"):
    """Run upstream definitions, replacing only external engine dependencies."""
    Attention.implementation = attention
    official_root = source.parents[2]
    common = official_root / "model_executor/layers/rotary_embedding/common.py"
    if common.exists():
        parsed = ast.parse(common.read_text())
        cls = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "ApplyRotaryEmb")
        native = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward_static")
        native.decorator_list = []
        scope = {"torch": torch}
        exec(compile(ast.Module(body=[native], type_ignores=[]), str(common), "exec"), scope)
        Rotary.official_native = staticmethod(scope["forward_static"])
    if attention == "official_sdpa":
        wrappers = official_root / "v1/attention/ops/vit_attn_wrappers.py"
        functions = ast.parse(wrappers.read_text())
        functions.body = [node for node in functions.body if isinstance(node, ast.FunctionDef)
                          and node.name in {"apply_sdpa", "torch_sdpa_wrapper"}]
        scope = {"torch": torch, "F": F, "einops": einops,
                 "current_platform": SimpleNamespace(is_zen_cpu=lambda: False, is_rocm=lambda: False),
                 "gpu_sync_allowed": nullcontext}
        exec(compile(functions, str(wrappers), "exec"), scope)
        Attention.official_sdpa = staticmethod(scope["torch_sdpa_wrapper"])
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("vllm")
    )]
    namespace = {
        "__name__": "independent_pinned_vllm_vision",
        "divide": lambda n, d: n // d,
        "get_tensor_model_parallel_world_size": lambda: 1,
        "init_logger": logging.getLogger,
        "get_act_fn": lambda name: nn.GELU(approximate="tanh" if name == "gelu_pytorch_tanh" else "none"),
        "MMEncoderAttention": Attention,
        "Conv2dLayer": Conv,
        "IdentityInputNorm": InputNorm,
        "ColumnParallelLinear": Linear, "RowParallelLinear": Linear,
        "ReplicatedLinear": Linear, "QKVParallelLinear": QKVLinear,
        "QuantizationConfig": type(None), "ApplyRotaryEmb": Rotary,
        "maybe_prefix": lambda prefix, name: f"{prefix}.{name}" if prefix else name,
        "is_vit_use_data_parallel": lambda *args: False,
        "KimiK3VisionConfig": SimpleNamespace, "HAS_TRITON": False,
    }
    if rotary == "official_packed":
        import triton
        import triton.language as tl
        path = official_root / "model_executor/layers/rotary_embedding/packed_qk_rope.py"
        kernel = ast.parse(path.read_text())
        kernel.body = [node for node in kernel.body if not isinstance(node, ast.ImportFrom)]
        scope = {"__name__": "independent_official_packed_rope", "torch": torch, "triton": triton, "tl": tl}
        exec(compile(kernel, str(path), "exec"), scope)
        namespace.update(HAS_TRITON=True, packed_qk_rope_=scope["packed_qk_rope_"])
    exec(compile(tree, str(source), "exec"), namespace)
    return SimpleNamespace(**namespace)
