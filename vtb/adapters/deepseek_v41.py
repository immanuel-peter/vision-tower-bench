from collections.abc import Iterator
from functools import lru_cache, partial
from math import isqrt
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from vtb.adapters.base import Inputs
from vtb.adapters.common import HALF, PatchPreprocess, default_device, depth_points, make_batch
from vtb.feature_batch import FeatureBatch
from vtb.provenance import open_release
from vtb.shards import load_prefixed

MODEL_ID = "immanuelpeter/DeepSeek-ViT"
REVISION = "9783c022bd626549c6c20460cad37edcc0490f0f"

SOURCE_REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
SOURCE_REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
VISION_SHARD = "model-00001-of-00048.safetensors"
TOWER_PREFIX = "vision."
ALIGNER_PREFIX = "aligner."
PATCH_SIZE = 14
DOWNSAMPLE = 3

ARGS = SimpleNamespace(
    vision_dim=1024,
    vision_n_heads=16,
    vision_n_layers=32,
    vision_inter_dim=2816,
    vision_patch_size=PATCH_SIZE,
    vision_downsample_ratio=DOWNSAMPLE,
    vision_rope_theta=10000,
    dim=5120,
)


@lru_cache(8)
def get_vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class PatchEmbed(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.proj = nn.Linear(3 * args.vision_patch_size**2, args.vision_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.vision_n_heads
        self.head_dim = args.vision_dim // args.vision_n_heads
        self.wqkv = nn.Linear(args.vision_dim, 3 * args.vision_dim)
        self.wo = nn.Linear(args.vision_dim, args.vision_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        q, k, v = (t.view(n, self.n_heads, self.head_dim) for t in self.wqkv(x).chunk(3, dim=-1))
        q = apply_rotary(q, cos.to(device=x.device), sin.to(device=x.device))
        k = apply_rotary(k, cos.to(device=x.device), sin.to(device=x.device))
        o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        return self.wo(o.transpose(0, 1).reshape(n, -1))


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.w1 = nn.Linear(args.vision_dim, 2 * args.vision_inter_dim, bias=False)
        self.w2 = nn.Linear(args.vision_inter_dim, args.vision_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.norm1 = RMSNorm(args.vision_dim)
        self.attn = Attention(args)
        self.norm2 = RMSNorm(args.vision_dim)
        self.mlp = MLP(args)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    def __init__(self, args=ARGS):
        super().__init__()
        self.rope_dim = args.vision_dim // args.vision_n_heads // 2
        self.rope_theta = args.vision_rope_theta
        self.patch_embed = PatchEmbed(args)
        self.blocks = nn.ModuleList([Block(args) for _ in range(args.vision_n_layers)])
        self.norm = RMSNorm(args.vision_dim)

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        x = self.patch_embed(patches)
        cos, sin = get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class Aligner(nn.Module):
    def __init__(self, args=ARGS):
        super().__init__()
        self.downsample_ratio = args.vision_downsample_ratio
        in_dim = args.vision_dim * self.downsample_ratio**2
        self.w1 = nn.Linear(in_dim, args.dim)
        self.w2 = nn.Linear(args.dim, args.dim)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))

    def unfold(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        return F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)


Preprocess = partial(PatchPreprocess, mean=HALF, std=HALF, patch_size=PATCH_SIZE, flatten=False)


def collate(samples: list[torch.Tensor]) -> Inputs:
    side = isqrt(samples[0].shape[0])
    return {"patches": torch.stack(samples), "grid": torch.tensor([side, side])}


def load_source_parts(dtype: torch.dtype = torch.bfloat16) -> tuple[ViT, Aligner]:
    tower = ViT()
    weights = load_prefixed(SOURCE_REPO, [VISION_SHARD], TOWER_PREFIX, revision=SOURCE_REVISION)
    tower.load_state_dict({name: tensor.to(dtype) for name, tensor in weights.items()})
    aligner = Aligner()
    aligner.load_state_dict({
        name: tensor.to(dtype)
        for name, tensor in load_prefixed(
            SOURCE_REPO, [VISION_SHARD], ALIGNER_PREFIX, revision=SOURCE_REVISION
        ).items()
    })
    return tower.to(dtype).eval(), aligner.to(dtype).eval()


def load_parts(dtype: torch.dtype) -> tuple[ViT, Aligner]:
    bundle = open_release(MODEL_ID, REVISION, SOURCE_REPO, SOURCE_REVISION)
    tower = ViT()
    tower.load_state_dict({
        name: tensor.to(dtype) for name, tensor in load_file(bundle / "model.safetensors").items()
    })
    aligner = Aligner()
    aligner.load_state_dict({
        name: tensor.to(dtype) for name, tensor in load_file(bundle / "projector.safetensors").items()
    })
    return tower.to(dtype).eval(), aligner.to(dtype).eval()


class DeepSeekV41Adapter:
    model_id = MODEL_ID
    default_resolution = 448
    stages = ("tower", "merged", "projected")
    collate = staticmethod(collate)

    def __init__(self, resolution: int | None = None, dtype: torch.dtype = torch.bfloat16, device: str | None = None):
        self.resolution = self.default_resolution if resolution is None else resolution
        self.device = default_device() if device is None else device
        self.dtype = dtype
        tower, aligner = load_parts(dtype)
        self.model = tower.to(self.device)
        self.aligner = aligner.to(self.device)
        self.num_layers = len(self.model.blocks)

    def preprocess(self) -> PatchPreprocess:
        return Preprocess(self.resolution)

    def depth_points(self, n: int = 8) -> list[int]:
        return depth_points(self.num_layers, n)

    @torch.inference_mode()
    def extract(self, inputs: Inputs, image_ids: list[str]) -> Iterator[FeatureBatch]:
        points = self.depth_points()
        n_h, n_w = int(inputs["grid"][0]), int(inputs["grid"][1])
        rows = len(image_ids)
        width = ARGS.vision_dim
        towers: dict[int, list[torch.Tensor]] = {layer: [] for layer in points}
        last: list[torch.Tensor] = []
        merged: list[torch.Tensor] = []
        projected: list[torch.Tensor] = []

        for patches in inputs["patches"]:
            captured: dict[int, torch.Tensor] = {}
            handles = []

            def capture(layer: int):
                def hook(_module, _args, output):
                    tokens = output[0] if isinstance(output, tuple) else output
                    captured[layer] = tokens.detach()

                return hook

            for layer in points:
                handles.append(self.model.blocks[layer - 1].register_forward_hook(capture(layer)))
            try:
                tokens = self.model(patches.to(self.device, self.dtype), n_h, n_w)
            finally:
                for handle in handles:
                    handle.remove()
            for layer in points:
                towers[layer].append(captured[layer])
            last.append(tokens)
            merged.append(self.aligner.unfold(tokens, n_h, n_w))
            projected.append(self.aligner(tokens, n_h, n_w))

        for layer in points:
            yield make_batch(self, torch.stack(towers[layer]).view(rows, -1, width), image_ids, "tower", layer)
        yield make_batch(self, torch.stack(merged), image_ids, "merged", self.num_layers)
        yield make_batch(self, torch.stack(projected), image_ids, "projected", self.num_layers)
