"""Standalone Tencent Hy4 Tower and Projector with checkpoint-native names."""

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig, PreTrainedModel


class Hy4VisionConfig(PretrainedConfig):
    model_type = "vtb_hy4"

    def __init__(
        self,
        hidden_size=1536,
        intermediate_size=6144,
        num_hidden_layers=40,
        num_attention_heads=16,
        out_hidden_size=6144,
        patch_size=16,
        max_image_size=2048,
        rms_norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.out_hidden_size = out_hidden_size
        self.patch_size = patch_size
        self.max_image_size = max_image_size
        self.rms_norm_eps = rms_norm_eps


class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        patch = config.patch_size
        self.edge = config.max_image_size // patch
        self.patch_embedding = nn.Conv2d(3, config.hidden_size, patch, stride=patch)
        self.position_embedding = nn.Embedding(self.edge**2 + 1, config.hidden_size)

    def forward(self, pixels):
        height, width = pixels.shape[-2:]
        patch = self.patch_embedding.kernel_size[0]
        rows, cols = height // patch, width // patch
        patches = F.unfold(pixels, patch, stride=patch).transpose(1, 2)
        embedded = self.patch_embedding(patches.reshape(-1, 3, patch, patch))
        tokens = embedded.reshape(pixels.shape[0], rows * cols, -1)
        positions = self.position_embedding.weight[1:].T.reshape(
            1, -1, self.edge, self.edge
        )
        positions = F.interpolate(
            positions.float(),
            scale_factor=((rows + 0.1) / self.edge, (cols + 0.1) / self.edge),
            mode="bilinear",
            align_corners=False,
        )
        return tokens + positions.flatten(2).transpose(1, 2).to(tokens.dtype)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.heads = config.num_attention_heads
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.o_proj = nn.Linear(width, width)

    def forward(self, tokens):
        batch, length, width = tokens.shape
        q, k, v = [
            proj(tokens).view(batch, length, self.heads, -1).transpose(1, 2)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        ]
        attended = F.scaled_dot_product_attention(q, k, v)
        return self.o_proj(attended.transpose(1, 2).reshape(batch, length, width))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, tokens):
        return self.dense_4h_to_h(F.gelu(self.dense_h_to_4h(tokens)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = MLP(config)

    def forward(self, tokens):
        tokens = tokens + self.self_attn(self.input_layernorm(tokens))
        return tokens + self.mlp(self.post_attention_layernorm(tokens))


class RMSNorm(nn.RMSNorm):
    def forward(self, tokens):
        values = tokens.float()
        normalized = values * torch.rsqrt(
            values.square().mean(-1, keepdim=True) + self.eps
        )
        return self.weight * normalized.to(tokens.dtype)


class Projector(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.norm = RMSNorm(width, eps=config.rms_norm_eps)
        self.proj = nn.Sequential(
            nn.Conv2d(width, 4 * width, 2, stride=2),
            nn.GELU(),
            nn.Conv2d(4 * width, config.out_hidden_size, 1),
        )
        self.image_newline = nn.Parameter(torch.zeros(config.out_hidden_size))

    def forward(self, tokens, rows, cols):
        batch, _, width = tokens.shape
        grid = self.norm(tokens).reshape(batch, rows, cols, width)
        merged = (
            grid.reshape(batch, rows // 2, 2, cols // 2, 2, width)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, rows * cols // 4, 4 * width)
        )
        projected = self.proj(grid.permute(0, 3, 1, 2)).flatten(2).transpose(1, 2)
        return merged, projected


class Hy4VisionModel(PreTrainedModel):
    config_class = Hy4VisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = PatchEmbedding(config)
        self.layers = nn.ModuleList(
            Block(config) for _ in range(config.num_hidden_layers)
        )
        self.perceive = Projector(config)
        self.post_init()

    def forward(self, pixel_values, depths=()):
        tokens = self.embeddings(pixel_values)
        hidden = {}
        for depth, layer in enumerate(self.layers, 1):
            tokens = layer(tokens)
            if depth in depths:
                hidden[depth] = tokens
        rows, cols = (
            side // self.config.patch_size for side in pixel_values.shape[-2:]
        )
        merged, projected = self.perceive(tokens, rows, cols)
        return hidden, merged, projected
