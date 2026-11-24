# Copyright (c) 2026 Immanuel Peter. MIT License; see LICENSE-CODE.
#
# Standalone vision Towers and Projectors of tencent/Hy4 (c10c0de5b6f850703fb0a90e65d1c32b558b9167)
# and tencent/Hy4-flash (d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959), written for vision-tower-bench.
# The arithmetic follows the vision classes of Transformers commit 1031277cef60200fda98628e9adc8fb15f0e1549
# (Hy4) and of transformers PR #50903 at c2ec850510a37978ea83df226589623d5f2be776 (Hy4 flash).
# Parameter names are the checkpoints' own `vit.*` names with the `vit.` prefix removed, so the released
# tensors load without renaming. No language model code is imported.

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

CHANNELS = 3
MERGE = 2


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


class Hy4FlashVisionConfig(Hy4VisionConfig):
    model_type = "hy4_flash_vision"

    def __init__(
        self,
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        out_hidden_size=3072,
        **kwargs,
    ):
        kwargs.setdefault("num_channels", CHANNELS)
        kwargs.setdefault("spatial_merge_size", MERGE)
        kwargs.setdefault("hidden_act", "gelu")
        kwargs.setdefault("layer_norm_eps", kwargs.get("rms_norm_eps", 1e-5))
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            out_hidden_size=out_hidden_size,
            **kwargs,
        )


class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.position_edge = config.max_image_size // config.patch_size
        self.patch_embedding = nn.Conv2d(
            CHANNELS, config.hidden_size, config.patch_size, stride=config.patch_size
        )
        # Row 0 is an unused class-token slot; rows 1.. form the position grid.
        self.position_embedding = nn.Embedding(
            self.position_edge**2 + 1, config.hidden_size
        )

    def interpolate_pos_encoding(self, rows, cols, dtype):
        edge, width = self.position_edge, self.position_embedding.embedding_dim
        grid = self.position_embedding.weight[1:].reshape(1, edge, edge, width)
        # The parent passes a scale factor of (size + 0.1) / edge, not a target size.
        grid = F.interpolate(
            grid.permute(0, 3, 1, 2).float(),
            scale_factor=((rows + 0.1) / edge, (cols + 0.1) / edge),
            mode="bilinear",
            align_corners=False,
        )
        return grid.permute(0, 2, 3, 1).reshape(1, -1, width).to(dtype)

    def forward(self, pixel_values):
        batch, channels, height, width = pixel_values.shape
        patch = self.patch_size
        rows, cols = height // patch, width // patch
        patches = pixel_values.reshape(batch, channels, rows, patch, cols, patch)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(
            batch * rows * cols, channels, patch, patch
        )
        tokens = self.patch_embedding(
            patches.to(self.patch_embedding.weight.dtype)
        ).reshape(batch, rows * cols, -1)
        return tokens + self.interpolate_pos_encoding(rows, cols, tokens.dtype)


class Attention(nn.Module):
    def __init__(self, config, per_image=False):
        super().__init__()
        width = config.hidden_size
        self.heads = config.num_attention_heads
        self.per_image = per_image
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.o_proj = nn.Linear(width, width)

    def forward(self, tokens):
        batch, length, _ = tokens.shape

        def heads(projection):
            return (
                projection(tokens).view(batch, length, self.heads, -1).transpose(1, 2)
            )

        q, k, v = heads(self.q_proj), heads(self.k_proj), heads(self.v_proj)
        scale = q.shape[-1] ** -0.5
        if self.per_image:
            # One kernel per image, as the Hy4 reference dispatches it; the batched BF16 FlashAttention kernel rounds differently.
            attended = torch.cat(
                [
                    F.scaled_dot_product_attention(
                        q[i : i + 1], k[i : i + 1], v[i : i + 1], scale=scale
                    )
                    for i in range(batch)
                ]
            )
        else:
            attended = F.scaled_dot_product_attention(q, k, v, scale=scale)
        return self.o_proj(attended.transpose(1, 2).reshape(batch, length, -1))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, tokens):
        return self.dense_4h_to_h(F.gelu(self.dense_h_to_4h(tokens)))


class Block(nn.Module):
    def __init__(self, config, per_image=False):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config, per_image)
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = MLP(config)

    def forward(self, tokens):
        tokens = tokens + self.self_attn(self.input_layernorm(tokens))
        return tokens + self.mlp(self.post_attention_layernorm(tokens))


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, tokens):
        values = tokens.float()
        normalized = values * torch.rsqrt(
            values.square().mean(-1, keepdim=True) + self.eps
        )
        return self.weight * normalized.to(tokens.dtype)


class Projector(nn.Module):
    """RMSNorm, a learned 2x2 stride-2 merge conv, GELU, a 1x1 conv, then a newline per row."""

    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.norm = RMSNorm(width, config.rms_norm_eps)
        self.proj = nn.Sequential(
            nn.Conv2d(width, width * MERGE**2, MERGE, stride=MERGE),
            nn.GELU(),
            nn.Conv2d(width * MERGE**2, config.out_hidden_size, 1),
        )
        self.image_newline = nn.Parameter(torch.zeros(config.out_hidden_size))

    def merge(self, normed, grid, channels_last=False):
        """Each 2x2 block as one vector, in raster block order. The vector is (channel, row, column) as proj.0 reads it, or (row, column, channel) when ``channels_last``."""
        batch, (height, width), channels = normed.shape[0], grid, normed.shape[-1]
        blocks = normed.reshape(
            batch, height // MERGE, MERGE, width // MERGE, MERGE, channels
        )
        order = (0, 1, 3, 2, 4, 5) if channels_last else (0, 1, 3, 5, 2, 4)
        return blocks.permute(*order).reshape(
            batch, (height // MERGE) * (width // MERGE), channels * MERGE**2
        )

    def project(self, normed, grid):
        """proj.2 output as (batch, channels, rows, columns), before the newline column."""
        image = normed.reshape(normed.shape[0], *grid, normed.shape[-1]).permute(
            0, 3, 1, 2
        )
        return self.proj(image)

    def forward(self, hidden_states, grid):
        projected = self.project(self.norm(hidden_states), grid)
        batch, channels, rows, _ = projected.shape
        newline = self.image_newline.reshape(1, channels, 1, 1).expand(
            batch, channels, rows, 1
        )
        projected = torch.cat([projected, newline.to(projected.dtype)], dim=-1)
        return projected.flatten(2).transpose(1, 2)


class Hy4VisionBase(PreTrainedModel):
    main_input_name = "pixel_values"
    _no_split_modules = ["Block"]
    merged_channels_last = False
    per_image_attention = False

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = PatchEmbedding(config)
        self.layers = nn.ModuleList(
            Block(config, self.per_image_attention)
            for _ in range(config.num_hidden_layers)
        )
        self.perceive = Projector(config)
        self.post_init()

    def grid(self, pixel_values):
        return tuple(side // self.config.patch_size for side in pixel_values.shape[-2:])

    def stages(self, pixel_values, depths=()):
        """The block outputs at ``depths``, the merged tokens, and the projected tokens without newlines. There is no final norm: the last block feeds the Projector."""
        tokens = self.embeddings(pixel_values)
        hidden = {}
        for depth, layer in enumerate(self.layers, 1):
            tokens = layer(tokens)
            if depth in depths:
                hidden[depth] = tokens
        grid = self.grid(pixel_values)
        normed = self.perceive.norm(tokens)
        merged = self.perceive.merge(normed, grid, self.merged_channels_last)
        projected = self.perceive.project(normed, grid).flatten(2).transpose(1, 2)
        return hidden, merged, projected


class Hy4VisionModel(Hy4VisionBase):
    config_class = Hy4VisionConfig
    merged_channels_last = True
    per_image_attention = True

    def forward(self, pixel_values, depths=()):
        return self.stages(pixel_values, depths)


class Hy4FlashVisionModel(Hy4VisionBase):
    config_class = Hy4FlashVisionConfig

    def forward(self, pixel_values, output_hidden_states=False):
        """pixel_values (batch, 3, H, W), normalized with mean and std 0.5. There is no final norm: last_hidden_state is the last block's output and pooler_output the Projector's, with newlines."""
        hidden_states = self.embeddings(pixel_values)
        states = [hidden_states] if output_hidden_states else None
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            if states is not None:
                states.append(hidden_states)
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=self.perceive(hidden_states, self.grid(pixel_values)),
            hidden_states=tuple(states) if states is not None else None,
        )
