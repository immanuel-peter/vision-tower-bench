# Copyright (c) 2026 Immanuel Peter. MIT License; see LICENSE-CODE.
#
# Standalone vision Tower and Projector of tencent/Hy4-flash
# (d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959), written for vision-tower-bench.
# The arithmetic follows the hy_v4_flash vision classes in transformers PR
# #50903 at c2ec850510a37978ea83df226589623d5f2be776. Parameter names are the
# checkpoint's own `vit.*` names with the `vit.` prefix removed, so the released
# tensors load without renaming. No language model code is imported.
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling


class Hy4FlashVisionConfig(PretrainedConfig):
    model_type = "hy4_flash_vision"

    def __init__(self, **kwargs):
        defaults = dict(hidden_size=1152, intermediate_size=4304, num_hidden_layers=27,
                        num_attention_heads=16, num_channels=3, patch_size=16,
                        layer_norm_eps=1e-5, rms_norm_eps=1e-5, max_image_size=2048,
                        spatial_merge_size=2, out_hidden_size=3072, hidden_act="gelu")
        defaults.update(kwargs)
        super().__init__(**defaults)


class Hy4FlashRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(dtype)


class Hy4FlashEmbeddings(nn.Module):
    def __init__(self, config: Hy4FlashVisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.position_edge = config.max_image_size // config.patch_size
        self.patch_embedding = nn.Conv2d(config.num_channels, config.hidden_size,
                                         kernel_size=config.patch_size, stride=config.patch_size, bias=True)
        # Row 0 is an unused class-token slot; rows 1.. form the 128x128 grid.
        self.position_embedding = nn.Embedding(self.position_edge**2 + 1, config.hidden_size)

    def interpolate_pos_encoding(self, height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
        edge, width_dim = self.position_edge, self.position_embedding.embedding_dim
        grid = self.position_embedding.weight[1:].reshape(1, edge, edge, width_dim)
        grid = grid.permute(0, 3, 1, 2).float()
        # The parent passes a scale factor of (size + 0.1) / edge, not a target size.
        grid = F.interpolate(grid, scale_factor=((height + 0.1) / edge, (width + 0.1) / edge),
                             mode="bilinear", align_corners=False)
        return grid.permute(0, 2, 3, 1).reshape(1, -1, width_dim).to(dtype)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Embed each patch on its own, as the parent does on its packed patch rows.
        batch, channels, height, width = pixel_values.shape
        p = self.patch_size
        rows, cols = height // p, width // p
        patches = pixel_values.reshape(batch, channels, rows, p, cols, p).permute(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(batch * rows * cols, channels, p, p)
        tokens = self.patch_embedding(patches.to(self.patch_embedding.weight.dtype)).reshape(batch, rows * cols, -1)
        return tokens + self.interpolate_pos_encoding(rows, cols, tokens.dtype)


class Hy4FlashAttention(nn.Module):
    def __init__(self, config: Hy4FlashVisionConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, length, _ = hidden_states.shape

        def heads(projection: nn.Linear) -> torch.Tensor:
            return projection(hidden_states).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

        attended = F.scaled_dot_product_attention(heads(self.q_proj), heads(self.k_proj), heads(self.v_proj),
                                                  scale=self.head_dim**-0.5)
        return self.o_proj(attended.transpose(1, 2).reshape(batch, length, -1))


class Hy4FlashMLP(nn.Module):
    def __init__(self, config: Hy4FlashVisionConfig) -> None:
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dense_4h_to_h(F.gelu(self.dense_h_to_4h(hidden_states)))


class Hy4FlashBlock(nn.Module):
    def __init__(self, config: Hy4FlashVisionConfig) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = Hy4FlashAttention(config)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Hy4FlashMLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states))
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class Hy4FlashPerceive(nn.Module):
    """The Projector: RMSNorm, a learned 2x2 stride-2 merge conv, GELU, a 1x1 conv, then a newline per row."""

    def __init__(self, config: Hy4FlashVisionConfig) -> None:
        super().__init__()
        merge, width = config.spatial_merge_size, config.hidden_size
        self.norm = Hy4FlashRMSNorm(width, eps=config.rms_norm_eps)
        self.proj = nn.Sequential(
            nn.Conv2d(width, width * merge**2, kernel_size=merge, stride=merge),
            nn.GELU(),
            nn.Conv2d(width * merge**2, config.out_hidden_size, kernel_size=1),
        )
        self.image_newline = nn.Parameter(torch.zeros(config.out_hidden_size))
        self.merge_size = merge

    def merge(self, normed: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        """Each 2x2 block as the (channel, row, column) vector proj.0 reads, in raster block order."""
        batch, (height, width), channels, m = normed.shape[0], grid, normed.shape[-1], self.merge_size
        blocks = normed.reshape(batch, height // m, m, width // m, m, channels)
        return blocks.permute(0, 1, 3, 5, 2, 4).reshape(batch, (height // m) * (width // m), channels * m * m)

    def project(self, normed: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        """proj.2 output as (batch, channels, rows, columns), before the newline column."""
        image = normed.reshape(normed.shape[0], *grid, normed.shape[-1]).permute(0, 3, 1, 2)
        return self.proj(image)

    def forward(self, hidden_states: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        projected = self.project(self.norm(hidden_states), grid)
        batch, channels, rows, _ = projected.shape
        newline = self.image_newline.reshape(1, channels, 1, 1).expand(batch, channels, rows, 1)
        projected = torch.cat([projected, newline.to(projected.dtype)], dim=-1)
        return projected.flatten(2).transpose(1, 2)


class Hy4FlashVisionModel(PreTrainedModel):
    config_class = Hy4FlashVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["Hy4FlashBlock"]

    def __init__(self, config: Hy4FlashVisionConfig) -> None:
        super().__init__(config)
        self.embeddings = Hy4FlashEmbeddings(config)
        self.layers = nn.ModuleList(Hy4FlashBlock(config) for _ in range(config.num_hidden_layers))
        self.perceive = Hy4FlashPerceive(config)
        self.post_init()

    def forward(self, pixel_values: torch.Tensor, output_hidden_states: bool = False) -> BaseModelOutputWithPooling:
        """pixel_values (batch, 3, H, W), normalized with mean and std 0.5. There is no final norm:
        last_hidden_state is the last block's output and pooler_output the Projector's, with newlines."""
        grid = (pixel_values.shape[-2] // self.config.patch_size, pixel_values.shape[-1] // self.config.patch_size)
        hidden_states = self.embeddings(pixel_values)
        states = [hidden_states]
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            states.append(hidden_states)
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=self.perceive(hidden_states, grid),
            hidden_states=tuple(states) if output_hidden_states else None,
        )
