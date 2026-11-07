# Copyright 2026 Immanuel Peter. MIT License; see LICENSE in the vision-tower-bench repository.
"""Mistral Large 4 vision encoder with the parent's numeric order.

The weights are HF PixtralVisionModel weights, so the module is a
PixtralVisionModel subclass and loads the same state dict. Only the forward
changes, to follow mistral_inference (the parent's reference code) in BF16:

1. Patch embeddings are made contiguous before ln_pre, as the parent's
   torch.cat does. A transposed view changes the RMSNorm reduction order.
2. The rotary table is built on the CPU in float32 with torch.polar, as
   mistral_inference.rope.precompute_freqs_cis_2d does, and q/k are rotated by
   complex multiplication in float32 before a cast back. HF Pixtral instead
   casts cos/sin to BF16 and rotates in BF16.
3. Rotated q/k are attended in the parent's interleaved (re, im) channel
   order, so the kernel sums each q.k product in the parent's order.
4. Attention uses PyTorch's memory-efficient SDPA kernel, which PyTorch
   adapted from the xformers kernel the parent calls (MATH on CPU).

In float32 this forward and HF's agree to rounding. In BF16, at 448 and 1540
px, it matches mistral_inference with the same attention kernel bit for bit.
"""

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import PixtralVisionConfig, PixtralVisionModel
from transformers.modeling_outputs import BaseModelOutput

ATTENTION_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


class PixtralLarge4VisionConfig(PixtralVisionConfig):
    model_type = "pixtral_large4_vision"


def rotary_table(rows: int, cols: int, head_dim: int, theta: float) -> torch.Tensor:
    """Complex rotation per (row, col) position, built as mistral_inference builds it.

    Entry j rotates the HF pair (j, j + head_dim / 2), which is the parent's
    pair (2j, 2j + 1) after the export reorders the q/k rows.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs_h = torch.outer(torch.arange(rows), freqs[::2]).float()
    freqs_w = torch.outer(torch.arange(cols), freqs[1::2]).float()
    grid = torch.cat([freqs_h[:, None, :].repeat(1, cols, 1), freqs_w[None, :, :].repeat(rows, 1, 1)], dim=-1)
    return torch.polar(torch.ones_like(grid), grid).reshape(rows * cols, head_dim // 2)


def rotate(x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Rotate half-split q or k in float32 and return the interleaved layout in x's dtype."""
    half = x.shape[-1] // 2
    x32 = x.float()
    turned = torch.complex(x32[..., :half], x32[..., half:]) * table
    return torch.view_as_real(turned).flatten(-2).to(x.dtype)


def block_forward(layer, hidden: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """One PixtralAttentionLayer, with RoPE and attention in the parent's numeric order."""
    attention = layer.attention
    batch, tokens, _ = hidden.shape
    normed = layer.attention_norm(hidden)

    def heads(projection):
        return projection(normed).view(batch, tokens, attention.num_heads, attention.head_dim).transpose(1, 2)

    query = rotate(heads(attention.q_proj), table)
    key = rotate(heads(attention.k_proj), table)
    value = heads(attention.v_proj)
    with sdpa_kernel(ATTENTION_BACKENDS):
        mixed = F.scaled_dot_product_attention(query, key, value, scale=attention.scaling)
    hidden = hidden + attention.o_proj(mixed.transpose(1, 2).reshape(batch, tokens, -1))
    return hidden + layer.feed_forward(layer.ffn_norm(hidden))


def encode(
    tower: PixtralVisionModel, pixel_values: torch.Tensor, capture: tuple[int, ...] = ()
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Encode equal-size images as a batch; return the last block and captured depths (1-based).

    The parent packs images into one sequence with a block-diagonal mask, so
    images never attend to each other. For equal sizes that equals a batch.
    """
    embeds = tower.patch_conv(pixel_values.to(tower.patch_conv.weight.dtype))
    rows, cols = embeds.shape[-2:]
    hidden = tower.ln_pre(embeds.flatten(2).transpose(1, 2).contiguous())
    config = tower.config
    table = rotary_table(rows, cols, config.head_dim, config.rope_parameters["rope_theta"]).to(hidden.device)
    captured = {}
    for depth, layer in enumerate(tower.transformer.layers, start=1):
        hidden = block_forward(layer, hidden, table)
        if depth in capture:
            captured[depth] = hidden
    return hidden, captured


class PixtralLarge4VisionModel(PixtralVisionModel):
    """PixtralVisionModel weights with the parent's BF16 numeric order.

    ``forward`` takes padded ``pixel_values`` (batch, 3, H, W) and optional
    ``image_sizes`` like PixtralVisionModel, and returns one packed sequence
    (1, tokens, 2048) in raster order per image, as PixtralVisionModel does.
    ``hidden_states`` holds every block's output, block 1 first, when
    ``output_hidden_states`` is set.
    """

    config_class = PixtralLarge4VisionConfig

    def forward(self, pixel_values: torch.Tensor, image_sizes=None, output_hidden_states: bool = False, **kwargs):
        if image_sizes is None:
            image_sizes = [tuple(pixel_values.shape[-2:])] * len(pixel_values)
        patch = self.config.patch_size
        crops = [
            pixel_values[i : i + 1, :, : (size[0] // patch) * patch, : (size[1] // patch) * patch]
            for i, size in enumerate(image_sizes)
        ]
        depths = tuple(range(1, self.config.num_hidden_layers + 1)) if output_hidden_states else ()
        outputs, stacks = [], []
        index = 0
        while index < len(crops):
            # Run consecutive equal-size images together.
            end = index + 1
            while end < len(crops) and crops[end].shape == crops[index].shape:
                end += 1
            batch = torch.cat(crops[index:end])
            final, captured = encode(self, batch, depths)
            outputs.extend(final.unbind(0))
            if output_hidden_states:
                stacks.append([captured[d] for d in depths])
            index = end
        last = torch.cat(outputs).unsqueeze(0)
        hidden_states = None
        if output_hidden_states:
            hidden_states = tuple(
                torch.cat([h for stack in stacks for h in stack[d].unbind(0)]).unsqueeze(0) for d in range(len(depths))
            )
        return BaseModelOutput(last_hidden_state=last, hidden_states=hidden_states)


__all__ = ["PixtralLarge4VisionConfig", "PixtralLarge4VisionModel", "encode", "rotary_table", "rotate", "block_forward"]
