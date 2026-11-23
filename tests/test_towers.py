"""Per-Tower behaviour that parity cannot see: preprocessing contracts, input validation, batch isolation.

Release correctness, conversions and forwards are checked against each parent in vtb/parity/.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import timm
import torch
from PIL import Image
from torchvision import transforms
from transformers import (
    AutoImageProcessor,
    DINOv3ViTModel,
    PixtralImageProcessor,
    PixtralVisionConfig,
    PixtralVisionModel,
    Qwen2VLImageProcessor,
)

from vtb.adapters import (
    dinov3,
    hy4_flash,
    mimo26,
    minimax_m3,
    mistral_large4,
    moonvit_v2,
    nemotron35,
    pe,
)
from vtb.adapters.common import IMAGENET_MEAN, IMAGENET_STD
from vtb.models import hy4_flash as hy4_flash_model
from vtb.models import inkling2
from vtb.parity import fixed_images
from vtb.roster import BY_SLUG, SLUGS

needs_weights = pytest.mark.skipif(
    os.environ.get("VTB_SKIP_WEIGHTS") == "1", reason="VTB_SKIP_WEIGHTS=1"
)
DEVICE = os.environ.get("VTB_PARITY_DEVICE", "cpu")
DTYPE = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32


def preprocess_for(slug: str, resolution: int | None = None):
    adapter_class = BY_SLUG[slug].adapter()
    preprocess = getattr(
        adapter_class, "Preprocess", sys.modules[adapter_class.__module__].Preprocess
    )
    return preprocess(resolution or adapter_class.default_resolution)


def noise(height: int, width: int, seed: int = 7) -> Image.Image:
    return Image.fromarray(
        np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    )


def bench_crop(image: Image.Image, resolution: int) -> Image.Image:
    """The bench's short-side bicubic resize and centre crop, written out with PIL."""
    width, height = image.size
    scale = resolution / min(width, height)
    resized = image.resize(
        (int(width * scale), int(height * scale)), Image.Resampling.BICUBIC
    )
    left, top = (
        round((resized.width - resolution) / 2),
        round((resized.height - resolution) / 2),
    )
    return resized.crop((left, top, left + resolution, top + resolution))


# A 600x480 image at each default resolution: pixels for whole-image Towers, one row per patch otherwise.
PREPROCESSED = {
    "dinov2": (3, 448, 448),
    "siglip2": (3, 448, 448),
    "dinov3": (3, 512, 512),
    "nemotron_omni": (3, 448, 448),
    "nemotron35_super": (3, 448, 448),
    "nemotron35_bolt": (3, 448, 448),
    "mistral_large4": (3, 448, 448),
    "pe_lang_l": (3, 448, 448),
    "inkling2_small": (3, 640, 640),
    "inkling2_mini": (3, 640, 640),
    "hy4_flash": (3, 512, 512),
    "kimi_k26": (1024, 3, 14, 14),
    "moonvit_v2": (1024, 3, 14, 14),
    "kimi_k31_pro": (1024, 3, 14, 14),
    "deepseek_v41": (1024, 3, 14, 14),
    "muse_glimmer": (1024, 1176),
    "glm5": (1024, 1176),
    "minimax_m3": (1024, 1176),
    "minimax_m31_flash": (1024, 1176),
    "qwen3_5": (784, 1536),
    "qwen38_flash_next": (784, 1536),
    "mimo26_pro": (784, 1536),
    "mimo26_flash": (784, 1536),
    "gemma4": (900, 768),
}


@pytest.mark.parametrize("slug", SLUGS)
def test_preprocess_produces_the_expected_patch_grid(slug):
    assert (
        tuple(preprocess_for(slug)(Image.new("RGB", (600, 480), "gray")).shape)
        == PREPROCESSED[slug]
    )


REJECTED_RESOLUTIONS = {
    "dinov3": [0, -16, 449, 511],
    "mimo26_pro": [0, -32, 240, 447],
    "qwen3_5": [0, -32, 447, 16],
    "mistral_large4": [440],
    "nemotron35_super": [0, 16, 33, 447],
    "pe_lang_l": [0, 224, 434, 462],
    "inkling2_small": [0, 448, 600, 680],
    "inkling2_mini": [0, 448, 600, 680],
    "hy4_flash": [0, 448, 544, 1024],
}


@pytest.mark.parametrize(
    "slug,resolution",
    [(slug, r) for slug, rs in REJECTED_RESOLUTIONS.items() for r in rs],
)
def test_unsupported_resolution_fails_before_weights_load(
    monkeypatch, slug, resolution
):
    def forbidden(*_args, **_kwargs):
        pytest.fail("weights loaded before the resolution was checked")

    module = sys.modules[BY_SLUG[slug].adapter().__module__]
    for name in ("load_parts", "load_model", "load_tower"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(dinov3.DINOv3ViTModel, "from_pretrained", forbidden)
    with pytest.raises(ValueError):
        BY_SLUG[slug].adapter()(resolution=resolution, device="cpu")


def dinov3_reference(image: Image.Image, resolution: int) -> torch.Tensor:
    raw = (
        torch.from_numpy(np.array(bench_crop(image, resolution)))
        .permute(2, 0, 1)
        .float()
        / 255
    )
    return (raw - torch.tensor(IMAGENET_MEAN)[:, None, None]) / torch.tensor(
        IMAGENET_STD
    )[:, None, None]


def mimo_reference(image: Image.Image, resolution: int) -> torch.Tensor:
    processor = Qwen2VLImageProcessor(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        image_mean=mimo26.IMAGE_MEAN,
        image_std=mimo26.IMAGE_STD,
    )
    return processor(
        images=mimo26.square_crop(resolution)(image),
        do_resize=False,
        return_tensors="pt",
    )["pixel_values"]


def mistral_reference(image: Image.Image, resolution: int) -> torch.Tensor:
    processor = PixtralImageProcessor(
        size={"longest_edge": resolution},
        patch_size=14,
        image_mean=list(mistral_large4.IMAGE_MEAN),
        image_std=list(mistral_large4.IMAGE_STD),
        resample=3,
        do_pad=True,
    )
    return processor(images=image, return_tensors="pt")["pixel_values"][0]


def nemotron_reference(image: Image.Image, resolution: int) -> torch.Tensor:
    return (
        torch.from_numpy(np.array(bench_crop(image, resolution)))
        .permute(2, 0, 1)
        .float()
        / 255
    )


def pe_reference(image: Image.Image, resolution: int) -> torch.Tensor:
    config = timm.data.resolve_data_config(
        pretrained_cfg=timm.models.get_pretrained_cfg(
            pe.PE_LANG_L.repo.removeprefix("timm/")
        ).to_dict()
    )
    assert (
        config["input_size"] == (3, resolution, resolution)
        and config["crop_pct"] == 1.0
    )
    return timm.data.create_transform(**config)(image)


def hy4_flash_reference(image: Image.Image, resolution: int) -> torch.Tensor:
    """The parent processor's rescale by 1/255, then mean and std 0.5, on the bench square."""
    raw = torch.from_numpy(np.array(bench_crop(image, resolution))).permute(2, 0, 1)
    return (raw.float() * (1 / 255) - 0.5) / 0.5


def kimi_reference(checkpoint):
    def reference(image: Image.Image, resolution: int) -> torch.Tensor:
        processor = AutoImageProcessor.from_pretrained(
            checkpoint.model_id, trust_remote_code=True
        )
        crop = transforms.Compose(
            [
                transforms.Resize(
                    resolution, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(resolution),
            ]
        )
        return processor(crop(image), return_tensors="pt")["pixel_values"]

    return reference


PARENT_PREPROCESSING = [
    pytest.param("dinov3", dinov3_reference, 0, id="dinov3"),
    pytest.param("mimo26_pro", mimo_reference, 5e-7, id="mimo26"),
    pytest.param("mistral_large4", mistral_reference, 1e-5, id="mistral_large4"),
    pytest.param("nemotron35_super", nemotron_reference, 0, id="nemotron35"),
    pytest.param("pe_lang_l", pe_reference, 0, id="pe_lang_l"),
    pytest.param("hy4_flash", hy4_flash_reference, 1e-6, id="hy4_flash"),
    *(
        pytest.param(
            slug, kimi_reference(checkpoint), 1e-6, id=slug, marks=needs_weights
        )
        for slug, checkpoint in moonvit_v2.CHECKPOINTS.items()
    ),
]


@pytest.mark.parametrize("slug,reference,atol", PARENT_PREPROCESSING)
def test_preprocess_matches_the_parent_image_processor(slug, reference, atol):
    """The bench crops to a square first; after that, each Tower's pixels or patches equal its parent processor's."""
    image = noise(480, 640) if slug != "mistral_large4" else noise(448, 448)
    resolution = BY_SLUG[slug].adapter().default_resolution
    torch.testing.assert_close(
        preprocess_for(slug)(image), reference(image, resolution), rtol=0, atol=atol
    )


@pytest.mark.parametrize(
    "adapter_class", [minimax_m3.MiniMaxM3Adapter, minimax_m3.MiniMaxM31FlashAdapter]
)
def test_minimax_keeps_unrelated_images_out_of_each_others_attention(adapter_class):
    """The native HF model packs a batch into one attention sequence; the adapter runs each image alone."""

    class MixingLayer(torch.nn.Module):
        def forward(self, tokens):
            return tokens + tokens.mean(dim=1, keepdim=True)

    class Tower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([MixingLayer() for _ in range(8)])
            self.config = SimpleNamespace(hidden_size=4)

        def forward(self, pixels, grid):
            tokens = pixels.unsqueeze(0)
            for layer in self.layers:
                tokens = layer(tokens)
            return SimpleNamespace(last_hidden_state=tokens)

    class Projector(torch.nn.Module):
        def forward(self, tokens):
            return tokens.reshape(-1, 4, 4).sum(dim=1)

    adapter = object.__new__(adapter_class)
    adapter.model, adapter.projector = Tower(), Projector()
    adapter.num_layers, adapter.resolution = 8, 28
    adapter.device, adapter.dtype = "cpu", torch.float32
    adapter.model_id = adapter_class.checkpoint.model_id
    first, second = torch.ones(4, 4), torch.full((4, 4), 10.0)
    alone = list(adapter.extract(minimax_m3.collate([first]), ["first"]))
    paired = list(
        adapter.extract(minimax_m3.collate([first, second]), ["first", "second"])
    )
    reversed_pair = list(
        adapter.extract(minimax_m3.collate([second, first]), ["second", "first"])
    )
    assert len(alone) == len(paired) == 10
    for single, batch, reversed_batch in zip(alone, paired, reversed_pair):
        assert torch.equal(single.tokens[0], batch.tokens[0])
        assert torch.equal(single.tokens[0], reversed_batch.tokens[1])


def tiny_pe_adapter(resolution: int = 96) -> pe.PELangLAdapter:
    """PE-Lang L's trunk at width 32, so raster order and isolation are checked without its weights."""
    torch.manual_seed(0)
    adapter = object.__new__(pe.PELangLAdapter)
    patch = resolution // pe.POSITION_GRID
    adapter.model = timm.create_model(
        pe.ARCHITECTURE,
        pretrained=False,
        patch_size=patch,
        img_size=resolution,
        embed_dim=32,
        num_heads=2,
    ).eval()
    adapter.num_layers, adapter.resolution = len(adapter.model.blocks), resolution
    adapter.device, adapter.dtype, adapter.model_id = (
        "cpu",
        torch.float32,
        pe.PE_LANG_L.model_id,
    )
    return adapter


def test_pe_tower_drops_the_class_token_and_keeps_raster_order():
    """With LayerScale zeroed every block is the identity, so each token is its own patch's embedding."""
    adapter = tiny_pe_adapter()
    for block in adapter.model.blocks:
        block.gamma_1.data.zero_()
        block.gamma_2.data.zero_()
    pixels = torch.randn(2, 3, 96, 96)
    batches = list(adapter.extract({"pixel_values": pixels}, ["a", "b"]))
    assert [batch.layer_index for batch in batches] == [3, 6, 9, 12, 14, 17, 20, 23]
    assert {batch.stage for batch in batches} == {"tower"}
    patches = pixels.unfold(2, 3, 3).unfold(3, 3, 3)
    weight = adapter.model.patch_embed.proj.weight
    embedded = torch.einsum("bchwyx,dcyx->bhwd", patches, weight).reshape(2, 1024, 32)
    expected = adapter.model.norm_pre(embedded + adapter.model.pos_embed[:, 1:])
    for batch in batches:
        assert batch.tokens.shape == (2, 1024, 32)
        torch.testing.assert_close(batch.tokens, expected, rtol=0, atol=1e-5)


def test_pe_tower_keeps_images_out_of_each_others_features():
    adapter = tiny_pe_adapter()
    first, second = torch.randn(1, 3, 96, 96), torch.randn(1, 3, 96, 96)
    alone = list(adapter.extract({"pixel_values": first}, ["first"]))
    paired = list(
        adapter.extract(
            {"pixel_values": torch.cat([second, first])}, ["second", "first"]
        )
    )
    for single, batch in zip(alone, paired, strict=True):
        torch.testing.assert_close(single.tokens[0], batch.tokens[1], rtol=0, atol=1e-5)


def test_pe_tower_rejects_mismatched_pixels_and_ids():
    adapter = tiny_pe_adapter()
    with pytest.raises(ValueError, match="pixels"):
        list(adapter.extract({"pixel_values": torch.randn(1, 3, 112, 112)}, ["a"]))
    with pytest.raises(ValueError, match="same length"):
        list(adapter.extract({"pixel_values": torch.randn(2, 3, 96, 96)}, ["a"]))


def tiny_inkling_weights(
    width: int = 8, layers: int = 2, patch: int = 2
) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    shapes = {
        "embeddings.patch_embedding.weight": (width, patch * patch * 3),
        "embeddings.position_embedding": (16, width),
        "embeddings.class_embedding": (width,),
        "embeddings.cls_position_embedding": (width,),
        "ln_pre.weight": (width,),
        "ln_pre.bias": (width,),
    }
    for index in range(layers):
        for name, shape in {
            "ln_1.weight": (width,),
            "ln_1.bias": (width,),
            "ln_2.weight": (width,),
            "ln_2.bias": (width,),
            "ls_1.gamma": (width,),
            "ls_2.gamma": (width,),
            "mlp.c_fc.weight": (4 * width, width),
            "mlp.c_fc.bias": (4 * width,),
            "mlp.c_proj.weight": (width, 4 * width),
            "mlp.c_proj.bias": (width,),
            **{
                f"self_attn.{p}_proj.{kind}": (width, width)
                if kind == "weight"
                else (width,)
                for p in ("q", "k", "v", "out")
                for kind in ("weight", "bias")
            },
        }.items():
            shapes[f"layers.{index}.{name}"] = shape
    return {"backbone." + name: torch.randn(shape) for name, shape in shapes.items()}


def test_inkling_remap_uses_every_tensor_once_and_names_every_timm_tensor():
    weights = tiny_inkling_weights()
    mapped = inkling2.to_timm(weights)
    assert len(weights) == 6 + 2 * 18 and len(mapped) == 5 + 2 * 14
    assert mapped["pos_embed"].shape == (1, 17, 8) and torch.equal(
        mapped["pos_embed"][0, 0], weights["backbone.embeddings.cls_position_embedding"]
    )
    qkv = mapped["blocks.1.attn.qkv.weight"]
    assert torch.equal(
        qkv,
        torch.cat(
            [weights[f"backbone.layers.1.self_attn.{p}_proj.weight"] for p in "qkv"]
        ),
    )
    assert torch.equal(
        mapped["blocks.0.gamma_2"], weights["backbone.layers.0.ls_2.gamma"]
    )
    with pytest.raises(KeyError, match="unmapped"):
        inkling2.to_timm({**weights, "backbone.extra.weight": torch.zeros(1)})
    with pytest.raises(KeyError):
        inkling2.to_timm(
            {
                name: tensor
                for name, tensor in weights.items()
                if not name.endswith("v_proj.bias")
            }
        )


def test_inkling_patch_kernel_reads_patches_flattened_row_column_channel():
    weights = tiny_inkling_weights(patch=3)
    linear = weights["backbone.embeddings.patch_embedding.weight"]
    kernel = inkling2.to_timm(weights)["patch_embed.proj.weight"]
    image = torch.randn(1, 3, 6, 9)
    patches = (
        image.permute(0, 2, 3, 1)
        .reshape(1, 2, 3, 3, 3, 3)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(1, 6, 27)
    )
    torch.testing.assert_close(
        torch.nn.functional.conv2d(image, kernel, stride=3).flatten(2).transpose(1, 2),
        patches @ linear.T,
    )


def test_inkling_merge_concatenates_each_block_row_major():
    side, width = 4, 2
    ids = torch.arange(side * side, dtype=torch.float32)
    merged = inkling2.merge(ids.view(1, -1, 1).expand(1, -1, width), side)
    assert merged.shape == (1, 4, 4 * width)
    assert merged[0, :, ::width].tolist() == [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15],
    ]


def test_inkling_preprocess_rounds_like_the_parent_and_stays_near_half_normalization():
    image = noise(480, 640)
    pixels = preprocess_for("inkling2_small")(image)
    assert pixels.shape == (3, 640, 640) and pixels.dtype == torch.float32
    assert torch.equal(pixels, pixels.bfloat16().float())
    half = (transforms.ToTensor()(bench_crop(image, 640)) - 0.5) / 0.5
    assert (pixels - half).abs().max() < 0.02


@pytest.mark.parametrize("checkpoint", pe.INKLING2, ids=lambda c: c.name)
def test_inkling_remap_flags_a_sibling_and_reads_the_projector_width_from_the_parent(
    monkeypatch, checkpoint
):
    weights = {
        **tiny_inkling_weights(),
        "proj_in.weight": torch.randn(4096, 32),
        "proj_out.weight": torch.randn(2304, 4096),
    }
    weights = {
        **weights,
        **{
            f"backbone.extra.{index}": torch.zeros(1)
            for index in range(pe.TOWER_TENSORS - len(weights) + 2)
        },
    }
    sibling = {name: tensor.clone() for name, tensor in weights.items()}
    sibling["backbone.ln_pre.weight"] += 1
    monkeypatch.setattr(pe, "timm_weights", lambda _checkpoint: {})
    monkeypatch.setattr(
        pe, "source_json", lambda *_: {"text_config": {"hidden_size": 2304}}
    )
    monkeypatch.setattr(pe, "source_weights", lambda _checkpoint: sibling)
    monkeypatch.setattr(
        inkling2,
        "to_timm",
        lambda _tower: {
            name: torch.zeros(1) for name in ("patch_embed.proj.weight", "pos_embed")
        },
    )
    report = pe.remap_checks(checkpoint, weights)
    (other,) = (c.name for c in pe.INKLING2 if c is not checkpoint)
    assert report["shapes"]["proj_out.weight"]["expected"] == [2304, 4096]
    assert "backbone.ln_pre.weight" not in report["sibling_equal"][other]
    assert "proj_out.weight" in report["sibling_equal"][other]
    assert not report["passed"]
    with pytest.raises(ValueError, match="no remap"):
        pe.remap_checks(pe.PE_LANG_L, weights)


def test_inkling_projector_uses_exact_gelu():
    projector = inkling2.Projector(8, 6, 4)
    tokens = torch.randn(2, 3, 8)
    hidden = projector.proj_in(tokens)
    expected = projector.proj_out(hidden * 0.5 * (1 + torch.erf(hidden / 2**0.5)))
    torch.testing.assert_close(projector(tokens), expected)


def test_mistral_release_packs_mixed_image_sizes_like_stock_pixtral():
    """Parity uses square images; the shipped forward must also pack non-square ones as Pixtral does."""
    torch.manual_seed(0)
    config = PixtralVisionConfig(
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=4,
        num_attention_heads=2,
        head_dim=16,
        image_size=112,
        patch_size=14,
        hidden_act="silu",
        rope_parameters={"rope_theta": 10000.0, "rope_type": "axial"},
    )
    stock = PixtralVisionModel(config).eval()
    shipped = mistral_large4.PixtralLarge4VisionModel(
        mistral_large4.PixtralLarge4VisionConfig(
            **{k: v for k, v in config.to_dict().items() if k != "model_type"}
        )
    ).eval()
    shipped.load_state_dict(stock.state_dict())
    pixels = torch.randn(2, 3, 112, 112)
    sizes = [(112, 84), (56, 112)]
    with torch.inference_mode():
        packed = shipped(pixels, image_sizes=sizes).last_hidden_state[0]
        first, _ = mistral_large4.encode(shipped, pixels[:1, :, :112, :84])
        second, _ = mistral_large4.encode(shipped, pixels[1:, :, :56, :112])
        reference = PixtralVisionModel.forward(
            stock, pixels, image_sizes=sizes
        ).last_hidden_state[0]
    assert packed.shape == (8 * 6 + 4 * 8, 32)
    assert torch.equal(packed, torch.cat([first[0], second[0]]))
    torch.testing.assert_close(packed, reference, rtol=1e-4, atol=1e-4)


def test_nemotron35_super_restores_the_parents_fp32_conditioner(monkeypatch):
    """The parent keeps the input conditioner in FP32; casting it to BF16 shifts every pixel."""

    class Radio(torch.nn.Module):
        def __init__(self, _config):
            super().__init__()
            self.radio_model = torch.nn.Module()
            self.radio_model.register_buffer(
                "summary_idxs", torch.zeros(3, dtype=torch.int64)
            )
            self.radio_model.input_conditioner = torch.nn.Module()
            for name in ("norm_mean", "norm_std"):
                self.radio_model.input_conditioner.register_buffer(
                    name, torch.zeros(3, 1, 1)
                )

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
    monkeypatch.setattr(
        nemotron35.AutoConfig,
        "from_pretrained",
        lambda *_a, **_kw: SimpleNamespace(vision_config=None),
    )
    monkeypatch.setattr(
        nemotron35, "get_class_from_dynamic_module", lambda *_a, **_kw: Radio
    )
    monkeypatch.setattr(nemotron35, "attach_video_embedder", lambda _: None)
    monkeypatch.setattr(nemotron35, "make_projector", torch.nn.Identity)
    parts = {
        "tower": {
            "summary_idxs": torch.tensor([0, 1]),
            "radio_model.input_conditioner.norm_mean": mean,
            "radio_model.input_conditioner.norm_std": std,
        },
        "projector": {},
        "final_norm": {},
    }
    radio, _, _ = nemotron35.build_parts(nemotron35.BOLT, parts, torch.bfloat16)
    conditioner = radio.radio_model.input_conditioner
    assert conditioner.norm_mean.dtype == torch.float32
    assert torch.equal(conditioner.norm_mean, mean) and torch.equal(
        conditioner.norm_std, std
    )


@pytest.fixture(scope="module")
def dinov3_parent():
    return (
        DINOv3ViTModel.from_pretrained(
            dinov3.SOURCE_REPO, revision=dinov3.SOURCE_REVISION, dtype=DTYPE
        )
        .to(DEVICE)
        .eval()
    )


@needs_weights
@pytest.mark.parametrize("resolution", [448, 512])
def test_dinov3_every_depth_matches_independent_parent_block_hooks(
    dinov3_parent, resolution
):
    """DINOv3 has no release of ours, so its check is here: a separate parent load, hooked by block, never via the adapter."""
    assert sum(p.numel() for p in dinov3_parent.parameters()) == 303_129_600
    assert dinov3_parent.config.num_register_tokens == 4
    adapter = dinov3.DINOv3Adapter(resolution=resolution, dtype=DTYPE, device=DEVICE)
    pixels = adapter.collate([adapter.preprocess()(image) for image in fixed_images()])
    streams, handles = {}, []
    for depth in range(3, 25, 3):

        def capture(_module, _inputs, output, depth=depth):
            streams[depth] = output[:, dinov3.PREFIX_TOKENS :].cpu()

        handles.append(
            dinov3_parent.model.layer[depth - 1].register_forward_hook(capture)
        )
    try:
        with torch.inference_mode():
            final = (
                dinov3_parent(pixels["pixel_values"].to(DEVICE, DTYPE))
                .last_hidden_state[:, dinov3.PREFIX_TOKENS :]
                .cpu()
            )
    finally:
        for handle in handles:
            handle.remove()
    batches = list(adapter.extract(pixels, ["landscape", "portrait"]))
    assert [batch.layer_index for batch in batches] == list(range(3, 25, 3))
    for batch in batches:
        expected = final if batch.layer_index == 24 else streams[batch.layer_index]
        assert batch.tokens.shape == (2, (resolution // 16) ** 2, 1024)
        assert torch.equal(batch.tokens, expected), batch.layer_index
    assert not torch.equal(streams[24], final)


def tiny_hy4_flash_adapter(side: int = 64) -> hy4_flash.Hy4FlashAdapter:
    """Eight blocks on a 4x4 position table, so the adapter's eight depth points are every block."""
    torch.manual_seed(0)
    config = hy4_flash_model.Hy4FlashVisionConfig(
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=8,
        num_attention_heads=2,
        patch_size=16,
        max_image_size=64,
        out_hidden_size=12,
    )
    adapter = hy4_flash.Hy4FlashAdapter.__new__(hy4_flash.Hy4FlashAdapter)
    adapter.model = hy4_flash_model.Hy4FlashVisionModel(config).eval()
    for parameter in adapter.model.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    adapter.num_layers, adapter.resolution = 8, side
    adapter.device, adapter.dtype, adapter.model_id = (
        "cpu",
        torch.float32,
        hy4_flash.MODEL_ID,
    )
    return adapter


def test_hy4_flash_positions_use_the_parents_shifted_scale_factor():
    """A linear ramp sampled at (out + 0.5) * 128 / (size + 0.1) - 0.5, not at size= coordinates."""
    config = hy4_flash_model.Hy4FlashVisionConfig(hidden_size=2, max_image_size=2048)
    embeddings = hy4_flash_model.Hy4FlashEmbeddings(config)
    edge = embeddings.position_edge
    rows, cols = torch.meshgrid(
        torch.arange(edge, dtype=torch.float32),
        torch.arange(edge, dtype=torch.float32),
        indexing="ij",
    )
    with torch.no_grad():
        embeddings.position_embedding.weight[0] = float("nan")
        embeddings.position_embedding.weight[1:] = torch.stack(
            [rows.flatten(), cols.flatten()], -1
        )
    table = embeddings.interpolate_pos_encoding(32, 24, torch.float32)
    sample_rows = ((torch.arange(32) + 0.5) * edge / 32.1 - 0.5).clamp(0, edge - 1)
    sample_cols = ((torch.arange(24) + 0.5) * edge / 24.1 - 0.5).clamp(0, edge - 1)
    expected = torch.stack(torch.meshgrid(sample_rows, sample_cols, indexing="ij"), -1)
    torch.testing.assert_close(table, expected.reshape(1, -1, 2), rtol=0, atol=1e-4)
    plain = ((torch.arange(32) + 0.5) * edge / 32 - 0.5).clamp(0, edge - 1)
    assert (table.reshape(32, 24, 2)[:, 0, 0] - plain).abs().max() > 0.1


def test_hy4_flash_embeds_each_patch_in_raster_order():
    adapter = tiny_hy4_flash_adapter()
    pixels = torch.randn(2, 3, 64, 64)
    tokens = adapter.model.embeddings(pixels)
    conv = adapter.model.embeddings.patch_embedding(pixels).flatten(2).transpose(1, 2)
    positions = adapter.model.embeddings.interpolate_pos_encoding(4, 4, torch.float32)
    torch.testing.assert_close(tokens, conv + positions)


def test_hy4_flash_merged_is_what_proj0_reads_and_projected_drops_the_newline():
    adapter = tiny_hy4_flash_adapter()
    pixels = torch.randn(2, 3, 64, 64)
    batches = {
        (b.stage, b.layer_index): b.tokens
        for b in adapter.extract({"pixel_values": pixels}, ["a", "b"])
    }
    assert list(batches) == [("tower", n) for n in range(1, 9)] + [
        ("merged", 8),
        ("projected", 8),
    ]
    with torch.inference_mode():
        out = adapter.model(pixels)
    perceive = adapter.model.perceive
    torch.testing.assert_close(batches[("tower", 8)], out.last_hidden_state)
    merged = batches[("merged", 8)]
    assert merged.shape == (2, 4, 64)
    conv = perceive.proj[0]
    with torch.inference_mode():
        first = merged @ conv.weight.reshape(64, -1).T + conv.bias
        normed = perceive.norm(out.last_hidden_state)
        reference = conv(normed.reshape(2, 4, 4, 16).permute(0, 3, 1, 2))
    torch.testing.assert_close(first, reference.flatten(2).transpose(1, 2))
    pooled = out.pooler_output.reshape(2, 2, 3, 12)
    torch.testing.assert_close(
        batches[("projected", 8)], pooled[:, :, :2].reshape(2, 4, 12)
    )
    assert torch.equal(pooled[:, :, 2], perceive.image_newline.expand(2, 2, 12))


def test_hy4_flash_projector_rmsnorm_and_gelu_match_the_parent_arithmetic():
    perceive = tiny_hy4_flash_adapter().model.perceive
    tokens = torch.randn(1, 16, 16)
    expected = tokens * torch.rsqrt(tokens.pow(2).mean(-1, keepdim=True) + 1e-5)
    torch.testing.assert_close(perceive.norm(tokens), perceive.norm.weight * expected)
    assert perceive.norm.eps == 1e-5 and perceive.proj[1].approximate == "none"


def test_hy4_flash_keeps_images_out_of_each_others_features():
    adapter = tiny_hy4_flash_adapter()
    first, second = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)
    alone = list(adapter.extract({"pixel_values": first}, ["first"]))
    paired = list(
        adapter.extract(
            {"pixel_values": torch.cat([second, first])}, ["second", "first"]
        )
    )
    for single, batch in zip(alone, paired, strict=True):
        torch.testing.assert_close(single.tokens[0], batch.tokens[1], rtol=0, atol=1e-5)


def test_hy4_flash_rejects_mismatched_pixels_and_ids():
    adapter = tiny_hy4_flash_adapter()
    with pytest.raises(ValueError, match="shape"):
        list(adapter.extract({"pixel_values": torch.randn(1, 3, 96, 96)}, ["a"]))
    with pytest.raises(ValueError, match="count"):
        list(adapter.extract({"pixel_values": torch.randn(2, 3, 64, 64)}, ["a"]))
