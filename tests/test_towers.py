"""Per-Tower behaviour that parity cannot see: preprocessing contracts, input validation, batch isolation.

Release correctness, conversions and forwards are checked against each parent in vtb/parity/.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, DINOv3ViTModel, PixtralImageProcessor, PixtralVisionConfig, PixtralVisionModel, Qwen2VLImageProcessor

from vtb.adapters import dinov3, mimo26, minimax_m3, mistral_large4, moonvit_v2, nemotron35
from vtb.adapters.common import IMAGENET_MEAN, IMAGENET_STD
from vtb.parity import fixed_images
from vtb.roster import BY_SLUG, SLUGS

needs_weights = pytest.mark.skipif(os.environ.get("VTB_SKIP_WEIGHTS") == "1", reason="VTB_SKIP_WEIGHTS=1")
DEVICE = os.environ.get("VTB_PARITY_DEVICE", "cpu")
DTYPE = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32


def preprocess_for(slug: str, resolution: int | None = None):
    adapter_class = BY_SLUG[slug].adapter()
    return sys.modules[adapter_class.__module__].Preprocess(resolution or adapter_class.default_resolution)


def noise(height: int, width: int, seed: int = 7) -> Image.Image:
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8))


def bench_crop(image: Image.Image, resolution: int) -> Image.Image:
    """The bench's short-side bicubic resize and centre crop, written out with PIL."""
    width, height = image.size
    scale = resolution / min(width, height)
    resized = image.resize((int(width * scale), int(height * scale)), Image.Resampling.BICUBIC)
    left, top = round((resized.width - resolution) / 2), round((resized.height - resolution) / 2)
    return resized.crop((left, top, left + resolution, top + resolution))


# A 600x480 image at each default resolution: pixels for whole-image Towers, one row per patch otherwise.
PREPROCESSED = {
    "dinov2": (3, 448, 448), "siglip2": (3, 448, 448), "dinov3": (3, 512, 512),
    "nemotron_omni": (3, 448, 448), "nemotron35_super": (3, 448, 448), "nemotron35_bolt": (3, 448, 448), "mistral_large4": (3, 448, 448),
    "kimi_k26": (1024, 3, 14, 14), "moonvit_v2": (1024, 3, 14, 14), "kimi_k31_pro": (1024, 3, 14, 14), "deepseek_v41": (1024, 3, 14, 14),
    "muse_glimmer": (1024, 1176), "glm5": (1024, 1176), "minimax_m3": (1024, 1176), "minimax_m31_flash": (1024, 1176),
    "qwen3_5": (784, 1536), "qwen38_flash_next": (784, 1536), "mimo26_pro": (784, 1536), "mimo26_flash": (784, 1536),
    "gemma4": (900, 768),
}


@pytest.mark.parametrize("slug", SLUGS)
def test_preprocess_produces_the_expected_patch_grid(slug):
    assert tuple(preprocess_for(slug)(Image.new("RGB", (600, 480), "gray")).shape) == PREPROCESSED[slug]


REJECTED_RESOLUTIONS = {
    "dinov3": [0, -16, 449, 511],
    "mimo26_pro": [0, -32, 240, 447],
    "qwen3_5": [0, -32, 447, 16],
    "mistral_large4": [440],
    "nemotron35_super": [0, 16, 33, 447],
}


@pytest.mark.parametrize("slug,resolution", [(slug, r) for slug, rs in REJECTED_RESOLUTIONS.items() for r in rs])
def test_unsupported_resolution_fails_before_weights_load(monkeypatch, slug, resolution):
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
    raw = torch.from_numpy(np.array(bench_crop(image, resolution))).permute(2, 0, 1).float() / 255
    return (raw - torch.tensor(IMAGENET_MEAN)[:, None, None]) / torch.tensor(IMAGENET_STD)[:, None, None]


def mimo_reference(image: Image.Image, resolution: int) -> torch.Tensor:
    processor = Qwen2VLImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2, image_mean=mimo26.IMAGE_MEAN, image_std=mimo26.IMAGE_STD)
    return processor(images=mimo26.square_crop(resolution)(image), do_resize=False, return_tensors="pt")["pixel_values"]


def mistral_reference(image: Image.Image, resolution: int) -> torch.Tensor:
    processor = PixtralImageProcessor(size={"longest_edge": resolution}, patch_size=14, image_mean=list(mistral_large4.IMAGE_MEAN),
                                      image_std=list(mistral_large4.IMAGE_STD), resample=3, do_pad=True)
    return processor(images=image, return_tensors="pt")["pixel_values"][0]


def nemotron_reference(image: Image.Image, resolution: int) -> torch.Tensor:
    return torch.from_numpy(np.array(bench_crop(image, resolution))).permute(2, 0, 1).float() / 255


def kimi_reference(checkpoint):
    def reference(image: Image.Image, resolution: int) -> torch.Tensor:
        processor = AutoImageProcessor.from_pretrained(checkpoint.model_id, trust_remote_code=True)
        crop = transforms.Compose([transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(resolution)])
        return processor(crop(image), return_tensors="pt")["pixel_values"]
    return reference


PARENT_PREPROCESSING = [
    pytest.param("dinov3", dinov3_reference, 0, id="dinov3"),
    pytest.param("mimo26_pro", mimo_reference, 5e-7, id="mimo26"),
    pytest.param("mistral_large4", mistral_reference, 1e-5, id="mistral_large4"),
    pytest.param("nemotron35_super", nemotron_reference, 0, id="nemotron35"),
    *(pytest.param(slug, kimi_reference(checkpoint), 1e-6, id=slug, marks=needs_weights)
      for slug, checkpoint in moonvit_v2.CHECKPOINTS.items()),
]


@pytest.mark.parametrize("slug,reference,atol", PARENT_PREPROCESSING)
def test_preprocess_matches_the_parent_image_processor(slug, reference, atol):
    """The bench crops to a square first; after that, each Tower's pixels or patches equal its parent processor's."""
    image = noise(480, 640) if slug != "mistral_large4" else noise(448, 448)
    resolution = BY_SLUG[slug].adapter().default_resolution
    torch.testing.assert_close(preprocess_for(slug)(image), reference(image, resolution), rtol=0, atol=atol)


@pytest.mark.parametrize("adapter_class", [minimax_m3.MiniMaxM3Adapter, minimax_m3.MiniMaxM31FlashAdapter])
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
    paired = list(adapter.extract(minimax_m3.collate([first, second]), ["first", "second"]))
    reversed_pair = list(adapter.extract(minimax_m3.collate([second, first]), ["second", "first"]))
    assert len(alone) == len(paired) == 10
    for single, batch, reversed_batch in zip(alone, paired, reversed_pair):
        assert torch.equal(single.tokens[0], batch.tokens[0])
        assert torch.equal(single.tokens[0], reversed_batch.tokens[1])


def test_mistral_release_packs_mixed_image_sizes_like_stock_pixtral():
    """Parity uses square images; the shipped forward must also pack non-square ones as Pixtral does."""
    torch.manual_seed(0)
    config = PixtralVisionConfig(hidden_size=32, intermediate_size=48, num_hidden_layers=4, num_attention_heads=2, head_dim=16, image_size=112,
                                 patch_size=14, hidden_act="silu", rope_parameters={"rope_theta": 10000.0, "rope_type": "axial"})
    stock = PixtralVisionModel(config).eval()
    shipped = mistral_large4.PixtralLarge4VisionModel(mistral_large4.PixtralLarge4VisionConfig(
        **{k: v for k, v in config.to_dict().items() if k != "model_type"})).eval()
    shipped.load_state_dict(stock.state_dict())
    pixels = torch.randn(2, 3, 112, 112)
    sizes = [(112, 84), (56, 112)]
    with torch.inference_mode():
        packed = shipped(pixels, image_sizes=sizes).last_hidden_state[0]
        first, _ = mistral_large4.encode(shipped, pixels[:1, :, :112, :84])
        second, _ = mistral_large4.encode(shipped, pixels[1:, :, :56, :112])
        reference = PixtralVisionModel.forward(stock, pixels, image_sizes=sizes).last_hidden_state[0]
    assert packed.shape == (8 * 6 + 4 * 8, 32)
    assert torch.equal(packed, torch.cat([first[0], second[0]]))
    torch.testing.assert_close(packed, reference, rtol=1e-4, atol=1e-4)


def test_nemotron35_super_restores_the_parents_fp32_conditioner(monkeypatch):
    """The parent keeps the input conditioner in FP32; casting it to BF16 shifts every pixel."""
    class Radio(torch.nn.Module):
        def __init__(self, _config):
            super().__init__()
            self.radio_model = torch.nn.Module()
            self.radio_model.register_buffer("summary_idxs", torch.zeros(3, dtype=torch.int64))
            self.radio_model.input_conditioner = torch.nn.Module()
            for name in ("norm_mean", "norm_std"):
                self.radio_model.input_conditioner.register_buffer(name, torch.zeros(3, 1, 1))

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
    monkeypatch.setattr(nemotron35.AutoConfig, "from_pretrained", lambda *_a, **_kw: SimpleNamespace(vision_config=None))
    monkeypatch.setattr(nemotron35, "get_class_from_dynamic_module", lambda *_a, **_kw: Radio)
    monkeypatch.setattr(nemotron35, "attach_video_embedder", lambda _: None)
    monkeypatch.setattr(nemotron35, "make_projector", torch.nn.Identity)
    parts = {"tower": {"summary_idxs": torch.tensor([0, 1]), "radio_model.input_conditioner.norm_mean": mean,
                       "radio_model.input_conditioner.norm_std": std}, "projector": {}, "final_norm": {}}
    radio, _, _ = nemotron35.build_parts(nemotron35.BOLT, parts, torch.bfloat16)
    conditioner = radio.radio_model.input_conditioner
    assert conditioner.norm_mean.dtype == torch.float32
    assert torch.equal(conditioner.norm_mean, mean) and torch.equal(conditioner.norm_std, std)


@pytest.fixture(scope="module")
def dinov3_parent():
    return DINOv3ViTModel.from_pretrained(dinov3.SOURCE_REPO, revision=dinov3.SOURCE_REVISION, dtype=DTYPE).to(DEVICE).eval()


@needs_weights
@pytest.mark.parametrize("resolution", [448, 512])
def test_dinov3_every_depth_matches_independent_parent_block_hooks(dinov3_parent, resolution):
    """DINOv3 has no release of ours, so its check is here: a separate parent load, hooked by block, never via the adapter."""
    assert sum(p.numel() for p in dinov3_parent.parameters()) == 303_129_600
    assert dinov3_parent.config.num_register_tokens == 4
    adapter = dinov3.DINOv3Adapter(resolution=resolution, dtype=DTYPE, device=DEVICE)
    pixels = adapter.collate([adapter.preprocess()(image) for image in fixed_images()])
    streams, handles = {}, []
    for depth in range(3, 25, 3):
        def capture(_module, _inputs, output, depth=depth):
            streams[depth] = output[:, dinov3.PREFIX_TOKENS:].cpu()
        handles.append(dinov3_parent.model.layer[depth - 1].register_forward_hook(capture))
    try:
        with torch.inference_mode():
            final = dinov3_parent(pixels["pixel_values"].to(DEVICE, DTYPE)).last_hidden_state[:, dinov3.PREFIX_TOKENS:].cpu()
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
