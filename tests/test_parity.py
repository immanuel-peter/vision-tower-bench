"""Every published Tower against the checkpoint it came from. See vtb/parity/."""

import os

import pytest
import torch
from transformers import Mistral3Config, PixtralVisionConfig, PixtralVisionModel
from transformers.models.mistral3.modeling_mistral3 import Mistral3MultiModalProjector

from vtb import parity
from vtb.adapters import mistral_large4 as ml4
from vtb.parity import mistral_large4 as validate

DEVICE = os.environ.get("VTB_PARITY_DEVICE", "cpu")

needs_weights = pytest.mark.skipif(
    os.environ.get("VTB_SKIP_WEIGHTS") == "1", reason="VTB_SKIP_WEIGHTS=1"
)


@needs_weights
@pytest.mark.parametrize("slug", parity.released())
def test_release_matches_its_pinned_parent(slug):
    if parity.requires_cuda(slug) and not DEVICE.startswith("cuda"):
        pytest.skip(f"{slug} parity needs VTB_PARITY_DEVICE=cuda")
    report = parity.validate(slug, DEVICE)
    failing = {
        key: cell
        for section in ("weights", "forward")
        for key, cell in report.get(section, {}).items()
        if not cell.get("bit_exact", cell.get("passed", True))
    }
    assert report["passed"], failing


# The checkers themselves: Mistral's reference needs a stand-in for xformers attention.


def tiny_tower_config(**overrides) -> PixtralVisionConfig:
    settings = dict(
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
    settings.update(overrides)
    return PixtralVisionConfig(**settings)


def tiny_projector_config(tower: PixtralVisionConfig) -> Mistral3Config:
    return Mistral3Config(
        vision_config=tower.to_dict(),
        text_config={"model_type": "mistral", "hidden_size": 40, "rms_norm_eps": 1e-5},
        spatial_merge_size=2,
        projector_hidden_act="gelu",
        multimodal_projector_bias=False,
        vision_feature_layer=-1,
    )


def tiny_parts():
    torch.manual_seed(0)
    config = tiny_tower_config()
    tower = PixtralVisionModel(config).eval()
    projector = Mistral3MultiModalProjector(tiny_projector_config(config)).eval()
    return tower, projector


def test_attention_substitute_matches_softmax_attention_with_block_masks():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 9, 3, 8) for _ in range(3))
    mask = validate.BlockDiagonalMask.from_seqlens([4, 5])
    actual = validate.memory_efficient_attention(q, k, v, mask)
    for start, stop in ((0, 4), (4, 9)):
        qs, ks, vs = (t[:, start:stop].transpose(1, 2) for t in (q, k, v))
        weights = torch.softmax(qs @ ks.transpose(-1, -2) / 8**0.5, dim=-1)
        torch.testing.assert_close(
            actual[:, start:stop], (weights @ vs).transpose(1, 2), rtol=1e-5, atol=1e-6
        )
    unmasked = validate.memory_efficient_attention(q, k, v)
    expected = torch.softmax(
        q.transpose(1, 2) @ k.permute(0, 2, 3, 1) / 8**0.5, -1
    ) @ v.transpose(1, 2)
    torch.testing.assert_close(unmasked, expected.transpose(1, 2), rtol=1e-5, atol=1e-6)
    half = validate.memory_efficient_attention(q.bfloat16(), k.bfloat16(), v.bfloat16())
    assert half.dtype == torch.bfloat16


def tiny_corrected_bundle(tmp_path):
    from transformers import AutoModel

    torch.manual_seed(0)
    config = ml4.PixtralLarge4VisionConfig(
        **{k: v for k, v in tiny_tower_config().to_dict().items() if k != "model_type"}
    )
    config.auto_map = {
        "AutoConfig": "modeling_pixtral_large4.PixtralLarge4VisionConfig",
        "AutoModel": "modeling_pixtral_large4.PixtralLarge4VisionModel",
    }
    model = ml4.PixtralLarge4VisionModel(config).eval()
    model.save_pretrained(tmp_path)
    (tmp_path / "modeling_pixtral_large4.py").write_bytes(
        (ml4.ROOT / "vtb/models/pixtral_large4.py").read_bytes()
    )
    loaded = AutoModel.from_pretrained(
        tmp_path, trust_remote_code=True, local_files_only=True
    ).eval()
    return model, loaded


def test_validator_release_path_equals_the_adapter_path(tmp_path):
    _, loaded = tiny_corrected_bundle(tmp_path)
    _, projector = tiny_parts()
    pixels = torch.randn(2, 3, 112, 112).bfloat16()
    loaded, projector = loaded.bfloat16(), projector.bfloat16()
    with torch.inference_mode():
        released = validate.release_stages(loaded, projector, pixels, (2, 4))
        single = validate.release_stages(loaded, projector, pixels[:1], (2, 4))
        shared = validate.candidate_stages(loaded, projector, pixels[:1], (2, 4))
    assert set(released) == {"tower_L2", "tower_L4", "merged", "projected"}
    for name, tensor in single.items():
        assert torch.equal(tensor, shared[name])
        assert torch.equal(released[name][0], tensor)
