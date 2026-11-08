import pytest
import torch
from safetensors.torch import save_file

from examples.kimi_k31_pro_inference import Projector
from vtb.adapters.moonvit_v2 import load_projector


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_example_projector_matches_adapter_and_exact_gelu(tmp_path, dtype):
    config = dict(input_size=4, hidden_size=4, output_size=6, norm_eps=1e-5)
    weights = {
        "proj.0.weight": torch.eye(4, dtype=dtype),
        "proj.2.weight": torch.arange(24, dtype=dtype).reshape(6, 4) / 24,
        "post_norm.weight": torch.linspace(.5, 1.5, 6, dtype=dtype),
    }
    save_file(weights, tmp_path / "projector.safetensors")
    example = Projector(config).to(dtype).eval()
    example.load_state_dict(weights, strict=True)
    adapter = load_projector(dtype, str(tmp_path))
    assert example.proj[1].approximate == adapter.proj[1].approximate == "none"
    assert set(example.state_dict()) == set(adapter.state_dict()) == set(weights)
    tokens = torch.tensor([[[-2., -.5, .5, 2.], [-1., 0., 1., 3.]]], dtype=dtype)
    with torch.inference_mode():
        expected = adapter.post_norm(adapter.proj(tokens))
        torch.testing.assert_close(example(tokens), expected, atol=0, rtol=0)
