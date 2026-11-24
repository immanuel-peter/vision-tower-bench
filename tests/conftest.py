import pytest
import torch

from vtb.adapters import hy4

ADAPTERS = {"hy4": hy4.Hy4Adapter, "hy4_flash": hy4.Hy4FlashAdapter}


@pytest.fixture
def tiny_hy4_adapter():
    """Eight blocks on a 4x4 position table, so the adapter's eight depth points are every block."""

    def build(checkpoint: hy4.Checkpoint, side: int = 64) -> hy4.Hy4TowerAdapter:
        torch.manual_seed(0)
        config = checkpoint.config_class(
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=8,
            num_attention_heads=2,
            patch_size=16,
            max_image_size=64,
            out_hidden_size=12,
        )
        adapter = ADAPTERS[checkpoint.name].__new__(ADAPTERS[checkpoint.name])
        adapter.model = checkpoint.model_class(config).eval()
        for parameter in adapter.model.parameters():
            torch.nn.init.normal_(parameter, std=0.2)
        adapter.num_layers, adapter.resolution = 8, side
        adapter.device, adapter.dtype = "cpu", torch.float32
        adapter.model_id = checkpoint.model_id
        return adapter

    return build
