"""Check DINOv3 adapter features against block hooks on the pinned parent.

The reference is a separate parent load that never calls the adapter or requests
hidden states, so shifted depths and a doubled final norm both fail.
"""

import os

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import DINOv3ViTModel

from vtb.adapters.dinov3 import PREFIX_TOKENS, SOURCE_REPO, SOURCE_REVISION, DINOv3Adapter

DEVICE = os.environ.get("VTB_RELEASE_DEVICE", "cpu")
DTYPE = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32

pytestmark = pytest.mark.skipif(
    os.environ.get("VTB_SKIP_WEIGHTS") == "1",
    reason="VTB_SKIP_WEIGHTS=1",
)


def patterned_images() -> list[Image.Image]:
    y, x = np.mgrid[:376, :613]
    image = np.stack([x % 256, y % 256, (x // 16 * 31 + y // 16 * 17) % 256], -1).astype("uint8")
    return [Image.fromarray(image), Image.fromarray(image[::-1, ::-1].copy()).rotate(90, expand=True)]


@pytest.fixture(scope="module")
def parent():
    return DINOv3ViTModel.from_pretrained(SOURCE_REPO, revision=SOURCE_REVISION, dtype=DTYPE).to(DEVICE).eval()


def test_pinned_parent_is_vit_large_with_four_registers(parent):
    assert sum(p.numel() for p in parent.parameters()) == 303_129_600
    assert parent.config.num_register_tokens == 4


@pytest.mark.parametrize("resolution", [448, 512])
def test_every_depth_matches_independent_parent_block_hooks(parent, resolution):
    adapter = DINOv3Adapter(resolution=resolution, dtype=DTYPE, device=DEVICE)
    pixels = adapter.collate([adapter.preprocess()(image) for image in patterned_images()])
    streams = {}
    handles = []
    for depth in range(3, 25, 3):
        def capture(_module, _inputs, output, depth=depth):
            streams[depth] = output[:, PREFIX_TOKENS:].cpu()
        handles.append(parent.model.layer[depth - 1].register_forward_hook(capture))
    try:
        with torch.inference_mode():
            final = parent(pixels.to(DEVICE, DTYPE)).last_hidden_state[:, PREFIX_TOKENS:].cpu()
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
