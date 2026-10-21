"""The batch-invariance check: final-cell selection and leakage detection."""

import torch

from scripts import batch_invariance
from vtb.feature_batch import FeatureBatch


class Adapter:
    """Tower depths 1 and 2; 'leak' mixes the batch mean into every image."""

    num_layers = 2
    model_id = "toy"

    def __init__(self, leak: float = 0.0):
        self.leak = leak

    @staticmethod
    def collate(samples):
        return torch.stack(samples)

    def extract(self, inputs, image_ids):
        tokens = inputs + self.leak * inputs.mean(0, keepdim=True)
        for layer in (1, 2):
            yield FeatureBatch(tokens * layer, image_ids, self.model_id, "tower", layer, 2, 8)
        yield FeatureBatch(tokens.sum(-1, keepdim=True), image_ids, self.model_id, "merged", 2, 2, 8)


def values():
    torch.manual_seed(0)
    return [torch.randn(4, 3) for _ in range(5)], [f"image{i}" for i in range(5)]


def test_final_cells_keep_only_the_final_layer_and_requested_stages():
    images, ids = values()
    cells = batch_invariance.final_cells(Adapter(), torch.stack(images[:2]), ids[:2])
    assert set(cells) == {"tower", "merged"}
    assert torch.equal(cells["tower"], torch.stack(images[:2]) * 2)
    assert set(batch_invariance.final_cells(Adapter(), torch.stack(images[:2]), ids[:2], ["merged"])) == {"merged"}


def test_independent_images_are_bit_exact_across_batch_sizes():
    images, ids = values()
    adapter = Adapter()
    report = batch_invariance.compare(
        batch_invariance.in_batches(adapter, images, ids, 2, None),
        batch_invariance.one_at_a_time(adapter, images, ids, None),
    )
    assert all(cell["bit_exact"] and cell["max_abs_diff"] == 0 for cell in report.values())
    assert report["tower"]["shape_per_image"] == [4, 3]


def test_cross_image_leakage_is_reported_per_image():
    images, ids = values()
    adapter = Adapter(leak=0.5)
    report = batch_invariance.compare(
        batch_invariance.in_batches(adapter, images, ids, 2, None),
        batch_invariance.one_at_a_time(adapter, images, ids, None),
    )
    tower = report["tower"]
    assert not tower["bit_exact"] and tower["relative_l2"] > 0.01
    assert len(tower["per_image_max_abs_diff"]) == 5
    # The last image is alone in its batch, so it is unaffected.
    assert tower["per_image_max_abs_diff"][-1] == 0
