import torch

from vtb.standardized_label_budget import StandardizedReducer, fit_reducer
from vtb.probe_run import split_indices


def test_standardization_matches_the_archived_patch_sample_std_rule():
    tokens = torch.arange(48).reshape(4, 2, 6).float()
    torch.manual_seed(0)
    reducer = StandardizedReducer.fit(tokens, 3)
    flat = tokens.reshape(-1, 6)
    mean = flat.mean(dim=0, keepdim=True)
    scale = flat.std(dim=0, keepdim=True).clamp_min(1e-6)
    torch.manual_seed(0)
    _, _, basis = torch.svd_lowrank((flat - mean) / scale, q=6)
    torch.testing.assert_close(reducer(tokens), ((tokens - mean) / scale) @ basis[:, :3], rtol=0, atol=0)
    torch.testing.assert_close(reducer.scale, scale, rtol=0, atol=0)


def test_fit_uses_only_training_images_and_preserves_rng():
    features = torch.randn(20, 4, 6)
    split = split_indices(20)
    before = torch.random.get_rng_state()
    original = fit_reducer(features, split, 3, standardize_channels=True)
    assert torch.equal(before, torch.random.get_rng_state())
    features[split.test] += 1e6
    features[split.val] -= 1e6
    changed = fit_reducer(features, split, 3, standardize_channels=True)
    torch.testing.assert_close(original.basis, changed.basis, rtol=0, atol=0)
    torch.testing.assert_close(original.scale, changed.scale, rtol=0, atol=0)

