import torch

from vtb import probe, probe_run
from vtb.channel_standardization import StandardizedReducer


def test_recovered_formula_including_sample_std_and_constant_channels():
    tokens = torch.randn(12, 3, 6, generator=torch.Generator().manual_seed(9))
    tokens[..., 0] = 7
    flat = tokens.reshape(-1, 6).float()
    mean = flat.mean(0, keepdim=True)
    scale = flat.std(0, keepdim=True).clamp_min(1e-6)
    torch.manual_seed(4)
    _, _, basis = torch.svd_lowrank((flat - mean) / scale, q=min(3 + 16, min(flat.shape)))
    torch.manual_seed(4)
    reducer = StandardizedReducer.fit(tokens, 3)
    assert torch.equal(reducer.mean, mean)
    assert torch.equal(reducer.scale, scale)
    assert torch.equal(reducer.basis, basis[:, :3].contiguous())
    torch.testing.assert_close(reducer(tokens), ((tokens - mean) / scale) @ basis[:, :3])
    assert torch.isfinite(reducer(tokens)).all()


def test_standardized_fit_uses_only_training_split_and_preserves_rng():
    tokens = torch.randn(32, 4, 24, generator=torch.Generator().manual_seed(9))
    split = probe_run.split_indices(32)
    altered = tokens.clone()
    altered[split.val] = 1e6
    altered[split.test] = -1e6
    torch.manual_seed(11)
    rng = torch.random.get_rng_state()
    first = probe_run.fit_reducer(tokens, split, 8, standardize_channels=True)
    assert torch.equal(rng, torch.random.get_rng_state())
    second = probe_run.fit_reducer(altered, split, 8, standardize_channels=True)
    assert torch.equal(first.basis, second.basis)
    assert torch.equal(first.scale, second.scale)
    assert torch.equal(first.mean, second.mean)
    assert isinstance(probe_run.fit_reducer(tokens, split, 8), probe.Reducer)


def test_recovered_scale_invariance_and_primary_default():
    torch.manual_seed(0)
    tokens = torch.randn(32, 4, 6) @ torch.randn(6, 6)
    massive = tokens * torch.tensor([1., 1., 1e4, 1., 1., 1.])
    def gram(values):
        torch.manual_seed(0)
        reduced = StandardizedReducer.fit(values, 3)(values).flatten(0, 1)
        return reduced @ reduced.T
    torch.testing.assert_close(gram(tokens), gram(massive), rtol=1e-3, atol=1e-3)
    args = probe_run.build_parser().parse_args(["--run", "cache", "--labels", "labels.json"])
    assert not args.standardize_channels
