from types import SimpleNamespace

import torch
import pytest

from vtb.probe_run import run_cell, split_indices


@pytest.mark.parametrize("readout", ("mean", "attention"))
def test_prediction_retention_preserves_fit_scores_and_random_state(readout):
    features = torch.arange(40 * 3 * 5, dtype=torch.float32).reshape(40, 3, 5) / 100
    labels = torch.arange(40) % 2
    split = split_indices(40)
    args = SimpleNamespace(learning_rates=[.001, .01], readout=readout, device="cpu",
                           epochs=1, batch_size=8, seeds=3, save_predictions=False)
    baseline = run_cell(features, labels, split, 2, args)
    state = torch.random.get_rng_state()
    args.save_predictions = True
    saved = run_cell(features, labels, split, 2, args)
    assert torch.equal(state, torch.random.get_rng_state())
    assert baseline == {key: saved[key] for key in baseline}
    assert saved["test_labels"] == labels[split.test].tolist()
    for index, predictions in enumerate(saved["test_predictions_by_seed"]):
        actual = sum(x == y for x, y in zip(predictions, saved["test_labels"], strict=True)) / len(predictions)
        assert actual == saved["test_accuracy_by_seed"][index]
