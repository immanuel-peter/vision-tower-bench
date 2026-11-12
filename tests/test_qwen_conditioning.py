import torch
from scripts.qwen_conditioning import PRIMARY_TRAIN, train_with_loss_log
from vtb.probe_run import split_indices


def test_loss_instrumentation_preserves_training_weights_and_scores():
    torch.manual_seed(11)
    features = torch.randn(32, 4, 8)
    labels = torch.arange(32) % 2
    parameters = (features, labels, split_indices(32), 2, "mean", 0.01, 7, "cpu", 2, 8)
    expected = PRIMARY_TRAIN(*parameters)
    actual, log = train_with_loss_log(*parameters)
    assert actual[1:] == expected[1:]
    for name, value in expected[0].state_dict().items():
        assert torch.equal(value, actual[0].state_dict()[name])
    assert len(log["step_losses"]) == 6
    assert len(log["epoch_losses"]) == 2
    assert log["nonfinite_loss_steps"] == 0

