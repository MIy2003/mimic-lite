"""CHIP terms follow the framework's environment-deferred component API."""
import inspect

from mimic_lite.tasks.chip import (
    ChipTracking, chip_history, chip_tracking_reward, chip_metric,
)


def test_chip_terms_construct_without_environment():
    history = chip_history(history_length=10)
    reward = chip_tracking_reward(weight=2.0, sigma=0.03)
    metric = chip_metric(weight=1.0)
    for term in (history, reward, metric):
        assert not term.initialized
    assert not hasattr(history, "buffers")
    assert not hasattr(reward, "weights")
    assert "env" not in inspect.signature(ChipTracking).parameters


def test_reset_callbacks_accept_reset_tensordict():
    for cls in (ChipTracking, chip_history):
        inspect.signature(cls.reset).bind(None, None, None)
