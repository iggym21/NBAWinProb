import numpy as np
import pytest
from scripts.evaluate import brier_score, log_loss_score, calibration_curve, accuracy_by_time_bucket, TIME_BUCKETS


def test_brier_score_perfect_predictions_is_zero():
    probs = np.array([1.0, 0.0, 1.0])
    targets = np.array([1.0, 0.0, 1.0])
    assert brier_score(probs, targets) == pytest.approx(0.0)


def test_brier_score_worst_predictions_is_one():
    probs = np.array([0.0, 1.0])
    targets = np.array([1.0, 0.0])
    assert brier_score(probs, targets) == pytest.approx(1.0)


def test_log_loss_score_reasonable():
    probs = np.array([0.9, 0.1])
    targets = np.array([1.0, 0.0])
    loss = log_loss_score(probs, targets)
    assert loss > 0
    assert loss == pytest.approx(-np.log(0.9), abs=1e-4)


def test_calibration_curve_bins_and_shape():
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 1, 200)
    targets = (rng.uniform(0, 1, 200) < probs).astype(float)
    curve = calibration_curve(probs, targets, n_bins=5)
    assert len(curve) > 0
    for bucket in curve:
        assert 0.0 <= bucket["predicted_mean"] <= 1.0
        assert 0.0 <= bucket["observed_frequency"] <= 1.0
        assert bucket["count"] > 0


def test_accuracy_by_time_bucket_perfect_predictions():
    probs = np.array([0.9, 0.1, 0.9, 0.1])
    targets = np.array([1.0, 0.0, 1.0, 0.0])
    seconds_remaining = np.array([2000, 2000, 100, 100])  # first two in one bucket, last two in another
    result = accuracy_by_time_bucket(probs, targets, seconds_remaining)
    assert set(result.keys()) == {name for name, _, _ in TIME_BUCKETS}
    populated = {k: v for k, v in result.items() if v["count"] > 0}
    assert all(v["accuracy"] == pytest.approx(1.0) for v in populated.values())
