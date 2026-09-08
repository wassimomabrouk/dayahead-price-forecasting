"""
Metrics, checked against values computed by hand.

These are the numbers every claim in the project ultimately rests on, so they
are tested against arithmetic rather than against each other.
"""

import numpy as np
import pytest

from dayahead.evaluation.metrics import (
    bias, coverage, interval_width, mae, mean_pinball, pinball_loss, rmse,
    skill_score,
)


def test_mae_and_rmse_by_hand():
    t = [10.0, 20.0, 30.0]
    p = [12.0, 17.0, 30.0]
    # errors 2, 3, 0
    assert mae(t, p) == pytest.approx(5 / 3)
    assert rmse(t, p) == pytest.approx(np.sqrt(13 / 3))


def test_bias_signs_are_prediction_minus_truth():
    assert bias([10.0], [12.0]) == pytest.approx(2.0)
    assert bias([10.0], [8.0]) == pytest.approx(-2.0)


def test_metrics_ignore_non_finite_pairs():
    t = [10.0, np.nan, 30.0]
    p = [12.0, 17.0, np.nan]
    assert mae(t, p) == pytest.approx(2.0)


def test_negative_prices_are_handled():
    """The whole reason MAPE is excluded. MAE must stay well defined."""
    assert mae([-50.0, 0.0, 50.0], [-45.0, 5.0, 45.0]) == pytest.approx(5.0)


def test_skill_score_is_zero_against_itself():
    t = [1.0, 2.0, 3.0]
    p = [1.5, 2.5, 2.0]
    assert skill_score(t, p, p) == pytest.approx(0.0)


def test_skill_score_positive_when_better():
    t = [10.0, 20.0, 30.0]
    good = [11.0, 21.0, 31.0]     # MAE 1
    bad = [14.0, 24.0, 34.0]      # MAE 4
    assert skill_score(t, good, bad) == pytest.approx(0.75)
    assert skill_score(t, bad, good) == pytest.approx(-3.0)


def test_skill_score_is_undefined_against_a_perfect_baseline():
    """
    A baseline with zero error makes the ratio undefined. Returning nan is
    deliberate: silently reporting minus infinity, or clamping to some large
    negative number, would put a fabricated value into a results table.
    """
    t = [10.0, 20.0]
    assert np.isnan(skill_score(t, [0.0, 0.0], t))


def test_pinball_at_median_is_half_absolute_error():
    t = np.array([10.0, 20.0, 30.0])
    p = np.array([12.0, 17.0, 30.0])
    assert pinball_loss(t, p, 0.5) == pytest.approx(mae(t, p) / 2)


def test_pinball_penalises_asymmetrically():
    """At q=0.9 under-prediction should hurt more than over-prediction."""
    under = pinball_loss([10.0], [8.0], 0.9)   # predicted too low
    over = pinball_loss([10.0], [12.0], 0.9)   # predicted too high
    assert under > over


def test_pinball_asymmetry_flips_at_low_quantile():
    under = pinball_loss([10.0], [8.0], 0.1)
    over = pinball_loss([10.0], [12.0], 0.1)
    assert over > under


def test_mean_pinball_averages_the_nine():
    t = np.zeros(100)
    preds = {q: np.zeros(100) for q in
             [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    assert mean_pinball(t, preds) == pytest.approx(0.0)


def test_coverage_counts_inclusive_boundaries():
    assert coverage([1.0, 5.0, 9.0], [2.0] * 3, [8.0] * 3) == pytest.approx(1 / 3)
    assert coverage([2.0], [2.0], [8.0]) == pytest.approx(1.0)


def test_interval_width_is_mean_of_differences():
    assert interval_width([0.0, 1.0], [10.0, 3.0]) == pytest.approx(6.0)


def test_wide_intervals_get_free_coverage():
    """
    Why width is always reported next to coverage: a useless interval scores
    perfectly on coverage alone.
    """
    t = np.random.default_rng(0).normal(size=500)
    assert coverage(t, np.full(500, -1e6), np.full(500, 1e6)) == 1.0
    assert interval_width(np.full(500, -1e6), np.full(500, 1e6)) == 2e6
