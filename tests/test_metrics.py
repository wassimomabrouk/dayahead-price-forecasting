"""
Metrics, checked against values computed by hand.

These are the numbers every claim in the project ultimately rests on, so they
are tested against arithmetic rather than against each other.
"""

import numpy as np
import pytest

# Standard normal quantiles at the nine levels, for building synthetic
# calibration frames.
NORMAL_Z = {0.1: -1.2816, 0.2: -0.8416, 0.3: -0.5244, 0.4: -0.2533,
            0.5: 0.0, 0.6: 0.2533, 0.7: 0.5244, 0.8: 0.8416, 0.9: 1.2816}

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


# ------------------------------------------------- conformal, section 10
def _cal_frame(y, quantile_preds):
    import pandas as pd
    df = pd.DataFrame({"y_true": y})
    for q, v in quantile_preds.items():
        df[f"q{int(q * 100):02d}"] = v
    return df


def test_an_already_calibrated_model_gets_no_correction():
    """
    The property that makes this a correction and not a second spread. An
    earlier version scored each quantile against the point forecast instead
    of against itself, which double-counted the model's own spread and
    overshot a nominal 0.80 to 0.912.
    """
    import numpy as np
    from dayahead.evaluation.conformal import QUANTILES, _deltas

    rng = np.random.default_rng(0)
    y = rng.normal(50, 10, 200_000)
    perfect = {q: np.full(len(y), 50 + 10 * NORMAL_Z[q]) for q in QUANTILES}
    d = _deltas(_cal_frame(y, perfect))
    for q, v in d.items():
        assert abs(v) < 0.3, f"q{q} corrected by {v:.3f} when already calibrated"


def test_conformal_widens_intervals_that_are_too_narrow():
    import numpy as np
    from dayahead.evaluation.conformal import QUANTILES, _deltas

    rng = np.random.default_rng(0)
    y = rng.normal(50, 10, 200_000)
    # Half the spread it should have.
    narrow = {q: np.full(len(y), 50 + 5 * NORMAL_Z[q]) for q in QUANTILES}
    d = _deltas(_cal_frame(y, narrow))
    assert d[0.1] < -3, "lower quantile should be pushed down"
    assert d[0.9] > 3, "upper quantile should be pushed up"
    assert abs(d[0.5]) < 0.3, "the median was already right"


def test_conformal_recovers_a_constant_bias():
    import numpy as np
    from dayahead.evaluation.conformal import QUANTILES, _deltas

    rng = np.random.default_rng(0)
    y = rng.normal(50, 10, 200_000)
    shifted = {q: np.full(len(y), 40 + 10 * NORMAL_Z[q]) for q in QUANTILES}
    d = _deltas(_cal_frame(y, shifted))
    for q, v in d.items():
        assert abs(v - 10.0) < 0.4


def test_monotonicity_is_enforced_after_correction():
    """Independent per-quantile shifts can reorder the quantiles."""
    import pandas as pd
    from dayahead.evaluation.conformal import _enforce_monotone

    df = pd.DataFrame({
        "q10": [50.0], "q20": [40.0], "q30": [45.0], "q40": [46.0],
        "q50": [47.0], "q60": [48.0], "q70": [49.0], "q80": [51.0],
        "q90": [52.0],
    })
    fixed, crossings = _enforce_monotone(df)
    assert crossings >= 1
    vals = [fixed[f"q{q}0"].iloc[0] for q in range(1, 10)]
    assert vals == sorted(vals)


def test_calibration_never_uses_the_fold_it_corrects():
    """
    The guarantee rests on the calibration rows preceding the test fold. A
    fold leaking into its own calibration set would produce coverage that
    looks excellent and means nothing.
    """
    import numpy as np
    import pandas as pd
    from dayahead.evaluation.conformal import CALIBRATION_FOLDS, calibrate_model

    rng = np.random.default_rng(0)
    rows = []
    for i in range(30):
        fold = f"2021-{i % 12 + 1:02d}-{i:02d}"
        # The last fold is wildly different; if it calibrated itself the
        # correction would absorb the shift.
        shift = 500.0 if i == 29 else 0.0
        for h in range(24):
            base = rng.normal(50, 5)
            rows.append({
                "timestamp": pd.Timestamp("2021-01-01", tz="UTC")
                             + pd.Timedelta(hours=i * 24 + h),
                "model": "m", "fold": fold, "regime": "post-crisis",
                "y_true": base + shift, "y_pred": base,
                **{f"q{q}0": base for q in range(1, 10)},
            })
    preds = pd.DataFrame(rows)
    out, meta = calibrate_model(preds)

    last = out[out["fold"] == "2021-06-29"]
    if len(last):
        # Corrections come from earlier folds, which had no shift, so the
        # calibrated interval must fail to cover the shifted outturn.
        assert (last["y_true"] > last["q90"]).all()
