"""
Backtest harness. The leakage guards matter more than the plumbing.
"""

import numpy as np
import pandas as pd
import pytest

from dayahead import config as cfg
from dayahead.evaluation.splits import (
    backtest_frame, eval_months, folds, locked_test_frame, regime_of,
)


def _frame(start="2018-10-01", end="2026-08-31 23:00"):
    idx = pd.date_range(start, end, freq="h", tz=cfg.LOCAL_TZ)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "price_d1_same_hour": rng.normal(50, 10, len(idx)),
        "price_d7_same_hour": rng.normal(50, 10, len(idx)),
        "x_fc_residual": rng.normal(30000, 8000, len(idx)),
        "y": rng.normal(50, 10, len(idx)),
    }, index=idx)
    df["is_usable"] = True
    return df


# ------------------------------------------------------------ locked test
def test_locked_test_set_refuses_casual_access():
    X = _frame()
    with pytest.raises(PermissionError, match="locked test set"):
        locked_test_frame(X)


def test_locked_test_set_opens_with_the_explicit_flag():
    X = _frame()
    t = locked_test_frame(X, i_am_opening_the_locked_test_set=True)
    assert len(t) > 8000
    assert t.index.min() >= pd.Timestamp("2025-09-01", tz=cfg.LOCAL_TZ)


def test_backtest_frame_never_contains_test_rows():
    X = _frame()
    bt = backtest_frame(X)
    assert bt.index.max() < pd.Timestamp("2025-09-01", tz=cfg.LOCAL_TZ)


# ----------------------------------------------------------------- folds
def test_fold_count_matches_the_design():
    """DESIGN.md section 7 pre-commits to 59 folds, 2020-10 through 2025-08."""
    assert len(eval_months()) == 59
    f = folds(_frame())
    assert len(f) == 59
    assert f[0]["month"] == "2020-10"
    assert f[-1]["month"] == "2025-08"


def test_training_always_ends_before_testing_begins():
    for fold in folds(_frame()):
        assert fold["train_index"].max() < fold["test_index"].min()
        assert len(fold["train_index"].intersection(fold["test_index"])) == 0


def test_training_window_expands():
    f = folds(_frame())
    sizes = [len(x["train_index"]) for x in f]
    assert all(b > a for a, b in zip(sizes, sizes[1:]))


def test_folds_cover_all_three_regimes():
    seen = {fold["regime"] for fold in folds(_frame())}
    assert seen == {"pre-crisis", "crisis", "post-crisis"}


def test_regime_boundaries_are_where_the_design_says():
    assert regime_of(pd.Timestamp("2021-08-31 23:00", tz=cfg.LOCAL_TZ)) == "pre-crisis"
    assert regime_of(pd.Timestamp("2021-09-01 00:00", tz=cfg.LOCAL_TZ)) == "crisis"
    assert regime_of(pd.Timestamp("2022-12-31 23:00", tz=cfg.LOCAL_TZ)) == "crisis"
    assert regime_of(pd.Timestamp("2023-01-01 00:00", tz=cfg.LOCAL_TZ)) == "post-crisis"


# ---------------------------------------------------------------- models
def test_baselines_run_through_the_harness():
    from dayahead.evaluation.backtest import run_backtest, summarise
    from dayahead.models.naive import all_baselines

    X = _frame(end="2021-06-30 23:00")
    preds = run_backtest(X, all_baselines(), verbose=False)
    assert set(preds["model"]) == {"B1_weekly_naive", "B2_daily_naive"}
    assert preds["q10"].notna().all()
    assert (preds["q10"] <= preds["q90"]).all()

    table = summarise(preds)
    assert len(table) == 2
    b1 = table[table["model"] == "B1_weekly_naive"].iloc[0]
    assert b1["skill"] == pytest.approx(0.0, abs=1e-9)


def test_harness_raises_if_a_fold_would_leak(monkeypatch):
    """The guard must actually fire, not merely exist."""
    from dayahead.evaluation import backtest as bt

    idx = pd.date_range("2020-01-01", periods=100, freq="h", tz=cfg.LOCAL_TZ)
    with pytest.raises(AssertionError, match="training data reaches"):
        bt._assert_no_overlap(idx, idx[50:], "fake-fold")


# ------------------------------------------- state space conditioning, section 7
def test_state_space_models_condition_on_realised_history():
    """
    Regression test for a silent failure.

    A state space model extended with NaN endogenous values receives no new
    information, so what looks like a one-day-ahead forecast becomes a
    month-long extrapolation. It does not raise, it does not warn, and on a
    falling market it produced a bias of +148 EUR/MWh where the correct
    figure was +24.

    The distinction matters because passing the realised series is easily
    mistaken for leakage. It is not: get_prediction with dynamic=False
    returns the one-step-ahead prediction for position t, conditioning on
    observations through t-1 only.
    """
    import numpy as np
    from dayahead.models.arima import PerHourSARIMAX

    idx = pd.date_range("2021-01-01", "2021-06-30 23:00", freq="h",
                        tz=cfg.LOCAL_TZ)
    rng = np.random.default_rng(0)
    # A clear downward level shift partway through the test window.
    level = np.where(np.arange(len(idx)) < len(idx) * 0.8, 200.0, 60.0)
    y = pd.Series(level + rng.normal(0, 5, len(idx)), index=idx)
    X = pd.DataFrame(index=idx)

    split = int(len(idx) * 0.8)
    m = PerHourSARIMAX("t", order=(1, 1, 1), exog_cols=[]).fit(
        X.iloc[:split], y.iloc[:split])

    blind = m.predict(X.iloc[split:])
    informed = m.predict(X.iloc[split:], y=y.iloc[split:])
    truth = y.iloc[split:].to_numpy()

    # The mechanism, not the magnitude. How fast an ARIMA converges after a
    # level shift depends on its fitted MA coefficient, and on constant
    # synthetic data that coefficient goes to roughly -1, making it an
    # extreme smoother. So assert that the conditioning state moves at all,
    # which is the thing that was broken.
    hour0 = X.iloc[split:].index.hour == 0
    b, i = blind[hour0], informed[hour0]

    # Blind predictions receive no information and barely move.
    assert np.nanstd(b) < 1.0

    # Informed predictions track the realised series downward.
    assert np.nanstd(i) > 10 * max(np.nanstd(b), 1e-9)
    assert i[-1] < i[0] - 20

    # And they end closer to the truth than the blind ones do.
    assert abs(i[-1] - truth[hour0][-1]) < abs(b[-1] - truth[hour0][-1])


# ------------------------------------------------- LightGBM, section 8
def test_quantile_crossing_is_repaired():
    """
    Nine independently fitted quantile models are not guaranteed to be
    ordered. Left alone that produces negative interval widths and a
    meaningless coverage figure, so predictions are sorted per row.
    """
    import numpy as np
    from dayahead.models.gbm import PerHourLightGBM

    qs = [0.1, 0.5, 0.9]
    crossed = {0.1: np.array([50.0, 10.0]),
               0.5: np.array([30.0, 20.0]),
               0.9: np.array([40.0, 30.0])}
    fixed, n = PerHourLightGBM._rearrange(crossed, qs)
    # One adjacent pair is out of order: [50, 30, 40] has a single negative
    # step. The second row is already sorted.
    assert n == 1
    assert list(fixed[0.1]) == [30.0, 10.0]
    assert list(fixed[0.9]) == [50.0, 30.0]
    for a, b in zip(qs, qs[1:]):
        assert (fixed[a] <= fixed[b]).all()


def test_rearrangement_leaves_ordered_predictions_alone():
    import numpy as np
    from dayahead.models.gbm import PerHourLightGBM

    qs = [0.1, 0.5, 0.9]
    ok = {0.1: np.array([10.0]), 0.5: np.array([20.0]), 0.9: np.array([30.0])}
    fixed, n = PerHourLightGBM._rearrange(ok, qs)
    assert n == 0
    assert fixed[0.5][0] == 20.0


def test_anchored_model_can_predict_outside_the_training_range():
    """
    The defect that section 8 exposed, stated as a test.

    A tree predicts leaf averages, so a model trained on the price level can
    never emit a value above the maximum it saw. Measured on the real crisis
    folds: true prices reached 871 EUR/MWh, predictions capped at 418.6.

    Anchoring the target on a naive baseline removes the level, so the
    residual stays inside the training range even when the price does not.
    This test asserts the ceiling is gone, which is the whole reason the
    variant exists.
    """
    import numpy as np
    from dayahead.models.gbm import lightgbm_anchored

    idx = pd.date_range("2021-01-01", "2021-12-31 23:00", freq="h",
                        tz=cfg.LOCAL_TZ)
    rng = np.random.default_rng(0)
    # The level triples on a fixed date, and the split is on that same date,
    # so no part of the high regime can leak into training. An earlier
    # version cut by position after filtering rows, which silently moved the
    # boundary and put high prices in the training window.
    shift_date = pd.Timestamp("2021-10-01", tz=cfg.LOCAL_TZ)
    level = np.where(idx < shift_date, 50.0, 150.0)
    y = pd.Series(level + rng.normal(0, 5, len(idx)), index=idx)

    X = pd.DataFrame(index=idx)
    X["price_d1_same_hour"] = y.shift(freq=pd.Timedelta(days=1)).reindex(idx)
    X["noise"] = rng.normal(0, 1, len(idx))
    ok = X["price_d1_same_hour"].notna()
    X, y = X[ok], y[ok]

    train = X.index < shift_date
    m = lightgbm_anchored().fit(X[train], y[train])
    # Skip the first day after the shift: its anchor is still the old level,
    # so no model could know the level had moved.
    future = X.index >= shift_date + pd.Timedelta(days=2)
    pred = m.predict(X[future])

    train_max = y[train].max()
    assert np.nanmax(pred) > train_max * 1.5, (
        f"anchored predictions capped at {np.nanmax(pred):.1f} "
        f"against a training maximum of {train_max:.1f}"
    )
