"""Validation helpers, tested on synthetic panels rather than the real store."""

import pandas as pd

from dayahead.data.validate import interior_nulls, null_runs


def _series(values):
    idx = pd.date_range("2020-01-01", periods=len(values), freq="h", tz="UTC")
    return pd.Series(values, index=idx, dtype="Float64")


def test_trailing_nulls_are_not_interior():
    s = _series([1.0, 2.0, 3.0, None, None])
    assert len(interior_nulls(s)) == 0


def test_leading_nulls_are_not_interior():
    s = _series([None, None, 1.0, 2.0])
    assert len(interior_nulls(s)) == 0


def test_interior_nulls_are_found():
    s = _series([1.0, None, None, 4.0, None])
    assert len(interior_nulls(s)) == 2


def test_null_runs_groups_consecutive_hours():
    s = _series([1.0, None, None, 4.0, None, 6.0])
    runs = null_runs(interior_nulls(s))
    assert len(runs) == 2
    assert (runs[0][1] - runs[0][0]) == pd.Timedelta(hours=1)


def test_null_runs_on_empty_index():
    s = _series([1.0, 2.0])
    assert null_runs(interior_nulls(s)) == []


# --------------------------------------------------- DST regression, section 4
def test_calendar_day_offset_survives_dst():
    """
    Regression test. Building price features with a fixed 24-hour offset
    silently emptied every day following a DST transition, because a 23- or
    25-hour day is not 24 hours of elapsed time. Calendar-day arithmetic must
    keep those days intact.
    """
    from dayahead import config as cfg
    from dayahead.features.lags import build as build_lags

    idx = pd.date_range("2019-10-20", "2019-11-03 23:00", freq="h",
                        tz=cfg.LOCAL_TZ)
    price = pd.Series(range(len(idx)), index=idx, dtype="float64")
    df, _ = build_lags(price, idx)

    # The day after the autumn transition must not be entirely missing.
    day_after = df.loc["2019-10-28"]
    assert day_after["price_d1_same_hour"].notna().sum() >= 23
    assert day_after["price_d1_mean"].notna().all()


def test_no_whole_day_is_lost_after_warmup():
    from dayahead import config as cfg
    from dayahead.features.lags import build as build_lags

    idx = pd.date_range("2019-01-01", "2019-12-31 23:00", freq="h",
                        tz=cfg.LOCAL_TZ)
    price = pd.Series(range(len(idx)), index=idx, dtype="float64")
    df, _ = build_lags(price, idx)

    after_warmup = df.loc["2019-01-15":]
    per_day = after_warmup["price_d1_same_hour"].notna().groupby(
        after_warmup.index.normalize()).sum()
    empty_days = per_day[per_day == 0]
    assert len(empty_days) == 0, f"days lost: {list(empty_days.index)}"
