"""
Gate closure.

This is the project's central claim, so these are the tests that matter most
in the suite. Tests that pass vacuously are worse than no tests, which is why
test_a_deliberately_leaky_feature_is_rejected exists: it proves the check has
teeth by feeding it something that must fail.

None of these touch the raw store, so they run anywhere.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dayahead import config as cfg
from dayahead.features import calendar as cal, exog, lags
from dayahead.features.gate import (
    FeatureSpec,
    availability_time,
    check_spec,
    delivery_date,
    issuance_time,
)


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=cfg.LOCAL_TZ)


# Note the hours chosen on the DST days. 02:00 does not exist on the
# spring-forward day and is ambiguous on the autumn-back day, so both are
# avoided here and tested explicitly below instead.
SAMPLE = pd.DatetimeIndex([
    ts("2019-03-31 03:00"),   # short DST day, first hour after the jump
    ts("2019-10-27 03:00"),   # long DST day, after the repeated hour
    ts("2020-02-29 12:00"),   # leap day
    ts("2021-01-01 00:00"),   # year boundary, first hour
    ts("2022-06-15 13:00"),
    ts("2024-12-31 23:00"),   # year boundary, last hour
])


def _synthetic():
    idx = pd.date_range(ts("2024-06-01 00:00"), periods=24 * 40, freq="h")
    price = pd.Series(range(len(idx)), index=idx, dtype="float64")
    panel = pd.DataFrame({c: 1000.0 for c in cfg.ALL_SERIES}, index=idx)
    panel["fc_residual"] = 20000.0
    panel["fc_load"] = 50000.0
    return idx, price, panel


# ------------------------------------------------------------------- timing
def test_issuance_is_noon_on_the_previous_day():
    assert issuance_time(ts("2024-06-15 00:00")) == ts("2024-06-14 12:00")
    assert issuance_time(ts("2024-06-15 23:00")) == ts("2024-06-14 12:00")


def test_all_hours_of_a_day_share_one_issuance():
    day = pd.date_range(ts("2024-06-15 00:00"), periods=24, freq="h")
    assert len({issuance_time(t) for t in day}) == 1


def test_issuance_crosses_month_and_year_boundaries():
    assert issuance_time(ts("2025-01-01 05:00")) == ts("2024-12-31 12:00")
    assert issuance_time(ts("2024-03-01 05:00")) == ts("2024-02-29 12:00")


def test_spring_forward_hour_does_not_exist():
    """
    Writing this test suite tripped over exactly this. 02:00 local is skipped
    on the spring-forward day, so any code constructing local timestamps by
    hour arithmetic will raise or silently shift. Recorded as a test so the
    trap stays visible.
    """
    with pytest.raises(Exception) as caught:
        pd.Timestamp("2019-03-31 02:00", tz=cfg.LOCAL_TZ)

    # Both pandas majors refuse the timestamp, but they say so differently.
    # 3.x raises ValueError through the stdlib zoneinfo backend, with
    # "nonexistent" in the message. 2.x raises
    # pytz.exceptions.NonExistentTimeError, whose message is only the
    # timestamp itself, so the meaning sits in the class name instead.
    # Matching either keeps the test valid on both, which matters because
    # neuralforecast pins this project to pandas 2.x and that pin may not
    # be permanent.
    where = f"{type(caught.value).__name__} {caught.value}".lower()
    assert "nonexistent" in where, f"unexpected exception: {where}"


def test_autumn_back_hour_is_ambiguous():
    with pytest.raises(Exception):
        pd.Timestamp("2019-10-27 02:00", tz=cfg.LOCAL_TZ)


def test_issuance_works_on_both_dst_days():
    assert issuance_time(ts("2019-03-31 03:00")) == ts("2019-03-30 12:00")
    assert issuance_time(ts("2019-10-27 03:00")) == ts("2019-10-26 12:00")


def test_delivery_date_is_local_not_utc():
    # 23:30 UTC in winter is already the next day in Berlin.
    t = pd.Timestamp("2024-01-15 23:30", tz="UTC")
    assert delivery_date(t) == ts("2024-01-16 00:00")


# ------------------------------------------------------------- availability
def test_todays_own_price_is_published_after_issuance():
    """The auction for D clears at noon on D-1, after our forecast is made."""
    t = ts("2024-06-15 10:00")
    assert availability_time(t, "price", 0) > issuance_time(t)


def test_yesterdays_price_vector_is_available():
    t = ts("2024-06-15 10:00")
    assert availability_time(t, "price", 1) < issuance_time(t)


def test_exog_for_the_delivery_day_is_available_at_issuance():
    t = ts("2024-06-15 10:00")
    assert availability_time(t, "exog", 0) <= issuance_time(t)


def test_exog_for_the_following_day_is_not_available():
    """Tomorrow's forecast has not been published when we forecast today."""
    t = ts("2024-06-15 10:00")
    assert availability_time(t, "exog", -1) > issuance_time(t)


def test_calendar_is_always_available():
    for t in SAMPLE:
        assert availability_time(t, "calendar", 0) < issuance_time(t)


# ------------------------------------------------------------ spec checking
def test_every_feature_carries_an_availability_timestamp():
    idx, price, panel = _synthetic()
    _, s1 = cal.build(idx)
    _, s2 = lags.build(price, idx)
    _, s3 = exog.build(panel, idx)
    assert len(s1 + s2 + s3) > 20
    for spec in s1 + s2 + s3:
        assert spec.kind in ("calendar", "exog", "price")
        assert isinstance(spec.availability(idx[100]), pd.Timestamp)


def test_availability_never_exceeds_issuance_for_real_specs():
    idx, price, panel = _synthetic()
    _, s1 = cal.build(idx)
    _, s2 = lags.build(price, idx)
    _, s3 = exog.build(panel, idx)
    for spec in s1 + s2 + s3:
        check_spec(spec, SAMPLE)


def test_no_forbidden_series_reaches_the_feature_set():
    leaky = FeatureSpec(name="x_load_real", kind="exog", offset_days=0,
                        source="load_real")
    with pytest.raises(ValueError, match="forbidden"):
        check_spec(leaky, SAMPLE)


# ------------------------------------------ the tests that give this teeth
def test_a_deliberately_leaky_feature_is_rejected():
    """
    A same-day price feature is the classic leak: it reads like an ordinary
    lag and it encodes the answer. The checker must refuse it.
    """
    leak = FeatureSpec(name="price_same_day", kind="price", offset_days=0,
                       source=cfg.TARGET)
    with pytest.raises(ValueError, match="gate closure"):
        check_spec(leak, SAMPLE)


def test_a_future_exog_feature_is_rejected():
    """Using tomorrow's published forecast to predict today is also a leak."""
    leak = FeatureSpec(name="x_fc_load_tomorrow", kind="exog", offset_days=-1,
                       source="fc_load")
    with pytest.raises(ValueError, match="gate closure"):
        check_spec(leak, SAMPLE)


def test_a_one_hour_price_lag_would_not_pass():
    """
    A naive lag_1h reads as conservative and is leakage, because for most
    delivery hours it points at the same delivery day. As a spec that is
    offset_days=0, and it must be refused.
    """
    leak = FeatureSpec(name="price_lag_1h", kind="price", offset_days=0,
                       source=cfg.TARGET)
    assert not leak.is_admissible_at(ts("2024-06-15 10:00"))


# ------------------------------------------------ end to end, if data present
@pytest.mark.skipif(not cfg.PANEL.exists(),
                    reason="panel not built, run: py -m dayahead.cli validate")
def test_built_matrix_contains_no_forbidden_columns():
    from dayahead.features.build import build_features
    X, _ = build_features()
    for forbidden in cfg.FORBIDDEN:
        assert not any(forbidden in c for c in X.columns)


@pytest.mark.skipif(not cfg.PANEL.exists(),
                    reason="panel not built, run: py -m dayahead.cli validate")
def test_built_matrix_target_is_not_a_feature():
    from dayahead.features.build import build_features
    X, specs = build_features()
    names = {s.name for s in specs}
    assert "y" not in names
    assert cfg.TARGET not in names
