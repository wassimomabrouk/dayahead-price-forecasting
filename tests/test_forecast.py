"""
The daily forecast run and its track record.

The track record is the one artifact in the repository that cannot be
reconstructed after the fact, so the rules protecting it are tested.
"""

import numpy as np
import pandas as pd
import pytest

from dayahead import config as cfg
from dayahead.forecast.daily import (
    LOG_COLUMNS, append_forecast, build_calibration, next_delivery_day,
    score_log, track_record_summary,
)


def _panel(last_price_hour: str, last_exog_hour: str) -> pd.DataFrame:
    idx = pd.date_range("2026-09-01", last_exog_hour, freq="h", tz="UTC")
    df = pd.DataFrame({c: 1000.0 for c in cfg.ALL_SERIES}, index=idx)
    df[cfg.TARGET] = 50.0
    df.loc[df.index > pd.Timestamp(last_price_hour, tz="UTC"), cfg.TARGET] = np.nan
    df["is_usable"] = True
    return df


def test_next_delivery_day_is_the_first_with_exog_and_no_price():
    """
    The asymmetry that makes forecasting possible: tomorrow's drivers are
    published, tomorrow's price is not.
    """
    panel = _panel("2026-09-10 21:00", "2026-09-11 21:00")
    day = next_delivery_day(panel)
    assert day is not None
    assert day.strftime("%Y-%m-%d") == "2026-09-11"


def test_no_delivery_day_when_prices_are_current():
    """Most of the day there is nothing new to forecast. Not an error."""
    panel = _panel("2026-09-11 21:00", "2026-09-11 21:00")
    assert next_delivery_day(panel) is None


def test_a_partial_day_of_exog_is_not_forecast():
    panel = _panel("2026-09-10 21:00", "2026-09-11 05:00")
    assert next_delivery_day(panel) is None


# ------------------------------------------------------------ track record
def _forecast_rows(day: str, pred: float = 50.0) -> pd.DataFrame:
    hours = pd.date_range(f"{day} 00:00", periods=24, freq="h",
                          tz=cfg.LOCAL_TZ)
    df = pd.DataFrame({
        "issued_utc": "2026-09-10T09:00:00+00:00",
        "delivery_hour_local": hours,
        "delivery_date_local": hours.normalize(),
        "hour": hours.hour,
        "model": "N-HiTS",
        "y_pred": pred,
    })
    for q in range(1, 10):
        df[f"q{q}0"] = pred + (q - 5) * 5.0
    df["y_true"] = np.nan
    df["abs_error"] = np.nan
    return df[LOG_COLUMNS]


def test_a_forecast_is_never_silently_replaced():
    """
    A rerun that overwrote an earlier forecast would let a bad day be quietly
    reissued. The log is append only for a reason.
    """
    first = _forecast_rows("2026-09-11", pred=50.0)
    log = append_forecast(first.copy())
    assert len(log) == 24

    # Same delivery hours, different numbers.
    second = _forecast_rows("2026-09-11", pred=999.0)
    log2 = append_forecast(second)
    assert len(log2) == len(log)


def test_scoring_fills_outturn_once_it_exists():
    log = _forecast_rows("2026-09-11", pred=45.0)
    idx = pd.date_range("2026-09-11", periods=24, freq="h", tz=cfg.LOCAL_TZ)
    panel = pd.DataFrame({cfg.TARGET: 60.0}, index=idx.tz_convert("UTC"))

    scored = score_log(log, panel)
    assert scored["y_true"].notna().all()
    assert scored["abs_error"].iloc[0] == pytest.approx(15.0)


def test_summary_reports_coverage_against_the_stated_nominal():
    log = _forecast_rows("2026-09-11", pred=50.0)
    log["y_true"] = 50.0     # inside q10..q90, which span 30 to 70
    log["abs_error"] = 0.0
    s = track_record_summary(log)
    assert s["scored_hours"] == 24
    assert s["coverage"] == pytest.approx(1.0)
    assert s["mae"] == pytest.approx(0.0)


def test_calibration_uses_only_recent_history():
    """
    A window fitted over eight years would correct today's forecast with
    residuals from a market structure that no longer exists.
    """
    n = 24 * 800
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    preds = pd.DataFrame({
        "timestamp": idx, "model": "N-HiTS", "y_true": 50.0,
        **{f"q{q}0": 50.0 for q in range(1, 10)},
    })
    cal = build_calibration(preds, "N-HiTS", window_days=365)
    assert cal["n_rows"] < n
    assert cal["window_days"] == 365


# ------------------------------------------- ingestion freshness, section 14
def test_recent_blocks_are_always_refetched():
    """
    Regression test for a bug that made the daily job impossible.

    SMARD keeps filling the current week's block as the week progresses. A
    cache rule that skips anything already on disk therefore freezes the
    store at whatever the block contained when it was first downloaded. The
    ingest summary still reports every block present, so nothing looks wrong
    until a forecast is requested and there is no new data to forecast from.
    """
    from dayahead.data.smard import ALWAYS_REFETCH_BLOCKS

    assert ALWAYS_REFETCH_BLOCKS >= 1

    import inspect

    from dayahead.data import smard

    src = inspect.getsource(smard.ingest_series)
    assert "ALWAYS_REFETCH_BLOCKS" in src, (
        "ingest_series must refetch the trailing blocks, not trust the cache"
    )
    assert "t in stale" in src
