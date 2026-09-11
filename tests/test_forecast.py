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


def test_a_forecast_is_never_silently_replaced(tmp_path, monkeypatch):
    """
    A rerun that overwrote an earlier forecast would let a bad day be quietly
    reissued. The log is append only for a reason.

    Pointed at a temporary file: an earlier version read the real log, so it
    passed or failed depending on what happened to be on disk. A test whose
    result depends on the developer's working directory is not a test.
    """
    import dayahead.forecast.daily as daily
    monkeypatch.setattr(daily, "LOG_PATH", tmp_path / "log.csv")

    first = _forecast_rows("2026-09-11", pred=50.0)
    log = daily.append_forecast(first.copy())
    assert len(log) == 24
    log.to_csv(daily.LOG_PATH, index=False)

    # Same delivery hours, different numbers.
    second = _forecast_rows("2026-09-11", pred=999.0)
    log2 = daily.append_forecast(second)
    assert len(log2) == 24
    assert (log2["y_pred"] == 50.0).all(), "the original forecast was replaced"


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


# ------------------------------------------------- published page, section 15
def test_page_renders_with_an_empty_log():
    """
    The page must exist before the first forecast does, otherwise the
    scheduled job has nothing to publish on its first run.
    """
    from dayahead.forecast.page import build_page

    html = build_page()
    assert "<!DOCTYPE html>" in html
    assert "Page generated" in html


def test_page_has_no_external_dependencies():
    """
    A hosted dashboard sleeps and a CDN link rots. A single file with inline
    SVG renders identically in a year whether or not anything is running.
    """
    import re

    from dayahead.forecast.page import build_page

    html = build_page()
    assert not re.search(r'(src|href)="https?://', html), (
        "the page must not fetch anything at load time"
    )
    assert "<script" not in html.lower()


def test_page_states_when_it_was_generated():
    """A stale page should be visibly stale rather than silently wrong."""
    from dayahead.forecast.page import build_page

    html = build_page()
    assert "UTC" in html
    assert "the daily job has not run" in html


# ------------------------------------- operationally available set, section 16
def test_reduced_set_keeps_only_obtainable_exogenous_columns():
    """
    Section 27 established which exogenous inputs SMARD publishes before the
    auction. Everything else has to go, including fc_residual, which derives
    from load but also from wind and solar and therefore inherits their
    unavailability.
    """
    from dayahead.features.build import reduced_columns

    cols = ["cal_hour", "cal_sin_day_1", "price_d1_same_hour",
            "price_d7_same_hour", "price_d1_mean",
            "x_fc_load", "x_fc_residual", "x_fc_solar", "x_fc_wind_on",
            "x_wind_total", "x_renewable_share", "x_residual_day_max",
            "y", "is_usable"]
    keep = reduced_columns(pd.DataFrame(columns=cols))

    assert "x_fc_load" in keep
    assert "x_fc_residual" not in keep, (
        "fc_residual is load minus wind minus solar and inherits their "
        "publication delay"
    )
    for banned in ("x_fc_solar", "x_fc_wind_on", "x_wind_total",
                   "x_renewable_share", "x_residual_day_max"):
        assert banned not in keep

    # Price history and calendar are unconditionally available.
    for kept in ("cal_hour", "cal_sin_day_1", "price_d1_same_hour",
                 "price_d7_same_hour", "price_d1_mean"):
        assert kept in keep


def test_reduced_set_excludes_the_target_and_the_flag():
    from dayahead.features.build import reduced_columns

    keep = reduced_columns(pd.DataFrame(columns=["cal_hour", "y", "is_usable"]))
    assert "y" not in keep
    assert "is_usable" not in keep


def test_variant_names_map_back_to_their_base_model():
    """
    Regression test. A variant whose suffix is unknown calibrates from the
    test folds alone and loses the first six to the minimum-history rule,
    reporting metrics on half the year with nothing but a smaller n to show
    for it.
    """
    from dayahead.evaluation.locked_test import base_model_name

    assert base_model_name("N-HiTS_operational") == "N-HiTS"
    assert base_model_name("N-HiTS_LEAKY") == "N-HiTS"
    assert base_model_name("N-HiTS_post2023") == "N-HiTS"
    assert base_model_name("N-HiTS") == "N-HiTS"
    assert base_model_name("B2_daily_naive") == "B2_daily_naive"


def test_a_variant_without_history_raises_rather_than_silently_shrinking():
    import pandas as pd

    from dayahead.evaluation.locked_test import calibrate_test

    test = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=48, freq="h",
                                   tz="UTC"),
        "model": "N-HiTS_unknownvariant", "fold": "2026-01",
        "regime": "post-crisis", "y_true": 50.0, "y_pred": 50.0,
        **{f"q{q}0": 50.0 for q in range(1, 10)},
    })
    with pytest.raises(ValueError, match="no backtest history"):
        calibrate_test(test, pd.DataFrame(columns=test.columns))
