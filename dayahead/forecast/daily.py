"""
The daily forecast run.

Produces tomorrow's 24 hourly prices with calibrated intervals, appends them
to a track record, and scores earlier forecasts once their outturn is
published.

Which feature set
-----------------

The reduced one, from section 16. Section 27 established that four of the six
exogenous inputs are published by SMARD after the auction they would have
informed, so a live system cannot use them however good they are in a
backtest. This module uses price history, calendar terms and the day-ahead
load forecast, which costs 8.8% on MAE against the full set and is the
difference between a system that runs and one that does not.

Why the untrimmed panel
-----------------------

`validate` trims the panel to the point where every series ends, which is
correct for evaluation and wrong here. Forecasting tomorrow depends on
exactly the asymmetry the trim removes: SMARD publishes tomorrow's load,
wind and solar forecasts, and does not publish tomorrow's price, because the
auction that sets it has not cleared. So this module reads panel_full and
selects the delivery day where the exogenous inputs are complete and the
target is absent.

That asymmetry is the whole forecasting problem, and it only becomes visible
when the system runs forward rather than over history.

The track record
----------------

Every run appends its forecast to reports/forecast_log.csv before the outturn
exists, and fills in the realised price on a later run. The file is committed,
so the history of what was predicted, and when, is not something the author
can revise afterwards. That is the point of keeping it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .. import config as cfg
from ..evaluation.splits import QUANTILES

LOG_PATH = cfg.REPORTS / "forecast_log.csv"
CALIBRATION_PATH = cfg.REPORTS / "calibration.json"

LOG_COLUMNS = [
    "issued_utc", "delivery_hour_local", "delivery_date_local", "hour",
    "model", "y_pred", *[f"q{int(q * 100):02d}" for q in QUANTILES],
    "y_true", "abs_error",
]


def _qcol(q: float) -> str:
    return f"q{int(q * 100):02d}"


# --------------------------------------------------------------- calibration
def build_calibration(preds: pd.DataFrame, model: str,
                      window_days: int = 365) -> dict:
    """
    Conformal corrections from the model's own recent out-of-sample errors.

    Same construction as section 10: for level q the correction is the q-th
    quantile of y - qhat_q. Fitted on the most recent window of stored
    predictions, all of which precede any forecast this file will produce.
    """
    g = preds[preds["model"] == model].copy()
    g["timestamp"] = pd.to_datetime(g["timestamp"], utc=True)
    cutoff = g["timestamp"].max() - pd.Timedelta(days=window_days)
    g = g[g["timestamp"] >= cutoff]

    deltas = {}
    y = g["y_true"].to_numpy(dtype=float)
    for q in QUANTILES:
        c = _qcol(q)
        if c not in g:
            deltas[str(q)] = 0.0
            continue
        r = y - g[c].to_numpy(dtype=float)
        r = r[np.isfinite(r)]
        deltas[str(q)] = float(np.quantile(r, q)) if len(r) else 0.0

    return {
        "model": model,
        "fitted_utc": datetime.now(timezone.utc).isoformat(),
        "window_days": window_days,
        "n_rows": int(len(g)),
        "deltas": deltas,
    }


def load_calibration() -> dict | None:
    if not CALIBRATION_PATH.exists():
        return None
    with CALIBRATION_PATH.open(encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------- the next day
# The exogenous series a live system can actually obtain before gate closure.
# Section 27. Requiring all six here would make the job refuse every day,
# because four of them arrive after the auction.
OPERATIONAL_EXOG = ["fc_load"]


def next_delivery_day(panel_full: pd.DataFrame,
                      required_exog: list[str] | None = None,
                      now: pd.Timestamp | None = None) -> pd.Timestamp | None:
    """
    The earliest *forecastable* local delivery date: inputs present, price
    absent, and the auction that sets it still ahead.

    Returns None when there is nothing to forecast, which is the normal state
    for most of the day and is not an error.

    Why the auction check exists
    ---------------------------

    An earlier version asked only for inputs present and price absent. That
    is wrong whenever the price series has a hole rather than a ragged end,
    and SMARD produced exactly that on 2026-09-13: it published the 14th and
    skipped the 13th. The job then fixed on the 13th, a day whose auction had
    closed two days earlier, and would have stayed there until the gap filled,
    missing every forecastable day in between.

    A missing price in the past is a data gap. A missing price in the future
    is the thing being forecast. Only the second is an opportunity, and the
    auction time is what separates them.
    """
    local = panel_full.tz_convert(cfg.LOCAL_TZ)
    cols = required_exog if required_exog is not None else OPERATIONAL_EXOG
    exog_ok = local[cols].notna().all(axis=1)
    price_missing = local[cfg.TARGET].isna()

    by_day = pd.DataFrame({
        "exog": exog_ok.groupby(local.index.normalize()).all(),
        "no_price": price_missing.groupby(local.index.normalize()).all(),
        "hours": exog_ok.groupby(local.index.normalize()).size(),
    })
    candidates = by_day[by_day["exog"] & by_day["no_price"]
                        & (by_day["hours"] >= 23)]
    if candidates.empty:
        return None

    # A delivery day is forecastable only while its auction is still ahead.
    # The auction for day D clears at gate closure on D-1.
    now = now if now is not None else pd.Timestamp.now(tz=cfg.LOCAL_TZ)
    still_open = [
        day for day in candidates.index
        if (day - pd.Timedelta(days=1)
            + pd.Timedelta(hours=cfg.GATE_CLOSURE_HOUR_LOCAL)) > now
    ]
    return min(still_open) if still_open else None


def produce_forecast(model_name: str = "N-HiTS",
                     verbose: bool = True) -> pd.DataFrame | None:
    """Fit on all published history, forecast the next delivery day."""
    from ..data.validate import load_panel
    from ..features.build import build_reduced_features
    from ..models.nhits import NHiTSForecaster

    panel_full = load_panel(full=True)
    target_day = next_delivery_day(panel_full)
    if target_day is None:
        if verbose:
            print("  no delivery day with complete inputs and no price yet")
        return None

    day_end = target_day + pd.Timedelta(hours=23)

    # Check the log before fitting, not after. append_forecast already
    # refuses to overwrite an existing forecast, but by then the model has
    # been refitted for nothing. SMARD's price publication lags over
    # weekends, so the same delivery day can remain the target for two or
    # three consecutive runs.
    existing = load_log()
    if len(existing):
        already = set(pd.to_datetime(existing["delivery_hour_local"],
                                     utc=True, errors="coerce")
                      .dt.tz_convert(cfg.LOCAL_TZ).dt.normalize())
        if target_day in already:
            if verbose:
                print(f"  {target_day:%Y-%m-%d} already forecast, nothing to do")
            return None

    if verbose:
        print(f"  delivery day: {target_day:%Y-%m-%d}")

    # Features are built on the untrimmed panel, cut at the end of the target
    # day so nothing later can enter.
    #
    # panel_full is written before the trim, so it carries no is_usable
    # column: that flag is added by mark_usable during validation. The
    # feature builder expects it, so it is recomputed here on the columns
    # this model actually consumes. Copying the trimmed panel's flag would be
    # wrong, since it was computed against the full exogenous set and would
    # mark the target day unusable for missing wind and solar that this model
    # does not use.
    usable = panel_full.tz_convert(cfg.LOCAL_TZ).loc[:day_end].copy()
    required = [cfg.TARGET] + OPERATIONAL_EXOG
    usable["is_usable"] = usable[required].notna().all(axis=1)
    X, _ = build_reduced_features(panel=usable.tz_convert("UTC"))

    # Filtered on the target alone, not on is_usable.
    #
    # is_usable requires every feature to be present, including the price lag
    # columns. A single unpublished delivery day therefore removes two days
    # from training: the day itself, which has no price, and the day after,
    # whose price_d1_same_hour reads from it. The conditioning history then
    # stops two days short and the model can only forecast a day already in
    # the past.
    #
    # The sequence model does not consume the lag columns. It needs the price
    # series and its own exogenous inputs, and the frame builder drops any row
    # missing those. So filtering on the target here is both sufficient and
    # two days less destructive.
    train = X[X["y"].notna()]
    future = X.loc[target_day:day_end]
    if future.empty:
        if verbose:
            print("  target day produced no feature rows")
        return None

    cols = [c for c in X.columns if c not in ("y", "is_usable")]
    if verbose:
        print(f"  training rows: {len(train):,}   forecast hours: {len(future)}")

    # Same architecture as the champion, exogenous list restricted to what is
    # present in the reduced matrix.
    model = NHiTSForecaster(
        name=model_name,
        futr_exog=[c for c in ("x_fc_load", "cal_is_nonworking",
                               "cal_is_holiday") if c in cols],
    )
    model.fit(train[cols], train["y"])
    quantiles = model.predict_quantiles(future[cols])

    out = pd.DataFrame({
        "issued_utc": datetime.now(timezone.utc).isoformat(),
        "delivery_hour_local": future.index,
        "delivery_date_local": future.index.normalize(),
        "hour": future.index.hour,
        "model": model_name,
    })
    for q in QUANTILES:
        out[_qcol(q)] = quantiles[q]
    out["y_pred"] = out[_qcol(0.5)]

    cal = load_calibration()
    if cal and cal.get("model") == model_name:
        for q in QUANTILES:
            out[_qcol(q)] = out[_qcol(q)] + cal["deltas"].get(str(q), 0.0)
        out["y_pred"] = out[_qcol(0.5)]
        if verbose:
            print(f"  calibration applied, fitted {cal['fitted_utc'][:10]} "
                  f"on {cal['n_rows']:,} rows")
    elif verbose:
        print("  WARNING: no calibration file, intervals are uncalibrated")

    # Independent per-quantile shifts can reorder the quantiles.
    qcols = [_qcol(q) for q in QUANTILES]
    block = out[qcols].to_numpy(dtype=float)
    block.sort(axis=1)
    out[qcols] = block

    out["y_true"] = np.nan
    out["abs_error"] = np.nan
    return out[LOG_COLUMNS]


# ------------------------------------------------------------ track record
def load_log() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(LOG_PATH)


def append_forecast(new: pd.DataFrame) -> pd.DataFrame:
    """
    Add a forecast, refusing to overwrite one already made for those hours.

    A rerun that silently replaced an earlier forecast would let a bad day be
    quietly reissued, which is exactly what a track record exists to prevent.
    """
    log = load_log()
    if len(log):
        already = set(log["delivery_hour_local"].astype(str))
        new = new[~new["delivery_hour_local"].astype(str).isin(already)]
    if new.empty:
        return log
    return pd.concat([log, new], ignore_index=True)


def score_log(log: pd.DataFrame, panel_full: pd.DataFrame) -> pd.DataFrame:
    """Fill in realised prices for forecasts whose outturn has since cleared."""
    if log.empty:
        return log
    local = panel_full.tz_convert(cfg.LOCAL_TZ)
    truth = local[cfg.TARGET].astype("float64")
    truth.index = truth.index.astype(str)

    log = log.copy()
    missing = log["y_true"].isna()
    filled = log.loc[missing, "delivery_hour_local"].astype(str).map(truth)
    log.loc[missing, "y_true"] = filled.to_numpy()
    log["abs_error"] = (log["y_true"] - log["y_pred"]).abs()
    return log


def track_record_summary(log: pd.DataFrame) -> dict:
    scored = log[log["y_true"].notna()]
    if scored.empty:
        return {"scored_hours": 0}

    lo, hi = _qcol(0.1), _qcol(0.9)
    covered = ((scored["y_true"] >= scored[lo])
               & (scored["y_true"] <= scored[hi])).mean()
    return {
        "forecast_hours": int(len(log)),
        "scored_hours": int(len(scored)),
        "pending_hours": int(log["y_true"].isna().sum()),
        "mae": float(scored["abs_error"].mean()),
        "coverage": float(covered),
        "first_delivery": str(log["delivery_hour_local"].min()),
        "last_delivery": str(log["delivery_hour_local"].max()),
    }
