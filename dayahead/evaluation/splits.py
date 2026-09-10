"""
Split protocol.

Implements DESIGN.md sections 7 and 8. The numbers here are pre-committed and
are not tuned: changing them after results exist would void the point of
having written them down first.

The locked test set is guarded rather than merely documented. `backtest_frame`
is the only function model development calls, and it physically cannot return
test rows. Reaching the test set requires `locked_test_frame(i_am_opening_the_
locked_test_set=True)`, which is deliberately awkward to type by accident.
"""

from __future__ import annotations

import pandas as pd

from .. import config as cfg

# ------------------------------------------------------------------ windows
BACKTEST_START = "2018-10-01"
BACKTEST_END = "2025-08-31 23:00"
TEST_START = "2025-09-01"
TEST_END = "2026-08-31 23:00"

# Fold structure. First training set ends 2020-09-30, giving 24 months of
# history, so the first evaluated month is 2020-10.
FIRST_EVAL_MONTH = "2020-10"
LAST_EVAL_MONTH = "2025-08"
MIN_TRAIN_MONTHS = 24

# Regime boundaries, anchored to market history. See DESIGN.md section 8.
REGIMES = [
    ("pre-crisis", "2018-10-01", "2021-08-31 23:00"),
    ("crisis", "2021-09-01", "2022-12-31 23:00"),
    ("post-crisis", "2023-01-01", "2099-12-31 23:00"),
]

# Nine quantiles, per DESIGN.md section 6.
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
INTERVAL_LOW, INTERVAL_HIGH = 0.1, 0.9
NOMINAL_COVERAGE = INTERVAL_HIGH - INTERVAL_LOW


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=cfg.LOCAL_TZ)


def backtest_frame(X: pd.DataFrame) -> pd.DataFrame:
    """
    Rows available for model development. Cannot return locked test rows.
    """
    return X.loc[_ts(BACKTEST_START):_ts(BACKTEST_END)]


def locked_test_frame(X: pd.DataFrame, *,
                      i_am_opening_the_locked_test_set: bool = False) -> pd.DataFrame:
    """
    The locked test set. Opened once, at section 12, after the champion is
    fixed. The keyword argument exists to make an accidental call impossible
    to write without noticing.
    """
    if not i_am_opening_the_locked_test_set:
        raise PermissionError(
            "The locked test set is not available during model development. "
            "See DESIGN.md section 7. If this really is section 12, pass "
            "i_am_opening_the_locked_test_set=True."
        )
    return X.loc[_ts(TEST_START):_ts(TEST_END)]


def eval_months() -> pd.PeriodIndex:
    return pd.period_range(FIRST_EVAL_MONTH, LAST_EVAL_MONTH, freq="M")


def folds(X: pd.DataFrame) -> list[dict]:
    """
    Expanding window, monthly origins.

    Fold m trains on everything up to the last hour of month m-1 and forecasts
    every delivery hour of month m. No embargo: the training set ends at the
    last delivery hour of m-1, and the first forecast of month m is issued at
    noon on the last day of m-1, by which time those prices have cleared.
    """
    bt = backtest_frame(X)
    out = []
    for month in eval_months():
        test_start = _ts(str(month.start_time.date()))
        test_end = _ts(str(month.end_time.date())) + pd.Timedelta(hours=23)
        train_end = test_start - pd.Timedelta(hours=1)

        train_idx = bt.index[bt.index <= train_end]
        test_idx = bt.index[(bt.index >= test_start) & (bt.index <= test_end)]
        if len(test_idx) == 0 or len(train_idx) < MIN_TRAIN_MONTHS * 28 * 24:
            continue
        out.append({
            "month": str(month),
            "train_index": train_idx,
            "test_index": test_idx,
            "train_end": train_end,
            "regime": regime_of(test_start),
        })
    return out


def test_folds(X: pd.DataFrame, *,
               i_am_opening_the_locked_test_set: bool = False) -> list[dict]:
    """
    Monthly folds across the locked test window.

    Same protocol as the backtest: fold m trains on everything up to the last
    hour of month m-1, which here includes the entire backtest window, and
    forecasts every delivery hour of month m. Using a different protocol for
    the final evaluation than for model development would make the two
    incomparable.

    Guarded exactly as locked_test_frame is. Section 12 is the only caller.
    """
    if not i_am_opening_the_locked_test_set:
        raise PermissionError(
            "test_folds reads the locked test set. See DESIGN.md section 7."
        )
    months = pd.period_range(TEST_START, TEST_END, freq="M")
    out = []
    for month in months:
        test_start = _ts(str(month.start_time.date()))
        test_end = _ts(str(month.end_time.date())) + pd.Timedelta(hours=23)
        train_end = test_start - pd.Timedelta(hours=1)

        train_idx = X.index[X.index <= train_end]
        test_idx = X.index[(X.index >= test_start) & (X.index <= test_end)]
        if len(test_idx) == 0:
            continue
        out.append({
            "month": str(month),
            "train_index": train_idx,
            "test_index": test_idx,
            "train_end": train_end,
            "regime": regime_of(test_start),
        })
    return out


def regime_of(ts: pd.Timestamp) -> str:
    for name, start, end in REGIMES:
        if _ts(start) <= ts <= _ts(end):
            return name
    return "unassigned"


def add_regime(index: pd.DatetimeIndex) -> pd.Series:
    out = pd.Series("unassigned", index=index, dtype="object")
    for name, start, end in REGIMES:
        mask = (index >= _ts(start)) & (index <= _ts(end))
        out[mask] = name
    return out
