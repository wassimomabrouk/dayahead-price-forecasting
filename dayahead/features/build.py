"""
Feature matrix assembly.

Every spec is checked against the gate before its column is admitted. The
check runs inside the builder rather than only in the test suite, so a leaky
feature cannot reach a model even if nobody runs pytest.

The frozen feature set, per DESIGN.md section 9 rule 7, is whatever this
module produces at the commit that closes section 4. It is not revised in
response to backtest results.
"""

from __future__ import annotations

import pandas as pd

from .. import config as cfg
from ..data.validate import load_panel
from . import calendar as cal, exog, lags
from .gate import FeatureSpec, audit, check_spec, issuance_time

# Sample of delivery timestamps the gate check is evaluated on. Includes both
# DST transitions and a leap day so the arithmetic is exercised, not assumed.
# 02:00 does not exist on spring-forward days and is ambiguous on
# autumn-back days, so the hour after each transition is used instead.
CHECK_SAMPLE_LOCAL = [
    "2019-03-31 03:00", "2019-10-27 03:00", "2020-02-29 12:00",
    "2021-01-01 00:00", "2022-06-15 13:00", "2024-03-31 03:00",
    "2025-10-26 03:00", "2026-01-01 23:00",
]


def _check_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    wanted = pd.DatetimeIndex([pd.Timestamp(t, tz=cfg.LOCAL_TZ)
                               for t in CHECK_SAMPLE_LOCAL])
    inside = wanted[(wanted >= index[0]) & (wanted <= index[-1])]
    # Always include the endpoints, plus an evenly spaced spine.
    spine = index[:: max(1, len(index) // 200)]
    return inside.union(spine).union(index[[0, -1]])


def build_leaky_features(*, i_am_measuring_the_leak: bool = False,
                         panel: pd.DataFrame | None = None
                         ) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """
    A deliberately leaky feature matrix, for DESIGN.md section 13 check 1.

    Every published day-ahead forecast is replaced by the realised outturn:
    fc_load becomes load_real and fc_residual becomes residual_real. Wind and
    solar have no realised counterpart in the ingested set, so they are left
    as forecasts, which makes this a lower bound on the size of the leak
    rather than the full extent of it.

    This exists to put a number on the project's central claim. Section 2
    asserts that excluding realised outturn costs accuracy; this measures how
    much. It is not a model, it is an instrument, and nothing produced here
    is ever reported as a forecast.

    The gate closure check is deliberately bypassed, so the guard keyword is
    mandatory and the function is called from exactly one place.
    """
    if not i_am_measuring_the_leak:
        raise PermissionError(
            "build_leaky_features constructs a matrix that violates gate "
            "closure. It exists only for the section 13 leak quantification."
        )
    if panel is None:
        panel = load_panel()
    swapped = panel.copy()
    swapped["fc_load"] = panel["load_real"]
    swapped["fc_residual"] = panel["residual_real"]

    local = swapped.tz_convert(cfg.LOCAL_TZ)
    index = local.index
    price = local[cfg.TARGET].astype("float64")

    frames, specs = [], []
    for frame, spec in (cal.build(index), lags.build(price, index),
                        exog.build(local, index)):
        frames.append(frame)
        specs.extend(spec)

    X = pd.concat(frames, axis=1)
    X["y"] = price
    X["is_usable"] = local["is_usable"] & X.drop(
        columns=["y", "is_usable"], errors="ignore").notna().all(axis=1)
    return X, specs


def build_features(panel: pd.DataFrame | None = None,
                   verbose: bool = False) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """
    Returns the feature matrix on a local-time delivery index, plus the specs.

    Rows where the target or any required input is missing are kept in the
    index and flagged, matching the treatment in section 2. Dropping them
    would break the delivery-date arithmetic the price features rely on.
    """
    if panel is None:
        panel = load_panel()
    local = panel.tz_convert(cfg.LOCAL_TZ)
    index = local.index

    price = local[cfg.TARGET].astype("float64")

    frames, specs = [], []
    for name, (frame, spec) in {
        "calendar": cal.build(index),
        "price": lags.build(price, index),
        "exog": exog.build(local, index),
    }.items():
        frames.append(frame)
        specs.extend(spec)
        if verbose:
            print(f"    {name:<10}{frame.shape[1]:>4} columns")

    sample = _check_index(index)
    for spec in specs:
        check_spec(spec, sample)

    X = pd.concat(frames, axis=1)
    if len(X.columns) != len(set(X.columns)):
        dupes = X.columns[X.columns.duplicated()].tolist()
        raise ValueError(f"duplicate feature names: {dupes}")

    forbidden_hits = [c for c in X.columns
                      if any(f in c for f in cfg.FORBIDDEN)]
    if forbidden_hits:
        raise ValueError(f"forbidden series in column names: {forbidden_hits}")

    X["y"] = price
    X["is_usable"] = local["is_usable"] & X.drop(columns=["y", "is_usable"],
                                                 errors="ignore").notna().all(axis=1)
    return X, specs


def feature_names(specs: list[FeatureSpec]) -> list[str]:
    return [s.name for s in specs]


def build_audit(index: pd.DatetimeIndex,
                specs: list[FeatureSpec]) -> pd.DataFrame:
    return audit(specs, _check_index(index))
