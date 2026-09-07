"""
Calendar features.

Known arbitrarily far ahead, so every spec here is kind="calendar" and is
trivially admissible.

German public holidays are computed here rather than pulled from a package,
which keeps the project dependency free and makes the choice inspectable.

Scope of the holiday set, per DESIGN.md section 9 rule 6: nationwide holidays
only. Germany has additional holidays that vary by federal state, for example
Fronleichnam in NRW and Bavaria and Reformationstag in the northern and
eastern states. Those shift load in the affected states only, so their price
effect is partial and diluted. Modelling them properly would require
population weighting across 16 states. That is deliberately not done, and the
omission is recorded rather than hidden, because a state holiday that lowers
load in NRW will show up as unexplained error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as cfg
from .gate import FeatureSpec


def easter_sunday(year: int) -> pd.Timestamp:
    """Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month, day = divmod(h + m - 7 * n + 114, 31)
    return pd.Timestamp(year=year, month=month, day=day + 1)


def german_national_holidays(years) -> dict[pd.Timestamp, str]:
    """Nationwide German public holidays for the given years."""
    out: dict[pd.Timestamp, str] = {}
    for y in years:
        easter = easter_sunday(y)
        fixed = {
            pd.Timestamp(y, 1, 1): "Neujahr",
            pd.Timestamp(y, 5, 1): "Tag der Arbeit",
            pd.Timestamp(y, 10, 3): "Tag der Deutschen Einheit",
            pd.Timestamp(y, 12, 25): "1. Weihnachtstag",
            pd.Timestamp(y, 12, 26): "2. Weihnachtstag",
        }
        moving = {
            easter - pd.Timedelta(days=2): "Karfreitag",
            easter + pd.Timedelta(days=1): "Ostermontag",
            easter + pd.Timedelta(days=39): "Christi Himmelfahrt",
            easter + pd.Timedelta(days=50): "Pfingstmontag",
        }
        out.update(fixed)
        out.update(moving)
    return out


def build(index: pd.DatetimeIndex) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Calendar columns for a local-time delivery index."""
    idx = index.tz_convert(cfg.LOCAL_TZ)
    years = range(idx.year.min() - 1, idx.year.max() + 2)
    hol = german_national_holidays(years)
    hol_dates = set(hol)

    dates = pd.DatetimeIndex(idx.normalize().tz_localize(None))
    is_hol = dates.isin(hol_dates)
    is_bridge = (dates + pd.Timedelta(days=1)).isin(hol_dates) | \
                (dates - pd.Timedelta(days=1)).isin(hol_dates)

    doy = idx.dayofyear.to_numpy()
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()

    df = pd.DataFrame(index=index)
    df["cal_hour"] = hour
    df["cal_dow"] = dow
    df["cal_month"] = idx.month.to_numpy()
    df["cal_is_weekend"] = (dow >= 5).astype(int)
    df["cal_is_holiday"] = is_hol.astype(int)
    df["cal_is_bridge_day"] = (is_bridge & ~is_hol & (dow < 5)).astype(int)
    df["cal_is_nonworking"] = ((dow >= 5) | is_hol).astype(int)

    # Fourier terms carry the daily and annual cycles smoothly. The weekly
    # cycle is left to the dow dummy, which is only 7 levels.
    for k in (1, 2, 3):
        df[f"cal_sin_day_{k}"] = np.sin(2 * np.pi * k * hour / 24)
        df[f"cal_cos_day_{k}"] = np.cos(2 * np.pi * k * hour / 24)
    for k in (1, 2):
        df[f"cal_sin_year_{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        df[f"cal_cos_year_{k}"] = np.cos(2 * np.pi * k * doy / 365.25)

    specs = [FeatureSpec(name=c, kind="calendar",
                         note="known arbitrarily far ahead")
             for c in df.columns]
    return df, specs
