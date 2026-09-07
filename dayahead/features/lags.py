"""
Price history features.

Indexed by delivery date offset, never by uniform hour lag. See the module
docstring in gate.py for why: at issuance the whole 24-hour price vector for
D-1 is known, and a uniform lag cannot express that without discarding
information for early delivery hours.

Every feature here comes from delivery date D-n with n >= 1. The D-0 vector
is the target and is never an input.

Calendar arithmetic, not elapsed time
-------------------------------------

Lookups go through a (local date, hour) key and step back with DateOffset,
which is calendar aware. An earlier version used shift(freq=Timedelta(days=1)),
a fixed 24-hour offset, and it silently produced an all-NaN feature row for
every day following a DST transition: 16 transitions times 24 hours of the
sample lost. A 23-hour or 25-hour day is not 24 hours of elapsed time, so a
fixed offset lands between grid points and reindexes to nothing.

Two hours a year remain genuinely unavailable and are not worked around: the
hour skipped at spring forward has no counterpart on the following day, and
the hour repeated at autumn back collapses to a single key. Both are left
missing rather than filled.
"""

from __future__ import annotations

import pandas as pd

from .. import config as cfg
from .gate import FeatureSpec

# Which past delivery days contribute a same-hour price.
#   1  yesterday, the freshest admissible auction
#   2  short persistence beyond one day
#   7  same weekday, the weekly cycle the ACF showed at lag 168
PRICE_DAY_OFFSETS = (1, 2, 7)

# Hours of the D-1 vector taken directly, regardless of target hour. The
# overnight trough and the two daily peaks summarise the shape of the
# previous auction without adding 24 columns.
D1_PROFILE_HOURS = (3, 8, 12, 19)

WEEK_MEAN_DAYS = 7


def _date_hour_lookup(s: pd.Series) -> pd.Series:
    """
    Reindex a local-time series by (calendar date, hour).

    The repeated hour on autumn-back days gives two rows the same key. The
    first is kept, which is the CEST occurrence.
    """
    key = pd.MultiIndex.from_arrays(
        [s.index.normalize(), s.index.hour], names=["date", "hour"])
    out = pd.Series(s.to_numpy(), index=key)
    return out[~out.index.duplicated(keep="first")]


def build(price_local: pd.Series,
          index: pd.DatetimeIndex) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """
    price_local  full local-time price series, used as the source
    index        delivery timestamps to build features for
    """
    s = price_local.astype("float64")
    lookup = _date_hour_lookup(s)

    dates = index.normalize()
    hours = index.hour

    df = pd.DataFrame(index=index)
    specs: list[FeatureSpec] = []

    def add(name: str, values, offset_days: int, note: str = ""):
        df[name] = values
        specs.append(FeatureSpec(name=name, kind="price", offset_days=offset_days,
                                 source=cfg.TARGET, note=note))

    # Same hour, n calendar days back.
    for n in PRICE_DAY_OFFSETS:
        want = pd.MultiIndex.from_arrays([dates - pd.DateOffset(days=n), hours])
        add(f"price_d{n}_same_hour", lookup.reindex(want).to_numpy(), n,
            f"price at this hour on D-{n}")

    # Daily aggregates of the D-1 auction.
    by_date = s.groupby(s.index.normalize())
    prev_dates = dates - pd.DateOffset(days=1)
    for label in ("mean", "min", "max", "std"):
        agg = getattr(by_date, label)()
        add(f"price_d1_{label}", agg.reindex(prev_dates).to_numpy(), 1,
            f"{label} price across all hours of D-1")

    df["price_d1_spread"] = df["price_d1_max"] - df["price_d1_min"]
    specs.append(FeatureSpec(name="price_d1_spread", kind="price", offset_days=1,
                             source=cfg.TARGET, note="D-1 max minus min"))

    # Selected hours of the D-1 profile, identical for every target hour.
    for h in D1_PROFILE_HOURS:
        want = pd.MultiIndex.from_arrays([prev_dates, pd.Index([h] * len(index))])
        add(f"price_d1_h{h:02d}", lookup.reindex(want).to_numpy(), 1,
            f"price at hour {h} on D-1")

    # Weekly level. Built from daily means so the window is 7 calendar days,
    # not 168 hours of elapsed time.
    daily_mean = by_date.mean()
    week_mean = daily_mean.rolling(WEEK_MEAN_DAYS, min_periods=WEEK_MEAN_DAYS).mean()
    add("price_d1_week_mean", week_mean.reindex(prev_dates).to_numpy(), 1,
        f"{WEEK_MEAN_DAYS} day mean of daily mean price, ending at D-1")

    return df, specs
