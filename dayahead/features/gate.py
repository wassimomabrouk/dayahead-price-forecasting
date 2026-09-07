"""
The gate closure filter.

This module is the project's central claim expressed as code. Every feature
passes through it and every feature carries an availability timestamp
assigned here.

SMARD publishes a delivery timestamp, never a publication timestamp, so
availability cannot be read off the data. It is assigned from the market
rules below. See DESIGN.md section 2a for what that assignment rests on.

Timeline for delivery day D
---------------------------

    D-2 12:00   auction for D-1 clears
    D-2 12:45   prices for all 24 hours of D-1 published
    D-1 10:00   day-ahead load forecast for D published (regulatory deadline
                is two hours before gate closure)
    D-1 12:00   ISSUANCE. Our forecast for all 24 hours of D is made here.
    D-1 12:45   auction for D clears, prices for D published. Too late.

Why price features are indexed by delivery date, not by hour lag
----------------------------------------------------------------

A uniform "lag h hours" framing is the natural first instinct and it is
clumsy here. At issuance the entire 24-hour price vector for D-1 is known,
because it cleared a full day earlier. Relative to a target hour that vector
sits anywhere from 1 to 24 hours back, so no single hour lag describes it.

Taking the smallest always-safe uniform lag, 24 hours, is correct but throws
away information for early delivery hours. Indexing by delivery date instead
keeps everything that was genuinely known and remains exactly as safe,
because admissibility depends on which auction published a value, not on how
many hours separate it from the target.

So price features are "hour k of day D-n" for n >= 1, and the D-0 vector is
the target, never an input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from .. import config as cfg

# Auction results are published shortly after the noon clearing. The exact
# minute does not matter; what matters is that it is strictly after issuance,
# so that day D's own prices are inadmissible when forecasting day D.
RESULTS_PUBLISHED_MINUTE = 45

# Availability stamp for features known arbitrarily far ahead.
CALENDAR_SENTINEL = pd.Timestamp("1900-01-01", tz=cfg.LOCAL_TZ)

Kind = Literal["calendar", "exog", "price"]


# ------------------------------------------------------------------- timing
def delivery_date(ts: pd.Timestamp) -> pd.Timestamp:
    """Local calendar date of a delivery timestamp, at local midnight."""
    return ts.tz_convert(cfg.LOCAL_TZ).normalize()


def issuance_time(ts: pd.Timestamp) -> pd.Timestamp:
    """
    When the forecast for this delivery timestamp is made.

    Gate closure, 12:00 local, on the day before the delivery date. Every
    delivery hour of a given day shares one issuance time, because the
    forecast is a single 24-value decision.
    """
    d = delivery_date(ts)
    return (d - pd.Timedelta(days=1)) + pd.Timedelta(hours=cfg.GATE_CLOSURE_HOUR_LOCAL)


def gate_closure_of(date: pd.Timestamp) -> pd.Timestamp:
    """Gate closure instant at which the auction for `date` clears."""
    return (date - pd.Timedelta(days=1)) + pd.Timedelta(hours=cfg.GATE_CLOSURE_HOUR_LOCAL)


def availability_time(ts: pd.Timestamp, kind: Kind, offset_days: int) -> pd.Timestamp:
    """
    When a source value became known.

    ts          the delivery timestamp being forecast
    kind        which publication rule applies
    offset_days how many delivery days back the source value sits

    calendar  known arbitrarily far ahead
    exog      day-ahead forecast for delivery date d, treated as available at
              that date's gate closure, subject to DESIGN.md section 2a
    price     cleared at the auction for delivery date d and published shortly
              after, so strictly later than that date's gate closure
    """
    if kind == "calendar":
        # A sentinel far before the sample. Not Timestamp.min: the gap to the
        # present would exceed the maximum representable Timedelta, so the
        # audit's slack subtraction would overflow.
        return CALENDAR_SENTINEL

    source_date = delivery_date(ts) - pd.Timedelta(days=offset_days)
    gate = gate_closure_of(source_date)
    if kind == "exog":
        return gate
    if kind == "price":
        return gate + pd.Timedelta(minutes=RESULTS_PUBLISHED_MINUTE)
    raise ValueError(f"unknown kind: {kind}")


# ------------------------------------------------------------------- specs
@dataclass(frozen=True)
class FeatureSpec:
    """
    One feature column and the publication rule that governs it.

    The spec is what the gate closure tests inspect. A feature that cannot
    state its kind and offset cannot enter the matrix.
    """
    name: str
    kind: Kind
    offset_days: int = 0
    source: str | None = None
    note: str = ""

    def availability(self, ts: pd.Timestamp) -> pd.Timestamp:
        return availability_time(ts, self.kind, self.offset_days)

    def is_admissible_at(self, ts: pd.Timestamp) -> bool:
        return self.availability(ts) <= issuance_time(ts)


def check_spec(spec: FeatureSpec, sample: pd.DatetimeIndex) -> None:
    """
    Raise if a spec would ever use information published after issuance, or
    if it draws on a forbidden series.

    Called by the builder for every feature, so a leaky spec cannot reach the
    matrix even if a test is not run.
    """
    if spec.source in cfg.FORBIDDEN:
        raise ValueError(
            f"feature '{spec.name}' draws on forbidden series '{spec.source}'. "
            "Realized outturn never enters the feature matrix."
        )
    for ts in sample:
        if not spec.is_admissible_at(ts):
            raise ValueError(
                f"feature '{spec.name}' violates gate closure at {ts}: "
                f"available {spec.availability(ts)}, issuance {issuance_time(ts)}"
            )


def audit(specs: list[FeatureSpec], sample: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Human readable admissibility table, for the README and for inspection.

    One row per feature with its worst-case slack: how long before issuance
    the value became available. Zero slack is allowed, negative is not.
    """
    rows = []
    for s in specs:
        slacks = [(issuance_time(t) - s.availability(t)).total_seconds() / 3600
                  for t in sample]
        finite = [x for x in slacks if x < 1e6]
        rows.append({
            "feature": s.name,
            "kind": s.kind,
            "offset_days": s.offset_days,
            "source": s.source or "",
            "min_slack_hours": round(min(finite), 2) if finite else float("inf"),
            "admissible": all(x >= 0 for x in slacks),
        })
    return pd.DataFrame(rows)
