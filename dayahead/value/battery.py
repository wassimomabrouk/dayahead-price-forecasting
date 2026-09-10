"""
Battery arbitrage valuation.

Implements the specification fixed in DESIGN.md section 25, which was
committed before any figure here was computed. Nothing in this module chooses
a parameter; they are all read from the constants below, which match that
section.

What this measures, and what it does not
----------------------------------------

It measures the relative value of forecast information under one dispatch
rule, holding everything except the forecast constant. It is not a profit and
loss statement: degradation, grid fees, imbalance exposure and the intraday
market are all absent, and a real operator's rule would differ.

The comparison across information sets is the meaningful part. The absolute
euro figures are illustrative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as cfg

# Section 25, fixed before computation.
POWER_MW = 1.0
CAPACITY_MWH = 2.0
ROUND_TRIP_EFFICIENCY = 0.85
CHARGE_HOURS = int(CAPACITY_MWH / POWER_MW)
DISCHARGE_HOURS = CHARGE_HOURS
ABSTAIN_QUANTILE = 0.80        # rule B skips the top quintile of width


def dispatch_day(prices: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Choose charge and discharge hours for one delivery day.

    The battery starts empty, so every charging hour must precede every
    discharging hour. The schedule is found by trying each admissible split
    of the day: for a given last-charge hour, take the cheapest hours before
    it and the dearest after it. Twenty-four candidates, evaluated exactly
    rather than approximated.

    Returns the chosen hour indices, or None if the day is too short to hold
    a full cycle.
    """
    n = len(prices)
    if n < CHARGE_HOURS + DISCHARGE_HOURS:
        return None

    best_value = -np.inf
    best: tuple[np.ndarray, np.ndarray] | None = None

    for split in range(CHARGE_HOURS, n - DISCHARGE_HOURS + 1):
        before, after = prices[:split], prices[split:]
        charge_idx = np.argsort(before)[:CHARGE_HOURS]
        discharge_idx = np.argsort(after)[-DISCHARGE_HOURS:] + split

        value = (ROUND_TRIP_EFFICIENCY * prices[discharge_idx].sum()
                 - prices[charge_idx].sum())
        if value > best_value:
            best_value = value
            best = (np.sort(charge_idx), np.sort(discharge_idx))

    return best


def settle(realised: np.ndarray, charge_idx: np.ndarray,
           discharge_idx: np.ndarray) -> float:
    """
    Revenue at realised prices.

    The schedule was committed on a forecast; the money is made or lost at
    the prices that actually cleared. Settling at forecast prices would
    measure how confident the model was, not how right it was.
    """
    return float(ROUND_TRIP_EFFICIENCY * realised[discharge_idx].sum()
                 - realised[charge_idx].sum())


def run_strategy(frame: pd.DataFrame, signal_col: str,
                 abstain_days: set | None = None) -> pd.DataFrame:
    """
    Dispatch across every delivery day on one forecast signal.

    frame         one row per delivery hour, with y_true and the signal
    signal_col    the column the schedule is chosen on
    abstain_days  days on which the battery does not trade, for rule B
    """
    rows = []
    local_date = frame.index.tz_convert(cfg.LOCAL_TZ).normalize()

    for day, block in frame.groupby(local_date):
        signal = block[signal_col].to_numpy(dtype=float)
        realised = block["y_true"].to_numpy(dtype=float)
        if not np.isfinite(signal).all() or not np.isfinite(realised).all():
            rows.append({"date": day, "revenue": 0.0, "traded": False,
                         "reason": "missing forecast"})
            continue
        if abstain_days is not None and day in abstain_days:
            rows.append({"date": day, "revenue": 0.0, "traded": False,
                         "reason": "abstained"})
            continue

        plan = dispatch_day(signal)
        if plan is None:
            rows.append({"date": day, "revenue": 0.0, "traded": False,
                         "reason": "short day"})
            continue

        rows.append({"date": day, "revenue": settle(realised, *plan),
                     "traded": True, "reason": ""})

    return pd.DataFrame(rows)


def wide_interval_days(frame: pd.DataFrame, low: str = "q10",
                       high: str = "q90") -> set:
    """Days in the top quintile of mean forecast interval width."""
    width = (frame[high] - frame[low]).astype(float)
    daily = width.groupby(frame.index.tz_convert(cfg.LOCAL_TZ).normalize()).mean()
    threshold = daily.quantile(ABSTAIN_QUANTILE)
    return set(daily[daily >= threshold].index)


def valuation(preds: pd.DataFrame, champion: str = "N-HiTS",
              baseline: str = "B2_daily_naive") -> dict:
    """
    The full comparison: three information sets, two rules.

    Perfect foresight dispatches on the realised prices themselves. It is not
    attainable and is reported only as the denominator.
    """
    champ = (preds[preds["model"] == champion]
             .set_index("timestamp").sort_index())
    base = (preds[preds["model"] == baseline]
            .set_index("timestamp").sort_index())

    # Every information set is evaluated on the same days, so the comparison
    # is not confounded by one of them trading more often than another.
    common = champ.index.intersection(base.index)
    champ, base = champ.loc[common], base.loc[common]

    perfect = champ.copy()
    perfect["oracle"] = perfect["y_true"]

    results = {
        "perfect_foresight": run_strategy(perfect, "oracle"),
        "champion_rule_A": run_strategy(champ, "y_pred"),
        "baseline_rule_A": run_strategy(base, "y_pred"),
    }

    if {"q10", "q90"}.issubset(champ.columns):
        skip = wide_interval_days(champ)
        results["champion_rule_B"] = run_strategy(champ, "y_pred",
                                                  abstain_days=skip)
        results["baseline_rule_B"] = run_strategy(base, "y_pred",
                                                  abstain_days=skip)

    oracle_total = results["perfect_foresight"]["revenue"].sum()

    summary = []
    for name, frame in results.items():
        traded = frame[frame["traded"]]
        total = float(frame["revenue"].sum())
        summary.append({
            "strategy": name,
            "days": int(len(frame)),
            "days_traded": int(len(traded)),
            "revenue_eur": total,
            "revenue_per_traded_day": (float(traded["revenue"].mean())
                                       if len(traded) else np.nan),
            "share_of_perfect": total / oracle_total if oracle_total else np.nan,
            "loss_making_days": int((frame["revenue"] < 0).sum()),
        })

    return {"summary": pd.DataFrame(summary), "daily": results,
            "oracle_total": float(oracle_total)}
