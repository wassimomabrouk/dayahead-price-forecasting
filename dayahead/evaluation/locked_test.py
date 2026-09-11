"""
Section 12: the locked test evaluation.

Opened once. Everything it depends on was fixed before it ran: the champion
in section 11, the calibration method in section 22, the metrics and split
protocol in sections 6 and 7, and the expectations in section 12 of the
design.

Protocol
--------

The same monthly expanding-window scheme as the backtest. Fold m trains on
everything up to the last hour of month m-1, which here includes the whole
backtest window, and forecasts every hour of month m. Using a different
protocol for the final evaluation than for development would make the two
figures incomparable, which is the thing a held-out set exists to avoid.

Calibration uses the twelve folds preceding each test month, which for the
early test months means backtest folds. Those closed before the test month
began, so the information constraint holds exactly as it did in section 10.

The six robustness checks
-------------------------

Pre-committed in DESIGN.md section 13 and run whether or not they flatter the
result. Check 1, the leak quantification, is the one that matters most: it
converts the project's central claim from an assertion into a number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as cfg
from ..models.naive import all_baselines
from .backtest import run_backtest
from .conformal import calibrate_model
from .metrics import evaluate, mae, skill_score
from .splits import QUANTILES, backtest_frame, test_folds

SPIKE_THRESHOLD = 200.0


def _run_folds(X: pd.DataFrame, models, folds, verbose=True) -> pd.DataFrame:
    """Run models over an explicit fold list, reusing the backtest harness."""
    import types

    from . import backtest as bt

    original = bt.folds
    bt.folds = lambda _X: folds
    original_frame = bt.backtest_frame
    bt.backtest_frame = lambda _X: _X
    try:
        return run_backtest(X, models, verbose=verbose)
    finally:
        bt.folds = original
        bt.backtest_frame = original_frame


def evaluate_operational(verbose: bool = True) -> pd.DataFrame:
    """
    Section 16. The champion on the feature set a live system can assemble.

    Section 27 established that four of the six exogenous inputs are
    published by SMARD after the auction they would have informed. This
    refits the champion using only price history, calendar terms and the
    day-ahead load forecast, over the same locked test folds, so the gap
    against the section 12 result is the cost of being able to run.

    It is the mirror of the section 13 leak quantification. That measured
    what using forbidden information would buy; this measures what being
    restricted to obtainable information costs.
    """
    from ..features.build import build_features, build_reduced_features
    from ..models.nhits import NHiTSForecaster

    full, _ = build_features()
    folds = test_folds(full, i_am_opening_the_locked_test_set=True)

    X, _ = build_reduced_features(verbose=verbose)
    # The champion's exogenous list must be restricted too, or the model will
    # ask for columns the reduced matrix no longer has.
    available = [c for c in X.columns if c.startswith(("x_", "cal_"))]
    model = NHiTSForecaster(name="N-HiTS_operational",
                            futr_exog=[c for c in available
                                       if c in ("x_fc_load", "cal_is_nonworking",
                                                "cal_is_holiday")])
    preds = _run_folds(X, [model], folds, verbose)
    preds["model"] = "N-HiTS_operational"
    return preds


def evaluate_locked_test(verbose: bool = True) -> dict:
    from ..features.build import build_features, build_leaky_features
    from ..models.nhits import nhits

    X, _ = build_features()
    folds = test_folds(X, i_am_opening_the_locked_test_set=True)
    if verbose:
        print(f"  test folds: {len(folds)}  "
              f"{folds[0]['month']} to {folds[-1]['month']}")

    results: dict[str, pd.DataFrame] = {}

    # Champion plus both baselines, the headline evaluation.
    models = [nhits()] + all_baselines()
    results["main"] = _run_folds(X, models, folds, verbose)

    # Check 1. Realised outturn substituted for the published forecasts.
    leaky_X, _ = build_leaky_features(i_am_measuring_the_leak=True)
    leak = _run_folds(leaky_X, [nhits()], folds, verbose)
    leak["model"] = "N-HiTS_LEAKY"
    results["leak"] = leak

    # Check 2. Champion trained on post-2023 data only.
    cutoff = pd.Timestamp("2023-01-01", tz=cfg.LOCAL_TZ)
    restricted = [
        {**f, "train_index": f["train_index"][f["train_index"] >= cutoff]}
        for f in folds
    ]
    recent = _run_folds(X, [nhits()], restricted, verbose)
    recent["model"] = "N-HiTS_post2023"
    results["recent"] = recent

    return results


# Variant suffixes that share a base model's backtest history. A variant
# whose suffix is missing here silently loses its calibration history, and
# with it the first folds of the test year: the section 16 run produced
# predictions for all twelve folds and reported metrics on six, because
# "_operational" was not on this list.
VARIANT_SUFFIXES = ("_LEAKY", "_post2023", "_operational")


def base_model_name(name: str) -> str:
    for suffix in VARIANT_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def calibrate_test(test_preds: pd.DataFrame,
                   backtest_preds: pd.DataFrame) -> pd.DataFrame:
    """
    Calibrate the test predictions using history that precedes them.

    Backtest and test predictions for the same model are concatenated so that
    the first test months draw their calibration from the final backtest
    folds. Only test rows are returned.

    A variant with no backtest history calibrates from the test folds alone,
    which costs the first six of them to the minimum-history rule. That is a
    silent loss: the metrics simply report a smaller n.
    """
    out = []
    test_folds_set = set(test_preds["fold"])
    for name, group in test_preds.groupby("model"):
        history_name = base_model_name(name)
        history = backtest_preds[backtest_preds["model"] == history_name]
        if history.empty:
            raise ValueError(
                f"no backtest history for '{name}' (base '{history_name}'). "
                "Add its suffix to VARIANT_SUFFIXES, or the first folds of "
                "the test year will be dropped without warning."
            )
        joined = pd.concat([history.assign(model=name), group], ignore_index=True)
        cal, _ = calibrate_model(joined)
        out.append(cal[cal["fold"].isin(test_folds_set)])
    return pd.concat(out, ignore_index=True)


def anatomy(preds: pd.DataFrame, baseline_name: str = "B1_weekly_naive") -> dict:
    """Error anatomy and the conditional subsets from check 6."""
    base = (preds[preds["model"] == baseline_name]
            .set_index("timestamp")["y_pred"].rename("y_baseline"))
    df = preds.merge(base, left_on="timestamp", right_index=True, how="left")

    out: dict[str, pd.DataFrame] = {}

    rows = []
    for name, g in df.groupby("model"):
        r = {"model": name}
        r.update(evaluate(g, baseline_col="y_baseline"))
        rows.append(r)
    out["overall"] = pd.DataFrame(rows).sort_values("mae")

    out["by_hour"] = (df.groupby(["model", "hour"])
                      .apply(lambda g: pd.Series({"mae": mae(g["y_true"], g["y_pred"])}),
                             include_groups=False)
                      .reset_index())

    out["by_month"] = (df.groupby(["model", "fold"])
                       .apply(lambda g: pd.Series({"mae": mae(g["y_true"], g["y_pred"])}),
                              include_groups=False)
                       .reset_index())

    subsets = []
    conditions = {
        "negative_price": df["y_true"] < 0,
        "spike_above_200": df["y_true"] > SPIKE_THRESHOLD,
        "ordinary": (df["y_true"] >= 0) & (df["y_true"] <= SPIKE_THRESHOLD),
    }
    for label, mask in conditions.items():
        sub = df[mask]
        if sub.empty:
            continue
        for name, g in sub.groupby("model"):
            subsets.append({
                "subset": label, "model": name, "n": len(g),
                "mae": mae(g["y_true"], g["y_pred"]),
                "skill_vs_B1": skill_score(g["y_true"], g["y_pred"],
                                           g["y_baseline"]),
            })
    out["by_condition"] = pd.DataFrame(subsets)
    return out
