"""
Rolling origin backtest.

Runs every model over the same expanding-window folds and returns predictions
in long format, one row per delivery hour per model, with the point forecast
and all nine quantiles.

Predictions are stored rather than metrics. Metrics computed on the fly cannot
be re-sliced later, and DESIGN.md sections 8 and 13 demand slices by regime,
by delivery hour, and conditional on negative prices and spikes. Keeping the
predictions means every one of those can be produced without refitting.

Leakage guards run inside the harness, not only in tests: the training index
is asserted to end strictly before the test index begins, on every fold, for
every model.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .. import config as cfg
from ..models.base import Forecaster
from .splits import QUANTILES, add_regime, backtest_frame, folds

TARGET_COL = "y"


def _feature_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in (TARGET_COL, "is_usable")]


def _assert_no_overlap(train_idx: pd.DatetimeIndex,
                       test_idx: pd.DatetimeIndex, month: str) -> None:
    if train_idx.max() >= test_idx.min():
        raise AssertionError(
            f"fold {month}: training data reaches {train_idx.max()}, "
            f"test begins {test_idx.min()}"
        )
    if len(train_idx.intersection(test_idx)):
        raise AssertionError(f"fold {month}: train and test indices overlap")


def run_backtest(X: pd.DataFrame, models: list[Forecaster],
                 verbose: bool = True, every: int = 1) -> pd.DataFrame:
    """
    X       feature matrix from section 4, with y and is_usable
    models  fitted fresh on every fold
    """
    bt = backtest_frame(X)
    fold_list = folds(X)[:: max(1, every)]
    feature_cols = _feature_columns(bt)

    if verbose:
        print(f"  folds: {len(fold_list)}   "
              f"{fold_list[0]['month']} to {fold_list[-1]['month']}")
        print(f"  features: {len(feature_cols)}   models: {len(models)}")

    records = []
    t0 = time.time()
    for i, fold in enumerate(fold_list, 1):
        train_idx, test_idx = fold["train_index"], fold["test_index"]
        _assert_no_overlap(train_idx, test_idx, fold["month"])

        train = bt.loc[train_idx]
        train = train[train["is_usable"]]
        test = bt.loc[test_idx]
        test = test[test["is_usable"]]
        if len(train) == 0 or len(test) == 0:
            continue

        Xtr, ytr = train[feature_cols], train[TARGET_COL]
        Xte, yte = test[feature_cols], test[TARGET_COL]

        for model in models:
            fitted = model.fit(Xtr, ytr)
            # State space models need the realised series to advance their
            # conditioning state one step at a time. They use it for filtering
            # only, never for parameter estimation, and each prediction still
            # conditions on t-1 alone. Models that ignore the argument are
            # unaffected.
            try:
                point = fitted.predict(Xte, y=yte)
            except TypeError:
                point = fitted.predict(Xte)
            try:
                # Sequence models need the realised series to advance their
                # conditioning window, exactly as the state space models do.
                # Used for conditioning only, never for fitting.
                try:
                    qs = fitted.predict_quantiles(Xte, y=yte)
                except TypeError:
                    qs = fitted.predict_quantiles(Xte)
            except NotImplementedError:
                qs = {}

            rec = pd.DataFrame({
                "timestamp": Xte.index,
                "model": fitted.name,
                "fold": fold["month"],
                "regime": fold["regime"],
                "y_true": yte.to_numpy(dtype=float),
                "y_pred": np.asarray(point, dtype=float),
            })
            for q in QUANTILES:
                if q in qs:
                    rec[f"q{int(q * 100):02d}"] = np.asarray(qs[q], dtype=float)
            records.append(rec)

        if verbose and (i % 12 == 0 or i == len(fold_list)):
            print(f"    fold {i}/{len(fold_list)}  {fold['month']}  "
                  f"train={len(train):,}  test={len(test):,}")

    out = pd.concat(records, ignore_index=True)
    out["hour"] = out["timestamp"].dt.hour
    if verbose:
        print(f"  elapsed: {time.time() - t0:.1f}s   rows: {len(out):,}")
    return out


def attach_baseline(preds: pd.DataFrame, baseline_name: str) -> pd.DataFrame:
    """
    Add the baseline's point forecast as a column on every row, so skill
    scores can be computed within any slice.
    """
    base = (preds[preds["model"] == baseline_name]
            .set_index("timestamp")["y_pred"]
            .rename("y_baseline"))
    if base.empty:
        raise KeyError(f"baseline '{baseline_name}' not present in predictions")
    return preds.merge(base, left_on="timestamp", right_index=True, how="left")


def summarise(preds: pd.DataFrame, by: list[str] | None = None,
              baseline_name: str = "B1_weekly_naive") -> pd.DataFrame:
    """Metrics table, grouped by model plus any additional keys."""
    from .metrics import evaluate

    df = attach_baseline(preds, baseline_name)
    keys = ["model"] + (by or [])
    rows = []
    for values, group in df.groupby(keys, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        row = dict(zip(keys, values))
        row.update(evaluate(group, baseline_col="y_baseline"))
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["model"] + (by or [])).reset_index(drop=True)


def fold_table(preds: pd.DataFrame,
               baseline_name: str = "B1_weekly_naive") -> pd.DataFrame:
    """
    Per-fold metrics. DESIGN.md section 13 check 3 requires the distribution
    of fold performance, not just its mean, because a good average across 59
    folds can hide catastrophic individual months.
    """
    return summarise(preds, by=["fold", "regime"], baseline_name=baseline_name)
