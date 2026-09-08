"""
Metrics.

Implements DESIGN.md section 6. MAPE and sMAPE are absent on purpose: the
price crosses zero in 3.9% of hours, so percentage error is undefined or
unbounded. That is a property of the market, not a stylistic preference.

Every function takes plain arrays and returns plain floats, so they are
testable without any model or data on disk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .splits import INTERVAL_HIGH, INTERVAL_LOW, NOMINAL_COVERAGE, QUANTILES


def _clean(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[ok], y_pred[ok]


def mae(y_true, y_pred) -> float:
    t, p = _clean(y_true, y_pred)
    return float(np.mean(np.abs(t - p))) if len(t) else float("nan")


def rmse(y_true, y_pred) -> float:
    t, p = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((t - p) ** 2))) if len(t) else float("nan")


def bias(y_true, y_pred) -> float:
    t, p = _clean(y_true, y_pred)
    return float(np.mean(p - t)) if len(t) else float("nan")


def skill_score(y_true, y_pred, y_baseline) -> float:
    """
    1 - MAE(model) / MAE(baseline).

    Positive means better than the baseline, zero means indistinguishable,
    negative means worse. Computed on rows where all three are finite, so the
    comparison is like for like.
    """
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    b = np.asarray(y_baseline, dtype=float)
    ok = np.isfinite(t) & np.isfinite(p) & np.isfinite(b)
    if not ok.any():
        return float("nan")
    m = np.mean(np.abs(t[ok] - p[ok]))
    n = np.mean(np.abs(t[ok] - b[ok]))
    return float(1 - m / n) if n > 0 else float("nan")


def pinball_loss(y_true, q_pred, quantile: float) -> float:
    """
    Pinball loss for a single quantile.

    Under-prediction is penalised by quantile * error, over-prediction by
    (1 - quantile) * error. At q=0.5 it reduces to half the absolute error.
    """
    t, p = _clean(y_true, q_pred)
    if not len(t):
        return float("nan")
    d = t - p
    return float(np.mean(np.maximum(quantile * d, (quantile - 1) * d)))


def mean_pinball(y_true, quantile_preds: dict[float, np.ndarray]) -> float:
    """Average pinball loss across the nine quantiles."""
    vals = [pinball_loss(y_true, quantile_preds[q], q)
            for q in QUANTILES if q in quantile_preds]
    return float(np.mean(vals)) if vals else float("nan")


def coverage(y_true, lower, upper) -> float:
    """Share of observations falling inside the interval."""
    t = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    ok = np.isfinite(t) & np.isfinite(lo) & np.isfinite(hi)
    if not ok.any():
        return float("nan")
    return float(np.mean((t[ok] >= lo[ok]) & (t[ok] <= hi[ok])))


def interval_width(lower, upper) -> float:
    """
    Mean interval width. Reported next to coverage because any coverage
    target is trivially met by a wide enough interval.
    """
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    ok = np.isfinite(lo) & np.isfinite(hi)
    return float(np.mean(hi[ok] - lo[ok])) if ok.any() else float("nan")


def evaluate(df: pd.DataFrame, baseline_col: str | None = None) -> dict:
    """
    All metrics for one long-format prediction frame.

    Expected columns: y_true, y_pred, and q10 through q90 where available.
    """
    out = {
        "n": int(df["y_true"].notna().sum()),
        "mae": mae(df["y_true"], df["y_pred"]),
        "rmse": rmse(df["y_true"], df["y_pred"]),
        "bias": bias(df["y_true"], df["y_pred"]),
    }
    if baseline_col and baseline_col in df:
        out["skill"] = skill_score(df["y_true"], df["y_pred"], df[baseline_col])

    qcols = {q: f"q{int(q * 100):02d}" for q in QUANTILES}
    have = {q: df[c].to_numpy() for q, c in qcols.items() if c in df}
    if have:
        out["pinball"] = mean_pinball(df["y_true"], have)
        lo_c, hi_c = qcols[INTERVAL_LOW], qcols[INTERVAL_HIGH]
        if lo_c in df and hi_c in df:
            out["coverage"] = coverage(df["y_true"], df[lo_c], df[hi_c])
            out["nominal_coverage"] = NOMINAL_COVERAGE
            out["interval_width"] = interval_width(df[lo_c], df[hi_c])
    return out
