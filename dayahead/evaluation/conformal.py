"""
Conformal calibration of the forecast quantiles.

Why this section is load bearing
--------------------------------

Every model in the ladder is under-covered. Against a nominal 80% interval,
section 21 measured 0.693 for N-HiTS, 0.646 for anchored LightGBM and 0.576
for SARIMAX, and within the crisis regime SARIMAX reached 0.192. Section 11
selects the champion on pinball loss, so ranking models on intervals known to
be wrong would make the choice arbitrary.

Where the calibration data comes from
-------------------------------------

The usual split-conformal recipe holds out part of the training window and
refits. That would cost another full backtest. It is unnecessary here because
the backtest already produced 59 folds of genuine out-of-sample predictions,
stored per timestamp. Fold m can be calibrated on folds strictly before m.

This is not a shortcut around the information constraint. The calibration
residuals for fold m come from delivery days that closed before fold m began,
so they were available at issuance in exactly the way the price history was.
The rolling window means calibration also tracks regime change rather than
averaging over eight years.

Global against conditional
--------------------------

Section 3b measured price variance as U-shaped in forecast residual load:
21.67 EUR/MWh in the middle deciles against 37.62 and 65.66 at the ends. A
single global width cannot express that, so it will be too wide mid-range and
too narrow in the tails, which is where the money is.

Both variants are therefore computed and reported. Global is the standard
method and the honest reference point; conditional bins the calibration
residuals by forecast residual load and corrects each bin separately. If the
conditional version does not improve on the global one, that is a finding
about the section 3b reasoning and is reported as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as cfg
from .splits import INTERVAL_HIGH, INTERVAL_LOW, QUANTILES

# Folds of history used to calibrate each fold. Twelve months tracks regime
# change without becoming so short that the tail quantiles are estimated on a
# handful of observations.
CALIBRATION_FOLDS = 12
MIN_CALIBRATION_FOLDS = 6

# Bins for the conditional variant, cut on forecast residual load. Five keeps
# roughly 1,700 calibration rows per bin at a twelve month window, which is
# enough to estimate a 0.1 quantile without it being driven by single points.
N_BINS = 5
BIN_COLUMN = "x_fc_residual"
MIN_BIN_ROWS = 200


def qcol(q: float) -> str:
    return f"q{int(q * 100):02d}"


def _deltas(cal: pd.DataFrame, quantiles=QUANTILES) -> dict[float, float]:
    """
    Additive correction per quantile, measured against that quantile.

    For level q the conformity score is y - qhat_q, the outturn minus the
    model's own q-th quantile, and the correction is the q-th quantile of
    that score on the calibration set. Adding it back makes the marginal
    coverage of qhat_q correct on the calibration distribution.

    The self-referential form matters. An earlier version scored every
    quantile against the point forecast instead, then added the result on
    top of the model's existing spread. That double-counts the spread: on a
    synthetic case with true coverage 0.442 it produced 0.912 against a
    nominal 0.80, overshooting as far in one direction as the raw model
    missed in the other. Scored against qhat_q, a model that is already
    calibrated receives a correction of zero, which is the property that
    makes the method a correction rather than a second spread.
    """
    y = cal["y_true"].to_numpy(dtype=float)
    out: dict[float, float] = {}
    for q in quantiles:
        c = qcol(q)
        if c not in cal:
            out[q] = 0.0
            continue
        r = y - cal[c].to_numpy(dtype=float)
        r = r[np.isfinite(r)]
        out[q] = float(np.quantile(r, q)) if len(r) else 0.0
    return out


def _apply(preds: pd.DataFrame, deltas: dict[float, float]) -> pd.DataFrame:
    out = preds.copy()
    for q, d in deltas.items():
        c = qcol(q)
        if c in out:
            out[c] = out[c] + d
    return out


def _enforce_monotone(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Sort quantiles per row. Independent corrections can reorder them."""
    cols = [qcol(q) for q in QUANTILES if qcol(q) in df]
    if len(cols) < 2:
        return df, 0
    block = df[cols].to_numpy(dtype=float)
    crossings = int(np.sum(np.diff(block, axis=1) < 0))
    block.sort(axis=1)
    out = df.copy()
    out[cols] = block
    return out, crossings


def calibrate_model(preds: pd.DataFrame, feature_col: pd.Series | None = None,
                    conditional: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Calibrate one model's predictions, fold by fold.

    preds        long format rows for a single model, with fold and quantiles
    feature_col  the binning variable, aligned on timestamp, for conditional
    """
    preds = preds.sort_values("timestamp").reset_index(drop=True)
    folds = sorted(preds["fold"].unique())
    fold_pos = {f: i for i, f in enumerate(folds)}

    if conditional:
        if feature_col is None:
            raise ValueError("conditional calibration needs a binning column")
        binvals = feature_col.reindex(preds["timestamp"]).to_numpy(dtype=float)
        preds = preds.assign(_bin_value=binvals)

    calibrated, skipped = [], []
    for f in folds:
        i = fold_pos[f]
        history_folds = folds[max(0, i - CALIBRATION_FOLDS):i]
        if len(history_folds) < MIN_CALIBRATION_FOLDS:
            skipped.append(f)
            continue

        cal = preds[preds["fold"].isin(history_folds)]
        cur = preds[preds["fold"] == f]
        if cal.empty or cur.empty:
            skipped.append(f)
            continue

        if not conditional:
            calibrated.append(_apply(cur, _deltas(cal)))
            continue

        # Conditional: bin edges from the calibration window only, so the
        # test fold never informs its own binning.
        edges = np.nanquantile(cal["_bin_value"].to_numpy(dtype=float),
                               np.linspace(0, 1, N_BINS + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        edges = np.unique(edges)

        cal_bin = np.digitize(cal["_bin_value"].to_numpy(dtype=float), edges[1:-1])
        cur_bin = np.digitize(cur["_bin_value"].to_numpy(dtype=float), edges[1:-1])
        global_deltas = _deltas(cal)

        pieces = []
        for b in np.unique(cur_bin):
            mask_cal = cal_bin == b
            # Too few calibration rows in a bin gives a tail quantile driven
            # by two or three points. Fall back to the global correction
            # rather than emit a confidently wrong interval.
            if mask_cal.sum() < MIN_BIN_ROWS:
                deltas = global_deltas
            else:
                deltas = _deltas(cal[mask_cal])
            pieces.append(_apply(cur[cur_bin == b], deltas))
        calibrated.append(pd.concat(pieces))

    if not calibrated:
        raise RuntimeError("no fold had enough calibration history")

    out = pd.concat(calibrated, ignore_index=True)
    out, crossings = _enforce_monotone(out)
    if "_bin_value" in out:
        out = out.drop(columns="_bin_value")
    return out, {
        "folds_calibrated": len(folds) - len(skipped),
        "folds_skipped": skipped,
        "quantile_crossings_repaired": crossings,
        "conditional": conditional,
    }


def calibrate_all(preds: pd.DataFrame, features: pd.DataFrame | None = None,
                  conditional: bool = False) -> tuple[pd.DataFrame, dict]:
    """Calibrate every model in a prediction frame."""
    feature_col = None
    if conditional:
        if features is None or BIN_COLUMN not in features:
            raise ValueError(f"conditional calibration needs {BIN_COLUMN}")
        feature_col = features[BIN_COLUMN].astype("float64")

    out, info = [], {}
    for name, group in preds.groupby("model"):
        cal, meta = calibrate_model(group, feature_col, conditional)
        cal["model"] = name
        out.append(cal)
        info[name] = meta
    return pd.concat(out, ignore_index=True), info


def coverage_table(before: pd.DataFrame, after_global: pd.DataFrame,
                   after_cond: pd.DataFrame,
                   by: list[str] | None = None) -> pd.DataFrame:
    """Coverage and width, uncalibrated against both calibrated variants."""
    from .metrics import coverage, interval_width, mean_pinball

    lo, hi = qcol(INTERVAL_LOW), qcol(INTERVAL_HIGH)
    keys = ["model"] + (by or [])

    def rows(df: pd.DataFrame, label: str) -> list[dict]:
        acc = []
        for values, g in df.groupby(keys, dropna=False):
            values = values if isinstance(values, tuple) else (values,)
            have = {q: g[qcol(q)].to_numpy() for q in QUANTILES if qcol(q) in g}
            r = dict(zip(keys, values))
            r.update({
                "variant": label,
                "n": int(g["y_true"].notna().sum()),
                "coverage": coverage(g["y_true"], g[lo], g[hi]),
                "width": interval_width(g[lo], g[hi]),
                "pinball": mean_pinball(g["y_true"], have) if have else np.nan,
            })
            acc.append(r)
        return acc

    # Compare on the folds all three variants share, so the difference is
    # calibration rather than a different set of evaluated days.
    common = set(after_global["fold"]) & set(after_cond["fold"])
    b = before[before["fold"].isin(common)]
    g = after_global[after_global["fold"].isin(common)]
    c = after_cond[after_cond["fold"].isin(common)]

    return pd.DataFrame(rows(b, "uncalibrated") + rows(g, "conformal_global")
                        + rows(c, "conformal_conditional"))
