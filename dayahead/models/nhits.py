"""
N-HiTS, the neural forecaster in the ladder.

Framing, and why it differs from every other model here
-------------------------------------------------------

The classical models and LightGBM are fitted per delivery hour, 24 models
each, per DESIGN.md section 18. N-HiTS is not. It is a sequence model: it
consumes a window of recent history and emits the whole 24-hour horizon at
once, so splitting the data by hour would remove the structure it exists to
learn.

So this is one model on the hourly series, input window 168 hours, output
horizon 24 hours. That is what DESIGN.md section 10 step 6 specifies, so no
deviation is recorded. But it does make N-HiTS the only entry in the ladder
using a different data arrangement, and that has to be said when the result
is reported: if it loses, the framing is a candidate explanation alongside
the architecture, and the two cannot be separated from this experiment alone.

Gate closure
------------

Parameters are estimated once per fold on the training window. Predictions
are then produced one delivery day at a time: history through the end of D-1,
future covariates for D, horizon 24. The model never sees a price from day D
or later when forecasting day D, and the exogenous inputs are the same
day-ahead forecasts audited in section 4.

Compute budget
--------------

Fixed at max_steps and no hyperparameter search, per DESIGN.md section 12
point 5, which predicted this model would not win. A search run only for the
model expected to lose, or only for the model expected to win, would make the
comparison meaningless either way. Every model in the ladder gets one honest
configuration.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from .. import config as cfg
from ..evaluation.splits import QUANTILES
from .base import Forecaster

INPUT_SIZE = 168          # one week of hourly history
HORIZON = 24              # one delivery day
MAX_STEPS = 300
BATCH_SIZE = 32

# Longest hole in the conditioning history that will be bridged rather than
# refused. SMARD skipped a whole delivery day on 2026-09-13, and
# neuralforecast requires a contiguous hourly series to condition on, so a
# single missing day made the model emit nothing at all. Bridging up to two
# days keeps the forecaster running through an outage of that size; anything
# longer is a different problem and is refused loudly.
MAX_CONDITIONING_GAP_HOURS = 48

# Any prediction beyond this is a numerical failure, not a forecast. The
# observed price range across the whole sample is -500 to 936 EUR/MWh.
#
# arima.py has carried this guard since section 7, where SARIMA diverged to
# an MAE of 1.4e10 on a single fold. It was not added here, and section 12
# paid for the omission: 18 hours across 2026-02-27 and 28 produced
# predictions as extreme as -5,276 EUR/MWh against an outturn near zero,
# inflating the locked test MAE by 22% and the RMSE threefold. The guard is
# added now and is not retroactive; section 24 reports what was measured.
DIVERGENCE_LIMIT = 10_000

# Future-known covariates. These are the published day-ahead forecasts, so
# they are available for the delivery day at issuance. One side of the
# residual identity only, as in the classical models.
FUTR_EXOG = ["x_fc_residual", "x_fc_solar", "x_wind_total",
             "cal_is_nonworking", "cal_is_holiday"]


def _quiet():
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")


def _quantile_columns(cols: list[str], prefix: str = "NHITS") -> dict[float, str]:
    """
    Map our nine quantiles onto neuralforecast's MQLoss column names.

    MQLoss labels outputs by symmetric prediction interval rather than by
    quantile: 'NHITS-lo-80.0' is the 10th percentile, 'NHITS-median' is the
    50th, 'NHITS-hi-80.0' is the 90th. The mapping is derived here rather
    than hardcoded, so a change in the library's naming produces a missing
    key instead of a silently mislabelled interval.
    """
    out: dict[float, str] = {}
    for c in cols:
        if c == f"{prefix}-median":
            out[0.5] = c
        elif "-lo-" in c:
            level = float(c.rsplit("-", 1)[1])
            out[round((1 - level / 100) / 2, 2)] = c
        elif "-hi-" in c:
            level = float(c.rsplit("-", 1)[1])
            out[round(1 - (1 - level / 100) / 2, 2)] = c
    return out


class NHiTSForecaster(Forecaster):
    """One sequence model on the hourly series."""

    name = "N-HiTS"
    native_quantiles = True

    def __init__(self, name: str = "N-HiTS", input_size: int = INPUT_SIZE,
                 max_steps: int = MAX_STEPS, futr_exog: list[str] | None = None):
        self.name = name
        self.input_size = input_size
        self.max_steps = max_steps
        self.futr_exog = list(futr_exog if futr_exog is not None else FUTR_EXOG)
        self._nf = None
        self._train_df: pd.DataFrame | None = None
        self._bridged_hours = 0
        self._failed_days = 0
        self._qcols: dict[float, str] = {}
        self._skipped_days = 0

    # ------------------------------------------------------------ internals
    def _frame(self, X: pd.DataFrame, y: pd.Series | None) -> pd.DataFrame:
        """
        Model frame with ds in UTC.

        UTC rather than local time, because neuralforecast requires a
        strictly consecutive hourly ds and local time is not: an hour is
        skipped every spring and repeated every autumn. The delivery-day
        grouping stays local, since that is what the gate closure rule is
        defined on; only the internal time axis is UTC.
        """
        cols = [c for c in self.futr_exog if c in X.columns]
        df = pd.DataFrame({
            "unique_id": "DE_LU",
            "ds": X.index.tz_convert("UTC").tz_localize(None),
        })
        for c in cols:
            df[c] = X[c].to_numpy(dtype=float)
        if y is not None:
            df["y"] = y.to_numpy(dtype=float)
        return df

    def _contiguous(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill short holes in the conditioning history.

        neuralforecast requires a gap-free hourly series and refuses one that
        is not, which on 2026-09-14 meant a single unpublished delivery day
        stopped the forecaster entirely.

        Bridged values are linear interpolations used only to condition the
        network. They never enter training, never enter any evaluation, and
        never appear in a reported figure. The count is exposed in report()
        so a forecast resting on a bridged history is identifiable rather
        than indistinguishable from one that is not.
        """
        self._bridged_hours = 0
        if df is None or df.empty:
            return df

        full = pd.date_range(df["ds"].min(), df["ds"].max(), freq="h")
        if len(full) == len(df):
            return df

        out = (df.set_index("ds").reindex(full)
               .rename_axis("ds").reset_index())
        missing = int(out["y"].isna().sum())
        if missing > MAX_CONDITIONING_GAP_HOURS:
            raise RuntimeError(
                f"{self.name}: {missing} hours missing from the conditioning "
                f"history, above the {MAX_CONDITIONING_GAP_HOURS} hour limit. "
                "Bridging a hole this size would be invention, not repair."
            )

        out["unique_id"] = df["unique_id"].iloc[0]
        numeric = [c for c in out.columns if c not in ("ds", "unique_id")]
        out[numeric] = out[numeric].interpolate(limit_direction="both")
        self._bridged_hours = missing
        return out

    # ----------------------------------------------------------------- api
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NHiTSForecaster":
        _quiet()
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import MQLoss
        from neuralforecast.models import NHITS

        df = self._frame(X, y)
        df = df.dropna().sort_values("ds").reset_index(drop=True)
        if len(df) < self.input_size + HORIZON * 30:
            raise RuntimeError(f"{self.name}: training window too short")

        model = NHITS(
            h=HORIZON,
            input_size=self.input_size,
            loss=MQLoss(quantiles=QUANTILES),
            futr_exog_list=[c for c in self.futr_exog if c in df.columns],
            max_steps=self.max_steps,
            batch_size=BATCH_SIZE,
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
            scaler_type="robust",
        )
        self._nf = NeuralForecast(models=[model], freq="h")
        self._nf.fit(df=df, verbose=False)
        self._train_df = df
        return self

    def predict_quantiles(self, X: pd.DataFrame,
                          quantiles=None, y: pd.Series | None = None
                          ) -> dict[float, np.ndarray]:
        """
        One delivery day at a time.

        Each call conditions on history through the end of D-1 and the
        published covariates for D. Realised prices for the test window are
        appended to the history as they become known, which mirrors what a
        forecaster holds at gate closure and matches the treatment of the
        state space models in section 7. Parameters are not re-estimated.
        """
        _quiet()
        if self._nf is None:
            raise RuntimeError("fit before predict")

        qs = list(quantiles or QUANTILES)
        out = {q: pd.Series(np.nan, index=X.index, dtype=float) for q in qs}

        history = self._contiguous(self._train_df)

        future = self._frame(X, y)
        # Group by local delivery date, which is what gate closure is defined
        # on, while the ds axis itself stays in UTC.
        future["_date"] = X.index.tz_convert(cfg.LOCAL_TZ).normalize().tz_localize(None)
        self._skipped_days = 0
        self._failed_days = 0

        for _, block in future.groupby("_date", sort=True):
            block = block.drop(columns="_date")
            futr = block.drop(columns=[c for c in ("y",) if c in block])

            # A local delivery day spans 23, 24 or 25 hours across a DST
            # transition, and the horizon is fixed at 24. Predict for the
            # rows that fit and map back by timestamp; at most one hour a
            # year goes uncovered.
            #
            # The earlier version skipped such a day entirely with a bare
            # continue, which also skipped the history update below. The
            # conditioning window then stopped advancing and every remaining
            # day of that fold produced nothing. Two DST days cost 837 hours
            # of the locked test year, 9.6%, before this was found.
            usable = futr.iloc[:HORIZON]
            if len(usable) == HORIZON:
                try:
                    pred = self._nf.predict(df=history, futr_df=usable,
                                            verbose=False)
                except Exception as exc:
                    # Counted rather than swallowed. A silent failure here
                    # produces a full set of NaN predictions that look like a
                    # successful run: the log gains its rows, the page draws
                    # an empty chart, and nothing reports an error. That is
                    # exactly what happened on 2026-09-14.
                    self._failed_days += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    pred = None

                if pred is not None:
                    if not self._qcols:
                        self._qcols = _quantile_columns(list(pred.columns))
                    stamps = (pd.DatetimeIndex(pred["ds"])
                              .tz_localize("UTC").tz_convert(cfg.LOCAL_TZ))
                    keep = stamps.isin(X.index)
                    for q in qs:
                        col = self._qcols.get(q)
                        if col is None:
                            continue
                        out[q].loc[stamps[keep]] = pred[col].to_numpy(float)[keep]
            else:
                self._skipped_days += 1

            # Always advance the conditioning window, whether or not this day
            # could be predicted. This is the line whose absence caused the
            # cascade described above.
            if "y" in block:
                history = pd.concat([history, block], ignore_index=True)

        result = {}
        for q, v in out.items():
            arr = v.to_numpy(dtype=float)
            arr[np.abs(arr) > DIVERGENCE_LIMIT] = np.nan
            result[q] = arr
        return result

    def predict(self, X: pd.DataFrame, y: pd.Series | None = None) -> np.ndarray:
        return self.predict_quantiles(X, [0.5], y=y)[0.5]

    def report(self) -> dict:
        return {
            "name": self.name,
            "input_size": self.input_size,
            "max_steps": self.max_steps,
            "futr_exog": self.futr_exog,
            "train_rows": None if self._train_df is None else len(self._train_df),
            "skipped_days": self._skipped_days,
            "failed_days": self._failed_days,
            "bridged_hours": self._bridged_hours,
            "last_error": getattr(self, "_last_error", None),
        }


def nhits() -> NHiTSForecaster:
    return NHiTSForecaster()
