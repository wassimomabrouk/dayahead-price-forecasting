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
        self._qcols: dict[float, str] = {}

    # ------------------------------------------------------------ internals
    def _frame(self, X: pd.DataFrame, y: pd.Series | None) -> pd.DataFrame:
        cols = [c for c in self.futr_exog if c in X.columns]
        df = pd.DataFrame({
            "unique_id": "DE_LU",
            "ds": X.index.tz_localize(None),
        })
        for c in cols:
            df[c] = X[c].to_numpy(dtype=float)
        if y is not None:
            df["y"] = y.to_numpy(dtype=float)
        return df

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

        future = self._frame(X, y)
        future["_date"] = future["ds"].dt.normalize()
        history = self._train_df.copy()

        for day, block in future.groupby("_date", sort=True):
            block = block.drop(columns="_date")
            futr = block.drop(columns=[c for c in ("y",) if c in block])
            if len(futr) != HORIZON:
                # Short days at DST transitions cannot fill the fixed horizon.
                continue
            try:
                pred = self._nf.predict(df=history, futr_df=futr, verbose=False)
            except Exception:
                continue

            if not self._qcols:
                self._qcols = _quantile_columns(list(pred.columns))
            stamps = pd.DatetimeIndex(pred["ds"]).tz_localize(cfg.LOCAL_TZ,
                                                              nonexistent="shift_forward",
                                                              ambiguous="NaT")
            keep = stamps.notna() & stamps.isin(X.index)
            for q in qs:
                col = self._qcols.get(q)
                if col is None:
                    continue
                out[q].loc[stamps[keep]] = pred[col].to_numpy(float)[keep]

            # Advance the conditioning window with what is now known.
            if "y" in block:
                history = pd.concat([history, block], ignore_index=True)

        return {q: v.to_numpy(dtype=float) for q, v in out.items()}

    def predict(self, X: pd.DataFrame, y: pd.Series | None = None) -> np.ndarray:
        return self.predict_quantiles(X, [0.5], y=y)[0.5]

    def report(self) -> dict:
        return {
            "name": self.name,
            "input_size": self.input_size,
            "max_steps": self.max_steps,
            "futr_exog": self.futr_exog,
            "train_rows": None if self._train_df is None else len(self._train_df),
        }


def nhits() -> NHiTSForecaster:
    return NHiTSForecaster()
