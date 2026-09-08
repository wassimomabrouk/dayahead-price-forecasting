"""
Classical models: ARIMA, SARIMA, SARIMAX.

Fitted per delivery hour, per DESIGN.md section 18. For each of the 24
delivery hours a separate series is formed at daily frequency, so one step
ahead is exactly one day ahead, which is exactly the forecast horizon.

Conditioning, and why it is not multi-step forecasting
-----------------------------------------------------

A fold retrains parameters once per month, but the information set updates
every day. On day D the forecaster knows prices through D-1, because those
cleared at the auction on D-2.

So the model is fitted on the training window, then `append(refit=False)`
extends it with the test observations and one-step-ahead in-sample
predictions are read off. Each prediction conditions on everything up to the
previous day and nothing later. Parameters never see test data; only the
conditioning state advances.

Forecasting 30 steps ahead from the fold boundary would be a different and
much harder problem, and it would not be the one section 4 defined. It would
also make the comparison against LightGBM meaningless, since LightGBM's
price features update daily.

The three specifications
------------------------

    ARIMA     price history only, no seasonal term, no regressors
    SARIMA    adds weekly seasonality and deterministic calendar terms
    SARIMAX   adds the published day-ahead forecasts

The gap from SARIMA to SARIMAX is the measured value of the wind, solar and
load information, which is the headline of section 7.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .base import Forecaster, ResidualQuantileMixin

# Deterministic regressors. The weekly cycle is carried by the seasonal term,
# so these cover the annual cycle and the working-day effect.
CALENDAR_EXOG = [
    "cal_is_nonworking", "cal_is_holiday",
    "cal_sin_year_1", "cal_cos_year_1",
    "cal_sin_year_2", "cal_cos_year_2",
]

# Fundamentals. One side of the residual identity only, per DESIGN.md
# section 3: residual load, solar and total wind are mutually independent,
# whereas adding load as well would make the design matrix singular.
FUNDAMENTAL_EXOG = ["x_fc_residual", "x_fc_solar", "x_wind_total"]

# Order chosen after a failure, not from a grid search.
#
# The first implementation used (2,0,1) with enforce_stationarity=False.
# ARIMA and SARIMAX behaved, but SARIMA diverged catastrophically: MAE
# 1.4e10 on a single fold. Without differencing, and with the stationarity
# constraint switched off, the fitted AR polynomial had roots inside the
# unit circle, and the state recursion then exploded once the conditioning
# state was extended over the test window. SARIMAX escaped it because the
# fundamentals regressors anchor the level; SARIMA had nothing to hold it.
#
# Differencing removes the near unit root that caused it, and enforcing
# stationarity prevents the explosive region from being reachable at all.
# Measured on the same fold: (2,0,1) enforced 19.32, (1,1,1) unenforced
# 17.22, (1,1,1) enforced 17.14.
DEFAULT_ORDER = (1, 1, 1)
DEFAULT_SEASONAL = (1, 0, 1, 7)
ENFORCE = True
MAXITER = 30

# Any prediction beyond this is a numerical failure, not a forecast. The
# observed price range across the whole sample is -500 to 936 EUR/MWh.
DIVERGENCE_LIMIT = 10_000


class PerHourSARIMAX(ResidualQuantileMixin, Forecaster):
    """One statsmodels state space model per delivery hour."""

    def __init__(self, name: str, order=DEFAULT_ORDER,
                 seasonal_order=(0, 0, 0, 0), exog_cols: list[str] | None = None,
                 description: str = ""):
        self.name = name
        self.order = order
        self.seasonal_order = seasonal_order
        self.exog_cols = list(exog_cols or [])
        self.description = description
        self._diverged = 0
        self._results: dict[int, object] = {}
        self._train: dict[int, pd.DataFrame] = {}
        self._failed: list[int] = []

    # ------------------------------------------------------------ internals
    @staticmethod
    def _hour_frame(X: pd.DataFrame, y: pd.Series | None, hour: int,
                    cols: list[str]) -> pd.DataFrame:
        """
        The daily series for one delivery hour, with any incomplete rows
        dropped and a positional index.

        A positional index is used rather than a daily DatetimeIndex because
        the hour-02 series is missing one day a year at the spring-forward
        transition. Asserting a daily frequency on a series with a hole would
        either fail or silently interpolate; dropping the row and counting
        positions does neither.
        """
        mask = X.index.hour == hour
        out = X.loc[mask, cols].copy() if cols else pd.DataFrame(index=X.index[mask])
        out["_y"] = y.loc[mask].to_numpy(dtype=float) if y is not None else np.nan
        out["_date"] = X.index[mask]
        out = out.replace([np.inf, -np.inf], np.nan)
        check = cols + (["_y"] if y is not None else [])
        return out.dropna(subset=check).reset_index(drop=True)

    def _fit_one(self, frame: pd.DataFrame):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        exog = frame[self.exog_cols].to_numpy(float) if self.exog_cols else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(frame["_y"].to_numpy(float), exog=exog,
                            order=self.order, seasonal_order=self.seasonal_order,
                            enforce_stationarity=ENFORCE,
                            enforce_invertibility=ENFORCE)
            return model.fit(disp=False, maxiter=MAXITER)

    # ----------------------------------------------------------------- api
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PerHourSARIMAX":
        self._results, self._train, self._failed = {}, {}, []
        fitted_all, actual_all = [], []

        for hour in range(24):
            frame = self._hour_frame(X, y, hour, self.exog_cols)
            if len(frame) < 60:
                self._failed.append(hour)
                continue
            try:
                res = self._fit_one(frame)
            except Exception:
                self._failed.append(hour)
                continue
            self._results[hour] = res
            self._train[hour] = frame
            fitted_all.append(np.asarray(res.fittedvalues, dtype=float))
            actual_all.append(frame["_y"].to_numpy(float))

        if not self._results:
            raise RuntimeError(f"{self.name}: every hourly model failed to fit")

        self._fit_residual_quantiles(np.concatenate(actual_all),
                                     np.concatenate(fitted_all))
        return self

    def predict(self, X: pd.DataFrame, y: pd.Series | None = None) -> np.ndarray:
        out = pd.Series(np.nan, index=X.index, dtype=float)

        for hour, res in self._results.items():
            test = self._hour_frame(X, y, hour, self.exog_cols)
            if test.empty:
                continue
            n_train = len(self._train[hour])
            exog = test[self.exog_cols].to_numpy(float) if self.exog_cols else None

            # Extend the conditioning state with the test window without
            # re-estimating parameters, then read one-step-ahead predictions.
            # The conditioning state must advance with the realised prices,
            # which is what makes each prediction one day ahead rather than a
            # month-long extrapolation. Appending NaN instead leaves the
            # filter with no new information and silently turns the model into
            # a multi-step forecaster: measured bias +31.9 EUR/MWh for ARIMA
            # and an MAE of 184.5 on the fold following the 2022 price peak.
            #
            # This is not leakage. get_prediction with dynamic=False returns
            # the one-step-ahead prediction for position t, which conditions
            # on observations through t-1 only. In the per-hour framing t-1 is
            # the previous day at the same hour, cleared at the auction two
            # days before delivery and therefore known at gate closure.
            endog = (test["_y"].to_numpy(float) if "_y" in test
                     else np.full(len(test), np.nan))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    extended = res.append(endog, exog=exog, refit=False)
                    pred = extended.get_prediction(
                        start=n_train, end=n_train + len(test) - 1,
                        dynamic=False).predicted_mean
            except Exception:
                pred = np.full(len(test), np.nan)

            pred = np.asarray(pred, dtype=float)
            # A diverged state recursion produces values orders of magnitude
            # outside any plausible price. Emitting them would poison the
            # aggregate metrics silently, so they are dropped and counted.
            bad = ~np.isfinite(pred) | (np.abs(pred) > DIVERGENCE_LIMIT)
            if bad.any():
                self._diverged += int(bad.sum())
                pred[bad] = np.nan
            out.loc[test["_date"].to_numpy()] = pred

        return out.to_numpy(dtype=float)

    def report(self) -> dict:
        return {
            "name": self.name,
            "hours_fitted": len(self._results),
            "hours_failed": sorted(self._failed),
            "n_exog": len(self.exog_cols),
            "diverged_predictions": self._diverged,
        }


# ------------------------------------------------------------ the three fits
def arima() -> PerHourSARIMAX:
    """7a. Price history alone. A diagnostic, not a competitor."""
    return PerHourSARIMAX(
        name="ARIMA",
        seasonal_order=(0, 0, 0, 0),
        exog_cols=[],
        description="price history only, no seasonal term, no regressors",
    )


def sarima() -> PerHourSARIMAX:
    """7b. Adds weekly seasonality and deterministic calendar structure."""
    return PerHourSARIMAX(
        name="SARIMA",
        seasonal_order=DEFAULT_SEASONAL,
        exog_cols=CALENDAR_EXOG,
        description="weekly seasonal term plus calendar regressors",
    )


def sarimax() -> PerHourSARIMAX:
    """7c. Adds the published day-ahead fundamentals."""
    return PerHourSARIMAX(
        name="SARIMAX",
        seasonal_order=DEFAULT_SEASONAL,
        exog_cols=CALENDAR_EXOG + FUNDAMENTAL_EXOG,
        description="seasonal plus calendar plus day-ahead fundamentals",
    )


def ladder() -> list[PerHourSARIMAX]:
    return [arima(), sarima(), sarimax()]
