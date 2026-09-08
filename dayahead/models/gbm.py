"""
Gradient boosting, one model per delivery hour.

Same framing as the classical ladder in section 18, so the comparison in
expectation 12.4 is between model classes rather than between data
arrangements.

Quantiles are native
--------------------

Every other model so far derives its intervals from an unconditional spread
of training residuals, which section 17 showed is badly miscalibrated: the
same width is applied at every hour regardless of how uncertain the hour
actually is. Section 3b established that price variance is U-shaped in
forecast residual load, 21.67 EUR/MWh mid-range against 37.62 and 65.66 in
the tails.

LightGBM fits a separate model per quantile with the pinball objective, so
its intervals are conditional on the features. This is the first model in the
ladder that can widen where uncertainty is genuinely larger, and it is the
reason section 8 matters for section 10 as well as for section 11.

The point forecast is the median quantile rather than a separately fitted
model. Minimising pinball loss at alpha=0.5 is minimising absolute error, so
a separate L1 model would be the same estimator fitted twice.

Quantile crossing
-----------------

Nine independently fitted quantile models are not guaranteed to be ordered:
the q30 model can predict above the q40 model for a given row. Left alone
this produces negative interval widths and meaningless coverage. Predictions
are therefore sorted across quantiles per row, which is the standard
rearrangement fix and cannot worsen pinball loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluation.splits import QUANTILES
from .base import Forecaster

# Small and fixed. No search was run: with 216 models per fold across 59
# folds, any tuning loop would be expensive, and a search whose budget is
# chosen to fit the compute available is not an honest search. These are
# conventional defaults for tabular data of this size.
PARAMS = dict(
    objective="quantile",
    num_leaves=31,
    learning_rate=0.05,
    min_child_samples=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    verbose=-1,
    n_jobs=-1,
)
N_ESTIMATORS = 400
EARLY_STOPPING = 40

# Share of the training window held out, chronologically, for early stopping.
# Inside the training window only. The test fold is never seen.
VALID_FRACTION = 0.15


class PerHourLightGBM(Forecaster):
    """One gradient boosting model per delivery hour per quantile."""

    name = "LightGBM"
    native_quantiles = True

    def __init__(self, name: str = "LightGBM", quantiles=QUANTILES,
                 n_estimators: int = N_ESTIMATORS, params: dict | None = None):
        self.name = name
        self.quantiles = list(quantiles)
        self.n_estimators = n_estimators
        self.params = {**PARAMS, **(params or {})}
        self._models: dict[tuple[int, float], object] = {}
        self._features: list[str] = []
        self._best_iters: list[int] = []
        self._crossings = 0

    # ------------------------------------------------------------ internals
    @staticmethod
    def _usable_features(X: pd.DataFrame) -> list[str]:
        """
        Drop columns that are constant within an hourly slice.

        cal_hour and the daily Fourier terms are constant once the data is
        split by delivery hour, so they carry no information here. Dropping
        them is cosmetic for a tree but keeps the reported feature count
        honest.
        """
        return [c for c in X.columns if X[c].nunique(dropna=True) > 1]

    def _split_valid(self, n: int) -> int:
        return max(1, int(n * (1 - VALID_FRACTION)))

    # ----------------------------------------------------------------- api
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PerHourLightGBM":
        import lightgbm as lgb

        self._models, self._best_iters = {}, []
        self._features = self._usable_features(X)

        for hour in range(24):
            mask = X.index.hour == hour
            Xh, yh = X.loc[mask, self._features], y.loc[mask]
            ok = Xh.notna().all(axis=1) & yh.notna()
            Xh, yh = Xh[ok], yh[ok]
            if len(Xh) < 100:
                continue

            cut = self._split_valid(len(Xh))
            Xtr, ytr = Xh.iloc[:cut], yh.iloc[:cut]
            Xva, yva = Xh.iloc[cut:], yh.iloc[cut:]

            for q in self.quantiles:
                model = lgb.LGBMRegressor(alpha=q, n_estimators=self.n_estimators,
                                          **self.params)
                model.fit(
                    Xtr, ytr,
                    eval_set=[(Xva, yva)],
                    callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
                )
                self._models[(hour, q)] = model
                self._best_iters.append(model.best_iteration_ or self.n_estimators)

        if not self._models:
            raise RuntimeError(f"{self.name}: no hourly model could be fitted")
        return self

    def _predict_raw(self, X: pd.DataFrame) -> dict[float, np.ndarray]:
        out = {q: np.full(len(X), np.nan) for q in self.quantiles}
        pos = {ts: i for i, ts in enumerate(X.index)}

        for hour in range(24):
            mask = X.index.hour == hour
            if not mask.any():
                continue
            Xh = X.loc[mask, self._features]
            idx = np.array([pos[t] for t in X.index[mask]])
            for q in self.quantiles:
                model = self._models.get((hour, q))
                if model is None:
                    continue
                out[q][idx] = model.predict(Xh)
        return out

    @staticmethod
    def _rearrange(preds: dict[float, np.ndarray],
                   quantiles: list[float]) -> tuple[dict[float, np.ndarray], int]:
        """Sort predictions across quantiles per row, and count crossings."""
        stack = np.column_stack([preds[q] for q in quantiles])
        crossings = int(np.sum(np.diff(stack, axis=1) < 0))
        stack.sort(axis=1)
        return {q: stack[:, i] for i, q in enumerate(quantiles)}, crossings

    def predict_quantiles(self, X: pd.DataFrame,
                          quantiles=None) -> dict[float, np.ndarray]:
        qs = list(quantiles or self.quantiles)
        raw = self._predict_raw(X)
        fixed, crossings = self._rearrange(raw, self.quantiles)
        self._crossings += crossings
        return {q: fixed[q] for q in qs if q in fixed}

    def predict(self, X: pd.DataFrame, y: pd.Series | None = None) -> np.ndarray:
        """Median quantile. Minimising pinball at 0.5 is minimising L1."""
        return self.predict_quantiles(X, [0.5])[0.5]

    def report(self) -> dict:
        return {
            "name": self.name,
            "models": len(self._models),
            "features": len(self._features),
            "mean_best_iteration": (float(np.mean(self._best_iters))
                                    if self._best_iters else None),
            "quantile_crossings": self._crossings,
        }


def lightgbm() -> PerHourLightGBM:
    return PerHourLightGBM()
