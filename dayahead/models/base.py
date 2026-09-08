"""
The forecaster interface.

Every model in the ladder implements this and nothing else, so the backtest
harness never branches on model type. Adding N-HiTS at section 9 should be a
matter of writing one subclass, not touching the harness.

Quantiles are part of the interface rather than an optional extra. A model
that cannot express its own uncertainty cannot be scored on pinball loss, and
DESIGN.md section 11 selects the champion on pinball loss. `ResidualQuantile
Mixin` gives any point forecaster a defensible empirical quantile spread, so
even the naive baselines compete fairly on the probabilistic metrics rather
than being penalised for a missing capability.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ..evaluation.splits import QUANTILES


class Forecaster(ABC):
    """Fit on a training window, predict a future window."""

    name: str = "unnamed"
    #: Set by subclasses that produce genuinely conditional quantiles.
    native_quantiles: bool = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Forecaster":
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def predict_quantiles(self, X: pd.DataFrame,
                          quantiles=QUANTILES) -> dict[float, np.ndarray]:
        raise NotImplementedError(
            f"{self.name} produces no quantiles. Mix in ResidualQuantileMixin "
            "or implement predict_quantiles."
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"


class ResidualQuantileMixin:
    """
    Quantiles from the empirical distribution of training residuals.

    Unconditional: the same spread is applied at every hour, so it cannot
    widen where uncertainty is genuinely larger. Section 3b showed the price
    standard deviation is U-shaped in residual load, 21.67 mid-range against
    37.62 and 65.66 at the tails, so this will be too narrow in the tails and
    too wide in the middle. That is the expected weakness and it is exactly
    what the conformal work in section 10 addresses.

    Stated here so the limitation travels with the code rather than being
    rediscovered from a bad coverage number.
    """

    _residual_quantiles: dict[float, float] | None = None

    def _fit_residual_quantiles(self, y_true, y_fitted, quantiles=QUANTILES):
        r = np.asarray(y_true, dtype=float) - np.asarray(y_fitted, dtype=float)
        r = r[np.isfinite(r)]
        self._residual_quantiles = (
            {q: float(np.quantile(r, q)) for q in quantiles}
            if len(r) else {q: 0.0 for q in quantiles}
        )

    def predict_quantiles(self, X: pd.DataFrame,
                          quantiles=QUANTILES) -> dict[float, np.ndarray]:
        if self._residual_quantiles is None:
            raise RuntimeError("fit before predict_quantiles")
        point = self.predict(X)
        return {q: point + self._residual_quantiles.get(q, 0.0) for q in quantiles}
