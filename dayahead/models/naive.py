"""
Seasonal naive baselines, B1 and B2 from DESIGN.md section 5.

Both are column selectors. That is not a shortcut, it is a consequence of the
delivery-date framing in section 4: "the price at this hour one week ago" is
already an admissible feature, `price_d7_same_hour`, built and gate-checked
alongside everything else. The baselines borrow the same audited columns
every other model sees, so no separate leakage surface exists for them.

Both carry empirical residual quantiles, so they compete on pinball loss
rather than being scored as if they had no uncertainty at all. A baseline
that cannot be beaten on the probabilistic metrics is a meaningful result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Forecaster, ResidualQuantileMixin


class ColumnNaive(ResidualQuantileMixin, Forecaster):
    """Predict the target with an existing admissible column."""

    def __init__(self, column: str, name: str, description: str = ""):
        self.column = column
        self.name = name
        self.description = description

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ColumnNaive":
        if self.column not in X.columns:
            raise KeyError(f"{self.name}: column '{self.column}' not in matrix")
        self._fit_residual_quantiles(y, X[self.column])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.column].to_numpy(dtype=float)


def b1() -> ColumnNaive:
    """Primary denominator: same hour, one week earlier."""
    return ColumnNaive(
        column="price_d7_same_hour",
        name="B1_weekly_naive",
        description="price at this hour on D-7",
    )


def b2() -> ColumnNaive:
    """Secondary: same hour, previous day."""
    return ColumnNaive(
        column="price_d1_same_hour",
        name="B2_daily_naive",
        description="price at this hour on D-1",
    )


def all_baselines() -> list[ColumnNaive]:
    return [b1(), b2()]
