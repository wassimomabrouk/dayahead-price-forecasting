"""Validation helpers, tested on synthetic panels rather than the real store."""

import pandas as pd

from dayahead.data.validate import interior_nulls, null_runs


def _series(values):
    idx = pd.date_range("2020-01-01", periods=len(values), freq="h", tz="UTC")
    return pd.Series(values, index=idx, dtype="Float64")


def test_trailing_nulls_are_not_interior():
    s = _series([1.0, 2.0, 3.0, None, None])
    assert len(interior_nulls(s)) == 0


def test_leading_nulls_are_not_interior():
    s = _series([None, None, 1.0, 2.0])
    assert len(interior_nulls(s)) == 0


def test_interior_nulls_are_found():
    s = _series([1.0, None, None, 4.0, None])
    assert len(interior_nulls(s)) == 2


def test_null_runs_groups_consecutive_hours():
    s = _series([1.0, None, None, 4.0, None, 6.0])
    runs = null_runs(interior_nulls(s))
    assert len(runs) == 2
    assert (runs[0][1] - runs[0][0]) == pd.Timedelta(hours=1)


def test_null_runs_on_empty_index():
    s = _series([1.0, 2.0])
    assert null_runs(interior_nulls(s)) == []
