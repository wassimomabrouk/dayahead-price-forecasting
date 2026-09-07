"""
Gate closure. This is the project's central claim, so the test suite for it
is written before the features it will police.

A forecast for delivery day D is issued at 12:00 local on D-1. A feature is
admissible only if it was published at or before that instant.

Section 4 fills these in against the real feature matrix. The placeholders
are deliberate: a test that passes because it checks nothing is worse than
no test at all, so each one fails loudly until it is implemented.
"""

import pytest

from dayahead import config as cfg


def test_forbidden_series_are_registered_as_forbidden():
    assert set(cfg.FORBIDDEN) == {"load_real", "residual_real"}


def test_no_forbidden_series_reaches_the_feature_set():
    # Once dayahead.features exists, this asserts the built matrix shares no
    # column with cfg.FORBIDDEN.
    pytest.skip("implement in section 4 alongside dayahead.features")


def test_every_feature_carries_an_availability_timestamp():
    pytest.skip("implement in section 4 alongside dayahead.features")


def test_availability_never_exceeds_issuance():
    pytest.skip("implement in section 4 alongside dayahead.features")


def test_a_deliberately_leaky_feature_is_rejected():
    # Proves the check has teeth: injecting a known-bad column must fail.
    pytest.skip("implement in section 4 alongside dayahead.features")
