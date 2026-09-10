"""
Battery dispatch and settlement.

The valuation is the project's headline business number, so the arithmetic
behind it is tested against hand-computed cases rather than against itself.
"""

import numpy as np
import pandas as pd
import pytest

from dayahead.value.battery import (
    CHARGE_HOURS, DISCHARGE_HOURS, ROUND_TRIP_EFFICIENCY,
    dispatch_day, settle,
)


def test_specification_matches_the_design():
    """Section 25 fixed these before any figure was computed."""
    assert ROUND_TRIP_EFFICIENCY == 0.85
    assert CHARGE_HOURS == 2
    assert DISCHARGE_HOURS == 2


def test_charging_precedes_discharging():
    """
    The battery starts empty. A schedule that discharges before it has
    charged would book revenue from energy it never held.
    """
    prices = np.array([100.0, 90.0, 80.0, 70.0] + [10.0] * 20)
    charge, discharge = dispatch_day(prices)
    assert charge.max() < discharge.min()


def test_it_buys_low_and_sells_high():
    prices = np.array([10.0, 12.0] + [50.0] * 20 + [200.0, 210.0])
    charge, discharge = dispatch_day(prices)
    assert set(charge) == {0, 1}
    assert set(discharge) == {22, 23}


def test_settlement_applies_the_round_trip_loss():
    realised = np.zeros(24)
    realised[[0, 1]] = 10.0     # bought 2 MWh at 10
    realised[[22, 23]] = 100.0  # sold at 100
    revenue = settle(realised, np.array([0, 1]), np.array([22, 23]))
    # 0.85 * 200 - 20
    assert revenue == pytest.approx(150.0)


def test_settlement_can_lose_money():
    """
    A schedule chosen on a wrong forecast settles at realised prices and can
    lose. A valuation that could not report a loss would be measuring the
    forecast's confidence rather than its accuracy.
    """
    realised = np.zeros(24)
    realised[[0, 1]] = 100.0
    realised[[22, 23]] = 10.0
    revenue = settle(realised, np.array([0, 1]), np.array([22, 23]))
    assert revenue < 0


def test_a_flat_day_earns_nothing_after_losses():
    """With no spread, the round trip loss makes the cycle unprofitable."""
    flat = np.full(24, 50.0)
    charge, discharge = dispatch_day(flat)
    assert settle(flat, charge, discharge) < 0


def test_short_days_are_refused_not_guessed():
    assert dispatch_day(np.arange(3, dtype=float)) is None


def test_dispatch_beats_a_naive_split_when_prices_favour_it():
    """
    The optimiser searches every admissible split rather than assuming the
    cheapest hours come first. This day rewards charging in the middle.
    """
    prices = np.array([80.0] * 8 + [5.0, 6.0] + [300.0] * 14)
    charge, discharge = dispatch_day(prices)
    assert set(charge) == {8, 9}
    assert settle(prices, charge, discharge) > 0


def test_perfect_foresight_is_an_upper_bound():
    """
    Dispatching on the realised prices must never earn less than dispatching
    on any forecast of them.
    """
    rng = np.random.default_rng(0)
    for _ in range(50):
        realised = rng.normal(60, 30, 24)
        forecast = realised + rng.normal(0, 20, 24)
        oracle = settle(realised, *dispatch_day(realised))
        guess = settle(realised, *dispatch_day(forecast))
        assert oracle >= guess - 1e-9
