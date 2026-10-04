"""The measurement object every length in plan.json is built from."""

import pytest

from cozmo_scan.geometry import meas

REQUIRED = ("value", "sigma", "ci95_low", "ci95_high")


def test_meas_has_the_contract_fields():
    got = meas(2.4, 0.01, 0.012)
    assert set(REQUIRED) <= set(got)


def test_meas_interval_contains_the_value():
    got = meas(7.097, 0.05, 0.012)
    assert got["sigma"] >= 0
    assert got["ci95_low"] <= got["value"] <= got["ci95_high"]


def test_meas_floor_is_at_least_as_wide_as_the_tier_floor():
    # A tiny sigma must not publish a millimetre interval on the lidar floor.
    got = meas(2.4, sigma=0.0, abs_floor=0.012, rel_floor=0.0)
    assert got["ci95_high"] - got["ci95_low"] >= 0.024 - 1e-6


def test_meas_relative_floor_widens_a_thin_tier():
    got = meas(4.0, sigma=0.0, abs_floor=0.0, rel_floor=0.08)
    low_side = got["value"] - got["ci95_low"]
    high_side = got["ci95_high"] - got["value"]
    assert low_side == pytest.approx(high_side, abs=1e-9)
    assert (got["ci95_high"] - got["ci95_low"]) == pytest.approx(0.64, abs=1e-6)
