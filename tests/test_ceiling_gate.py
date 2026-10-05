"""Ceiling gate. A FARO number that matches the prediction is still BLOCKED."""

from cozmo_scan.benchmark import score_ceiling_gate
from cozmo_scan.benchmark_eval import CEILING_MAX_ERROR_M, CEILING_REPEAT_SPREAD_M


def _room(error=0.0, spread=0.0, predicted=True):
    truth = 2.50
    return {
        "room_id": "room_01",
        "ground_truth_m": truth,
        "predicted_m": None if not predicted else truth + error,
        "repeat_spread_m": spread,
    }


def test_within_15mm_and_10mm_spread_passes():
    report = score_ceiling_gate([_room(0.015, 0.010)], "tape")
    assert report["status"] == "PASS"
    assert report["threshold_m"] == CEILING_MAX_ERROR_M == 0.015
    assert report["repeat_spread_m"] == CEILING_REPEAT_SPREAD_M == 0.01


def test_bias_over_15mm_fails():
    report = score_ceiling_gate([_room(0.02, 0.0)], "laser")
    assert report["status"] == "FAIL"
    assert report["bias_m"] == 0.02


def test_spread_over_10mm_fails():
    report = score_ceiling_gate([_room(0.0, 0.011)], "tape")
    assert report["status"] == "FAIL"


def test_missing_ceiling_is_blocked():
    report = score_ceiling_gate([], "tape")
    assert report["status"] == "BLOCKED"
    assert report["reason"] == "missing_ceiling"


def test_faro_agreement_is_blocked():
    report = score_ceiling_gate([_room(0.0, 0.0)], "faro")
    assert report["status"] == "BLOCKED"
