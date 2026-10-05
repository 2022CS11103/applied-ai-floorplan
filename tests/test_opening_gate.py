"""Opening gate. The 2 cm and 85% bars are the PDF constants."""

from cozmo_scan.benchmark import score_opening_gate
from cozmo_scan.benchmark_eval import OPENING_MAX_ERROR_M, OPENING_PASS_RATE


def _gt(width=0.80, opening_id="door_a"):
    return [{"opening_id": opening_id, "room_id": "room_01", "wall_id": "w0", "width_m": width}]


def _pred(width=0.80, opening_id="door_a"):
    return [{"opening_id": opening_id, "room_id": "room_01", "wall_id": "w0", "width_m": width}]


def test_exact_match_passes_and_the_threshold_is_unchanged():
    report = score_opening_gate(_gt(), _pred(), "tape")
    assert report["status"] == "PASS"
    assert report["accuracy"] == 1.0
    assert report["threshold"] == OPENING_PASS_RATE == 0.85
    assert report["max_error_m"] == OPENING_MAX_ERROR_M == 0.02
    assert report["within_2cm"] == 1


def test_two_centimetre_miss_fails():
    report = score_opening_gate(_gt(0.80), _pred(0.85), "laser")
    assert report["status"] == "FAIL"
    assert report["within_2cm"] == 0


def test_missed_opening_is_a_failure():
    report = score_opening_gate(_gt(), [], "tape")
    assert report["status"] == "FAIL"
    assert report["missed"] == 1
    assert report["matched"] == 0


def test_phantom_opening_is_a_failure():
    report = score_opening_gate(_gt(), _pred() + _pred(0.70, "extra"), "tape")
    assert report["status"] == "FAIL"
    assert report["phantom"] == 1


def test_missing_ground_truth_is_blocked():
    report = score_opening_gate([], _pred(), "tape")
    assert report["status"] == "BLOCKED"
    assert report["reason"] == "missing_ground_truth"


def test_faro_match_stays_blocked():
    report = score_opening_gate(_gt(), _pred(), "faro")
    assert report["status"] == "BLOCKED"
