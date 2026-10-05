"""Incumbent comparison. A tie counts. A missing export stays BLOCKED."""

from cozmo_scan.assessment import score_incumbent
from cozmo_scan.benchmark import incumbent_bundle


def _row(ours, theirs, truth=4.0, name="w0"):
    return {
        "dimension": name,
        "ours_error_m": abs(ours - truth),
        "incumbent_error_m": abs(theirs - truth),
        "ground_truth_m": truth,
    }


def test_tie_counts_as_a_beat():
    report = score_incumbent([_row(4.0, 4.0)])
    assert report["status"] == "PASS"
    assert report["tied"] == 1
    assert report["beat_or_tie_rate"] == 1.0


def test_incumbent_win_for_us_passes():
    report = score_incumbent([_row(4.01, 4.05), _row(3.0, 3.04, truth=3.0, name="w1")])
    assert report["status"] == "PASS"
    assert report["won"] == 2


def test_incumbent_loss_fails_the_seventy_percent_bar():
    report = score_incumbent([_row(4.05, 4.01), _row(3.04, 3.0, truth=3.0, name="w1")])
    assert report["status"] == "FAIL"
    assert report["won"] == 0


def test_missing_export_is_blocked():
    report = score_incumbent([])
    assert report["status"] == "BLOCKED"


def test_polycam_rooms_normalize_without_inventing_a_length():
    doc = incumbent_bundle(
        {
            "app": "polycam",
            "version": "1.2",
            "property_id": "home_a",
            "rooms": [{"room_id": "room_01", "walls": [{"wall_id": "w0", "length_m": 4.1}, {"wall_id": "w1"}]}],
        }
    )
    assert doc["app"] == "polycam"
    assert doc["rooms"][0]["walls"] == [{"wall_id": "w0", "length_m": 4.1}]
    assert doc["dimensions"][0]["incumbent_m"] == 4.1
    assert "ground_truth_m" not in doc["dimensions"][0]
