"""Two captures. Even/odd frames are not this function."""

from cozmo_scan.benchmark_eval import REPEATABILITY_ABS_M, REPEATABILITY_REL
from cozmo_scan.repeatability import compare_two_captures


def _plan(length, wall_id="w0", room_id="room_01"):
    return {
        "property": {
            "rooms": [
                {
                    "id": room_id,
                    "walls": [{"id": wall_id, "length_m": length}],
                }
            ]
        }
    }


def test_one_centimetre_passes():
    report = compare_two_captures(_plan(3.0), _plan(3.0 + REPEATABILITY_ABS_M), "room_01")
    assert report["status"] == "PASS"
    assert report["proxy_experiment"] is False
    assert report["rows"][0]["pass"] is True
    assert report["threshold"]["absolute_m"] == REPEATABILITY_ABS_M
    assert report["threshold"]["relative"] == REPEATABILITY_REL


def test_short_wall_over_one_centimetre_fails():
    report = compare_two_captures(_plan(2.0), _plan(2.012), "room_01")
    assert report["status"] == "FAIL"
    row = report["rows"][0]
    assert row["absolute_difference"] > 0.01
    assert row["relative_difference"] > 0.005


def test_unmatched_wall_fails():
    report = compare_two_captures(_plan(3.0, "w0"), _plan(3.0, "w1"), "room_01")
    assert report["status"] == "FAIL"
    assert any(row["matched"] is False for row in report["rows"])


def test_missing_capture_b_is_blocked():
    report = compare_two_captures(_plan(3.0), None, "room_01")
    assert report["status"] == "BLOCKED"
    assert report["reason"] == "missing_capture_b"
