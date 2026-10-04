"""The benchmark harness. These cases check the arithmetic, not a tape."""

import json
from pathlib import Path

import pytest

from cozmo_scan.benchmark_eval import (
    OPENING_MAX_ERROR_M,
    EVALUATOR_LABEL,
    evaluate,
    evaluate_case,
    load_ground_truth,
    load_prediction,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "benchmarks" / "fixtures"


def _case(name: str) -> dict:
    return evaluate_case(FIXTURES / name)


def _row(report: dict, metric: str, kind: str | None = None) -> dict:
    rows = report[metric]["rows"]
    if kind is None:
        return rows[0]
    found = [row for row in rows if row["kind"] == kind]
    assert found, rows
    return found[0]


def test_exact_match_passes_the_calculator():
    report = _case("exact_match")
    assert report["assessment_gate"]["status"] == "blocked"
    assert report["evaluator_only"] is True
    wall = _row(report, "wall_accuracy", "wall")
    assert wall["predicted"] == 4.0
    assert wall["ground_truth"] == 4.0
    assert wall["absolute_error"] == 0.0
    assert wall["relative_error"] == 0.0
    assert wall["pass"] is True
    opening = _row(report, "opening_accuracy", "opening")
    assert opening["pass"] is True
    assert opening["threshold"]["absolute_m"] == OPENING_MAX_ERROR_M
    assert report["opening_accuracy"]["gate_pass"] is True
    ceiling = _row(report, "ceiling_accuracy", "ceiling")
    assert ceiling["pass"] is True
    assert report["photo_property"]["status"] == "not_applicable"


def test_known_wall_error_fails_the_video_gate():
    report = _case("wall_error")
    wall = _row(report, "wall_accuracy", "wall")
    assert wall["predicted"] == 4.0
    assert wall["ground_truth"] == 3.8
    assert wall["absolute_error"] == pytest.approx(0.2)
    assert wall["relative_error"] == pytest.approx(0.2 / 3.8, abs=1e-6)
    assert wall["pass"] is False
    assert report["wall_accuracy"]["worst_abs_error_m"] == pytest.approx(0.2)


def test_opening_error_miss_and_phantom():
    wrong = _case("opening_error")
    row = _row(wrong, "opening_accuracy", "opening")
    assert row["absolute_error"] == pytest.approx(0.05)
    assert row["pass"] is False
    assert wrong["opening_accuracy"]["gate_pass"] is False

    missed = _case("missed_opening")
    assert _row(missed, "opening_accuracy", "missed")["pass"] is False
    assert missed["opening_accuracy"]["n_missed"] == 1
    assert missed["opening_accuracy"]["n_phantom"] == 0

    phantom = _case("phantom_opening")
    assert _row(phantom, "opening_accuracy", "phantom")["predicted"] == 0.8
    assert _row(phantom, "opening_accuracy", "phantom")["ground_truth"] is None
    assert phantom["opening_accuracy"]["n_phantom"] == 1
    assert phantom["opening_accuracy"]["n_missed"] == 0


def test_ceiling_error_fails_1_5_cm():
    report = _case("ceiling_error")
    row = _row(report, "ceiling_accuracy", "ceiling")
    assert row["predicted"] == 2.53
    assert row["ground_truth"] == 2.5
    assert row["absolute_error"] == pytest.approx(0.03)
    assert row["pass"] is False


def test_repeatability_pass_and_fail():
    passed = _case("repeatability_pass")
    row = _row(passed, "repeatability", "repeat_wall")
    assert row["absolute_error"] == pytest.approx(0.004)
    assert row["ground_truth"] is None
    assert row["compared_m"] == 4.004
    assert row["pass"] is True

    failed = _case("repeatability_fail")
    row = _row(failed, "repeatability", "repeat_wall")
    assert row["absolute_error"] == pytest.approx(0.08)
    assert row["pass"] is False


def test_photo_footprint_and_adjacency():
    passed = _case("photo_footprint_pass")
    footprint = passed["photo_property"]["footprint"]
    assert footprint["rows"][0]["pass"] is True
    assert footprint["rows"][0]["relative_error"] == pytest.approx(0.5 / 12.5)
    assert passed["photo_property"]["adjacency"]["pass"] is True
    assert passed["photo_property"]["overlap"]["pass"] is True

    failed = _case("photo_footprint_fail")
    assert failed["photo_property"]["footprint"]["rows"][0]["pass"] is False

    correct = _case("adjacency_correct")
    assert correct["photo_property"]["adjacency"]["pass"] is True
    wrong = _case("adjacency_wrong")
    assert wrong["photo_property"]["adjacency"]["pass"] is False
    assert wrong["photo_property"]["adjacency"]["n_missed"] == 1


def test_malformed_ground_truth_is_rejected():
    with pytest.raises(ValueError, match="rooms"):
        load_ground_truth({"case_id": "bad"})
    with pytest.raises(ValueError, match="length_m"):
        load_ground_truth({"rooms": [{"room_id": "room_0", "walls": [{}]}]})
    with pytest.raises(ValueError, match="wall_id"):
        load_ground_truth({"rooms": [{"room_id": "room_0", "openings": [{"width_m": 0.8}]}]})


def test_missing_ground_truth_is_blocked_and_does_not_invent_scores():
    prediction = load_prediction(json.loads((FIXTURES / "exact_match" / "prediction.json").read_text(encoding="utf-8")))
    report = evaluate(prediction, None, case_id="no_tape")
    assert report["assessment_gate"]["status"] == "blocked"
    assert report["wall_accuracy"]["status"] == "unavailable"
    assert report["opening_accuracy"]["rows"] == []
    assert report["ceiling_accuracy"]["status"] == "unavailable"
    assert "assessment_gate" in report["blocked_metrics"]
    assert report["calibration"]["status"] == "unavailable"


def test_geometric_wall_match_and_room_map():
    prediction = {
        "tier": "video",
        "capture_id": "geom",
        "property": {
            "rooms": [
                {
                    "id": "capture_a",
                    "room_id": "capture_a",
                    "walls": [
                        {
                            "id": "capture_a_w0",
                            "length_m": {"value": 4.0, "sigma": 0.01, "ci95_low": 3.98, "ci95_high": 4.02},
                            "p0_m": [0.0, 0.0],
                            "p1_m": [4.0, 0.0],
                            "openings": [],
                        }
                    ],
                    "ceiling_height_m": None,
                }
            ],
            "adjacencies": [],
            "footprint_area_m2": None,
        },
    }
    ground_truth = {
        "case_id": "geom",
        "tier": "video",
        "evaluator_only": True,
        "provenance": "synthetic_evaluator",
        "room_map": {"capture_a": "room_0"},
        "rooms": [
            {
                "room_id": "room_0",
                "walls": [{"length_m": 4.0, "p0_m": [4.0, 0.0], "p1_m": [0.0, 0.0]}],
                "openings": [],
            }
        ],
    }
    report = evaluate(prediction, ground_truth)
    assert report["rooms"][0]["status"] == "matched"
    assert report["rooms"][0]["ground_truth_room_id"] == "room_0"
    wall = _row(report, "wall_accuracy", "wall")
    assert wall["absolute_error"] == 0.0
    assert wall["pass"] is True


def test_report_is_deterministic_and_reads_plan_json(tmp_path):
    folder = FIXTURES / "exact_match"
    prediction = load_prediction(json.loads((folder / "prediction.json").read_text(encoding="utf-8")))
    ground_truth = load_ground_truth(json.loads((folder / "ground_truth.json").read_text(encoding="utf-8")))
    assert EVALUATOR_LABEL == ground_truth["label"]
    first = evaluate(prediction, ground_truth)
    second = evaluate(prediction, ground_truth)
    assert first == second
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(prediction), encoding="utf-8")
    loaded = load_prediction(json.loads(plan_path.read_text(encoding="utf-8")))
    assert evaluate(loaded, ground_truth) == first


def test_drift_comparison_uses_a_second_plan_when_one_is_given():
    base = json.loads((FIXTURES / "repeatability_pass" / "prediction.json").read_text(encoding="utf-8"))
    other = json.loads((FIXTURES / "repeatability_fail" / "prediction_repeat.json").read_text(encoding="utf-8"))
    report = evaluate(base, None, drift_off=other, case_id="drift")
    assert report["drift"]["status"] == "compared"
    assert report["drift"]["rows"][0]["pass"] is False
    untouched = evaluate(base, None, case_id="drift")
    assert untouched["drift"]["wall_comparison"] == "unavailable"
    assert untouched["drift"]["evidence"]["method"]
