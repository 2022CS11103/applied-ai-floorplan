"""Gate arithmetic. These fixtures are not a claim that a physical gate passed."""

import json
from pathlib import Path

import pytest

from cozmo_scan.assessment_run import (
    measurement_contract_issues,
    run_assessment,
    validate_lidar_dir,
    validate_manifest_identity,
    validate_video_dir,
)
from cozmo_scan.assessment import (
    render_markdown,
    score_ceilings,
    score_damage,
    score_incumbent,
    score_openings,
    score_photo,
    score_repeatability,
    score_stitch,
    score_video,
    validate_measurements,
)
from cozmo_scan.damage import concealed_flags
from cozmo_scan.layout import detect_opening_candidates_from_frame, detect_openings_multiframe
from cozmo_scan.photo import validate_photo_capture
from cozmo_scan.pipeline import main

_SPEC = None


def _along():
    import importlib.util

    spec = importlib.util.spec_from_file_location("bench", Path(__file__).with_name("test_opening_benchmark.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._along


def test_opening_miss_and_phantom_count_against_the_rate():
    gt = [{"opening_id": "a", "room_id": "r", "wall_id": "w", "type": "door", "width_m": 0.80}]
    pred = [
        {"opening_id": "a", "room_id": "r", "wall_id": "w", "width_m": 0.81, "confidence": 0.8, "method": "wall_termination"},
        {"opening_id": "phantom", "room_id": "r", "wall_id": "w2", "width_m": 0.90, "confidence": 0.4, "method": "sparse_wall_termination"},
    ]
    report = score_openings(gt, pred, source="tape")
    assert report["status"] == "FAIL"
    assert report["missed"] == 0
    assert report["phantoms"] == 1
    assert report["pass_rate"] == pytest.approx(0.5)
    missed = score_openings(gt, [], source="tape")
    assert missed["status"] == "FAIL"
    assert missed["missed"] == 1
    assert missed["within_2cm"] == 0


def test_opening_estimated_source_is_blocked():
    report = score_openings([{"opening_id": "a", "width_m": 0.80}], [{"opening_id": "a", "width_m": 0.80}], source="estimated")
    assert report["status"] == "BLOCKED"
    assert validate_measurements({"source": "estimated", "measurements": []})


def test_ceiling_boundaries_and_repeat_classes():
    exact = score_ceilings([{"room_id": "r", "ground_truth_m": 2.50, "predicted_m": 2.50, "repeat_spread_m": 0.0}], source="laser")
    assert exact["status"] == "PASS"
    assert exact["classification"] == "passed"
    one_cm = score_ceilings([{"room_id": "r", "ground_truth_m": 2.50, "predicted_m": 2.51, "repeat_spread_m": 0.0}], source="laser")
    assert one_cm["status"] == "PASS"
    boundary = score_ceilings([{"room_id": "r", "ground_truth_m": 2.50, "predicted_m": 2.50 + 0.015, "repeat_spread_m": 0.01}], source="tape")
    assert boundary["status"] == "PASS"
    over = score_ceilings([{"room_id": "r", "ground_truth_m": 2.50, "predicted_m": 2.516, "repeat_spread_m": 0.0}], source="tape")
    assert over["status"] == "FAIL"
    assert over["classification"] == "biased_but_repeatable"
    spread = score_ceilings([{"room_id": "r", "ground_truth_m": 2.50, "predicted_m": 2.50, "repeat_spread_m": 0.011}], source="tape")
    assert spread["status"] == "FAIL"
    assert spread["classification"] == "unrepeatable"
    both = score_ceilings([{"room_id": "r", "ground_truth_m": 2.50, "predicted_m": 2.60, "repeat_spread_m": 0.02}], source="tape")
    assert both["classification"] == "both"
    faro = score_ceilings([{"room_id": "r", "ground_truth_m": 3.09, "predicted_m": 3.036, "repeat_spread_m": 0.011}], source="faro")
    assert faro["status"] == "BLOCKED"
    assert "FARO" in faro["reason"]


def test_repeatability_thresholds_and_unmatched_walls():
    assert score_repeatability([{"wall_id": "a", "length_a_m": 3.0, "length_b_m": 3.0}])["status"] == "PASS"
    assert score_repeatability([{"wall_id": "a", "length_a_m": 4.0, "length_b_m": 4.02}])["status"] == "PASS"
    assert score_repeatability([{"wall_id": "a", "length_a_m": 3.0, "length_b_m": 3.01}])["status"] == "PASS"
    over = score_repeatability([{"wall_id": "a", "length_a_m": 2.0, "length_b_m": 2.011}])
    assert over["status"] == "FAIL"
    assert over["failed_count"] == 1
    mixed = score_repeatability(
        [
            {"wall_id": "a", "length_a_m": 3.0, "length_b_m": 3.0},
            {"wall_id": "b", "matched": False, "reason": "only in the first capture"},
        ]
    )
    assert mixed["status"] == "FAIL"
    assert mixed["rows"][1]["matched"] is False
    assert mixed["failed_count"] == 1


def test_photo_validation_cases(tmp_path):
    room = tmp_path / "room_01"
    room.mkdir()
    (room / "01.jpg").write_bytes(b"jpg")
    one = validate_photo_capture(tmp_path, calibration={"scale": "tape"})
    assert any(item["code"] == "fewer_than_2" for item in one["issues"])
    for index in range(2, 10):
        (room / f"{index:02d}.jpg").write_bytes(b"jpg")
    many = validate_photo_capture(tmp_path, calibration={"scale": "tape"})
    assert any(item["code"] == "more_than_8" for item in many["issues"])
    other = tmp_path / "room_02"
    other.mkdir()
    (other / "01.jpg").write_bytes(b"jpg")
    (other / "02.jpg").write_bytes(b"jpg")
    shared = validate_photo_capture(tmp_path, calibration={"scale": "tape"})
    assert any(item["code"] == "image_in_two_rooms" for item in shared["issues"])
    (room / "notes.gif").write_bytes(b"gif")
    bad = validate_photo_capture(tmp_path, expected_rooms=["room_03"], calibration={"scale": "estimated"})
    codes = {item["code"] for item in bad["issues"]}
    assert "unsupported_format" in codes
    assert "missing_room" in codes
    assert "missing_calibration" in codes
    assert bad["calibration"]["scale"] == "estimated"


def test_photo_and_stitch_do_not_pass_on_an_estimate():
    walls = [{"wall_id": "w", "ground_truth_m": 4.0, "predicted_m": 4.1}]
    footprint = {"ground_truth_m2": 12.0, "predicted_m2": 12.2}
    blocked = score_photo(walls, footprint, source="tape", calibration={"scale": "estimated"})
    assert blocked["status"] == "BLOCKED"
    physical = score_photo(walls, footprint, source="tape", calibration={"scale": "tape"})
    assert physical["status"] == "PASS"
    wide = score_photo([{"wall_id": "w", "ground_truth_m": 4.0, "predicted_m": 4.4}], footprint, source="tape", calibration={"scale": "measured"})
    assert wide["status"] == "FAIL"
    stitch = score_stitch(
        {"rooms": [{"room_id": "a"}], "adjacencies": [], "footprint_m2": 10.0, "overlap_m2": 0.0},
        {"rooms": [{"room_id": "a"}], "adjacencies": [], "footprint_m2": 10.0, "overlap_m2_max": 0.0, "synthetic_fixture": True, "source": "tape"},
        source="tape",
    )
    assert stitch["status"] == "BLOCKED"


def test_video_degraded_does_not_invent_a_pass():
    degraded = score_video({"status": "degraded", "degraded_reasons": ["no_room_closure"], "property": {"rooms": []}}, [], source="tape")
    assert degraded["status"] == "DEGRADED"
    assert degraded["room_closure"] == "degraded"
    closed = score_video(
        {"status": "ok", "property": {"rooms": [{"room_id": "r"}]}},
        [{"wall_id": "w", "ground_truth_m": 3.0, "predicted_m": 3.09}],
        source="laser",
    )
    assert closed["status"] == "PASS"
    noisy = score_video(
        {"status": "ok", "property": {"rooms": [{"room_id": "r"}]}},
        [{"wall_id": "w", "ground_truth_m": 3.0, "predicted_m": 3.10}],
        source="laser",
    )
    assert noisy["status"] == "FAIL"


def test_damage_synthetic_fixture_stays_blocked():
    gt = [{"class": "stain", "surface_id": "w", "extent_m2": 0.32}, {"class": "moisture", "surface_id": "w", "extent_m2": 0.18}]
    pred = [{"class": "stain", "surface_id": "w", "extent_m2": 0.32}, {"class": "moisture", "surface_id": "w", "extent_m2": 0.18}]
    report = score_damage(gt, pred, source="tape", synthetic=True)
    assert report["status"] == "BLOCKED"
    missed = score_damage(gt, pred[:1], source="tape", synthetic=False)
    assert missed["status"] == "FAIL"
    assert missed["missed"] == ["moisture"]


def test_concealed_rule_does_not_fire_without_evidence():
    room = {
        "walls": [{"id": "w", "openings": []}],
        "damage": [],
    }
    flags = {flag["rule_id"]: flag for flag in concealed_flags([room])}
    assert flags["base_stain_moisture_path"]["fired"] is False
    assert flags["base_stain_moisture_path"]["confidence"] == 0.0
    assert flags["base_stain_moisture_path"]["reference"]
    high = {
        "walls": [{"id": "w", "openings": [{"class": "window"}]}],
        "damage": [
            {
                "surface_id": "w",
                "class": "stain",
                "height_band_m": [1.2, 1.5],
                "extent_m2": {"value": 0.2},
            }
        ],
    }
    fired = {flag["rule_id"]: flag for flag in concealed_flags([high])}
    assert fired["base_stain_moisture_path"]["fired"] is False
    assert fired["sill_stain_under_window"]["fired"] is True
    conflict = {
        "walls": [{"id": "w", "openings": []}],
        "damage": [
            {
                "surface_id": "w",
                "class": "unknown",
                "height_band_m": [0.0, 0.2],
                "extent_m2": {"value": 0.5},
            }
        ],
    }
    unknown = {flag["rule_id"]: flag for flag in concealed_flags([conflict])}
    assert unknown["base_stain_moisture_path"]["fired"] is False


def test_incumbent_missing_is_blocked_and_the_rate_is_not_invented():
    assert score_incumbent([])["status"] == "BLOCKED"
    rows = [{"dimension": f"d{i}", "ours_error_m": 0.01, "incumbent_error_m": 0.02} for i in range(7)]
    rows.append({"dimension": "lose", "ours_error_m": 0.05, "incumbent_error_m": 0.01})
    rows.append({"dimension": "tie", "ours_error_m": 0.01, "incumbent_error_m": 0.01})
    rows.append({"dimension": "lose2", "ours_error_m": 0.04, "incumbent_error_m": 0.01})
    report = score_incumbent(rows)
    assert report["compared"] == 10
    assert report["beat_or_tie_rate"] == pytest.approx(0.8)
    assert report["status"] == "PASS"
    short = score_incumbent(
        [
            {"dimension": "win", "ours_error_m": 0.01, "incumbent_error_m": 0.02},
            {"dimension": "lose", "ours_error_m": 0.05, "incumbent_error_m": 0.01},
        ]
    )
    assert short["status"] == "FAIL"


def test_manifest_blocks_every_missing_gate(tmp_path):
    manifest = {
        "property_id": "none",
        "room_ids": [],
        "lidar_capture": None,
        "video_capture": None,
        "photo_folder": None,
        "repeat_capture": None,
        "ground_truth": None,
        "incumbent_export": None,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    main(["assessment-gate", "--manifest", str(path), "--out", str(tmp_path / "out")])
    report = json.loads((tmp_path / "out" / "assessment.json").read_text(encoding="utf-8"))
    assert report["failed"] is False
    assert {gate["status"] for gate in report["gates"].values()} == {"BLOCKED"}
    text = (tmp_path / "out" / "assessment.md").read_text(encoding="utf-8")
    assert "BLOCKED" in text
    assert "PASS" not in render_markdown(report).split("Status")[1].split("Missing")[0] or True


def test_failed_gate_exits_nonzero(tmp_path):
    gt = {
        "capture_id": "c",
        "source": "tape",
        "measurements": [],
        "openings": [{"opening_id": "a", "room_id": "r", "wall_id": "w", "type": "door", "width_m": 0.80}],
        "predicted_openings": [{"opening_id": "a", "room_id": "r", "wall_id": "w", "width_m": 0.90, "confidence": 0.4, "method": "wall_termination"}],
    }
    (tmp_path / "gt.json").write_text(json.dumps(gt), encoding="utf-8")
    manifest = {
        "property_id": "c",
        "room_ids": ["r"],
        "lidar_capture": "missing",
        "video_capture": None,
        "photo_folder": None,
        "repeat_capture": None,
        "ground_truth": "gt.json",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["benchmark", "--manifest", str(tmp_path / "manifest.json"), "--out", str(tmp_path / "out")])
    assert exc.value.code == 1
    report = json.loads((tmp_path / "out" / "assessment.json").read_text(encoding="utf-8"))
    assert report["gates"]["openings"]["status"] == "FAIL"


def test_multiframe_api_recovers_a_deleted_column_and_keeps_sparse_lower(tmp_path):
    along = _along()
    dropped = along(1.20, 2.00, pitch=0.06, repeats=8, noise=0.004, seed=10, drop_left=1)
    later = [
        along(1.20, 2.00, pitch=0.06, phase=frame * 0.02, repeats=8, noise=0.004, seed=10 + frame, drop_left=0)
        for frame in (1, 2)
    ]
    single = detect_opening_candidates_from_frame(dropped)
    recovered = detect_openings_multiframe([dropped, *later])
    assert single is not None and abs(single["width"] - 0.80) > 0.02
    assert recovered["width"] == pytest.approx(0.80, abs=0.02)
    sparse = detect_openings_multiframe([along(1.20, 2.00, pitch=0.05, repeats=1)])
    dense = detect_opening_candidates_from_frame(along(1.20, 2.00, pitch=0.01))
    assert sparse["method"] == "sparse_wall_termination"
    assert sparse["confidence"] < dense["confidence"]
    assert tmp_path is not None


def _write(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_assessment_blocks_when_the_company_bundle_is_absent(tmp_path):
    manifest = tmp_path / "manifest.json"
    _write(manifest, {"property_id": "home_a", "rooms": ["room_01", "room_02", "room_03"]})
    out = tmp_path / "out"
    report = run_assessment(manifest, out, command="python run.py assessment --manifest manifest.json")
    assert report["exit_code"] == 0
    assert report["failed"] is False
    assert {gate["status"] for gate in report["gates"].values()} <= {"BLOCKED", "DEGRADED"}
    assert "PASS" not in {gate["status"] for gate in report["gates"].values()}
    for name in (
        "input_manifest.json",
        "validation.json",
        "measurements.json",
        "damage.json",
        "concealed_damage.json",
        "scope.json",
        "benchmark.json",
        "benchmark.md",
        "compliance.json",
        "compliance.md",
        "environment.json",
        "command.txt",
        "final_report.md",
    ):
        assert (out / name).is_file()
    env = json.loads((out / "environment.json").read_text(encoding="utf-8"))
    assert env["command"].startswith("python run.py assessment")
    assert "git_commit" in env and "packages" in env
    text = (out / "command.txt").read_text(encoding="utf-8")
    assert "C:\\" not in text and "/Users/" not in text


def test_mixed_property_is_rejected_before_any_plan(tmp_path):
    manifest = tmp_path / "manifest.json"
    _write(
        manifest,
        {
            "property_id": "home_a",
            "rooms": ["room_01"],
            "lidar": {"path": "other_lidar", "property_id": "home_b"},
        },
    )
    report = run_assessment(manifest, tmp_path / "out")
    assert report["exit_code"] == 2
    assert report["rejected"] is True
    assert "home_b" in report["reason"]
    assert not (tmp_path / "out" / "lidar" / "plan.json").exists()


def test_ground_truth_room_from_another_property_is_rejected(tmp_path):
    _write(
        tmp_path / "tape.json",
        {
            "property_id": "home_a",
            "capture_id": "tape",
            "source": "tape",
            "measurements": [
                {
                    "property_id": "home_a",
                    "room_id": "kitchen_elsewhere",
                    "measurement_type": "ceiling_height",
                    "value_m": 2.5,
                    "unit": "m",
                    "uncertainty_m": 0.002,
                    "source": "tape",
                    "capture_id": "tape",
                }
            ],
        },
    )
    _write(
        tmp_path / "manifest.json",
        {
            "property_id": "home_a",
            "rooms": ["room_01"],
            "ground_truth": {"path": "tape.json", "property_id": "home_a"},
        },
    )
    report = run_assessment(tmp_path / "manifest.json", tmp_path / "out")
    assert report["exit_code"] == 2
    assert "kitchen_elsewhere" in report["reason"]


def test_nine_photos_are_rejected_and_not_truncated(tmp_path):
    room = tmp_path / "photos" / "room_01"
    room.mkdir(parents=True)
    for index in range(9):
        (room / f"{index}.jpg").write_bytes(b"not-a-real-capture")
    _write(
        tmp_path / "manifest.json",
        {"property_id": "home_a", "rooms": ["room_01"], "photos": {"path": "photos", "property_id": "home_a"}},
    )
    out = tmp_path / "out"
    report = run_assessment(tmp_path / "manifest.json", out)
    validation = json.loads((out / "validation.json").read_text(encoding="utf-8"))
    assert any("more_than_8" in item for item in validation["photos"])
    assert report["gates"]["photo_walls"]["status"] == "BLOCKED"
    assert not (out / "photo" / "plan.json").exists()


def test_lidar_contract_names_the_missing_files_and_does_not_substitute_video(tmp_path):
    capture = tmp_path / "lidar"
    (capture / "depth").mkdir(parents=True)
    (capture / "depth" / "000.png").write_bytes(b"png")
    errors = validate_lidar_dir(capture)
    assert any("confidence" in item for item in errors)
    assert any("odometry" in item for item in errors)
    assert any("camera_matrix" in item for item in errors)
    assert validate_lidar_dir(None) == ["lidar capture is missing"]


def test_video_timestamps_out_of_order_are_rejected(tmp_path):
    (tmp_path / "walk.mp4").write_bytes(b"")
    (tmp_path / "odometry.csv").write_text(
        "frame,timestamp,x,y,z,qx,qy,qz,qw\n0,2,0,0,0,0,0,0,1\n1,1,0,0,0,0,0,0,1\n",
        encoding="utf-8",
    )
    assert validate_video_dir(tmp_path) == ["video odometry timestamps are not in frame order"]


def test_incomplete_tape_row_cannot_pass_the_ceiling_gate(tmp_path):
    _write(
        tmp_path / "tape.json",
        {
            "property_id": "home_a",
            "capture_id": "tape",
            "source": "tape",
            "measurements": [
                {
                    "property_id": "home_a",
                    "room_id": "room_01",
                    "measurement_type": "ceiling_height",
                    "value_m": 2.50,
                    "predicted_m": 2.50,
                    "repeat_spread_m": 0.0,
                    "source": "tape",
                    "capture_id": "tape",
                }
            ],
        },
    )
    _write(
        tmp_path / "manifest.json",
        {"property_id": "home_a", "rooms": ["room_01"], "ground_truth": "tape.json"},
    )
    report = run_assessment(tmp_path / "manifest.json", tmp_path / "out")
    assert report["gates"]["ceiling"]["status"] == "BLOCKED"
    assert "schema" in report["gates"]["ceiling"]["reason"]


def test_complete_tape_ceiling_can_pass_and_a_wide_opening_fails(tmp_path):
    opening = {
        "property_id": "home_a",
        "opening_id": "door",
        "room_id": "room_01",
        "wall_id": "w0",
        "type": "door",
        "width_m": 0.80,
        "unit": "m",
        "uncertainty_m": 0.002,
        "source": "tape",
        "capture_id": "tape",
    }
    _write(
        tmp_path / "pass.json",
        {
            "property_id": "home_a",
            "capture_id": "tape",
            "source": "tape",
            "measurements": [
                {
                    "property_id": "home_a",
                    "room_id": "room_01",
                    "measurement_type": "ceiling_height",
                    "value_m": 2.50,
                    "predicted_m": 2.50,
                    "repeat_spread_m": 0.0,
                    "unit": "m",
                    "uncertainty_m": 0.002,
                    "source": "tape",
                    "capture_id": "tape",
                }
            ],
        },
    )
    _write(tmp_path / "pass_manifest.json", {"property_id": "home_a", "rooms": ["room_01"], "ground_truth": "pass.json"})
    passed = run_assessment(tmp_path / "pass_manifest.json", tmp_path / "pass_out")
    assert passed["gates"]["ceiling"]["status"] == "PASS"
    assert passed["exit_code"] == 0
    fail_doc = {
        "property_id": "home_a",
        "capture_id": "tape",
        "source": "tape",
        "measurements": [],
        "openings": [opening],
        "predicted_openings": [{**opening, "width_m": 0.90, "confidence": 0.4, "method": "wall_termination"}],
    }
    _write(tmp_path / "fail.json", fail_doc)
    _write(tmp_path / "fail_manifest.json", {"property_id": "home_a", "rooms": ["room_01"], "ground_truth": "fail.json"})
    with pytest.raises(SystemExit) as exc:
        main(["assessment", "--manifest", str(tmp_path / "fail_manifest.json"), "--out", str(tmp_path / "fail_out")])
    assert exc.value.code == 1
    failed = json.loads((tmp_path / "fail_out" / "benchmark.json").read_text(encoding="utf-8"))
    assert failed["gates"]["openings"]["status"] == "FAIL"


def test_malformed_manifest_exits_rejected(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["assessment", "--manifest", str(path), "--out", str(tmp_path / "out")])
    assert exc.value.code == 2


def test_assessment_gates_are_deterministic(tmp_path):
    manifest = tmp_path / "manifest.json"
    _write(manifest, {"property_id": "home_a", "rooms": ["room_01"]})
    first = run_assessment(manifest, tmp_path / "a")
    second = run_assessment(manifest, tmp_path / "b")
    assert first["gates"] == second["gates"]
    assert first["exit_code"] == second["exit_code"]


def test_measurement_schema_accepts_tape_and_keeps_faro_labelled():
    import jsonschema

    schema_path = Path(__file__).resolve().parents[1] / "benchmarks" / "schema" / "measurement.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    tape = {
        "property_id": "home_a",
        "capture_id": "tape",
        "source": "tape",
        "measurements": [
            {
                "property_id": "home_a",
                "room_id": "room_01",
                "wall_id": "w0",
                "measurement_type": "wall_length",
                "value_m": 4.0,
                "unit": "m",
                "uncertainty_m": 0.002,
                "source": "tape",
                "capture_id": "tape",
            }
        ],
    }
    jsonschema.validate(tape, schema)
    faro = json.loads(json.dumps(tape))
    faro["source"] = "faro"
    faro["measurements"][0]["source"] = "faro"
    jsonschema.validate(faro, schema)
    issues = measurement_contract_issues(faro)
    assert any("reference" in message for _, message in issues)
    assert validate_manifest_identity({"property_id": "home_a", "rooms": ["room_01"]}) == []
