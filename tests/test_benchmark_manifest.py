"""Bundle loader. A missing tape file is BLOCKED. A broken file is exit code 2."""

import json

from cozmo_scan.benchmark import (
    bundle_status,
    load_benchmark_manifest,
    validate_benchmark_manifest,
    validate_ground_truth,
    validate_incumbent_export,
    validate_property_identity,
    validate_repeat_capture,
    validate_required_tiers,
)

ROOT = "benchmarks/properties/not_yet_captured/manifest.json"


def _manifest(tmp_path, **overrides):
    doc = {
        "property_id": "home_a",
        "rooms": ["room_01", "room_02", "room_03"],
        "connector": {"rooms": ["room_01", "room_02"]},
        "tiers": {
            "lidar": {"path": "lidar"},
            "video": {"path": "video"},
            "photo": {"path": "photo"},
        },
        "repeat": {"capture_a": "repeat/capture_a", "capture_b": "repeat/capture_b"},
        "ground_truth": {"path": "ground_truth/measurements.json", "source": "tape"},
        "incumbent": {"app": "polycam", "version": "1", "path": "incumbent/export.json"},
    }
    doc.update(overrides)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return load_benchmark_manifest(path)


def test_example_bundle_stays_blocked():
    result = bundle_status(ROOT)
    assert result["status"] == "BLOCKED"
    assert result["exit_code"] == 0
    assert result["missing"]


def test_malformed_manifest_is_exit_2(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{", encoding="utf-8")
    result = load_benchmark_manifest(path)
    assert result["reason"] == "malformed_manifest"
    assert result["exit_code"] == 2
    assert bundle_status(path)["exit_code"] == 2


def test_missing_keys_are_malformed(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"property_id": "home_a"}), encoding="utf-8")
    result = validate_benchmark_manifest(load_benchmark_manifest(path))
    assert result["exit_code"] == 2
    assert result["reason"] == "malformed_manifest"


def test_mixed_property_is_exit_2(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["ground_truth"]["property_id"] = "home_b"
    result = validate_property_identity(manifest)
    assert result["exit_code"] == 2
    assert result["reason"] == "mixed_property"


def test_missing_ground_truth_is_blocked_not_an_exception(tmp_path):
    manifest = _manifest(tmp_path)
    result = validate_ground_truth(manifest)
    assert result == {
        "status": "BLOCKED",
        "reason": "missing_ground_truth",
        "property_id": "home_a",
        "missing": ["ground_truth"],
        "exit_code": 0,
    }


def test_empty_tier_folders_are_missing(tmp_path):
    (tmp_path / "lidar").mkdir()
    manifest = _manifest(tmp_path)
    result = validate_required_tiers(manifest)
    assert result["status"] == "BLOCKED"
    assert "lidar" in result["missing"]
    assert result["exit_code"] == 0


def test_missing_repeat_and_incumbent_are_blocked(tmp_path):
    manifest = _manifest(tmp_path)
    assert validate_repeat_capture(manifest)["reason"] == "missing_repeat_capture"
    assert "capture_b" in validate_repeat_capture(manifest)["missing"]
    assert validate_incumbent_export(manifest)["reason"] == "missing_incumbent_export"
    assert validate_incumbent_export(manifest)["exit_code"] == 0
