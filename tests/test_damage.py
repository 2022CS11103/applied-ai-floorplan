"""Damage on a labelled synthetic room. Not a physical staged capture.

The painted rectangles in ground_truth.json are the specification.
The detector result is checked for class, room, and a measurable area.
It is not a claim that those areas match a tape.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from cozmo_scan.damage import concealed_flags, detect_on_points
from cozmo_scan.damage_fixture import FIXTURE_LABEL, assess_scene, load_ground_truth, load_scene
from cozmo_scan.pipeline import run_one

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "synthetic_damage"
SCHEMA = json.loads((ROOT / "schema" / "floorplan.schema.json").read_text(encoding="utf-8"))
MEASURE = {"value", "sigma", "ci95_low", "ci95_high"}


def _scene_without(classes: set[str]) -> dict:
    scene = load_scene(FIXTURE)
    for wall in scene["walls"]:
        wall["patches"] = [patch for patch in wall.get("patches") or [] if patch["class"] not in classes]
    return scene


def _regions(result: dict) -> list[dict]:
    found = []
    for room in result["rooms"]:
        found.extend(room["damage"])
    return found


def _assert_measure(measure: dict) -> None:
    assert MEASURE <= set(measure)
    assert measure["ci95_low"] <= measure["value"] <= measure["ci95_high"]


def test_fixture_is_labelled_and_ground_truth_stays_separate():
    text = (FIXTURE / "FIXTURE.txt").read_text(encoding="utf-8")
    truth = load_ground_truth(FIXTURE)
    scene = load_scene(FIXTURE)
    assert FIXTURE_LABEL in text
    assert truth["label"] == FIXTURE_LABEL
    assert scene["label"] == FIXTURE_LABEL
    assert "regions" in truth
    result = assess_scene(scene)
    assert load_ground_truth(FIXTURE) == truth
    assert "rooms" in result and "regions" not in result


def test_no_damage_on_a_clean_room():
    result = assess_scene(_scene_without({"stain", "moisture"}))
    assert _regions(result) == []
    assert result["scope_line_items"] == []
    assert all(flag["fired"] is False for flag in result["concealed_damage"])


def test_one_stain_region():
    result = assess_scene(_scene_without({"moisture"}))
    found = _regions(result)
    assert [region["class"] for region in found] == ["stain"]
    assert found[0]["room_id"] == "room_01"
    _assert_measure(found[0]["extent_m2"])


def test_two_classes_have_extent_room_and_stable_ids():
    scene = load_scene(FIXTURE)
    first = assess_scene(scene)
    second = assess_scene(scene)
    found = _regions(first)
    again = _regions(second)
    assert [region["class"] for region in found] == ["moisture", "stain"]
    assert [region["damage_id"] for region in found] == [region["damage_id"] for region in again]
    assert [region["extent_m2"] for region in found] == [region["extent_m2"] for region in again]
    truth = {item["class"]: item for item in load_ground_truth(FIXTURE)["regions"]}
    for region in found:
        assert region["damage_id"] == region["id"]
        assert region["surface"] == region["surface_id"] == "room_01_w0"
        assert region["room_id"] == "room_01"
        assert region["extent"] is region["extent_m2"]
        _assert_measure(region["extent_m2"])
        assert region["extent_m2"]["value"] == pytest.approx(truth[region["class"]]["extent_m2"], abs=0.15)
        assert region["confidence"] is None
        assert region["evidence"]["method"] == "local_lab_residual"
        assert region["evidence"]["concealed_observed_directly"] is False
        assert region["region"]["height_band_m"] == region["height_band_m"]
    moisture = found[0]
    stain = found[1]
    assert moisture["concealed"] is True
    assert moisture["concealed_rules"] == ["base_stain_moisture_path"]
    assert stain["concealed"] is False
    assert stain["concealed_rules"] == []
    fired = [flag for flag in first["concealed_damage"] if flag["fired"]]
    assert [flag["rule"] for flag in fired] == ["base_stain_moisture_path"]
    assert {item["class"] for item in first["scope_line_items"]} == {"stain", "moisture"}
    assert {item["damage_id"] for item in first["scope_line_items"]} == {region["damage_id"] for region in found}


def test_damage_references_the_room_and_validates():
    scene = load_scene(FIXTURE)
    result = assess_scene(scene)
    room_ids = {room["room_id"] for room in result["rooms"]}
    for region in _regions(result):
        assert region["room_id"] in room_ids
    plan = {
        "schema_version": "1.0",
        "capture_id": "synthetic_damage",
        "tier": "lidar",
        "units": "meters",
        "status": "degraded",
        "degraded_reasons": ["synthetic_damage_fixture"],
        "property": {
            "rooms": [
                {
                    "id": room["id"],
                    "room_id": room["room_id"],
                    "polygon_m": room["polygon_m"],
                    "floor_polygon": room["floor_polygon"],
                    "floor_area_m2": {"value": 12.0, "sigma": 0.2, "ci95_low": 11.6, "ci95_high": 12.4},
                    "ceiling_height_m": None,
                    "ceiling_observed": False,
                    "walls": [
                        {
                            "id": wall["id"],
                            "length_m": wall["length_m"],
                            "height_m": None,
                            "p0_m": wall["p0_m"],
                            "p1_m": wall["p1_m"],
                            "openings": wall["openings"],
                            "measurements": {"length_m": wall["length_m"], "height_m": None},
                        }
                        for wall in room["walls"]
                    ],
                    "measurements": {
                        "floor_area_m2": {"value": 12.0, "sigma": 0.2, "ci95_low": 11.6, "ci95_high": 12.4},
                        "ceiling_height_m": None,
                    },
                    "status": "ok",
                    "reasons": [],
                    "damage": room["damage"],
                }
                for room in result["rooms"]
            ],
            "adjacencies": [],
        },
        "scope_line_items": result["scope_line_items"],
        "concealed_damage": result["concealed_damage"],
        "drift": {"method": ["not a capture"]},
    }
    errors = sorted(Draft202012Validator(SCHEMA).iter_errors(plan), key=lambda err: list(err.path))
    assert errors == [], "\n".join(f"{list(err.path)}: {err.message}" for err in errors)


def test_sill_rule_fires_only_with_a_window():
    stain = {
        "id": "room_01_w0_d0",
        "room_id": "room_01",
        "surface_id": "room_01_w0",
        "class": "stain",
        "extent_m2": {"value": 0.2, "sigma": 0.08, "ci95_low": 0.04, "ci95_high": 0.36},
        "height_band_m": [0.8, 1.1],
    }
    wall = {"id": "room_01_w0", "openings": [{"class": "window"}]}
    room = {"id": "room_01", "walls": [wall], "damage": [stain]}
    flags = {flag["rule"]: flag["fired"] for flag in concealed_flags([room])}
    assert flags["base_stain_moisture_path"] is False
    assert flags["sill_stain_under_window"] is True
    bare = {"id": "room_01_w0", "openings": []}
    flags = {flag["rule"]: flag["fired"] for flag in concealed_flags([{**room, "walls": [bare]}])}
    assert flags["sill_stain_under_window"] is False


def test_empty_or_mismatched_input_detects_nothing():
    xyz = np.zeros((0, 3), dtype=np.float32)
    colors = np.zeros((0, 3), dtype=np.uint8)
    assert detect_on_points(xyz, colors, [], "lidar") == []
    assert detect_on_points(xyz, None, [], "lidar") == []
    points = np.zeros((10, 3), dtype=np.float32)
    assert detect_on_points(points, colors, [{"id": "room_01", "walls": []}], "lidar") == []
    short = np.zeros((3, 3), dtype=np.uint8)
    assert detect_on_points(points, short, [{"id": "room_01", "walls": [{"id": "w", "p0_m": [0, 0], "p1_m": [1, 0], "openings": []}]}], "lidar") == []


def test_existing_photo_property_is_not_painted_with_damage(tmp_path):
    plan = run_one(ROOT / "fixtures" / "synthetic_photo_property", "photo", tmp_path / "out")
    assert plan["status"] == "ok"
    assert plan["scope_line_items"] == []
    assert plan["concealed_damage"]
    assert all(flag["fired"] is False for flag in plan["concealed_damage"])
    assert all(room["damage"] == [] for room in plan["property"]["rooms"])
