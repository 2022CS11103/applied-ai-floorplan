"""Photo tier on a labelled synthetic fixture.

The fixture is not a physical capture and it is not an accuracy claim.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from jsonschema import Draft202012Validator

from cozmo_scan.photo import FIXTURE_LABEL, write_property, write_room
from cozmo_scan.pipeline import run_one

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "schema" / "floorplan.schema.json").read_text(encoding="utf-8"))
MEASURE = {"value", "sigma", "ci95_low", "ci95_high"}


def _flat(folder: Path, n: int, value: int = 18) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    image = np.full((96, 128, 3), value, dtype=np.uint8)
    for i in range(n):
        cv2.imwrite(str(folder / f"{i + 1:02d}.jpg"), image)


def _plan(tmp_path: Path, folder: Path):
    return run_one(folder, "photo", tmp_path / "out")


def _assert_schema(plan: dict) -> None:
    errors = sorted(Draft202012Validator(SCHEMA).iter_errors(plan), key=lambda e: list(e.path))
    assert errors == [], "\n".join(f"{list(e.path)}: {e.message}" for e in errors)


def _assert_measure(measure: dict) -> None:
    assert MEASURE <= set(measure)
    assert measure["ci95_low"] <= measure["value"] <= measure["ci95_high"]


def test_single_room_photo_closes_and_uses_the_common_measurement(tmp_path):
    write_room(tmp_path / "room_01", (0.0, 0.0, 4.0, 3.0), n_images=4, seed=1)
    plan = _plan(tmp_path, tmp_path / "room_01")
    _assert_schema(plan)
    assert (tmp_path / "out" / "plan.json").is_file()
    assert (tmp_path / "out" / "plan.png").is_file()
    assert plan["status"] == "ok"
    assert plan["tier"] == "photo"
    assert plan["degraded_reasons"] == []
    assert FIXTURE_LABEL in plan["notes"]
    assert plan["quality"]["synthetic_fixture"] is True
    rooms = plan["property"]["rooms"]
    assert len(rooms) == 1
    _assert_measure(rooms[0]["floor_area_m2"])
    assert rooms[0]["floor_area_m2"]["value"] == pytest.approx(12.0, abs=0.2)
    assert rooms[0]["damage"] == []
    for wall in rooms[0]["walls"]:
        _assert_measure(wall["length_m"])
        assert isinstance(wall["openings"], list)


def test_two_images_is_the_minimum(tmp_path):
    write_room(tmp_path / "room", (0.0, 0.0, 3.0, 3.0), n_images=2, seed=2)
    plan = _plan(tmp_path, tmp_path / "room")
    assert plan["status"] == "ok"
    assert plan["quality"]["images_used"] == 2
    assert len(plan["property"]["rooms"]) == 1


def test_eight_images_is_the_maximum_used(tmp_path):
    write_room(tmp_path / "room8", (0.0, 0.0, 3.5, 3.0), n_images=8, seed=3)
    plan = _plan(tmp_path, tmp_path / "room8")
    assert plan["status"] == "ok"
    assert plan["quality"]["images_used"] == 8
    write_room(tmp_path / "room9", (0.0, 0.0, 3.5, 3.0), n_images=9, seed=4)
    capped = _plan(tmp_path, tmp_path / "room9")
    assert capped["status"] == "ok"
    assert capped["quality"]["images"] == 9
    assert capped["quality"]["images_used"] == 8
    assert capped["quality"]["images_capped_at_8"] is True


def test_missing_image_is_degraded(tmp_path):
    _flat(tmp_path / "one", 1)
    plan = _plan(tmp_path, tmp_path / "one")
    _assert_schema(plan)
    assert plan["status"] == "degraded"
    assert plan["property"]["rooms"] == []
    assert "missing_images" in plan["degraded_reasons"]


def test_low_texture_is_degraded(tmp_path):
    _flat(tmp_path / "blank", 4)
    plan = _plan(tmp_path, tmp_path / "blank")
    _assert_schema(plan)
    assert plan["status"] == "degraded"
    assert plan["property"]["rooms"] == []
    assert "low_texture" in plan["degraded_reasons"]
    assert "insufficient_points" in plan["degraded_reasons"]
    assert plan["status"] != "ok"


def test_property_keeps_every_room_and_ignores_the_sketch(tmp_path):
    root = write_property(tmp_path / "photos")
    plan = _plan(tmp_path, root)
    _assert_schema(plan)
    assert plan["status"] == "ok"
    assert FIXTURE_LABEL in plan["notes"]
    assert any("adjacency.json was not used" in note for note in plan["notes"])
    ids = {room["id"] for room in plan["property"]["rooms"]}
    assert ids == {"room_01", "room_02", "room_03", "connector"}
    links = {frozenset((link["a"], link["b"])) for link in plan["property"]["adjacencies"]}
    assert links == {
        frozenset(("room_01", "connector")),
        frozenset(("connector", "room_02")),
        frozenset(("room_01", "room_03")),
    }
    assert all(link["source"] == "geometry" for link in plan["property"]["adjacencies"])
    assert plan["quality"]["overlap_m2"] == 0.0
    # Outer rectangles sum to 33.84 m2. The layout cell is the inner face
    # of a 2 cm wall sheet, so the closed footprint sits a little inside that.
    assert plan["property"]["footprint_area_m2"] == pytest.approx(33.84, abs=0.6)
    for room in plan["property"]["rooms"]:
        _assert_measure(room["floor_area_m2"])
        for wall in room["walls"]:
            _assert_measure(wall["length_m"])


def test_overlapping_rooms_are_rejected(tmp_path):
    root = tmp_path / "overlap"
    write_room(root / "room_a", (0.0, 0.0, 4.0, 3.0), n_images=3, seed=5)
    write_room(root / "room_b", (0.5, 0.5, 4.5, 3.5), n_images=3, seed=6)
    plan = _plan(tmp_path, root)
    _assert_schema(plan)
    assert plan["status"] == "degraded"
    assert "overlap_rejected" in plan["degraded_reasons"]
    assert {room["id"] for room in plan["property"]["rooms"]} == {"room_a", "room_b"}
    assert plan["property"]["adjacencies"] == []
    assert plan["quality"]["overlap_m2"] > 0.05


def test_disconnected_rooms_are_not_given_a_link(tmp_path):
    root = tmp_path / "apart"
    write_room(root / "room_a", (0.0, 0.0, 3.0, 3.0), n_images=3, seed=7)
    write_room(root / "room_b", (12.0, 0.0, 15.0, 3.0), n_images=3, seed=8)
    plan = _plan(tmp_path, root)
    _assert_schema(plan)
    assert plan["status"] == "degraded"
    assert "adjacency_unresolved" in plan["degraded_reasons"]
    assert {room["id"] for room in plan["property"]["rooms"]} == {"room_a", "room_b"}
    assert plan["property"]["adjacencies"] == []


def test_empty_or_invalid_folder_is_not_a_success(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    plan = _plan(tmp_path, empty)
    _assert_schema(plan)
    assert plan["status"] == "degraded"
    assert plan["property"]["rooms"] == []
    assert "missing_images" in plan["degraded_reasons"]
    stray = tmp_path / "notes.txt"
    stray.write_text("not a capture", encoding="utf-8")
    invalid = run_one(stray, "photo", tmp_path / "invalid_out")
    assert invalid["status"] == "degraded"
    assert invalid["property"]["rooms"] == []
