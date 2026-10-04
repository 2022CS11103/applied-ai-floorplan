"""The common plan contract. These tests check shape, not tape accuracy."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from jsonschema import Draft202012Validator

from cozmo_scan.contract import apply_contract
from cozmo_scan.layout import build_layout, layout_to_dict
from cozmo_scan.photo import synthetic_cloud, write_property, write_room
from cozmo_scan.pipeline import run_one

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "schema" / "floorplan.schema.json").read_text(encoding="utf-8"))
MEASURE = {"value", "sigma", "ci95_low", "ci95_high"}


def _errors(plan: dict) -> list[str]:
    found = sorted(Draft202012Validator(SCHEMA).iter_errors(plan), key=lambda err: list(err.path))
    return [f"{list(err.path)}: {err.message}" for err in found]


def _assert_schema(plan: dict) -> None:
    assert _errors(plan) == []


def _assert_measure(measure: dict) -> None:
    assert MEASURE <= set(measure)
    assert measure["ci95_low"] <= measure["value"] <= measure["ci95_high"]


def _assert_graph(plan: dict) -> None:
    rooms = plan["property"]["rooms"]
    ids = [room["room_id"] for room in rooms]
    assert ids == sorted(ids)
    assert ids == [room["id"] for room in rooms]
    known = set(ids)
    for room in rooms:
        assert room["floor_polygon"] == room["polygon_m"]
        assert room["measurements"]["floor_area_m2"] == room["floor_area_m2"]
        _assert_measure(room["floor_area_m2"])
        if room["ceiling_height_m"] is not None:
            _assert_measure(room["ceiling_height_m"])
        for wall in room["walls"]:
            _assert_measure(wall["length_m"])
            assert wall["measurements"]["length_m"] == wall["length_m"]
            for opening in wall["openings"]:
                _assert_measure(opening["width_m"])
    links = plan["property"]["adjacencies"]
    order = [(link["room_a"], link["room_b"]) for link in links]
    assert order == sorted(order)
    for link in links:
        assert link["room_a"] in known and link["room_b"] in known
        assert link["a"] == link["room_a"] and link["b"] == link["room_b"]
        assert link["relationship"] == link["kind"]
        assert link["source"] == "geometry"
        assert link["shared_length_m"] is None or link["shared_length_m"] > 0.4


def _shell(tier: str, rooms: list[dict], links: list[dict], status: str = "ok", reasons: list[str] | None = None) -> dict:
    return apply_contract(
        {
            "schema_version": "1.0",
            "capture_id": "contract",
            "tier": tier,
            "units": "meters",
            "status": status,
            "degraded_reasons": [] if reasons is None else reasons,
            "property": {"rooms": rooms, "adjacencies": links},
            "scope_line_items": [],
            "concealed_damage": [],
            "drift": {"method": ["contract test"]},
        }
    )


def test_lidar_layout_matches_the_common_contract():
    cloud = synthetic_cloud({"x0": 0.0, "z0": 0.0, "x1": 4.0, "z1": 3.0, "height_m": 2.5, "synthetic_fixture": True})
    layout = build_layout(cloud.xyz, tier="lidar")
    assert layout.rooms, layout.notes
    prop = layout_to_dict(layout, "lidar")
    plan = _shell("lidar", prop["rooms"], prop["adjacencies"])
    _assert_schema(plan)
    _assert_graph(plan)
    assert plan["tier"] == "lidar"
    assert plan["status"] == "ok"
    assert plan["degraded_reasons"] == []
    assert [room["room_id"] for room in plan["property"]["rooms"]] == ["room_0"]
    assert plan["property"]["adjacencies"] == []


def test_video_degraded_plan_is_structurally_valid(tmp_path):
    folder = tmp_path / "clip"
    folder.mkdir()
    writer = cv2.VideoWriter(str(folder / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    assert writer.isOpened()
    frame = np.full((120, 160, 3), 30, dtype=np.uint8)
    for _ in range(4):
        writer.write(frame)
    writer.release()
    plan = run_one(folder, "video", tmp_path / "out")
    _assert_schema(plan)
    assert plan["tier"] == "video"
    assert plan["status"] == "degraded"
    assert plan["property"]["rooms"] == []
    assert plan["property"]["adjacencies"] == []
    assert plan["degraded_reasons"] == ["missing_pose"]


def test_photo_single_room_matches_the_contract(tmp_path):
    write_room(tmp_path / "room_01", (0.0, 0.0, 4.0, 3.0), n_images=4, seed=1)
    plan = run_one(tmp_path / "room_01", "photo", tmp_path / "out")
    _assert_schema(plan)
    _assert_graph(plan)
    assert plan["status"] == "ok"
    assert plan["tier"] == "photo"
    assert plan["degraded_reasons"] == []
    assert len(plan["property"]["rooms"]) == 1


def test_degraded_photo_room_stays_valid(tmp_path):
    folder = tmp_path / "blank"
    folder.mkdir()
    image = np.full((96, 128, 3), 18, dtype=np.uint8)
    for i in range(3):
        cv2.imwrite(str(folder / f"{i + 1:02d}.jpg"), image)
    plan = run_one(folder, "photo", tmp_path / "out")
    _assert_schema(plan)
    assert plan["status"] == "degraded"
    assert plan["property"]["rooms"] == []
    assert "low_texture" in plan["degraded_reasons"]
    assert plan["status"] != "ok"


def test_four_room_photo_property_keeps_geometry_links(tmp_path):
    root = write_property(tmp_path / "photos")
    first = run_one(root, "photo", tmp_path / "a")
    second = run_one(root, "photo", tmp_path / "b")
    _assert_schema(first)
    _assert_graph(first)
    assert first["status"] == "ok"
    ids = [room["room_id"] for room in first["property"]["rooms"]]
    assert ids == ["connector", "room_01", "room_02", "room_03"]
    assert ids == [room["room_id"] for room in second["property"]["rooms"]]
    links = [(link["room_a"], link["room_b"]) for link in first["property"]["adjacencies"]]
    assert links == [(link["room_a"], link["room_b"]) for link in second["property"]["adjacencies"]]
    assert {frozenset(pair) for pair in links} == {
        frozenset(("connector", "room_01")),
        frozenset(("connector", "room_02")),
        frozenset(("room_01", "room_03")),
    }


def test_room_order_does_not_follow_insertion_order():
    def room(room_id: str) -> dict:
        area = {"value": 1.0, "sigma": 0.1, "ci95_low": 0.8, "ci95_high": 1.2}
        length = {"value": 1.0, "sigma": 0.1, "ci95_low": 0.8, "ci95_high": 1.2}
        return {
            "id": room_id,
            "polygon_m": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            "floor_area_m2": area,
            "ceiling_height_m": None,
            "ceiling_observed": False,
            "walls": [
                {
                    "id": f"{room_id}_w1",
                    "length_m": length,
                    "height_m": None,
                    "p0_m": [0.0, 0.0],
                    "p1_m": [1.0, 0.0],
                    "openings": [],
                }
            ],
            "damage": [],
        }

    plan = _shell("lidar", [room("room_b"), room("room_a")], [])
    assert [item["room_id"] for item in plan["property"]["rooms"]] == ["room_a", "room_b"]
    assert plan["property"]["adjacencies"] == []


def test_contract_does_not_invent_adjacency_or_keep_a_dangling_link():
    area = {"value": 9.0, "sigma": 0.2, "ci95_low": 8.6, "ci95_high": 9.4}
    length = {"value": 3.0, "sigma": 0.05, "ci95_low": 2.9, "ci95_high": 3.1}

    def room(room_id: str, origin: float) -> dict:
        return {
            "id": room_id,
            "polygon_m": [[origin, 0.0], [origin + 3.0, 0.0], [origin + 3.0, 3.0], [origin, 3.0]],
            "floor_area_m2": area,
            "ceiling_height_m": None,
            "ceiling_observed": False,
            "walls": [
                {
                    "id": f"{room_id}_w0",
                    "length_m": length,
                    "height_m": None,
                    "p0_m": [origin, 0.0],
                    "p1_m": [origin + 3.0, 0.0],
                    "openings": [],
                }
            ],
        }

    plan = _shell(
        "photo",
        [room("room_a", 0.0), room("room_b", 12.0)],
        [
            {"a": "room_a", "b": "missing_room", "kind": "shared_wall", "gap_m": 0.0},
        ],
    )
    assert plan["property"]["adjacencies"] == []
    assert {room["room_id"] for room in plan["property"]["rooms"]} == {"room_a", "room_b"}


def test_empty_plan_marked_ok_is_rejected():
    plan = _shell("lidar", [], [], status="ok", reasons=[])
    _assert_schema(plan)
    assert plan["status"] == "degraded"
    assert plan["degraded_reasons"] == ["no_room_closure"]
    assert plan["property"]["rooms"] == []
