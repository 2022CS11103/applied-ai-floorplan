"""Write the synthetic evaluator fixtures. Not physical ground truth."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "fixtures"
LABEL = "synthetic evaluator fixture — not physical ground truth"


def measure(value: float) -> dict:
    return {"value": value, "sigma": 0.01, "ci95_low": round(value - 0.02, 4), "ci95_high": round(value + 0.02, 4)}


def wall(wall_id: str, length: float, p0, p1, openings=None) -> dict:
    return {
        "id": wall_id,
        "length_m": measure(length),
        "p0_m": p0,
        "p1_m": p1,
        "openings": openings or [],
        "height_m": None,
    }


def opening(opening_id: str, width: float) -> dict:
    return {
        "id": opening_id,
        "class": "door",
        "width_m": measure(width),
        "sill_height_m": None,
        "head_height_m": None,
    }


def room(room_id: str, walls: list[dict], ceiling: float | None) -> dict:
    return {
        "id": room_id,
        "room_id": room_id,
        "walls": walls,
        "ceiling_height_m": None if ceiling is None else measure(ceiling),
        "ceiling_observed": ceiling is not None,
    }


def plan(tier: str, case_id: str, rooms: list[dict], footprint: float | None = None, links=None, overlap: float | None = None) -> dict:
    quality = {}
    if overlap is not None:
        quality["overlap_m2"] = overlap
        quality["overlaps"] = [] if overlap == 0 else [{"area_m2": overlap}]
    if tier == "photo":
        quality["scale_source"] = "synthetic_fixture_meters"
        quality["scale_estimated"] = False
    return {
        "schema_version": "1.0",
        "capture_id": case_id,
        "tier": tier,
        "units": "meters",
        "status": "ok",
        "degraded_reasons": [],
        "property": {
            "rooms": rooms,
            "adjacencies": links or [],
            "footprint_area_m2": footprint,
        },
        "quality": quality,
        "drift": {
            "method": ["evaluator fixture, drift not re-run"],
            "footprint_area_m2_anchor_on": footprint,
            "footprint_area_m2_anchor_off": None if footprint is None else round(footprint - 0.4, 4),
        },
    }


def gt(case_id: str, tier: str, rooms: list[dict], **extra) -> dict:
    doc = {
        "case_id": case_id,
        "tier": tier,
        "evaluator_only": True,
        "label": LABEL,
        "provenance": "synthetic_evaluator",
        "rooms": rooms,
    }
    doc.update(extra)
    return doc


def gt_room(room_id: str, walls=None, openings=None, ceiling=None) -> dict:
    doc = {"room_id": room_id, "walls": walls or [], "openings": openings or []}
    if ceiling is not None:
        doc["ceiling_height_m"] = ceiling
    return doc


def gt_wall(wall_id: str, length: float, p0=None, p1=None) -> dict:
    doc = {"wall_id": wall_id, "length_m": length}
    if p0 is not None:
        doc["p0_m"] = p0
        doc["p1_m"] = p1
    return doc


def write(name: str, ground: dict, prediction: dict, repeat: dict | None = None) -> None:
    folder = ROOT / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "ground_truth.json").write_text(json.dumps(ground, indent=2), encoding="utf-8")
    (folder / "prediction.json").write_text(json.dumps(prediction, indent=2), encoding="utf-8")
    if repeat is not None:
        (folder / "prediction_repeat.json").write_text(json.dumps(repeat, indent=2), encoding="utf-8")


def main() -> None:
    door = opening("room_0_w0_o0", 0.80)
    base_wall = wall("room_0_w0", 4.0, [0, 0], [4, 0], [door])
    base_room = room("room_0", [base_wall], 2.50)
    base_gt = gt(
        "exact_match",
        "video",
        [gt_room("room_0", [gt_wall("room_0_w0", 4.0, [0, 0], [4, 0])], [{"opening_id": "room_0_w0_o0", "wall_id": "room_0_w0", "width_m": 0.80}], 2.50)],
    )
    write("exact_match", base_gt, plan("video", "exact_match", [base_room], 12.0))

    short = wall("room_0_w0", 4.0, [0, 0], [4, 0])
    write(
        "wall_error",
        gt("wall_error", "video", [gt_room("room_0", [gt_wall("room_0_w0", 3.80, [0, 0], [3.8, 0])])]),
        plan("video", "wall_error", [room("room_0", [short], None)]),
    )

    wide = opening("room_0_w0_o0", 0.85)
    write(
        "opening_error",
        gt("opening_error", "lidar", [gt_room("room_0", [gt_wall("room_0_w0", 4.0)], [{"opening_id": "room_0_w0_o0", "wall_id": "room_0_w0", "width_m": 0.80}])]),
        plan("lidar", "opening_error", [room("room_0", [wall("room_0_w0", 4.0, [0, 0], [4, 0], [wide])], None)]),
    )

    write(
        "missed_opening",
        gt("missed_opening", "lidar", [gt_room("room_0", [gt_wall("room_0_w0", 4.0)], [{"opening_id": "room_0_w0_o0", "wall_id": "room_0_w0", "width_m": 0.80}])]),
        plan("lidar", "missed_opening", [room("room_0", [wall("room_0_w0", 4.0, [0, 0], [4, 0])], None)]),
    )

    write(
        "phantom_opening",
        gt("phantom_opening", "lidar", [gt_room("room_0", [gt_wall("room_0_w0", 4.0)], [])]),
        plan("lidar", "phantom_opening", [room("room_0", [wall("room_0_w0", 4.0, [0, 0], [4, 0], [door])], None)]),
    )

    write(
        "ceiling_error",
        gt("ceiling_error", "lidar", [gt_room("room_0", [gt_wall("room_0_w0", 4.0)], [], 2.50)]),
        plan("lidar", "ceiling_error", [room("room_0", [wall("room_0_w0", 4.0, [0, 0], [4, 0])], 2.53)]),
    )

    repeat_gt = gt("repeatability_pass", "lidar", [gt_room("room_0", [gt_wall("room_0_w0", 4.0)])])
    first = plan("lidar", "repeatability_pass", [room("room_0", [wall("room_0_w0", 4.0, [0, 0], [4, 0])], None)])
    second = plan("lidar", "repeatability_pass", [room("room_0", [wall("room_0_w0", 4.004, [0, 0], [4.004, 0])], None)])
    write("repeatability_pass", repeat_gt, first, second)

    fail_gt = gt("repeatability_fail", "lidar", [gt_room("room_0", [gt_wall("room_0_w0", 4.0)])])
    fail_second = plan("lidar", "repeatability_fail", [room("room_0", [wall("room_0_w0", 4.08, [0, 0], [4.08, 0])], None)])
    write("repeatability_fail", fail_gt, plan("lidar", "repeatability_fail", [room("room_0", [wall("room_0_w0", 4.0, [0, 0], [4, 0])], None)]), fail_second)

    photo_rooms = [
        room("room_01", [wall("room_01_w0", 4.0, [0, 0], [4, 0])], None),
        room("room_02", [wall("room_02_w0", 3.0, [4, 0], [7, 0])], None),
    ]
    links = [{"a": "room_01", "b": "room_02", "room_a": "room_01", "room_b": "room_02", "kind": "shared_wall", "relationship": "shared_wall", "source": "geometry"}]
    photo_gt_rooms = [
        gt_room("room_01", [gt_wall("room_01_w0", 4.0)]),
        gt_room("room_02", [gt_wall("room_02_w0", 3.0)]),
    ]
    write(
        "photo_footprint_pass",
        gt("photo_footprint_pass", "photo", photo_gt_rooms, footprint_m2=12.5, adjacencies=[{"room_a": "room_01", "room_b": "room_02"}], overlap_m2_max=0.0),
        plan("photo", "photo_footprint_pass", photo_rooms, 12.0, links, 0.0),
    )
    write(
        "photo_footprint_fail",
        gt("photo_footprint_fail", "photo", photo_gt_rooms, footprint_m2=12.0, overlap_m2_max=0.0),
        plan("photo", "photo_footprint_fail", photo_rooms, 10.0, links, 0.0),
    )
    write(
        "adjacency_correct",
        gt("adjacency_correct", "photo", photo_gt_rooms, footprint_m2=12.0, adjacencies=[{"room_a": "room_01", "room_b": "room_02"}], overlap_m2_max=0.0),
        plan("photo", "adjacency_correct", photo_rooms, 12.0, links, 0.0),
    )
    write(
        "adjacency_wrong",
        gt(
            "adjacency_wrong",
            "photo",
            photo_gt_rooms,
            footprint_m2=12.0,
            adjacencies=[{"room_a": "room_01", "room_b": "room_02"}, {"room_a": "room_01", "room_b": "room_03"}],
            overlap_m2_max=0.0,
        ),
        plan("photo", "adjacency_wrong", photo_rooms, 12.0, links, 0.0),
    )


if __name__ == "__main__":
    main()
