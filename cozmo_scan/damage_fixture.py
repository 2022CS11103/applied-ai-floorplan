"""Synthetic staged damage. Not a physical capture and not a tape.

The scene file is the input. ground_truth.json is the painted specification.
The detector's plan is a separate result. Nothing here is written back onto
the LiDAR, video, or photo room fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .damage import concealed_flags, detect_on_points, scope_items

FIXTURE_LABEL = "synthetic/test fixture — not physical benchmark ground truth"


def load_scene(folder: Path) -> dict:
    return json.loads((Path(folder) / "input" / "scene.json").read_text(encoding="utf-8"))


def load_ground_truth(folder: Path) -> dict:
    return json.loads((Path(folder) / "ground_truth.json").read_text(encoding="utf-8"))


def build_cloud(scene: dict) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Wall returns plus one sofa. Patches are painted only where the scene says."""
    pitch = float(scene["pitch_m"])
    wall_rgb = np.array(scene["wall_rgb"], dtype=np.uint8)
    chunks_xyz = []
    chunks_rgb = []
    walls = []
    for wall in scene["walls"]:
        p0 = np.array(wall["p0_m"], dtype=np.float64)
        p1 = np.array(wall["p1_m"], dtype=np.float64)
        edge = p1 - p0
        length = float(np.linalg.norm(edge))
        direction = edge / length
        along = np.arange(pitch, length - pitch + 1e-9, pitch)
        height = np.arange(float(scene["height_min_m"]), float(scene["height_max_m"]) + 1e-9, pitch)
        aa, hh = np.meshgrid(along, height)
        xyz = np.column_stack(
            [
                p0[0] + aa.ravel() * direction[0],
                hh.ravel(),
                p0[1] + aa.ravel() * direction[1],
            ]
        )
        rgb = np.tile(wall_rgb, (len(xyz), 1))
        for patch in wall.get("patches") or []:
            x0, x1 = patch["along_m"]
            y0, y1 = patch["height_m"]
            mask = (aa.ravel() >= x0) & (aa.ravel() < x1) & (hh.ravel() >= y0) & (hh.ravel() < y1)
            rgb[mask] = np.array(patch["rgb"], dtype=np.uint8)
        chunks_xyz.append(xyz)
        chunks_rgb.append(rgb)
        walls.append(
            {
                "id": wall["id"],
                "p0_m": wall["p0_m"],
                "p1_m": wall["p1_m"],
                "openings": wall.get("openings") or [],
                "length_m": {"value": round(length, 4), "sigma": 0.02, "ci95_low": round(length - 0.04, 4), "ci95_high": round(length + 0.04, 4)},
            }
        )
    furniture = scene["furniture"]
    origin = np.array(furniture["origin_m"], dtype=np.float64)
    size = np.array(furniture["size_m"], dtype=np.float64)
    counts = np.maximum(np.round(size / pitch).astype(int), 1)
    grid = np.meshgrid(
        np.linspace(0.0, size[0], int(counts[0])),
        np.linspace(0.0, size[1], int(counts[1])),
        np.linspace(0.0, size[2], int(counts[2])),
        indexing="ij",
    )
    sofa = np.column_stack([axis.ravel() for axis in grid]) + origin
    chunks_xyz.append(sofa)
    chunks_rgb.append(np.tile(np.array(furniture["rgb"], dtype=np.uint8), (len(sofa), 1)))
    room = {
        "id": scene["room_id"],
        "room_id": scene["room_id"],
        "polygon_m": scene["floor_polygon_m"],
        "floor_polygon": scene["floor_polygon_m"],
        "walls": walls,
        "damage": [],
    }
    return (
        np.concatenate(chunks_xyz).astype(np.float32),
        np.concatenate(chunks_rgb).astype(np.uint8),
        [room],
    )


def assess_scene(scene: dict, tier: str = "lidar") -> dict:
    """Run the same damage functions the LiDAR plan uses. No second detector."""
    xyz, colors, rooms = build_cloud(scene)
    found = detect_on_points(xyz, colors, rooms, tier)
    by_room: dict[str, list] = {}
    for region in found:
        by_room.setdefault(region["room_id"], []).append(region)
    for room in rooms:
        room["damage"] = by_room.get(room["id"], [])
    return {
        "label": FIXTURE_LABEL,
        "tier": tier,
        "rooms": rooms,
        "scope_line_items": scope_items(rooms),
        "concealed_damage": concealed_flags(rooms),
    }
