"""Convert one HouseLayout3D scene into this repo's benchmark ground-truth shape.

HouseLayout3D publishes CAD polygons for walls, floors, and ceilings, plus
door and window rectangles, for 16 Matterport buildings. It does not publish
the RGB or depth frames. This module does not run the floor-plan pipeline
and it does not mark the result as laser or tape.

Room ids are the floor polygons in the release. They are not Matterport's
region ids, which are not in this download. A door becomes an adjacency
only when a short step along its normal lands in two different floor
polygons. Anything that fails that test is left unlinked.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

LABEL = (
    "EXTERNAL DATASET VALIDATION — HouseLayout3D CAD annotation derived from "
    "Matterport3D. Not laser/tape. Not the company physical benchmark."
)
SOURCE = "https://huggingface.co/datasets/houselayout3d/HouseLayout3D"
# Horizontal plane if |normal·up| is at least this. Up is +Z in this release.
_HORIZONTAL = 0.85
_DOOR_REACH_M = 0.45
_WALL_ATTACH_M = 0.40
# Adapter assumption, not a HouseLayout3D label. The paper does not define a
# minimum room area. Polygons under this area are kept in the output as
# excluded_horizontal_polygons and are not called rooms.
_MIN_FLOOR_M2 = 1.5
# Adapter assumption. The paper says each door is stored as the open leaf and
# the closed opening. A quad whose horizontal edge is only a few centimetres
# is recorded under openings_not_scored, not as an opening width.
_MIN_OPENING_M = 0.40


def convert_scene(scene_dir: Path, scene_id: str | None = None) -> dict:
    scene_dir = Path(scene_dir)
    scene_id = scene_id or scene_dir.name
    doors = _load_quads(scene_dir, "doors")
    windows = _load_quads(scene_dir, "windows")
    planes = np.load(_find(scene_dir, "plane_equations.npy"))
    entities = [_parse_ply(path) for path in _entity_paths(scene_dir)]
    if len(entities) != len(planes):
        raise ValueError(f"{scene_id}: {len(entities)} polygons and {len(planes)} plane rows")

    floors = []
    ceilings = []
    walls = []
    excluded_floors = []
    for index, (entity, plane) in enumerate(zip(entities, planes)):
        point, normal = plane[0], plane[1]
        normal = normal / max(float(np.linalg.norm(normal)), 1e-9)
        loop = _boundary_xy(entity)
        record = {
            "index": index,
            "point": point,
            "normal": normal,
            "loop": loop,
            "z": float(point[2]),
        }
        if abs(float(normal[2])) >= _HORIZONTAL:
            if float(point[2]) < 1.0 and loop is not None:
                area = _shoelace(loop)
                record["area_m2"] = area
                if area >= _MIN_FLOOR_M2:
                    floors.append(record)
                else:
                    excluded_floors.append(
                        {
                            "entity_index": index,
                            "area_m2": round(area, 4),
                            "field_status": "DERIVED_FROM_DATASET",
                            "used_as_room": False,
                            "assumption": (
                                f"adapter excluded horizontal polygons under {_MIN_FLOOR_M2} m^2. "
                                "HouseLayout3D does not label these as doors or as rooms."
                            ),
                        }
                    )
            elif float(point[2]) > 1.5:
                ceilings.append(record)
        elif abs(float(normal[2])) < 0.35 and loop is not None:
            length = _horizontal_span(entity)
            if length >= 0.40:
                record["length_m"] = length
                walls.append(record)

    wall_owner = _assign_walls(walls, floors)
    opening_owner, openings_not_scored = _assign_openings(doors, windows, walls, wall_owner)
    rooms = []
    for order, floor in enumerate(floors):
        room_id = f"{scene_id}_floor_{order:02d}"
        height = _ceiling_above(floor, ceilings)
        room_walls = [
            {
                "wall_id": f"{room_id}_e{wall['index']}",
                "length_m": round(wall["length_m"], 4),
                "entity_index": wall["index"],
                "derived": True,
            }
            for wall in walls
            if wall_owner.get(wall["index"]) == order
        ]
        openings = []
        for item in opening_owner:
            if item["room_order"] != order:
                continue
            openings.append(
                {
                    "opening_id": item["opening_id"],
                    "wall_id": f"{room_id}_e{item['entity_index']}",
                    "width_m": item["width_m"],
                    "class": item["class"],
                }
            )
        rooms.append(
            {
                "room_id": room_id,
                "ceiling_height_m": None if height is None else round(height, 4),
                "walls": room_walls,
                "openings": openings,
                "floor_area_m2": None if floor["loop"] is None else round(_shoelace(floor["loop"]), 4),
                "entity_index": floor["index"],
                "room_id_source": "derived_from_floor_polygon",
            }
        )

    links = _door_links(doors, floors, scene_id)
    areas = [room["floor_area_m2"] for room in rooms if room["floor_area_m2"] is not None]
    return {
        "case_id": f"houselayout3d_{scene_id}",
        "tier": "lidar",
        "evaluator_only": False,
        "label": LABEL,
        "external_dataset": {
            "name": "HouseLayout3D",
            "scene_id": scene_id,
            "url": SOURCE,
            "license": "MIT",
            "underlying_scans": "Matterport3D, not redistributed in this Hugging Face release",
            "replaces_company_physical_benchmark": False,
        },
        "coordinate_frame": {
            "up": "+Z",
            "units": "metres",
            "horizontal_axes": ["X", "Y"],
            "note": "Dataset frame. Not converted into the logger's Y-up frame.",
        },
        "calibration": {
            "scale": "unavailable",
            "reason": "CAD metres come from a Matterport reconstruction, not a tape measured for this assignment",
        },
        "footprint_m2": round(float(sum(areas)), 4) if areas else None,
        "footprint_note": "Sum of derived floor-polygon areas on this scene. Not a surveyed footprint.",
        "rooms": rooms,
        "adjacencies": links,
        "excluded_horizontal_polygons": excluded_floors,
        "openings_not_scored": openings_not_scored,
        "field_status": {
            "scene_id": "PROVIDED_BY_DATASET",
            "door_window_vertices": "PROVIDED_BY_DATASET",
            "opening_width_m": "DERIVED_FROM_DATASET",
            "room_id": "DERIVED_FROM_DATASET",
            "wall_length_m": "DERIVED_FROM_DATASET",
            "floor_area_m2": "DERIVED_FROM_DATASET",
            "ceiling_height_m": "DERIVED_FROM_DATASET",
            "adjacency": "DERIVED_FROM_DATASET",
            "rgb_depth_point_cloud": "UNAVAILABLE",
            "laser_tape": "UNAVAILABLE",
            "repeat_capture": "UNAVAILABLE",
            "imu": "UNAVAILABLE",
        },
        "not_in_release": [
            "rgb_images",
            "depth",
            "point_cloud",
            "matterport_room_ids",
            "repeat_capture",
            "pose_drift_ablation",
        ],
    }


def _assign_walls(walls: list[dict], floors: list[dict]) -> dict[int, int]:
    owner = {}
    for wall in walls:
        if wall["loop"] is None:
            continue
        mid = np.mean(np.asarray(wall["loop"], dtype=np.float64), axis=0)
        best_order = None
        best_distance = 0.35
        for order, floor in enumerate(floors):
            if floor["loop"] is None:
                continue
            distance = 0.0 if _point_in_polygon(mid, floor["loop"]) else _distance_to_loop(mid, floor["loop"])
            if distance < best_distance:
                best_order = order
                best_distance = distance
        if best_order is not None:
            owner[wall["index"]] = best_order
    return owner


def _assign_openings(doors: list[dict], windows: list[dict], walls: list[dict], owner: dict[int, int]) -> tuple[list[dict], list[dict]]:
    assigned = []
    not_scored = []
    for kind, quads in (("door", doors), ("window", windows)):
        for item_index, quad in enumerate(quads):
            width, mid = _opening_width(quad["vertices"])
            if width < _MIN_OPENING_M:
                not_scored.append(
                    {
                        "opening_id": f"{kind}_{item_index}",
                        "class": kind,
                        "horizontal_edge_m": round(width, 4),
                        "field_status": "PROVIDED_BY_DATASET",
                        "used_as_opening_width": False,
                        "assumption": (
                            "Paper states doors are annotated in both the open and closed positions. "
                            f"This quad's horizontal edge is under {_MIN_OPENING_M} m, so the adapter "
                            "does not treat it as the opening width."
                        ),
                    }
                )
                continue
            best = None
            best_distance = _WALL_ATTACH_M
            for wall in walls:
                if wall["index"] not in owner or wall["loop"] is None:
                    continue
                distance = _distance_to_loop(mid[:2], wall["loop"])
                if distance < best_distance:
                    best = wall
                    best_distance = distance
            if best is None:
                continue
            assigned.append(
                {
                    "opening_id": f"{kind}_{item_index}",
                    "width_m": round(width, 4),
                    "class": kind,
                    "entity_index": best["index"],
                    "room_order": owner[best["index"]],
                }
            )
    return assigned, not_scored


def _door_links(doors: list[dict], floors: list[dict], scene_id: str) -> list[dict]:
    links = []
    seen = set()
    for index, door in enumerate(doors):
        width, _mid = _opening_width(door["vertices"])
        if width < _MIN_OPENING_M:
            continue
        normal = np.asarray(door.get("normal") or [0, 0, 0], dtype=np.float64)
        horizontal = normal[:2]
        norm = float(np.linalg.norm(horizontal))
        if norm < 0.2:
            continue
        horizontal = horizontal / norm
        mid = np.mean(np.asarray(door["vertices"], dtype=np.float64), axis=0)
        hits = []
        for step in (-_DOOR_REACH_M, _DOOR_REACH_M):
            point = mid[:2] + horizontal * step
            for order, floor in enumerate(floors):
                if floor["loop"] is not None and _point_in_polygon(point, floor["loop"]):
                    hits.append(f"{scene_id}_floor_{order:02d}")
                    break
        if len(hits) == 2 and hits[0] != hits[1]:
            key = tuple(sorted(hits))
            if key in seen:
                continue
            seen.add(key)
            links.append(
                {
                    "room_a": key[0],
                    "room_b": key[1],
                    "relationship": "door",
                    "source": "derived_from_door_normal",
                    "opening_id": f"door_{index}",
                }
            )
    return links


def _ceiling_above(floor: dict, ceilings: list[dict]) -> float | None:
    if floor["loop"] is None:
        return None
    heights = []
    for ceiling in ceilings:
        xy = ceiling["point"][:2]
        if _point_in_polygon(xy, floor["loop"]):
            delta = float(ceiling["z"] - floor["z"])
            if 1.8 <= delta <= 6.0:
                heights.append(delta)
    if not heights:
        return None
    return min(heights)


def _near_floor(wall: dict, floor: dict) -> bool:
    if floor["loop"] is None or wall["loop"] is None:
        return False
    mid = np.mean(np.asarray(wall["loop"], dtype=np.float64), axis=0)
    return _point_in_polygon(mid, floor["loop"]) or _distance_to_loop(mid, floor["loop"]) <= 0.35


def _nearest_wall(mid: np.ndarray, room_walls: list[dict], walls: list[dict]) -> str | None:
    by_index = {wall["index"]: wall for wall in walls}
    best = None
    best_distance = _WALL_ATTACH_M
    for room_wall in room_walls:
        wall = by_index[room_wall["entity_index"]]
        if wall["loop"] is None:
            continue
        distance = _distance_to_loop(mid[:2], wall["loop"])
        if distance < best_distance:
            best = room_wall["wall_id"]
            best_distance = distance
    return best


def _opening_width(vertices: list) -> tuple[float, np.ndarray]:
    pts = np.asarray(vertices, dtype=np.float64)
    horizontal = []
    for index in range(len(pts)):
        edge = pts[(index + 1) % len(pts)] - pts[index]
        if abs(float(edge[2])) < 0.25:
            horizontal.append(float(np.linalg.norm(edge)))
    if not horizontal:
        raise ValueError("opening quad has no horizontal edge")
    return float(np.median(horizontal)), np.mean(pts, axis=0)


def _horizontal_span(entity: dict) -> float:
    verts = entity["xyz"]
    best = 0.0
    for face in entity["faces"]:
        for index, start in enumerate(face):
            end = face[(index + 1) % len(face)]
            delta = verts[end] - verts[start]
            if abs(float(delta[2])) < 0.15:
                best = max(best, float(np.linalg.norm(delta[:2])))
    return best


def _boundary_xy(entity: dict) -> np.ndarray | None:
    counts: dict[tuple[int, int], int] = {}
    for face in entity["faces"]:
        for index, start in enumerate(face):
            end = face[(index + 1) % len(face)]
            key = (start, end) if start < end else (end, start)
            counts[key] = counts.get(key, 0) + 1
    neighbours: dict[int, list[int]] = {}
    for (start, end), count in counts.items():
        if count != 1:
            continue
        neighbours.setdefault(start, []).append(end)
        neighbours.setdefault(end, []).append(start)
    if not neighbours:
        return None
    start = min(neighbours)
    loop = [start]
    previous = None
    current = start
    for _ in range(len(neighbours) + 2):
        options = [item for item in neighbours[current] if item != previous]
        if not options:
            break
        nxt = options[0]
        if nxt == start:
            break
        loop.append(nxt)
        previous, current = current, nxt
    if len(loop) < 3:
        return None
    return entity["xyz"][loop][:, :2]


def _shoelace(loop: np.ndarray) -> float:
    x, y = loop[:, 0], loop[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) * 0.5)


def _point_in_polygon(point: np.ndarray, loop: np.ndarray) -> bool:
    x, y = float(point[0]), float(point[1])
    inside = False
    j = len(loop) - 1
    for i in range(len(loop)):
        xi, yi = float(loop[i, 0]), float(loop[i, 1])
        xj, yj = float(loop[j, 0]), float(loop[j, 1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside


def _distance_to_loop(point: np.ndarray, loop: np.ndarray) -> float:
    best = 1e9
    for index in range(len(loop)):
        best = min(best, _distance_to_segment(point, loop[index], loop[(index + 1) % len(loop)]))
    return best


def _distance_to_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    edge = end - start
    length2 = float(np.dot(edge, edge))
    if length2 < 1e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, edge) / length2, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * edge)))


def _parse_ply(path: Path) -> dict:
    data = path.read_bytes()
    header, body = data.split(b"end_header\n", 1)
    text = header.decode("ascii", "replace")
    n_vertices = _element_count(text, "vertex")
    n_faces = _element_count(text, "face")
    xyz = np.empty((n_vertices, 3), dtype=np.float64)
    offset = 0
    for index in range(n_vertices):
        xyz[index] = struct.unpack_from("<ddd", body, offset)
        offset += 27  # 3 float64 + 3 uchar
    faces = []
    for _ in range(n_faces):
        count = body[offset]
        offset += 1
        indices = struct.unpack_from(f"<{count}I", body, offset)
        offset += 4 * count
        faces.append(indices)
    return {"xyz": xyz, "faces": faces}


def _element_count(header: str, name: str) -> int:
    for line in header.splitlines():
        if line.startswith(f"element {name} "):
            return int(line.split()[-1])
    raise ValueError(f"ply has no element {name}")


def _load_quads(scene_dir: Path, kind: str) -> list[dict]:
    path = _find(scene_dir, f"{kind}.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    quads = document.get(kind) or document.get(kind.rstrip("s")) or []
    if not isinstance(quads, list):
        raise ValueError(f"{path} has no {kind} list")
    return quads


def _entity_paths(scene_dir: Path) -> list[Path]:
    paths = [path for path in scene_dir.glob("*.ply") if path.stem.isdigit()]
    entity_dir = scene_dir / "entities"
    if entity_dir.is_dir():
        paths.extend(path for path in entity_dir.glob("*.ply") if path.stem.isdigit())
    return sorted(paths, key=lambda path: int(path.stem))


def _find(scene_dir: Path, name: str) -> Path:
    direct = scene_dir / name
    if direct.is_file():
        return direct
    matches = list(scene_dir.rglob(name))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(name)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert one local HouseLayout3D scene to benchmark ground truth")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    document = convert_scene(args.scene)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(f"wrote {args.out} rooms={len(document['rooms'])} links={len(document['adjacencies'])}")


if __name__ == "__main__":
    main()
