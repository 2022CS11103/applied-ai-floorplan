"""Photo tier: one folder of stills per room, then a geometric property stitch.

A room folder is 2 to 8 images. If it carries synthetic_walls.json, the cloud
is the labelled fixture (metres on the walls), not a solved photograph.
Otherwise the stills go through sequential SfM and the 1.40 m camera-height
prior, which is an estimate, not a tape.

Rooms are laid out with the same layout core as LiDAR. They are connected
only when their floor polygons share an edge. A sketch file is not a pose.
Overlapping floors are rejected. Rooms that do not touch are not given a link.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .fuse import Cloud, align_floor
from .geometry import shoelace
from .layout import _rotate2, _shared_edge, build_layout, layout_to_dict
from .reconstruct import load_photo_cloud
from .render import render_plan

FIXTURE_LABEL = "synthetic/test fixture — not physical benchmark ground truth"
PHOTO_REASONS = (
    "low_texture",
    "weak_overlap",
    "insufficient_points",
    "missing_images",
    "scale_unavailable",
    "no_room_closure",
    "adjacency_unresolved",
    "overlap_rejected",
)
_OVERLAP_M2 = 0.05
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def list_stills(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES and not p.name.startswith("."))


def room_folders(root: Path) -> list[Path]:
    """Child folders that are rooms. Images in the root itself are one room."""
    if not root.is_dir():
        return []
    children = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if list_stills(child) or (child / "synthetic_walls.json").is_file():
            children.append(child)
    return children


def write_room(folder: Path, bounds: tuple[float, float, float, float], n_images: int, seed: int = 0) -> None:
    """Write stills plus the labelled wall rectangle. The stills are not a capture."""
    folder.mkdir(parents=True, exist_ok=True)
    x0, z0, x1, z1 = bounds
    spec = {
        "synthetic_fixture": True,
        "label": FIXTURE_LABEL,
        "x0": x0,
        "z0": z0,
        "x1": x1,
        "z1": z1,
        "height_m": 2.5,
    }
    (folder / "synthetic_walls.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    rng = np.random.default_rng(seed)
    for i in range(n_images):
        image = rng.integers(0, 40, size=(120, 160, 3), dtype=np.uint8)
        cv2.imwrite(str(folder / f"{i + 1:02d}.jpg"), image)


def write_property(root: Path) -> Path:
    """Three rooms and a connector, in one metre frame, sharing walls."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "FIXTURE.txt").write_text(FIXTURE_LABEL + "\n", encoding="utf-8")
    # A misleading sketch must not become the stitch.
    (root / "adjacency.json").write_text(
        json.dumps({"rooms": [], "links": [{"a": "room_01", "b": "room_02", "side": "north"}]}),
        encoding="utf-8",
    )
    write_room(root / "room_01", (0.0, 0.0, 4.0, 3.0), 4, seed=1)
    write_room(root / "connector", (4.0, 0.6, 5.4, 2.4), 3, seed=2)
    write_room(root / "room_02", (5.4, 0.0, 9.0, 3.2), 4, seed=3)
    write_room(root / "room_03", (0.4, 3.0, 3.4, 5.6), 4, seed=4)
    return root


def _ticks(start: float, stop: float, pitch: float) -> np.ndarray:
    """Samples on the pitch, built as an integer count so the step does not drift."""
    i0 = int(round(start / pitch))
    i1 = int(round(stop / pitch))
    return np.arange(i0, i1 + 1, dtype=np.float64) * pitch


def synthetic_cloud(spec: dict) -> Cloud:
    """Wall and floor returns for one rectangle. Y is up. XZ is the plan.

    Points are generated in a frame whose corner is the origin. Float32 cannot
    keep a 5 cm step once the room sits many metres from the origin, and the
    layout then reads a bent wall. The corner is stored and added back onto
    the finished polygon.
    """
    origin_x, origin_z = float(spec["x0"]), float(spec["z0"])
    x0, z0 = 0.0, 0.0
    x1 = float(spec["x1"]) - origin_x
    z1 = float(spec["z1"]) - origin_z
    height = float(spec.get("height_m", 2.5))
    # Match the layout's 5 cm occupancy grid. A 6 cm step aliases into a
    # diagonal and the wall angle comes out wrong on a small room.
    pitch = 0.05
    chunks = [_grid(x0, x1, z0, z1, 0.0, pitch)]
    # Wall bases stay above the floor band so the floor anchor is not pulled
    # onto the walls. The floor sheet itself is the y=0 plane.
    ys = _ticks(0.35, height, pitch)
    xs = _ticks(x0, x1, pitch)
    zs = _ticks(z0, z1, pitch)
    x_near, x_far = float(xs[0]), float(xs[-1])
    z_near, z_far = float(zs[0]), float(zs[-1])
    # A second sheet, 2 cm toward the room, gives the wall a little thickness.
    # One occupied cell is a perfect line, and the layout's local line fit
    # skips those (the cross-axis spread is zero). The sheet stays inside the
    # room so neighbouring rooms do not share the same points.
    thick = 0.02
    for y in ys:
        chunks.append(np.column_stack([xs, np.full(xs.shape, y), np.full(xs.shape, z_near)]))
        chunks.append(np.column_stack([xs, np.full(xs.shape, y), np.full(xs.shape, z_near + thick)]))
        chunks.append(np.column_stack([xs, np.full(xs.shape, y), np.full(xs.shape, z_far)]))
        chunks.append(np.column_stack([xs, np.full(xs.shape, y), np.full(xs.shape, z_far - thick)]))
        chunks.append(np.column_stack([np.full(zs.shape, x_near), np.full(zs.shape, y), zs]))
        chunks.append(np.column_stack([np.full(zs.shape, x_near + thick), np.full(zs.shape, y), zs]))
        chunks.append(np.column_stack([np.full(zs.shape, x_far), np.full(zs.shape, y), zs]))
        chunks.append(np.column_stack([np.full(zs.shape, x_far - thick), np.full(zs.shape, y), zs]))
    xyz = np.concatenate(chunks, axis=0).astype(np.float32)
    return Cloud(
        xyz=xyz,
        frame_index=np.zeros(len(xyz), dtype=np.int32),
        source="synthetic_photo_fixture",
        meta={
            "points": int(len(xyz)),
            "points_raw": int(len(xyz)),
            "synthetic_fixture": True,
            "label": FIXTURE_LABEL,
            "scale": 1.0,
            "scale_source": "synthetic_fixture_meters",
            "scale_estimated": False,
            "degraded": False,
            "reconstruction_reasons": [],
            "frame_origin_xz": [origin_x, origin_z],
        },
    )


def _grid(x0, x1, z0, z1, y, pitch) -> np.ndarray:
    xs = _ticks(x0, x1, pitch)
    zs = _ticks(z0, z1, pitch)
    xx, zz = np.meshgrid(xs, zs)
    return np.column_stack([xx.ravel(), np.full(xx.size, y), zz.ravel()])


def reconstruct_room(folder: Path) -> Cloud:
    stills = list_stills(folder)
    spec_path = folder / "synthetic_walls.json"
    if spec_path.is_file():
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if spec.get("synthetic_fixture") and len(stills) >= 2:
            cloud = synthetic_cloud(spec)
            cloud.meta["images"] = len(stills)
            cloud.meta["images_used"] = min(len(stills), 8)
            if len(stills) > 8:
                cloud.meta["images_capped_at_8"] = True
            return cloud
    if len(stills) < 2:
        cloud = load_photo_cloud(folder)
        cloud.meta["images"] = len(stills)
        return cloud
    # More than eight stills: the tier is defined on 2–8. Keep the first eight.
    cloud = load_photo_cloud(folder)
    cloud.meta["images"] = len(stills)
    if len(stills) > 8:
        cloud.meta["images_capped_at_8"] = True
    return cloud


def _reasons_from(cloud: Cloud, layout_notes: list[str], n_rooms: int) -> list[str]:
    found = set(cloud.meta.get("reconstruction_reasons") or [])
    if cloud.meta.get("scale_source") in {"failed_no_floor_plane", "failed_degenerate_scale"}:
        found.add("scale_unavailable")
    if n_rooms == 0 and len(cloud.xyz) >= 80 and "missing_images" not in found:
        notes = " ".join(layout_notes)
        if "not enough wall points" in notes or "too few points" in notes:
            found.add("insufficient_points")
        found.add("no_room_closure")
    return [reason for reason in PHOTO_REASONS if reason in found]


def _drop_histogram_bin_rotation(layout) -> None:
    """Put an axis-aligned fixture room back on the cloud axes.

    The wall-angle histogram reports the centre of a 1 degree bin, so a room
    whose walls already lie on the axes comes back turned by half a degree.
    Each room is laid out on its own, and that half degree is about its own
    corner, so a shared wall would not fall on one line. Half a degree is the
    bin centre, not a measured twist, and it is removed. A larger angle is a
    real rotation and is kept.
    """
    theta = float(layout.theta)
    if abs(theta) > np.deg2rad(1.0) or not layout.rooms:
        return
    for room in layout.rooms:
        room.polygon = _rotate2(room.polygon, theta)
        for wall in room.walls:
            wall.p0 = _rotate2(np.asarray(wall.p0, dtype=float).reshape(1, 2), theta)[0]
            wall.p1 = _rotate2(np.asarray(wall.p1, dtype=float).reshape(1, 2), theta)[0]
    layout.theta = 0.0


def layout_room(cloud: Cloud, endpoint_mode: str) -> dict:
    if len(cloud.xyz) < 80:
        prop = {"rooms": [], "adjacencies": [], "notes": ["too few points to build a plan"], "manhattan_theta_deg": 0.0}
        return {"prop": prop, "area_on": 0.0, "drift": {"method": ["not run: photo cloud is empty"]}}
    if cloud.meta.get("synthetic_fixture"):
        # Layout runs in the local frame (corner at the origin). The floor
        # anchor is a float32 shift, and that shift knocks a 5 cm wall step
        # into the next bin. The fixture floor is already y=0.
        layout = build_layout(cloud.xyz, tier="photo", endpoint_mode=endpoint_mode)
        _drop_histogram_bin_rotation(layout)
        prop = layout_to_dict(layout, "photo")
        prop = _shift_plan(prop, cloud.meta.get("frame_origin_xz") or [0.0, 0.0])
        return {
            "prop": prop,
            "area_on": float(sum(shoelace(np.array(r["polygon_m"])) for r in prop["rooms"])),
            "drift_report": {"method": ["fixture laid out in a local metre frame; floor already at y=0"]},
            "aligned_points": int(len(cloud.xyz)),
        }
    aligned = align_floor(cloud, enable_anchor=True, enable_loop=False)
    layout = build_layout(aligned["xyz"], tier="photo", endpoint_mode=endpoint_mode)
    prop = layout_to_dict(layout, "photo")
    prop = _put_back_in_cloud_frame(prop, aligned["replay"])
    return {
        "prop": prop,
        "area_on": float(sum(shoelace(np.array(r["polygon_m"])) for r in prop["rooms"])),
        "drift_report": aligned["report"],
        "aligned_points": int(len(aligned["xyz"])),
        "replay": aligned["replay"],
    }


def _put_back_in_cloud_frame(prop: dict, replay: dict) -> dict:
    """Undo the floor-anchor shift so rooms keep the cloud's horizontal frame.

    The anchor moves each room onto its own floor origin. Independent photos
    do not share that origin. The synthetic fixture does share one metre
    frame, and this puts the polygon back into it. A real SfM cloud has no
    second room in that frame, so nothing is matched that was not already there.
    """
    rotation = np.asarray(replay["R"], dtype=float)
    origin = np.asarray(replay["origin"], dtype=float)

    def restore(xz):
        raw = np.array([float(xz[0]), 0.0, float(xz[1])]) @ rotation + origin
        return [round(float(raw[0]), 4), round(float(raw[2]), 4)]

    for room in prop["rooms"]:
        room["polygon_m"] = [restore(p) for p in room["polygon_m"]]
        room["placement"] = {
            "rotation_deg": 0.0,
            "translation_m": [round(float(origin[0]), 4), round(float(origin[2]), 4)],
            "frame": "cloud_xz_before_floor_anchor",
        }
        for wall in room["walls"]:
            wall["p0_m"] = restore(wall["p0_m"])
            wall["p1_m"] = restore(wall["p1_m"])
    return prop


def _shift_plan(prop: dict, origin_xz) -> dict:
    """Move a local-frame plan by the fixture corner, in metres."""
    ox, oz = float(origin_xz[0]), float(origin_xz[1])

    def shift(xz):
        return [round(float(xz[0]) + ox, 4), round(float(xz[1]) + oz, 4)]

    for room in prop["rooms"]:
        room["polygon_m"] = [shift(p) for p in room["polygon_m"]]
        placed = room.get("placement") or {"rotation_deg": 0.0, "translation_m": [0.0, 0.0]}
        tx, tz = placed.get("translation_m") or [0.0, 0.0]
        room["placement"] = {
            "rotation_deg": float(placed.get("rotation_deg", 0.0)),
            "translation_m": [round(float(tx) + ox, 4), round(float(tz) + oz, 4)],
            "frame": "synthetic_fixture_meters",
        }
        for wall in room["walls"]:
            wall["p0_m"] = shift(wall["p0_m"])
            wall["p1_m"] = shift(wall["p1_m"])
    return prop


def convex_intersection_area(poly_a, poly_b) -> float:
    """Area shared by two convex floor polygons. A shared wall is about zero."""
    a = np.asarray(poly_a, dtype=float)
    b = np.asarray(poly_b, dtype=float)
    if len(a) < 3 or len(b) < 3:
        return 0.0
    clipped = a
    for i in range(len(b)):
        p = b[i]
        q = b[(i + 1) % len(b)]
        clipped = _clip_halfplane(clipped, p, q, b.mean(axis=0))
        if len(clipped) < 3:
            return 0.0
    return abs(shoelace(clipped))


def _clip_halfplane(poly: np.ndarray, p: np.ndarray, q: np.ndarray, interior: np.ndarray) -> np.ndarray:
    edge = q - p
    normal = np.array([-edge[1], edge[0]], dtype=float)
    if float(np.dot(interior - p, normal)) < 0:
        normal = -normal
    def inside(point: np.ndarray) -> bool:
        return float(np.dot(point - p, normal)) >= -1e-7

    def cross(s: np.ndarray, e: np.ndarray) -> np.ndarray:
        direction = e - s
        denom = float(np.dot(direction, normal))
        if abs(denom) < 1e-12:
            return s
        t = float(np.dot(p - s, normal) / denom)
        return s + float(np.clip(t, 0.0, 1.0)) * direction

    output = []
    for i in range(len(poly)):
        start = poly[i]
        end = poly[(i + 1) % len(poly)]
        if inside(end):
            if not inside(start):
                output.append(cross(start, end))
            output.append(end)
        elif inside(start):
            output.append(cross(start, end))
    if not output:
        return np.zeros((0, 2))
    return np.asarray(output, dtype=float)


def stitch_rooms(rooms: list[dict]) -> dict:
    """Connect rooms whose polygons already share an edge. Do not invent a pose.

    Each photo room is reconstructed in the frame of its own cloud. The
    synthetic fixture builds every room in one metre frame, so a shared wall
    is a geometric fact. A real SfM room that does not land on that wall is
    left unlinked.
    """
    placed = []
    for room in rooms:
        copy = json.loads(json.dumps(room))
        copy.setdefault(
            "placement",
            {"rotation_deg": 0.0, "translation_m": [0.0, 0.0], "frame": "room_cloud"},
        )
        placed.append(copy)
    overlaps = []
    overlap_area = 0.0
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            area = convex_intersection_area(placed[i]["polygon_m"], placed[j]["polygon_m"])
            if area > _OVERLAP_M2:
                overlaps.append({"a": placed[i]["id"], "b": placed[j]["id"], "area_m2": round(area, 4)})
                overlap_area += area
    links = []
    if not overlaps:
        for i in range(len(placed)):
            for j in range(i + 1, len(placed)):
                gap = _shared_edge(np.array(placed[i]["polygon_m"]), np.array(placed[j]["polygon_m"]))
                if gap is None:
                    continue
                links.append(
                    {
                        "a": placed[i]["id"],
                        "b": placed[j]["id"],
                        "kind": "shared_wall",
                        "gap_m": round(float(gap), 4),
                        "source": "geometry",
                    }
                )
    component = _connected([room["id"] for room in placed], links)
    return {
        "rooms": placed,
        "adjacencies": links,
        "overlaps": overlaps,
        "overlap_m2": round(overlap_area, 4),
        "connected": bool(component) and not overlaps,
    }


def _connected(ids: list[str], links: list[dict]) -> bool:
    if not ids:
        return False
    parent = {i: i for i in ids}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for link in links:
        parent[find(link["a"])] = find(link["b"])
    return len({find(i) for i in ids}) == 1


def run_photo(capture_path: Path, out_dir: Path, endpoint_mode: str = "intersections") -> dict:
    t0 = time.perf_counter()
    capture_path = Path(capture_path)
    if not capture_path.exists() or not capture_path.is_dir():
        return _emit(capture_path, out_dir, [], [], ["missing_images"], {"images": 0}, t0, endpoint_mode, ["photo path is missing or not a folder"])
    children = room_folders(capture_path)
    if len(children) >= 2:
        return _run_property(capture_path, children, out_dir, endpoint_mode, t0)
    folder = capture_path
    if len(children) == 1 and not list_stills(capture_path):
        folder = children[0]
    return _run_single(folder, out_dir, endpoint_mode, t0, capture_id=capture_path.name)


def _run_single(folder: Path, out_dir: Path, endpoint_mode: str, t0: float, capture_id: str) -> dict:
    cloud = reconstruct_room(folder)
    laid = layout_room(cloud, endpoint_mode)
    prop = laid["prop"]
    reasons = _reasons_from(cloud, prop["notes"], len(prop["rooms"]))
    if cloud.meta.get("scale_estimated") and prop["rooms"]:
        reasons = [r for r in reasons if r != "no_room_closure"]
    if cloud.meta.get("synthetic_fixture"):
        pass
    elif cloud.meta.get("scale_estimated") and "scale_unavailable" not in reasons and prop["rooms"]:
        # A closed room from the height prior is still an estimate. Keep it,
        # and say so. It is not marked ok unless scale succeeded and the
        # layout closed, which the prior can do. The prior is labelled.
        pass
    notes = list(prop["notes"])
    if cloud.meta.get("synthetic_fixture"):
        notes.insert(0, FIXTURE_LABEL)
    if cloud.meta.get("scale_estimated"):
        notes.append("photo scale is the camera-height prior, not a tape")
    status = "degraded" if reasons or not prop["rooms"] else "ok"
    if not prop["rooms"]:
        status = "degraded"
    for room in prop["rooms"]:
        room.setdefault(
            "placement",
            {"rotation_deg": 0.0, "translation_m": [0.0, 0.0], "frame": "room_cloud"},
        )
    quality = {k: v for k, v in cloud.meta.items() if not isinstance(v, np.ndarray)}
    quality["points_aligned"] = laid.get("aligned_points", 0)
    quality["overlap_m2"] = 0.0
    quality["stitch"] = "single_room"
    return _emit(
        folder if capture_id == folder.name else Path(capture_id),
        out_dir,
        prop["rooms"],
        [],
        reasons,
        quality,
        t0,
        endpoint_mode,
        notes,
        capture_id=capture_id,
        source=cloud.source,
        status=status,
    )


def _run_property(root: Path, children: list[Path], out_dir: Path, endpoint_mode: str, t0: float) -> dict:
    rooms = []
    notes = []
    reasons: set[str] = set()
    images = 0
    points = 0
    synthetic = False
    per_room = []
    if (root / "adjacency.json").is_file():
        notes.append("adjacency.json was not used; links come from room polygons")
    if (root / "FIXTURE.txt").is_file():
        notes.append(FIXTURE_LABEL)
        synthetic = True
    for folder in children:
        cloud = reconstruct_room(folder)
        images += int(cloud.meta.get("images") or 0)
        points += int(cloud.meta.get("points") or 0)
        synthetic = synthetic or bool(cloud.meta.get("synthetic_fixture"))
        for reason in cloud.meta.get("reconstruction_reasons") or []:
            reasons.add(reason)
        laid = layout_room(cloud, endpoint_mode)
        got = laid["prop"]["rooms"]
        if not got:
            reasons.add("no_room_closure")
            notes.append(f"{folder.name}: " + "; ".join(laid["prop"]["notes"]))
            per_room.append({"id": folder.name, "closed": False, "points": int(cloud.meta.get("points") or 0)})
            continue
        room = got[0]
        room["id"] = folder.name
        for wall_i, wall in enumerate(room["walls"]):
            wall["id"] = f"{folder.name}_w{wall_i}"
        rooms.append(room)
        per_room.append({"id": folder.name, "closed": True, "points": int(cloud.meta.get("points") or 0)})
    stitched = stitch_rooms(rooms)
    if stitched["overlaps"]:
        reasons.add("overlap_rejected")
        notes.append("floor polygons overlap; the stitch was rejected")
    elif rooms and not stitched["connected"]:
        reasons.add("adjacency_unresolved")
        notes.append("no shared wall connects every room; adjacency was not invented")
    if not rooms:
        reasons.add("no_room_closure")
    ordered = [reason for reason in PHOTO_REASONS if reason in reasons]
    status = "ok" if rooms and stitched["connected"] and not ordered else "degraded"
    # A labelled height-prior failure on one room is already in `reasons`.
    # Overlap and a broken adjacency must not look successful.
    if "overlap_rejected" in ordered or "adjacency_unresolved" in ordered or "no_room_closure" in ordered:
        status = "degraded"
    footprint = 0.0
    if status == "ok":
        footprint = float(sum(room["floor_area_m2"]["value"] for room in stitched["rooms"]))
    quality = {
        "images": images,
        "points": points,
        "rooms_reconstructed": len(rooms),
        "rooms_in_folders": len(children),
        "per_room": per_room,
        "overlap_m2": stitched["overlap_m2"],
        "overlaps": stitched["overlaps"],
        "stitch": "shared_wall" if status == "ok" else "rejected",
        "synthetic_fixture": synthetic,
        "scale_source": "synthetic_fixture_meters" if synthetic else "per_room",
    }
    return _emit(
        root,
        out_dir,
        stitched["rooms"],
        [] if stitched["overlaps"] else stitched["adjacencies"],
        ordered,
        quality,
        t0,
        endpoint_mode,
        notes,
        capture_id=root.name,
        source="synthetic_photo_fixture" if synthetic else "photo_sfm",
        status=status,
        footprint=footprint,
    )


def _emit(
    capture_path: Path,
    out_dir: Path,
    rooms: list[dict],
    adjacencies: list[dict],
    reasons: list[str],
    quality: dict,
    t0: float,
    endpoint_mode: str,
    notes: list[str],
    capture_id: str | None = None,
    source: str = "photo_sfm",
    status: str | None = None,
    footprint: float | None = None,
) -> dict:
    if status is None:
        status = "degraded" if reasons or not rooms else "ok"
    if not rooms:
        status = "degraded"
    if footprint is None:
        footprint = float(sum(room["floor_area_m2"]["value"] for room in rooms)) if rooms and status == "ok" else 0.0
    document = {
        "schema_version": "1.0",
        "pipeline_version": __version__,
        "status": status,
        "degraded_reasons": [reason for reason in PHOTO_REASONS if reason in reasons],
        "capture_id": capture_id or Path(capture_path).name,
        "tier": "photo",
        "units": "meters",
        "capture_kind": "photo_property" if adjacencies or quality.get("rooms_in_folders", 0) else "photo_folder",
        "source": source,
        "endpoint_mode": endpoint_mode,
        "timing_s": {"total_s": round(time.perf_counter() - t0, 3)},
        "quality": quality,
        "drift": {
            "method": (
                [
                    "fixture laid out in a local metre frame; floor already at y=0",
                    "no property-level pose graph",
                ]
                if quality.get("synthetic_fixture")
                else ["per-room floor alignment", "no property-level pose graph"]
            ),
            "floor_tilt_deg": None,
            "floor_residual_m": None,
            "chunk_floor_offsets_m": [],
            "loop": {"fired": False, "reason": "disabled"},
            "footprint_area_m2_anchor_on": round(footprint, 4),
            "footprint_area_m2_anchor_off": round(footprint, 4),
            "note": (
                "Synthetic fixture. Not a floor-anchor replay and not a lidar walk."
                if quality.get("synthetic_fixture")
                else "Each photo room is aligned on its own. Rooms are not a lidar walk."
            ),
        },
        "property": {
            "rooms": rooms,
            "adjacencies": adjacencies,
            "footprint_area_m2": round(footprint, 4),
            "manhattan_theta_deg": None,
        },
        "scope_line_items": [],
        "concealed_damage": [],
        "notes": notes,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plan.json").write_text(json.dumps(_jsonable(document), indent=2), encoding="utf-8")
    render_plan(document, out_dir / "plan.png")
    lines = [
        f"capture: {document['capture_id']}",
        f"tier: photo",
        f"status: {status}",
        f"rooms: {len(rooms)}",
        f"reasons: {', '.join(document['degraded_reasons']) or 'none'}",
        FIXTURE_LABEL if quality.get("synthetic_fixture") else "photo",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return document


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj
