"""Per-surface damage from color, not from a trained detector.

The sample captures are not a labelled damage set, and this machine has
no second model weights to fetch. The detector is a local color residual
on the wall surface:

* wall points are binned, 10 cm along the wall by 10 cm in height
* each bin keeps a median color
* a bin is anomalous when it differs from the median of its neighbors
  (a lighting gradient moves the neighbors too, a stain does not)
* classes are stain, moisture, discoloration
* concealed-damage rules are explicit and reported even when they do not fire

Scope lines are emitted only for regions that pass the area cut.
"""

from __future__ import annotations

import cv2
import numpy as np

from .capture import Capture
from .fuse import apply_replay
from .geometry import backproject, meas, quat_to_matrix


RULES = (
    "base_stain_moisture_path",
    "sill_stain_under_window",
)


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    img = rgb.reshape(-1, 1, 3).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)
    return lab


def detect_on_points(xyz: np.ndarray, colors: np.ndarray, rooms: list[dict], tier: str) -> list[dict]:
    """rooms is the JSON room list (polygons and walls in the same XZ frame as xyz)."""
    if colors is None or len(xyz) == 0 or len(colors) != len(xyz):
        return []
    rel = 0.0 if tier == "lidar" else (0.03 if tier == "video" else 0.08)
    abs_floor = 0.015 if tier == "lidar" else 0.03
    found = []
    for room in rooms:
        for wall in room["walls"]:
            regions = _wall_regions(xyz, colors, wall, room["id"])
            for reg in regions:
                reg["extent_m2"] = meas(reg["extent"], reg["sigma"], abs_floor, rel)
                reg.pop("extent")
                reg.pop("sigma")
                found.append(reg)
    return found


def _wall_regions(xyz: np.ndarray, colors: np.ndarray, wall: dict, room_id: str) -> list[dict]:
    p0 = np.array(wall["p0_m"], dtype=np.float64)
    p1 = np.array(wall["p1_m"], dtype=np.float64)
    direction = p1 - p0
    length = np.linalg.norm(direction)
    if length < 0.4:
        return []
    direction = direction / length
    normal = np.array([-direction[1], direction[0]])
    xz = xyz[:, [0, 2]]
    rel = xz - p0
    along = rel @ direction
    across = rel @ normal
    h = xyz[:, 1]
    sel = (np.abs(across) < 0.1) & (along > 0.05) & (along < length - 0.05) & (h > 0.15) & (h < 1.9)
    if sel.sum() < 30:
        return []
    along_s = along[sel]
    h_s = h[sel]
    cols = colors[sel]
    cell = 0.10
    ia = np.floor(along_s / cell).astype(np.int32)
    ih = np.floor(h_s / cell).astype(np.int32)
    lab = _rgb_to_lab(cols)
    # aggregate median per cell — mean is enough at this resolution and is stable
    keys = {}
    for a, hh, color in zip(ia, ih, lab):
        keys.setdefault((int(a), int(hh)), []).append(color)
    if len(keys) < 8:
        return []
    cells = {k: np.median(np.stack(v), axis=0) for k, v in keys.items() if len(v) >= 2}
    if len(cells) < 8:
        return []
    # local residual against neighbor median
    anomalous = []
    for (a, hh), color in cells.items():
        neigh = []
        for da in range(-2, 3):
            for dh in range(-2, 3):
                if da == 0 and dh == 0:
                    continue
                other = cells.get((a + da, hh + dh))
                if other is not None:
                    neigh.append(other)
        if len(neigh) < 4:
            continue
        base = np.median(np.stack(neigh), axis=0)
        delta = float(np.linalg.norm(color - base))
        # OpenCV LAB units. ~18 is a clear local patch, not a shadow gradient.
        # 32 in OpenCV LAB is a hard local step, not a lighting falloff.
        if delta >= 32 and (color[0] - base[0]) < -12:
            anomalous.append((a, hh, delta, color, base))
    if len(anomalous) < 8:
        return []
    # connected components on the cell grid (4-neighbour)
    remaining = {(a, hh): (delta, color, base) for a, hh, delta, color, base in anomalous}
    regions = []
    while remaining:
        start = next(iter(remaining))
        stack = [start]
        comp = []
        while stack:
            cur = stack.pop()
            if cur not in remaining:
                continue
            comp.append((cur, remaining.pop(cur)))
            a, hh = cur
            for nxt in ((a + 1, hh), (a - 1, hh), (a, hh + 1), (a, hh - 1)):
                if nxt in remaining:
                    stack.append(nxt)
        if len(comp) < 8:
            continue
        area = len(comp) * cell * cell
        if area < 0.12:
            continue
        # compact: reject a speckle field spread over the whole wall
        aa = [c[0][0] for c in comp]
        hh = [c[0][1] for c in comp]
        box = (max(aa) - min(aa) + 1) * (max(hh) - min(hh) + 1) * cell * cell
        if area / max(box, 1e-6) < 0.35:
            continue
        deltas = np.array([c[1][0] for c in comp])
        colors_c = np.stack([c[1][1] for c in comp])
        bases = np.stack([c[1][2] for c in comp])
        dL = float(np.median(colors_c[:, 0] - bases[:, 0]))
        db = float(np.median(colors_c[:, 2] - bases[:, 2]))
        h0 = min(hh) * cell
        h1 = (max(hh) + 1) * cell
        # Lighting is not damage. A region has to be clearly darker than the
        # wall around it. Moisture is that, plus a yellow shift, at the base.
        if dL > -15:
            continue
        if h1 <= 0.45 and db > 8:
            klass = "moisture"
        else:
            klass = "stain"
        along_c = float(np.mean(aa) + 0.5) * cell
        center = p0 + direction * along_c
        sigma = max(0.4 * area, 0.01)  # color extent is coarser than the tape
        regions.append(
            {
                "id": f"{wall['id']}_d{len(regions)}",
                "room_id": room_id,
                "surface_id": wall["id"],
                "class": klass,
                "extent": float(area),
                "sigma": float(sigma),
                "height_band_m": [round(float(h0), 3), round(float(h1), 3)],
                "centroid_m": [round(float(center[0]), 3), round(float(center[1]), 3)],
                "mean_delta_e": round(float(deltas.mean()), 2),
            }
        )
    return regions


def concealed_flags(rooms: list[dict]) -> list[dict]:
    flags = []
    for room in rooms:
        damage = room.get("damage", [])
        for wall in room["walls"]:
            on_wall = [d for d in damage if d["surface_id"] == wall["id"]]
            windows = [o for o in wall["openings"] if o["class"] == "window"]
            base_hit = any(
                d["class"] in {"stain", "moisture"}
                and d["height_band_m"][0] <= 0.30
                and d["extent_m2"]["value"] >= 0.04
                for d in on_wall
            )
            flags.append(
                {
                    "id": f"{wall['id']}_base_stain_moisture_path",
                    "surface_id": wall["id"],
                    "rule": "base_stain_moisture_path",
                    "fired": bool(base_hit),
                    "evidence": (
                        "stain or moisture intersects 0-0.30 m above the floor and covers at least 0.04 m^2"
                        if base_hit
                        else "no stain or moisture region met the base-of-wall predicate"
                    ),
                }
            )
            sill_hit = False
            if windows and on_wall:
                sill_hit = any(d["class"] in {"stain", "moisture"} and d["height_band_m"][1] >= 0.7 for d in on_wall)
            flags.append(
                {
                    "id": f"{wall['id']}_sill_stain_under_window",
                    "surface_id": wall["id"],
                    "rule": "sill_stain_under_window",
                    "fired": bool(sill_hit),
                    "evidence": (
                        "a window is on this wall and a stain reaches the sill band"
                        if sill_hit
                        else "no window, or no stain in the sill band"
                    ),
                }
            )
    return flags


def scope_items(rooms: list[dict]) -> list[dict]:
    catalog = {
        "stain": ("INTERIOR_PAINT", "m2"),
        "discoloration": ("INTERIOR_PAINT", "m2"),
        "moisture": ("MOISTURE_OPEN_AND_INSPECT", "m2"),
    }
    items = []
    for room in rooms:
        for d in room.get("damage", []):
            code, unit = catalog.get(d["class"], ("INSPECT", "m2"))
            items.append(
                {
                    "id": f"scope_{d['id']}",
                    "surface_id": d["surface_id"],
                    "damage_id": d["id"],
                    "code": code,
                    "class": d["class"],
                    "quantity": d["extent_m2"]["value"],
                    "unit": unit,
                    "ci95": [d["extent_m2"]["ci95_low"], d["extent_m2"]["ci95_high"]],
                }
            )
    return items


def colorize_lidar(capture: Capture, replay: dict, max_frames: int = 28, pixel_stride: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Second pass: RGB color on a thinned depth cloud, in the plan frame."""
    import numpy as np
    from PIL import Image

    video = capture.video_path
    if video is None:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    pose_map = capture.pose_by_frame()
    pose_index = {p.frame: i for i, p in enumerate(capture.poses)}
    frames = sorted(p.stem for p in capture.depth_dir.glob("*.png") if p.stem in pose_map)
    if not frames:
        cap.release()
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    step = max(1, len(frames) // max_frames)
    chosen = frames[::step][:max_frames]
    chunks = []
    cols = []
    ids = []
    rgb_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    rgb_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1440
    for fr in chosen:
        depth = np.array(Image.open(capture.depth_dir / f"{fr}.png"))
        z = depth.astype(np.float32) / 1000.0
        conf_path = capture.confidence_dir / f"{fr}.png"
        if conf_path.exists():
            conf = np.array(Image.open(conf_path))
            mask = (conf >= 2) & (z > 0.3) & (z < 4.5)
        else:
            mask = (z > 0.3) & (z < 4.5)
        vs, us = np.nonzero(mask)
        if len(vs) < 20:
            continue
        vs = vs[::pixel_stride]
        us = us[::pixel_stride]
        frame_i = int(fr)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dh, dw = depth.shape
        u_rgb = np.clip((us.astype(np.float64) + 0.5) * (rgb_w / dw) - 0.5, 0, rgb_w - 1).astype(np.int32)
        v_rgb = np.clip((vs.astype(np.float64) + 0.5) * (rgb_h / dh) - 0.5, 0, rgb_h - 1).astype(np.int32)
        color = rgb[v_rgb, u_rgb]
        pose = pose_map[fr]
        zz = z[vs, us].astype(np.float64)
        sx = dw / float(rgb_w)
        sy = dh / float(rgb_h)
        cam = backproject(us.astype(np.float64), vs.astype(np.float64), zz, pose.fx * sx, pose.fy * sy, pose.cx * sx, pose.cy * sy)
        R = quat_to_matrix(pose.q)
        world = cam @ R.T + pose.t
        chunks.append(world.astype(np.float32))
        cols.append(color)
        ids.append(np.full(len(world), pose_index[fr], dtype=np.int32))
    cap.release()
    if not chunks:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    raw = np.concatenate(chunks)
    colors = np.concatenate(cols)
    frame_index = np.concatenate(ids)
    aligned = apply_replay(raw, frame_index, replay)
    return aligned.astype(np.float32), colors
