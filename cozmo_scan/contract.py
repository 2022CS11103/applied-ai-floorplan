"""One plan document for LiDAR, video, and photo.

This pass runs after a tier has already built its rooms. It does not
reconstruct, stitch, or invent a measurement. It gives every tier the same
room-graph fields and a stable order.
"""

from __future__ import annotations

import numpy as np

_MEASUREMENT_KEYS = ("value", "sigma", "ci95_low", "ci95_high")


def apply_contract(document: dict) -> dict:
    """Fill the common room-graph fields on a plan that was already built.

    Existing ids, polygons, and measurement objects are kept. Aliases are
    added so a reader can use one set of names on every tier. Adjacency
    entries that do not name two rooms in this plan are dropped. None are added.
    """
    prop = document.setdefault("property", {})
    rooms = list(prop.get("rooms") or [])
    for room in rooms:
        _normalize_room(room)
    rooms.sort(key=lambda room: room["room_id"])
    prop["rooms"] = rooms

    known = {room["room_id"] for room in rooms}
    links = []
    for link in list(prop.get("adjacencies") or []):
        normalized = _normalize_link(link, rooms)
        if normalized is None:
            continue
        if normalized["room_a"] not in known or normalized["room_b"] not in known:
            continue
        if normalized["room_a"] == normalized["room_b"]:
            continue
        links.append(normalized)
    links.sort(key=lambda link: (link["room_a"], link["room_b"], link["relationship"]))
    prop["adjacencies"] = links

    reasons = list(document.get("degraded_reasons") or [])
    claimed_ok = document.get("status") == "ok" and not reasons
    if not rooms:
        document["status"] = "degraded"
        if claimed_ok and "no_room_closure" not in reasons:
            reasons.append("no_room_closure")
    document["degraded_reasons"] = reasons
    return document


def _normalize_room(room: dict) -> None:
    room_id = str(room.get("room_id") or room.get("id") or "")
    room["id"] = room_id
    room["room_id"] = room_id
    polygon = room.get("floor_polygon", room.get("polygon_m"))
    room["polygon_m"] = polygon
    room["floor_polygon"] = polygon
    closed = bool(polygon) and len(polygon) >= 3 and bool(room.get("walls"))
    room["status"] = "ok" if closed else "degraded"
    room["reasons"] = [] if closed else ["no_room_closure"]
    area = room.get("floor_area_m2")
    ceiling = room.get("ceiling_height_m")
    room["measurements"] = {
        "floor_area_m2": area,
        "ceiling_height_m": ceiling,
    }
    walls = list(room.get("walls") or [])
    for wall in walls:
        _normalize_wall(wall)
    walls.sort(key=lambda wall: wall.get("id") or "")
    room["walls"] = walls


def _normalize_wall(wall: dict) -> None:
    wall["measurements"] = {
        "length_m": wall.get("length_m"),
        "height_m": wall.get("height_m"),
    }
    openings = list(wall.get("openings") or [])
    for opening in openings:
        opening["measurements"] = {
            "width_m": opening.get("width_m"),
            "sill_height_m": opening.get("sill_height_m"),
            "head_height_m": opening.get("head_height_m"),
        }
    openings.sort(key=lambda opening: opening.get("id") or "")
    wall["openings"] = openings


def _normalize_link(link: dict, rooms: list[dict]) -> dict | None:
    relationship = link.get("relationship") or link.get("kind")
    if not relationship:
        return None
    room_a = str(link.get("room_a") or link.get("a") or "")
    room_b = str(link.get("room_b") or link.get("b") or "")
    if not room_a or not room_b:
        return None
    by_id = {room["room_id"]: room for room in rooms}
    shared = link.get("shared_length_m")
    if shared is None and room_a in by_id and room_b in by_id:
        shared = _shared_length(by_id[room_a].get("floor_polygon"), by_id[room_b].get("floor_polygon"))
    out = dict(link)
    out["a"] = room_a
    out["b"] = room_b
    out["room_a"] = room_a
    out["room_b"] = room_b
    out["kind"] = relationship
    out["relationship"] = relationship
    out["source"] = link.get("source") or "geometry"
    out["shared_length_m"] = None if shared is None else round(float(shared), 4)
    return out


def _shared_length(poly_a, poly_b, tol: float = 0.25) -> float | None:
    """Along-wall overlap of the closest edge pair. None when the edges do not meet.

    This describes a link the tier already recorded. It does not create one.
    The length is geometric metadata, not a new confidence interval.
    """
    a = np.asarray(poly_a if poly_a is not None else [], dtype=float)
    b = np.asarray(poly_b if poly_b is not None else [], dtype=float)
    if len(a) < 2 or len(b) < 2:
        return None
    best_gap = None
    best_overlap = None
    for i in range(len(a)):
        p0 = a[i]
        p1 = a[(i + 1) % len(a)]
        edge = p1 - p0
        length = float(np.linalg.norm(edge))
        if length < 0.4:
            continue
        direction = edge / length
        normal = np.array([-direction[1], direction[0]])
        for j in range(len(b)):
            q0 = b[j]
            q1 = b[(j + 1) % len(b)]
            gap = abs(float(np.dot(q0 - p0, normal)))
            if gap > tol:
                continue
            t0 = float(np.dot(q0 - p0, direction))
            t1 = float(np.dot(q1 - p0, direction))
            lo, hi = sorted((t0, t1))
            overlap = min(hi, length) - max(lo, 0.0)
            if overlap <= 0.4:
                continue
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_overlap = overlap
    return best_overlap
