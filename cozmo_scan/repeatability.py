"""Compare two real captures of the same room.

Even and odd frames of one walk are not this comparison. A missing second
capture stays BLOCKED. Thresholds are the PDF constants in benchmark_eval.
"""

from __future__ import annotations

from .benchmark_eval import REPEATABILITY_ABS_M, REPEATABILITY_REL


def _length(wall: dict) -> float | None:
    value = wall.get("length_m")
    if isinstance(value, dict):
        value = value.get("value")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _walls(plan: dict | None, room_id: str | None) -> list[dict]:
    if not plan:
        return []
    rooms = (plan.get("property") or {}).get("rooms") or plan.get("rooms") or []
    rows = []
    for room in rooms:
        this = room.get("id") or room.get("room_id")
        if room_id and this != room_id:
            continue
        for wall in room.get("walls") or []:
            rows.append(
                {
                    "room": this,
                    "wall": wall.get("id") or wall.get("wall_id"),
                    "length_m": _length(wall),
                    "ground_truth_length": wall.get("ground_truth_length_m") or wall.get("ground_truth_m"),
                }
            )
    return rows


def _match(wall: dict, others: list[dict], used: set[int]) -> int | None:
    for index, other in enumerate(others):
        if index in used:
            continue
        if wall["room"] == other["room"] and wall["wall"] and wall["wall"] == other["wall"]:
            return index
    return None


def compare_two_captures(capture_a_plan: dict | None, capture_b_plan: dict | None, room_id: str | None = None) -> dict:
    """One row per wall. An unmatched wall fails. Capture B missing is BLOCKED."""
    if capture_b_plan is None:
        return {
            "status": "BLOCKED",
            "reason": "missing_capture_b",
            "rows": [],
            "proxy_experiment": False,
            "threshold": {"absolute_m": REPEATABILITY_ABS_M, "relative": REPEATABILITY_REL},
        }
    if capture_a_plan is None:
        return {
            "status": "BLOCKED",
            "reason": "missing_capture_a",
            "rows": [],
            "proxy_experiment": False,
            "threshold": {"absolute_m": REPEATABILITY_ABS_M, "relative": REPEATABILITY_REL},
        }
    left = _walls(capture_a_plan, room_id)
    right = _walls(capture_b_plan, room_id)
    if not left and not right:
        return {
            "status": "BLOCKED",
            "reason": "no walls in either capture",
            "rows": [],
            "proxy_experiment": False,
            "threshold": {"absolute_m": REPEATABILITY_ABS_M, "relative": REPEATABILITY_REL},
        }
    used: set[int] = set()
    rows = []
    for wall in left:
        partner = _match(wall, right, used)
        if partner is None or wall["length_m"] is None or right[partner]["length_m"] is None:
            rows.append(
                {
                    "room": wall["room"],
                    "wall": wall["wall"],
                    "capture_a": wall["length_m"],
                    "capture_b": None if partner is None else right[partner]["length_m"],
                    "absolute_difference": None,
                    "relative_difference": None,
                    "pass": False,
                    "matched": False,
                }
            )
            if partner is not None:
                used.add(partner)
            continue
        used.add(partner)
        other = right[partner]
        error = abs(wall["length_m"] - other["length_m"])
        truth = wall.get("ground_truth_length") or other.get("ground_truth_length")
        if truth in (None, 0):
            truth = max(abs(wall["length_m"]), abs(other["length_m"]))
        relative = None if not truth else error / abs(float(truth))
        passed = bool(error <= REPEATABILITY_ABS_M or (relative is not None and relative <= REPEATABILITY_REL))
        rows.append(
            {
                "room": wall["room"],
                "wall": wall["wall"],
                "capture_a": wall["length_m"],
                "capture_b": other["length_m"],
                "absolute_difference": round(error, 4),
                "relative_difference": None if relative is None else round(relative, 6),
                "pass": passed,
                "matched": True,
            }
        )
    for index, other in enumerate(right):
        if index in used:
            continue
        rows.append(
            {
                "room": other["room"],
                "wall": other["wall"],
                "capture_a": None,
                "capture_b": other["length_m"],
                "absolute_difference": None,
                "relative_difference": None,
                "pass": False,
                "matched": False,
            }
        )
    failed = [row for row in rows if not row["pass"]]
    return {
        "status": "FAIL" if failed else "PASS",
        "reason": f"{len(rows) - len(failed)}/{len(rows)} walls within 1 cm or 0.5%",
        "rows": rows,
        "proxy_experiment": False,
        "threshold": {"absolute_m": REPEATABILITY_ABS_M, "relative": REPEATABILITY_REL},
    }
