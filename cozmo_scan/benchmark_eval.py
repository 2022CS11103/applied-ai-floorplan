"""Score a plan.json against laser/tape ground truth.

This module does not reconstruct anything. A missing ground-truth file stays
blocked. A synthetic fixture only checks the arithmetic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Assessment thresholds. Tests may pass a Thresholds object with other values.
OPENING_MAX_ERROR_M = 0.02
OPENING_PASS_RATE = 0.85
CEILING_MAX_ERROR_M = 0.015
CEILING_REPEAT_SPREAD_M = 0.01
REPEATABILITY_ABS_M = 0.01
REPEATABILITY_REL = 0.005
PHOTO_FOOTPRINT_REL = 0.08
PHOTO_WALL_REL = 0.08
VIDEO_WALL_REL = 0.03

EVALUATOR_LABEL = "synthetic evaluator fixture — not physical ground truth"
BLOCKED_REASON = "physical laser/tape ground truth is not available"


class Thresholds:
    def __init__(
        self,
        opening_max_error_m: float = OPENING_MAX_ERROR_M,
        opening_pass_rate: float = OPENING_PASS_RATE,
        ceiling_max_error_m: float = CEILING_MAX_ERROR_M,
        ceiling_repeat_spread_m: float = CEILING_REPEAT_SPREAD_M,
        repeatability_abs_m: float = REPEATABILITY_ABS_M,
        repeatability_rel: float = REPEATABILITY_REL,
        photo_footprint_rel: float = PHOTO_FOOTPRINT_REL,
        photo_wall_rel: float = PHOTO_WALL_REL,
        video_wall_rel: float = VIDEO_WALL_REL,
        wall_abs_m: float | None = None,
    ):
        self.opening_max_error_m = opening_max_error_m
        self.opening_pass_rate = opening_pass_rate
        self.ceiling_max_error_m = ceiling_max_error_m
        self.ceiling_repeat_spread_m = ceiling_repeat_spread_m
        self.repeatability_abs_m = repeatability_abs_m
        self.repeatability_rel = repeatability_rel
        self.photo_footprint_rel = photo_footprint_rel
        self.photo_wall_rel = photo_wall_rel
        self.video_wall_rel = video_wall_rel
        self.wall_abs_m = wall_abs_m


def load_json(path: Path) -> dict:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{path} must be a JSON object")
    return doc


def load_prediction(doc: dict) -> dict:
    """A prediction is a plan.json. There is no second format."""
    if not isinstance(doc, dict) or "property" not in doc:
        raise ValueError("prediction must be a plan.json with property.rooms")
    rooms = doc["property"].get("rooms")
    if not isinstance(rooms, list):
        raise ValueError("prediction property.rooms must be a list")
    return doc


def load_ground_truth(doc: dict) -> dict:
    if not isinstance(doc, dict):
        raise ValueError("ground truth must be a JSON object")
    rooms = doc.get("rooms")
    if not isinstance(rooms, list):
        raise ValueError("ground truth needs a rooms list")
    for room in rooms:
        if not isinstance(room, dict) or "room_id" not in room:
            raise ValueError("each ground-truth room needs room_id")
        for wall in room.get("walls") or []:
            if "length_m" not in wall:
                raise ValueError(f"room {room['room_id']} has a wall without length_m")
        for opening in room.get("openings") or []:
            if "width_m" not in opening or "wall_id" not in opening:
                raise ValueError(f"room {room['room_id']} has an opening without wall_id and width_m")
    if "adjacencies" in doc and not isinstance(doc["adjacencies"], list):
        raise ValueError("ground truth adjacencies must be a list")
    return doc


def evaluate(
    prediction: dict,
    ground_truth: dict | None = None,
    repeat: dict | None = None,
    drift_off: dict | None = None,
    thresholds: Thresholds | None = None,
    case_id: str | None = None,
) -> dict:
    prediction = load_prediction(prediction)
    limits = thresholds or Thresholds()
    tier = prediction.get("tier") or (ground_truth or {}).get("tier")
    case = case_id or (ground_truth or {}).get("case_id") or prediction.get("capture_id")
    if ground_truth is not None:
        ground_truth = load_ground_truth(ground_truth)
    report = {
        "case_id": case,
        "tier": tier,
        "evaluator_only": bool(ground_truth and ground_truth.get("evaluator_only")),
        "assessment_gate": _assessment_gate(ground_truth),
        "rooms": [],
        "wall_accuracy": _unavailable("no ground truth"),
        "opening_accuracy": _unavailable("no ground truth"),
        "ceiling_accuracy": _unavailable("no ground truth"),
        "repeatability": _repeatability(prediction, repeat, limits) if repeat is not None else _unavailable("no second capture"),
        "photo_property": _not_applicable("tier is not photo") if tier != "photo" else _unavailable("no ground truth"),
        "calibration": _calibration(prediction, ground_truth),
        "drift": _drift(prediction, drift_off, limits),
    }
    if ground_truth is None:
        report["blocked_metrics"] = _blocked(report)
        return report
    pairs, room_notes = _match_rooms(prediction, ground_truth)
    report["rooms"] = room_notes
    report["wall_accuracy"] = _walls(pairs, tier, limits)
    report["opening_accuracy"] = _openings(pairs, limits)
    report["ceiling_accuracy"] = _ceilings(pairs, repeat, limits)
    if tier == "photo":
        report["photo_property"] = _photo(prediction, ground_truth, report["wall_accuracy"], limits)
    elif tier == "video":
        report["wall_accuracy"]["video_threshold_rel"] = limits.video_wall_rel
    report["blocked_metrics"] = _blocked(report)
    return report


def evaluate_case(case_dir: Path, thresholds: Thresholds | None = None) -> dict:
    folder = Path(case_dir)
    prediction = load_prediction(load_json(folder / "prediction.json"))
    ground_truth = load_ground_truth(load_json(folder / "ground_truth.json"))
    repeat = load_json(folder / "prediction_repeat.json") if (folder / "prediction_repeat.json").is_file() else None
    drift_off = load_json(folder / "prediction_drift_off.json") if (folder / "prediction_drift_off.json").is_file() else None
    return evaluate(prediction, ground_truth, repeat, drift_off, thresholds, case_id=ground_truth.get("case_id") or folder.name)


def write_report(report: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "report.txt").write_text(render_text(report), encoding="utf-8")


def render_text(report: dict) -> str:
    lines = [
        f"case: {report.get('case_id')}",
        f"tier: {report.get('tier')}",
        f"assessment gate: {report['assessment_gate']['status']} — {report['assessment_gate']['reason']}",
        "",
    ]
    for name in (
        "wall_accuracy",
        "opening_accuracy",
        "ceiling_accuracy",
        "repeatability",
        "photo_property",
    ):
        block = report.get(name) or {}
        lines.append(f"[{name}] {block.get('status')}")
        if block.get("reason"):
            lines.append(f"  reason: {block['reason']}")
        summary = []
        for key in ("n_evaluated", "n_pass", "pass_rate", "n_missed", "n_phantom", "worst_abs_error_m"):
            if key in block and block[key] is not None:
                summary.append(f"{key}={block[key]}")
        if summary:
            lines.append("  " + "  ".join(summary))
        for row in block.get("rows") or []:
            lines.extend(_row_lines(row))
        lines.append("")
    cal = report["calibration"]
    lines.append(f"[calibration] {cal.get('status')}")
    if cal.get("reason"):
        lines.append(f"  reason: {cal['reason']}")
    drift = report["drift"]
    lines.append(f"[drift] {drift.get('status')}")
    if drift.get("reason"):
        lines.append(f"  reason: {drift['reason']}")
    if report.get("blocked_metrics"):
        lines.append("")
        lines.append("blocked or unavailable: " + ", ".join(report["blocked_metrics"]))
    lines.append("")
    return "\n".join(lines)


def _row_lines(row: dict) -> list[str]:
    label = row.get("kind") or row.get("wall_id") or row.get("opening_id") or row.get("room_id") or "row"
    lines = [f"  {label}"]
    for key in ("predicted", "ground_truth", "compared_m", "absolute_error", "relative_error", "threshold", "pass"):
        if key in row:
            lines.append(f"    {key}: {row[key]}")
    return lines


def _assessment_gate(ground_truth: dict | None) -> dict:
    if ground_truth and ground_truth.get("provenance") == "laser_tape":
        return {"status": "recorded", "reason": "ground truth is marked laser_tape"}
    if ground_truth and ground_truth.get("evaluator_only"):
        return {"status": "blocked", "reason": "synthetic evaluator fixture is not laser/tape ground truth"}
    return {"status": "blocked", "reason": BLOCKED_REASON}


def _unavailable(reason: str) -> dict:
    return {"status": "unavailable", "reason": reason, "rows": []}


def _not_applicable(reason: str) -> dict:
    return {"status": "not_applicable", "reason": reason, "rows": []}


def _blocked(report: dict) -> list[str]:
    names = []
    for key in ("wall_accuracy", "opening_accuracy", "ceiling_accuracy", "repeatability", "photo_property", "calibration", "drift"):
        status = (report.get(key) or {}).get("status")
        if status in {"unavailable", "blocked"}:
            names.append(key)
    if report["assessment_gate"]["status"] == "blocked":
        names.append("assessment_gate")
    return names


def _value(measure) -> float | None:
    if measure is None:
        return None
    if isinstance(measure, (int, float)):
        return float(measure)
    if isinstance(measure, dict) and measure.get("value") is not None:
        return float(measure["value"])
    return None


def _room_id(room: dict) -> str:
    return str(room.get("room_id") or room.get("id"))


def _match_rooms(prediction: dict, ground_truth: dict) -> tuple[list[tuple[dict, dict]], list[dict]]:
    """Pair rooms by room_id. room_map sends a prediction id to a ground-truth id."""
    mapping = {str(k): str(v) for k, v in (ground_truth.get("room_map") or {}).items()}
    gt_by_id = {_room_id(room): room for room in ground_truth["rooms"]}
    notes = []
    pairs = []
    used = set()
    for room in prediction["property"]["rooms"]:
        pred_id = _room_id(room)
        gt_id = mapping.get(pred_id, pred_id)
        gt = gt_by_id.get(gt_id)
        if gt is None:
            notes.append({"room_id": pred_id, "status": "unmatched_prediction"})
            continue
        used.add(gt_id)
        notes.append({"room_id": pred_id, "ground_truth_room_id": gt_id, "status": "matched"})
        pairs.append((room, gt))
    for gt_id in gt_by_id:
        if gt_id not in used:
            notes.append({"room_id": gt_id, "status": "unmatched_ground_truth"})
    notes.sort(key=lambda note: (note["room_id"], note["status"]))
    return pairs, notes


def _wall_threshold(tier: str, limits: Thresholds) -> dict:
    if tier == "photo":
        return {"relative": limits.photo_wall_rel, "absolute_m": limits.wall_abs_m}
    if tier == "video":
        return {"relative": limits.video_wall_rel, "absolute_m": limits.wall_abs_m}
    return {"relative": None, "absolute_m": limits.wall_abs_m}


def _passes_wall(abs_error: float, rel_error: float | None, threshold: dict) -> bool | None:
    if threshold.get("absolute_m") is None and threshold.get("relative") is None:
        return None
    ok_abs = threshold.get("absolute_m") is None or abs_error <= threshold["absolute_m"]
    ok_rel = threshold.get("relative") is None or (rel_error is not None and rel_error <= threshold["relative"])
    if threshold.get("absolute_m") is not None and threshold.get("relative") is not None:
        return bool(ok_abs or ok_rel)
    if threshold.get("absolute_m") is not None:
        return bool(ok_abs)
    return bool(ok_rel)


def _walls(pairs: list[tuple[dict, dict]], tier: str, limits: Thresholds) -> dict:
    threshold = _wall_threshold(tier, limits)
    rows = []
    for pred_room, gt_room in pairs:
        matched, missed, phantom = _match_walls(pred_room.get("walls") or [], gt_room.get("walls") or [])
        for pred, gt in matched:
            predicted = _value(pred.get("length_m"))
            truth = _value(gt.get("length_m"))
            rows.append(_error_row(pred.get("id"), truth, predicted, threshold, "wall", gt.get("wall_id") or gt.get("id")))
        for gt in missed:
            rows.append(_missing_row(gt.get("wall_id") or gt.get("id"), _value(gt.get("length_m")), "missed_wall"))
        for pred in phantom:
            rows.append(_missing_row(pred.get("id"), None, "phantom_wall", predicted=_value(pred.get("length_m"))))
    return _summarize("wall", rows, threshold)


def _match_walls(pred_walls: list[dict], gt_walls: list[dict]) -> tuple[list[tuple[dict, dict]], list[dict], list[dict]]:
    """Id match first. Leftovers pair by orientation and midpoint, greedily, in id order."""
    pred_by_id = {wall.get("id"): wall for wall in pred_walls if wall.get("id")}
    used_pred = set()
    used_gt = set()
    matched = []
    for index, gt in enumerate(gt_walls):
        gt_id = gt.get("wall_id") or gt.get("id")
        if gt_id and gt_id in pred_by_id:
            matched.append((pred_by_id[gt_id], gt))
            used_pred.add(id(pred_by_id[gt_id]))
            used_gt.add(index)
    left_pred = [wall for wall in pred_walls if id(wall) not in used_pred]
    left_gt = [(index, wall) for index, wall in enumerate(gt_walls) if index not in used_gt]
    left_gt.sort(key=lambda item: _geom_key(item[1], item[0]))
    left_pred.sort(key=lambda wall: (wall.get("id") or "", _geom_key(wall, 0)[1], _geom_key(wall, 0)[2]))
    still_pred = []
    for gt_index, gt in left_gt:
        choice = _closest_wall(gt, left_pred)
        if choice is None:
            continue
        matched.append((choice, gt))
        used_gt.add(gt_index)
        left_pred.remove(choice)
    missed = [wall for index, wall in enumerate(gt_walls) if index not in used_gt]
    phantom = [wall for wall in pred_walls if id(wall) not in {id(pred) for pred, _ in matched}]
    return matched, missed, phantom


def _geom_key(wall: dict, index: int) -> tuple:
    p0, p1 = wall.get("p0_m"), wall.get("p1_m")
    if not p0 or not p1:
        return (index, 0.0, 0.0)
    ang, mid = _angle_mid(p0, p1)
    return (index, round(ang, 5), round(float(mid[0]), 5), round(float(mid[1]), 5))


def _angle_mid(p0, p1) -> tuple[float, np.ndarray]:
    a = np.asarray(p0, dtype=float)
    b = np.asarray(p1, dtype=float)
    delta = b - a
    angle = float(np.arctan2(delta[1], delta[0]) % np.pi)
    return angle, (a + b) / 2.0


def _closest_wall(gt: dict, preds: list[dict]) -> dict | None:
    if not gt.get("p0_m") or not gt.get("p1_m"):
        return None
    gt_angle, gt_mid = _angle_mid(gt["p0_m"], gt["p1_m"])
    best = None
    best_key = None
    for pred in preds:
        if not pred.get("p0_m") or not pred.get("p1_m"):
            continue
        angle, mid = _angle_mid(pred["p0_m"], pred["p1_m"])
        dang = abs(gt_angle - angle)
        dang = min(dang, np.pi - dang)
        if dang > np.radians(20):
            continue
        dist = float(np.linalg.norm(mid - gt_mid))
        key = (round(dist, 6), pred.get("id") or "")
        if best_key is None or key < best_key:
            best_key = key
            best = pred
    return best


def _openings(pairs: list[tuple[dict, dict]], limits: Thresholds) -> dict:
    threshold = {"absolute_m": limits.opening_max_error_m, "relative": None, "pass_rate": limits.opening_pass_rate}
    rows = []
    any_opening = False
    for pred_room, gt_room in pairs:
        pred = _openings_on_room(pred_room)
        gt = list(gt_room.get("openings") or [])
        if pred or gt:
            any_opening = True
        matched, missed, phantom = _match_openings(pred, gt)
        for left, right in matched:
            rows.append(
                _error_row(
                    left.get("id"),
                    _value(right.get("width_m")),
                    _value(left.get("width_m")),
                    {"absolute_m": limits.opening_max_error_m, "relative": None},
                    "opening",
                    right.get("opening_id") or right.get("id"),
                )
            )
        for right in missed:
            rows.append(_missing_row(right.get("opening_id") or right.get("id"), _value(right.get("width_m")), "missed"))
        for left in phantom:
            rows.append(_missing_row(left.get("id"), None, "phantom", predicted=_value(left.get("width_m"))))
    if not any_opening and not rows:
        block = _not_applicable("no openings on either side")
        block["n_missed"] = 0
        block["n_phantom"] = 0
        return block
    summary = _summarize("opening", rows, threshold)
    summary["pass_rate_threshold"] = limits.opening_pass_rate
    if summary["pass_rate"] is None:
        summary["gate_pass"] = None
    else:
        summary["gate_pass"] = bool(summary["pass_rate"] >= limits.opening_pass_rate)
    return summary


def _openings_on_room(room: dict) -> list[dict]:
    found = []
    for wall in room.get("walls") or []:
        for opening in wall.get("openings") or []:
            item = dict(opening)
            item["wall_id"] = wall.get("id")
            found.append(item)
    return found


def _match_openings(pred: list[dict], gt: list[dict]) -> tuple[list[tuple[dict, dict]], list[dict], list[dict]]:
    pred_by_id = {item.get("id"): item for item in pred if item.get("id")}
    used_pred = set()
    used_gt = set()
    matched = []
    for index, item in enumerate(gt):
        gt_id = item.get("opening_id") or item.get("id")
        if gt_id and gt_id in pred_by_id:
            matched.append((pred_by_id[gt_id], item))
            used_pred.add(id(pred_by_id[gt_id]))
            used_gt.add(index)
    left_pred = [item for item in pred if id(item) not in used_pred]
    left_gt = [item for index, item in enumerate(gt) if index not in used_gt]
    # Same wall, then width, then id. Pair in that order. Leftovers are miss or phantom.
    left_pred.sort(key=lambda item: (item.get("wall_id") or "", _value(item.get("width_m")) or 0.0, item.get("id") or ""))
    left_gt.sort(key=lambda item: (item.get("wall_id") or "", _value(item.get("width_m")) or 0.0, item.get("opening_id") or item.get("id") or ""))
    buckets: dict[str, list[dict]] = {}
    for item in left_pred:
        buckets.setdefault(item.get("wall_id") or "", []).append(item)
    still_missed = []
    for item in left_gt:
        bucket = buckets.get(item.get("wall_id") or "", [])
        if not bucket:
            still_missed.append(item)
            continue
        matched.append((bucket.pop(0), item))
    phantom = [item for items in buckets.values() for item in items]
    missed_ids = {id(item) for item in still_missed}
    missed = [item for item in left_gt if id(item) in missed_ids]
    return matched, missed, phantom


def _ceilings(pairs: list[tuple[dict, dict]], repeat: dict | None, limits: Thresholds) -> dict:
    threshold = {"absolute_m": limits.ceiling_max_error_m, "relative": None, "repeat_spread_m": limits.ceiling_repeat_spread_m}
    rows = []
    repeat_rooms = {}
    if repeat is not None:
        repeat_rooms = {_room_id(room): room for room in load_prediction(repeat)["property"]["rooms"]}
    any_ceiling = False
    for pred_room, gt_room in pairs:
        if "ceiling_height_m" not in gt_room:
            continue
        any_ceiling = True
        truth = _value(gt_room.get("ceiling_height_m"))
        predicted = _value(pred_room.get("ceiling_height_m"))
        if truth is None:
            continue
        if predicted is None:
            rows.append(_missing_row(_room_id(pred_room), truth, "missed_ceiling"))
            continue
        row = _error_row(_room_id(pred_room), truth, predicted, {"absolute_m": limits.ceiling_max_error_m, "relative": None}, "ceiling", _room_id(gt_room))
        other = repeat_rooms.get(_room_id(pred_room))
        if other is not None:
            other_value = _value(other.get("ceiling_height_m"))
            if other_value is None:
                row["repeat_spread_m"] = None
                row["repeat_pass"] = None
            else:
                spread = abs(predicted - other_value)
                row["repeat_spread_m"] = round(spread, 4)
                row["repeat_pass"] = bool(spread <= limits.ceiling_repeat_spread_m)
        rows.append(row)
    if not any_ceiling:
        return _not_applicable("ground truth has no ceiling")
    return _summarize("ceiling", rows, threshold)


def _repeatability(first: dict, second: dict, limits: Thresholds) -> dict:
    first = load_prediction(first)
    second = load_prediction(second)
    threshold = {
        "absolute_m": limits.repeatability_abs_m,
        "relative": limits.repeatability_rel,
        "rule": "pass when absolute difference <= 1 cm OR relative difference <= 0.5%",
    }
    second_rooms = {_room_id(room): room for room in second["property"]["rooms"]}
    rows = []
    for room in first["property"]["rooms"]:
        other = second_rooms.get(_room_id(room))
        if other is None:
            continue
        matched, _, _ = _match_walls(room.get("walls") or [], _as_gt_walls(other.get("walls") or []))
        for pred, gt in matched:
            a = _value(pred.get("length_m"))
            b = _value(gt.get("length_m"))
            if a is None or b is None:
                continue
            abs_error = abs(a - b)
            denom = max(abs(a), abs(b))
            rel = None if denom == 0 else abs_error / denom
            passed = bool(abs_error <= limits.repeatability_abs_m or (rel is not None and rel <= limits.repeatability_rel))
            rows.append(
                {
                    "kind": "repeat_wall",
                    "wall_id": pred.get("id"),
                    "room_id": _room_id(room),
                    "predicted": round(a, 4),
                    "ground_truth": None,
                    "compared_m": round(b, 4),
                    "absolute_error": round(abs_error, 4),
                    "relative_error": None if rel is None else round(rel, 6),
                    "threshold": threshold,
                    "pass": passed,
                }
            )
    if not rows:
        return _unavailable("no wall could be matched across the two captures")
    return _summarize("repeatability", rows, threshold)


def _as_gt_walls(walls: list[dict]) -> list[dict]:
    copied = []
    for wall in walls:
        item = dict(wall)
        item["wall_id"] = wall.get("id")
        item["length_m"] = wall.get("length_m")
        copied.append(item)
    return copied


def _photo(prediction: dict, ground_truth: dict, wall_accuracy: dict, limits: Thresholds) -> dict:
    rows = []
    if "footprint_m2" not in ground_truth:
        footprint = _unavailable("ground truth has no footprint_m2")
    else:
        predicted = prediction["property"].get("footprint_area_m2")
        truth = _value(ground_truth.get("footprint_m2"))
        threshold = {"relative": limits.photo_footprint_rel, "absolute_m": None}
        if predicted is None or truth is None:
            footprint = _unavailable("footprint value missing")
        else:
            row = _error_row("footprint", truth, float(predicted), threshold, "footprint", "footprint")
            rows.append(row)
            footprint = _summarize("footprint", [row], threshold)
    adjacency = _adjacency(prediction, ground_truth)
    overlap = _overlap(prediction, ground_truth)
    status = "evaluated"
    if footprint.get("status") == "unavailable" and adjacency.get("status") != "evaluated":
        status = "unavailable"
    return {
        "status": status,
        "threshold_footprint_rel": limits.photo_footprint_rel,
        "threshold_wall_rel": limits.photo_wall_rel,
        "footprint": footprint,
        "adjacency": adjacency,
        "overlap": overlap,
        "wall_pass_rate": wall_accuracy.get("pass_rate"),
        "rows": rows,
    }


def _adjacency(prediction: dict, ground_truth: dict) -> dict:
    if "adjacencies" not in ground_truth:
        return _not_applicable("ground truth has no adjacencies")
    truth = {_link_key(link) for link in ground_truth["adjacencies"]}
    pred = {_link_key(link) for link in prediction["property"].get("adjacencies") or []}
    missing = sorted(truth - pred)
    phantom = sorted(pred - truth)
    passed = not missing and not phantom
    return {
        "status": "evaluated",
        "predicted": sorted(pred),
        "ground_truth": sorted(truth),
        "missing": missing,
        "phantom": phantom,
        "pass": passed,
        "n_evaluated": len(truth | pred),
        "n_missed": len(missing),
        "n_phantom": len(phantom),
    }


def _link_key(link: dict) -> str:
    a = str(link.get("room_a") or link.get("a"))
    b = str(link.get("room_b") or link.get("b"))
    left, right = sorted((a, b))
    return f"{left}|{right}"


def _overlap(prediction: dict, ground_truth: dict) -> dict:
    if "overlap_m2_max" not in ground_truth:
        return _not_applicable("ground truth has no overlap_m2_max")
    quality = prediction.get("quality") or {}
    if "overlap_m2" not in quality and "overlaps" not in quality:
        return _unavailable("prediction has no overlap field")
    value = float(quality.get("overlap_m2") or 0.0)
    limit = float(ground_truth["overlap_m2_max"])
    return {
        "status": "evaluated",
        "predicted": value,
        "ground_truth": limit,
        "absolute_error": round(max(0.0, value - limit), 4),
        "relative_error": None,
        "threshold": limit,
        "pass": bool(value <= limit),
    }


def _calibration(prediction: dict, ground_truth: dict | None) -> dict:
    quality = prediction.get("quality") or {}
    evidence = {}
    for key in ("scale_source", "scale_estimated", "scale"):
        if key in quality:
            evidence[key] = quality[key]
    record = (ground_truth or {}).get("calibration") if ground_truth else None
    if isinstance(record, dict) and record.get("scale") in {"measured", "estimated", "unavailable"}:
        return {"status": record["scale"], "source": "ground_truth", "evidence": evidence, "record": record}
    source = evidence.get("scale_source")
    estimated = evidence.get("scale_estimated")
    if estimated is True or (isinstance(source, str) and "estimated" in source):
        return {"status": "estimated", "source": "plan", "evidence": evidence, "reason": "plan marks scale as an estimate, not a tape"}
    if isinstance(source, str) and (source.startswith("failed") or source == "synthetic_fixture_meters"):
        reason = "fixture metres are not a tape calibration" if source == "synthetic_fixture_meters" else "plan reported a failed scale"
        return {"status": "unavailable", "source": "plan", "evidence": evidence, "reason": reason}
    return {"status": "unavailable", "source": "plan", "evidence": evidence, "reason": "plan has no calibration record"}


def _drift(prediction: dict, drift_off: dict | None, limits: Thresholds) -> dict:
    block = prediction.get("drift") if isinstance(prediction.get("drift"), dict) else {}
    evidence = {
        "method": block.get("method"),
        "floor_tilt_deg": block.get("floor_tilt_deg"),
        "floor_residual_m": block.get("floor_residual_m"),
        "chunk_floor_offsets_m": block.get("chunk_floor_offsets_m"),
        "loop": block.get("loop"),
        "footprint_area_m2_anchor_on": block.get("footprint_area_m2_anchor_on"),
        "footprint_area_m2_anchor_off": block.get("footprint_area_m2_anchor_off"),
    }
    on = evidence["footprint_area_m2_anchor_on"]
    off = evidence["footprint_area_m2_anchor_off"]
    delta = None
    if isinstance(on, (int, float)) and isinstance(off, (int, float)):
        delta = round(float(on) - float(off), 4)
    out = {
        "status": "evidence_only",
        "mode": evidence["method"],
        "evidence": evidence,
        "footprint_delta_m2": delta,
        "wall_comparison": "unavailable",
        "reason": "no drift-off plan was supplied for a wall-by-wall comparison",
        "rows": [],
    }
    if not block and drift_off is None:
        return _unavailable("plan has no drift record and no drift-off plan")
    if drift_off is None:
        return out
    comparison = _repeatability(prediction, drift_off, limits)
    out["status"] = "compared"
    out["wall_comparison"] = comparison.get("status")
    out["reason"] = None
    out["rows"] = comparison.get("rows") or []
    out["n_evaluated"] = comparison.get("n_evaluated")
    out["n_pass"] = comparison.get("n_pass")
    out["pass_rate"] = comparison.get("pass_rate")
    out["worst_abs_error_m"] = comparison.get("worst_abs_error_m")
    return out


def _error_row(pred_id, truth, predicted, threshold, kind: str, gt_id) -> dict:
    if truth is None or predicted is None:
        return _missing_row(pred_id or gt_id, truth, kind, predicted=predicted)
    abs_error = abs(predicted - truth)
    rel = None if truth == 0 else abs_error / abs(truth)
    passed = _passes_wall(abs_error, rel, threshold)
    return {
        "kind": kind,
        "wall_id": pred_id if kind == "wall" else None,
        "opening_id": pred_id if kind == "opening" else None,
        "room_id": pred_id if kind == "ceiling" else None,
        "id": pred_id,
        "ground_truth_id": gt_id,
        "predicted": round(float(predicted), 4),
        "ground_truth": round(float(truth), 4),
        "absolute_error": round(abs_error, 4),
        "relative_error": None if rel is None else round(rel, 6),
        "threshold": threshold,
        "pass": passed,
    }


def _missing_row(item_id, truth, kind: str, predicted=None) -> dict:
    return {
        "kind": kind,
        "id": item_id,
        "wall_id": item_id if "wall" in kind else None,
        "opening_id": item_id if kind in {"opening", "missed", "phantom"} else None,
        "predicted": None if predicted is None else round(float(predicted), 4),
        "ground_truth": None if truth is None else round(float(truth), 4),
        "absolute_error": None,
        "relative_error": None,
        "threshold": None,
        "pass": False,
    }


def _summarize(name: str, rows: list[dict], threshold: dict) -> dict:
    decided = [row for row in rows if isinstance(row.get("pass"), bool)]
    n_pass = sum(1 for row in decided if row["pass"] is True)
    missed = sum(1 for row in rows if row.get("kind") in {"missed", "missed_wall", "missed_ceiling"})
    phantom = sum(1 for row in rows if "phantom" in str(row.get("kind")))
    errors = [row["absolute_error"] for row in rows if isinstance(row.get("absolute_error"), (int, float))]
    return {
        "status": "evaluated",
        "metric": name,
        "threshold": threshold,
        "n_evaluated": len(rows),
        "n_pass": n_pass,
        "pass_rate": None if not decided else round(n_pass / len(decided), 4),
        "n_missed": missed,
        "n_phantom": phantom,
        "worst_abs_error_m": None if not errors else round(max(errors), 4),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score a plan.json against ground truth")
    parser.add_argument("--prediction", type=Path, help="a plan.json from run.py")
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--repeat", type=Path, help="second plan.json of the same room")
    parser.add_argument("--drift-off", type=Path, help="plan.json with drift correction off")
    parser.add_argument("--case", type=Path, help="folder with prediction.json and ground_truth.json")
    parser.add_argument("--case-id", type=str)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.case:
        report = evaluate_case(args.case)
    else:
        if args.prediction is None:
            raise SystemExit("pass --prediction or --case")
        prediction = load_prediction(load_json(args.prediction))
        ground_truth = load_ground_truth(load_json(args.ground_truth)) if args.ground_truth else None
        repeat = load_prediction(load_json(args.repeat)) if args.repeat else None
        drift_off = load_prediction(load_json(args.drift_off)) if args.drift_off else None
        report = evaluate(prediction, ground_truth, repeat, drift_off, case_id=args.case_id)
    write_report(report, args.out)
    print(
        f"wrote {args.out / 'report.json'} "
        f"assessment_gate={report['assessment_gate']['status']} "
        f"walls={report['wall_accuracy']['status']}"
    )


if __name__ == "__main__":
    main()
