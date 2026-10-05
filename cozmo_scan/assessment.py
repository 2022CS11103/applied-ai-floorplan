"""Assessment gates. A missing capture stays BLOCKED. A miss is FAIL. PASS needs the measured bar.

Estimated scales and synthetic fixtures are not laser or tape. FARO is a
reference depth, not a tape of each wall and opening, so it cannot PASS a
tape gate. Thresholds are the constants in benchmark_eval.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .benchmark_eval import (
    CEILING_MAX_ERROR_M,
    CEILING_REPEAT_SPREAD_M,
    OPENING_MAX_ERROR_M,
    OPENING_PASS_RATE,
    PHOTO_FOOTPRINT_REL,
    PHOTO_WALL_REL,
    REPEATABILITY_ABS_M,
    REPEATABILITY_REL,
    VIDEO_WALL_REL,
)

PHYSICAL_SOURCES = {"laser", "tape", "survey"}
REFERENCE_SOURCES = {"faro", "external_reference"}
INCUMBENT_BEAT_OR_TIE = 0.70


def _status(measured_pass: bool | None, *, eligible: bool, reason: str, **extra) -> dict:
    if not eligible or measured_pass is None:
        state = "BLOCKED"
    elif measured_pass:
        state = "PASS"
    else:
        state = "FAIL"
    out = {"status": state, "reason": reason}
    out.update(extra)
    return out


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def physical_source(source: str | None) -> bool:
    return str(source or "").lower() in PHYSICAL_SOURCES


def load_json(path: Path) -> dict:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{path} must be a JSON object")
    return doc


def validate_measurements(doc: dict) -> list[str]:
    """Reject a file that asks an estimate to stand in for ground truth."""
    problems = []
    if not isinstance(doc, dict):
        return ["ground truth must be an object"]
    source = str(doc.get("source") or "").lower()
    if source in {"estimated", "synthetic", "chest_height_prior"}:
        problems.append("estimated source is not ground truth")
    rows = doc.get("measurements") or []
    if rows and not isinstance(rows, list):
        problems.append("measurements must be a list")
    for index, row in enumerate(rows if isinstance(rows, list) else []):
        kind = row.get("measurement_type")
        if kind not in {"wall_length", "opening_width", "ceiling_height", "floor_area"}:
            problems.append(f"measurements[{index}] has an unknown measurement_type")
        if "value_m" not in row:
            problems.append(f"measurements[{index}] needs value_m")
        row_source = str(row.get("source") or source or "").lower()
        if row_source in {"estimated", "synthetic", "chest_height_prior"}:
            problems.append(f"measurements[{index}] source is estimated")
    return problems


def _eligible(source: str | None) -> tuple[bool, str]:
    name = str(source or "").lower()
    if name in PHYSICAL_SOURCES:
        return True, name
    if name in REFERENCE_SOURCES:
        if name == "faro":
            return False, "FARO is a depth reference, not a tape"
        return False, "external reference is not a tape measurement"
    if not name:
        return False, "no measurement source"
    return False, f"source {name} is not laser, tape, or survey"


def score_openings(ground_truth: list[dict], predicted: list[dict], source: str | None = None) -> dict:
    """Match openings. A miss and a phantom both count against the 85% bar."""
    eligible, why = _eligible(source)
    gt = list(ground_truth or [])
    pred = list(predicted or [])
    if not gt and not pred:
        return _status(None, eligible=False, reason="no openings to score", true_positives=0, missed=0, phantoms=0)
    used_pred: set[int] = set()
    rows = []
    missed = 0
    within = 0
    errors = []
    for item in gt:
        gt_id = item.get("opening_id") or item.get("id")
        partner = None
        for index, candidate in enumerate(pred):
            if index in used_pred:
                continue
            same_id = gt_id and candidate.get("opening_id") == gt_id
            same_wall = item.get("wall_id") and candidate.get("wall_id") == item.get("wall_id") and item.get("room_id") == candidate.get("room_id")
            if same_id or (partner is None and same_wall and not gt_id):
                partner = index
                if same_id:
                    break
        if partner is None:
            missed += 1
            rows.append({"opening_id": gt_id, "kind": "miss", "pass": False, "ground_truth_m": item.get("width_m") or item.get("value_m")})
            continue
        used_pred.add(partner)
        chosen = pred[partner]
        truth = item.get("width_m") if item.get("width_m") is not None else item.get("value_m")
        estimate = chosen.get("width_m") if chosen.get("width_m") is not None else chosen.get("predicted_width_m")
        if truth is None or estimate is None:
            missed += 1
            rows.append({"opening_id": gt_id, "kind": "miss", "pass": False})
            continue
        error = abs(float(estimate) - float(truth))
        errors.append(error)
        ok = error <= OPENING_MAX_ERROR_M
        within += int(ok)
        rows.append(
            {
                "opening_id": gt_id,
                "room_id": item.get("room_id"),
                "wall_id": item.get("wall_id"),
                "type": item.get("type") or item.get("opening_type"),
                "ground_truth_m": float(truth),
                "predicted_m": float(estimate),
                "absolute_error_m": _round(error),
                "confidence": chosen.get("confidence"),
                "method": chosen.get("method"),
                "kind": "true_positive",
                "pass": ok,
            }
        )
    phantoms = []
    for index, candidate in enumerate(pred):
        if index in used_pred:
            continue
        phantoms.append(candidate.get("opening_id") or candidate.get("id"))
        rows.append({"opening_id": candidate.get("opening_id"), "kind": "phantom", "pass": False, "predicted_m": candidate.get("width_m")})
    trials = len(gt) + len(phantoms)
    rate = None if trials == 0 else within / trials
    passed = None if rate is None else bool(rate >= OPENING_PASS_RATE and missed == 0 and not phantoms or rate >= OPENING_PASS_RATE)
    # Misses and phantoms are already excluded from `within`, and they sit in `trials`.
    passed = None if rate is None else bool(rate >= OPENING_PASS_RATE)
    arr = np.array(errors, dtype=float) if errors else np.array([])
    reason = why if not eligible else (
        f"{within}/{trials} within {OPENING_MAX_ERROR_M * 100:.0f} cm; threshold {OPENING_PASS_RATE:.0%}"
    )
    return _status(
        passed,
        eligible=eligible and trials > 0,
        reason=reason,
        threshold={"absolute_m": OPENING_MAX_ERROR_M, "pass_rate": OPENING_PASS_RATE},
        true_positives=within,
        matched=len(errors),
        missed=missed,
        phantoms=len(phantoms),
        within_2cm=within,
        trials=trials,
        pass_rate=None if rate is None else round(rate, 4),
        mae_m=None if len(arr) == 0 else _round(float(arr.mean())),
        median_m=None if len(arr) == 0 else _round(float(np.median(arr))),
        p95_m=None if len(arr) == 0 else _round(float(np.percentile(arr, 95))),
        worst_m=None if len(arr) == 0 else _round(float(arr.max())),
        rows=rows,
    )


def score_ceilings(rooms: list[dict], source: str | None = None) -> dict:
    """Per-room ceiling error, plus the repeat spread when a second capture exists.

    `passed` is both bars. `biased_but_repeatable` and `unrepeatable` are FAIL.
    A missing repeat capture does not get to claim the spread bar.
    """
    eligible, why = _eligible(source)
    if not rooms:
        return _status(None, eligible=False, reason="no ceiling measurements", classification="blocked")
    errors = []
    spreads = []
    rows = []
    for room in rooms:
        truth = room.get("ground_truth_m")
        predicted = room.get("predicted_m")
        if truth is None or predicted is None:
            rows.append({"room_id": room.get("room_id"), "kind": "missing", "pass": False})
            continue
        error = abs(float(predicted) - float(truth))
        errors.append(error)
        spread = room.get("repeat_spread_m")
        spread_pass = None
        if spread is not None:
            spreads.append(float(spread))
            spread_pass = round(float(spread), 6) <= CEILING_REPEAT_SPREAD_M
        row_pass = error <= CEILING_MAX_ERROR_M and spread_pass is not False
        rows.append(
            {
                "capture_id": room.get("capture_id"),
                "room_id": room.get("room_id"),
                "absolute_error_m": _round(error),
                "repeat_spread_m": None if spread is None else _round(float(spread)),
                "pass": bool(error <= CEILING_MAX_ERROR_M and (spread_pass is not False)),
                "room_pass": bool(error <= CEILING_MAX_ERROR_M),
                "spread_pass": spread_pass,
            }
        )
        _ = row_pass
    if not errors:
        return _status(None, eligible=False, reason="ceiling values missing", classification="blocked", rows=rows)
    arr = np.array(errors, dtype=float)
    biased = bool(np.any(np.round(arr, 6) > CEILING_MAX_ERROR_M))
    if not spreads:
        classification = "biased_repeat_not_measured" if biased else "repeat_not_measured"
        passed = None
        reason = "repeat spread was not measured"
        if biased:
            reason = f"max error {_round(float(arr.max()))} m exceeds {CEILING_MAX_ERROR_M} m, and repeat spread was not measured"
            passed = False
    else:
        unrepeatable = bool(np.any(np.round(np.array(spreads), 6) > CEILING_REPEAT_SPREAD_M))
        if biased and unrepeatable:
            classification = "both"
        elif biased:
            classification = "biased_but_repeatable"
        elif unrepeatable:
            classification = "unrepeatable"
        else:
            classification = "passed"
        passed = classification == "passed"
        reason = classification
    if not eligible:
        passed_out = None
        reason = why
    else:
        passed_out = passed
    return _status(
        passed_out,
        eligible=eligible and passed is not None or (eligible and passed is False),
        reason=reason,
        classification=classification,
        threshold={"absolute_m": CEILING_MAX_ERROR_M, "repeat_spread_m": CEILING_REPEAT_SPREAD_M},
        mean_error_m=_round(float(arr.mean())),
        median_error_m=_round(float(np.median(arr))),
        max_error_m=_round(float(arr.max())),
        max_repeat_spread_m=None if not spreads else _round(float(max(spreads))),
        rows=rows,
    )


def score_repeatability(walls: list[dict], source: str | None = None) -> dict:
    """Two captures, same room and tier. Unmatched walls stay in the report."""
    eligible, why = _eligible(source if source is not None else "tape")
    if source is None:
        eligible, why = True, "comparison of two plans"
    else:
        eligible, why = _eligible(source)
    rows = []
    for wall in walls:
        if wall.get("matched") is False or wall.get("length_a_m") is None or wall.get("length_b_m") is None:
            rows.append(
                {
                    "wall_id": wall.get("wall_id"),
                    "room_id": wall.get("room_id"),
                    "matched": False,
                    "pass": False,
                    "reason": wall.get("reason") or "unmatched wall",
                }
            )
            continue
        a = float(wall["length_a_m"])
        b = float(wall["length_b_m"])
        absolute = abs(a - b)
        denom = max(abs(a), abs(b))
        relative = None if denom == 0 else absolute / denom
        passed = bool(absolute <= REPEATABILITY_ABS_M or (relative is not None and relative <= REPEATABILITY_REL))
        rows.append(
            {
                "wall_id": wall.get("wall_id"),
                "room_id": wall.get("room_id"),
                "absolute_difference_m": _round(absolute),
                "relative_difference_percent": None if relative is None else _round(relative * 100.0, 4),
                "pass": passed,
                "matched": True,
            }
        )
    if not rows:
        return _status(None, eligible=False, reason="no walls to compare", passed_count=0, failed_count=0)
    failed = [row for row in rows if not row["pass"]]
    absolutes = [row["absolute_difference_m"] for row in rows if row.get("absolute_difference_m") is not None]
    relatives = [row["relative_difference_percent"] for row in rows if row.get("relative_difference_percent") is not None]
    passed = len(failed) == 0
    reason = why if not eligible else f"{len(rows) - len(failed)}/{len(rows)} walls within 1 cm or 0.5%"
    return _status(
        passed if eligible else None,
        eligible=eligible,
        reason=reason,
        threshold={"absolute_m": REPEATABILITY_ABS_M, "relative": REPEATABILITY_REL},
        passed_count=len(rows) - len(failed),
        failed_count=len(failed),
        max_absolute_difference_m=None if not absolutes else max(absolutes),
        max_relative_difference_percent=None if not relatives else max(relatives),
        rows=rows,
    )


def score_photo(walls: list[dict], footprint: dict | None, source: str | None, calibration: dict | None = None) -> dict:
    """Wall and footprint percentages. An estimated scale cannot PASS."""
    from .photo import _calibration_status

    cal = _calibration_status(calibration)
    eligible, why = _eligible(source)
    if not cal["is_physical"]:
        eligible = False
        why = cal["reason"]
    wall_rows = []
    for wall in walls:
        truth = wall.get("ground_truth_m")
        predicted = wall.get("predicted_m")
        if truth in (None, 0) or predicted is None:
            wall_rows.append({"wall_id": wall.get("wall_id"), "pass": False, "reason": "missing length"})
            continue
        rel = abs(float(predicted) - float(truth)) / abs(float(truth))
        wall_rows.append(
            {
                "wall_id": wall.get("wall_id"),
                "room_id": wall.get("room_id"),
                "error_percent": _round(rel * 100.0, 4),
                "pass": bool(rel <= PHOTO_WALL_REL),
            }
        )
    foot_rel = None
    foot_pass = None
    if footprint and footprint.get("ground_truth_m2") not in (None, 0) and footprint.get("predicted_m2") is not None:
        foot_rel = abs(float(footprint["predicted_m2"]) - float(footprint["ground_truth_m2"])) / abs(float(footprint["ground_truth_m2"]))
        foot_pass = bool(foot_rel <= PHOTO_FOOTPRINT_REL)
    walls_ok = bool(wall_rows) and all(row.get("pass") for row in wall_rows)
    if foot_pass is None or not wall_rows:
        passed = None
        reason = "photo ground truth is incomplete"
        eligible = False
    else:
        passed = bool(walls_ok and foot_pass)
        reason = why if not eligible else f"walls {'ok' if walls_ok else 'outside ±8%'}, footprint error {None if foot_rel is None else _round(foot_rel * 100, 2)}%"
    return _status(
        passed if eligible else None,
        eligible=eligible and passed is not None,
        reason=reason,
        threshold={"wall_relative": PHOTO_WALL_REL, "footprint_relative": PHOTO_FOOTPRINT_REL},
        calibration=cal,
        walls=wall_rows,
        footprint_error_percent=None if foot_rel is None else _round(foot_rel * 100.0, 4),
        footprint_pass=foot_pass,
    )


def score_stitch(prediction: dict, ground_truth: dict, source: str | None = None) -> dict:
    """Room count, adjacency, overlap, and footprint. A synthetic fixture stays ineligible."""
    eligible, why = _eligible(source)
    if ground_truth.get("evaluator_only") or ground_truth.get("synthetic_fixture"):
        eligible = False
        why = "synthetic fixture is not a physical property"
    pred_rooms = prediction.get("rooms") or []
    gt_rooms = ground_truth.get("rooms") or []
    pred_links = {_link(link) for link in prediction.get("adjacencies") or []}
    gt_links = {_link(link) for link in ground_truth.get("adjacencies") or []}
    missing = sorted(gt_links - pred_links)
    false = sorted(pred_links - gt_links)
    overlap = float(prediction.get("overlap_m2") or 0.0)
    overlap_max = float(ground_truth.get("overlap_m2_max") or 0.0)
    overlap_ok = overlap <= overlap_max
    foot = None
    if ground_truth.get("footprint_m2") not in (None, 0) and prediction.get("footprint_m2") is not None:
        foot = abs(float(prediction["footprint_m2"]) - float(ground_truth["footprint_m2"])) / abs(float(ground_truth["footprint_m2"]))
    count_ok = len(pred_rooms) == len(gt_rooms)
    passed = bool(count_ok and not missing and not false and overlap_ok and foot is not None and foot <= PHOTO_FOOTPRINT_REL)
    if foot is None:
        passed_out = None
        eligible = False
        why = "footprint ground truth is missing"
    else:
        passed_out = passed
    return _status(
        passed_out if eligible else None,
        eligible=eligible,
        reason=why if not eligible else "stitch comparison",
        room_count_predicted=len(pred_rooms),
        room_count_ground_truth=len(gt_rooms),
        missing_adjacency=missing,
        false_adjacency=false,
        overlap_m2=_round(overlap),
        overlapping_rooms=bool(not overlap_ok),
        footprint_error_percent=None if foot is None else _round(foot * 100.0, 4),
    )


def _link(link: dict) -> tuple[str, str]:
    a = str(link.get("room_a") or link.get("a") or "")
    b = str(link.get("room_b") or link.get("b") or "")
    return tuple(sorted((a, b)))


def score_video(plan: dict | None, walls: list[dict], source: str | None = None) -> dict:
    """No invented lengths. A room that did not close is BLOCKED, not a 3% pass."""
    if plan is None:
        return _status(None, eligible=False, reason="no video plan")
    reasons = list(plan.get("degraded_reasons") or [])
    rooms = (plan.get("property") or {}).get("rooms") or []
    if plan.get("status") == "degraded" or not rooms or "no_room_closure" in reasons:
        return {
            "status": "DEGRADED",
            "reason": "no_room_closure" if "no_room_closure" in reasons or not rooms else "video plan is degraded",
            "room_closure": "degraded",
            "degraded_reasons": reasons or ["no_room_closure"],
            "walls": [],
        }
    eligible, why = _eligible(source)
    rows = []
    for wall in walls:
        truth = wall.get("ground_truth_m")
        predicted = wall.get("predicted_m")
        if truth in (None, 0) or predicted is None:
            rows.append({"wall_id": wall.get("wall_id"), "pass": False, "reason": "missing length"})
            continue
        rel = abs(float(predicted) - float(truth)) / abs(float(truth))
        rows.append({"wall_id": wall.get("wall_id"), "error_percent": _round(rel * 100.0, 4), "pass": bool(rel <= VIDEO_WALL_REL)})
    if not rows:
        return _status(None, eligible=False, reason="video plan closed but no wall ground truth was supplied", room_closure="closed")
    passed = all(row.get("pass") for row in rows)
    return _status(
        passed if eligible else None,
        eligible=eligible,
        reason=why if not eligible else f"video walls against ±{VIDEO_WALL_REL:.0%}",
        threshold={"relative": VIDEO_WALL_REL},
        room_closure="closed",
        walls=rows,
    )


def score_damage(ground_truth: list[dict], predicted: list[dict], source: str | None = None, synthetic: bool = False) -> dict:
    """Class, surface, and extent. A synthetic fixture cannot PASS the physical gate."""
    eligible, why = _eligible(source)
    if synthetic:
        eligible = False
        why = "synthetic damage fixture is regression coverage, not a furnished room"
    if not ground_truth:
        return _status(None, eligible=False, reason="no damage ground truth")
    used = set()
    missed = []
    class_hits = 0
    extent_errors = []
    rows = []
    for item in ground_truth:
        partner = None
        for index, candidate in enumerate(predicted):
            if index in used:
                continue
            if candidate.get("surface_id") == item.get("surface_id") and candidate.get("class") == item.get("class"):
                partner = index
                break
        if partner is None:
            missed.append(item.get("class"))
            rows.append({"class": item.get("class"), "surface_id": item.get("surface_id"), "kind": "miss", "pass": False})
            continue
        used.add(partner)
        chosen = predicted[partner]
        class_hits += 1
        truth = item.get("extent_m2")
        estimate = (chosen.get("extent_m2") or {}).get("value") if isinstance(chosen.get("extent_m2"), dict) else chosen.get("extent_m2")
        error = None if truth is None or estimate is None else abs(float(estimate) - float(truth))
        if error is not None:
            extent_errors.append(error)
        rows.append(
            {
                "class": item.get("class"),
                "surface_id": item.get("surface_id"),
                "kind": "matched",
                "class_match": True,
                "extent_error_m2": _round(error) if error is not None else None,
            }
        )
    false_pos = [predicted[index].get("class") for index in range(len(predicted)) if index not in used]
    for name in false_pos:
        rows.append({"class": name, "kind": "false_positive", "pass": False})
    passed = bool(not missed and not false_pos and class_hits == len(ground_truth))
    return _status(
        passed if eligible else None,
        eligible=eligible,
        reason=why if not eligible else "damage regions",
        missed=missed,
        false_positives=false_pos,
        class_matches=class_hits,
        mean_extent_error_m2=None if not extent_errors else _round(float(np.mean(extent_errors))),
        rows=rows,
    )


def _metres(wall: dict) -> float | None:
    for key in ("length_m", "lengthInMeters", "length", "measure"):
        if wall.get(key) is None:
            continue
        try:
            return float(wall[key])
        except (TypeError, ValueError):
            return None
    return None


def normalize_incumbent(doc: dict) -> list[dict]:
    """Turn a Polycam or Magicplan export into one row per wall.

    A file that already has ``dimensions`` is used as given. A room export
    contributes a length only when that length is in the file. Missing ground
    truth stays missing, so the comparison cannot invent a winner.
    """
    if not isinstance(doc, dict):
        return []
    if doc.get("dimensions"):
        return [row for row in doc["dimensions"] if isinstance(row, dict)]
    rooms = [room for room in (doc.get("rooms") or []) if isinstance(room, dict)]
    for floor in doc.get("floors") or []:
        if isinstance(floor, dict):
            rooms.extend(room for room in (floor.get("rooms") or []) if isinstance(room, dict))
    vendor = str(doc.get("vendor") or doc.get("export") or doc.get("source") or "incumbent")
    rows = []
    for room in rooms:
        room_id = room.get("room_id") or room.get("id") or room.get("name")
        for wall in room.get("walls") or []:
            if not isinstance(wall, dict):
                continue
            length = _metres(wall)
            if length is None:
                continue
            truth = wall.get("ground_truth_m")
            ours = wall.get("ours_m")
            row = {
                "dimension": str(wall.get("wall_id") or wall.get("id") or room_id),
                "room_id": room_id,
                "wall_id": wall.get("wall_id") or wall.get("id"),
                "incumbent_m": length,
                "vendor": vendor,
            }
            if ours is not None:
                row["ours_m"] = float(ours)
            if truth is not None:
                truth_m = float(truth)
                row["ground_truth_m"] = truth_m
                row["incumbent_error_m"] = abs(length - truth_m)
                if truth_m:
                    row["incumbent_relative_error"] = abs(length - truth_m) / abs(truth_m)
                if ours is not None:
                    row["ours_error_m"] = abs(float(ours) - truth_m)
                    if truth_m:
                        row["ours_relative_error"] = abs(float(ours) - truth_m) / abs(truth_m)
            rows.append(row)
    return rows


def score_incumbent(rows: list[dict]) -> dict:
    """Dimension by dimension. No export means BLOCKED, not a 0% loss."""
    if not rows:
        return _status(None, eligible=False, reason="no Polycam or Magicplan export", won=0, tied=0, compared=0, beat_or_tie_rate=None)
    compared = []
    for row in rows:
        ours = row.get("ours_error_m")
        theirs = row.get("incumbent_error_m")
        if ours is None or theirs is None:
            winner = "unscored"
        elif abs(float(ours) - float(theirs)) <= 1e-6:
            winner = "tie"
        elif abs(float(ours)) < abs(float(theirs)):
            winner = "ours"
        else:
            winner = "incumbent"
        compared.append({**row, "winner": winner})
    scored = [row for row in compared if row["winner"] != "unscored"]
    if not scored:
        return _status(None, eligible=False, reason="incumbent rows have no paired errors", rows=compared, won=0, tied=0, compared=0)
    won = sum(row["winner"] == "ours" for row in scored)
    tied = sum(row["winner"] == "tie" for row in scored)
    rate = (won + tied) / len(scored)
    return _status(
        bool(rate >= INCUMBENT_BEAT_OR_TIE),
        eligible=True,
        reason=f"{won + tied}/{len(scored)} beat or tie; threshold {INCUMBENT_BEAT_OR_TIE:.0%}",
        threshold={"beat_or_tie": INCUMBENT_BEAT_OR_TIE},
        won=won,
        tied=tied,
        compared=len(scored),
        beat_or_tie_rate=round(rate, 4),
        rows=compared,
    )


def validate_manifest(doc: dict) -> list[str]:
    missing = []
    for key in ("property_id", "room_ids", "lidar_capture", "video_capture", "photo_folder", "repeat_capture", "ground_truth"):
        if not doc.get(key):
            missing.append(key)
    if doc.get("room_ids") is not None and not isinstance(doc.get("room_ids"), list):
        missing.append("room_ids_not_a_list")
    return missing


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
    except OSError:
        return "unknown"
    if out.returncode != 0:
        return "unknown"
    return out.stdout.strip() or "unknown"


def _measurement_rows(doc: dict, kind: str) -> list[dict]:
    return [row for row in (doc.get("measurements") or []) if row.get("measurement_type") == kind]


def run_manifest(manifest: dict, root: Path | None = None) -> dict:
    """Score whatever files the manifest actually points at. Missing files stay BLOCKED."""
    root = Path(root or ".").resolve()
    missing = validate_manifest(manifest)
    gates = {}
    evidence = {}

    def resolve(value: str | None) -> Path | None:
        if not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        return path

    gt_path = resolve(manifest.get("ground_truth"))
    gt = None
    gt_problems: list[str] = []
    if gt_path is None or not gt_path.is_file():
        gt_problems.append("ground truth file is missing")
    else:
        gt = load_json(gt_path)
        gt_problems.extend(validate_measurements(gt))
        evidence["ground_truth"] = str(gt_path)

    source = None if gt is None else gt.get("source")
    if gt_problems and gt is None:
        blocked = ", ".join(gt_problems)
        for name in ("openings", "ceiling", "repeatability", "photo_walls", "photo_footprint", "video_walls", "damage", "incumbent", "stitch"):
            gates[name] = _status(None, eligible=False, reason=blocked)
    else:
        if gt is None:
            gates["openings"] = _status(None, eligible=False, reason="; ".join(gt_problems) or "no ground truth")
            gates["ceiling"] = _status(None, eligible=False, reason="; ".join(gt_problems) or "no ground truth")
        else:
            opening_gt = list(gt.get("openings") or [])
            for row in _measurement_rows(gt, "opening_width"):
                opening_gt.append(row)
            pred_openings = list(gt.get("predicted_openings") or [])
            pred_path = resolve((manifest.get("predictions") or {}).get("openings"))
            if pred_path and pred_path.is_file():
                pred_openings = load_json(pred_path).get("openings") or pred_openings
                evidence["openings"] = str(pred_path)
            if not opening_gt:
                gates["openings"] = _status(None, eligible=False, reason="ground truth has no opening widths")
            else:
                gates["openings"] = score_openings(opening_gt, pred_openings, source=source)
            ceiling_rows = []
            for row in _measurement_rows(gt, "ceiling_height"):
                ceiling_rows.append(
                    {
                        "capture_id": row.get("capture_id") or gt.get("capture_id"),
                        "room_id": row.get("room_id"),
                        "ground_truth_m": row.get("value_m"),
                        "predicted_m": row.get("predicted_m"),
                        "repeat_spread_m": row.get("repeat_spread_m"),
                    }
                )
            if not ceiling_rows:
                gates["ceiling"] = _status(None, eligible=False, reason="ground truth has no ceiling_height")
            else:
                gates["ceiling"] = score_ceilings(ceiling_rows, source=source)

        repeat_path = resolve(manifest.get("repeat_capture"))
        if repeat_path is None or not repeat_path.is_file():
            gates["repeatability"] = _status(None, eligible=False, reason="repeat capture is missing")
        else:
            evidence["repeat_capture"] = str(repeat_path)
            repeat_doc = load_json(repeat_path)
            gates["repeatability"] = score_repeatability(repeat_doc.get("walls") or [], source=source if gt else None)

        photo_path = resolve(manifest.get("photo_folder"))
        if photo_path is None or not photo_path.is_dir():
            gates["photo_walls"] = _status(None, eligible=False, reason="photo folder is missing")
            gates["photo_footprint"] = gates["photo_walls"]
            gates["stitch"] = _status(None, eligible=False, reason="photo folder is missing")
        else:
            evidence["photo_folder"] = str(photo_path)
            walls = [] if gt is None else [
                {"wall_id": row.get("wall_id"), "room_id": row.get("room_id"), "ground_truth_m": row.get("value_m"), "predicted_m": row.get("predicted_m")}
                for row in _measurement_rows(gt, "wall_length")
            ]
            footprint = None
            if gt is not None:
                areas = _measurement_rows(gt, "floor_area")
                if areas:
                    footprint = {"ground_truth_m2": areas[0].get("value_m"), "predicted_m2": areas[0].get("predicted_m")}
            photo = score_photo(walls, footprint, source, calibration=None if gt is None else gt.get("calibration"))
            gates["photo_walls"] = photo
            gates["photo_footprint"] = photo
            if gt is None or "rooms" not in gt:
                gates["stitch"] = _status(None, eligible=False, reason="no room polygons in ground truth")
            else:
                gates["stitch"] = score_stitch(gt.get("predicted_stitch") or {"rooms": [], "adjacencies": []}, gt, source=source)

        video_path = resolve(manifest.get("video_capture"))
        plan_path = resolve((manifest.get("predictions") or {}).get("video"))
        if plan_path is None or not plan_path.is_file():
            reason = "video plan is missing" if video_path else "video capture is missing"
            gates["video_walls"] = _status(None, eligible=False, reason=reason)
        else:
            evidence["video_plan"] = str(plan_path)
            plan = load_json(plan_path)
            walls = [] if gt is None else [
                {"wall_id": row.get("wall_id"), "ground_truth_m": row.get("value_m"), "predicted_m": row.get("predicted_m")}
                for row in _measurement_rows(gt, "wall_length")
            ]
            gates["video_walls"] = score_video(plan, walls, source=source)

        damage_path = resolve(manifest.get("damage_ground_truth"))
        if damage_path is None or not damage_path.is_file():
            gates["damage"] = _status(None, eligible=False, reason="damage ground truth is missing")
        else:
            evidence["damage"] = str(damage_path)
            damage_doc = load_json(damage_path)
            synthetic = bool(damage_doc.get("label") or damage_doc.get("synthetic_fixture") or "synthetic" in str(damage_doc.get("label") or "").lower())
            regions = damage_doc.get("regions") or []
            predicted = damage_doc.get("predicted") or []
            gates["damage"] = score_damage(regions, predicted, source=damage_doc.get("source") or source, synthetic=synthetic or not physical_source(damage_doc.get("source")))

        incumbent_path = resolve(manifest.get("incumbent_export"))
        if incumbent_path is None or not incumbent_path.is_file():
            gates["incumbent"] = _status(None, eligible=False, reason="no Polycam or Magicplan export")
        else:
            evidence["incumbent"] = str(incumbent_path)
            incumbent = load_json(incumbent_path)
            gates["incumbent"] = score_incumbent(normalize_incumbent(incumbent))

    for key in missing:
        evidence.setdefault("missing_manifest_fields", [])
        if isinstance(evidence["missing_manifest_fields"], list):
            evidence["missing_manifest_fields"].append(key)

    any_fail = any(gate.get("status") == "FAIL" for gate in gates.values())
    return {
        "property_id": manifest.get("property_id"),
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git_commit": _git_commit(),
        "manifest_missing": missing,
        "evidence": evidence,
        "gates": gates,
        "failed": any_fail,
    }


def render_markdown(report: dict) -> str:
    lines = [
        f"# Assessment {report.get('property_id') or ''}".strip(),
        "",
        f"Commit `{report.get('git_commit')}` at {report.get('generated_at')}.",
        "",
        "| Gate | Status | Reason |",
        "| --- | --- | --- |",
    ]
    for name, gate in (report.get("gates") or {}).items():
        lines.append(f"| {name} | {gate.get('status')} | {gate.get('reason')} |")
    if report.get("manifest_missing"):
        lines.extend(["", "Missing manifest fields: " + ", ".join(report["manifest_missing"]) + "."])
    lines.append("")
    lines.append("BLOCKED means the file or the physical source is absent. DEGRADED means the pipeline ran and did not produce a measurement. Neither one is a pass.")
    return "\n".join(lines) + "\n"


def write_assessment(report: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "assessment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "assessment.md").write_text(render_markdown(report), encoding="utf-8")


def run_manifest_file(path: Path, out_dir: Path | None = None) -> dict:
    path = Path(path)
    manifest = load_json(path)
    report = run_manifest(manifest, root=path.parent)
    destination = Path(out_dir) if out_dir is not None else path.parent / "assessment_out"
    write_assessment(report, destination)
    report["output_dir"] = str(destination)
    return report
