"""Load a physical benchmark bundle. Missing evidence stays BLOCKED.

This module does not invent tape, laser, or incumbent numbers. Thresholds
come from benchmark_eval and are not restated as different constants.
"""

from __future__ import annotations

import json
from pathlib import Path

from .assessment import (
    normalize_incumbent,
    score_ceilings,
    score_openings,
    score_photo,
    score_video,
)
from .benchmark_eval import (
    CEILING_MAX_ERROR_M,
    CEILING_REPEAT_SPREAD_M,
    OPENING_MAX_ERROR_M,
    OPENING_PASS_RATE,
    PHOTO_FOOTPRINT_REL,
    PHOTO_WALL_REL,
    VIDEO_WALL_REL,
)

_KIND = {
    "wall": "wall_length",
    "opening": "opening_width",
    "ceiling": "ceiling_height",
    "area": "floor_area",
}
_PHYSICAL = {"tape", "laser", "survey"}
_REFERENCE = {"faro", "external_reference"}


def _blocked(reason: str, property_id: str | None, missing: list) -> dict:
    return {"status": "BLOCKED", "reason": reason, "property_id": property_id, "missing": missing, "exit_code": 0}


def _malformed(reason: str, detail: str) -> dict:
    return {"status": "BLOCKED", "reason": reason, "property_id": None, "missing": [], "detail": detail, "exit_code": 2}


def load_benchmark_manifest(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        return _blocked("missing_manifest", None, [path.name])
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _malformed("malformed_manifest", str(exc))
    if not isinstance(doc, dict):
        return _malformed("malformed_manifest", "manifest must be a JSON object")
    doc["_manifest_path"] = str(path)
    return doc


def validate_benchmark_manifest(manifest: dict) -> dict:
    if manifest.get("exit_code") == 2 or manifest.get("reason") == "malformed_manifest":
        return manifest
    if not isinstance(manifest, dict):
        return _malformed("malformed_manifest", "manifest must be a JSON object")
    missing = []
    for key in ("property_id", "rooms", "tiers", "repeat", "ground_truth", "incumbent"):
        if key not in manifest:
            missing.append(key)
    if missing:
        return _malformed("malformed_manifest", "missing keys: " + ", ".join(missing))
    if not isinstance(manifest.get("rooms"), list) or not manifest["rooms"]:
        return _malformed("malformed_manifest", "rooms must be a non-empty list")
    tiers = manifest.get("tiers") or {}
    if not isinstance(tiers, dict):
        return _malformed("malformed_manifest", "tiers must be an object")
    for name in ("lidar", "video", "photo"):
        if name not in tiers:
            missing.append(f"tiers.{name}")
    if missing:
        return _malformed("malformed_manifest", "missing keys: " + ", ".join(missing))
    return {"status": "ok", "property_id": manifest.get("property_id"), "missing": [], "exit_code": 0}


def validate_property_identity(manifest: dict) -> dict:
    checked = validate_benchmark_manifest(manifest)
    if checked.get("exit_code") == 2:
        return checked
    property_id = manifest.get("property_id")
    errors = []
    for label in ("ground_truth", "incumbent", "repeat", "damage"):
        block = manifest.get(label) or {}
        if isinstance(block, dict) and block.get("property_id") and block["property_id"] != property_id:
            errors.append(f"{label} belongs to {block['property_id']}, not {property_id}")
    tiers = manifest.get("tiers") or {}
    for name, block in tiers.items():
        if isinstance(block, dict) and block.get("property_id") and block["property_id"] != property_id:
            errors.append(f"{name} belongs to {block['property_id']}, not {property_id}")
    rooms = set(manifest.get("rooms") or [])
    connector = manifest.get("connector") or {}
    for room_id in connector.get("rooms") or []:
        if rooms and room_id not in rooms:
            errors.append(f"connector room {room_id} is not in this property")
    if errors:
        return _malformed("mixed_property", "; ".join(errors)) | {"property_id": property_id}
    return {"status": "ok", "property_id": property_id, "missing": [], "exit_code": 0}


def _root(manifest: dict) -> Path:
    raw = manifest.get("_manifest_path")
    return Path(raw).parent if raw else Path(".")


def _resolve(manifest: dict, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = _root(manifest) / path
    return path


def _tier_has_capture(name: str, path: Path) -> bool:
    if name == "lidar":
        return any(path.rglob("odometry.csv")) and any(path.rglob("*.png"))
    if name == "video":
        return any(path.rglob("*.mp4")) and any(path.rglob("odometry.csv"))
    if name == "photo":
        images = list(path.rglob("*.jpg")) + list(path.rglob("*.jpeg")) + list(path.rglob("*.png"))
        return len(images) >= 2
    return False


def validate_required_tiers(manifest: dict) -> dict:
    identity = validate_property_identity(manifest)
    if identity.get("exit_code") == 2:
        return identity
    missing = []
    for name, block in (manifest.get("tiers") or {}).items():
        path = _resolve(manifest, (block or {}).get("path") if isinstance(block, dict) else None)
        if path is None or not path.exists() or not _tier_has_capture(name, path):
            missing.append(name)
    if missing:
        return _blocked("missing_tier_capture", manifest.get("property_id"), missing)
    return {"status": "ok", "property_id": manifest.get("property_id"), "missing": [], "exit_code": 0}


def validate_ground_truth(manifest: dict) -> dict:
    block = manifest.get("ground_truth") or {}
    path = _resolve(manifest, block.get("path") if isinstance(block, dict) else None)
    if path is None or not path.is_file():
        return _blocked("missing_ground_truth", manifest.get("property_id"), ["ground_truth"])
    loaded = load_ground_truth(path)
    problems = validate_ground_truth_rows(loaded.get("rows") or [], manifest.get("property_id"))
    structural = [item for item in problems if "reference" not in item]
    if structural:
        return {
            "status": "BLOCKED",
            "reason": "invalid_ground_truth",
            "property_id": manifest.get("property_id"),
            "missing": [],
            "errors": structural,
            "exit_code": 2,
        }
    source = loaded.get("source")
    if source in _REFERENCE or problems:
        return _blocked("reference_is_not_tape", manifest.get("property_id"), [source or "faro"])
    return {"status": "ok", "property_id": manifest.get("property_id"), "missing": [], "exit_code": 0, "source": source}


def validate_repeat_capture(manifest: dict) -> dict:
    block = manifest.get("repeat") or {}
    if not isinstance(block, dict):
        return _malformed("malformed_manifest", "repeat must be an object")
    missing = []
    for name in ("capture_a", "capture_b"):
        path = _resolve(manifest, block.get(name))
        has_files = path is not None and path.exists() and any(item for item in path.rglob("*") if item.is_file() and item.name != ".gitkeep")
        if not has_files:
            missing.append(name)
    if missing:
        return _blocked("missing_repeat_capture", manifest.get("property_id"), missing)
    return {"status": "ok", "property_id": manifest.get("property_id"), "missing": [], "exit_code": 0}


def validate_incumbent_export(manifest: dict) -> dict:
    block = manifest.get("incumbent") or {}
    path = _resolve(manifest, block.get("path") if isinstance(block, dict) else None)
    if path is None or not path.is_file():
        return _blocked("missing_incumbent_export", manifest.get("property_id"), ["incumbent"])
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _malformed("malformed_incumbent", str(exc))
    if not isinstance(doc, dict):
        return _malformed("malformed_incumbent", "export must be a JSON object")
    return {"status": "ok", "property_id": manifest.get("property_id"), "missing": [], "exit_code": 0}


def _row_from_alias(row: dict, parent: dict) -> dict:
    kind = row.get("kind")
    measurement_type = row.get("measurement_type") or _KIND.get(kind)
    value = row.get("value_m")
    if value is None and row.get("value") is not None:
        value = row.get("value")
    unit = row.get("unit") or parent.get("unit")
    if unit == "cm" and value is not None:
        value = float(value) / 100.0
        unit = "m"
    uncertainty = row.get("uncertainty_m")
    if uncertainty is None and row.get("uncertainty") is not None:
        uncertainty = row.get("uncertainty")
        if row.get("unit") == "cm":
            uncertainty = float(uncertainty) / 100.0
    return {
        "property_id": row.get("property_id") or parent.get("property_id"),
        "capture_id": row.get("capture_id") or parent.get("capture_id"),
        "room_id": row.get("room_id"),
        "measurement_id": row.get("measurement_id") or row.get("opening_id") or row.get("wall_id"),
        "measurement_type": measurement_type,
        "kind": kind or {v: k for k, v in _KIND.items()}.get(measurement_type),
        "name": row.get("name") or row.get("wall_id") or row.get("opening_id"),
        "value_m": value,
        "unit": unit,
        "uncertainty_m": uncertainty,
        "source": str(row.get("source") or parent.get("source") or "").lower(),
        "notes": row.get("notes"),
        "predicted_m": row.get("predicted_m"),
        "repeat_spread_m": row.get("repeat_spread_m"),
        "wall_id": row.get("wall_id"),
        "opening_id": row.get("opening_id"),
    }


def load_ground_truth(path: Path) -> dict:
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        return {"rows": [], "source": None, "error": "ground truth must be an object"}
    parent_source = str(doc.get("source") or "").lower()
    rows = [_row_from_alias(row, doc) for row in (doc.get("measurements") or []) if isinstance(row, dict)]
    for item in doc.get("openings") or []:
        if isinstance(item, dict):
            alias = dict(item)
            alias.setdefault("kind", "opening")
            alias.setdefault("value", item.get("width_m"))
            rows.append(_row_from_alias(alias, doc))
    return {"property_id": doc.get("property_id"), "capture_id": doc.get("capture_id"), "source": parent_source, "rows": rows}


def validate_ground_truth_rows(rows: list[dict], property_id: str | None = None) -> list[str]:
    problems = []
    for index, row in enumerate(rows):
        if row.get("measurement_type") not in {"wall_length", "opening_width", "ceiling_height", "floor_area"}:
            problems.append(f"rows[{index}] has an unknown kind")
        if row.get("value_m") is None:
            problems.append(f"rows[{index}] needs a value in metres")
        if row.get("unit") not in {"m", "m2"}:
            problems.append(f"rows[{index}] unit must be m or m2 after conversion")
        if row.get("measurement_type") == "floor_area" and row.get("unit") != "m2":
            problems.append(f"rows[{index}] floor area unit must be m2")
        if row.get("uncertainty_m") is None:
            problems.append(f"rows[{index}] missing uncertainty")
        source = row.get("source") or ""
        if source in {"estimated", "synthetic", "chest_height_prior", "guess", "guessed"} or row.get("guessed") is True:
            problems.append(f"rows[{index}] source is guessed")
        if source in _REFERENCE:
            problems.append(f"rows[{index}] source {source} is a reference, not a tape")
        if source and source not in _PHYSICAL and source not in _REFERENCE:
            problems.append(f"rows[{index}] source {source} is not tape or laser")
        for field in ("property_id", "capture_id", "room_id"):
            if not row.get(field):
                problems.append(f"rows[{index}] missing {field}")
        if property_id and row.get("property_id") and row["property_id"] != property_id:
            problems.append(f"rows[{index}] property_id does not match the bundle")
    return problems


def group_ground_truth_by_room(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row.get("room_id") or "unknown", []).append(row)
    return grouped


def compare_prediction_to_ground_truth(rows: list[dict], predicted: list[dict], source: str | None) -> dict:
    """Score each kind with the PDF threshold. A missing prediction is not dropped."""
    openings_gt = []
    openings_pred = []
    ceilings = []
    walls = []
    footprint = None
    for row in rows:
        kind = row.get("measurement_type")
        if kind == "opening_width":
            openings_gt.append(
                {
                    "opening_id": row.get("opening_id") or row.get("measurement_id"),
                    "room_id": row.get("room_id"),
                    "wall_id": row.get("wall_id") or row.get("name"),
                    "width_m": row.get("value_m"),
                }
            )
        elif kind == "ceiling_height":
            ceilings.append(
                {
                    "room_id": row.get("room_id"),
                    "ground_truth_m": row.get("value_m"),
                    "predicted_m": row.get("predicted_m"),
                    "repeat_spread_m": row.get("repeat_spread_m"),
                }
            )
        elif kind == "wall_length":
            walls.append(
                {
                    "wall_id": row.get("wall_id") or row.get("measurement_id"),
                    "room_id": row.get("room_id"),
                    "ground_truth_m": row.get("value_m"),
                    "predicted_m": row.get("predicted_m"),
                }
            )
        elif kind == "floor_area" and footprint is None:
            footprint = {"ground_truth_m2": row.get("value_m"), "predicted_m2": row.get("predicted_m")}
    for item in predicted or []:
        if item.get("width_m") is not None or item.get("measurement_type") == "opening_width":
            openings_pred.append(item)
    report = {
        "openings": score_opening_gate(openings_gt, openings_pred, source),
        "ceiling": score_ceiling_gate(ceilings, source),
        "photo": score_photo(walls, footprint, source, calibration={"scale": source if source in _PHYSICAL else None}),
    }
    return report


def score_opening_gate(ground_truth: list[dict], predicted: list[dict], source: str | None) -> dict:
    if not ground_truth:
        return {
            "status": "BLOCKED",
            "reason": "missing_ground_truth",
            "total_openings": 0,
            "matched": 0,
            "missed": 0,
            "phantom": 0,
            "within_2cm": 0,
            "accuracy": None,
            "threshold": OPENING_PASS_RATE,
            "max_error_m": OPENING_MAX_ERROR_M,
        }
    report = score_openings(ground_truth, predicted, source=source)
    return {
        "status": report["status"],
        "reason": report.get("reason"),
        "total_openings": report.get("trials"),
        "matched": report.get("matched"),
        "missed": report.get("missed"),
        "phantom": report.get("phantoms"),
        "within_2cm": report.get("within_2cm"),
        "accuracy": report.get("pass_rate"),
        "threshold": OPENING_PASS_RATE,
        "max_error_m": OPENING_MAX_ERROR_M,
        "rows": report.get("rows"),
    }


def score_ceiling_gate(rooms: list[dict], source: str | None) -> dict:
    if not rooms:
        return {
            "status": "BLOCKED",
            "reason": "missing_ceiling",
            "threshold_m": CEILING_MAX_ERROR_M,
            "repeat_spread_m": CEILING_REPEAT_SPREAD_M,
            "per_room": [],
        }
    report = score_ceilings(rooms, source=source)
    errors = [row.get("absolute_error_m") for row in report.get("rows") or [] if row.get("absolute_error_m") is not None]
    bias = None if not errors else round(sum(errors) / len(errors), 4)
    return {
        "status": report["status"],
        "reason": report.get("reason"),
        "classification": report.get("classification"),
        "bias_m": bias,
        "repeatability_m": report.get("max_repeat_spread_m"),
        "threshold_m": CEILING_MAX_ERROR_M,
        "repeat_spread_m": CEILING_REPEAT_SPREAD_M,
        "per_room": report.get("rows"),
        "missing_ceiling": [row.get("room_id") for row in report.get("rows") or [] if row.get("kind") == "missing"],
    }


def validate_video_capture(path: Path | None) -> dict:
    """Official video needs an ordered walk and poses. Missing poses are DEGRADED, not a fake reconstruction."""
    from .assessment_run import validate_video_dir

    if path is None or not Path(path).exists():
        return {"status": "BLOCKED", "reason": "missing_video", "errors": ["video capture is missing"]}
    folder = Path(path) if Path(path).is_dir() else Path(path).parent
    videos = [Path(path)] if Path(path).is_file() and Path(path).suffix.lower() == ".mp4" else sorted(folder.glob("*.mp4"))
    if not videos:
        return {"status": "BLOCKED", "reason": "missing_video", "errors": ["video capture needs an mp4 walkthrough"]}
    if not (folder / "odometry.csv").is_file():
        return {
            "status": "DEGRADED",
            "reason": "missing_pose",
            "errors": ["official video route needs odometry.csv; the 1.40 m photo prior is not used"],
        }
    errors = validate_video_dir(path)
    if errors:
        return {"status": "BLOCKED", "reason": "invalid_video", "errors": errors}
    return {"status": "ok", "reason": None, "errors": []}


def score_video_against_ground_truth(plan: dict | None, walls: list[dict], source: str | None) -> dict:
    report = score_video(plan, walls, source=source)
    reasons = list((plan or {}).get("degraded_reasons") or [])
    notes = " ".join(str(item) for item in ((plan or {}).get("notes") or []))
    if report.get("status") == "DEGRADED":
        if "missing_pose" in reasons:
            chosen = "missing_pose"
        elif "low_texture" in reasons:
            chosen = "low_texture"
        elif "weak_overlap" in reasons:
            chosen = "weak_overlap"
        elif "insufficient_points" in reasons or "not enough wall points" in notes:
            chosen = "insufficient_wall_support"
        else:
            chosen = "no_room_closure"
        report = {**report, "reason": chosen}
    report["threshold"] = VIDEO_WALL_REL
    return report


def incumbent_bundle(doc: dict) -> dict:
    """Normalize a Polycam or Magicplan export. Lengths that are absent stay absent."""
    if not isinstance(doc, dict):
        return {"app": None, "version": None, "property_id": None, "rooms": []}
    app = doc.get("app") or doc.get("vendor") or doc.get("export")
    rooms_out = []
    source_rooms = [room for room in (doc.get("rooms") or []) if isinstance(room, dict)]
    for floor in doc.get("floors") or []:
        if isinstance(floor, dict):
            source_rooms.extend(room for room in (floor.get("rooms") or []) if isinstance(room, dict))
    for room in source_rooms:
        walls = []
        for wall in room.get("walls") or []:
            if not isinstance(wall, dict):
                continue
            length = wall.get("length_m")
            if length is None:
                length = wall.get("lengthInMeters")
            if length is None and wall.get("unit") in {None, "m"} and wall.get("length") is not None:
                length = wall.get("length")
            if length is None:
                continue
            walls.append({"wall_id": wall.get("wall_id") or wall.get("id"), "length_m": float(length)})
        rooms_out.append({"room_id": room.get("room_id") or room.get("id") or room.get("name"), "walls": walls})
    return {
        "app": app,
        "version": doc.get("version"),
        "property_id": doc.get("property_id"),
        "rooms": rooms_out,
        "dimensions": normalize_incumbent(doc),
    }


def write_ground_truth_report(destination: Path, manifest: dict, rows: list[dict], problems: list[str]) -> None:
    folder = Path(destination) / "benchmark"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "ground_truth_validation.json").write_text(
        json.dumps({"property_id": manifest.get("property_id"), "rows": len(rows), "source": (manifest.get("ground_truth") or {}).get("source")}, indent=2),
        encoding="utf-8",
    )
    (folder / "errors.json").write_text(json.dumps(problems, indent=2), encoding="utf-8")
    by_room = group_ground_truth_by_room(rows)
    summary = {
        "property_id": manifest.get("property_id"),
        "rooms": list(by_room),
        "measurements": len(rows),
        "blocked": bool(problems) or not rows,
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        f"# Ground truth {manifest.get('property_id') or ''}",
        "",
        f"Measurements: {len(rows)}.",
        "",
    ]
    if problems:
        lines.append("The file is not tape or laser ground truth:")
        lines.extend(f"- {item}" for item in problems)
    elif not rows:
        lines.append("No ground-truth file was supplied. The physical gates stay BLOCKED.")
    else:
        lines.append("Rows parsed. A FARO source cannot satisfy a tape gate.")
    (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _gate_value(gate: dict) -> str:
    for key in ("pass_rate", "accuracy", "max_error_m", "beat_or_tie_rate", "footprint_error_percent", "classification"):
        if gate.get(key) is not None:
            return str(gate.get(key))
    return ""


def render_gate_table(scored: dict, manifest: dict | None = None, elapsed_s: float | None = None) -> str:
    """Human report. BLOCKED stays BLOCKED."""
    manifest = manifest or {}
    gates = scored.get("gates") or {}
    lines = [
        f"# Assessment {manifest.get('property_id') or scored.get('property_id') or ''}",
        "",
        f"Rooms: {', '.join(manifest.get('rooms') or []) or 'not declared'}.",
        f"Ground-truth source: {(manifest.get('ground_truth') or {}).get('source') or 'not supplied'}.",
        f"Elapsed seconds: {'' if elapsed_s is None else round(elapsed_s, 3)}.",
        "",
        "| Gate | Result | Value | Threshold | Evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    allowed = {"PASS", "FAIL", "BLOCKED", "DEGRADED"}
    for name, gate in gates.items():
        if not isinstance(gate, dict):
            continue
        status = gate.get("status") if gate.get("status") in allowed else "BLOCKED"
        evidence = str(gate.get("reason") or "").replace("|", "/")
        lines.append(f"| {name} | {status} | {_gate_value(gate)} | {gate.get('threshold') or ''} | {evidence} |")
    missing = (scored.get("evidence") or {}).get("missing_manifest_fields") or []
    if missing:
        lines.extend(["", "Missing evidence: " + ", ".join(str(item) for item in missing) + "."])
    if scored.get("contract_errors"):
        lines.extend(["", "Errors:"])
        lines.extend(f"- {item}" for item in scored["contract_errors"])
    lines.append("")
    lines.append("A confidence interval is the one stored on each measurement. A missing interval is not filled in here.")
    lines.append("")
    return "\n".join(lines)


def write_assessment_artifacts(destination: Path, scored: dict, manifest: dict | None = None, elapsed_s: float | None = None) -> None:
    """Extra files the walk-in command must leave behind. Existing plan files are not replaced."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = manifest or {}
    gates = scored.get("gates") or {}
    incumbent = gates.get("incumbent") or {"status": "BLOCKED", "reason": "no Polycam or Magicplan export"}
    repeat = dict(gates.get("repeatability") or {"status": "BLOCKED", "reason": "repeat capture is missing"})
    repeat["proxy_experiment"] = repeat.get("proxy_experiment", True)
    (destination / "incumbent.json").write_text(json.dumps(incumbent, indent=2), encoding="utf-8")
    (destination / "repeatability.json").write_text(json.dumps(repeat, indent=2), encoding="utf-8")
    (destination / "fix_loop.json").write_text(
        json.dumps(
            {
                "status": "BLOCKED",
                "reason": "two physical captures were not supplied to the fix loop",
                "proxy_experiment": True,
                "gate": "wall_repeatability",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    rows = []
    problems = []
    gt = manifest.get("ground_truth") or {}
    path = gt.get("path") if isinstance(gt, dict) else None
    if path and Path(path).is_file():
        loaded = load_ground_truth(Path(path))
        rows = loaded.get("rows") or []
        problems = validate_ground_truth_rows(rows, manifest.get("property_id"))
    elif not rows:
        problems = ["ground truth file is missing"]
    write_ground_truth_report(destination, manifest, rows, problems)
    report_path = destination / "final_report.md"
    table = render_gate_table(scored, manifest, elapsed_s)
    existing = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
    report_path.write_text(table + "\n" + existing, encoding="utf-8")


def bundle_status(manifest_path: Path) -> dict:
    """Validate a bundle and return one structured result. No traceback for a bad file."""
    manifest = load_benchmark_manifest(manifest_path)
    if manifest.get("exit_code") == 2 or manifest.get("reason") in {"malformed_manifest", "missing_manifest"}:
        return manifest
    for check in (
        validate_benchmark_manifest,
        validate_property_identity,
        validate_required_tiers,
        validate_ground_truth,
        validate_repeat_capture,
        validate_incumbent_export,
    ):
        result = check(manifest)
        if result.get("exit_code") == 2 or result.get("status") == "BLOCKED":
            return result
    return {"status": "ok", "property_id": manifest.get("property_id"), "missing": [], "exit_code": 0, "photo_wall_threshold": PHOTO_WALL_REL, "photo_footprint_threshold": PHOTO_FOOTPRINT_REL}
