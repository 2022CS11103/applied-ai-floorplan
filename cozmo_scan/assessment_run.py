"""One command that runs every tier the manifest actually contains.

A capture from another property is rejected before any plan is built.
A missing tape file stays BLOCKED. A room that does not close is DEGRADED.
Nothing here invents a measurement or marks that case PASS.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
import platform
import sys
from pathlib import Path

from .assessment import (
    _git_commit,
    load_json,
    render_markdown,
    run_manifest,
    score_video,
    write_assessment,
)
from .photo import validate_photo_capture


def _as_block(value, property_id: str | None) -> dict:
    if isinstance(value, dict):
        return {
            "path": value.get("path") or value.get("capture_b"),
            "property_id": value.get("property_id") or property_id,
            "tier": value.get("tier"),
        }
    if isinstance(value, str):
        return {"path": value, "property_id": property_id, "tier": None}
    return {"path": None, "property_id": property_id, "tier": None}


def normalize_manifest(doc: dict) -> dict:
    """Accept the canonical blocks and the older flat path keys."""
    if not isinstance(doc, dict):
        raise ValueError("manifest must be a JSON object")
    property_id = doc.get("property_id")
    rooms = doc.get("rooms") or doc.get("room_ids") or []
    tiers = doc.get("tiers") if isinstance(doc.get("tiers"), dict) else {}
    return {
        "property_id": property_id,
        "rooms": list(rooms),
        "connector": doc.get("connector"),
        "lidar": _as_block(doc.get("lidar") or doc.get("lidar_capture") or tiers.get("lidar"), property_id),
        "video": _as_block(doc.get("video") or doc.get("video_capture") or tiers.get("video"), property_id),
        "photos": _as_block(doc.get("photos") or doc.get("photo_folder") or tiers.get("photo") or tiers.get("photos"), property_id),
        "repeat": _as_block(doc.get("repeat") or doc.get("repeat_capture"), property_id),
        "ground_truth": _as_block(doc.get("ground_truth"), property_id),
        "damage": _as_block(doc.get("damage") or doc.get("damage_ground_truth"), property_id),
        "incumbent": _as_block(doc.get("incumbent") or doc.get("incumbent_export"), property_id),
        "frame_profiles": doc.get("frame_profiles"),
        "predictions": doc.get("predictions") or {},
    }


def validate_manifest_identity(manifest: dict, root: Path | None = None) -> list[str]:
    """Reject a bundle that mixes two properties or an undeclared room."""
    errors = []
    property_id = manifest.get("property_id")
    if not property_id:
        errors.append("property_id is required")
    rooms = list(manifest.get("rooms") or [])
    if rooms and len(set(rooms)) != len(rooms):
        errors.append("room ids must be unique")
    for name in ("lidar", "video", "photos", "repeat", "ground_truth", "damage", "incumbent"):
        block = manifest.get(name) or {}
        other = block.get("property_id")
        if block.get("path") and other and property_id and other != property_id:
            errors.append(f"{name} belongs to property {other}, not {property_id}")
    connector = manifest.get("connector") or {}
    for room_id in connector.get("rooms") or []:
        if rooms and room_id not in rooms:
            errors.append(f"connector room {room_id} is not in this property")
    if root is None:
        return errors
    allowed = set(rooms)
    allowed.update(connector.get("rooms") or [])
    allowed.add("connector")
    for label in ("ground_truth", "damage", "incumbent", "repeat"):
        path = _resolve(root, (manifest.get(label) or {}).get("path"))
        if path is None or not path.is_file():
            continue
        try:
            doc = load_json(path)
        except ValueError:
            errors.append(f"{label} is not a JSON object")
            continue
        other = doc.get("property_id")
        if other and property_id and other != property_id:
            errors.append(f"{label} file belongs to property {other}, not {property_id}")
        for row in list(doc.get("measurements") or []) + list(doc.get("regions") or []) + list(doc.get("openings") or []):
            room_id = row.get("room_id") or row.get("room")
            if rooms and room_id and room_id not in rooms:
                errors.append(f"{label} room {room_id} is not in this property")
    photo_path = _resolve(root, (manifest.get("photos") or {}).get("path"))
    if photo_path is not None and photo_path.is_dir() and rooms:
        from .photo import room_folders

        for folder in room_folders(photo_path):
            if folder.name not in allowed:
                errors.append(f"photo room {folder.name} is not in this property")
    return errors


def _resolve(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def validate_lidar_dir(path: Path | None) -> list[str]:
    if path is None:
        return ["lidar capture is missing"]
    if not path.is_dir():
        return [f"lidar capture is not a directory: {path.name}"]
    errors = []
    depth = path / "depth"
    depth_frames = sorted(depth.glob("*.png")) if depth.is_dir() else []
    if not depth_frames:
        errors.append("lidar capture needs depth/*.png")
    confidence = path / "confidence"
    confidence_frames = sorted(confidence.glob("*.png")) if confidence.is_dir() else []
    if not confidence_frames:
        errors.append("lidar capture needs confidence/*.png")
    if depth_frames and confidence_frames and len(depth_frames) != len(confidence_frames):
        errors.append(f"depth has {len(depth_frames)} frames and confidence has {len(confidence_frames)}")
    if depth_frames:
        import cv2

        sample = cv2.imread(str(depth_frames[0]), cv2.IMREAD_UNCHANGED)
        if sample is None:
            errors.append("depth frame is not a readable image")
        elif getattr(sample, "ndim", 0) != 2:
            errors.append("depth frame must be a single-channel range image")
    if not (path / "odometry.csv").is_file():
        errors.append("lidar capture needs odometry.csv")
    if not (path / "camera_matrix.csv").is_file():
        errors.append("lidar capture needs camera_matrix.csv")
    return errors


def _timestamp_errors(folder: Path) -> list[str]:
    odom = folder / "odometry.csv"
    if not odom.is_file():
        return []
    from .capture import load_intrinsics, load_poses

    k_path = folder / "camera_matrix.csv"
    k = load_intrinsics(k_path) if k_path.is_file() else None
    poses = load_poses(odom, k)
    stamps = [pose.timestamp for pose in poses]
    if stamps != sorted(stamps):
        return ["video odometry timestamps are not in frame order"]
    return []


def validate_video_dir(path: Path | None) -> list[str]:
    if path is None:
        return ["video capture is missing"]
    if path.is_file() and path.suffix.lower() == ".mp4":
        return _timestamp_errors(path.parent)
    if not path.is_dir():
        return [f"video capture is not a file or directory: {path.name}"]
    videos = sorted(path.glob("*.mp4"))
    if not videos:
        return ["video capture needs an mp4 walkthrough"]
    return _timestamp_errors(path)


def validate_photo_dir(path: Path | None, rooms: list[str], calibration: dict | None) -> list[str]:
    if path is None:
        return ["photo folder is missing"]
    if not path.is_dir():
        return [f"photo folder is missing: {path.name}"]
    report = validate_photo_capture(path, expected_rooms=rooms or None, calibration=calibration)
    return [f"{item.get('code')} {item.get('room_id') or item.get('name') or ''}".strip() for item in report["issues"]]


def _sha256(path: Path) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _fingerprint(path: Path | None) -> str | None:
    """Hash a file, or the names and sizes of a capture directory. Depth frames are not re-read."""
    if path is None or not path.exists():
        return None
    if path.is_file():
        return _sha256(path)
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if child.is_file():
            digest.update(child.relative_to(path).as_posix().encode())
            digest.update(str(child.stat().st_size).encode())
    return digest.hexdigest()


def _portable(value, cwd: Path):
    if isinstance(value, dict):
        return {key: _portable(item, cwd) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable(item, cwd) for item in value]
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute():
            try:
                return candidate.resolve().relative_to(cwd.resolve()).as_posix()
            except ValueError:
                return candidate.name
    return value


def _package_versions() -> dict:
    versions = {}
    for name, module in (
        ("numpy", "numpy"),
        ("pillow", "PIL"),
        ("opencv", "cv2"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("jsonschema", "jsonschema"),
    ):
        try:
            if name == "jsonschema":
                from importlib.metadata import version

                versions[name] = version("jsonschema")
            else:
                imported = __import__(module)
                versions[name] = getattr(imported, "__version__", "unknown")
        except ImportError:
            versions[name] = "missing"
    return versions


def _environment(command: str) -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _package_versions(),
        "command": command,
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git_commit": _git_commit(),
    }


_ROW_GATES = {
    "ceiling_height": ("ceiling",),
    "wall_length": ("photo_walls", "photo_footprint", "video_walls"),
    "floor_area": ("photo_footprint", "stitch"),
    "opening_width": ("openings",),
}


def measurement_contract_issues(doc: dict) -> list[tuple[str, str]]:
    """A tape row must name the property, room, value, unit, uncertainty, source, and capture.

    FARO and any other external reference stay on the row. They do not become tape.
    """
    issues: list[tuple[str, str]] = []
    parent_property = doc.get("property_id")
    parent_capture = doc.get("capture_id")
    parent_source = str(doc.get("source") or "").lower()
    if parent_source in {"estimated", "synthetic", "chest_height_prior", "1.40m"}:
        issues.append(("ceiling", "estimated source is not ground truth"))
    for index, row in enumerate(doc.get("measurements") or []):
        gates = _ROW_GATES.get(row.get("measurement_type"), ("ceiling",))
        if not isinstance(row, dict):
            issues.append((gates[0], f"measurements[{index}] is not an object"))
            continue
        if "property_id" not in row and not parent_property:
            issues.append((gates[0], f"measurements[{index}] missing property_id"))
        if "capture_id" not in row and not parent_capture:
            issues.append((gates[0], f"measurements[{index}] missing capture_id"))
        for field in ("room_id", "measurement_type", "value_m", "unit", "uncertainty_m"):
            if field not in row:
                for gate in gates:
                    issues.append((gate, f"measurements[{index}] missing {field}"))
        if "source" not in row and not parent_source:
            for gate in gates:
                issues.append((gate, f"measurements[{index}] missing source"))
        kind = row.get("measurement_type")
        if kind == "wall_length" and "wall_id" not in row:
            for gate in gates:
                issues.append((gate, f"measurements[{index}] missing wall_id"))
        if kind == "opening_width" and "opening_id" not in row:
            issues.append(("openings", f"measurements[{index}] missing opening_id"))
        unit = row.get("unit")
        if unit is not None and kind == "floor_area" and unit != "m2":
            issues.append(("photo_footprint", f"measurements[{index}] floor area unit must be m2"))
        if unit is not None and kind != "floor_area" and unit != "m":
            for gate in gates:
                issues.append((gate, f"measurements[{index}] unit must be m"))
        row_source = str(row.get("source") or parent_source).lower()
        if row_source in {"faro", "external_reference"}:
            for gate in gates:
                issues.append((gate, f"measurements[{index}] source {row_source} is a reference, not a tape"))
    for index, item in enumerate(doc.get("openings") or []):
        for field in ("opening_id", "room_id", "wall_id", "width_m", "unit", "uncertainty_m", "source", "capture_id"):
            if field not in item and not (field == "capture_id" and parent_capture) and not (field == "source" and parent_source):
                issues.append(("openings", f"openings[{index}] missing {field}"))
        if "property_id" not in item and not parent_property:
            issues.append(("openings", f"openings[{index}] missing property_id"))
        item_source = str(item.get("source") or parent_source).lower()
        if item_source in {"faro", "external_reference"}:
            issues.append(("openings", f"openings[{index}] source {item_source} is a reference, not a tape"))
    return issues


def _tier_status(plan: dict | None, validation_errors: list[str]) -> str:
    if validation_errors and any("missing" in item for item in validation_errors) and plan is None:
        return "BLOCKED"
    if plan is None:
        return "BLOCKED"
    if plan.get("status") == "degraded" or not (plan.get("property") or {}).get("rooms"):
        return "DEGRADED"
    return "RAN"


def run_assessment(manifest_path: Path, out_dir: Path | None = None, command: str | None = None) -> dict:
    started = time.perf_counter()
    manifest_path = Path(manifest_path)
    raw = load_json(manifest_path)
    manifest = normalize_manifest(raw)
    root = manifest_path.parent
    identity_errors = validate_manifest_identity(manifest, root)
    destination = Path(out_dir) if out_dir is not None else Path("runs") / "assessment" / str(manifest.get("property_id") or "unknown")
    destination.mkdir(parents=True, exist_ok=True)

    lidar_path = _resolve(root, manifest["lidar"]["path"])
    video_path = _resolve(root, manifest["video"]["path"])
    photo_path = _resolve(root, manifest["photos"]["path"])
    repeat_path = _resolve(root, manifest["repeat"]["path"])
    gt_path = _resolve(root, manifest["ground_truth"]["path"])

    validation = {
        "property_id": manifest.get("property_id"),
        "rejected": bool(identity_errors),
        "identity_errors": identity_errors,
        "lidar": validate_lidar_dir(lidar_path) if not identity_errors else [],
        "video": validate_video_dir(video_path) if not identity_errors else [],
        "photos": validate_photo_dir(
            photo_path,
            manifest.get("rooms") or [],
            None if gt_path is None or not gt_path.is_file() else (load_json(gt_path).get("calibration")),
        )
        if not identity_errors
        else [],
    }
    (destination / "input_manifest.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    (destination / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    (destination / "command.txt").write_text((command or f"python run.py assessment --manifest {manifest_path.name}") + "\n", encoding="utf-8")
    env = _environment(command or f"python run.py assessment --manifest {manifest_path.name}")
    env["input_sha256"] = {
        "manifest": _sha256(manifest_path),
        "lidar": _fingerprint(lidar_path),
        "video": _fingerprint(video_path),
        "photos": _fingerprint(photo_path),
        "ground_truth": _sha256(gt_path) if gt_path is not None else None,
    }
    (destination / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    if identity_errors:
        report = {
            "property_id": manifest.get("property_id"),
            "rejected": True,
            "gates": {},
            "failed": False,
            "reason": "; ".join(identity_errors),
        }
        (destination / "final_report.md").write_text(
            "# Assessment rejected\n\n" + "\n".join(f"- {item}" for item in identity_errors) + "\n",
            encoding="utf-8",
        )
        for name in ("lidar", "video", "photo", "room_plans", "property_plan"):
            (destination / name).mkdir(exist_ok=True)
        for name in ("measurements.json", "damage.json", "concealed_damage.json", "scope.json", "benchmark.json", "compliance.json"):
            (destination / name).write_text(json.dumps({"rejected": True, "reason": report["reason"]}, indent=2), encoding="utf-8")
        report["output_dir"] = str(destination)
        report["exit_code"] = 2
        from .benchmark import write_assessment_artifacts

        reported = dict(manifest)
        if gt_path is not None:
            reported["ground_truth"] = {**manifest["ground_truth"], "path": str(gt_path)}
        write_assessment_artifacts(destination, report, reported, time.perf_counter() - started)
        return report

    plans = {}
    from .pipeline import run_one

    for tier, path, errors in (
        ("lidar", lidar_path, validation["lidar"]),
        ("video", video_path, validation["video"]),
        ("photo", photo_path, validation["photos"]),
    ):
        tier_dir = destination / tier
        if path is None or errors:
            plans[tier] = None
            continue
        plans[tier] = run_one(path, tier, tier_dir)

    for tier, plan in plans.items():
        (destination / "measurements.json").parent.mkdir(parents=True, exist_ok=True)
    measurements = {
        tier: {
            "status": _tier_status(plan, validation[tier if tier != "photo" else "photos"]),
            "rooms": 0 if plan is None else len((plan.get("property") or {}).get("rooms") or []),
            "opening_evidence": None if plan is None else (plan.get("quality") or {}).get("opening_evidence"),
        }
        for tier, plan in plans.items()
    }
    if manifest.get("frame_profiles"):
        from .layout import detect_openings_multiframe

        profiles = manifest["frame_profiles"]
        if isinstance(profiles, str):
            profiles = load_json(_resolve(root, profiles) or Path(profiles))
        opening_rows = []
        for profile in profiles.get("walls") or []:
            frames = [__import__("numpy").asarray(frame, dtype=float) for frame in profile.get("frames") or []]
            aggregated = detect_openings_multiframe(frames) if frames else None
            opening_rows.append({"wall_id": profile.get("wall_id"), "opening": aggregated, "source": "per_frame_profiles"})
        measurements["per_frame_openings"] = opening_rows
    (destination / "measurements.json").write_text(json.dumps(measurements, indent=2), encoding="utf-8")

    damage_doc = []
    concealed = []
    scope = []
    for plan in plans.values():
        if not plan:
            continue
        for room in (plan.get("property") or {}).get("rooms") or []:
            for region in room.get("damage") or []:
                damage_doc.append(region)
        concealed.extend(plan.get("concealed_damage") or [])
        scope.extend(plan.get("scope_line_items") or [])
    (destination / "damage.json").write_text(json.dumps(damage_doc, indent=2), encoding="utf-8")
    (destination / "concealed_damage.json").write_text(json.dumps(concealed, indent=2), encoding="utf-8")
    (destination / "scope.json").write_text(json.dumps(scope, indent=2), encoding="utf-8")
    (destination / "room_plans").mkdir(exist_ok=True)
    (destination / "property_plan").mkdir(exist_ok=True)
    room_index = {}
    for tier, plan in plans.items():
        rooms = [] if plan is None else (plan.get("property") or {}).get("rooms") or []
        room_index[tier] = [room.get("id") or room.get("room_id") for room in rooms]
    (destination / "room_plans" / "index.json").write_text(json.dumps(room_index, indent=2), encoding="utf-8")
    if plans.get("photo") is not None:
        from .pipeline import _jsonable

        (destination / "property_plan" / "plan.json").write_text(
            json.dumps(_jsonable(plans["photo"]), indent=2),
            encoding="utf-8",
        )

    flat = {
        "property_id": manifest.get("property_id"),
        "room_ids": manifest.get("rooms") or [],
        "lidar_capture": None if lidar_path is None or validation["lidar"] else str(lidar_path),
        "video_capture": None if video_path is None or validation["video"] else str(video_path),
        "photo_folder": None if photo_path is None or validation["photos"] else str(photo_path),
        "repeat_capture": None if repeat_path is None or not repeat_path.is_file() else str(repeat_path),
        "ground_truth": None if gt_path is None else str(gt_path),
        "damage_ground_truth": manifest["damage"]["path"],
        "incumbent_export": manifest["incumbent"]["path"],
        "predictions": {},
    }
    video_plan = plans.get("video")
    if video_plan is not None:
        video_file = destination / "video" / "plan.json"
        flat["predictions"]["video"] = str(video_file)
    scored = run_manifest(flat, root=root)
    if validation["video"] and video_plan is None:
        scored["gates"]["video_walls"] = {"status": "BLOCKED", "reason": "; ".join(validation["video"])}
    elif video_plan is not None and (video_plan.get("status") == "degraded" or not (video_plan.get("property") or {}).get("rooms")):
        scored["gates"]["video_walls"] = score_video(video_plan, [], source=None)
    if validation["photos"] and plans.get("photo") is None:
        reason = "; ".join(validation["photos"])
        for name in ("photo_walls", "photo_footprint", "stitch"):
            scored["gates"][name] = {"status": "BLOCKED", "reason": reason}
    if validation["lidar"] and plans.get("lidar") is None:
        scored.setdefault("capture_errors", {})["lidar"] = validation["lidar"]
    if gt_path is not None and gt_path.is_file():
        for gate, message in measurement_contract_issues(load_json(gt_path)):
            current = scored["gates"].get(gate) or {}
            if current.get("status") in {"PASS", "FAIL"}:
                scored["gates"][gate] = {
                    "status": "BLOCKED",
                    "reason": "ground truth does not meet the measurement schema: " + message,
                }
        validation["ground_truth_schema"] = [message for _, message in measurement_contract_issues(load_json(gt_path))]
        (destination / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    scored["failed"] = any(gate.get("status") == "FAIL" for gate in scored["gates"].values())
    structural = [message for message in validation.get("ground_truth_schema") or [] if "reference" not in message]
    if structural:
        scored["contract_errors"] = structural
    scored = _portable(scored, Path.cwd())
    scored["tier_status"] = {tier: measurements[tier]["status"] for tier in ("lidar", "video", "photo")}
    write_assessment(scored, destination)
    (destination / "benchmark.json").write_text(json.dumps(scored, indent=2), encoding="utf-8")
    (destination / "benchmark.md").write_text(render_markdown(scored), encoding="utf-8")
    compliance = {
        "property_id": manifest.get("property_id"),
        "gates": {name: {"status": gate.get("status"), "reason": gate.get("reason")} for name, gate in scored["gates"].items()},
        "tiers": scored["tier_status"],
    }
    (destination / "compliance.json").write_text(json.dumps(compliance, indent=2), encoding="utf-8")
    (destination / "compliance.md").write_text(render_markdown(scored), encoding="utf-8")
    (destination / "final_report.md").write_text(render_markdown(scored), encoding="utf-8")
    scored["output_dir"] = str(destination)
    if structural:
        scored["exit_code"] = 2
    else:
        scored["exit_code"] = 1 if scored.get("failed") else 0
    from .benchmark import write_assessment_artifacts

    reported = dict(manifest)
    if gt_path is not None:
        reported["ground_truth"] = {**manifest["ground_truth"], "path": str(gt_path)}
    write_assessment_artifacts(destination, scored, reported, time.perf_counter() - started)
    return scored
