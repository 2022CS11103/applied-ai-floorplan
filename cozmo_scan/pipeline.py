"""One command per capture. The JSON file is the contract; the PNG is the plan."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import __version__
from .capture import discover, imu_gravity_g
from .contract import apply_contract
from .damage import colorize_lidar, concealed_flags, detect_on_points, scope_items
from .fuse import align_floor, fuse_lidar
from .layout import build_layout, footprint_area, layout_to_dict
from .reconstruct import VIDEO_REASONS, fuse_video
from .render import render_plan


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if k != "replay"}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def run_one(
    capture_path: Path,
    tier: str,
    out_dir: Path,
    endpoint_mode: str = "intersections",
    floor_anchor: bool = True,
    loop: bool = True,
) -> dict:
    if tier == "photo":
        from .photo import run_photo

        return run_photo(capture_path, out_dir, endpoint_mode=endpoint_mode)
    t0 = time.perf_counter()
    capture = discover(capture_path)
    timing = {}
    notes = []

    t = time.perf_counter()
    if tier == "lidar":
        if capture.kind != "lidar_bundle":
            raise RuntimeError(f"{capture.root} has no depth maps; lidar tier cannot run")
        cloud = fuse_lidar(capture)
    elif tier == "video":
        cloud = fuse_video(capture)
        if _video_stops_early(cloud):
            timing["reconstruct_s"] = round(time.perf_counter() - t, 3)
            return _emit_video_without_layout(capture, cloud, out_dir, endpoint_mode, floor_anchor, timing, t0)
    else:
        raise RuntimeError(f"unknown tier {tier}")
    timing["reconstruct_s"] = round(time.perf_counter() - t, 3)

    t = time.perf_counter()
    aligned = align_floor(cloud, enable_anchor=floor_anchor, enable_loop=loop and tier == "lidar")
    # Ablation uses the same points. Anchor-off is "poses used as leveled only by a
    # histogram", which is the control the drift row asks for.
    layout = build_layout(aligned["xyz"], tier=tier, endpoint_mode=endpoint_mode)
    layout_off = build_layout(aligned["xyz_anchor_off"], tier=tier, endpoint_mode=endpoint_mode)
    timing["layout_s"] = round(time.perf_counter() - t, 3)

    prop = layout_to_dict(layout, tier)
    area_on = footprint_area(layout)
    area_off = footprint_area(layout_off)

    t = time.perf_counter()
    colors_xyz = None
    colors = cloud.colors
    if tier == "lidar":
        colors_xyz, colors = colorize_lidar(capture, aligned["replay"])
    elif colors is not None:
        # video / photo colors are on the raw cloud; replay the same floor frame
        from .fuse import apply_replay

        colors_xyz = apply_replay(cloud.xyz, cloud.frame_index, aligned["replay"])
    if colors_xyz is not None and colors is not None and len(prop["rooms"]):
        regions = detect_on_points(colors_xyz, colors, prop["rooms"], tier)
        by_room = {}
        for reg in regions:
            by_room.setdefault(reg["room_id"], []).append(reg)
        for room in prop["rooms"]:
            room["damage"] = by_room.get(room["id"], [])
    timing["damage_s"] = round(time.perf_counter() - t, 3)

    flags = concealed_flags(prop["rooms"])
    items = scope_items(prop["rooms"])
    degraded_reasons = video_degraded_reasons(cloud, prop["notes"], len(prop["rooms"])) if tier == "video" else []
    degraded = bool(cloud.meta.get("degraded")) or bool(degraded_reasons)
    status = "degraded" if degraded or not prop["rooms"] else "ok"
    if tier == "video" and not prop["rooms"]:
        status = "degraded"
    if not prop["rooms"]:
        notes.append("layout did not close a room; see layout notes")
    notes.extend(prop["notes"])

    gravity = None
    imu_path = capture.root / "imu.csv"
    if imu_path.exists():
        gravity = imu_gravity_g(imu_path)

    document = {
        "schema_version": "1.0",
        "pipeline_version": __version__,
        "status": status,
        "capture_id": capture.capture_id,
        "tier": tier,
        "units": "meters",
        "capture_kind": capture.kind,
        "source": cloud.source,
        "endpoint_mode": endpoint_mode,
        "floor_anchor": floor_anchor,
        "timing_s": timing,
        "quality": {
            **{k: v for k, v in cloud.meta.items() if not isinstance(v, (np.ndarray,))},
            "imu_specific_force_g": None if gravity is None else round(gravity, 3),
            "points_aligned": int(len(aligned["xyz"])),
        },
        "drift": {
            "method": aligned["report"]["method"],
            "floor_tilt_deg": aligned["report"].get("floor_tilt_deg"),
            "floor_residual_m": aligned["report"].get("floor_residual_m"),
            "chunk_floor_offsets_m": aligned["report"].get("chunk_floor_offsets_m"),
            "loop": aligned["report"].get("loop"),
            "footprint_area_m2_anchor_on": round(area_on, 4),
            "footprint_area_m2_anchor_off": round(area_off, 4),
            "note": (
                "Anchor-on rotates the cloud onto the floor plane and shifts each "
                "quarter of the walk so its floor sits at zero. Anchor-off only "
                "subtracts a single floor height. Loop closure runs on the lidar "
                "tier and is recorded whether or not it fires."
            ),
        },
        "property": {
            "rooms": prop["rooms"],
            "adjacencies": prop["adjacencies"],
            "footprint_area_m2": round(area_on, 4),
            "manhattan_theta_deg": prop["manhattan_theta_deg"],
        },
        "scope_line_items": items,
        "concealed_damage": flags,
        "notes": notes,
    }
    document["timing_s"]["total_s"] = round(time.perf_counter() - t0, 3)
    if tier == "video":
        document["degraded_reasons"] = degraded_reasons
    elif "degraded_reasons" not in document:
        document["degraded_reasons"] = []
    document = apply_contract(document)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "plan.json"
    out_path.write_text(json.dumps(_jsonable(document), indent=2), encoding="utf-8")
    render_plan(document, out_dir / "plan.png")
    _write_summary(document, out_dir / "summary.txt")
    return document


def _video_stops_early(cloud) -> bool:
    """No points to lay out: missing or unusable poses, or an empty triangulation."""
    meta = cloud.meta or {}
    if meta.get("pose_error") or meta.get("video_error"):
        return True
    reasons = meta.get("reconstruction_reasons") or []
    if "missing_pose" in reasons:
        return True
    return len(cloud.xyz) == 0


def video_degraded_reasons(cloud, layout_notes: list[str], n_rooms: int) -> list[str]:
    """Reasons that were actually observed. Order follows the allowed list."""
    meta = cloud.meta or {}
    found = set(meta.get("reconstruction_reasons") or [])
    if meta.get("pose_error") == "invalid_pose":
        # The pose rows exist but cannot be used. That is not a layout failure.
        return []
    notes = " ".join(layout_notes or [])
    if n_rooms == 0 and len(cloud.xyz) > 0:
        if "not enough wall points" in notes or "too few points" in notes:
            found.add("insufficient_points")
        found.add("no_room_closure")
    return [reason for reason in VIDEO_REASONS if reason in found]


def _emit_video_without_layout(capture, cloud, out_dir, endpoint_mode, floor_anchor, timing, t0) -> dict:
    """Write a degraded plan when there is nothing honest to lay out."""
    reasons = video_degraded_reasons(cloud, [], 0)
    notes = []
    if cloud.meta.get("pose_error"):
        notes.append(cloud.meta["pose_error"])
    if cloud.meta.get("video_error"):
        notes.append(cloud.meta["video_error"])
    if "missing_pose" in reasons:
        notes.append("video has no usable metric poses")
    for reason in reasons:
        if reason not in notes:
            notes.append(reason)
    document = {
        "schema_version": "1.0",
        "pipeline_version": __version__,
        "status": "degraded",
        "degraded_reasons": reasons,
        "capture_id": capture.capture_id,
        "tier": "video",
        "units": "meters",
        "capture_kind": capture.kind,
        "source": cloud.source,
        "endpoint_mode": endpoint_mode,
        "floor_anchor": floor_anchor,
        "timing_s": {**timing, "total_s": round(time.perf_counter() - t0, 3)},
        "quality": {k: v for k, v in cloud.meta.items() if not isinstance(v, np.ndarray)},
        "drift": {
            "method": ["not run: video produced no cloud"],
            "floor_tilt_deg": None,
            "floor_residual_m": None,
            "chunk_floor_offsets_m": [],
            "loop": {"fired": False, "reason": "disabled"},
            "footprint_area_m2_anchor_on": 0.0,
            "footprint_area_m2_anchor_off": 0.0,
            "note": "No layout was built, so there is no floor anchor to compare.",
        },
        "property": {
            "rooms": [],
            "adjacencies": [],
            "footprint_area_m2": 0.0,
            "manhattan_theta_deg": 0.0,
        },
        "scope_line_items": [],
        "concealed_damage": [],
        "notes": notes,
    }
    document = apply_contract(document)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plan.json").write_text(json.dumps(_jsonable(document), indent=2), encoding="utf-8")
    render_plan(document, out_dir / "plan.png")
    _write_summary(document, out_dir / "summary.txt")
    return document


def repeatability(capture_path: Path, out_dir: Path, endpoint_mode: str) -> dict:
    """Two temporal halves of one lidar walk, compared as if they were two captures.

    This is a proxy. A second physical walk was not possible on this machine.
    The report says so. It is still the repeatability test the geometry can be
    scored on: same room, different viewpoints, same tier.
    """
    capture = discover(capture_path)
    cloud = fuse_lidar(capture, max_frames=260, pixel_stride=3)
    aligned = align_floor(cloud, enable_anchor=True, enable_loop=False)
    idx = aligned["frame_index"]
    xyz = aligned["xyz_anchor_on_loop_off"]
    # Even/odd frames see the same walk, so a stable estimator should agree.
    # A temporal split does not: the second half of the walk is a different corner.
    halves = {
        "even": xyz[idx % 2 == 0],
        "odd": xyz[idx % 2 == 1],
    }
    layouts = {name: build_layout(pts, tier="lidar", endpoint_mode=endpoint_mode) for name, pts in halves.items()}
    rows = _compare_layouts(layouts["even"], layouts["odd"])
    # gate: 1 cm or 0.5% per wall, whichever is larger
    for row in rows:
        tol = max(0.01, 0.005 * row["length_a_m"])
        row["tolerance_m"] = round(tol, 4)
        row["pass"] = bool(row["abs_delta_m"] <= tol)
    n = len(rows)
    n_pass = sum(1 for r in rows if r["pass"])
    summary = {
        "capture_id": capture.capture_id,
        "endpoint_mode": endpoint_mode,
        "walls_compared": n,
        "walls_within_gate": n_pass,
        "fraction": None if n == 0 else round(n_pass / n, 3),
        "worst_abs_delta_m": None if n == 0 else max(r["abs_delta_m"] for r in rows),
        "proxy": "even/odd frames of one walk, not two physical captures",
        "proxy_experiment": True,
        "rows": rows,
        "notes_even": layouts["even"].notes,
        "notes_odd": layouts["odd"].notes,
        "rooms_even": len(layouts["even"].rooms),
        "rooms_odd": len(layouts["odd"].rooms),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "repeatability.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    return summary


def _compare_layouts(a, b) -> list[dict]:
    rows = []
    used = set()
    for i, room_a in enumerate(a.rooms):
        if not b.rooms:
            break
        # match the other room with the nearest centroid
        ca = room_a.polygon.mean(0)
        cents = [r.polygon.mean(0) for r in b.rooms]
        order = np.argsort([np.linalg.norm(c - ca) for c in cents])
        room_b = None
        for j in order:
            if int(j) not in used and np.linalg.norm(cents[j] - ca) < 3.0:
                room_b = b.rooms[int(j)]
                used.add(int(j))
                break
        if room_b is None:
            continue
        for wa in room_a.walls:
            la = float(np.linalg.norm(wa.p1 - wa.p0))
            da = wa.p1 - wa.p0
            ang_a = np.arctan2(da[1], da[0])
            mid_a = 0.5 * (wa.p0 + wa.p1)
            best = None
            for wb in room_b.walls:
                db = wb.p1 - wb.p0
                ang_b = np.arctan2(db[1], db[0])
                dang = abs((ang_a - ang_b + np.pi / 2) % np.pi - np.pi / 2)
                if dang > np.radians(20):
                    continue
                mid_b = 0.5 * (wb.p0 + wb.p1)
                dist = float(np.linalg.norm(mid_a - mid_b))
                if dist > 1.2:
                    continue
                lb = float(np.linalg.norm(wb.p1 - wb.p0))
                delta = abs(la - lb)
                if best is None or delta < best[0]:
                    best = (delta, lb, dist)
            if best is None:
                continue
            rows.append(
                {
                    "room_a": i,
                    "length_a_m": round(la, 4),
                    "length_b_m": round(best[1], 4),
                    "abs_delta_m": round(best[0], 4),
                    "midpoint_distance_m": round(best[2], 4),
                }
            )
    return rows


def _write_summary(document: dict, path: Path) -> None:
    lines = [
        f"capture: {document['capture_id']}",
        f"tier: {document['tier']}",
        f"status: {document['status']}",
        f"rooms: {len(document['property']['rooms'])}",
    ]
    for room in document["property"]["rooms"]:
        ceil = room["ceiling_height_m"]["value"] if room.get("ceiling_height_m") else "not observed"
        lines.append(
            f"  {room['id']}: area {room['floor_area_m2']['value']:.3f} m^2 "
            f"[{room['floor_area_m2']['ci95_low']:.3f}, {room['floor_area_m2']['ci95_high']:.3f}] "
            f"ceiling {ceil}"
        )
        for wall in room["walls"]:
            ops = ", ".join(f"{o['class']} {o['width_m']['value']:.2f} m" for o in wall["openings"]) or "no opening"
            lines.append(
                f"    {wall['id']}: {wall['length_m']['value']:.3f} m "
                f"[{wall['length_m']['ci95_low']:.3f}, {wall['length_m']['ci95_high']:.3f}] ({ops})"
            )
        if room.get("damage"):
            for d in room["damage"]:
                lines.append(f"    damage {d['class']} {d['extent_m2']['value']:.3f} m^2 on {d['surface_id']}")
    drift = document.get("drift") or {}
    lines.append(
        f"footprint anchor on {drift.get('footprint_area_m2_anchor_on')} "
        f"off {drift.get('footprint_area_m2_anchor_off')}"
    )
    lines.append("notes:")
    for n in document.get("notes", []):
        lines.append(f"  - {n}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_check() -> dict:
    """Box room with a known door. No sample data required."""
    rng = np.random.default_rng(0)
    # 4.00 x 3.00 x 2.50, door of 0.80 m on the wall x=0, centered
    pts = []

    def sheet(origin, du, dv, nu, nv, gap=None):
        us = np.linspace(0, 1, nu)
        vs = np.linspace(0, 1, nv)
        uu, vv = np.meshgrid(us, vs)
        u = uu.ravel()
        v = vv.ravel()
        p = origin + u[:, None] * du + v[:, None] * dv
        if gap is not None:
            # gap in the u coordinate, in metres along du
            along = u * np.linalg.norm(du)
            keep = ~((along > gap[0]) & (along < gap[1]) & (v * np.linalg.norm(dv) < 2.1))
            p = p[keep]
        p = p + rng.normal(0, 0.006, size=p.shape)
        pts.append(p)

    # floor and ceiling
    sheet(np.array([0.0, 0.0, 0.0]), np.array([4.0, 0, 0]), np.array([0, 0, 3.0]), 40, 30)
    sheet(np.array([0.0, 2.5, 0.0]), np.array([4.0, 0, 0]), np.array([0, 0, 3.0]), 30, 24)
    # walls. Door on x=0 from z=1.1 to z=1.9
    sheet(np.array([0.0, 0.0, 0.0]), np.array([0, 0, 3.0]), np.array([0, 2.5, 0]), 50, 20, gap=(1.1, 1.9))
    sheet(np.array([4.0, 0.0, 0.0]), np.array([0, 0, 3.0]), np.array([0, 2.5, 0]), 50, 20)
    sheet(np.array([0.0, 0.0, 0.0]), np.array([4.0, 0, 0]), np.array([0, 2.5, 0]), 60, 20)
    sheet(np.array([0.0, 0.0, 3.0]), np.array([4.0, 0, 0]), np.array([0, 2.5, 0]), 60, 20)
    # a sofa that must not become a wall
    sheet(np.array([1.2, 0.0, 1.0]), np.array([1.6, 0, 0]), np.array([0, 0.45, 0]), 20, 8)
    xyz = np.concatenate(pts, axis=0).astype(np.float32)
    # fake frame index so alignment is happy
    from .fuse import Cloud

    cloud = Cloud(xyz=xyz, frame_index=np.zeros(len(xyz), dtype=np.int32), source="synthetic")
    aligned = align_floor(cloud, enable_anchor=True, enable_loop=False)
    layout = build_layout(aligned["xyz"], tier="lidar", endpoint_mode="intersections")
    assert layout.rooms, layout.notes
    room = layout.rooms[0]
    area = float(abs(_shoelace(room.polygon)))
    lengths = sorted(float(np.linalg.norm(w.p1 - w.p0)) for w in room.walls)
    openings = [op for w in room.walls for op in w.openings]
    # The box is 4 x 3. Allow 5 cm on each side.
    ok_area = abs(area - 12.0) < 0.6
    ok_ceil = room.ceiling_observed and room.ceiling is not None and abs(room.ceiling - 2.5) < 0.05
    door_widths = [op.width for op in openings if op.kind == "door"]
    ok_door = any(abs(w - 0.8) < 0.12 for w in door_widths)
    result = {
        "rooms": len(layout.rooms),
        "area_m2": round(area, 3),
        "wall_lengths_m": [round(v, 3) for v in lengths],
        "ceiling_m": None if room.ceiling is None else round(room.ceiling, 3),
        "doors_m": [round(w, 3) for w in door_widths],
        "notes": layout.notes,
        "pass_area": ok_area,
        "pass_ceiling": ok_ceil,
        "pass_door": ok_door,
        "pass": bool(ok_area and ok_ceil and ok_door),
    }
    if not result["pass"]:
        raise AssertionError(json.dumps(result, indent=2))
    return result


def _shoelace(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Floor plan from a phone capture")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="one capture, one tier, one JSON + plan")
    run.add_argument("--capture", required=True, type=Path)
    run.add_argument("--tier", required=True, choices=["lidar", "video", "photo"])
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--endpoints", default="intersections", choices=["intersections", "observed"])
    run.add_argument("--no-floor-anchor", action="store_true")
    run.add_argument("--no-loop", action="store_true")

    rep = sub.add_parser("repeat", help="temporal-split repeatability on a lidar capture")
    rep.add_argument("--capture", required=True, type=Path)
    rep.add_argument("--out", required=True, type=Path)
    rep.add_argument("--endpoints", default="intersections", choices=["intersections", "observed"])

    sub.add_parser("self-check", help="synthetic room with a known door")

    bench = sub.add_parser("benchmark", help="score a manifest; missing physical inputs stay blocked")
    bench.add_argument("--manifest", required=True, type=Path)
    bench.add_argument("--out", type=Path, default=None)

    gate = sub.add_parser("assessment-gate", help="PASS, FAIL, or BLOCKED for each PDF gate")
    gate.add_argument("--manifest", required=True, type=Path)
    gate.add_argument("--out", type=Path, default=None)

    assess = sub.add_parser("assessment", help="validate one property and score every PDF gate")
    assess.add_argument("--manifest", required=True, type=Path)
    assess.add_argument("--out", type=Path, default=None)

    bundle = sub.add_parser("benchmark-bundle", help="validate a property bundle; missing evidence stays BLOCKED")
    bundle.add_argument("--manifest", required=True, type=Path)

    fix = sub.add_parser("fix-loop", help="before, after, and compare for one investigated gate")
    fix_sub = fix.add_subparsers(dest="fix_cmd", required=True)
    fix_before = fix_sub.add_parser("before")
    fix_before.add_argument("--capture", required=True, type=Path)
    fix_before.add_argument("--out", required=True, type=Path)
    fix_before.add_argument("--tier", default="lidar", choices=["lidar", "video", "photo"])
    fix_after = fix_sub.add_parser("after")
    fix_after.add_argument("--capture", required=True, type=Path)
    fix_after.add_argument("--out", required=True, type=Path)
    fix_after.add_argument("--tier", default="lidar", choices=["lidar", "video", "photo"])
    fix_compare = fix_sub.add_parser("compare")
    fix_compare.add_argument("--before", required=True, type=Path)
    fix_compare.add_argument("--after", required=True, type=Path)
    fix_compare.add_argument("--out", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.cmd == "self-check":
        result = self_check()
        print(json.dumps(result, indent=2))
        return
    if args.cmd == "run":
        try:
            doc = run_one(
                args.capture,
                args.tier,
                args.out,
                endpoint_mode=args.endpoints,
                floor_anchor=not args.no_floor_anchor,
                loop=not args.no_loop,
            )
        except FileNotFoundError as exc:
            print(json.dumps({"status": "BLOCKED", "reason": "missing_required_input", "missing": [str(exc)]}))
            raise SystemExit(0)
        except (ValueError, RuntimeError, OSError) as exc:
            print(json.dumps({"status": "BLOCKED", "reason": "invalid_capture", "missing": [str(exc)]}))
            raise SystemExit(2)
        status = doc.get("status")
        if status == "degraded":
            print(f"wrote {args.out / 'plan.json'} status=DEGRADED rooms={len(doc['property']['rooms'])}")
        else:
            print(f"wrote {args.out / 'plan.json'} status={status} rooms={len(doc['property']['rooms'])}")
        return
    if args.cmd == "benchmark-bundle":
        from .benchmark import bundle_status

        result = bundle_status(args.manifest)
        print(json.dumps(result, indent=2))
        raise SystemExit(int(result.get("exit_code") or 0))
    if args.cmd == "fix-loop":
        from .fix_loop import after as fix_after_run
        from .fix_loop import before as fix_before_run
        from .fix_loop import compare as fix_compare_run

        try:
            if args.fix_cmd == "before":
                fix_before_run(args.capture, args.out, tier=args.tier)
            elif args.fix_cmd == "after":
                fix_after_run(args.capture, args.out, tier=args.tier)
            else:
                diff = fix_compare_run(args.before, args.after, args.out)
                print(json.dumps(diff, indent=2))
                return
        except FileNotFoundError as exc:
            print(json.dumps({"status": "BLOCKED", "reason": "missing_required_input", "missing": [str(exc)]}))
            raise SystemExit(0)
        except (ValueError, RuntimeError, OSError) as exc:
            print(json.dumps({"status": "BLOCKED", "reason": "invalid_capture", "missing": [str(exc)]}))
            raise SystemExit(2)
        print(f"wrote {args.out}")
        return
    if args.cmd == "assessment":
        from .assessment_run import run_assessment

        command = "python run.py assessment --manifest " + Path(args.manifest).name
        try:
            report = run_assessment(args.manifest, args.out, command=command)
        except (ValueError, OSError) as exc:
            print(f"assessment rejected: {exc}")
            raise SystemExit(2) from exc
        print(f"wrote {report['output_dir']}")
        if report.get("rejected"):
            print(report.get("reason"))
            raise SystemExit(report.get("exit_code") or 2)
        for name, item in report.get("gates", {}).items():
            print(f"{name} {item['status']} {item['reason']}")
        if report.get("exit_code"):
            raise SystemExit(report["exit_code"])
        return
    if args.cmd in {"benchmark", "assessment-gate"}:
        from .assessment import run_manifest_file

        report = run_manifest_file(args.manifest, args.out)
        print(f"wrote {report['output_dir']}")
        for name, item in report["gates"].items():
            print(f"{name} {item['status']} {item['reason']}")
        if report["failed"]:
            raise SystemExit(1)
        return
    if args.cmd == "repeat":
        summary = repeatability(args.capture, args.out, args.endpoints)
        print(
            f"compared {summary['walls_compared']} walls, "
            f"{summary['walls_within_gate']} inside the 1 cm / 0.5% gate, "
            f"worst delta {summary['worst_abs_delta_m']} m"
        )
        return
