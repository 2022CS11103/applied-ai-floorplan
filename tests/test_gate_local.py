"""Local checks for every gate scorer.

These fixtures use known numbers. A PASS here means the scorer applied the
PDF threshold. It does not mean a tape, a phone walk, or an incumbent export
was measured. Threshold constants are imported and are not rewritten.
"""

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from scipy.spatial.transform import Rotation

from cozmo_scan.assessment import (
    normalize_incumbent,
    score_ceilings,
    score_damage,
    score_incumbent,
    score_openings,
    score_photo,
    score_repeatability,
    score_video,
)
from cozmo_scan.assessment_run import measurement_contract_issues, run_assessment
from cozmo_scan.benchmark_eval import (
    OPENING_MAX_ERROR_M,
    OPENING_PASS_RATE,
    PHOTO_FOOTPRINT_REL,
    PHOTO_WALL_REL,
    REPEATABILITY_ABS_M,
    REPEATABILITY_REL,
    VIDEO_WALL_REL,
)
from cozmo_scan.damage import scope_items
from cozmo_scan.fuse import Cloud, align_floor
from cozmo_scan.layout import build_layout
from cozmo_scan.photo import validate_photo_capture
from cozmo_scan.pipeline import run_one
from cozmo_scan.reconstruct import reject_distant_points

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "benchmarks" / "schema" / "measurement.schema.json").read_text(encoding="utf-8")
)


def _opening(opening_id: str, truth: float, predicted: float | None):
    gt = {"opening_id": opening_id, "room_id": "r", "wall_id": "w", "type": "door", "width_m": truth}
    pred = []
    if predicted is not None:
        pred.append(
            {
                "opening_id": opening_id,
                "room_id": "r",
                "wall_id": "w",
                "width_m": predicted,
                "confidence": 0.8,
                "method": "wall_termination",
            }
        )
    return gt, pred


def test_opening_scorer_pass_fail_boundary_and_missing():
    truth = 0.80
    # 0.80 + 0.02 is a hair over 0.02 in binary. Step back to the last width the comparison accepts.
    on_bar = math.nextafter(truth + OPENING_MAX_ERROR_M, truth)
    inside, pred_in = _opening("in", truth, on_bar)
    passed = score_openings([inside], pred_in, source="tape")
    assert abs(pred_in[0]["width_m"] - truth) <= OPENING_MAX_ERROR_M
    assert passed["status"] == "PASS"
    assert passed["pass_rate"] == 1.0

    outside, pred_out = _opening("out", truth, truth + OPENING_MAX_ERROR_M + 0.001)
    failed = score_openings([outside], pred_out, source="tape")
    assert failed["status"] == "FAIL"
    assert failed["within_2cm"] == 0

    n = 20
    rows_gt = []
    rows_pred = []
    keep = int(round(OPENING_PASS_RATE * n))
    for index in range(n):
        gt, pred = _opening(f"d{index}", truth, truth + 0.01 if index < keep else None)
        rows_gt.append(gt)
        rows_pred.extend(pred)
    at_rate = score_openings(rows_gt, rows_pred, source="tape")
    assert at_rate["trials"] == n
    assert at_rate["pass_rate"] == pytest.approx(OPENING_PASS_RATE)
    assert at_rate["status"] == "PASS"
    short_gt = rows_gt
    short_pred = [row for row in rows_pred if row["opening_id"] != f"d{keep - 1}"]
    below = score_openings(short_gt, short_pred, source="tape")
    assert below["pass_rate"] < OPENING_PASS_RATE
    assert below["status"] == "FAIL"

    assert score_openings([], [], source="tape")["status"] == "BLOCKED"
    broken = score_openings([{"opening_id": "a", "room_id": "r", "wall_id": "w"}], [{"opening_id": "a", "width_m": 0.8}], source="tape")
    assert broken["status"] == "FAIL"
    assert broken["missed"] == 1


def test_ceiling_scorer_pass_fail_boundary_and_missing():
    assert score_ceilings([{"room_id": "r", "ground_truth_m": 2.5, "predicted_m": 2.5, "repeat_spread_m": 0.0}], source="tape")["status"] == "PASS"
    over = score_ceilings([{"room_id": "r", "ground_truth_m": 2.5, "predicted_m": 2.5 + 0.016, "repeat_spread_m": 0.0}], source="tape")
    assert over["status"] == "FAIL"
    missing_spread = score_ceilings([{"room_id": "r", "ground_truth_m": 2.5, "predicted_m": 2.5}], source="tape")
    assert missing_spread["status"] == "BLOCKED"
    assert missing_spread["classification"] == "repeat_not_measured"
    assert score_ceilings([], source="tape")["status"] == "BLOCKED"
    assert score_ceilings([{"room_id": "r", "ground_truth_m": 2.5, "predicted_m": None, "repeat_spread_m": 0.0}], source="tape")["status"] == "BLOCKED"
    faro = score_ceilings([{"room_id": "r", "ground_truth_m": 2.5, "predicted_m": 2.5, "repeat_spread_m": 0.0}], source="faro")
    assert faro["status"] == "BLOCKED"


def test_repeatability_scorer_pass_fail_boundary_and_unmatched():
    assert score_repeatability([{"wall_id": "a", "length_a_m": 2.0, "length_b_m": 2.0 + REPEATABILITY_ABS_M}], source="tape")["status"] == "PASS"
    short = score_repeatability([{"wall_id": "a", "length_a_m": 2.0, "length_b_m": 2.0 + REPEATABILITY_ABS_M + 0.001}], source="tape")
    assert short["status"] == "FAIL"
    long_wall = 10.0
    relative_ok = long_wall * REPEATABILITY_REL
    assert score_repeatability([{"wall_id": "a", "length_a_m": long_wall, "length_b_m": long_wall + relative_ok}], source="tape")["status"] == "PASS"
    unmatched = score_repeatability(
        [
            {"wall_id": "a", "length_a_m": 3.0, "length_b_m": 3.0},
            {"wall_id": "b", "matched": False},
        ],
        source="tape",
    )
    assert unmatched["status"] == "FAIL"
    assert unmatched["rows"][1]["matched"] is False
    assert score_repeatability([], source="tape")["status"] == "BLOCKED"
    assert score_repeatability([{"wall_id": "a", "length_a_m": 3.0, "length_b_m": 3.0}], source="faro")["status"] == "BLOCKED"


def test_photo_count_scale_and_error_boundaries(tmp_path):
    room = tmp_path / "room_01"
    room.mkdir()
    for index in range(2):
        (room / f"{index}.jpg").write_bytes(b"jpg")
    two = validate_photo_capture(tmp_path, expected_rooms=["room_01"], calibration={"scale": "tape"})
    assert two["ok"] is True
    assert two["issues"] == []
    for index in range(2, 8):
        (room / f"{index}.jpg").write_bytes(b"jpg")
    eight = validate_photo_capture(tmp_path, expected_rooms=["room_01"], calibration={"scale": "laser"})
    assert eight["ok"] is True
    (room / "08.jpg").write_bytes(b"jpg")
    nine = validate_photo_capture(tmp_path, expected_rooms=["room_01"], calibration={"scale": "survey"})
    assert any(item["code"] == "more_than_8" for item in nine["issues"])
    assert nine["ok"] is False

    truth = 4.0
    wall_on_bar = math.nextafter(truth * (1.0 + PHOTO_WALL_REL), truth)
    foot_on_bar = math.nextafter(12.0 * (1.0 + PHOTO_FOOTPRINT_REL), 12.0)
    foot = {"ground_truth_m2": 12.0, "predicted_m2": foot_on_bar}
    at_wall = score_photo(
        [{"wall_id": "w", "ground_truth_m": truth, "predicted_m": wall_on_bar}],
        {"ground_truth_m2": 12.0, "predicted_m2": 12.0},
        source="tape",
        calibration={"scale": "tape"},
    )
    assert at_wall["status"] == "PASS"
    over_wall = score_photo(
        [{"wall_id": "w", "ground_truth_m": truth, "predicted_m": truth * (1.0 + PHOTO_WALL_REL) + 0.01}],
        {"ground_truth_m2": 12.0, "predicted_m2": 12.0},
        source="tape",
        calibration={"scale": "tape"},
    )
    assert over_wall["status"] == "FAIL"
    at_foot = score_photo(
        [{"wall_id": "w", "ground_truth_m": truth, "predicted_m": truth}],
        foot,
        source="tape",
        calibration={"scale": "measured"},
    )
    assert at_foot["status"] == "PASS"
    over_foot = score_photo(
        [{"wall_id": "w", "ground_truth_m": truth, "predicted_m": truth}],
        {"ground_truth_m2": 12.0, "predicted_m2": 12.0 * (1.0 + PHOTO_FOOTPRINT_REL) + 0.05},
        source="tape",
        calibration={"scale": "tape"},
    )
    assert over_foot["status"] == "FAIL"
    assert score_photo(
        [{"wall_id": "w", "ground_truth_m": truth, "predicted_m": truth}],
        {"ground_truth_m2": 12.0, "predicted_m2": 12.0},
        source="tape",
        calibration={"scale": "chest_height_prior"},
    )["status"] == "BLOCKED"
    assert score_photo([], None, source="tape", calibration={"scale": "tape"})["status"] == "BLOCKED"


def test_video_scorer_closed_room_boundary_and_degraded():
    closed = {"status": "ok", "property": {"rooms": [{"room_id": "r"}]}}
    limit = 3.0 * (1.0 + VIDEO_WALL_REL)
    assert score_video(closed, [{"wall_id": "w", "ground_truth_m": 3.0, "predicted_m": limit}], source="tape")["status"] == "PASS"
    assert score_video(closed, [{"wall_id": "w", "ground_truth_m": 3.0, "predicted_m": limit + 0.01}], source="tape")["status"] == "FAIL"
    assert score_video(closed, [], source="tape")["status"] == "BLOCKED"
    assert score_video(None, [{"wall_id": "w", "ground_truth_m": 3.0, "predicted_m": 3.0}], source="tape")["status"] == "BLOCKED"
    degraded = score_video({"status": "degraded", "degraded_reasons": ["no_room_closure"], "property": {"rooms": []}}, [], source="tape")
    assert degraded["status"] == "DEGRADED"
    assert score_video(closed, [{"wall_id": "w", "ground_truth_m": 3.0, "predicted_m": 3.0}], source="faro")["status"] == "BLOCKED"


def _box_cloud() -> np.ndarray:
    rng = np.random.default_rng(0)
    pts = []

    def sheet(origin, du, dv, nu, nv):
        us = np.linspace(0, 1, nu)
        vs = np.linspace(0, 1, nv)
        uu, vv = np.meshgrid(us, vs)
        p = origin + uu.ravel()[:, None] * du + vv.ravel()[:, None] * dv
        pts.append(p + rng.normal(0, 0.006, size=p.shape))

    sheet(np.array([0.0, 0.0, 0.0]), np.array([4.0, 0, 0]), np.array([0, 0, 3.0]), 40, 30)
    sheet(np.array([0.0, 2.5, 0.0]), np.array([4.0, 0, 0]), np.array([0, 0, 3.0]), 30, 24)
    sheet(np.array([0.0, 0.0, 0.0]), np.array([0, 0, 3.0]), np.array([0, 2.5, 0]), 50, 20)
    sheet(np.array([4.0, 0.0, 0.0]), np.array([0, 0, 3.0]), np.array([0, 2.5, 0]), 50, 20)
    sheet(np.array([0.0, 0.0, 0.0]), np.array([4.0, 0, 0]), np.array([0, 2.5, 0]), 60, 20)
    sheet(np.array([0.0, 0.0, 3.0]), np.array([4.0, 0, 0]), np.array([0, 2.5, 0]), 60, 20)
    return np.concatenate(pts, axis=0).astype(np.float32)


def test_video_tier_layout_closes_a_known_box_and_scores_within_3_percent():
    xyz = _box_cloud()
    cloud = Cloud(xyz=xyz, frame_index=np.zeros(len(xyz), dtype=np.int32), source="synthetic")
    aligned = align_floor(cloud, enable_anchor=True, enable_loop=False)
    layout = build_layout(aligned["xyz"], tier="video")
    assert layout.rooms, layout.notes
    lengths = sorted(float(np.linalg.norm(wall.p1 - wall.p0)) for wall in layout.rooms[0].walls)
    assert lengths[0] == pytest.approx(3.0, abs=0.05)
    assert lengths[-1] == pytest.approx(4.0, abs=0.05)
    plan = {"status": "ok", "property": {"rooms": [{"room_id": "box"}]}}
    walls = [
        {"wall_id": "short", "ground_truth_m": 3.0, "predicted_m": lengths[0]},
        {"wall_id": "long", "ground_truth_m": 4.0, "predicted_m": lengths[-1]},
    ]
    scored = score_video(plan, walls, source="tape")
    assert scored["status"] == "PASS"
    assert scored["room_closure"] == "closed"


def _render_room(R, t, width, height, fx, fy, cx, cy):
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    uu, vv = np.meshgrid(np.arange(width), np.arange(height))
    pix = np.stack([uu.ravel(), vv.ravel(), np.ones(width * height)], axis=1)
    dirs = (np.linalg.inv(k) @ pix.T).T @ R.T
    dirs = dirs / np.clip(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-8, None)
    best = np.full(len(dirs), np.inf)
    hit = np.zeros((len(dirs), 3))
    faces = (
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
        (np.array([-1.0, 0.0, 0.0]), np.array([4.0, 0.0, 0.0])),
        (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0])),
        (np.array([0.0, -1.0, 0.0]), np.array([0.0, 2.5, 0.0])),
        (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0])),
        (np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 3.0])),
    )
    for normal, point in faces:
        denom = dirs @ normal
        numer = (point - t) @ normal
        with np.errstate(divide="ignore", invalid="ignore"):
            dist = np.where(np.abs(denom) > 1e-6, numer / denom, np.inf)
        ok = np.isfinite(dist) & (dist > 0.3) & (dist < 8.0) & (dist < best)
        if not ok.any():
            continue
        pts = t + dirs[ok] * dist[ok, None]
        inside = (
            (pts[:, 0] >= -0.02)
            & (pts[:, 0] <= 4.02)
            & (pts[:, 1] >= -0.02)
            & (pts[:, 1] <= 2.52)
            & (pts[:, 2] >= -0.02)
            & (pts[:, 2] <= 3.02)
        )
        idx = np.flatnonzero(ok)[inside]
        best[idx] = dist[ok][inside]
        hit[idx] = pts[inside]
    seen = np.isfinite(best) & (best < np.inf)
    image = np.zeros((height * width, 3), dtype=np.uint8)
    cell = np.floor(hit[seen] / 0.05).astype(np.int64)
    hashed = (cell[:, 0] * 73856093) ^ (cell[:, 1] * 19349663) ^ (cell[:, 2] * 83492791)
    shade = ((hashed >> 8) & np.int64(255)).astype(np.uint8)
    image[seen, 0] = shade
    image[seen, 1] = (shade * 3) & np.uint8(255)
    image[seen, 2] = (shade * 7) & np.uint8(255)
    return image.reshape(height, width, 3)


def test_synthetic_video_walk_tracks_and_does_not_invent_a_closed_room(tmp_path):
    """Frames, poses, and a cloud are real. This walk still does not close a room."""
    width, height = 320, 240
    fx, fy, cx, cy = 240.0, 240.0, 160.0, 120.0
    scale_x, scale_y = 1920 / width, 1440 / height
    views = []
    specs = (
        (0.0, np.array([1.6, 1.4, 1.15])),
        (np.pi / 2, np.array([1.15, 1.4, 1.5])),
        (np.pi, np.array([2.4, 1.4, 1.85])),
        (-np.pi / 2, np.array([2.85, 1.4, 1.5])),
    )
    for yaw, origin in specs:
        look = np.array([np.sin(yaw), 0.0, np.cos(yaw)])
        cam_y = np.array([0.0, 1.0, 0.0])
        cam_x = np.cross(cam_y, look)
        level = np.column_stack([cam_x, cam_y, look])
        step = np.array([np.cos(yaw), 0.0, -np.sin(yaw)]) * 0.22
        for pitch in (-0.35, 0.0, 0.28):
            pitched = Rotation.from_rotvec(cam_x * pitch).as_matrix() @ level
            for k in range(2):
                views.append((pitched, origin + step * k))
    writer = cv2.VideoWriter(str(tmp_path / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (width, height))
    assert writer.isOpened()
    rows = []
    for index, (R, t) in enumerate(views):
        frame = cv2.cvtColor(_render_room(R, t, width, height, fx, fy, cx, cy), cv2.COLOR_RGB2BGR)
        writer.write(frame)
        quat = Rotation.from_matrix(R).as_quat()
        rows.append(
            {
                "timestamp": f"{index * 0.1:.3f}",
                "frame": str(index),
                "x": f"{t[0]:.4f}",
                "y": f"{t[1]:.4f}",
                "z": f"{t[2]:.4f}",
                "qx": f"{quat[0]:.8f}",
                "qy": f"{quat[1]:.8f}",
                "qz": f"{quat[2]:.8f}",
                "qw": f"{quat[3]:.8f}",
                "fx": f"{fx * scale_x:.4f}",
                "fy": f"{fy * scale_y:.4f}",
                "cx": f"{cx * scale_x:.4f}",
                "cy": f"{cy * scale_y:.4f}",
            }
        )
    writer.release()
    with (tmp_path / "odometry.csv").open("w", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    plan = run_one(tmp_path, "video", tmp_path / "out")
    assert plan["quality"]["frames_used"] >= 5
    assert plan["quality"]["points"] > 200
    assert plan["status"] == "degraded"
    assert plan["property"]["rooms"] == []
    assert "no_room_closure" in plan["degraded_reasons"]
    scored = score_video(plan, [{"wall_id": "w", "ground_truth_m": 4.0, "predicted_m": 4.0}], source="tape")
    assert scored["status"] == "DEGRADED"


def test_damage_scorer_both_classes_false_positive_and_clean_room():
    gt = [
        {"class": "stain", "surface_id": "w", "extent_m2": 0.32},
        {"class": "moisture", "surface_id": "w", "extent_m2": 0.18},
    ]
    pred = [
        {"class": "stain", "surface_id": "w", "extent_m2": 0.32},
        {"class": "moisture", "surface_id": "w", "extent_m2": 0.18},
    ]
    assert score_damage(gt, pred, source="tape", synthetic=False)["status"] == "PASS"
    assert score_damage(gt, pred, source="tape", synthetic=True)["status"] == "BLOCKED"
    missed = score_damage(gt, pred[:1], source="tape", synthetic=False)
    assert missed["status"] == "FAIL"
    assert missed["missed"] == ["moisture"]
    extra = score_damage(gt, pred + [{"class": "stain", "surface_id": "other", "extent_m2": 0.05}], source="tape", synthetic=False)
    assert extra["status"] == "FAIL"
    assert extra["false_positives"]
    assert scope_items([{"id": "room_01", "damage": []}]) == []
    assert score_damage([], [], source="tape")["status"] == "BLOCKED"


def test_incumbent_scorer_boundary_tie_and_fail():
    def row(name, ours, theirs):
        return {"dimension": name, "ours_error_m": ours, "incumbent_error_m": theirs}

    seven = [row(f"w{i}", 0.01, 0.02) for i in range(7)] + [row(f"l{i}", 0.05, 0.01) for i in range(3)]
    assert score_incumbent(seven)["status"] == "PASS"
    assert score_incumbent(seven)["beat_or_tie_rate"] == pytest.approx(0.70)
    six = [row(f"w{i}", 0.01, 0.02) for i in range(6)] + [row(f"l{i}", 0.05, 0.01) for i in range(4)]
    assert score_incumbent(six)["status"] == "FAIL"
    tied = [row(f"w{i}", 0.01, 0.02) for i in range(6)] + [row("tie", 0.02, 0.02)] + [row(f"l{i}", 0.05, 0.01) for i in range(3)]
    assert score_incumbent(tied)["status"] == "PASS"
    assert score_incumbent(tied)["tied"] == 1
    assert score_incumbent([])["status"] == "BLOCKED"
    assert score_incumbent([{"dimension": "x"}])["status"] == "BLOCKED"


def _manifest(path: Path, body: dict) -> Path:
    target = path / "manifest.json"
    target.write_text(json.dumps(body), encoding="utf-8")
    return target


def test_same_property_manifest_is_accepted_and_a_mixed_one_is_rejected(tmp_path):
    same = _manifest(
        tmp_path,
        {
            "property_id": "home_a",
            "rooms": ["room_01", "room_02", "room_03"],
            "lidar": {"path": "property/lidar", "property_id": "home_a"},
            "video": {"path": "property/video", "property_id": "home_a"},
            "photos": {"path": "property/photos", "property_id": "home_a"},
            "repeat": {"path": "repeats/second.json", "property_id": "home_a"},
            "ground_truth": {"path": "ground_truth/measurements.json", "property_id": "home_a"},
            "incumbent": {"path": "incumbent/polycam.json", "property_id": "home_a"},
        },
    )
    report = run_assessment(same, tmp_path / "out")
    assert report.get("rejected") is not True
    assert report["exit_code"] == 0
    assert {gate["status"] for gate in report["gates"].values()} == {"BLOCKED"}
    (tmp_path / "mixed").mkdir()
    mixed = _manifest(
        tmp_path / "mixed",
        {
            "property_id": "home_a",
            "rooms": ["room_01"],
            "video": {"path": "property/video", "property_id": "home_b"},
        },
    )
    rejected = run_assessment(mixed, tmp_path / "mixed_out")
    assert rejected["exit_code"] == 2
    assert rejected["rejected"] is True


def test_tape_schema_accepts_a_complete_row_and_rejects_the_rest():
    tape = {
        "property_id": "home_a",
        "capture_id": "tape",
        "source": "tape",
        "measurements": [
            {
                "property_id": "home_a",
                "room_id": "room_01",
                "wall_id": "w0",
                "measurement_type": "wall_length",
                "value_m": 4.0,
                "unit": "m",
                "uncertainty_m": 0.002,
                "source": "tape",
                "capture_id": "tape",
            }
        ],
    }
    Draft202012Validator(SCHEMA).validate(tape)
    invalid = json.loads(json.dumps(tape))
    invalid["source"] = "guess"
    with pytest.raises(ValidationError):
        Draft202012Validator(SCHEMA).validate(invalid)
    wrong_unit = json.loads(json.dumps(tape))
    wrong_unit["measurements"][0]["unit"] = "cm"
    with pytest.raises(ValidationError):
        Draft202012Validator(SCHEMA).validate(wrong_unit)
    incomplete = json.loads(json.dumps(tape))
    del incomplete["measurements"][0]["uncertainty_m"]
    assert any("uncertainty_m" in message for _, message in measurement_contract_issues(incomplete))
    faro = json.loads(json.dumps(tape))
    faro["source"] = "faro"
    faro["measurements"][0]["source"] = "faro"
    faro["measurements"][0]["measurement_type"] = "ceiling_height"
    faro["measurements"][0]["predicted_m"] = 4.0
    faro["measurements"][0]["repeat_spread_m"] = 0.0
    Draft202012Validator(SCHEMA).validate(faro)
    assert any("reference" in message for _, message in measurement_contract_issues(faro))
    assert score_ceilings(
        [{"room_id": "room_01", "ground_truth_m": 4.0, "predicted_m": 4.0, "repeat_spread_m": 0.0}],
        source="faro",
    )["status"] == "BLOCKED"


def test_distant_video_points_are_removed_and_the_room_is_kept():
    wall = np.column_stack([np.linspace(0, 4, 80), np.full(80, 1.4), np.full(80, 3.0)])
    far = np.array([[80.0, 1.4, 3.0], [-40.0, 1.4, 3.0]])
    kept, _, dropped = reject_distant_points(np.vstack([wall, far]).astype(np.float32))
    assert dropped == 2
    assert len(kept) == 80
    assert float(kept[:, 0].max()) < 10


def test_polycam_and_magicplan_exports_need_a_shared_truth():
    polycam = {"vendor": "polycam", "rooms": [{"id": "room_01", "walls": [{"id": "w0", "length_m": 4.10}]}]}
    alone = normalize_incumbent(polycam)
    assert alone[0]["incumbent_m"] == pytest.approx(4.10)
    assert "ours_error_m" not in alone[0]
    assert score_incumbent(alone)["status"] == "BLOCKED"
    magicplan = {
        "vendor": "magicplan",
        "floors": [
            {
                "rooms": [
                    {
                        "room_id": "room_01",
                        "walls": [{"wall_id": "w0", "lengthInMeters": 4.10, "ours_m": 4.02, "ground_truth_m": 4.00}],
                    }
                ]
            }
        ],
    }
    rows = normalize_incumbent(magicplan)
    assert rows[0]["ours_error_m"] == pytest.approx(0.02)
    assert rows[0]["incumbent_error_m"] == pytest.approx(0.10)
    assert score_incumbent(rows)["status"] == "PASS"
    assert score_incumbent(rows)["rows"][0]["winner"] == "ours"
    pixels = {"rooms": [{"id": "room_01", "walls": [{"id": "w0", "pixels": 400}]}]}
    assert normalize_incumbent(pixels) == []


def test_incomplete_ground_truth_exits_as_an_invalid_contract(tmp_path):
    tape = {
        "property_id": "home_a",
        "capture_id": "tape",
        "source": "tape",
        "measurements": [
            {
                "property_id": "home_a",
                "room_id": "room_01",
                "measurement_type": "ceiling_height",
                "value_m": 2.5,
                "predicted_m": 2.5,
                "repeat_spread_m": 0.0,
                "source": "tape",
                "capture_id": "tape",
            }
        ],
    }
    (tmp_path / "tape.json").write_text(json.dumps(tape), encoding="utf-8")
    (tmp_path / "bad").mkdir()
    manifest = _manifest(
        tmp_path / "bad",
        {"property_id": "home_a", "rooms": ["room_01"], "ground_truth": {"path": "../tape.json", "property_id": "home_a"}},
    )
    report = run_assessment(manifest, tmp_path / "out")
    assert report["gates"]["ceiling"]["status"] == "BLOCKED"
    assert report["exit_code"] == 2


def test_faro_ceiling_stays_blocked_with_exit_zero(tmp_path):
    tape = {
        "property_id": "home_a",
        "capture_id": "faro_1",
        "source": "faro",
        "measurements": [
            {
                "property_id": "home_a",
                "room_id": "room_01",
                "measurement_type": "ceiling_height",
                "value_m": 2.5,
                "predicted_m": 2.5,
                "repeat_spread_m": 0.0,
                "unit": "m",
                "uncertainty_m": 0.002,
                "source": "faro",
                "capture_id": "faro_1",
            }
        ],
    }
    (tmp_path / "faro.json").write_text(json.dumps(tape), encoding="utf-8")
    (tmp_path / "ok").mkdir()
    manifest = _manifest(
        tmp_path / "ok",
        {"property_id": "home_a", "rooms": ["room_01"], "ground_truth": {"path": "../faro.json", "property_id": "home_a"}},
    )
    report = run_assessment(manifest, tmp_path / "faro_out")
    assert report["gates"]["ceiling"]["status"] == "BLOCKED"
    assert report["exit_code"] == 0


def test_assessment_exit_codes_keep_degraded_separate_from_fail(tmp_path):
    folder = tmp_path / "video"
    folder.mkdir()
    writer = cv2.VideoWriter(str(folder / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    assert writer.isOpened()
    blank = np.full((120, 160, 3), 30, dtype=np.uint8)
    for _ in range(8):
        writer.write(blank)
    writer.release()
    fields = ["timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw", "fx", "fy", "cx", "cy"]
    with (folder / "odometry.csv").open("w", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=fields)
        csv_writer.writeheader()
        for index in range(8):
            csv_writer.writerow(
                {
                    "timestamp": str(index * 0.1),
                    "frame": f"{index:06d}",
                    "x": str(index * 0.25),
                    "y": "1.4",
                    "z": "0",
                    "qx": "0",
                    "qy": "0",
                    "qz": "0",
                    "qw": "1",
                    "fx": "120",
                    "fy": "120",
                    "cx": "80",
                    "cy": "60",
                }
            )
    manifest = _manifest(
        tmp_path,
        {"property_id": "home_a", "rooms": ["room_01"], "video": {"path": "video", "property_id": "home_a"}},
    )
    report = run_assessment(manifest, tmp_path / "out")
    assert report["gates"]["video_walls"]["status"] == "DEGRADED"
    assert report["exit_code"] == 0
    assert report["failed"] is False
