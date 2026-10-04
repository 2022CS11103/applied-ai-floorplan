"""Video tier: metric triangulation, and a degraded plan when it cannot close.

These tests check the JSON. They do not claim a wall-length accuracy.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from jsonschema import Draft202012Validator

from cozmo_scan.pipeline import run_one, video_degraded_reasons
from cozmo_scan.fuse import Cloud

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "single_room" / "c00a170fe1"
SCHEMA = json.loads((ROOT / "schema" / "floorplan.schema.json").read_text(encoding="utf-8"))
ALLOWED = {"low_texture", "weak_overlap", "insufficient_points", "no_room_closure", "missing_pose"}


def _write_clip(folder: Path, n: int = 8, value: int = 40) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "rgb.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    assert writer.isOpened()
    frame = np.full((120, 160, 3), value, dtype=np.uint8)
    for _ in range(n):
        writer.write(frame)
    writer.release()


def _write_poses(folder: Path, rows: list[dict]) -> None:
    fields = ["timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw", "fx", "fy", "cx", "cy"]
    with (folder / "odometry.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pose_row(i: int, qw: str = "1", x: float | None = None) -> dict:
    return {
        "timestamp": str(i * 0.1),
        "frame": f"{i:06d}",
        "x": "0" if x is None else str(x),
        "y": "1.4",
        "z": "0",
        "qx": "0",
        "qy": "0",
        "qz": "0",
        "qw": qw,
        "fx": "120",
        "fy": "120",
        "cx": "80",
        "cy": "60",
    }


def _assert_degraded_plan(plan: dict) -> None:
    errors = sorted(Draft202012Validator(SCHEMA).iter_errors(plan), key=lambda e: list(e.path))
    assert errors == [], "\n".join(f"{list(e.path)}: {e.message}" for e in errors)
    assert plan["status"] == "degraded"
    assert plan["property"]["rooms"] == []
    assert plan["status"] != "ok"
    reasons = plan["degraded_reasons"]
    assert isinstance(reasons, list)
    assert set(reasons) <= ALLOWED


def test_missing_pose_is_degraded(tmp_path):
    _write_clip(tmp_path)
    plan = run_one(tmp_path, "video", tmp_path / "out")
    _assert_degraded_plan(plan)
    assert plan["degraded_reasons"] == ["missing_pose"]
    assert (tmp_path / "out" / "plan.json").is_file()
    assert (tmp_path / "out" / "plan.png").is_file()


def test_invalid_pose_is_degraded(tmp_path):
    _write_clip(tmp_path)
    _write_poses(tmp_path, [_pose_row(i, qw="nan", x=i * 0.25) for i in range(8)])
    plan = run_one(tmp_path, "video", tmp_path / "out")
    _assert_degraded_plan(plan)
    assert plan["quality"]["pose_error"] == "invalid_pose"
    assert "missing_pose" not in plan["degraded_reasons"]
    assert "no_room_closure" not in plan["degraded_reasons"]


def test_blank_video_reports_insufficient_points(tmp_path):
    _write_clip(tmp_path, value=30)
    _write_poses(tmp_path, [_pose_row(i, x=i * 0.25) for i in range(8)])
    plan = run_one(tmp_path, "video", tmp_path / "out")
    _assert_degraded_plan(plan)
    assert "insufficient_points" in plan["degraded_reasons"]
    assert plan["quality"]["points"] < 200


def test_unclosed_room_gets_a_checked_reason():
    cloud = Cloud(
        xyz=np.zeros((30, 3), dtype=np.float32),
        frame_index=np.zeros(30, dtype=np.int32),
        source="video_triangulation",
        meta={"reconstruction_reasons": [], "points": 30},
    )
    reasons = video_degraded_reasons(cloud, ["not enough wall points"], 0)
    assert reasons == ["insufficient_points", "no_room_closure"]
    assert "low_texture" not in reasons


def test_video_sample_writes_a_plan_and_does_not_look_successful_when_empty(tmp_path):
    if not (CAPTURE / "rgb.mp4").is_file() or not (CAPTURE / "odometry.csv").is_file():
        pytest.skip("sample capture single_room/c00a170fe1 is not on disk")
    out = tmp_path / "video"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "run", "--capture", str(CAPTURE), "--tier", "video", "--out", str(out)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    assert (out / "plan.png").is_file()
    errors = sorted(Draft202012Validator(SCHEMA).iter_errors(plan), key=lambda e: list(e.path))
    assert errors == [], "\n".join(f"{list(e.path)}: {e.message}" for e in errors)
    assert plan["tier"] == "video"
    assert plan["source"] == "video_triangulation"
    assert "points" in plan["quality"]
    assert "input_frames" in plan["quality"]
    rooms = plan["property"]["rooms"]
    if rooms:
        assert plan["status"] == "ok"
        assert plan["degraded_reasons"] == []
        for room in rooms:
            assert set(room["floor_area_m2"]) >= {"value", "sigma", "ci95_low", "ci95_high"}
            for wall in room["walls"]:
                measure = wall["length_m"]
                assert set(measure) >= {"value", "sigma", "ci95_low", "ci95_high"}
                assert measure["ci95_low"] <= measure["value"] <= measure["ci95_high"]
    else:
        _assert_degraded_plan(plan)
        assert plan["degraded_reasons"]
        assert "status=ok" not in proc.stdout or "rooms=0" in proc.stdout
