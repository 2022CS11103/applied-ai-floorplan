"""Lock the LiDAR sample that already produces three rooms.

These numbers are the current pipeline output on the supplied capture.
They are not laser ground truth.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from cozmo_scan.pipeline import self_check

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "single_room" / "c00a170fe1"
SCHEMA = ROOT / "schema" / "floorplan.schema.json"


def test_self_check_still_passes():
    result = self_check()
    assert result["pass"] is True
    assert result["rooms"] == 1


def test_lidar_sample_still_closes_three_rooms(tmp_path):
    if not (CAPTURE / "depth").is_dir() or not (CAPTURE / "odometry.csv").exists():
        pytest.skip("sample capture single_room/c00a170fe1 is not on disk")

    out = tmp_path / "lidar"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run.py"),
            "run",
            "--capture",
            str(CAPTURE),
            "--tier",
            "lidar",
            "--out",
            str(out),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "status=ok" in proc.stdout
    assert "rooms=3" in proc.stdout

    plan_path = out / "plan.json"
    assert plan_path.is_file()
    assert (out / "plan.png").is_file()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(plan), key=lambda e: list(e.path))
    assert errors == [], "\n".join(f"{list(e.path)}: {e.message}" for e in errors)

    assert plan["status"] == "ok"
    assert plan["tier"] == "lidar"
    rooms = plan["property"]["rooms"]
    assert len(rooms) == 3

    areas = sorted(room["floor_area_m2"]["value"] for room in rooms)
    # Current output is about 2.85, 4.05 and 7.10 m^2. Tolerance is slack
    # for voxel noise, not a ground-truth band.
    assert areas == pytest.approx([2.85, 4.05, 7.097], abs=0.15)

    for room in rooms:
        lengths = [wall["length_m"]["value"] for wall in room["walls"]]
        assert any(abs(length - 2.40) <= 0.08 for length in lengths), lengths
        assert room["damage"] == []
        for wall in room["walls"]:
            measure = wall["length_m"]
            assert set(("value", "sigma", "ci95_low", "ci95_high")) <= set(measure)
            assert measure["ci95_low"] <= measure["value"] <= measure["ci95_high"]
    assert plan["scope_line_items"] == []
    assert plan["concealed_damage"]
    assert all(flag["fired"] is False for flag in plan["concealed_damage"])
