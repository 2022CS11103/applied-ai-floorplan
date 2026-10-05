"""Walk-in command. Missing input is JSON, not a traceback."""

import json
import subprocess
import sys
from pathlib import Path

from cozmo_scan.assessment_run import run_assessment

ROOT = Path(__file__).resolve().parents[1]


def _run(args):
    return subprocess.run(
        [sys.executable, "run.py", *args],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )


def test_missing_capture_is_blocked_json(tmp_path):
    proc = _run(["run", "--capture", str(tmp_path / "missing"), "--tier", "lidar", "--out", str(tmp_path / "out")])
    assert proc.returncode == 0
    assert "Traceback" not in proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "BLOCKED"
    assert payload["reason"] == "missing_required_input"
    assert payload["missing"]


def test_empty_folder_is_malformed_exit_2(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    proc = _run(["run", "--capture", str(empty), "--tier", "lidar", "--out", str(tmp_path / "out")])
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "BLOCKED"
    assert payload["reason"] == "invalid_capture"


def test_example_assessment_writes_the_report_and_stays_blocked(tmp_path):
    report = run_assessment(ROOT / "benchmarks/manifests/assessment.example.json", tmp_path / "assessment")
    assert report["exit_code"] == 0
    assert all(gate["status"] == "BLOCKED" for gate in report["gates"].values())
    out = tmp_path / "assessment"
    for name in ("incumbent.json", "repeatability.json", "fix_loop.json", "final_report.md", "benchmark.json"):
        assert (out / name).is_file()
    text = (out / "final_report.md").read_text(encoding="utf-8")
    assert "| Gate | Result | Value | Threshold | Evidence |" in text
    assert "| PASS |" not in text
    summary = json.loads((out / "benchmark" / "summary.json").read_text(encoding="utf-8"))
    assert summary["blocked"] is True
