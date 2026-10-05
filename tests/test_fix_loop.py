"""Fix-loop files. One capture is a proxy and stays labelled that way."""

import json

from cozmo_scan.fix_loop import compare


def _plan(length, folder):
    folder.mkdir(parents=True, exist_ok=True)
    plan = {"property": {"rooms": [{"id": "room_01", "walls": [{"id": "w0", "length_m": length}]}]}}
    (folder / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (folder / "benchmark.json").write_text(
        json.dumps({"status": "BLOCKED", "proxy_experiment": True, "reason": "one capture"}),
        encoding="utf-8",
    )


def test_compare_writes_diff_and_report(tmp_path):
    _plan(4.05, tmp_path / "before")
    _plan(4.00, tmp_path / "after")
    diff = compare(tmp_path / "before", tmp_path / "after", tmp_path)
    assert diff["gate"] == "wall_repeatability"
    assert diff["before"] == 4.05
    assert diff["after"] == 4.00
    assert diff["delta"] == -0.05
    assert diff["threshold"] == 0.01
    assert diff["status_before"] == "BLOCKED"
    assert diff["status_after"] == "BLOCKED"
    assert diff["proxy_experiment"] is True
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    for line in (
        "Worst gate",
        "Failing number",
        "Root-cause hypothesis",
        "Evidence",
        "Intended fix",
        "Predicted result",
        "Actual after result",
        "Honest explanation",
    ):
        assert line in text


def test_missing_after_plan_stays_blocked(tmp_path):
    _plan(4.0, tmp_path / "before")
    (tmp_path / "after").mkdir()
    diff = compare(tmp_path / "before", tmp_path / "after", tmp_path)
    assert diff["status_after"] == "BLOCKED"
    assert "after/plan.json" in diff["missing"]
