"""Video gate. Missing odometry is DEGRADED, and a closed room uses the 3% bar."""

from cozmo_scan.benchmark import score_video_against_ground_truth, validate_video_capture
from cozmo_scan.benchmark_eval import VIDEO_WALL_REL


def test_missing_folder_is_blocked(tmp_path):
    report = validate_video_capture(tmp_path / "absent")
    assert report["status"] == "BLOCKED"
    assert report["reason"] == "missing_video"


def test_mp4_without_odometry_is_degraded(tmp_path):
    (tmp_path / "walk.mp4").write_bytes(b"")
    report = validate_video_capture(tmp_path)
    assert report["status"] == "DEGRADED"
    assert report["reason"] == "missing_pose"


def test_degraded_plan_does_not_invent_a_pass():
    plan = {"status": "degraded", "degraded_reasons": ["missing_pose"], "property": {"rooms": []}}
    report = score_video_against_ground_truth(plan, [], "tape")
    assert report["status"] == "DEGRADED"
    assert report["reason"] == "missing_pose"
    assert report["threshold"] == VIDEO_WALL_REL == 0.03


def test_unclosed_room_is_degraded():
    plan = {"status": "degraded", "degraded_reasons": ["no_room_closure"], "property": {"rooms": []}}
    report = score_video_against_ground_truth(plan, [], "tape")
    assert report["status"] == "DEGRADED"
    assert report["reason"] == "no_room_closure"


def test_closed_room_outside_three_percent_fails():
    plan = {"status": "ok", "degraded_reasons": [], "property": {"rooms": [{"id": "room_01"}]}}
    walls = [{"wall_id": "w0", "ground_truth_m": 4.0, "predicted_m": 4.2}]
    report = score_video_against_ground_truth(plan, walls, "tape")
    assert report["status"] == "FAIL"


def test_closed_room_inside_three_percent_passes():
    plan = {"status": "ok", "degraded_reasons": [], "property": {"rooms": [{"id": "room_01"}]}}
    walls = [{"wall_id": "w0", "ground_truth_m": 4.0, "predicted_m": 4.1}]
    report = score_video_against_ground_truth(plan, walls, "laser")
    assert report["status"] == "PASS"


def test_faro_closed_room_stays_blocked():
    plan = {"status": "ok", "degraded_reasons": [], "property": {"rooms": [{"id": "room_01"}]}}
    walls = [{"wall_id": "w0", "ground_truth_m": 4.0, "predicted_m": 4.0}]
    report = score_video_against_ground_truth(plan, walls, "faro")
    assert report["status"] == "BLOCKED"
