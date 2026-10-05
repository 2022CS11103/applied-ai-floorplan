"""Photo still counts and the 8% bars. The 1.40 m prior cannot pass."""

from cozmo_scan.assessment import score_photo
from cozmo_scan.benchmark_eval import PHOTO_FOOTPRINT_REL, PHOTO_WALL_REL
from cozmo_scan.photo import validate_photo_capture


def _stills(folder, count):
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (folder / f"{index}.jpg").write_bytes(b"jpg")


def test_two_and_eight_stills_are_accepted_and_nine_is_not(tmp_path):
    room = tmp_path / "room_01"
    _stills(room, 2)
    two = validate_photo_capture(tmp_path, expected_rooms=["room_01"], calibration={"scale": "tape"})
    assert not any(item["code"] in {"fewer_than_2", "more_than_8"} for item in two["issues"])
    for path in room.glob("*.jpg"):
        path.unlink()
    _stills(room, 8)
    eight = validate_photo_capture(tmp_path, expected_rooms=["room_01"], calibration={"scale": "tape"})
    assert not any(item["code"] == "more_than_8" for item in eight["issues"])
    (room / "extra.jpg").write_bytes(b"jpg")
    nine = validate_photo_capture(tmp_path, expected_rooms=["room_01"], calibration={"scale": "tape"})
    assert any(item["code"] == "more_than_8" for item in nine["issues"])


def test_one_still_is_rejected(tmp_path):
    _stills(tmp_path / "room_01", 1)
    report = validate_photo_capture(tmp_path, expected_rooms=["room_01"])
    assert any(item["code"] == "fewer_than_2" for item in report["issues"])


def test_chest_height_prior_cannot_pass_the_eight_percent_gate():
    walls = [{"wall_id": "w0", "room_id": "room_01", "ground_truth_m": 4.0, "predicted_m": 4.0}]
    footprint = {"ground_truth_m2": 12.0, "predicted_m2": 12.0}
    report = score_photo(walls, footprint, "tape", calibration={"scale": "chest_height_prior"})
    assert report["status"] == "BLOCKED"
    assert report["threshold"]["wall_relative"] == PHOTO_WALL_REL == 0.08
    assert report["threshold"]["footprint_relative"] == PHOTO_FOOTPRINT_REL == 0.08


def test_tape_calibrated_wall_outside_eight_percent_fails():
    walls = [{"wall_id": "w0", "room_id": "room_01", "ground_truth_m": 4.0, "predicted_m": 4.4}]
    footprint = {"ground_truth_m2": 12.0, "predicted_m2": 12.0}
    report = score_photo(walls, footprint, "tape", calibration={"scale": "tape"})
    assert report["status"] == "FAIL"


def test_tape_calibrated_agreement_passes():
    walls = [{"wall_id": "w0", "room_id": "room_01", "ground_truth_m": 4.0, "predicted_m": 4.0}]
    footprint = {"ground_truth_m2": 12.0, "predicted_m2": 12.0}
    report = score_photo(walls, footprint, "laser", calibration={"scale": "laser"})
    assert report["status"] == "PASS"
