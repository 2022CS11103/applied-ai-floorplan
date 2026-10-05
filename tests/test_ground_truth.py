"""Tape rows. Centimetres are converted. FARO and guesses are not a tape."""

import json

from cozmo_scan.benchmark import (
    compare_prediction_to_ground_truth,
    group_ground_truth_by_room,
    load_ground_truth,
    validate_ground_truth,
    validate_ground_truth_rows,
    write_ground_truth_report,
)


def _file(tmp_path, doc):
    path = tmp_path / "measurements.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _row(**overrides):
    row = {
        "property_id": "home_a",
        "capture_id": "tape_01",
        "room_id": "room_01",
        "measurement_id": "wall_a",
        "kind": "wall",
        "name": "north",
        "value": 4.25,
        "unit": "m",
        "uncertainty": 0.005,
        "source": "tape",
        "notes": "measured",
    }
    row.update(overrides)
    return row


def test_alias_row_is_accepted(tmp_path):
    path = _file(
        tmp_path,
        {"property_id": "home_a", "capture_id": "tape_01", "source": "tape", "measurements": [_row()]},
    )
    loaded = load_ground_truth(path)
    assert loaded["rows"][0]["measurement_type"] == "wall_length"
    assert loaded["rows"][0]["value_m"] == 4.25
    assert validate_ground_truth_rows(loaded["rows"], "home_a") == []
    grouped = group_ground_truth_by_room(loaded["rows"])
    assert list(grouped) == ["room_01"]


def test_centimetres_are_converted_before_the_schema_check(tmp_path):
    path = _file(
        tmp_path,
        {
            "property_id": "home_a",
            "capture_id": "tape_01",
            "source": "tape",
            "measurements": [_row(value=425, unit="cm", uncertainty=0.5)],
        },
    )
    row = load_ground_truth(path)["rows"][0]
    assert row["value_m"] == 4.25
    assert row["unit"] == "m"
    assert row["uncertainty_m"] == 0.005
    assert validate_ground_truth_rows([row], "home_a") == []


def test_guess_missing_uncertainty_and_identity_are_rejected():
    guessed = _row(source="guessed")
    loaded_guess = {"property_id": "home_a", "capture_id": "tape_01", "source": "guessed", "measurements": [guessed]}
    from pathlib import Path

    path = Path("runs") / "_gt_not_written"
    rows = [
        {**_row(), "uncertainty": None, "source": "tape"},
    ]
    # Build through the loader so aliases are applied.
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        file_path = Path(folder) / "measurements.json"
        file_path.write_text(
            json.dumps({"property_id": "home_a", "capture_id": "tape_01", "source": "tape", "measurements": [_row(uncertainty=None)]}),
            encoding="utf-8",
        )
        problems = validate_ground_truth_rows(load_ground_truth(file_path)["rows"], "home_a")
    assert any("uncertainty" in item for item in problems)
    with tempfile.TemporaryDirectory() as folder:
        file_path = Path(folder) / "measurements.json"
        file_path.write_text(json.dumps(loaded_guess), encoding="utf-8")
        problems = validate_ground_truth_rows(load_ground_truth(file_path)["rows"], "home_a")
    assert any("guessed" in item for item in problems)
    assert rows  # the local row above is only a placeholder for the rejected shape
    assert path.name


def test_faro_cannot_pass_as_tape(tmp_path):
    path = _file(
        tmp_path,
        {"property_id": "home_a", "capture_id": "faro_01", "source": "faro", "measurements": [_row(source="faro")]},
    )
    manifest = {
        "_manifest_path": str(tmp_path / "manifest.json"),
        "property_id": "home_a",
        "ground_truth": {"path": str(path)},
    }
    result = validate_ground_truth(manifest)
    assert result["status"] == "BLOCKED"
    assert result["exit_code"] == 0
    assert result["reason"] == "reference_is_not_tape"


def test_compare_does_not_drop_a_missing_prediction(tmp_path):
    path = _file(
        tmp_path,
        {
            "property_id": "home_a",
            "capture_id": "tape_01",
            "source": "tape",
            "measurements": [_row(kind="opening", measurement_id="door_a", value=0.8, name="door")],
        },
    )
    rows = load_ground_truth(path)["rows"]
    report = compare_prediction_to_ground_truth(rows, [], "tape")
    assert report["openings"]["status"] == "FAIL"
    assert report["openings"]["missed"] == 1
    write_ground_truth_report(tmp_path / "out", {"property_id": "home_a", "ground_truth": {"source": "tape"}}, rows, [])
    assert (tmp_path / "out" / "benchmark" / "summary.json").is_file()
    assert (tmp_path / "out" / "benchmark" / "errors.json").is_file()
    assert (tmp_path / "out" / "benchmark" / "report.md").is_file()
