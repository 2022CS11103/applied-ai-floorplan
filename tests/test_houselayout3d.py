"""HouseLayout3D is an external CAD check. It is not the company tape benchmark."""

import json
from pathlib import Path

from cozmo_scan.benchmark_eval import _assessment_gate, load_ground_truth

ROOT = Path(__file__).resolve().parents[1]
GT = ROOT / "benchmarks" / "external" / "houselayout3d_2t7WUuJeko7" / "ground_truth.json"


def test_external_fixture_stays_blocked_and_labelled():
    document = json.loads(GT.read_text(encoding="utf-8"))
    assert document["label"].startswith("EXTERNAL DATASET VALIDATION")
    assert document.get("provenance") != "laser_tape"
    assert document["external_dataset"]["replaces_company_physical_benchmark"] is False
    assert document["calibration"]["scale"] == "unavailable"
    assert document["field_status"]["laser_tape"] == "UNAVAILABLE"
    assert document["field_status"]["room_id"] == "DERIVED_FROM_DATASET"
    assert all(item["used_as_room"] is False for item in document["excluded_horizontal_polygons"])
    loaded = load_ground_truth(document)
    assert _assessment_gate(loaded)["status"] == "blocked"
    assert len(loaded["rooms"]) >= 3
    for room in loaded["rooms"]:
        assert room["floor_area_m2"] >= 1.5
        if room["ceiling_height_m"] is not None:
            assert 2.0 <= room["ceiling_height_m"] <= 5.0
        for opening in room["openings"]:
            assert opening["width_m"] >= 0.4
            assert opening["wall_id"]
    opening_ids = [opening["opening_id"] for room in loaded["rooms"] for opening in room["openings"]]
    assert len(opening_ids) == len(set(opening_ids))
