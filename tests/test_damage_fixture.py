"""Synthetic damage only. It must not satisfy the physical damage gate."""

from cozmo_scan.assessment import score_damage
from cozmo_scan.damage import concealed_flags, concealed_summary, scope_items
from cozmo_scan.damage_fixture import assess_scene, load_ground_truth, load_scene

FIXTURE = "fixtures/synthetic_damage"


def test_fixture_is_labelled_and_finds_both_classes():
    scene = load_scene(FIXTURE)
    truth = load_ground_truth(FIXTURE)
    result = assess_scene(scene)
    assert result["source"] == "synthetic_test_fixture"
    assert truth["source"] == "synthetic_test_fixture"
    found = result["rooms"][0]["damage"]
    assert {region["class"] for region in found} == {"stain", "moisture"}
    gate = score_damage(truth["regions"], found, source="synthetic_test_fixture", synthetic=True)
    assert gate["status"] == "BLOCKED"


def test_miss_false_positive_and_clean_room():
    truth = [{"class": "stain", "surface_id": "room_01_w0", "extent_m2": 0.2}]
    missed = score_damage(truth, [], source="tape")
    assert missed["status"] == "FAIL"
    assert missed["missed"] == ["stain"]
    false = score_damage(
        truth,
        [
            {"class": "stain", "surface_id": "room_01_w0", "extent_m2": 0.2},
            {"class": "moisture", "surface_id": "room_01_w0", "extent_m2": 0.1},
        ],
        source="tape",
    )
    assert false["status"] == "FAIL"
    assert false["false_positives"] == ["moisture"]
    clean = {"id": "room_01", "walls": [{"id": "room_01_w0", "openings": []}], "damage": []}
    assert scope_items([clean]) == []
    flags = concealed_flags([clean])
    assert flags
    assert all(flag["flag"] is False for flag in flags)
    assert concealed_summary(flags) == {"flag": False, "rule": None}


def test_fired_flag_names_the_rule_and_the_scope_item_is_keyed_to_the_surface():
    region = {
        "id": "room_01_w0_d0",
        "damage_id": "room_01_w0_d0",
        "room_id": "room_01",
        "surface_id": "room_01_w0",
        "class": "stain",
        "extent_m2": {"value": 0.2, "sigma": 0.04, "ci95_low": 0.12, "ci95_high": 0.28},
        "height_band_m": [0.0, 0.3],
    }
    room = {"id": "room_01", "walls": [{"id": "room_01_w0", "openings": []}], "damage": [region]}
    moisture = {
        **region,
        "id": "room_01_w0_d1",
        "class": "moisture",
        "height_band_m": [0.0, 0.2],
    }
    room["damage"].append(moisture)
    flags = [flag for flag in concealed_flags([room]) if flag["flag"]]
    assert flags
    assert {"rule", "surface_id", "evidence", "confidence"} <= set(flags[0])
    items = scope_items([room])
    assert items
    assert items[0]["scope_id"].startswith("scope_")
    assert items[0]["surface_id"] == "room_01_w0"
    assert items[0]["metric_extent_m2"] == 0.2
