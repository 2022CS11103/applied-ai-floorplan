"""Adapter math for ARKitScenes. No download required."""

import json
from pathlib import Path

import numpy as np

from datasets.arkitscenes_adapter import camera_to_world, scale_intrinsics_to_rgb_slot

ROOT = Path(__file__).resolve().parents[1]


def test_traj_inverse_puts_the_camera_center_in_y_up():
    # World-to-camera is identity with translation (1, 2, 3) in Z-up.
    # Camera center in Z-up is (-1, -2, -3). Y-up swaps Y and Z with a sign
    # so the center becomes (-1, -3, 2).
    timestamp, rotation, center = camera_to_world("12.3456 0 0 0 1 2 3")
    assert timestamp == "12.346"
    np.testing.assert_allclose(rotation, np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]), atol=1e-8)
    np.testing.assert_allclose(center, [-1.0, -3.0, 2.0], atol=1e-8)


def test_recorded_faro_comparison_does_not_unlock_the_tape_gate():
    path = ROOT / "benchmarks" / "external" / "arkitscenes_42444949" / "comparison.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["assessment_gate"]["status"] == "blocked"
    assert report["openings"]["field_status"] == "UNAVAILABLE"
    assert report["wall_match"]["field_status"] == "DERIVED_FROM_DATASET"
    assert report["depth_agreement"]["field_status"] == "PROVIDED_BY_DATASET"
    assert "laser_tape" not in json.dumps(report)


def test_pincam_survives_the_pipeline_rgb_slot():
    fx, fy, cx, cy = scale_intrinsics_to_rgb_slot(212.148, 212.148, 128.725, 95.7578, (256, 192))
    assert abs(fx * (256 / 1920) - 212.148) < 1e-6
    assert abs(fy * (192 / 1440) - 212.148) < 1e-6
    assert abs(cx * (256 / 1920) - 128.725) < 1e-6
    assert abs(cy * (192 / 1440) - 95.7578) < 1e-6
