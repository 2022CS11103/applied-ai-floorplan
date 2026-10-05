"""Sparse jambs, and the captures that still must not become a door.

A width is accepted when it is within 2 cm. Insufficient evidence is
rejected, or it is left outside 2 cm. These bounds are not loosened
to force a pass, and no width is snapped to 0.80 m.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from cozmo_scan.layout import _openings_on_line, _support_runs

_SPEC = importlib.util.spec_from_file_location(
    "opening_benchmark",
    Path(__file__).with_name("test_opening_benchmark.py"),
)
_BENCH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BENCH)

_MF_SPEC = importlib.util.spec_from_file_location(
    "opening_multiframe",
    Path(__file__).with_name("test_opening_multiframe.py"),
)
_MF = importlib.util.module_from_spec(_MF_SPEC)
_MF_SPEC.loader.exec_module(_MF)


def _opening(gap0, gap1, **kwargs):
    along = _BENCH._along(gap0, gap1, **kwargs)
    height = np.full(along.shape, 1.0)
    door = (height > 0.35) & (height < 1.65)
    runs = _support_runs(along[door] if int(door.sum()) >= 10 else along)
    line = {"runs": runs, "along": along, "height": height, "sigma": 0.01, "n": int(len(along))}
    found = [op for op in _openings_on_line(line, 0.0, 4.0, 4.0, 0.012) if op.kind == "door"]
    return found


def _one(gap0, gap1, **kwargs):
    found = _opening(gap0, gap1, **kwargs)
    assert len(found) == 1, [(op.method, round(op.width, 3)) for op in found]
    return found[0]


def _row(name, n_frames):
    return next(row for row in _MF.multiframe_rows(n_frames) if row["name"] == name)


def test_normal_jamb_is_the_dense_path():
    opening = _one(1.20, 2.00, pitch=0.01)
    assert opening.method == "wall_termination"
    assert opening.confidence > 0.52
    assert opening.width == pytest.approx(0.80, abs=0.02)


def test_4cm_phase_needs_another_view():
    single = _row("mid_4cm_0.70", 1)
    three = _row("mid_4cm_0.70", 3)
    assert single["absolute_error_cm"] > 2.0
    assert three["pass_2cm"]


def test_6cm_phase_needs_another_view():
    single = _row("phase6cm_0.000", 1)
    five = _row("phase6cm_0.000", 5)
    assert single["absolute_error_cm"] > 2.0
    assert five["absolute_error_cm"] < single["absolute_error_cm"]
    assert five["pass_2cm"]


def test_deleted_left_column_is_not_invented_from_one_frame():
    single = _row("missing_left_6cm", 1)
    three = _row("missing_left_6cm", 3)
    assert single["absolute_error_cm"] > 2.0
    assert three["pass_2cm"]


def test_deleted_right_column_is_not_invented_from_one_frame():
    single = _row("missing_right_fine", 1)
    three = _row("missing_right_fine", 3)
    assert single["absolute_error_cm"] > 2.0
    assert three["pass_2cm"]


def test_sparse_one_hit_is_a_low_confidence_jamb():
    opening = _one(1.20, 2.00, pitch=0.05, repeats=1)
    dense = _one(1.20, 2.00, pitch=0.01)
    assert _support_runs(_BENCH._along(1.20, 2.00, pitch=0.05, repeats=1)) == []
    assert opening.method == "sparse_wall_termination"
    assert opening.confidence == pytest.approx(0.52)
    assert opening.confidence < dense.confidence
    assert opening.width == pytest.approx(0.80, abs=0.02)


def test_sparse_two_hits_stay_off_the_strong_threshold():
    along = _BENCH._along(1.20, 2.00, pitch=0.05, repeats=2)
    assert _support_runs(along) == []
    opening = _one(1.20, 2.00, pitch=0.05, repeats=2)
    assert opening.method == "sparse_wall_termination"
    assert opening.confidence == pytest.approx(0.52)
    assert opening.width == pytest.approx(0.80, abs=0.02)


def test_sparse_sensor_noise_keeps_the_width_when_the_wall_continues():
    opening = _one(1.20, 2.00, pitch=0.05, repeats=1, noise=0.004, seed=3)
    assert opening.method == "sparse_wall_termination"
    assert opening.width == pytest.approx(0.80, abs=0.02)


def test_sparse_opening_filled_with_returns_is_rejected():
    wall = _BENCH._along(1.20, 2.00, pitch=0.05, repeats=1)
    filled = np.concatenate([wall, np.arange(1.25, 1.95, 0.05)])
    height = np.full(filled.shape, 1.0)
    line = {"runs": _support_runs(filled), "along": filled, "height": height, "sigma": 0.01, "n": int(len(filled))}
    assert _openings_on_line(line, 0.0, 4.0, 4.0, 0.012) == []


def test_offgrid_jamb_is_not_fixed_by_one_grid():
    single = _row("voxel_offgrid_0.73", 1)
    three = _row("voxel_offgrid_0.73", 3)
    assert single["absolute_error_cm"] > 2.0
    assert three["pass_2cm"]


def test_one_missing_jamb_is_not_an_opening():
    along = np.arange(0.0, 1.20, 0.05)
    height = np.full(along.shape, 1.0)
    line = {"runs": _support_runs(along), "along": along, "height": height, "sigma": 0.01, "n": int(len(along))}
    assert _openings_on_line(line, 0.0, 4.0, 4.0, 0.012) == []


def test_both_jambs_sparse_reports_the_low_confidence_path():
    opening = _one(1.20, 1.90, pitch=0.05, repeats=1)
    assert opening.method == "sparse_wall_termination"
    assert opening.confidence == pytest.approx(0.52)
    assert opening.width == pytest.approx(0.70, abs=0.02)


def test_isolated_offplane_noise_does_not_invent_a_jamb():
    cluster = 1.50 + np.array([-0.01, 0.0, 0.01, 0.02])
    height = np.full(cluster.shape, 1.0)
    line = {
        "runs": [],
        "along": cluster,
        "height": height,
        "sigma": 0.01,
        "n": int(len(cluster)),
        "step_along": cluster,
        "step_depth": np.full(cluster.shape, 0.20),
    }
    assert _openings_on_line(line, 0.0, 4.0, 4.0, 0.012) == []

    opening = _one(1.20, 2.00, pitch=0.01)
    along = _BENCH._along(1.20, 2.00, pitch=0.01)
    height = np.full(along.shape, 1.0)
    line = {
        "runs": _support_runs(along),
        "along": along,
        "height": height,
        "sigma": 0.01,
        "n": int(len(along)),
        "step_along": np.array([1.40]),
        "step_depth": np.array([0.20]),
    }
    found = [op for op in _openings_on_line(line, 0.0, 4.0, 4.0, 0.012) if op.kind == "door"]
    assert len(found) == 1
    assert found[0].width == pytest.approx(opening.width, abs=0.005)
