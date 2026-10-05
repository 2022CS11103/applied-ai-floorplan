"""Same opening, several camera positions, edges aggregated before any voxel fuse.

Each extra frame shifts the sample grid by a fraction of the column pitch.
That is a camera moved along the wall, not a phase chosen to land on the
true jamb. A dropped column is missing in the first view only: another
position can still return it. One-hit bins stay off the 3-hit wall test.
They are a separate sparse candidate when neighboring bins continue.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from cozmo_scan.layout import _openings_on_line, _support_runs, aggregate_jambs, openings_from_wall_frames

_SPEC = importlib.util.spec_from_file_location(
    "opening_benchmark",
    Path(__file__).with_name("test_opening_benchmark.py"),
)
_BENCH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BENCH)


def _frame_opening(case, frame_index: int, n_frames: int):
    gap0 = case["gap0"]
    gap1 = gap0 + case["width"]
    pitch = case.get("pitch") or max(case.get("left_pitch", 0.0), case.get("right_pitch", 0.0))
    kwargs = {
        key: case[key]
        for key in (
            "pitch",
            "phase",
            "noise",
            "repeats",
            "seed",
            "left_pitch",
            "right_pitch",
            "outliers",
            "drop_left",
            "drop_right",
        )
        if key in case
    }
    kwargs["phase"] = case.get("phase", 0.0) + frame_index * pitch / n_frames
    if frame_index > 0:
        kwargs["drop_left"] = 0
        kwargs["drop_right"] = 0
        if "seed" in kwargs:
            kwargs["seed"] = int(kwargs["seed"]) + 1000 * frame_index
    along = _BENCH._along(gap0, gap1, **kwargs)
    if len(along) < 10:
        return None
    height = np.full(along.shape, 1.0)
    door = (height > 0.35) & (height < 1.65)
    runs = _support_runs(along[door] if int(door.sum()) >= 10 else along)
    line = {"runs": runs, "along": along, "height": height, "sigma": 0.01, "n": int(len(along))}
    found = [op for op in _openings_on_line(line, 0.0, 4.0, 4.0, 0.012) if op.kind == "door"]
    if len(found) != 1:
        return None
    return found[0]


def _frame_edges(case, frame_index: int, n_frames: int):
    opening = _frame_opening(case, frame_index, n_frames)
    if opening is None:
        return None
    return (float(opening.along0), float(opening.along1), float(opening.sigma), float(opening.sigma))


def _is_unobserved(case, single) -> bool:
    """A detected opening whose single grid left the jamb between returns."""
    return single is not None and not case.get("drop_left") and not case.get("drop_right")


def multiframe_rows(n_frames: int) -> list[dict]:
    rows = []
    for case in _BENCH.benchmark_cases():
        single = _frame_opening(case, 0, 1)
        candidates = []
        methods = []
        for frame in range(n_frames):
            opening = _frame_opening(case, frame, n_frames)
            if opening is None:
                continue
            candidates.append((opening.along0, opening.along1, opening.sigma, opening.sigma))
            methods.append(opening.method)
        aggregated = aggregate_jambs(candidates, methods=methods)
        estimated = None if aggregated is None else aggregated["width"]
        if estimated is None:
            error_cm = None
            passed = False
        else:
            error_cm = abs(estimated - case["width"]) * 100.0
            passed = error_cm <= 2.0
        single_err = None if single is None else abs(single.width - case["width"]) * 100.0
        sparse = case["name"].startswith("sparse_")
        rows.append(
            {
                "name": case["name"],
                "actual": case["width"],
                "estimated": estimated,
                "absolute_error_cm": None if error_cm is None else round(error_cm, 2),
                "pass_2cm": passed,
                "unobserved_interval": (not sparse) and _is_unobserved(case, single) and (single_err is not None and single_err > 2.0),
                "deleted_column": bool(case.get("drop_left") or case.get("drop_right")),
                "sparse": sparse,
                "method": None if aggregated is None else aggregated["method"],
                "confidence": None if aggregated is None else aggregated["confidence"],
                "frames_used": 0 if aggregated is None else aggregated["frames"],
                "localization": None if aggregated is None else aggregated["localization"],
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    errors = [row["absolute_error_cm"] for row in rows if row["absolute_error_cm"] is not None]
    arr = np.array(errors, dtype=float)
    passed = sum(row["pass_2cm"] for row in rows)
    unobserved = [row for row in rows if row["unobserved_interval"]]
    deleted = [row for row in rows if row["deleted_column"]]
    sparse = [row for row in rows if row["sparse"]]
    return {
        "n": len(rows),
        "pass": passed,
        "pass_rate": passed / len(rows),
        "mae_cm": None if len(arr) == 0 else round(float(arr.mean()), 2),
        "median_cm": None if len(arr) == 0 else round(float(np.median(arr)), 2),
        "p95_cm": None if len(arr) == 0 else round(float(np.percentile(arr, 95)), 2),
        "worst_cm": None if len(arr) == 0 else round(float(arr.max()), 2),
        "undetected": sum(row["estimated"] is None for row in rows),
        "unobserved_n": len(unobserved),
        "unobserved_pass": sum(row["pass_2cm"] for row in unobserved),
        "deleted_n": len(deleted),
        "deleted_pass": sum(row["pass_2cm"] for row in deleted),
        "sparse_n": len(sparse),
        "sparse_pass": sum(row["pass_2cm"] for row in sparse),
    }


def test_two_frames_use_the_median_not_a_fused_grid():
    aggregated = aggregate_jambs([(1.10, 1.90), (1.16, 1.96)])
    assert aggregated["left"] == pytest.approx(1.13)
    assert aggregated["right"] == pytest.approx(1.93)
    assert aggregated["width"] == pytest.approx(0.80)
    assert aggregated["method"] == "multi_frame_wall_termination"
    assert aggregated["frames"] == 2


def test_one_frame_keeps_the_low_confidence_fallback_name():
    single = aggregate_jambs([(1.10, 1.90)])
    agreed = aggregate_jambs([(1.10, 1.90), (1.11, 1.91)])
    assert single["method"] == "single_frame_unobserved_interval"
    assert single["confidence"] < agreed["confidence"]


def test_single_frame_still_misses_the_unobserved_intervals():
    """Sparse bins can now be detected. The 19 interval misses stay misses."""
    summary = summarize(multiframe_rows(1))
    assert summary["n"] == 62
    assert summary["unobserved_n"] == 19
    assert summary["unobserved_pass"] == 0
    assert summary["sparse_n"] == 5
    assert summary["sparse_pass"] == 5
    assert summary["pass"] == 39


def test_extra_frames_do_not_voxel_merge_before_the_edge():
    """Five phase shifts of one opening. The width is the overlap of the brackets."""
    case = next(case for case in _BENCH.benchmark_cases() if case["name"] == "phase6cm_0.000")
    edges = [_frame_edges(case, frame, 5) for frame in range(5)]
    assert all(edge is not None for edge in edges)
    aggregated = aggregate_jambs(edges)
    assert aggregated["localization"] == "interval_intersection"
    assert abs(aggregated["width"] - 0.80) < abs((edges[0][1] - edges[0][0]) - 0.80)


def test_three_frames_clear_85_percent():
    """Measured on this phase schedule. Three views leave one 6 cm case out."""
    three = summarize(multiframe_rows(3))
    five = summarize(multiframe_rows(5))
    assert three["pass"] == 61
    assert three["unobserved_pass"] == 18
    assert three["deleted_pass"] == 4
    assert three["sparse_pass"] == 5
    assert five["pass"] == 62
    assert five["undetected"] == 0
    assert five["unobserved_pass"] == 19


def test_far_6cm_0_70_uses_the_bracket_the_midpoint_gate_used_to_drop():
    """0.70 m on a 6 cm grid is 2 cm wide of the midpoint on four phases.

    0.70 = 11 * 0.06 + 0.04, so those four unobserved intervals sit on the
    same side of the true jamb. The fifth phase puts the midpoint at about
    0.664 m. The histogram gap on that view is still 0.70 m, so the bracket
    is real evidence. Overlap of the five brackets is the estimate. It is
    not snapped to 0.70.
    """
    case = next(case for case in _BENCH.benchmark_cases() if case["name"] == "far_6cm_0.70")
    single = _frame_opening(case, 0, 1)
    assert single is not None
    assert abs(single.width - 0.70) > 0.02
    five = next(row for row in multiframe_rows(5) if row["name"] == "far_6cm_0.70")
    assert five["frames_used"] == 5
    assert five["localization"] == "interval_intersection"
    assert five["estimated"] == pytest.approx(0.70, abs=0.02)
    assert five["estimated"] != pytest.approx(0.70, abs=1e-4)


def test_deleted_column_is_recovered_from_later_frames_not_from_fusion():
    case = next(case for case in _BENCH.benchmark_cases() if case["name"] == "missing_left_6cm")
    frames = []
    gap0 = case["gap0"]
    gap1 = gap0 + case["width"]
    pitch = case["pitch"]
    for frame in range(3):
        kwargs = {
            "pitch": pitch,
            "phase": frame * pitch / 3,
            "repeats": case["repeats"],
            "noise": case["noise"],
            "seed": case["seed"] + 1000 * frame,
            "drop_left": case["drop_left"] if frame == 0 else 0,
        }
        frames.append(_BENCH._along(gap0, gap1, **kwargs))
    single = openings_from_wall_frames([frames[0]])
    recovered = openings_from_wall_frames(frames)
    assert single is not None and abs(single["width"] - 0.80) > 0.02
    assert recovered["width"] == pytest.approx(0.80, abs=0.02)
    assert recovered["method"] == "multi_frame_wall_termination"
    assert recovered["frames"] == 3


def test_sparse_frame_profiles_stay_less_confident_than_dense_ones():
    sparse = openings_from_wall_frames(
        [_BENCH._along(1.20, 2.00, pitch=0.05, phase=phase, repeats=1) for phase in (0.0, 0.02)]
    )
    dense = openings_from_wall_frames(
        [_BENCH._along(1.20, 2.00, pitch=0.01, phase=phase) for phase in (0.0, 0.004)]
    )
    assert sparse["method"] == "sparse_wall_termination"
    assert dense["method"] == "multi_frame_wall_termination"
    assert sparse["confidence"] < dense["confidence"]
    assert sparse["confidence"] <= 0.62
