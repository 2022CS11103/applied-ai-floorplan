"""Synthetic opening widths at several sample spacings.

The rows measure the column-midpoint estimator. They are not a tape survey,
and a high pass rate here does not clear the assessment's 2 cm gate.
"""

import numpy as np
import pytest

from cozmo_scan.layout import _openings_on_line, _support_runs


def _along(
    gap0,
    gap1,
    pitch=0.01,
    phase=0.0,
    noise=0.0,
    repeats=1,
    seed=0,
    left_pitch=None,
    right_pitch=None,
    outliers=0,
    drop_left=0,
    drop_right=0,
    span=4.0,
):
    """Wall returns on one or two grids, with the opening cut out."""
    if left_pitch is None and right_pitch is None:
        stations = phase + np.arange(0, int(span / pitch) + 3) * pitch
        stations = stations[(stations >= 0.0) & (stations <= span)]
        stations = stations[(stations < gap0) | (stations >= gap1)]
    else:
        lp = pitch if left_pitch is None else left_pitch
        rp = pitch if right_pitch is None else right_pitch
        left = phase + np.arange(0, int(span / lp) + 3) * lp
        left = left[(left >= 0.0) & (left < gap0)]
        k0 = int(np.ceil((gap1 - phase) / rp))
        right = phase + np.arange(k0, k0 + int(span / rp) + 3) * rp
        right = right[(right >= gap1) & (right <= span)]
        stations = np.concatenate([left, right])
    if drop_left or drop_right:
        left = stations[stations < gap0]
        right = stations[stations >= gap1]
        if drop_left:
            left = left[:-drop_left] if len(left) > drop_left else left[:0]
        if drop_right:
            right = right[drop_right:] if len(right) > drop_right else right[:0]
        stations = np.concatenate([left, right])
    rng = np.random.default_rng(seed)
    if repeats > 1:
        chunks = [s + rng.normal(0.0, noise, size=repeats) for s in stations]
        along = np.concatenate(chunks) if chunks else np.zeros(0)
    elif noise:
        along = stations + rng.normal(0.0, noise, size=stations.shape)
    else:
        along = stations.copy()
    if outliers and len(along):
        mid = 0.5 * (gap0 + gap1)
        along = np.concatenate([along, mid + rng.normal(0.0, 0.02, size=outliers)])
    return along


def _estimate(gap0, gap1, **kwargs):
    along = _along(gap0, gap1, **kwargs)
    if len(along) < 10:
        return None
    height = np.full(along.shape, 1.0)
    door = (height > 0.35) & (height < 1.65)
    runs = _support_runs(along[door] if int(door.sum()) >= 10 else along)
    line = {"runs": runs, "along": along, "height": height, "sigma": 0.01, "n": int(len(along))}
    found = [op for op in _openings_on_line(line, 0.0, 4.0, 4.0, 0.012) if op.kind == "door"]
    if len(found) != 1:
        return None
    return float(found[0].width)


def _case(name, width, gap0=1.20, **kwargs):
    row = {"name": name, "width": width, "gap0": gap0}
    row.update(kwargs)
    return row


def benchmark_cases():
    """Widths, spacings, noise, repeats, asymmetry, outliers, and missing columns."""
    cases = []
    widths = (0.70, 0.80, 0.90, 1.00, 1.10)
    for width in widths:
        cases.append(_case(f"dense_1cm_{width:.2f}", width, pitch=0.01))
    # Sample step stands in for range: a wall farther from the camera is coarser.
    for pitch, label in ((0.02, "near_2cm"), (0.04, "mid_4cm"), (0.06, "far_6cm")):
        for width in widths:
            cases.append(
                _case(f"{label}_{width:.2f}", width, pitch=pitch, repeats=8, noise=0.004, seed=1)
            )
    for phase in np.linspace(0.0, 0.05, 8, endpoint=False):
        cases.append(
            _case(
                f"phase6cm_{phase:.3f}",
                0.80,
                pitch=0.06,
                phase=float(phase),
                repeats=8,
                noise=0.006,
                seed=2,
            )
        )
    for phase in (0.0, 0.005, 0.01, 0.015):
        cases.append(_case(f"phase1cm_{phase:.3f}", 0.80, pitch=0.01, phase=float(phase)))
    for width in widths:
        cases.append(_case(f"gauss4mm_{width:.2f}", width, pitch=0.01, noise=0.004, seed=3))
    for width in (0.80, 1.00):
        cases.append(_case(f"gauss8mm_{width:.2f}", width, pitch=0.02, noise=0.008, repeats=6, seed=4))
    # Self-check sheet: 50 samples over 3 m, many returns on each column.
    cases.append(
        _case(
            "self_check_sheet_0.80",
            0.80,
            gap0=1.10,
            pitch=3.0 / 49.0,
            repeats=12,
            noise=0.006,
            seed=0,
        )
    )
    for phase in (0.0, 0.008, 0.016, 0.024):
        cases.append(
            _case(
                f"voxel2p5cm_{phase:.3f}",
                0.80,
                pitch=0.025,
                phase=float(phase),
                repeats=6,
                noise=0.006,
                seed=5,
            )
        )
    # 0.80 m is an exact multiple of a 2.5 cm step, so the two jamb errors
    # cancel. Widths that are not a whole number of steps do not.
    for width in (0.73, 0.86, 0.94):
        cases.append(
            _case(f"voxel_offgrid_{width:.2f}", width, pitch=0.025, repeats=6, noise=0.005, seed=5)
        )
    for width in (0.80, 1.00, 1.10):
        cases.append(
            _case(
                f"asymmetric_{width:.2f}",
                width,
                left_pitch=0.02,
                right_pitch=0.05,
                repeats=8,
                noise=0.004,
                seed=6,
            )
        )
    for width in (0.80, 0.90):
        cases.append(_case(f"outliers_{width:.2f}", width, pitch=0.01, outliers=3, seed=7))
    cases.append(_case("outliers_coarse", 0.80, pitch=0.03, repeats=6, outliers=3, noise=0.004, seed=8))
    for width in (0.80, 1.00):
        cases.append(_case(f"missing_left_{width:.2f}", width, pitch=0.02, repeats=6, drop_left=1, seed=9))
    cases.append(_case("missing_left_6cm", 0.80, pitch=0.06, repeats=8, drop_left=1, noise=0.004, seed=10))
    cases.append(_case("missing_right_fine", 0.90, pitch=0.015, repeats=4, drop_right=2, seed=11))
    for width in widths:
        cases.append(_case(f"sparse_{width:.2f}", width, pitch=0.05, repeats=1))
    return cases


def benchmark_rows():
    rows = []
    for case in benchmark_cases():
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
        estimated = _estimate(case["gap0"], case["gap0"] + case["width"], **kwargs)
        if estimated is None:
            error_cm = None
            passed = False
        else:
            error_cm = abs(estimated - case["width"]) * 100.0
            passed = error_cm <= 2.0
        rows.append(
            {
                "name": case["name"],
                "actual": case["width"],
                "estimated": estimated,
                "absolute_error_cm": None if error_cm is None else round(error_cm, 2),
                "pass_2cm": passed,
                "pitch": case.get("pitch") or max(case.get("left_pitch", 0.0), case.get("right_pitch", 0.0)),
            }
        )
    return rows


def _rows_named(prefix):
    return [row for row in benchmark_rows() if row["name"].startswith(prefix)]


def test_benchmark_covers_the_requested_widths_and_conditions():
    names = [case["name"] for case in benchmark_cases()]
    blob = " ".join(names)
    for width in (0.70, 0.80, 0.90, 1.00, 1.10):
        assert f"{width:.2f}" in blob
    for token in ("dense", "near", "mid", "far", "gauss", "phase", "voxel", "asymmetric", "outliers", "missing", "sparse", "self_check"):
        assert any(token in name for name in names)


def test_benchmark_rows_report_error_and_the_2cm_flag():
    rows = benchmark_rows()
    assert len(rows) >= 40
    for row in rows:
        assert set(row) >= {"name", "actual", "estimated", "absolute_error_cm", "pass_2cm"}
        assert row["pass_2cm"] in (True, False)
        if row["estimated"] is not None:
            assert row["absolute_error_cm"] == pytest.approx(abs(row["estimated"] - row["actual"]) * 100.0, abs=0.02)


def test_one_centimetre_grid_stays_within_2cm():
    rows = _rows_named("dense_1cm_") + _rows_named("phase1cm_") + _rows_named("gauss4mm_")
    assert rows
    assert all(row["pass_2cm"] for row in rows)


def test_six_centimetre_sampling_does_not_clear_the_2cm_gate():
    """A column step of 6 cm leaves the jamb inside an unobserved interval.

    The midpoint is the estimate. Across phases, that cannot put 85% of
    openings inside 2 cm. This locks the finding so a constant tweak aimed
    at one 0.80 m sheet does not count as a gate pass.
    """
    rows = _rows_named("phase6cm_")
    assert len(rows) >= 8
    assert all(row["estimated"] is not None for row in rows)
    rate = sum(row["pass_2cm"] for row in rows) / len(rows)
    assert rate < 0.85
    assert max(row["absolute_error_cm"] for row in rows) > 2.0


def test_self_check_sheet_error_is_the_column_step_not_a_tuned_width():
    row = _rows_named("self_check_sheet_")[0]
    assert row["estimated"] is not None
    # The 5 cm bins reported 0.90 m (10 cm off). The midpoint should beat that
    # and still miss 2 cm: this phase leaves almost a full column unobserved.
    assert abs(row["estimated"] - 0.80) < 0.10
    assert row["absolute_error_cm"] > 2.0
    assert row["absolute_error_cm"] < 8.0


def test_isolated_outliers_do_not_set_the_jamb():
    rows = _rows_named("outliers_")
    assert rows
    for row in rows:
        assert row["estimated"] is not None
        assert row["absolute_error_cm"] < 4.0


def test_voxel_sized_spacing_still_misses_2cm_on_some_widths():
    """Fused LiDAR keeps one return per 2.5 cm voxel.

    A width that is not a whole number of those steps leaves a residual of
    about one step. That is already past 2 cm, so this sampling does not
    clear the 85% gate by itself.
    """
    rows = _rows_named("voxel_offgrid_")
    assert len(rows) >= 3
    assert all(row["estimated"] is not None for row in rows)
    assert any(not row["pass_2cm"] for row in rows)
    assert max(row["absolute_error_cm"] for row in rows) > 2.0


def test_a_missing_column_is_not_invented():
    row = _rows_named("missing_left_6cm")[0]
    assert row["estimated"] is not None
    # Dropping the last 6 cm column moves the midpoint by about that step.
    assert row["absolute_error_cm"] > 2.0
