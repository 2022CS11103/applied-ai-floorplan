"""Opening width from wall returns, not from the 5 cm bin edges.

These are synthetic point walls. They do not validate the assessment's
2 cm gate against a tape.
"""

import numpy as np
import pytest

from cozmo_scan.layout import _openings_on_line, _support_runs


def _wall(gap0, gap1, pitch=0.01, noise=0.0, seed=0, one_sided=False, specks=0):
    along = np.arange(0.0, 4.0 + pitch * 0.5, pitch)
    if one_sided:
        along = along[along < gap0]
    else:
        along = along[(along < gap0) | (along >= gap1)]
    if noise:
        rng = np.random.default_rng(seed)
        along = along + rng.normal(0.0, noise, size=along.shape)
    if specks:
        rng = np.random.default_rng(seed + 1)
        mid = 0.5 * (gap0 + gap1)
        extra = mid + rng.normal(0.0, 0.02, size=specks)
        along = np.concatenate([along, extra])
    height = np.full(along.shape, 1.0)
    return along, height


def _openings(along, height):
    door = (height > 0.35) & (height < 1.65)
    runs = _support_runs(along[door] if int(door.sum()) >= 10 else along)
    line = {"runs": runs, "along": along, "height": height, "sigma": 0.01, "n": int(len(along))}
    return _openings_on_line(line, 0.0, 4.0, 4.0, 0.012)


def _door_width(gap0, gap1, **kwargs):
    found = [op for op in _openings(*_wall(gap0, gap1, **kwargs)) if op.kind == "door"]
    assert len(found) == 1, [(op.kind, round(op.width, 3)) for op in _openings(*_wall(gap0, gap1, **kwargs))]
    return found[0].width


@pytest.mark.parametrize("width", [0.70, 0.80, 0.90, 1.10])
def test_dense_opening_width_is_within_2cm(width):
    gap0 = 1.20
    estimated = _door_width(gap0, gap0 + width, pitch=0.01)
    assert estimated == pytest.approx(width, abs=0.02)


def test_noisy_wall_keeps_the_80cm_opening():
    estimated = _door_width(1.20, 2.00, pitch=0.01, noise=0.004, seed=3)
    assert estimated == pytest.approx(0.80, abs=0.03)


def test_specks_inside_the_opening_do_not_shrink_it():
    estimated = _door_width(1.20, 2.00, pitch=0.01, specks=3, seed=5)
    assert estimated == pytest.approx(0.80, abs=0.03)


def test_one_sided_gap_is_not_an_opening():
    found = _openings(*_wall(2.0, 4.0, one_sided=True))
    assert found == []


def test_short_flank_is_not_an_opening():
    # 0.20 m of wall, then a 0.80 m hole. The left flank is below 0.35 m.
    along = np.concatenate([np.arange(0.0, 0.20, 0.01), np.arange(1.00, 3.00, 0.01)])
    height = np.full(along.shape, 1.0)
    assert _openings(along, height) == []


def test_scan_hole_smaller_than_a_door_is_closed():
    found = _openings(*_wall(1.50, 1.65, pitch=0.01))
    assert found == []


def test_repeated_returns_on_each_column_do_not_ignore_the_pitch():
    # ~6 cm between columns, many heights on each column, same layout as the
    # self-check sheet. Half a column jump should beat the 5 cm bin width.
    pitch = 3.0 / 49
    stations = np.arange(0.0, 3.0 + pitch * 0.5, pitch)
    stations = stations[(stations <= 1.1) | (stations >= 1.9)]
    rng = np.random.default_rng(0)
    along = np.concatenate([s + rng.normal(0.0, 0.006, size=12) for s in stations])
    height = np.full(along.shape, 1.0)
    opening = [op for op in _openings(along, height) if op.kind == "door"][0]
    assert abs(opening.width - 0.80) < abs(0.90 - 0.80)
    assert opening.width == pytest.approx(0.80, abs=pitch)


def _opening_with_reveal(gap0, gap1, pitch=0.06, repeats=8):
    """Coarse wall grid plus a jamb face 12 cm off the wall plane."""
    stations = np.arange(0.0, 4.0 + pitch * 0.5, pitch)
    stations = stations[(stations < gap0) | (stations >= gap1)]
    along = np.repeat(stations, repeats)
    height = np.full(along.shape, 1.0)
    reveal = np.concatenate([np.full(repeats, gap0), np.full(repeats, gap1)])
    runs = _support_runs(along)
    line = {
        "runs": runs,
        "along": along,
        "height": height,
        "sigma": 0.01,
        "n": int(len(along)),
        "step_along": reveal,
        "step_depth": np.full(reveal.shape, 0.12),
    }
    found = [op for op in _openings_on_line(line, 0.0, 4.0, 4.0, 0.012) if op.kind == "door"]
    assert len(found) == 1, [(op.kind, round(op.width, 3)) for op in _openings_on_line(line, 0.0, 4.0, 4.0, 0.012)]
    return found[0]


def test_depth_step_places_the_jamb_on_a_coarse_grid():
    # A 6 cm column midpoint is several centimetres off this phase.
    # The reveal is the second surface, so the width is not half a column.
    opening = _opening_with_reveal(1.20, 2.00, pitch=0.06)
    assert opening.width == pytest.approx(0.80, abs=0.02)


@pytest.mark.parametrize("width", [0.73, 1.05])
def test_depth_step_is_not_a_fixed_door_width(width):
    opening = _opening_with_reveal(1.20, 1.20 + width, pitch=0.06)
    assert opening.width == pytest.approx(width, abs=0.02)


def test_one_off_plane_speck_does_not_move_the_jamb():
    along, height = _wall(1.20, 2.00, pitch=0.01)
    runs = _support_runs(along)
    line = {
        "runs": runs,
        "along": along,
        "height": height,
        "sigma": 0.01,
        "n": int(len(along)),
        "step_along": np.array([1.40]),
        "step_depth": np.array([0.20]),
    }
    opening = [op for op in _openings_on_line(line, 0.0, 4.0, 4.0, 0.012) if op.kind == "door"][0]
    assert opening.width == pytest.approx(0.80, abs=0.02)


def test_both_flanks_are_required_for_the_80cm_door():
    opening = [op for op in _openings(*_wall(1.20, 2.00)) if op.kind == "door"][0]
    assert opening.along0 > 0.35
    assert 4.0 - opening.along1 > 0.35
    assert opening.along1 - opening.along0 == pytest.approx(0.80, abs=0.02)
