"""Walls, rooms, openings, ceiling.

The plan is built in the floor plane after alignment:

1. Keep points that sit in a column tall enough to be a wall. A table is
   a thin slab; a wall shows up in more than one height band. This is
   what keeps furniture out of the plan.
2. Rotate that 2D cloud onto a Manhattan frame. Homes in this brief are
   orthogonal; the dominant gradient direction of the wall mask is the
   rotation, and it is recorded so a non-orthogonal scan can be spotted.
3. Wall positions are peaks in the X and Y histograms. Two peaks closer
   than about 0.3 m are the two noisy faces of one surface and are merged.
4. Those lines form a grid. A grid cell is a room when it is wide enough
   to stand in and its interior is empty. A cell that is full of points
   is a wall, not a closet.
5. Wall length is the distance between the perpendicular walls that close
   the room (the tape measurement), not the length of the points that
   happened to be observed. The observed-only length is kept as the
   "before" model in the fix loop: a second pass that saw a different
   portion of the same wall disagrees, even when the room did not change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import meas, shoelace


TIER_FLOOR = {
    # absolute metre floor on a length, and a relative floor
    "lidar": (0.012, 0.0),
    "video": (0.04, 0.03),
    "photo": (0.08, 0.08),
}


@dataclass
class Opening:
    kind: str
    width: float
    sigma: float
    along0: float
    along1: float
    sill: float | None
    head: float | None


@dataclass
class WallSeg:
    p0: np.ndarray
    p1: np.ndarray
    sigma: float
    support: int
    openings: list[Opening] = field(default_factory=list)


@dataclass
class Room:
    polygon: np.ndarray
    walls: list[WallSeg]
    ceiling: float | None
    ceiling_sigma: float | None
    ceiling_observed: bool


@dataclass
class Layout:
    rooms: list[Room]
    theta: float
    endpoint_mode: str
    notes: list[str]
    wall_sigma: float


def _rotate2(xy: np.ndarray, theta: float) -> np.ndarray:
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.array([[c, -s], [s, c]])
    return xy @ R.T


def _unrotate2(xy: np.ndarray, theta: float) -> np.ndarray:
    # inverse of _rotate2
    c, s = np.cos(theta), np.sin(theta)
    # _rotate2 uses rot(-theta), so inverse is rot(+theta) on row vectors? 
    # p' = R(-th) @ p, so p = R(th) @ p'. Row vectors: p = p' @ R(-th).T = p' @ R(th)
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.array([[c, -s], [s, c]])
    return xy @ R


def vertical_columns(xyz: np.ndarray, cell: float = 0.05, min_span: float = 0.65) -> np.ndarray:
    """Boolean mask of points whose floor-column spans enough height to be a wall."""
    if len(xyz) == 0:
        return np.zeros(0, dtype=bool)
    xz = xyz[:, [0, 2]]
    h = xyz[:, 1]
    key = np.floor(xz / cell).astype(np.int64)
    key -= key.min(axis=0)
    span = int(key[:, 0].max()) + 2
    code = key[:, 0] + key[:, 1] * span
    order = np.argsort(code, kind="mergesort")
    code_s = code[order]
    h_s = h[order]
    cuts = np.flatnonzero(np.diff(code_s)) + 1
    starts = np.r_[0, cuts]
    hmin = np.minimum.reduceat(h_s, starts)
    hmax = np.maximum.reduceat(h_s, starts)
    # Floor plus ceiling in the same column is not a wall. A wall has returns
    # in the band where a person would touch it.
    mid = ((h_s > 0.45) & (h_s < 1.6)).astype(np.int32)
    mid_count = np.add.reduceat(mid, starts)
    good = (hmax - hmin) >= min_span
    good &= hmax >= 1.15
    good &= mid_count >= 3
    mask_s = np.repeat(good, np.diff(np.r_[starts, len(code_s)]))
    mask = np.empty(len(xyz), dtype=bool)
    mask[order] = mask_s
    return mask


def _manhattan_theta(xy: np.ndarray) -> float:
    """Dominant wall direction from local line fits, folded into a right angle.

    A silhouette gradient locks onto the bounding box, which is axis-aligned
    even when the room is rotated in the capture frame. Local PCA on the
    occupancy grid follows the wall itself. Isotropic clutter (a plant, a
    pile of returns) has no long axis and is ignored.
    """
    if len(xy) < 50:
        return 0.0
    res = 0.05
    ij = np.floor(xy / res).astype(np.int32)
    cells = np.unique(ij, axis=0)
    if len(cells) < 30:
        return 0.0
    occupied = set(map(tuple, cells.tolist()))
    if len(cells) > 5000:
        step = max(1, len(cells) // 5000)
        sample = cells[::step]
    else:
        sample = cells
    angles = []
    weights = []
    for c0, c1 in sample:
        nbr = []
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                if (int(c0) + dx, int(c1) + dy) in occupied:
                    nbr.append((dx, dy))
        if len(nbr) < 8:
            continue
        arr = np.asarray(nbr, dtype=np.float64)
        cov = np.cov(arr.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        if eigvals[0] <= 1e-6:
            continue
        ratio = float(eigvals[1] / eigvals[0])
        if ratio < 4.0:
            continue
        tangent = eigvecs[:, 1]
        ang = float(np.arctan2(tangent[1], tangent[0]))
        angles.append(np.mod(ang, np.pi / 2.0))
        weights.append(ratio)
    if len(angles) < 25:
        return 0.0
    hist, edges = np.histogram(angles, bins=90, range=(0.0, np.pi / 2.0), weights=weights)
    i = int(hist.argmax())
    return float(0.5 * (edges[i] + edges[i + 1]))


def _peaks_1d(values: np.ndarray, bin_m: float, min_sep: float, min_frac: float) -> list[dict]:
    if len(values) < 30:
        return []
    lo, hi = np.percentile(values, [0.5, 99.5])
    if hi - lo < 0.4:
        return []
    bins = np.arange(lo, hi + bin_m, bin_m)
    hist, edges = np.histogram(values, bins=bins)
    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float64)
    kernel /= kernel.sum()
    smooth = np.convolve(hist.astype(np.float64), kernel, mode="same")
    peaks = []
    thresh = float(smooth.max()) * min_frac
    # End bins are included. The outer walls of a room sit on the edge of the
    # point cloud, and skipping them drops the only walls that close the plan.
    for i in range(len(smooth)):
        left = smooth[i - 1] if i > 0 else -1.0
        right = smooth[i + 1] if i < len(smooth) - 1 else -1.0
        if smooth[i] >= left and smooth[i] >= right and smooth[i] >= thresh and hist[i] >= 15:
            if 0 < i < len(smooth) - 1:
                y0, y1, y2 = smooth[i - 1], smooth[i], smooth[i + 1]
                denom = y0 - 2 * y1 + y2
                shift = 0.0 if abs(denom) < 1e-6 else 0.5 * (y0 - y2) / denom
                shift = float(np.clip(shift, -0.5, 0.5))
            else:
                shift = 0.0
            center = 0.5 * (edges[i] + edges[i + 1]) + shift * bin_m
            peaks.append({"pos": float(center), "score": float(smooth[i]), "count": int(hist[i])})
    peaks.sort(key=lambda p: p["pos"])
    merged: list[dict] = []
    for p in peaks:
        if not merged or abs(p["pos"] - merged[-1]["pos"]) > min_sep:
            merged.append(p)
        elif p["score"] > merged[-1]["score"]:
            merged[-1] = p
    return merged


def _edge_supported(line: dict, a0: float, a1: float, frac: float = 0.30) -> bool:
    span = a1 - a0
    if span <= 0.2:
        return False
    overlap = 0.0
    for a, b, _ in line["runs"]:
        overlap += max(0.0, min(b, a1) - max(a, a0))
    return overlap >= frac * span


def _drop_weak_partitions(lines: list[dict]) -> list[dict]:
    """Keep the outer walls and any interior wall that is almost as solid.

    A sofa or a counter produces a parallel ridge. It is weaker than the
    wall behind it. Dropping it is what stops one room being drawn as three.
    """
    if len(lines) <= 2:
        return lines
    strongest = max(L["score"] for L in lines)
    kept = [L for L in lines if L["score"] >= 0.55 * strongest and L["support_len"] >= 1.2]
    if len(kept) < 2:
        # fall back to the two strongest, which are the outer pair in a single room
        kept = sorted(lines, key=lambda L: L["score"], reverse=True)[:2]
        kept = sorted(kept, key=lambda L: L["pos"])
    return kept


def _band_faces(coord: np.ndarray, pos: float, tol: float = 0.12) -> dict | None:
    """Inner and outer face of one histogram peak, plus along-track values."""
    band = np.abs(coord - pos) <= tol
    # caller passes parallel arrays; this function only sees the coordinate
    return None


def _support_runs(along: np.ndarray, bin_m: float = 0.05, min_run: float = 0.45) -> list[tuple[float, float, int]]:
    if len(along) < 10:
        return []
    lo, hi = float(along.min()), float(along.max())
    if hi - lo < min_run:
        return []
    bins = np.arange(lo, hi + bin_m, bin_m)
    hist, edges = np.histogram(along, bins=bins)
    # A handful of returns marks the surface. Holes shorter than a door are
    # closed so a sparse pass does not shatter one wall into clutter.
    occupied = hist >= 3
    occ = occupied.copy()
    gap_bins = max(1, int(round(0.25 / bin_m)))
    i = 0
    while i < len(occ):
        if occ[i]:
            i += 1
            continue
        j = i
        while j < len(occ) and not occ[j]:
            j += 1
        if i > 0 and j < len(occ) and (j - i) <= gap_bins:
            occ[i:j] = True
        i = j
    runs = []
    i = 0
    while i < len(occ):
        if not occ[i]:
            i += 1
            continue
        j = i
        while j < len(occ) and occ[j]:
            j += 1
        a = float(edges[i])
        b = float(edges[j])
        if b - a >= min_run:
            runs.append((a, b, int(hist[i:j].sum())))
        i = j
    return runs


def _height_profile(heights: np.ndarray, along: np.ndarray, a: float, b: float, bin_m: float = 0.05):
    """For a gap [a,b], how the vertical coverage looks. Used to tell a door from a window."""
    sel = (along >= a) & (along <= b)
    h = heights[sel]
    if len(h) < 5:
        return {"n": int(len(h)), "low": 0, "mid": 0, "high": 0, "p10": None, "p90": None}
    return {
        "n": int(len(h)),
        "low": int(((h > 0.15) & (h < 0.7)).sum()),
        "mid": int(((h > 0.7) & (h < 1.5)).sum()),
        "high": int((h >= 1.5).sum()),
        "p10": float(np.percentile(h, 10)),
        "p90": float(np.percentile(h, 90)),
    }


def build_layout(
    xyz: np.ndarray,
    tier: str = "lidar",
    endpoint_mode: str = "intersections",
    min_room: float = 1.15,
) -> Layout:
    notes: list[str] = []
    abs_floor, rel_floor = TIER_FLOOR.get(tier, TIER_FLOOR["lidar"])
    if len(xyz) < 200:
        notes.append("too few points to build a plan")
        return Layout([], 0.0, endpoint_mode, notes, abs_floor)

    # Work on a subsample for the histogram. Full set is already voxelized.
    pts = xyz
    if len(pts) > 700_000:
        pts = pts[:: max(1, len(pts) // 700_000)]

    mask = vertical_columns(pts)
    above_furniture = mask & (pts[:, 1] > 0.8) & (pts[:, 1] < 1.9)
    wall_pts = pts[above_furniture] if int(above_furniture.sum()) > 400 else pts[mask]
    if len(wall_pts) < 150:
        notes.append("vertical-column filter kept too little; falling back to a mid-height band")
        h = pts[:, 1]
        wall_pts = pts[(h > 0.8) & (h < 1.8)]
    if len(wall_pts) < 80:
        notes.append("not enough wall points")
        return Layout([], 0.0, endpoint_mode, notes, abs_floor)

    xy = wall_pts[:, [0, 2]]
    height = wall_pts[:, 1]
    theta = _manhattan_theta(xy)
    q = _rotate2(xy, theta)
    # sigma of a wall position: robust spread inside a peak. Updated per peak.
    base_sigma = 0.01 if tier == "lidar" else (0.03 if tier == "video" else 0.06)

    x_peaks = _peaks_1d(q[:, 0], bin_m=0.02, min_sep=0.32, min_frac=0.22)
    y_peaks = _peaks_1d(q[:, 1], bin_m=0.02, min_sep=0.32, min_frac=0.22)

    def with_faces(peaks, axis: int) -> list[dict]:
        out = []
        for p in peaks:
            coord = q[:, axis]
            band = np.abs(coord - p["pos"]) <= 0.11
            if band.sum() < 40:
                continue
            vals = coord[band]
            along = q[band, 1 - axis]
            hh = height[band]
            lo = float(np.percentile(vals, 12))
            hi = float(np.percentile(vals, 88))
            thick = hi - lo
            if thick < 0.06:
                lo = hi = float(np.median(vals))
            # Door and window gaps live in the walking band. A header above a
            # door, or a sill below a window, must not fill the gap.
            door_band = (hh > 0.35) & (hh < 1.65)
            runs = _support_runs(along[door_band] if int(door_band.sum()) >= 10 else along)
            support_len = sum(b - a for a, b, _ in runs)
            if support_len < 0.8 and endpoint_mode == "intersections":
                # a short smear is clutter, unless we are in observed mode and it is the only evidence
                if support_len < 0.7:
                    continue
            spread = float(np.std(vals))
            out.append(
                {
                    "pos": p["pos"],
                    "face_lo": lo,
                    "face_hi": hi,
                    "score": p["score"],
                    "along": along,
                    "height": hh,
                    "coord": vals,
                    "runs": runs,
                    "support_len": support_len,
                    "sigma": max(base_sigma, spread),
                    "n": int(band.sum()),
                }
            )
        return out

    v_lines = _drop_weak_partitions(with_faces(x_peaks, 0))
    h_lines = _drop_weak_partitions(with_faces(y_peaks, 1))
    notes.append(f"manhattan theta {np.degrees(theta):.1f} deg, vertical lines {len(v_lines)}, horizontal lines {len(h_lines)}")

    if len(v_lines) < 2 or len(h_lines) < 2:
        notes.append("fewer than two walls in one direction; no closed room")
        return Layout([], theta, endpoint_mode, notes, base_sigma)

    # Grid coordinates. Interior face depends on which side the room is on,
    # so the grid line stores both faces.
    xs = v_lines  # already sorted by pos via peaks
    ys = h_lines
    # sort
    xs = sorted(xs, key=lambda L: L["pos"])
    ys = sorted(ys, key=lambda L: L["pos"])

    def x_face(line, toward_positive: bool) -> float:
        # toward_positive: the room is on the +x side, so the face we measure to is face_hi
        return line["face_hi"] if toward_positive else line["face_lo"]

    def y_face(line, toward_positive: bool) -> float:
        return line["face_hi"] if toward_positive else line["face_lo"]

    rooms: list[Room] = []
    # candidate cells
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            # room sits between line i and i+1, so left wall's interior face is the +x face of line i
            x0 = x_face(xs[i], True)
            x1 = x_face(xs[i + 1], False)
            y0 = y_face(ys[j], True)
            y1 = y_face(ys[j + 1], False)
            if x1 - x0 < min_room or y1 - y0 < min_room:
                continue
            # A grid line only counts as this cell's wall where it was actually
            # observed. Otherwise every pair of lines draws a room in the gaps.
            supports = (
                _edge_supported(ys[j], x0, x1),
                _edge_supported(xs[i + 1], y0, y1),
                _edge_supported(ys[j + 1], x0, x1),
                _edge_supported(xs[i], y0, y1),
            )
            if sum(supports) < 3:
                continue
            # interior emptiness: points strictly inside, inset by 8 cm
            # Inset past the wall thickness and the furniture pushed against it.
            inset = 0.22
            inside = (
                (q[:, 0] > x0 + inset)
                & (q[:, 0] < x1 - inset)
                & (q[:, 1] > y0 + inset)
                & (q[:, 1] < y1 - inset)
            )
            area = max((x1 - x0 - 2 * inset) * (y1 - y0 - 2 * inset), 0.1)
            density = float(inside.sum()) / area
            # ~1600 points/m^2 is a solid sheet at the 2.5 cm voxel.
            # A furnished room is well below that in the core.
            if density > 1400:
                notes.append(f"rejected cell {x1-x0:.2f}x{y1-y0:.2f} m as clutter (density {density:.0f}/m^2)")
                continue
            poly_m = _polygon_for_cell(
                xs[i], xs[i + 1], ys[j], ys[j + 1], x0, x1, y0, y1, theta, endpoint_mode, tier
            )
            if poly_m is None:
                continue
            ceil, ceil_sigma, ceil_ok = _ceiling_in_cell(xyz, theta, x0, x1, y0, y1)
            rooms.append(Room(poly_m["polygon"], poly_m["walls"], ceil, ceil_sigma, ceil_ok))

    if not rooms:
        notes.append("grid produced no room cell above the minimum size")
    return Layout(rooms, theta, endpoint_mode, notes, base_sigma)


def _polygon_for_cell(left, right, bottom, top, x0, x1, y0, y1, theta, endpoint_mode, tier) -> dict | None:
    """Rectangle in world XZ, plus four walls. Openings cut from the support gaps."""
    # corners in Manhattan frame, then unrotate to world XZ (x, z)
    corners_q = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    polygon = _unrotate2(corners_q, theta)

    edges = [
        # (p0, p1, line dict, along-axis is y for vertical walls and x for horizontal)
        (polygon[0], polygon[1], bottom, "h", y0, x0, x1),
        (polygon[1], polygon[2], right, "v", x1, y0, y1),
        (polygon[2], polygon[3], top, "h", y1, x0, x1),
        (polygon[3], polygon[0], left, "v", x0, y0, y1),
    ]
    walls: list[WallSeg] = []
    abs_floor, rel_floor = TIER_FLOOR.get(tier, TIER_FLOOR["lidar"])
    for p0, p1, line, kind, _pos, a0, a1 in edges:
        full_len = float(np.linalg.norm(p1 - p0))
        if endpoint_mode == "observed":
            if not line["runs"]:
                continue
            # observed length is the longest run, which is what a partial pass measures
            run = max(line["runs"], key=lambda r: r[1] - r[0])
            length = float(run[1] - run[0])
            # place the shorter segment in the middle of the full side so the drawing still sits on the wall
            # but the reported length is the observed one
            direction = (p1 - p0) / max(full_len, 1e-6)
            mid = 0.5 * (p0 + p1)
            p0 = mid - direction * (length / 2)
            p1 = mid + direction * (length / 2)
        else:
            length = full_len
        openings = _openings_on_line(line, a0, a1, length, abs_floor)
        walls.append(WallSeg(p0=p0, p1=p1, sigma=float(line["sigma"]), support=int(line["n"]), openings=openings))
    if len(walls) < 3:
        return None
    return {"polygon": polygon, "walls": walls}


def _openings_on_line(line: dict, a0: float, a1: float, wall_len: float, abs_floor: float) -> list[Opening]:
    """Gaps in the along-wall support, flanked on both sides.

    A missing end of a wall is an unscanned corner, not a door. Both
    flanks have to be present. Width is conservative on purpose: a phantom
    opening counts the same as a miss.
    """
    runs = sorted(line["runs"])
    if len(runs) < 2:
        return []
    openings = []
    along = line["along"]
    heights = line["height"]
    # flank occupancy scale
    for (a, b, _), (c, d, _) in zip(runs, runs[1:]):
        gap0, gap1 = b, c
        width = gap1 - gap0
        flank_left = b - a
        flank_right = d - c
        if flank_left < 0.35 or flank_right < 0.35:
            continue
        if width < 0.68 or width > 1.20:
            continue
        # the gap must sit on the wall we are measuring, not past its ends
        if gap0 < a0 - 0.05 or gap1 > a1 + 0.05:
            continue
        prof = _height_profile(heights, along, gap0, gap1)
        # empty through the door band: very few mid-height returns
        flank_mid = 1
        # compare with a flank slice
        flank = _height_profile(heights, along, a, b)
        if flank["mid"] > 20 and prof["mid"] > 0.25 * flank["mid"]:
            continue
        kind = "door"
        sill = 0.0
        head = None
        if prof["low"] > prof["mid"] and prof["low"] > 8 and prof["high"] > 8:
            kind = "window"
            sill = prof["p90"] if prof["p10"] is not None and prof["low"] > prof["mid"] else None
            # sill is the top of the low cluster: approximate with 0.7 if low points exist
            sill = 0.85 if prof["low"] > 8 else None
            head = None
        sigma = max(0.02, abs_floor)
        openings.append(
            Opening(kind=kind, width=float(width), sigma=sigma, along0=float(gap0), along1=float(gap1), sill=sill, head=head)
        )
    return openings


def _ceiling_in_cell(xyz: np.ndarray, theta: float, x0, x1, y0, y1):
    xy = _rotate2(xyz[:, [0, 2]], theta)
    h = xyz[:, 1]
    inside = (xy[:, 0] > x0) & (xy[:, 0] < x1) & (xy[:, 1] > y0) & (xy[:, 1] < y1)
    hh = h[inside]
    high = hh[hh > 1.85]
    if len(high) < 80:
        return None, None, False
    hist, edges = np.histogram(high, bins=np.arange(1.85, min(4.2, high.max() + 0.05), 0.02))
    if hist.max() < 40:
        return None, None, False
    i = int(hist.argmax())
    center = 0.5 * (edges[i] + edges[i + 1])
    band = high[np.abs(high - center) < 0.06]
    if len(band) < 40:
        return None, None, False
    # a real ceiling is a sharp peak, not a smear of high wall points
    if float(np.std(band)) > 0.04:
        return None, None, False
    return float(np.median(band)), float(np.std(band) + 0.005), True


def layout_to_dict(layout: Layout, tier: str, xyz_for_area_sigma: float | None = None) -> dict:
    abs_floor, rel_floor = TIER_FLOOR.get(tier, TIER_FLOOR["lidar"])
    rooms_out = []
    adjacencies = []
    for ri, room in enumerate(layout.rooms):
        area = shoelace(room.polygon)
        # area uncertainty grows with the perimeter and the wall-position sigma
        perim = float(np.linalg.norm(np.roll(room.polygon, -1, axis=0) - room.polygon, axis=1).sum())
        wall_sigma = float(np.median([w.sigma for w in room.walls])) if room.walls else layout.wall_sigma
        area_sigma = perim * wall_sigma * 0.5
        walls_out = []
        for wi, w in enumerate(room.walls):
            length = float(np.linalg.norm(w.p1 - w.p0))
            # position error on both ends
            length_sigma = float(np.sqrt(2) * w.sigma)
            openings_out = []
            for oi, op in enumerate(w.openings):
                openings_out.append(
                    {
                        "id": f"room{ri}_w{wi}_o{oi}",
                        "class": op.kind,
                        "width_m": meas(op.width, op.sigma, max(abs_floor, 0.02), rel_floor),
                        "sill_height_m": None
                        if op.sill is None
                        else meas(op.sill, max(op.sigma, 0.03), max(abs_floor, 0.03), rel_floor),
                        "head_height_m": None
                        if op.head is None
                        else meas(op.head, max(op.sigma, 0.04), max(abs_floor, 0.04), rel_floor),
                        "observed_head": op.head is not None,
                    }
                )
            height_m = None
            if room.ceiling_observed and room.ceiling is not None:
                height_m = meas(room.ceiling, room.ceiling_sigma or 0.02, max(abs_floor, 0.015), rel_floor)
            walls_out.append(
                {
                    "id": f"room{ri}_w{wi}",
                    "length_m": meas(length, length_sigma, abs_floor, rel_floor),
                    "height_m": height_m,
                    "p0_m": [round(float(w.p0[0]), 4), round(float(w.p0[1]), 4)],
                    "p1_m": [round(float(w.p1[0]), 4), round(float(w.p1[1]), 4)],
                    "support_points": w.support,
                    "openings": openings_out,
                }
            )
        ceil = None
        if room.ceiling_observed and room.ceiling is not None:
            ceil = meas(room.ceiling, room.ceiling_sigma or 0.02, max(abs_floor, 0.015), rel_floor)
        rooms_out.append(
            {
                "id": f"room_{ri}",
                "polygon_m": [[round(float(p[0]), 4), round(float(p[1]), 4)] for p in room.polygon],
                "floor_area_m2": meas(area, area_sigma, max(0.05, abs_floor * perim), rel_floor),
                "ceiling_height_m": ceil,
                "ceiling_observed": room.ceiling_observed,
                "walls": walls_out,
                "damage": [],
            }
        )
    # adjacency: rooms whose polygons share an edge within 0.2 m
    for i in range(len(layout.rooms)):
        for j in range(i + 1, len(layout.rooms)):
            link = _shared_edge(layout.rooms[i].polygon, layout.rooms[j].polygon)
            if link is not None:
                adjacencies.append(
                    {
                        "a": f"room_{i}",
                        "b": f"room_{j}",
                        "kind": "shared_wall",
                        "gap_m": round(link, 4),
                    }
                )
    return {"rooms": rooms_out, "adjacencies": adjacencies, "notes": layout.notes, "manhattan_theta_deg": round(float(np.degrees(layout.theta)), 3)}


def _shared_edge(poly_a: np.ndarray, poly_b: np.ndarray, tol: float = 0.25) -> float | None:
    """Return the gap between the closest parallel edges, if they overlap."""
    def edges(poly):
        out = []
        for i in range(len(poly)):
            p, q = poly[i], poly[(i + 1) % len(poly)]
            out.append((p, q))
        return out

    best = None
    for p0, p1 in edges(poly_a):
        d = p1 - p0
        L = np.linalg.norm(d)
        if L < 0.4:
            continue
        direction = d / L
        normal = np.array([-direction[1], direction[0]])
        for q0, q1 in edges(poly_b):
            gap = abs(float(np.dot(q0 - p0, normal)))
            if gap > tol:
                continue
            # overlap along the wall
            t0 = float(np.dot(q0 - p0, direction))
            t1 = float(np.dot(q1 - p0, direction))
            lo, hi = sorted([t0, t1])
            overlap = min(hi, L) - max(lo, 0.0)
            if overlap > 0.4:
                if best is None or gap < best:
                    best = gap
    return best


def footprint_area(layout: Layout) -> float:
    return float(sum(shoelace(r.polygon) for r in layout.rooms))
