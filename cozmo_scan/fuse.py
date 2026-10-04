"""LiDAR fusion and the drift correction that sits on top of it.

Drift is handled in two layers, and both can be switched off for the
ablation the brief asks for:

1. Floor-plane anchor. A single RANSAC floor removes the fixed tilt of
   the whole capture. Each time-chunk of the walk then has its own
   floor height shifted back to zero, so a slow vertical drift does not
   tilt the ceiling or smear wall histograms.
2. One-edge loop. If the camera finishes near where it started, a 2D
   rigid alignment of the late wall points onto the early wall points
   is spread linearly along the trajectory. If the walk does not close,
   this step reports that it did not fire. It is not applied silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .capture import DEPTH_WH, RGB_WH, Capture
from .geometry import backproject, quat_to_matrix, ransac_plane, rotation_from_to, voxel_downsample


@dataclass
class Cloud:
    xyz: np.ndarray  # raw logger world, meters
    frame_index: np.ndarray  # int, index into Capture.poses
    colors: np.ndarray | None = None  # uint8 Nx3, optional
    source: str = "lidar"
    meta: dict = field(default_factory=dict)


def _depth_intrinsics(pose, depth_wh, rgb_wh) -> tuple[float, float, float, float]:
    sx = depth_wh[0] / float(rgb_wh[0])
    sy = depth_wh[1] / float(rgb_wh[1])
    return pose.fx * sx, pose.fy * sy, pose.cx * sx, pose.cy * sy


def fuse_lidar(
    capture: Capture,
    frame_stride: int | None = None,
    pixel_stride: int = 3,
    max_frames: int = 220,
    conf_min: int = 2,
    z_min: float = 0.25,
    z_max: float = 4.8,
) -> Cloud:
    depth_dir = capture.depth_dir
    conf_dir = capture.confidence_dir
    frames = sorted(p.stem for p in depth_dir.glob("*.png"))
    pose_map = capture.pose_by_frame()
    frames = [f for f in frames if f in pose_map]
    if not frames:
        raise RuntimeError(f"no depth frames with poses in {capture.root}")

    if frame_stride is None:
        frame_stride = max(1, int(np.ceil(len(frames) / max_frames)))
    chosen = frames[::frame_stride]
    # Map frame string -> index in the full pose list for drift bucketing.
    pose_index = {p.frame: i for i, p in enumerate(capture.poses)}

    chunks = []
    ids = []
    used = 0
    for fr in chosen:
        depth = np.array(Image.open(depth_dir / f"{fr}.png"))
        if depth.ndim != 2:
            continue
        z = depth.astype(np.float32) / 1000.0
        conf_path = conf_dir / f"{fr}.png"
        if conf_path.exists():
            conf = np.array(Image.open(conf_path))
            mask = (conf >= conf_min) & (z > z_min) & (z < z_max)
        else:
            mask = (z > z_min) & (z < z_max)
        vs, us = np.nonzero(mask)
        if len(vs) < 40:
            continue
        vs = vs[::pixel_stride]
        us = us[::pixel_stride]
        zz = z[vs, us].astype(np.float64)
        pose = pose_map[fr]
        h, w = depth.shape
        fx, fy, cx, cy = _depth_intrinsics(pose, (w, h), capture.rgb_size)
        cam = backproject(us.astype(np.float64), vs.astype(np.float64), zz, fx, fy, cx, cy)
        R = quat_to_matrix(pose.q)
        world = cam @ R.T + pose.t
        chunks.append(world.astype(np.float32))
        ids.append(np.full(len(world), pose_index[fr], dtype=np.int32))
        used += 1

    if not chunks:
        raise RuntimeError("fusion produced no points; confidence or depth range filtered everything")

    xyz = np.concatenate(chunks, axis=0)
    frame_index = np.concatenate(ids, axis=0)
    xyz, [frame_index] = voxel_downsample(xyz, 0.025, [frame_index])
    return Cloud(
        xyz=xyz,
        frame_index=frame_index,
        source="lidar",
        meta={
            "depth_frames_on_disk": len(frames),
            "depth_frames_used": used,
            "frame_stride": frame_stride,
            "pixel_stride": pixel_stride,
            "points": int(len(xyz)),
        },
    )


def _floor_mode(heights: np.ndarray) -> float | None:
    if len(heights) < 80:
        return None
    lo = np.percentile(heights, 1)
    band = heights[(heights >= lo - 0.02) & (heights <= lo + 0.25)]
    if len(band) < 40:
        return None
    hist, edges = np.histogram(band, bins=40)
    i = int(hist.argmax())
    return float(0.5 * (edges[i] + edges[i + 1]))


def align_floor(cloud: Cloud, enable_anchor: bool, enable_loop: bool) -> dict:
    """Return layout-frame points (Y up, floor at 0) and an ablation record.

    `xyz` in the returned dict is the default (anchor on, loop if it fires).
    Ablation copies are included so one fusion feeds every reported number.
    """
    raw = cloud.xyz.astype(np.float64)
    report = {
        "method": [],
        "loop_fired": False,
        "floor_tilt_deg": None,
        "floor_residual_m": None,
        "chunk_floor_offsets_m": [],
    }

    def _package(points: np.ndarray) -> np.ndarray:
        return points.astype(np.float32)

    # --- no correction: histogram floor only, Y stays the logger Y ---
    mode = _floor_mode(raw[:, 1])
    if mode is None:
        mode = float(np.percentile(raw[:, 1], 5))
    uncorrected = raw.copy()
    uncorrected[:, 1] -= mode

    # --- plane anchor ---
    low = raw[raw[:, 1] < mode + 0.18]
    plane = ransac_plane(low, horizontal=True) if len(low) > 80 else None
    floor_h = 0.0
    if plane is None or not enable_anchor:
        R = np.eye(3)
        origin = np.array([0.0, 0.0, 0.0])
        floor_h = mode
        anchored = raw.copy()
        anchored[:, 1] -= floor_h
        report["method"].append("histogram floor, no rotation" if plane is None else "floor anchor disabled")
        tilt = 0.0
        residual = None
    else:
        R = rotation_from_to(plane["normal"], np.array([0.0, 1.0, 0.0]))
        origin = plane["point"]
        anchored = (raw - origin) @ R.T
        near = np.abs((raw - plane["point"]) @ plane["normal"]) < 0.03
        floor_h = float(np.median(anchored[near, 1])) if near.any() else 0.0
        anchored[:, 1] -= floor_h
        tilt = float(np.degrees(np.arccos(np.clip(plane["normal"][1], -1, 1))))
        residual = plane["residual_std"]
        report["method"].append("RANSAC floor-plane anchor")
        report["floor_tilt_deg"] = round(tilt, 3)
        report["floor_residual_m"] = round(residual, 4)

    # per-chunk vertical shift on the anchored cloud
    chunked = anchored.copy()
    offsets = []
    chunk_bounds = []
    qs = np.array([0.0, 1.0])
    if enable_anchor and cloud.frame_index.size == len(chunked) and len(cloud.frame_index):
        idx = cloud.frame_index
        # 4 temporal chunks
        qs = np.quantile(idx, [0.0, 0.25, 0.5, 0.75, 1.0])
        for a, b in zip(qs[:-1], qs[1:]):
            last = b == qs[-1]
            sel = (idx >= a) & (idx <= b if last else idx < b)
            if sel.sum() < 100:
                continue
            m = _floor_mode(chunked[sel, 1])
            if m is None:
                continue
            # only shift if the local floor is a real mode near zero, not a missing floor
            if abs(m) > 0.25:
                continue
            chunked[sel, 1] -= m
            offsets.append(round(float(m), 4))
            chunk_bounds.append((float(a), float(b), float(m), bool(last)))
        if offsets:
            report["method"].append("per-chunk vertical floor shift")
    report["chunk_floor_offsets_m"] = offsets

    looped = chunked.copy()
    loop_info = {"fired": False, "reason": "disabled"}
    if enable_loop:
        looped, loop_info = _loop_close(looped, cloud.frame_index)
        report["loop_fired"] = bool(loop_info.get("fired"))
        report["loop"] = loop_info
        if loop_info.get("fired"):
            report["method"].append("trajectory loop, 2D rigid, spread along the walk")
        else:
            report["method"].append("loop closure checked and not applied")
    else:
        report["loop"] = loop_info

    replay = {
        "R": R,
        "origin": origin,
        "floor_h": float(floor_h),
        "chunks": chunk_bounds,
    }
    return {
        "xyz": _package(looped if enable_loop else chunked),
        "xyz_anchor_off": _package(uncorrected),
        "xyz_anchor_on_loop_off": _package(chunked),
        "xyz_full": _package(looped),
        "frame_index": cloud.frame_index,
        "rotation": R,
        "origin": origin,
        "replay": replay,
        "report": report,
    }


def apply_replay(raw: np.ndarray, frame_index: np.ndarray, replay: dict) -> np.ndarray:
    """Same floor anchor as the cloud the plan was built from. Loop closure is not replayed."""
    R = replay["R"]
    origin = replay["origin"]
    out = (raw.astype(np.float64) - origin) @ R.T
    out[:, 1] -= replay["floor_h"]
    if frame_index is None or len(frame_index) != len(out):
        return out
    for a, b, shift, last in replay.get("chunks", []):
        sel = (frame_index >= a) & (frame_index <= b if last else frame_index < b)
        out[sel, 1] -= shift
    return out


def _loop_close(points: np.ndarray, frame_index: np.ndarray) -> tuple[np.ndarray, dict]:
    """Yaw+translation that maps the end of the walk onto the start, in the floor plane."""
    info = {"fired": False, "reason": ""}
    if frame_index is None or len(frame_index) != len(points) or len(points) < 500:
        info["reason"] = "no per-point frame index"
        return points, info
    lo, hi = np.quantile(frame_index, [0.12, 0.88])
    early = points[frame_index <= lo]
    late = points[frame_index >= hi]
    # wall band
    early = early[(early[:, 1] > 0.7) & (early[:, 1] < 1.8)]
    late = late[(late[:, 1] > 0.7) & (late[:, 1] < 1.8)]
    if len(early) < 200 or len(late) < 200:
        info["reason"] = "not enough wall points at both ends of the walk"
        return points, info
    e2 = _voxel2(early[:, [0, 2]], 0.05)
    l2 = _voxel2(late[:, [0, 2]], 0.05)
    if len(e2) < 30 or len(l2) < 30:
        info["reason"] = "wall footprint too small to close"
        return points, info
    # Only attempt a loop when the camera path ends near the start.
    # We don't have the path here, so use the wall-cloud centroids as a proxy:
    # a looped room still has similar centroids. Reject a large rigid jump.
    yaw, trans, rmse_before, rmse_after = _icp_2d(l2, e2)
    info.update(
        {
            "yaw_deg": round(float(np.degrees(yaw)), 3),
            "translation_m": [round(float(trans[0]), 3), round(float(trans[1]), 3)],
            "rmse_before_m": round(float(rmse_before), 3),
            "rmse_after_m": round(float(rmse_after), 3),
        }
    )
    if abs(yaw) > np.radians(8) or np.linalg.norm(trans) > 0.75:
        info["reason"] = "alignment is too large to be a loop; left poses unchanged"
        return points, info
    if rmse_after > rmse_before * 0.85:
        info["reason"] = "alignment did not tighten the loop"
        return points, info
    # Spread the correction from 0 at the start to the full rigid motion at the end.
    out = points.copy()
    alpha = (frame_index - frame_index.min()) / max(1, (frame_index.max() - frame_index.min()))
    alpha = alpha.astype(np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    xz = out[:, [0, 2]]
    # rotate around the early centroid so the start of the walk stays put
    center = e2.mean(axis=0)
    shifted = xz - center
    rotated = np.stack([c * shifted[:, 0] - s * shifted[:, 1], s * shifted[:, 0] + c * shifted[:, 1]], axis=1)
    rotated = rotated + center + trans
    # interpolate: result = (1-a)*original + a*corrected, but yaw should interpolate too.
    # Small-angle: blend the delta.
    delta = rotated - xz
    out[:, 0] += delta[:, 0] * alpha
    out[:, 2] += delta[:, 1] * alpha
    info["fired"] = True
    info["reason"] = "applied"
    return out, info


def _voxel2(xy: np.ndarray, voxel: float) -> np.ndarray:
    if len(xy) == 0:
        return xy
    key = np.floor(xy / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    pts = xy[idx]
    if len(pts) > 2500:
        rng = np.random.default_rng(0)
        pts = pts[rng.choice(len(pts), 2500, replace=False)]
    return pts


def _icp_2d(src: np.ndarray, dst: np.ndarray, iters: int = 12):
    """Rigid ICP. Returns yaw, translation that maps src onto dst, and RMSEs."""
    def rmse(a, b):
        d = _nn_dist(a, b)
        return float(np.sqrt(np.mean(d ** 2)))

    cur = src.copy()
    yaw_total = 0.0
    # translation is accumulated inside cur; recover it at the end relative to src
    before = rmse(src, dst)
    for _ in range(iters):
        nn = _nn_index(cur, dst)
        matched = dst[nn]
        yaw, t = _rigid_2d(cur, matched)
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        cur = cur @ R.T + t
        yaw_total += yaw
    after = rmse(cur, dst)
    # Recover the single rigid motion mapping src -> cur (approximately yaw_total about origin
    # is wrong because each step rotates about a different point). Fit once more.
    yaw, trans = _rigid_2d(src, cur)
    return yaw, trans, before, after


def _nn_index(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    # Chunked brute force. Both clouds are capped at 2500.
    best_d = np.full(len(src), np.inf)
    best_i = np.zeros(len(src), dtype=np.int32)
    chunk = 400
    for i in range(0, len(dst), chunk):
        block = dst[i : i + chunk]
        # (Ns, Nb)
        d = np.sum((src[:, None, :] - block[None, :, :]) ** 2, axis=2)
        j = np.argmin(d, axis=1)
        dd = d[np.arange(len(src)), j]
        take = dd < best_d
        best_d[take] = dd[take]
        best_i[take] = j[take] + i
    return best_i


def _nn_dist(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    idx = _nn_index(src, dst)
    return np.linalg.norm(src - dst[idx], axis=1)


def _rigid_2d(src: np.ndarray, dst: np.ndarray):
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    X = src - mu_s
    Y = dst - mu_d
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    t = mu_d - mu_s @ R.T
    return yaw, t
