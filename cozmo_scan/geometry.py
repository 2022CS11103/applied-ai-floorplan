"""Small geometry helpers.

World frame after fusion, before floor alignment, follows the logger:
camera X right, Y down, Z forward (OpenCV), pose quaternion is
camera-to-world. In that convention the sample scans have Y pointing up
and the camera sitting about 1.4 m above a flat floor. That was checked
on the provided captures; it is not an ARKit-native camera axis (ARKit
looks down -Z). Using the ARKit axis on this logger stretches the
vertical span to ~5 m, which is not a room.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def quat_to_matrix(q_xyzw: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(q_xyzw).as_matrix()


def backproject(u, v, z, fx, fy, cx, cy) -> np.ndarray:
    """OpenCV camera: X right, Y down, Z forward. z is meters."""
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def rotation_from_to(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rotation taking unit vector src onto dst."""
    a = src / np.linalg.norm(src)
    b = dst / np.linalg.norm(dst)
    c = float(np.dot(a, b))
    if c > 0.9999:
        return np.eye(3)
    if c < -0.9999:
        # 180 degrees around an arbitrary perpendicular axis
        axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = axis - a * np.dot(axis, a)
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(axis * np.pi).as_matrix()
    v = np.cross(a, b)
    vx = skew(v)
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def ransac_plane(points: np.ndarray, iters: int = 250, thresh: float = 0.02, horizontal: bool = True, seed: int = 0):
    """Fit n·x = n·p0. Normal is forced to have positive Y when horizontal."""
    if len(points) < 50:
        return None
    rng = np.random.default_rng(seed)
    sample = points
    if len(points) > 12000:
        sample = points[rng.choice(len(points), 12000, replace=False)]
    best = None
    n = len(sample)
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        a, b, c = sample[idx]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-8:
            continue
        normal = normal / length
        if horizontal:
            if normal[1] < 0:
                normal = -normal
            if normal[1] < 0.92:
                continue
        dist = np.abs((sample - a) @ normal)
        count = int((dist < thresh).sum())
        if best is None or count > best[0]:
            best = (count, normal, a)
    if best is None:
        return None
    _, normal, p0 = best
    dist = np.abs((sample - p0) @ normal)
    inliers = sample[dist < thresh]
    if len(inliers) < 30:
        return None
    centroid = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    if horizontal and normal[1] < 0:
        normal = -normal
    residual = np.abs((inliers - centroid) @ normal)
    return {
        "normal": normal,
        "point": centroid,
        "inliers": len(inliers),
        "residual_std": float(residual.std()),
    }


def meas(value: float, sigma: float, abs_floor: float, rel_floor: float = 0.0) -> dict:
    """95% interval. Floors stop a thin tier from publishing fake precision."""
    half = max(1.96 * float(sigma), float(abs_floor), abs(float(value)) * float(rel_floor))
    return {
        "value": round(float(value), 4),
        "sigma": round(float(sigma), 4),
        "ci95_low": round(float(value) - half, 4),
        "ci95_high": round(float(value) + half, 4),
    }


def shoelace(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def voxel_downsample(points: np.ndarray, voxel: float, extras: list[np.ndarray] | None = None):
    if len(points) == 0:
        return points if not extras else (points, extras)
    key = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    idx.sort()
    if extras is None:
        return points[idx]
    return points[idx], [e[idx] for e in extras]
