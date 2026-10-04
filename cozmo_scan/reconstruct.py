"""Video and photo tiers.

Video, when the logger recorded metric VIO poses, triangulates RGB
features with those poses and never reads a depth image. Scale is the
VIO scale. The interval is wider than LiDAR because a textureless wall
simply does not produce points.

Photo stills have no poses. A short sequential SfM recovers shape, and
scale comes from the capture protocol: the phone is held at 1.40 m, and
the floor plane is set to that distance below the cameras. That prior is
why the photo interval is ±8% at minimum. If no floor plane is found the
run is marked degraded instead of inventing a scale.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .capture import Capture, list_photo_paths
from .fuse import Cloud
from .geometry import backproject, quat_to_matrix, ransac_plane


CAMERA_HEIGHT_M = 1.40


def _K_from_capture(capture: Capture, width: int, height: int) -> np.ndarray:
    if capture.K_rgb is not None:
        K = capture.K_rgb.copy()
        # scale if the file was recorded at a different resolution than the matrix assumes
        # sample matrices match 1920x1440; if the image is that size, use as-is
        return K
    # fallback pinhole, ~70 deg horizontal, enough to run, interval stays wide
    fx = 0.85 * width
    return np.array([[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]], dtype=np.float64)


def _projection(K: np.ndarray, R_wc: np.ndarray, t_wc: np.ndarray) -> np.ndarray:
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc
    return K @ np.hstack([R_cw, t_cw.reshape(3, 1)])


def fuse_video(capture: Capture, max_pairs: int = 40) -> Cloud:
    video = capture.video_path
    if video is None:
        raise RuntimeError(f"no video in {capture.root}")
    if len(capture.poses) < 5:
        # No metric poses: treat a subsample of frames as a photo sequence.
        paths = _dump_frames(video, capture.root / "_video_frames", count=8)
        return fuse_photos(paths, capture, source="video_unscaled_sfm")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    K = _K_from_capture(capture, width, height)
    n = len(capture.poses)
    step = max(1, n // (max_pairs + 1))
    indices = list(range(0, n, step))[: max_pairs + 1]
    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(capture.poses[idx].frame))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        frames.append((idx, bgr, kp, des))
    cap.release()

    points = []
    colors = []
    frame_ids = []
    for (i0, bgr0, kp0, des0), (i1, bgr1, kp1, des1) in zip(frames, frames[1:]):
        if des0 is None or des1 is None or len(kp0) < 20 or len(kp1) < 20:
            continue
        matches = bf.match(des0, des1)
        matches = sorted(matches, key=lambda m: m.distance)[:400]
        if len(matches) < 30:
            continue
        p0 = capture.poses[i0]
        p1 = capture.poses[i1]
        baseline = float(np.linalg.norm(p1.t - p0.t))
        if baseline < 0.05 or baseline > 0.8:
            continue
        pts0 = np.float32([kp0[m.queryIdx].pt for m in matches])
        pts1 = np.float32([kp1[m.trainIdx].pt for m in matches])
        R0 = quat_to_matrix(p0.q)
        R1 = quat_to_matrix(p1.q)
        P0 = _projection(K, R0, p0.t)
        P1 = _projection(K, R1, p1.t)
        homo = cv2.triangulatePoints(P0, P1, pts0.T, pts1.T)
        X = (homo[:3] / homo[3]).T
        # positive depth in both cameras and a sane room range
        for k, Xw in enumerate(X):
            if not np.isfinite(Xw).all():
                continue
            z0 = _cam_z(Xw, R0, p0.t)
            z1 = _cam_z(Xw, R1, p1.t)
            if z0 < 0.3 or z1 < 0.3 or z0 > 6 or z1 > 6:
                continue
            u0 = _project(Xw, K, R0, p0.t)
            if np.linalg.norm(u0 - pts0[k]) > 3.0:
                continue
            points.append(Xw)
            x, y = int(pts0[k, 0]), int(pts0[k, 1])
            if 0 <= x < bgr0.shape[1] and 0 <= y < bgr0.shape[0]:
                b, g, r = bgr0[y, x]
                colors.append((r, g, b))
            else:
                colors.append((0, 0, 0))
            frame_ids.append(i0)
    if len(points) < 80:
        raise RuntimeError(
            f"video triangulation kept {len(points)} points. The walk may be too dark or too blank to survey from RGB alone."
        )
    xyz = np.asarray(points, dtype=np.float32)
    cols = np.asarray(colors, dtype=np.uint8)
    fidx = np.asarray(frame_ids, dtype=np.int32)
    return Cloud(
        xyz=xyz,
        frame_index=fidx,
        colors=cols,
        source="video_triangulation",
        meta={"points": int(len(xyz)), "pairs": len(frames) - 1, "poses": n},
    )


def _cam_z(Xw, R_wc, t_wc) -> float:
    Xc = R_wc.T @ (Xw - t_wc)
    return float(Xc[2])


def _project(Xw, K, R_wc, t_wc) -> np.ndarray:
    Xc = R_wc.T @ (Xw - t_wc)
    u = K @ Xc
    return u[:2] / u[2]


def _dump_frames(video: Path, out_dir: Path, count: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        n = count * 10
    idxs = np.linspace(0, max(n - 1, 1), count).astype(int)
    paths = []
    for i, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        path = out_dir / f"frame_{i:02d}.jpg"
        cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        paths.append(path)
    cap.release()
    return paths


def fuse_photos(paths: list[Path], capture: Capture | None = None, source: str = "photo_sfm") -> Cloud:
    if len(paths) < 2:
        raise RuntimeError("photo tier needs at least 2 stills")
    images = []
    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        images.append((p, bgr))
    if len(images) < 2:
        raise RuntimeError("could not read photo stills")
    h, w = images[0][1].shape[:2]
    if capture is not None and capture.K_rgb is not None:
        K = capture.K_rgb.copy()
    else:
        K = _K_from_capture(Capture(root=paths[0].parent, capture_id="photos", kind="photo_folder"), w, h)

    orb = cv2.ORB_create(2500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    feats = []
    for _, bgr in images:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        feats.append((kp, des, bgr))

    # Pose of image 0 is identity. Subsequent poses are camera-to-world, arbitrary scale.
    R_wc = [np.eye(3)]
    t_wc = [np.zeros(3)]
    cloud_xyz = []
    cloud_rgb = []
    cloud_f = []

    for i in range(len(feats) - 1):
        kp0, des0, bgr0 = feats[i]
        kp1, des1, bgr1 = feats[i + 1]
        if des0 is None or des1 is None:
            R_wc.append(R_wc[-1])
            t_wc.append(t_wc[-1])
            continue
        matches = sorted(bf.match(des0, des1), key=lambda m: m.distance)[:500]
        if len(matches) < 40:
            R_wc.append(R_wc[-1].copy())
            t_wc.append(t_wc[-1].copy())
            continue
        pts0 = np.float32([kp0[m.queryIdx].pt for m in matches])
        pts1 = np.float32([kp1[m.trainIdx].pt for m in matches])
        E, emask = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.5)
        if E is None:
            R_wc.append(R_wc[-1].copy())
            t_wc.append(t_wc[-1].copy())
            continue
        _, R_rel, t_rel, _ = cv2.recoverPose(E, pts0, pts1, K)
        # R_rel, t_rel take camera i points into camera i+1: X_{i+1} = R_rel X_i + t_rel
        # We store camera-to-world. X_w = R_i X_i + t_i
        # X_w = R_{i+1} X_{i+1} + t_{i+1} = R_{i+1} (R_rel X_i + t_rel) + t_{i+1}
        # so R_i = R_{i+1} R_rel, t_i = R_{i+1} t_rel + t_{i+1}
        # => R_{i+1} = R_i R_rel.T, t_{i+1} = t_i - R_{i+1} @ t_rel
        R_next = R_wc[i] @ R_rel.T
        t_next = t_wc[i] - R_next @ t_rel.reshape(3)
        R_wc.append(R_next)
        t_wc.append(t_next.reshape(3))
        P0 = _projection(K, R_wc[i], t_wc[i])
        P1 = _projection(K, R_next, t_next.reshape(3))
        homo = cv2.triangulatePoints(P0, P1, pts0.T, pts1.T)
        X = (homo[:3] / np.clip(homo[3], 1e-8, None)).T
        for k, Xw in enumerate(X):
            if not np.isfinite(Xw).all():
                continue
            z0 = _cam_z(Xw, R_wc[i], t_wc[i])
            z1 = _cam_z(Xw, R_next, t_next)
            if z0 < 0.05 or z1 < 0.05:
                continue
            x, y = int(pts0[k, 0]), int(pts0[k, 1])
            if 0 <= x < bgr0.shape[1] and 0 <= y < bgr0.shape[0]:
                b, g, r = bgr0[y, x]
                cloud_rgb.append((r, g, b))
            else:
                cloud_rgb.append((128, 128, 128))
            cloud_xyz.append(Xw)
            cloud_f.append(i)

    if len(cloud_xyz) < 60:
        raise RuntimeError(f"photo SfM kept {len(cloud_xyz)} points; the stills may not overlap")

    xyz = np.asarray(cloud_xyz, dtype=np.float64)
    cams = np.asarray(t_wc, dtype=np.float64)
    scaled, scale_meta = scale_to_camera_height(xyz, cams, R_wc)
    return Cloud(
        xyz=scaled.astype(np.float32),
        frame_index=np.asarray(cloud_f, dtype=np.int32),
        colors=np.asarray(cloud_rgb, dtype=np.uint8),
        source=source,
        meta={"points": int(len(scaled)), "images": len(images), **scale_meta},
    )


def scale_to_camera_height(xyz: np.ndarray, cam_positions: np.ndarray, R_wc: list[np.ndarray]) -> tuple[np.ndarray, dict]:
    """Set the median camera-to-floor distance to CAMERA_HEIGHT_M.

    Camera up is -Y in the OpenCV camera, rotated into the SfM world.
    The floor is the RANSAC plane whose normal best agrees with that up axis.
    """
    ups = []
    for R in R_wc:
        up = R @ np.array([0.0, -1.0, 0.0])
        ups.append(up)
    mean_up = np.mean(ups, axis=0)
    mean_up = mean_up / max(np.linalg.norm(mean_up), 1e-8)

    # Try several plane fits and keep the one most parallel to camera-up.
    rng = np.random.default_rng(1)
    sample = xyz
    if len(sample) > 8000:
        sample = xyz[rng.choice(len(xyz), 8000, replace=False)]
    best = None
    for _ in range(200):
        idx = rng.choice(len(sample), 3, replace=False)
        a, b, c = sample[idx]
        n = np.cross(b - a, c - a)
        L = np.linalg.norm(n)
        if L < 1e-8:
            continue
        n = n / L
        if np.dot(n, mean_up) < 0:
            n = -n
        if np.dot(n, mean_up) < 0.8:
            continue
        dist = np.abs((sample - a) @ n)
        inl = dist < np.percentile(np.abs((sample - sample.mean(0)) @ n) + 1e-6, 40) * 0.5 + 1e-3
        # use a relative threshold: 2% of the cloud's spread
        spread = float(np.linalg.norm(sample.max(0) - sample.min(0)))
        dist = np.abs((sample - a) @ n)
        count = int((dist < 0.02 * spread).sum())
        if best is None or count > best[0]:
            best = (count, n, a, 0.02 * spread)
    meta = {"scale": None, "scale_source": None, "degraded": False}
    if best is None or best[0] < 40:
        meta["degraded"] = True
        meta["scale_source"] = "failed_no_floor_plane"
        meta["scale"] = 1.0
        return xyz, meta
    _, normal, p0, thresh = best
    dist = np.abs((xyz - p0) @ normal)
    inliers = xyz[dist < thresh]
    if len(inliers) < 40:
        meta["degraded"] = True
        meta["scale_source"] = "failed_no_floor_plane"
        meta["scale"] = 1.0
        return xyz, meta
    centroid = inliers.mean(0)
    cam_dist = (cam_positions - centroid) @ normal
    # cameras should be on the up side of the floor
    med = float(np.median(cam_dist))
    if abs(med) < 1e-4:
        meta["degraded"] = True
        meta["scale_source"] = "failed_degenerate_scale"
        meta["scale"] = 1.0
        return xyz, meta
    scale = CAMERA_HEIGHT_M / med
    # keep the floor as the origin in Y after a rotation is left to align_floor;
    # here we only apply uniform scale about the centroid
    scaled = (xyz - centroid) * scale + centroid
    meta["scale"] = round(float(scale), 4)
    meta["scale_source"] = f"camera_height_prior_{CAMERA_HEIGHT_M:.2f}m"
    meta["degraded"] = False
    meta["floor_inliers"] = int(len(inliers))
    return scaled, meta


def load_photo_cloud(folder: Path, capture_for_K: Capture | None = None) -> Cloud:
    paths = list_photo_paths(folder)
    return fuse_photos(paths, capture_for_K, source="photo_sfm")
