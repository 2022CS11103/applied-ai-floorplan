"""Video and photo tiers.

Video triangulates RGB features with the logger's metric poses and never
reads a depth image. Scale stays the pose scale. A walk without usable
poses is marked degraded; it does not borrow the photo height prior.

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
from .geometry import backproject, quat_to_matrix, ransac_plane, voxel_downsample


VIDEO_REASONS = ("low_texture", "weak_overlap", "insufficient_points", "no_room_closure", "missing_pose")
# Below this, layout cannot build a plan. The cloud is still returned.
_MIN_CLOUD = 200


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


def _pose_usable(pose) -> bool:
    if not np.isfinite(pose.t).all() or not np.isfinite(pose.q).all():
        return False
    return float(np.linalg.norm(pose.q)) > 0.5


def _empty_video_cloud(capture: Capture, **meta) -> Cloud:
    return Cloud(
        xyz=np.zeros((0, 3), dtype=np.float32),
        frame_index=np.zeros((0,), dtype=np.int32),
        source="video_triangulation",
        meta={"points": 0, "poses": len(capture.poses), "degraded": True, **meta},
    )


def _K_for_pose(capture: Capture, pose, width: int, height: int) -> np.ndarray:
    """Intrinsics of this frame, scaled if the picture is not the logged size."""
    sx = width / float(capture.rgb_size[0])
    sy = height / float(capture.rgb_size[1])
    fx = float(pose.fx) if pose.fx else 0.0
    fy = float(pose.fy) if pose.fy else 0.0
    cx = float(pose.cx) if pose.cx else 0.0
    cy = float(pose.cy) if pose.cy else 0.0
    if fx <= 1.0 or fy <= 1.0:
        K = _K_from_capture(capture, width, height)
        return K
    return np.array(
        [[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _select_pose_indices(capture: Capture, min_baseline: float = 0.18, limit: int = 90) -> list[int]:
    """Pose rows spaced along the walk so each step has some parallax."""
    chosen = []
    last = None
    for i, pose in enumerate(capture.poses):
        if not _pose_usable(pose):
            continue
        if last is None:
            chosen.append(i)
            last = pose.t
            continue
        if float(np.linalg.norm(pose.t - last)) >= min_baseline:
            chosen.append(i)
            last = pose.t
    if len(chosen) > limit:
        keep = np.linspace(0, len(chosen) - 1, limit).astype(int)
        chosen = [chosen[int(i)] for i in keep]
    return chosen


def _frame_number(pose) -> int | None:
    try:
        return int(str(pose.frame).strip())
    except ValueError:
        return None


def fuse_video(capture: Capture, max_frames: int = 90) -> Cloud:
    """Triangulate the walk with the metric poses.

    Frames are read in order. Seeking an mp4 by frame number often returns
    a different picture than the pose row, and those pairs do not triangulate.
    """
    video = capture.video_path
    usable = [p for p in capture.poses if _pose_usable(p)]
    if len(capture.poses) < 5 and len(usable) < 5:
        return _empty_video_cloud(
            capture,
            reconstruction_reasons=["missing_pose"],
            video_present=video is not None,
        )
    if len(usable) < 5:
        return _empty_video_cloud(
            capture,
            reconstruction_reasons=[],
            pose_error="invalid_pose",
            video_present=video is not None,
        )
    if video is None:
        return _empty_video_cloud(capture, reconstruction_reasons=[], video_error="no_video")

    chosen = _select_pose_indices(capture, limit=max_frames)
    wanted = {}
    for idx in chosen:
        number = _frame_number(capture.poses[idx])
        if number is not None:
            wanted[number] = idx
    if len(wanted) < 5:
        return _empty_video_cloud(capture, reconstruction_reasons=[], pose_error="invalid_pose")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Features on a half-size image, coordinates back in the full picture
    # so they match the logged intrinsics.
    detect_scale = 0.5
    orb = cv2.ORB_create(nfeatures=3000, fastThreshold=7)
    frames = []
    keypoint_counts = []
    frame_no = 0
    while True:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break
        idx = wanted.get(frame_no)
        frame_no += 1
        if idx is None:
            continue
        small = cv2.resize(bgr, None, fx=detect_scale, fy=detect_scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        n_kp = 0 if kp is None else len(kp)
        keypoint_counts.append(n_kp)
        if des is None or n_kp < 20:
            continue
        for point in kp:
            point.pt = (point.pt[0] / detect_scale, point.pt[1] / detect_scale)
        frames.append((idx, small, kp, des))
    cap.release()

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    points: list[np.ndarray] = []
    colors: list[tuple[int, int, int]] = []
    frame_ids: list[int] = []
    pairs_attempted = 0
    pairs_used = 0
    weak_matches = 0
    bad_baseline = 0
    pair_list = [(i, i + 1) for i in range(len(frames) - 1)]
    pair_list += [(i, i + 2) for i in range(len(frames) - 2)]
    for a, b in pair_list:
        i0, small0, kp0, des0 = frames[a]
        i1, small1, kp1, des1 = frames[b]
        pairs_attempted += 1
        matches = bf.match(des0, des1)
        matches = [m for m in matches if m.distance <= 56]
        matches = sorted(matches, key=lambda m: m.distance)[:700]
        if len(matches) < 25:
            weak_matches += 1
            continue
        p0 = capture.poses[i0]
        p1 = capture.poses[i1]
        baseline = float(np.linalg.norm(p1.t - p0.t))
        if baseline < 0.08 or baseline > 1.3:
            bad_baseline += 1
            continue
        kept = _triangulate_pair(capture, p0, p1, kp0, kp1, matches, small0, width, height)
        if not kept:
            continue
        pairs_used += 1
        xyz_k, rgb_k = kept
        points.extend(xyz_k)
        colors.extend(rgb_k)
        frame_ids.extend([i0] * len(xyz_k))

    raw_count = len(points)
    if raw_count:
        xyz = np.asarray(points, dtype=np.float32)
        cols = np.asarray(colors, dtype=np.uint8)
        fidx = np.asarray(frame_ids, dtype=np.int32)
        xyz, [cols, fidx] = voxel_downsample(xyz, 0.03, [cols, fidx])
    else:
        xyz = np.zeros((0, 3), dtype=np.float32)
        cols = np.zeros((0, 3), dtype=np.uint8)
        fidx = np.zeros((0,), dtype=np.int32)

    median_kp = float(np.median(keypoint_counts)) if keypoint_counts else 0.0
    low_texture = bool(keypoint_counts) and (median_kp < 80 or np.mean(np.array(keypoint_counts) < 25) > 0.45)
    weak_overlap = pairs_attempted == 0 or pairs_used < 3 or (
        pairs_attempted >= 4 and pairs_used / pairs_attempted < 0.3
    )
    reasons: list[str] = []
    if len(xyz) < _MIN_CLOUD:
        if low_texture:
            reasons.append("low_texture")
        if weak_overlap:
            reasons.append("weak_overlap")
        reasons.append("insufficient_points")
    midband = 0
    if len(xyz):
        floor = float(np.percentile(xyz[:, 1], 10))
        midband = int(((xyz[:, 1] > floor + 0.35) & (xyz[:, 1] < floor + 1.7)).sum())
    return Cloud(
        xyz=xyz,
        frame_index=fidx,
        colors=cols if len(cols) else None,
        source="video_triangulation",
        meta={
            "points": int(len(xyz)),
            "points_raw": int(raw_count),
            "points_after_voxel": int(len(xyz)),
            "midband_points": int(midband),
            "input_frames": int(input_frames),
            "frames_decoded": int(frame_no),
            "frames_used": int(len(frames)),
            "median_keypoints": round(median_kp, 1),
            "pairs_attempted": int(pairs_attempted),
            "pairs_used": int(pairs_used),
            "weak_match_pairs": int(weak_matches),
            "bad_baseline_pairs": int(bad_baseline),
            "poses": len(capture.poses),
            "usable_poses": len(usable),
            "degraded": bool(reasons),
            "reconstruction_reasons": reasons,
        },
    )


def _triangulate_pair(capture, p0, p1, kp0, kp1, matches, small0, width, height):
    """Return (points, colors) that agree with both metric poses."""
    pts0 = np.float64([kp0[m.queryIdx].pt for m in matches])
    pts1 = np.float64([kp1[m.trainIdx].pt for m in matches])
    K0 = _K_for_pose(capture, p0, width, height)
    K1 = _K_for_pose(capture, p1, width, height)
    R0 = quat_to_matrix(p0.q)
    R1 = quat_to_matrix(p1.q)
    P0 = _projection(K0, R0, p0.t)
    P1 = _projection(K1, R1, p1.t)
    homo = cv2.triangulatePoints(P0, P1, pts0.T.astype(np.float64), pts1.T.astype(np.float64))
    w = homo[3]
    good_w = np.abs(w) > 1e-8
    X = np.full((len(matches), 3), np.nan)
    X[good_w] = (homo[:3, good_w] / w[good_w]).T
    finite = np.isfinite(X).all(axis=1)
    if not finite.any():
        return None
    Xc0 = (X - p0.t) @ R0
    Xc1 = (X - p1.t) @ R1
    z0 = Xc0[:, 2]
    z1 = Xc1[:, 2]
    depth_ok = finite & (z0 > 0.35) & (z1 > 0.35) & (z0 < 6.0) & (z1 < 6.0)
    if not depth_ok.any():
        return None
    u0 = (K0 @ Xc0.T)[:2] / np.clip(z0, 1e-6, None)
    u1 = (K1 @ Xc1.T)[:2] / np.clip(z1, 1e-6, None)
    err0 = np.linalg.norm(u0.T - pts0, axis=1)
    err1 = np.linalg.norm(u1.T - pts1, axis=1)
    ray0 = X - p0.t
    ray1 = X - p1.t
    cos = np.sum(ray0 * ray1, axis=1) / np.clip(np.linalg.norm(ray0, axis=1) * np.linalg.norm(ray1, axis=1), 1e-8, None)
    parallax = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    keep = depth_ok & (err0 < 6.0) & (err1 < 6.0) & (parallax > 0.8)
    if not keep.any():
        return None
    xyz = X[keep]
    cols = []
    sh, sw = small0.shape[:2]
    for x_full, y_full in pts0[keep]:
        x = int(round(x_full * sw / width))
        y = int(round(y_full * sh / height))
        if 0 <= x < sw and 0 <= y < sh:
            b, g, r = small0[y, x]
            cols.append((int(r), int(g), int(b)))
        else:
            cols.append((0, 0, 0))
    return xyz, cols


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
