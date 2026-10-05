"""Convert one ARKitScenes raw capture into the LiDAR bundle this repo already reads.

The bundle is real Apple LiDAR: uint16 depth in millimetres, confidence 0–2,
per-frame pinhole intrinsics, and a metric trajectory. It is not a vector
floor plan. FARO data, when present as highres_depth, is a depth image
projected from the laser mesh. It is not a tape measurement of a wall.

ARKitScenes stores the trajectory as a world-to-camera axis-angle pose, and
its 3D-object code treats world Z as the vertical axis. This adapter inverts
that pose the same way Apple's TrajStringToMatrix does, then swaps Z-up into
the Y-up frame the capture reader expects. The swap is a fixed change of
axes, not a floor fit.
"""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

LABEL = (
    "REAL DATA — ARKitScenes Apple LiDAR capture. "
    "FARO highres_depth is laser depth in the camera, not a tape schedule of walls."
)
SOURCE = "https://github.com/apple/ARKitScenes"
# Right-handed map taking ARKitScenes Z-up world onto this repo's Y-up world.
_Z_UP_TO_Y_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64)
# fuse_lidar scales pose intrinsics by depth_size / rgb_size, and the reader
# assumes the RGB slot is 1920x1440. Store K in that slot so the scale lands
# back on the published pincam.
_PIPELINE_RGB_WH = (1920, 1440)


def camera_to_world(traj_line: str) -> tuple[str, np.ndarray, np.ndarray]:
    """Return timestamp key, camera-to-world rotation, and camera center.

    Matches threedod/benchmark_scripts/utils/tenFpsDataLoader.TrajStringToMatrix:
    the file stores a world-to-camera pose, and the camera-to-world pose is
    its inverse. The returned rotation is already in the Y-up frame.
    """
    tokens = traj_line.split()
    if len(tokens) != 7:
        raise ValueError(f"traj line has {len(tokens)} columns, expected 7")
    timestamp = f"{round(float(tokens[0]), 3):.3f}"
    world_to_camera = Rotation.from_rotvec(np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])])).as_matrix()
    translation = np.array([float(tokens[4]), float(tokens[5]), float(tokens[6])], dtype=np.float64)
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = world_to_camera
    extrinsics[:3, 3] = translation
    camera_to_world_z_up = np.linalg.inv(extrinsics)
    rotation = _Z_UP_TO_Y_UP @ camera_to_world_z_up[:3, :3]
    center = _Z_UP_TO_Y_UP @ camera_to_world_z_up[:3, 3]
    return timestamp, rotation, center


def scale_intrinsics_to_rgb_slot(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    src_wh: tuple[int, int],
    rgb_wh: tuple[int, int] = _PIPELINE_RGB_WH,
) -> tuple[float, float, float, float]:
    """Scale a pincam so fuse_lidar's depth/rgb ratio recovers it."""
    sx = rgb_wh[0] / float(src_wh[0])
    sy = rgb_wh[1] / float(src_wh[1])
    return fx * sx, fy * sy, cx * sx, cy * sy


def _traj_poses(traj_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    poses = {}
    for line in traj_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        timestamp, rotation, center = camera_to_world(line)
        poses[timestamp] = (rotation, center)
    if not poses:
        raise ValueError(f"no poses in {traj_path}")
    return poses


def _zip_members(path: Path) -> dict[str, str]:
    """Map a 3-decimal timestamp to the member name inside a zip."""
    found = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            stem = Path(name).stem
            if "_" not in stem:
                continue
            timestamp = stem.split("_", 1)[1]
            try:
                key = f"{round(float(timestamp), 3):.3f}"
            except ValueError:
                continue
            found[key] = name
    return found


def _read_pincam(archive: zipfile.ZipFile, member: str) -> tuple[int, int, float, float, float, float]:
    width, height, fx, fy, cx, cy = archive.read(member).decode().split()
    return int(float(width)), int(float(height)), float(fx), float(fy), float(cx), float(cy)


def convert_capture(raw_dir: Path, out_dir: Path, video_id: str | None = None) -> dict:
    """Write depth/, confidence/, odometry.csv, and camera_matrix.csv.

    Only frames that have a trajectory pose are copied. A 60 fps depth image
    with no pose is not given an interpolated pose.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    video_id = video_id or raw_dir.name
    poses = _traj_poses(raw_dir / "lowres_wide.traj")
    depth_members = _zip_members(raw_dir / "lowres_depth.zip")
    confidence_members = _zip_members(raw_dir / "confidence.zip")
    intrinsic_members = _zip_members(raw_dir / "lowres_wide_intrinsics.zip")
    frames = sorted(set(poses) & set(depth_members) & set(confidence_members) & set(intrinsic_members), key=float)
    if len(frames) < 10:
        raise RuntimeError(f"{video_id} has {len(frames)} frames with depth, confidence, intrinsics, and a pose")

    depth_out = out_dir / "depth"
    confidence_out = out_dir / "confidence"
    depth_out.mkdir(parents=True, exist_ok=True)
    confidence_out.mkdir(parents=True, exist_ok=True)

    rows = []
    first_k = None
    pincam_wh = None
    with zipfile.ZipFile(raw_dir / "lowres_depth.zip") as depth_zip, zipfile.ZipFile(
        raw_dir / "confidence.zip"
    ) as confidence_zip, zipfile.ZipFile(raw_dir / "lowres_wide_intrinsics.zip") as intrinsic_zip:
        for timestamp in frames:
            (depth_out / f"{timestamp}.png").write_bytes(depth_zip.read(depth_members[timestamp]))
            (confidence_out / f"{timestamp}.png").write_bytes(confidence_zip.read(confidence_members[timestamp]))
            width, height, fx, fy, cx, cy = _read_pincam(intrinsic_zip, intrinsic_members[timestamp])
            pincam_wh = (width, height)
            fx_s, fy_s, cx_s, cy_s = scale_intrinsics_to_rgb_slot(fx, fy, cx, cy, (width, height))
            rotation, center = poses[timestamp]
            quat = Rotation.from_matrix(rotation).as_quat()
            if first_k is None:
                first_k = (fx_s, fy_s, cx_s, cy_s)
            rows.append(
                {
                    "frame": timestamp,
                    "timestamp": timestamp,
                    "x": f"{center[0]:.6f}",
                    "y": f"{center[1]:.6f}",
                    "z": f"{center[2]:.6f}",
                    "qx": f"{quat[0]:.8f}",
                    "qy": f"{quat[1]:.8f}",
                    "qz": f"{quat[2]:.8f}",
                    "qw": f"{quat[3]:.8f}",
                    "fx": f"{fx_s:.6f}",
                    "fy": f"{fy_s:.6f}",
                    "cx": f"{cx_s:.6f}",
                    "cy": f"{cy_s:.6f}",
                }
            )

    with (out_dir / "odometry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame", "timestamp", "x", "y", "z", "qx", "qy", "qz", "qw", "fx", "fy", "cx", "cy"],
        )
        writer.writeheader()
        writer.writerows(rows)
    fx_s, fy_s, cx_s, cy_s = first_k
    (out_dir / "camera_matrix.csv").write_text(
        f"{fx_s:.6f}, 0, {cx_s:.6f}\n0, {fy_s:.6f}, {cy_s:.6f}\n0, 0, 1\n",
        encoding="utf-8",
    )
    meta = {
        "label": LABEL,
        "source": SOURCE,
        "video_id": video_id,
        "frames_with_pose": len(frames),
        "traj_poses": len(poses),
        "depth_files_in_zip": len(depth_members),
        "pincam_wh": list(pincam_wh),
        "intrinsics_stored_for_rgb_wh": list(_PIPELINE_RGB_WH),
        "intrinsics_note": (
            "odometry fx/fy/cx/cy are the published pincam scaled into the 1920x1440 slot "
            "so the existing depth/rgb scale recovers the 256x192 pincam. They are not a second calibration."
        ),
        "world_frame": "ARKitScenes Z-up rotated to Y-up by a fixed axis swap",
        "field_status": {
            "depth": "PROVIDED_BY_DATASET",
            "confidence": "PROVIDED_BY_DATASET",
            "camera_pose": "PROVIDED_BY_DATASET",
            "intrinsics": "PROVIDED_BY_DATASET",
            "imu": "UNAVAILABLE",
            "rgb_video": "UNAVAILABLE",
            "vector_floor_plan": "UNAVAILABLE",
            "laser_tape_lengths": "UNAVAILABLE",
        },
    }
    (out_dir / "arkitscenes_source.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video-id")
    args = parser.parse_args()
    meta = convert_capture(args.raw, args.out, args.video_id)
    print(f"wrote {args.out} frames={meta['frames_with_pose']}")


if __name__ == "__main__":
    main()
