"""Read a Cozmo / ARKit-style capture bundle.

Expected layout (what the sample data uses)::

    <capture_id>/
        camera_matrix.csv     3x3 RGB intrinsics
        odometry.csv           metric camera poses, one row per RGB frame
        imu.csv                accelerometer in g, gyro in rad/s
        rgb.mp4                1920x1440, frame index == odometry frame
        depth/NNNNNN.png       uint16 millimetres, 256x192
        confidence/NNNNNN.png  0 low, 1 medium, 2 high

Photo tier is different: a folder of images, optionally with adjacency.json
one level up.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


DEPTH_WH = (256, 192)  # width, height of the LiDAR depth image
RGB_WH = (1920, 1440)


@dataclass
class Pose:
    frame: str
    timestamp: float
    t: np.ndarray  # camera position in world, meters
    q: np.ndarray  # xyzw quaternion, camera-to-world
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class Capture:
    root: Path
    capture_id: str
    kind: str  # lidar_bundle | video | photo_folder
    poses: list[Pose] = field(default_factory=list)
    rgb_size: tuple[int, int] = RGB_WH
    K_rgb: np.ndarray | None = None

    @property
    def depth_dir(self) -> Path:
        return self.root / "depth"

    @property
    def confidence_dir(self) -> Path:
        return self.root / "confidence"

    @property
    def video_path(self) -> Path | None:
        for name in ("rgb.mp4", "video.mp4", "walkthrough.mp4"):
            p = self.root / name
            if p.exists():
                return p
        videos = sorted(self.root.glob("*.mp4"))
        return videos[0] if videos else None

    def pose_by_frame(self) -> dict[str, Pose]:
        return {p.frame: p for p in self.poses}


def _open_csv(path: Path):
    # The sample files put a space after every comma.
    f = path.open(newline="")
    return csv.DictReader(f, skipinitialspace=True)


def load_intrinsics(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([float(x) for x in line.split(",")])
    K = np.array(rows, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"camera matrix at {path} is {K.shape}, expected 3x3")
    return K


def load_poses(path: Path, K_fallback: np.ndarray | None) -> list[Pose]:
    poses = []
    fx0 = float(K_fallback[0, 0]) if K_fallback is not None else 0.0
    fy0 = float(K_fallback[1, 1]) if K_fallback is not None else 0.0
    cx0 = float(K_fallback[0, 2]) if K_fallback is not None else 0.0
    cy0 = float(K_fallback[1, 2]) if K_fallback is not None else 0.0
    with path.open(newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            fx = float(row["fx"]) if row.get("fx") else fx0
            fy = float(row["fy"]) if row.get("fy") else fy0
            cx = float(row["cx"]) if row.get("cx") else cx0
            cy = float(row["cy"]) if row.get("cy") else cy0
            poses.append(
                Pose(
                    frame=row["frame"],
                    timestamp=float(row["timestamp"]),
                    t=np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64),
                    q=np.array(
                        [float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])],
                        dtype=np.float64,
                    ),
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                )
            )
    return poses


def imu_gravity_g(path: Path) -> float | None:
    """Median specific-force magnitude. A still phone reads about 1 g."""
    if not path.exists():
        return None
    norms = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for i, row in enumerate(reader):
            if i > 4000:
                break
            row = {k.strip(): v.strip() for k, v in row.items()}
            a = np.array([float(row["a_x"]), float(row["a_y"]), float(row["a_z"])])
            norms.append(float(np.linalg.norm(a)))
    if not norms:
        return None
    return float(np.median(norms))


def discover(path: Path) -> Capture:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    # A property sketch folder: contains room subfolders of images.
    if (path / "adjacency.json").exists() and not (path / "depth").exists():
        return Capture(root=path, capture_id=path.name, kind="photo_property")

    images = [p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}] if path.is_dir() else []
    has_depth = (path / "depth").is_dir() and any((path / "depth").glob("*.png"))
    has_odom = (path / "odometry.csv").exists()
    has_video = any(path.glob("*.mp4")) if path.is_dir() else path.suffix.lower() == ".mp4"

    if path.is_file() and path.suffix.lower() == ".mp4":
        return Capture(root=path.parent, capture_id=path.stem, kind="video")

    K = None
    if (path / "camera_matrix.csv").exists():
        K = load_intrinsics(path / "camera_matrix.csv")

    poses: list[Pose] = []
    if has_odom:
        poses = load_poses(path / "odometry.csv", K)

    if has_depth and has_odom:
        kind = "lidar_bundle"
    elif has_video or has_odom:
        kind = "video"
    elif len(images) >= 2:
        kind = "photo_folder"
    else:
        raise ValueError(
            f"{path} is not a capture. Need depth+odometry, a video, or a folder of stills."
        )

    return Capture(
        root=path,
        capture_id=path.name,
        kind=kind,
        poses=poses,
        K_rgb=K,
    )


def list_photo_paths(folder: Path) -> list[Path]:
    files = [
        p
        for p in sorted(folder.iterdir())
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and not p.name.startswith(".")
    ]
    return files
