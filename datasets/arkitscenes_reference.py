"""Measure an ARKitScenes capture against its FARO depth.

highres_depth is the laser mesh projected into the wide camera. Wall lengths
taken from that cloud are derived by the same floor-plan extractor that runs
on the phone cloud. They are a phone-versus-laser consistency check. They are
not a published vector plan and they are not laser/tape ground truth.

Nothing here changes fusion, layout, or the assessment thresholds.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from cozmo_scan.fuse import Cloud, align_floor
from cozmo_scan.geometry import backproject, voxel_downsample
from cozmo_scan.layout import build_layout, layout_to_dict
from datasets.arkitscenes_adapter import _traj_poses, _zip_members

LABEL = (
    "EXTERNAL LASER DEPTH — ARKitScenes highres_depth, projected from a FARO mesh. "
    "Lengths below are derived by this repo's layout extractor. Not laser/tape."
)


def _value(measure) -> float | None:
    if measure is None:
        return None
    if isinstance(measure, dict):
        return None if measure.get("value") is None else float(measure["value"])
    return float(measure)


def _wall_lengths(plan_rooms: list[dict]) -> list[float]:
    lengths = []
    for room in plan_rooms:
        for wall in room.get("walls") or []:
            length = _value(wall.get("length_m"))
            if length is not None and length > 0.2:
                lengths.append(length)
    return lengths


def _areas(plan_rooms: list[dict]) -> list[float]:
    areas = []
    for room in plan_rooms:
        area = _value(room.get("floor_area_m2"))
        if area is not None:
            areas.append(area)
    return areas


def _ceilings(plan_rooms: list[dict]) -> list[float]:
    heights = []
    for room in plan_rooms:
        if room.get("ceiling_observed") and room.get("ceiling_height_m") is not None:
            height = _value(room["ceiling_height_m"])
            if height is not None:
                heights.append(height)
    return heights


def height_histogram(xyz: np.ndarray) -> dict:
    """Floor is already at Y=0. The ceiling is the highest sharp peak above 1.8 m.

    This does not use the room extractor. A missing peak stays null.
    """
    heights = xyz[:, 1]
    heights = heights[(heights > -0.3) & (heights < 5.0)]
    if len(heights) < 500:
        return {"ceiling_height_m": None, "field_status": "UNAVAILABLE", "reason": "too few points"}
    bins = np.arange(1.8, 4.5, 0.02)
    counts, edges = np.histogram(heights, bins=bins)
    if counts.max() < 80:
        return {
            "ceiling_height_m": None,
            "field_status": "UNAVAILABLE",
            "reason": "no sharp peak above 1.8 m",
            "points": int(len(heights)),
        }
    center = float(0.5 * (edges[int(np.argmax(counts))] + edges[int(np.argmax(counts)) + 1]))
    return {
        "ceiling_height_m": round(center, 4),
        "field_status": "DERIVED_FROM_DATASET",
        "derivation": "20 mm histogram peak of FARO points above 1.8 m after the floor anchor",
        "peak_count": int(counts.max()),
        "points": int(len(heights)),
    }


def depth_agreement(raw_dir: Path, sample: int = 24) -> dict:
    """Median |Apple depth − FARO depth| on pixels that both cameras saw.

    The FARO image is 1920x1440 and the Apple depth is 256x192, the same wide
    camera at two resolutions. A low-res pixel is compared with the high-res
    pixel at the scaled coordinate. Frames farther than 50 ms apart are skipped.
    """
    raw_dir = Path(raw_dir)
    poses = _traj_poses(raw_dir / "lowres_wide.traj")
    low_members = _zip_members(raw_dir / "lowres_depth.zip")
    high_members = _zip_members(raw_dir / "highres_depth.zip")
    conf_members = _zip_members(raw_dir / "confidence.zip")
    high_keys = np.array(sorted(high_members, key=float), dtype=np.float64)
    common = sorted(set(poses) & set(low_members) & set(conf_members), key=float)
    if len(common) == 0 or len(high_keys) == 0:
        return {"status": "UNAVAILABLE", "reason": "depth or highres_depth missing"}
    step = max(1, len(common) // sample)
    chosen = common[::step][:sample]
    errors = []
    used = 0
    with zipfile.ZipFile(raw_dir / "lowres_depth.zip") as low_zip, zipfile.ZipFile(
        raw_dir / "highres_depth.zip"
    ) as high_zip, zipfile.ZipFile(raw_dir / "confidence.zip") as conf_zip:
        for timestamp in chosen:
            nearest = int(np.argmin(np.abs(high_keys - float(timestamp))))
            if abs(high_keys[nearest] - float(timestamp)) > 0.05:
                continue
            low = np.array(Image.open(io.BytesIO(low_zip.read(low_members[timestamp]))))
            high = np.array(Image.open(io.BytesIO(high_zip.read(high_members[f"{high_keys[nearest]:.3f}"]))))
            conf = np.array(Image.open(io.BytesIO(conf_zip.read(conf_members[timestamp]))))
            if low.ndim != 2 or high.ndim != 2:
                continue
            scale_x = high.shape[1] / float(low.shape[1])
            scale_y = high.shape[0] / float(low.shape[0])
            vs, us = np.nonzero((conf >= 2) & (low > 200) & (low < 5000))
            if len(vs) < 50:
                continue
            pick = np.linspace(0, len(vs) - 1, min(400, len(vs))).astype(int)
            vs, us = vs[pick], us[pick]
            hu = np.clip(np.round(us * scale_x).astype(int), 0, high.shape[1] - 1)
            hv = np.clip(np.round(vs * scale_y).astype(int), 0, high.shape[0] - 1)
            hz = high[hv, hu].astype(np.float64)
            lz = low[vs, us].astype(np.float64)
            ok = (hz > 200) & (hz < 8000)
            if ok.sum() < 30:
                continue
            errors.append(np.abs(lz[ok] - hz[ok]) / 1000.0)
            used += 1
    if not errors:
        return {"status": "UNAVAILABLE", "reason": "no overlapping confident pixels"}
    stacked = np.concatenate(errors)
    return {
        "status": "measured",
        "field_status": "PROVIDED_BY_DATASET",
        "frames": used,
        "pixels": int(len(stacked)),
        "median_abs_m": round(float(np.median(stacked)), 4),
        "p90_abs_m": round(float(np.percentile(stacked, 90)), 4),
        "note": "Apple lowres depth against FARO highres_depth. Not a wall length.",
    }


def faro_points(raw_dir: Path, pixel_stride: int = 8, max_frames: int = 70) -> tuple[np.ndarray, dict]:
    """Backproject a subsample of highres_depth into the Y-up world frame."""
    raw_dir = Path(raw_dir)
    poses = _traj_poses(raw_dir / "lowres_wide.traj")
    pose_times = np.array(sorted(poses, key=float), dtype=np.float64)
    high_members = _zip_members(raw_dir / "highres_depth.zip")
    intrinsic_members = _zip_members(raw_dir / "lowres_wide_intrinsics.zip")
    keys = sorted(high_members, key=float)
    step = max(1, len(keys) // max_frames)
    chosen = keys[::step][:max_frames]
    chunks = []
    used = 0
    skipped = 0
    with zipfile.ZipFile(raw_dir / "highres_depth.zip") as high_zip, zipfile.ZipFile(
        raw_dir / "lowres_wide_intrinsics.zip"
    ) as intrinsic_zip:
        for timestamp in chosen:
            nearest = int(np.argmin(np.abs(pose_times - float(timestamp))))
            if abs(pose_times[nearest] - float(timestamp)) > 0.05:
                skipped += 1
                continue
            pose_key = f"{pose_times[nearest]:.3f}"
            if pose_key not in intrinsic_members:
                skipped += 1
                continue
            depth = np.array(Image.open(io.BytesIO(high_zip.read(high_members[timestamp]))))
            if depth.ndim != 2:
                continue
            width, height, fx, fy, cx, cy = _pincam(intrinsic_zip, intrinsic_members[pose_key])
            scale_x = depth.shape[1] / float(width)
            scale_y = depth.shape[0] / float(height)
            z = depth.astype(np.float32) / 1000.0
            mask = (z > 0.3) & (z < 6.0)
            vs, us = np.nonzero(mask)
            vs = vs[::pixel_stride]
            us = us[::pixel_stride]
            if len(vs) < 40:
                continue
            zz = z[vs, us].astype(np.float64)
            cam = backproject(us.astype(np.float64), vs.astype(np.float64), zz, fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y)
            rotation, center = poses[pose_key]
            world = cam @ rotation.T + center
            chunks.append(world.astype(np.float32))
            used += 1
            if len(chunks) >= 8:
                chunks = [voxel_downsample(np.concatenate(chunks, axis=0), 0.03)]
    if not chunks:
        raise RuntimeError(f"no FARO points from {raw_dir}")
    points = voxel_downsample(np.concatenate(chunks, axis=0), 0.03)
    return points, {
        "frames_used": used,
        "frames_skipped_no_pose": skipped,
        "points": int(len(points)),
        "field_status": "DERIVED_FROM_DATASET",
        "derivation": "highres_depth backprojected with the lowres pincam scaled by the image-size ratio, then a 3 cm voxel",
        "assumption": "highres_depth is the same wide camera as lowres_wide, so intrinsics scale with resolution. Apple does not ship a separate wide pincam in the raw asset list.",
    }


def _pincam(archive: zipfile.ZipFile, member: str) -> tuple[int, int, float, float, float, float]:
    width, height, fx, fy, cx, cy = archive.read(member).decode().split()
    return int(float(width)), int(float(height)), float(fx), float(fy), float(cx), float(cy)


def layout_on_points(points: np.ndarray) -> dict:
    cloud = Cloud(xyz=points.astype(np.float32), frame_index=np.arange(len(points), dtype=np.int32), source="faro_depth")
    aligned = align_floor(cloud, enable_anchor=True, enable_loop=False)
    layout = build_layout(aligned["xyz"], tier="lidar")
    document = layout_to_dict(layout, tier="lidar")
    document["floor_tilt_deg"] = aligned["report"].get("floor_tilt_deg")
    document["points_aligned"] = int(len(aligned["xyz"]))
    return document


def _greedy_lengths(predicted: list[float], reference: list[float]) -> list[dict]:
    remaining = reference[:]
    rows = []
    for length in sorted(predicted, reverse=True):
        if not remaining:
            rows.append({"predicted_m": round(length, 4), "reference_m": None, "status": "unmatched"})
            continue
        index = int(np.argmin([abs(length - other) for other in remaining]))
        other = remaining.pop(index)
        error = abs(length - other)
        rows.append(
            {
                "predicted_m": round(length, 4),
                "reference_m": round(other, 4),
                "absolute_error_m": round(error, 4),
                "relative_error": round(error / other, 4) if other else None,
            }
        )
    for other in remaining:
        rows.append({"predicted_m": None, "reference_m": round(other, 4), "status": "reference_only"})
    return rows


def compare_plan(plan: dict, faro_layout: dict, depth: dict, heights: dict, cloud_meta: dict) -> dict:
    predicted_rooms = plan.get("property", {}).get("rooms") or []
    reference_rooms = faro_layout.get("rooms") or []
    pred_lengths = _wall_lengths(predicted_rooms)
    ref_lengths = _wall_lengths(reference_rooms)
    pred_areas = _areas(predicted_rooms)
    ref_areas = _areas(reference_rooms)
    pred_ceilings = _ceilings(predicted_rooms)
    ref_ceilings = _ceilings(reference_rooms)
    rows = _greedy_lengths(pred_lengths, ref_lengths)
    matched = [row for row in rows if row.get("absolute_error_m") is not None]
    ceiling_rows = []
    if heights.get("ceiling_height_m") is not None:
        for height in pred_ceilings:
            error = abs(height - heights["ceiling_height_m"])
            ceiling_rows.append(
                {
                    "predicted_m": round(height, 4),
                    "faro_histogram_m": heights["ceiling_height_m"],
                    "absolute_error_m": round(error, 4),
                    "within_1_5_cm": error <= 0.015,
                }
            )
    return {
        "label": LABEL,
        "video_id": plan.get("capture_id"),
        "pipeline_status": plan.get("status"),
        "assessment_gate": {
            "status": "blocked",
            "reason": (
                "FARO highres_depth is laser depth, not a tape of each wall, opening, and ceiling. "
                "The assessment gate stays blocked."
            ),
        },
        "phone_rooms": len(predicted_rooms),
        "faro_layout_rooms": len(reference_rooms),
        "phone_wall_lengths_m": [round(v, 4) for v in pred_lengths],
        "faro_layout_wall_lengths_m": [round(v, 4) for v in ref_lengths],
        "wall_match": {
            "field_status": "DERIVED_FROM_DATASET",
            "derivation": "same build_layout on the phone cloud and on the backprojected FARO depth",
            "rows": rows,
            "median_abs_error_m": None if not matched else round(float(np.median([r["absolute_error_m"] for r in matched])), 4),
        },
        "floor_area_m2": {
            "field_status": "DERIVED_FROM_DATASET",
            "phone_sum": round(float(sum(pred_areas)), 4) if pred_areas else None,
            "faro_layout_sum": round(float(sum(ref_areas)), 4) if ref_areas else None,
            "phone_rooms": [round(v, 4) for v in pred_areas],
            "faro_rooms": [round(v, 4) for v in ref_areas],
        },
        "ceiling": {
            "phone_m": [round(v, 4) for v in pred_ceilings],
            "faro_layout_m": [round(v, 4) for v in ref_ceilings],
            "faro_histogram": heights,
            "rows": ceiling_rows,
        },
        "depth_agreement": depth,
        "faro_cloud": cloud_meta,
        "phone_floor_tilt_deg": (plan.get("drift") or {}).get("floor_tilt_deg"),
        "faro_floor_tilt_deg": faro_layout.get("floor_tilt_deg"),
        "openings": {
            "field_status": "UNAVAILABLE",
            "reason": "ARKitScenes 3DOD labels are furniture boxes, not door or window widths.",
        },
        "imu": {"field_status": "UNAVAILABLE"},
        "repeat_capture": {"field_status": "see visit pair when a second video of the same visit was run"},
    }


def repeatability(plan_a: dict, plan_b: dict) -> dict:
    lengths_a = _wall_lengths(plan_a.get("property", {}).get("rooms") or [])
    lengths_b = _wall_lengths(plan_b.get("property", {}).get("rooms") or [])
    rows = _greedy_lengths(lengths_a, lengths_b)
    matched = [row for row in rows if row.get("absolute_error_m") is not None]
    return {
        "field_status": "DERIVED_FROM_DATASET",
        "derivation": "two Apple walks of one visit, each run through the unchanged LiDAR pipeline",
        "video_a": plan_a.get("capture_id"),
        "video_b": plan_b.get("capture_id"),
        "rooms_a": len(plan_a.get("property", {}).get("rooms") or []),
        "rooms_b": len(plan_b.get("property", {}).get("rooms") or []),
        "rows": rows,
        "median_abs_error_m": None if not matched else round(float(np.median([r["absolute_error_m"] for r in matched])), 4),
        "assessment_threshold": "<= 1 cm or 0.5% per wall. This pair is not the assessment protocol and is not tape.",
        "assessment_gate": "blocked",
    }


def write_case(raw_dir: Path, plan_path: Path, out_dir: Path) -> dict:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    points, cloud_meta = faro_points(raw_dir)
    faro_layout = layout_on_points(points)
    cloud = Cloud(xyz=points.astype(np.float32), frame_index=np.arange(len(points), dtype=np.int32), source="faro_depth")
    aligned = align_floor(cloud, enable_anchor=True, enable_loop=False)
    heights = height_histogram(aligned["xyz"])
    depth = depth_agreement(raw_dir)
    report = compare_plan(plan, faro_layout, depth, heights, cloud_meta)
    report["faro_layout_notes"] = faro_layout.get("notes") or []
    report["phone_notes"] = plan.get("notes") or []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "EXTERNAL.txt").write_text(
        "ARKitScenes real Apple LiDAR compared with FARO highres_depth.\n"
        "Wall lengths on the FARO side are derived by this repo's layout code.\n"
        "That is not a tape, and the assessment gate stays blocked.\n"
        f"Rebuild from {raw_dir} and {plan_path}.\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = write_case(args.raw, args.plan, args.out)
    print(
        f"rooms phone={report['phone_rooms']} faro={report['faro_layout_rooms']} "
        f"depth_median_m={report['depth_agreement'].get('median_abs_m')}"
    )


if __name__ == "__main__":
    main()
