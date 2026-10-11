"""Single-frame analysis (M1): depth -> points -> ground -> obstacles -> snapshot + evidence.

Reads only an ``RGBDFrame`` (sensor data + calibration + optional supplied pose).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from clearpath3d.config import PipelineConfig
from clearpath3d.contracts import (
    GroundPlane,
    GroundStatus,
    ObstacleDetection,
    PoseSource,
    RGBDFrame,
    SceneSnapshot,
)
from clearpath3d.geometry.frames import compose, invert, transform_points
from clearpath3d.geometry.ground import GroundParams, detect_ground
from clearpath3d.geometry.preprocess import (
    estimate_normals,
    remove_statistical_outliers,
    voxel_downsample,
)
from clearpath3d.geometry.projection import backproject
from clearpath3d.perception.clustering import aabb, dbscan

LABEL_OTHER, LABEL_GROUND, LABEL_OBSTACLE, LABEL_OUTLIER = 0, 1, 2, 3
GROUND_NORMAL_MAX_DEG = 30.0  # ground inliers must also have a roughly plane-parallel normal


@dataclass
class FrameAnalysis:
    snapshot: SceneSnapshot
    points_out: np.ndarray  # [M,3] in snapshot.output_frame
    points_base: np.ndarray  # [M,3] in base_link
    colors: np.ndarray  # [M,3] in [0,1]
    labels: np.ndarray  # [M] LABEL_*
    cluster: np.ndarray  # [M] detection id or -1
    pixel: np.ndarray  # [M,2] (v, u) of a representative source pixel
    explanation: dict


def analyze_frame(frame: RGBDFrame, cfg: PipelineConfig) -> FrameAnalysis:
    timings: dict[str, float] = {}
    warnings: list[str] = []
    t0 = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal t0
        t = time.perf_counter()
        timings[name] = round((t - t0) * 1000.0, 3)
        t0 = t

    n_valid = int(frame.depth_valid.sum())
    n_pix = frame.depth_valid.size
    if n_valid == 0:
        warnings.append("frame has no valid depth: nothing observed")

    pts_cam, pix = backproject(frame.depth_m, frame.depth_valid, frame.intrinsics)
    cols = (
        frame.color_rgb[pix[:, 0], pix[:, 1]].astype(np.float64) / 255.0
        if len(pix)
        else np.zeros((0, 3))
    )
    lap("backproject")

    if frame.T_base_camera is None:
        raise ValueError("T_base_camera calibration is required for the gravity prior")
    T_base_camera = frame.T_base_camera
    pts_base_full = transform_points(T_base_camera, pts_cam)

    ds = voxel_downsample(pts_base_full, cols, cfg.preprocess.voxel_m)
    pts_base, colors, pixel = ds.points, ds.colors, pix[ds.source_index] if len(pix) else pix
    lap("voxel_downsample")

    labels = np.full(len(pts_base), LABEL_OTHER, dtype=np.int8)
    if cfg.preprocess.outlier_removal and len(pts_base):
        keep = remove_statistical_outliers(
            pts_base, cfg.preprocess.outlier_nb_neighbors, cfg.preprocess.outlier_std_ratio
        )
        labels[~keep] = LABEL_OUTLIER
    lap("outliers")

    inlier_idx = np.nonzero(labels != LABEL_OUTLIER)[0]
    work = pts_base[inlier_idx]
    normals = (
        estimate_normals(work, cfg.preprocess.normals_radius_factor * cfg.preprocess.voxel_m)
        if len(work) >= 3
        else np.zeros((len(work), 3))
    )
    lap("normals")

    g = cfg.ground
    gres = detect_ground(
        work,
        GroundParams(
            distance_threshold_m=g.distance_threshold_m,
            max_tilt_deg=g.max_tilt_deg,
            expected_height_m=0.0,
            height_tolerance_m=g.height_tolerance_m,
            min_inliers=g.min_inliers,
            min_support_area_m2=g.min_support_area_m2,
            area_cell_m=g.area_cell_m,
            max_candidates=g.max_candidates,
            ransac_iterations=g.ransac_iterations,
            seed=g.seed,
        ),
    )
    lap("ground")

    plane = gres.plane
    if plane is not None:
        cos_n = np.abs(normals @ plane.normal)
        gmask = gres.inlier_mask & (cos_n >= np.cos(np.radians(GROUND_NORMAL_MAX_DEG)))
        labels[inlier_idx[gmask]] = LABEL_GROUND
        height = work @ plane.normal + plane.offset  # signed height above the plane
        ground_status = GroundStatus.found
    else:
        gmask = np.zeros(len(work), dtype=bool)
        height = work[:, 2]  # relative to the calibrated base frame only
        ground_status = GroundStatus.unavailable
        reasons = (
            "; ".join(f"cand{i}: {', '.join(c.reasons)}" for i, c in enumerate(gres.candidates))
            or "no planar candidate"
        )
        warnings.append(
            "ground unavailable: no traversability is asserted; obstacle heights are relative to "
            f"the calibrated base frame ({reasons})"
        )

    # Obstacle candidates: non-ground points inside the robot-relevant height band and range.
    cam_xy = T_base_camera[:2, 3]
    horiz = np.linalg.norm(work[:, :2] - cam_xy, axis=1)
    o = cfg.obstacles
    cand = (
        (~gmask)
        & (height >= o.min_height_m)
        & (height <= o.max_height_m)
        & (horiz <= o.max_range_m)
    )
    if plane is not None:
        below = int(((~gmask) & (height < -g.distance_threshold_m * 2)).sum())
        if below:
            warnings.append(
                f"{below} points lie below the ground plane (holes, noise or a step down)"
            )
    cand_idx = np.nonzero(cand)[0]
    cl = dbscan(work[cand_idx], o.dbscan_eps_m, o.dbscan_min_points)
    lap("cluster")

    # Output frame: map when a pose is available, otherwise base_link.
    if frame.T_map_camera is not None:
        T_out_base = compose(frame.T_map_camera, invert(T_base_camera))
        out_frame = "map"
    else:
        T_out_base = np.eye(4)
        out_frame = "base_link"
    pts_out = transform_points(T_out_base, pts_base)

    cluster = np.full(len(pts_base), -1, dtype=np.int32)
    detections: list[ObstacleDetection] = []
    for lab in sorted(set(cl.tolist()) - {-1}):
        members = inlier_idx[cand_idx[cl == lab]]
        if len(members) < o.min_cluster_points:
            continue
        did = len(detections)
        cluster[members] = did
        labels[members] = LABEL_OBSTACLE
        lo, hi = aabb(pts_out[members])
        detections.append(
            ObstacleDetection(
                detection_id=did,
                frame_id=out_frame,
                center_m=((lo + hi) / 2).round(4).tolist(),
                visible_extent_m=(hi - lo).round(4).tolist(),
                min_m=lo.round(4).tolist(),
                max_m=hi.round(4).tolist(),
                point_count=int(len(members)),
            )
        )
    lap("boxes")

    ground_plane = None
    if plane is not None:
        T_base_out = invert(T_out_base)
        n_out = T_base_out[:3, :3].T @ plane.normal
        d_out = float(plane.normal @ T_base_out[:3, 3] + plane.offset)
        ground_plane = GroundPlane(
            frame_id=out_frame,
            normal=n_out.round(6).tolist(),
            offset=round(d_out, 6),
            inlier_count=int((labels == LABEL_GROUND).sum()),
            residual_m=round(plane.residual_m, 5),
            support_area_m2=round(plane.support_area_m2, 4),
            tilt_deg=round(plane.tilt_deg, 3),
        )

    snapshot = SceneSnapshot(
        sequence_id=frame.sequence_id,
        frame_id=frame.frame_id,
        timestamp_s=frame.timestamp_s,
        config_hash=cfg.config_hash(),
        pose_status=frame.pose_source,
        ground_status=ground_status,
        output_frame=out_frame,
        ground_plane=ground_plane,
        detections=detections,
        warnings=warnings,
        timings_ms=timings,
        stats={
            "valid_depth_fraction": round(n_valid / n_pix, 5),
            "points_backprojected": float(len(pts_cam)),
            "points_downsampled": float(len(pts_base)),
            "points_outlier": float((labels == LABEL_OUTLIER).sum()),
            "points_ground": float((labels == LABEL_GROUND).sum()),
            "points_obstacle": float((labels == LABEL_OBSTACLE).sum()),
        },
    )
    if frame.pose_source == PoseSource.unavailable:
        snapshot.warnings.append("pose unavailable: results expressed in base_link, not in map")
    explanation = {
        "ground_candidates": [c.summary() for c in gres.candidates],
        "ground_selection_rule": (
            "accepted = tilt<=max_tilt AND |height-expected|<=tol AND inliers>=min AND area>=min; "
            "best accepted = most inliers (the largest plane overall is NOT preferred)"
        ),
        "gravity_prior": "base_link +z via T_base_camera (assumes a level robot base)",
        "pose_source": frame.pose_source.value,
        "calibration_id": frame.calibration_id,
    }
    return FrameAnalysis(snapshot, pts_out, pts_base, colors, labels, cluster, pixel, explanation)


_PALETTE = np.array(
    [
        [0.90, 0.10, 0.10],
        [0.10, 0.45, 0.90],
        [0.95, 0.60, 0.05],
        [0.60, 0.20, 0.80],
        [0.05, 0.75, 0.75],
        [0.90, 0.20, 0.60],
        [0.55, 0.55, 0.10],
        [0.30, 0.30, 0.95],
    ]
)


def label_colors(labels: np.ndarray, cluster: np.ndarray) -> np.ndarray:
    c = np.full((len(labels), 3), 0.55)
    c[labels == LABEL_GROUND] = [0.15, 0.75, 0.25]
    c[labels == LABEL_OUTLIER] = [0.2, 0.2, 0.2]
    obs = labels == LABEL_OBSTACLE
    c[obs] = _PALETTE[cluster[obs] % len(_PALETTE)]
    return c


def export_analysis(fa: FrameAnalysis, out_dir: Path | str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "snapshot.json").write_text(fa.snapshot.model_dump_json(indent=2))
    (out / "explanation.json").write_text(json.dumps(fa.explanation, indent=2))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(fa.points_out))
    pcd.colors = o3d.utility.Vector3dVector(label_colors(fa.labels, fa.cluster))
    o3d.io.write_point_cloud(str(out / "points_labelled.ply"), pcd)
    rgb = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(fa.points_out))
    rgb.colors = o3d.utility.Vector3dVector(fa.colors)
    o3d.io.write_point_cloud(str(out / "points_rgb.ply"), rgb)
    np.savez_compressed(
        out / "points.npz",
        points_out=fa.points_out,
        points_base=fa.points_base,
        labels=fa.labels,
        cluster=fa.cluster,
        pixel=fa.pixel,
    )
    return out
