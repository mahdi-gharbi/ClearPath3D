"""Reconstruct a sequence with supplied poses: TSDF fusion and the point-union baseline."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import open3d as o3d

from clearpath3d.datasets.base import SensorSequence
from clearpath3d.geometry.frames import transform_points
from clearpath3d.geometry.preprocess import voxel_downsample
from clearpath3d.geometry.projection import backproject
from clearpath3d.reconstruction.tsdf import KeyframeSelector, TSDFFusion, TSDFParams
from clearpath3d.runtime.profiling import hardware, memory_mb, percentiles


def point_union(seq: SensorSequence, params: TSDFParams) -> tuple[o3d.geometry.PointCloud, dict]:
    """Baseline: concatenate back-projected keyframes in map, then voxel-downsample."""
    kf = KeyframeSelector(params.keyframe_translation_m, params.keyframe_rotation_deg)
    pts, cols, used = [], [], 0
    t0 = time.perf_counter()
    for i in range(len(seq)):
        fr = seq[i]
        if fr.T_map_camera is None or (params.use_keyframes and not kf.accept(fr.T_map_camera)):
            continue
        valid = fr.depth_valid & (fr.depth_m <= params.max_depth_m)
        p, pix = backproject(fr.depth_m, valid, fr.intrinsics)
        pts.append(transform_points(fr.T_map_camera, p))
        cols.append(fr.color_rgb[pix[:, 0], pix[:, 1]] / 255.0)
        used += 1
    P, C = np.concatenate(pts), np.concatenate(cols)
    ds = voxel_downsample(P, C, params.voxel_m)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ds.points))
    pcd.colors = o3d.utility.Vector3dVector(ds.colors)
    return pcd, {
        "frames_used": used,
        "raw_points": int(len(P)),
        "points": int(len(ds.points)),
        "total_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


def reconstruct(seq: SensorSequence, params: TSDFParams, out_dir: Path | str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mem_before = memory_mb()
    fusion = TSDFFusion(seq.intrinsics, params)
    load_ms = []
    t_all = time.perf_counter()
    for i in range(len(seq)):
        t0 = time.perf_counter()
        fr = seq[i]
        load_ms.append((time.perf_counter() - t0) * 1000)
        fusion.integrate(fr)
    mesh, cloud = fusion.extract()
    total_ms = (time.perf_counter() - t_all) * 1000
    mem_after = memory_mb()
    o3d.io.write_triangle_mesh(str(out / "tsdf_mesh.ply"), mesh)
    o3d.io.write_point_cloud(str(out / "tsdf_cloud.ply"), cloud)

    union, union_stats = point_union(seq, params)
    o3d.io.write_point_cloud(str(out / "union_cloud.ply"), union)

    rep = fusion.report
    report = {
        "sequence": seq.sequence_id,
        "pose_source": rep.pose_source,
        "pose_note": "poses supplied by the dataset/simulator; this run does not evaluate odometry",
        "params": asdict(params),
        "tsdf": {
            "frames_seen": rep.frames_seen,
            "frames_integrated": rep.frames_integrated,
            "frames_rejected_no_pose": rep.frames_rejected_no_pose,
            "integrated_frame_ids": rep.integrated_frame_ids,
            "integrate": percentiles(rep.integrate_ms),
            "load": percentiles(load_ms),
            "extract_ms": rep.extract_ms,
            "total_ms": round(total_ms, 1),
            "mesh_vertices": rep.mesh_vertices,
            "mesh_triangles": rep.mesh_triangles,
            "cloud_points": rep.cloud_points,
        },
        "union_baseline": union_stats,
        "memory": {"before": mem_before, "after_tsdf": mem_after, "after_union": memory_mb()},
        "hardware": hardware(),
    }
    (out / "reconstruction_report.json").write_text(json.dumps(report, indent=2))
    return report
