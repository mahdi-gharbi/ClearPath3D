"""Surface metrics for a reconstruction against the simulator mesh (evaluator side only).

Protocol:
- accuracy: unsigned distance from each reconstructed point to the GT mesh (all static objects at t=0);
- completeness: reference = GT surface points actually *observed* by the integrated frames (clean GT
  depth back-projected, 2 cm voxels); distance from each reference point to the nearest
  reconstructed point. Unobserved surfaces are not counted against the reconstruction.
Thresholds 2 cm and 5 cm. Pose source is reported; with supplied poses this does not evaluate odometry.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from clearpath3d.datasets.base import SensorSequence
from clearpath3d.evaluation.ground_truth import GroundTruth
from clearpath3d.geometry.frames import transform_points
from clearpath3d.geometry.preprocess import voxel_downsample
from clearpath3d.geometry.projection import backproject

THRESHOLDS_M = (0.02, 0.05)


def _dist_stats(d: np.ndarray) -> dict:
    out = {
        "n": int(d.size),
        "mean_m": round(float(d.mean()), 4),
        "median_m": round(float(np.median(d)), 4),
        "p95_m": round(float(np.percentile(d, 95)), 4),
    }
    for t in THRESHOLDS_M:
        out[f"frac_below_{int(t * 100)}cm"] = round(float((d < t).mean()), 4)
    return out


def observed_gt_points(seq_root: Path, frame_ids: list[str], max_depth_m: float) -> np.ndarray:
    seq, gt = SensorSequence(seq_root), GroundTruth(seq_root)
    pts = []
    for fid in frame_ids:
        g = gt.frame(fid)
        valid = (g.depth_clean > 0) & (g.depth_clean <= max_depth_m)
        p, _ = backproject(g.depth_clean, valid, seq.intrinsics)
        pts.append(transform_points(g.T_map_camera, p))
    P = np.concatenate(pts)
    return voxel_downsample(P, np.zeros_like(P), 0.02).points


def evaluate_reconstruction(seq_root: Path | str, run_dir: Path | str) -> dict:
    seq_root, run_dir = Path(seq_root), Path(run_dir)
    rep = json.loads((run_dir / "reconstruction_report.json").read_text())
    gt_mesh = o3d.io.read_triangle_mesh(str(seq_root / "ground_truth" / "scene_mesh_t0.ply"))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(gt_mesh))
    ref = observed_gt_points(
        seq_root, rep["tsdf"]["integrated_frame_ids"], rep["params"]["max_depth_m"]
    )
    result = {"pose_source": rep["pose_source"], "protocol": __doc__.strip().splitlines()[2:6]}
    for key, fname in (
        ("tsdf", "tsdf_cloud.ply"),
        ("tsdf_mesh_vertices", "tsdf_mesh.ply"),
        ("union_baseline", "union_cloud.ply"),
    ):
        if fname.endswith("mesh.ply"):
            pts = np.asarray(o3d.io.read_triangle_mesh(str(run_dir / fname)).vertices)
        else:
            pts = np.asarray(o3d.io.read_point_cloud(str(run_dir / fname)).points)
        acc = scene.compute_distance(o3d.core.Tensor(pts.astype(np.float32))).numpy()
        comp, _ = cKDTree(pts).query(ref, k=1)
        result[key] = {"accuracy": _dist_stats(acc), "completeness": _dist_stats(comp)}
    (run_dir / "reconstruction_eval.json").write_text(json.dumps(result, indent=2))
    return result
