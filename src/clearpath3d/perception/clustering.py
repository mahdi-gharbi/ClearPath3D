"""Obstacle clustering (DBSCAN) and visible-extent boxes."""

from __future__ import annotations

import numpy as np
import open3d as o3d


def dbscan(points: np.ndarray, eps_m: float, min_points: int) -> np.ndarray:
    """Return labels (-1 = noise). Open3D's DBSCAN is deterministic for a given input."""
    if len(points) == 0:
        return np.zeros(0, dtype=np.int64)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return np.asarray(
        pcd.cluster_dbscan(eps=eps_m, min_points=min_points, print_progress=False), dtype=np.int64
    )


def aabb(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return points.min(axis=0), points.max(axis=0)


def aabb_iou(a_min, a_max, b_min, b_max) -> float:
    a_min, a_max, b_min, b_max = map(np.asarray, (a_min, a_max, b_min, b_max))
    inter = np.clip(np.minimum(a_max, b_max) - np.maximum(a_min, b_min), 0, None).prod()
    va = np.clip(a_max - a_min, 0, None).prod()
    vb = np.clip(b_max - b_min, 0, None).prod()
    union = va + vb - inter
    return float(inter / union) if union > 0 else 0.0
