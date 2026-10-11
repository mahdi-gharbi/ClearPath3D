"""Depth sanitisation and point-cloud preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


class DepthUnitError(ValueError):
    pass


def valid_depth_mask(depth_m: np.ndarray, min_depth_m: float, max_depth_m: float) -> np.ndarray:
    """True only for finite depths inside [min, max]. 0 / NaN / inf / out-of-range are *no data*."""
    d = np.asarray(depth_m)
    with np.errstate(invalid="ignore"):
        return np.isfinite(d) & (d >= min_depth_m) & (d <= max_depth_m)


def check_depth_plausibility(
    depth_m: np.ndarray, plausible_median_m: tuple[float, float] = (0.1, 20.0)
) -> float:
    """Catch unit errors (e.g. millimetres read as metres). Returns the median finite positive depth."""
    d = np.asarray(depth_m, dtype=np.float64)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        raise DepthUnitError("depth image has no finite positive value")
    med = float(np.median(d))
    lo, hi = plausible_median_m
    if not lo <= med <= hi:
        hint = " (looks like millimetres; check depth_scale)" if med > 100 else ""
        raise DepthUnitError(
            f"median depth {med:.3f} m outside plausible range {plausible_median_m}{hint}"
        )
    return med


@dataclass
class DownsampleResult:
    points: np.ndarray  # [M, 3]
    colors: np.ndarray  # [M, 3] float in [0, 1]
    source_index: np.ndarray  # [M] index into the input points of one representative point


def voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_m: float) -> DownsampleResult:
    """Voxel-grid downsampling that keeps, per voxel, the index of the point nearest the centroid.

    The kept index preserves the link back to pixels (needed for evidence and evaluation).
    """
    if len(points) == 0 or voxel_m <= 0:
        return DownsampleResult(points.copy(), colors.copy(), np.arange(len(points)))
    keys = np.floor(points / voxel_m).astype(np.int64)
    keys -= keys.min(axis=0)
    span = keys.max(axis=0) + 1
    flat = (keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2]  # 1-D key: much faster unique
    _, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    n = len(counts)
    centroid = np.zeros((n, 3))
    np.add.at(centroid, inverse, points)
    centroid /= counts[:, None]
    col = np.zeros((n, 3))
    np.add.at(col, inverse, colors)
    col /= counts[:, None]
    dist = np.linalg.norm(points - centroid[inverse], axis=1)
    order = np.lexsort((dist, inverse))  # by voxel, then distance
    first = np.ones(len(order), dtype=bool)
    first[1:] = inverse[order][1:] != inverse[order][:-1]
    rep = np.empty(n, dtype=np.int64)
    rep[inverse[order][first]] = order[first]
    return DownsampleResult(centroid, col, rep)


def remove_statistical_outliers(
    points: np.ndarray, nb_neighbors: int, std_ratio: float
) -> np.ndarray:
    """Return a boolean keep-mask (Open3D statistical outlier removal)."""
    if len(points) <= nb_neighbors:
        return np.ones(len(points), dtype=bool)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    _, idx = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    keep = np.zeros(len(points), dtype=bool)
    keep[np.asarray(idx, dtype=np.int64)] = True
    return keep


def estimate_normals(points: np.ndarray, radius_m: float, max_nn: int = 30) -> np.ndarray:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_m, max_nn=max_nn))
    return np.asarray(pcd.normals)
