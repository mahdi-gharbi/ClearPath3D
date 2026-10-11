"""Pinhole projection / back-projection. Integer pixel coords are pixel centres (no +0.5)."""

from __future__ import annotations

import numpy as np

from clearpath3d.contracts import Intrinsics


def pixel_grid(intr: Intrinsics) -> tuple[np.ndarray, np.ndarray]:
    """Return (u, v) integer grids of shape [H, W]."""
    v, u = np.mgrid[0 : intr.height, 0 : intr.width]
    return u, v


def ray_directions_optical(intr: Intrinsics) -> np.ndarray:
    """[H, W, 3] ray directions in camera_optical with unit z component (t along ray == z-depth)."""
    u, v = pixel_grid(intr)
    d = np.empty((intr.height, intr.width, 3), dtype=np.float64)
    d[..., 0] = (u - intr.cx) / intr.fx
    d[..., 1] = (v - intr.cy) / intr.fy
    d[..., 2] = 1.0
    return d


def ray_distance_to_z_depth(ray_dist: np.ndarray, intr: Intrinsics) -> np.ndarray:
    d = ray_directions_optical(intr)
    return (ray_dist / np.linalg.norm(d, axis=-1)).astype(ray_dist.dtype)


def z_depth_to_ray_distance(z: np.ndarray, intr: Intrinsics) -> np.ndarray:
    d = ray_directions_optical(intr)
    return (z * np.linalg.norm(d, axis=-1)).astype(z.dtype)


def backproject(
    depth_m: np.ndarray, valid: np.ndarray, intr: Intrinsics
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project valid pixels.

    Returns ``(points_optical [N,3] float64, pixel_index [N,2] int (v, u))``. Invalid pixels are
    dropped, never turned into points.
    """
    vs, us = np.nonzero(valid)
    z = depth_m[vs, us].astype(np.float64)
    x = (us - intr.cx) * z / intr.fx
    y = (vs - intr.cy) * z / intr.fy
    return np.stack([x, y, z], axis=1), np.stack([vs, us], axis=1)


def project(points_optical: np.ndarray, intr: Intrinsics) -> tuple[np.ndarray, np.ndarray]:
    """Project optical-frame points. Returns ``(uv [N,2] float, in_front [N] bool)``."""
    p = np.asarray(points_optical, dtype=np.float64)
    z = p[:, 2]
    in_front = z > 1e-9
    zs = np.where(in_front, z, np.nan)
    u = intr.fx * p[:, 0] / zs + intr.cx
    v = intr.fy * p[:, 1] / zs + intr.cy
    return np.stack([u, v], axis=1), in_front
