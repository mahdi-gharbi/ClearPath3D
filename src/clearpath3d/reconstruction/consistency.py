"""Sensor-only pose/depth consistency: re-project frame B's depth into frame A and compare.

Used to verify dataset conventions (pose direction, depth type) without any ground truth. A low
residual is evidence that poses and depth agree; it is not a probability that a pose is correct.
"""

from __future__ import annotations

import numpy as np

from clearpath3d.contracts import RGBDFrame
from clearpath3d.geometry.frames import invert, transform_points
from clearpath3d.geometry.projection import backproject, project


def reprojection_residual(
    a: RGBDFrame,
    b: RGBDFrame,
    T_map_cam_a: np.ndarray | None = None,
    T_map_cam_b: np.ndarray | None = None,
) -> dict:
    Ta = a.T_map_camera if T_map_cam_a is None else T_map_cam_a
    Tb = b.T_map_camera if T_map_cam_b is None else T_map_cam_b
    if Ta is None or Tb is None:
        raise ValueError("both frames need a pose")
    pts_b, _ = backproject(b.depth_m, b.depth_valid, b.intrinsics)
    pts_a = transform_points(invert(Ta) @ Tb, pts_b)
    uv, front = project(pts_a, a.intrinsics)
    with np.errstate(invalid="ignore"):
        u = np.round(uv[:, 0])
        v = np.round(uv[:, 1])
    inside = front & (u >= 0) & (u < a.intrinsics.width) & (v >= 0) & (v < a.intrinsics.height)
    ui, vi = u[inside].astype(int), v[inside].astype(int)
    has = a.depth_valid[vi, ui]
    r = np.abs(a.depth_m[vi, ui][has] - pts_a[inside, 2][has])
    if r.size == 0:
        return {"overlap": 0.0, "median_abs_dz_m": None, "frac_below_2cm": None}
    return {
        "overlap": round(float(has.sum() / max(len(pts_b), 1)), 4),
        "median_abs_dz_m": round(float(np.median(r)), 5),
        "frac_below_2cm": round(float((r < 0.02).mean()), 4),
    }
