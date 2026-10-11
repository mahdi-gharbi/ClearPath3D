"""Visibility-based voxel evidence map (log-odds), built from depth *rays*, not from the TSDF mesh.

For each valid depth pixel (optionally subsampled):
- every voxel strictly between the camera and the hit voxel receives ``delta_free``;
- the voxel containing the hit receives ``delta_occ``;
- nothing behind the hit is touched, and invalid pixels (holes, out of range) touch nothing.

Correlated rays of one frame are capped: a voxel receives at most one update per frame (occupied
wins over free inside the same frame). ``L = clip(L + delta, L_min, L_max)``; ``p = sigmoid(L)``.
These probabilities are sensor-model estimates, not calibrated confidences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from clearpath3d.contracts import RGBDFrame
from clearpath3d.geometry.frames import transform_points
from clearpath3d.geometry.projection import backproject

UPDATE_NONE, UPDATE_FREE, UPDATE_OCC, UPDATE_ROBOT = 0, 1, 2, 3


@dataclass
class EvidenceParams:
    resolution_m: float = 0.05
    # Fixed local map extent. Must cover the explored area; anything outside is treated as unknown by the
    # planner. Bounds are multiples of the resolution so cell boundaries do not depend on the extent.
    bounds_min: tuple[float, float, float] = (-1.0, -1.0, -0.15)
    bounds_max: tuple[float, float, float] = (8.5, 6.5, 2.0)
    delta_occ: float = 0.85
    delta_free: float = -0.40
    l_min: float = -2.0
    l_max: float = 2.0  # lower clamp: a moved object can be freed after ~6 free observations
    pixel_stride: int = 2
    max_range_m: float = 6.0


def bounds_around(
    xy: np.ndarray, margin_m: float, res: float = 0.05, z: tuple[float, float] = (-0.15, 2.0)
):
    """Map bounds covering positions ``xy`` (N,2) plus ``margin_m`` (e.g. the sensor range), snapped to ``res``.

    Fixed default bounds only fit the synthetic rooms; real sequences live elsewhere in their world frame
    (TUM pioneer_360 starts at x = -1.82 m, outside the default grid).
    """
    lo = np.floor((np.asarray(xy).min(0) - margin_m) / res) * res
    hi = np.ceil((np.asarray(xy).max(0) + margin_m) / res) * res
    return (float(lo[0]), float(lo[1]), z[0]), (float(hi[0]), float(hi[1]), z[1])


def dda_traverse(
    origin: np.ndarray,
    ends: np.ndarray,
    grid_min: np.ndarray,
    res: float,
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Exact voxel traversal (Amanatides & Woo), vectorised over rays.

    Returns ``(free_linear_idx, hit_linear_idx)``: voxels crossed strictly before the end voxel, and
    the end voxels. Voxels outside the grid are dropped (traversal itself continues through them).
    """
    shape_a = np.asarray(shape)
    o = (np.asarray(origin, dtype=np.float64) - grid_min) / res  # voxel units
    e = (np.asarray(ends, dtype=np.float64) - grid_min) / res
    n = len(e)
    cur = np.floor(np.broadcast_to(o, e.shape)).astype(np.int64)
    end = np.floor(e).astype(np.int64)
    d = e - o
    step = np.sign(d).astype(np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(d != 0, 1.0 / d, np.inf)
        next_boundary = cur + (step > 0)
        t_max = np.where(d != 0, (next_boundary - o) * inv, np.inf)
        t_delta = np.where(d != 0, np.abs(inv), np.inf)

    def linear(idx: np.ndarray) -> np.ndarray:
        inside = np.all((idx >= 0) & (idx < shape_a), axis=1)
        lin = np.ravel_multi_index(idx[inside].T, shape) if inside.any() else np.zeros(0, np.int64)
        return lin

    free_chunks = []
    active = np.any(cur != end, axis=1)
    max_steps = int(np.abs(end - cur).sum(axis=1).max(initial=0)) + 1
    for _ in range(max_steps):
        if not active.any():
            break
        idx = np.nonzero(active)[0]
        free_chunks.append(linear(cur[idx]))
        tm = t_max[idx]
        axis = np.argmin(tm, axis=1)
        rows = np.arange(len(idx))
        cur[idx, axis] += step[idx, axis]
        t_max[idx, axis] += t_delta[idx, axis]
        # Stop when the end voxel is reached (or numerically overshot: t beyond the end point).
        reached = np.all(cur[idx] == end[idx], axis=1) | (tm[rows, axis] > 1.0 + 1e-9)
        active[idx[reached]] = False
    free = np.concatenate(free_chunks) if free_chunks else np.zeros(0, np.int64)
    return free, linear(end) if n else np.zeros(0, np.int64)


class VoxelEvidenceGrid:
    def __init__(self, p: EvidenceParams):
        self.p = p
        self.min = np.asarray(p.bounds_min, dtype=np.float64)
        self.shape = tuple(
            int(np.ceil((hi - lo) / p.resolution_m))
            for lo, hi in zip(p.bounds_min, p.bounds_max, strict=True)
        )
        n = int(np.prod(self.shape))
        self.log_odds = np.zeros(n, dtype=np.float32)
        self.obs_count = np.zeros(n, dtype=np.uint16)
        self.hit_count = np.zeros(n, dtype=np.uint16)  # occupied updates ever received
        # Columns the robot body stood in (persistent: later sensor rays must not erase this support).
        self.robot_support = np.zeros(self.shape[:2], dtype=bool)
        self.last_seen_s = np.full(n, -np.inf, dtype=np.float64)
        self.last_frame = np.full(n, -1, dtype=np.int32)  # index into self.frame_ids
        self.last_update = np.zeros(n, dtype=np.uint8)
        self.frame_ids: list[str] = []
        self.version = 0

    # ------------------------------------------------------------------ geometry helpers
    def centers(self, lin: np.ndarray) -> np.ndarray:
        idx = np.stack(np.unravel_index(lin, self.shape), axis=1)
        return self.min + (idx + 0.5) * self.p.resolution_m

    def index_of(self, pts: np.ndarray) -> np.ndarray:
        """Linear index of points, -1 outside the grid."""
        idx = np.floor((np.asarray(pts) - self.min) / self.p.resolution_m).astype(np.int64)
        inside = np.all((idx >= 0) & (idx < np.asarray(self.shape)), axis=1)
        out = np.full(len(idx), -1, dtype=np.int64)
        if inside.any():
            out[inside] = np.ravel_multi_index(idx[inside].T, self.shape)
        return out

    def volume(self, arr: np.ndarray) -> np.ndarray:
        return arr.reshape(self.shape)

    # ------------------------------------------------------------------ updates
    def _apply(
        self, free: np.ndarray, occ: np.ndarray, t: float, frame_idx: int, free_kind: int
    ) -> None:
        occ = np.unique(occ)
        free = np.setdiff1d(
            np.unique(free), occ, assume_unique=True
        )  # one update per voxel per frame
        lo = self.log_odds
        lo[free] = np.clip(lo[free] + self.p.delta_free, self.p.l_min, self.p.l_max)
        lo[occ] = np.clip(lo[occ] + self.p.delta_occ, self.p.l_min, self.p.l_max)
        both = np.concatenate([free, occ])
        self.obs_count[both] = np.minimum(self.obs_count[both].astype(np.int64) + 1, 65535)
        self.hit_count[occ] = np.minimum(self.hit_count[occ].astype(np.int64) + 1, 65535)
        self.last_seen_s[both] = t
        self.last_frame[both] = frame_idx
        self.last_update[free] = free_kind
        self.last_update[occ] = UPDATE_OCC
        self.version += 1

    def integrate_frame(self, frame: RGBDFrame, exclude_mask: np.ndarray | None = None) -> dict:
        if frame.T_map_camera is None:
            raise ValueError("evidence integration needs a pose (supplied or estimated)")
        valid = frame.depth_valid & (frame.depth_m <= self.p.max_range_m)
        if exclude_mask is not None:
            valid &= ~exclude_mask
        s = self.p.pixel_stride
        if s > 1:
            sub = np.zeros_like(valid)
            sub[::s, ::s] = True
            valid &= sub
        pts_cam, _ = backproject(frame.depth_m, valid, frame.intrinsics)
        ends = transform_points(frame.T_map_camera, pts_cam)
        origin = frame.T_map_camera[:3, 3]
        free, occ = dda_traverse(origin, ends, self.min, self.p.resolution_m, self.shape)
        self.frame_ids.append(frame.frame_id)
        self._apply(free, occ, frame.timestamp_s, len(self.frame_ids) - 1, UPDATE_FREE)
        return {
            "rays": int(len(ends)),
            "free_updates": int(len(np.unique(free))),
            "occ_updates": int(len(np.unique(occ))),
        }

    def mark_robot_footprint(
        self,
        T_map_base: np.ndarray,
        radius_m: float,
        height_m: float,
        t: float,
        frame_id: str,
        T_map_base_prev: np.ndarray | None = None,
        max_gap_m: float = 0.5,
    ) -> int:
        """The robot body occupied this volume at a supplied pose: evidence that it is free of obstacles.

        Marks every voxel column whose centre lies within ``radius + res*sqrt(2)/2`` of the robot centre (the
        columns the body overlaps), from one voxel above the floor to ``height_m``. If the previous pose
        is closer than ``max_gap_m``, the segment between them is swept too (assumes continuous motion
        between supplied poses). Source is recorded as UPDATE_ROBOT, distinct from sensor rays.
        """
        r = self.p.resolution_m
        cx, cy, z0 = T_map_base[:3, 3]
        centers = [np.array([cx, cy])]
        if T_map_base_prev is not None:
            prev = T_map_base_prev[:2, 3]
            gap = float(np.linalg.norm(prev - centers[0]))
            if 0 < gap <= max_gap_m:
                n = int(np.ceil(gap / (r / 2)))
                centers = [prev + (centers[0] - prev) * k / n for k in range(n + 1)]
        # Columns within radius + half a cell diagonal: the planner reasons on cell centres, which can be
        # up to r*sqrt(2)/2 from the true pose. Over-claim bounded by 3.5 cm at 5 cm voxels (documented).
        reach = radius_m + r * np.sqrt(2) / 2
        nx, ny, nz = self.shape
        ix = np.arange(nx)
        iy = np.arange(ny)
        xc = self.min[0] + (ix + 0.5) * r
        yc = self.min[1] + (iy + 0.5) * r
        cols = np.zeros((nx, ny), dtype=bool)
        for c in centers:
            sx = np.abs(xc - c[0]) <= reach
            sy = np.abs(yc - c[1]) <= reach
            if not sx.any() or not sy.any():
                continue
            X, Y = np.meshgrid(xc[sx], yc[sy], indexing="ij")
            sub = np.hypot(X - c[0], Y - c[1]) <= reach
            cols[np.ix_(sx, sy)] |= sub
        zc = self.min[2] + (np.arange(nz) + 0.5) * r
        layers = np.nonzero((zc >= z0 + r / 2) & (zc <= z0 + height_m))[0]
        ci, cj = np.nonzero(cols)
        if len(ci) == 0 or len(layers) == 0:
            return 0
        ii = np.repeat(ci, len(layers))
        jj = np.repeat(cj, len(layers))
        kk = np.tile(layers, len(ci))
        lin = np.ravel_multi_index((ii, jj, kk), self.shape)
        self.frame_ids.append(f"{frame_id}:robot_footprint")
        self.robot_support[ci, cj] = True
        self._apply(lin, np.zeros(0, np.int64), t, len(self.frame_ids) - 1, UPDATE_ROBOT)
        return int(len(lin))

    def probability(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.log_odds))
