"""Project the voxel evidence onto a 2D traversability map for a cylindrical robot.

Cell states: OCCUPIED (100), FREE (0, *observed* free), UNKNOWN (-1).

Exact criterion (per 2D cell = one voxel column, resolution = voxel size):
- band = voxels whose centre height above the floor lies in ``[clearance_m, robot_height_m]``;
- OCCUPIED  if any band voxel has ``log_odds >= l_occ`` and is not expired;
- FREE      if ground support is established in the cell (the floor surface voxel was hit by a ray
            with ``log_odds >= l_occ``, or the robot stood there), AND at least ``min_free_fraction``
            of the band voxels have ``log_odds <= l_free`` (``l_free_near_obstacle`` for cells next to an
            occupied cell) and are not expired, AND no band voxel is occupied;
- UNKNOWN   otherwise (including: floor seen but band not covered; nothing seen; evidence expired).

"No obstacle detected" is therefore never turned into "free" by itself. Expired evidence returns to
UNKNOWN, never to FREE. Robot-footprint dilation is done by the planner, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from clearpath3d.mapping.voxel_evidence import VoxelEvidenceGrid

OCCUPIED, FREE, UNKNOWN = 100, 0, -1


@dataclass
class ProjectionParams:
    robot_height_m: float = 0.40
    clearance_m: float = 0.06  # obstacles lower than this above the floor are ignored (step height)
    l_occ: float = 0.85  # one confident occupied update
    l_free: float = -0.39  # one net free observation (delta_free = -0.40)
    # Cells next to an occupied cell need two net free observations: a single grazing ray through the free
    # sliver of a partly occupied voxel marked a cabinet edge cell free (proc_005, dev split). Applying -0.79
    # everywhere removed it but left every passable door blocked by unknown on dev/validation (0/7 + 0/7).
    l_free_near_obstacle: float = -0.79
    min_free_fraction: float = 0.8
    floor_tolerance_m: float = 0.05
    expiry_s: float | None = None  # evidence older than this (w.r.t. ``now_s``) counts as unknown


@dataclass
class Map2D:
    state: np.ndarray  # int8 [NX, NY]
    origin_xy: np.ndarray  # map coordinates of cell (0, 0)'s lower corner
    resolution_m: float
    floor_z_m: float
    free_fraction: np.ndarray  # [NX, NY]
    ground_support: np.ndarray  # bool [NX, NY]
    version: int
    params: ProjectionParams

    def cell_of(self, xy: np.ndarray) -> np.ndarray:
        return np.floor((np.asarray(xy) - self.origin_xy) / self.resolution_m).astype(np.int64)

    def center_of(self, ij: np.ndarray) -> np.ndarray:
        return self.origin_xy + (np.asarray(ij) + 0.5) * self.resolution_m

    def counts(self) -> dict[str, int]:
        return {
            "occupied": int((self.state == OCCUPIED).sum()),
            "free": int((self.state == FREE).sum()),
            "unknown": int((self.state == UNKNOWN).sum()),
        }


def project_to_2d(
    grid: VoxelEvidenceGrid, floor_z_m: float, p: ProjectionParams, now_s: float | None = None
) -> Map2D:
    res = grid.p.resolution_m
    nx, ny, nz = grid.shape
    zc = grid.min[2] + (np.arange(nz) + 0.5) * res - floor_z_m  # voxel centre height above floor
    band = (zc >= p.clearance_m) & (zc <= p.robot_height_m)
    floor_layer = np.abs(zc) <= max(p.floor_tolerance_m, res / 2)
    L = grid.volume(grid.log_odds)
    if p.expiry_s is not None and now_s is not None:
        alive = grid.volume(grid.last_seen_s) >= now_s - p.expiry_s
    else:
        alive = np.ones(grid.shape, dtype=bool)
    Lb, Ab = L[:, :, band], alive[:, :, band]
    occ = ((Lb >= p.l_occ) & Ab).any(axis=2)
    free_frac = ((Lb <= p.l_free) & Ab).mean(axis=2) if band.any() else np.zeros((nx, ny))
    strict_frac = (
        ((Lb <= p.l_free_near_obstacle) & Ab).mean(axis=2) if band.any() else np.zeros((nx, ny))
    )
    # "Next to an obstacle": any occupied voxel above the floor in the 3x3 column neighbourhood, at any height
    # (an object may only have been seen from the top, e.g. a cabinet whose side face was never observed).
    above = zc >= p.clearance_m
    near_occ = ndimage.binary_dilation(
        ((L[:, :, above] >= p.l_occ) & alive[:, :, above]).any(axis=2),
        structure=np.ones((3, 3), bool),
    )
    free_frac = np.where(near_occ, strict_frac, free_frac)
    # Floor seen: the floor voxel was hit at least once and has not since been confirmed free. (Requiring
    # log_odds >= l_occ lost support when a single grazing ray passed through the floor voxel near its hit.)
    H = grid.volume(grid.hit_count)[:, :, floor_layer]
    floor_hit = ((H >= 1) & (L[:, :, floor_layer] > p.l_free) & alive[:, :, floor_layer]).any(
        axis=2
    )
    # Robot stood here (persistent flag; reading last_update lost it as soon as a later ray passed).
    robot_stood = grid.robot_support
    support = floor_hit | robot_stood
    state = np.full((nx, ny), UNKNOWN, dtype=np.int8)
    state[support & (free_frac >= p.min_free_fraction) & ~occ] = FREE
    state[occ] = OCCUPIED
    return Map2D(state, grid.min[:2].copy(), res, floor_z_m, free_frac, support, grid.version, p)


def explain_cell(grid: VoxelEvidenceGrid, m: Map2D, ij: tuple[int, int]) -> dict:
    """Evidence behind one 2D cell: per-voxel log-odds, counts, last frame and update source."""
    i, j = ij
    res = grid.p.resolution_m
    nz = grid.shape[2]
    rows = []
    for k in range(nz):
        lin = int(np.ravel_multi_index((i, j, k), grid.shape))
        z = grid.min[2] + (k + 0.5) * res - m.floor_z_m
        if not (-res <= z <= m.params.robot_height_m + res):
            continue
        lf = int(grid.last_frame[lin])
        rows.append(
            {
                "height_above_floor_m": round(float(z), 3),
                "log_odds": round(float(grid.log_odds[lin]), 3),
                "observations": int(grid.obs_count[lin]),
                "last_seen_s": None if lf < 0 else float(grid.last_seen_s[lin]),
                "last_frame": None if lf < 0 else grid.frame_ids[lf],
                "last_update": ["none", "free_ray", "occupied_hit", "robot_footprint"][
                    int(grid.last_update[lin])
                ],
            }
        )
    state = int(m.state[i, j])
    return {
        "cell": [int(i), int(j)],
        "center_m": m.center_of(np.array([i, j])).round(3).tolist(),
        "state": {OCCUPIED: "occupied", FREE: "free", UNKNOWN: "unknown"}[state],
        "ground_support": bool(m.ground_support[i, j]),
        "band_free_fraction": round(float(m.free_fraction[i, j]), 3),
        "criterion": (
            f"free requires ground support and >= {m.params.min_free_fraction:.0%} of voxels between "
            f"{m.params.clearance_m} and {m.params.robot_height_m} m observed free; any occupied voxel in "
            "that band makes the cell occupied"
        ),
        "voxels": rows,
    }
