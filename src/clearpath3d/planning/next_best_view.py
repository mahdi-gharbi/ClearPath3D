"""Next-best-view on the *observed* map only (the planner never sees the simulator world).

Candidates: robot-centre cells reachable through known free space (A*-compatible: unknown and
obstacles dilated by the robot radius), on a regular grid, within ``max_travel_m``, each with
``n_yaws`` camera headings.

Predicted gain (``first_unknown`` model, the default and only implemented hypothesis): cast a coarse
grid of camera rays through the current voxel evidence; along each ray, free voxels let the ray
continue, an occupied or ambiguous (observed, neither free nor occupied) voxel stops it with no gain,
and the *first never-observed* voxel counts once and stops the ray (unknown is not assumed
transparent). Ambiguous voxels are excluded from the gain because more observation of a surface
boundary does not resolve them. Gain = number of distinct such voxels whose height lies in
the traversability slab ``[floor - 0.05, floor + robot_height]``. Unit: voxels (5 cm)^3.

    score(q) = alpha * gain(q) / gain_unit - beta * path_length_m(q) - gamma * |delta_yaw(q)|_rad

Defaults: gain_unit = 100 voxels, alpha = 1, beta = 0.2 / m, gamma = 0.1 / rad. They are design choices
to be validated, not tuned results. If no reachable candidate has gain >= ``min_gain_voxels``, the
result says so explicitly instead of proposing a view.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from clearpath3d.contracts import Intrinsics
from clearpath3d.geometry.frames import compose, planar_pose
from clearpath3d.mapping.projection import Map2D
from clearpath3d.mapping.voxel_evidence import VoxelEvidenceGrid
from clearpath3d.planning.astar import RobotSpec, blocked_grid
from clearpath3d.planning.frontiers import frontier_cells, travel_distance


@dataclass
class NBVParams:
    candidate_spacing_m: float = 0.35
    n_yaws: int = 8
    max_travel_m: float = 4.0
    ray_cols: int = 24
    ray_rows: int = 16
    max_range_m: float = 4.0
    # One sample per voxel length: a ray can occasionally skip a voxel corner. Acceptable for a gain
    # *estimate* (measured cost: 864 candidates took ~18 s with 0.025 m steps and 32x24 rays).
    step_m: float = 0.05
    gain_unit_voxels: float = 100.0
    alpha: float = 1.0
    beta_per_m: float = 0.2
    gamma_per_rad: float = 0.1
    min_gain_voxels: int = 20
    gain_model: str = "first_unknown"


@dataclass
class Candidate:
    x: float
    y: float
    yaw: float
    path_length_m: float
    rotation_rad: float
    gain_voxels: int
    score: float


@dataclass
class NBVResult:
    status: str  # ok | no_reachable_candidate | no_informative_candidate | no_map
    message: str
    best: Candidate | None
    candidates: list[Candidate]
    frontier_cells: int
    params: dict


def _ray_dirs(intr: Intrinsics, cols: int, rows: int) -> np.ndarray:
    us = np.linspace(0, intr.width - 1, cols)
    vs = np.linspace(0, intr.height - 1, rows)
    U, V = np.meshgrid(us, vs)
    d = np.stack(
        [(U - intr.cx) / intr.fx, (V - intr.cy) / intr.fy, np.ones_like(U)], axis=-1
    ).reshape(-1, 3)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def predicted_gain(
    grid: VoxelEvidenceGrid,
    T_map_camera: np.ndarray,
    dirs_cam: np.ndarray,
    l_occ: float,
    l_free: float,
    floor_z: float,
    robot_height: float,
    max_range_m: float,
    step_m: float,
) -> int:
    origin = T_map_camera[:3, 3]
    dirs = dirs_cam @ T_map_camera[:3, :3].T
    ts = np.arange(step_m, max_range_m + 1e-9, step_m)
    pts = origin + dirs[:, None, :] * ts[None, :, None]  # [R, S, 3]
    lin = grid.index_of(pts.reshape(-1, 3)).reshape(len(dirs), len(ts))
    safe = np.clip(lin, 0, None)
    L = grid.log_odds[safe]
    outside = lin < 0
    never = ~outside & (grid.obs_count[safe] == 0)
    passable = ~outside & ~never & (L <= l_free)
    stop = ~passable  # occupied, ambiguous, never observed, or outside the grid
    unknown = never
    first = np.argmax(stop, axis=1)
    any_stop = stop[np.arange(len(dirs)), first]
    is_unknown = any_stop & unknown[np.arange(len(dirs)), first]
    hit_lin = lin[np.arange(len(dirs)), first][is_unknown]
    if hit_lin.size == 0:
        return 0
    z = grid.centers(np.unique(hit_lin))[:, 2]
    return int(((z >= floor_z - 0.05) & (z <= floor_z + robot_height)).sum())


def _sample_reachable(dist: np.ndarray, max_travel_m: float, spacing_cells: float) -> np.ndarray:
    """Greedy spacing-based sampling of reachable cells, nearest first (includes the current cell).

    Works for thin reachable regions (e.g. a corridor swept by the robot) where an aligned grid would
    miss every cell.
    """
    cells = np.argwhere(np.isfinite(dist) & (dist <= max_travel_m))
    if len(cells) == 0:
        return cells
    order = np.argsort(dist[cells[:, 0], cells[:, 1]], kind="stable")
    chosen: list[np.ndarray] = []
    r2 = spacing_cells**2
    for c in cells[order]:
        if all(((c - q) ** 2).sum() >= r2 for q in chosen):
            chosen.append(c)
    return np.array(chosen)


def _angle_diff(a: float, b: float) -> float:
    return abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def propose_viewpoint(
    grid: VoxelEvidenceGrid,
    m: Map2D | None,
    robot_xy,
    robot_yaw: float,
    robot: RobotSpec,
    T_base_camera: np.ndarray,
    intr: Intrinsics,
    l_occ: float,
    l_free: float,
    p: NBVParams | None = None,
) -> NBVResult:
    p = p or NBVParams()
    if m is None:
        return NBVResult(
            "no_map",
            "no floor established: no traversable space to move in",
            None,
            [],
            0,
            asdict(p),
        )
    res = m.resolution_m
    blocked = blocked_grid(m.state, res, robot)
    s = tuple(int(v) for v in m.cell_of(np.asarray(robot_xy)))
    dist = travel_distance(blocked, s, res)
    fmask, _ = frontier_cells(m)
    ij = _sample_reachable(dist, p.max_travel_m, p.candidate_spacing_m / res)
    if len(ij) == 0:
        return NBVResult(
            "no_reachable_candidate",
            "no candidate position reachable through observed free space",
            None,
            [],
            int(fmask.sum()),
            asdict(p),
        )
    dirs = _ray_dirs(intr, p.ray_cols, p.ray_rows)
    yaws = np.linspace(-np.pi, np.pi, p.n_yaws, endpoint=False)
    cands = []
    for i, j in ij:
        x, y = m.center_of(np.array([i, j]))
        for yaw in yaws:
            T = compose(planar_pose(float(x), float(y), float(yaw)), T_base_camera)
            g = predicted_gain(
                grid, T, dirs, l_occ, l_free, m.floor_z_m, robot.height_m, p.max_range_m, p.step_m
            )
            rot = _angle_diff(float(yaw), robot_yaw)
            score = (
                p.alpha * g / p.gain_unit_voxels
                - p.beta_per_m * float(dist[i, j])
                - p.gamma_per_rad * rot
            )
            cands.append(
                Candidate(float(x), float(y), float(yaw), float(dist[i, j]), rot, g, score)
            )
    informative = [c for c in cands if c.gain_voxels >= p.min_gain_voxels]
    if not informative:
        return NBVResult(
            "no_informative_candidate",
            f"{len(cands)} reachable candidates, none expected to observe >= {p.min_gain_voxels} unknown voxels",
            None,
            cands,
            int(fmask.sum()),
            asdict(p),
        )
    best = max(informative, key=lambda c: c.score)
    return NBVResult(
        "ok", "best reachable viewpoint by score", best, cands, int(fmask.sum()), asdict(p)
    )
