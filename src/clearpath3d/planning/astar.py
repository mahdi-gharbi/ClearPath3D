"""Footprint-aware A* on the 2D map. Geometric simulation only: no friction, slope or dynamics.

- Forbidden cells = OCCUPIED cells, and UNKNOWN cells adjacent to them, dilated by ``robot_radius + half
  a cell diagonal``; other UNKNOWN cells (including outside the map) dilated by ``robot_radius``. The main
  mode never crosses unknown space.
  Cells within ``radius + margin`` are allowed but pay ``margin_cost_per_m`` per metre (like an inflated
  costmap layer), so the margin is used only when unavoidable - e.g. leaving a start pose whose
  surroundings were never observed. The result reports how many path cells lie in that margin band.
- 8-connected moves; a diagonal move is forbidden if either adjacent orthogonal cell is blocked.
- Cost = Euclidean step length + ``proximity_weight * max(0, proximity_m - clearance)`` per metre.
  The octile-distance heuristic ignores the (non-negative) penalty, so it stays admissible.
- On failure, a diagnostic re-plan treating unknown as free tells "blocked by obstacles" apart from
  "not enough observation".
- Optional shortcut smoothing re-checks every segment against the blocked grid.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from clearpath3d.mapping.projection import FREE, OCCUPIED, UNKNOWN, Map2D


@dataclass
class RobotSpec:
    radius_m: float = 0.25
    height_m: float = 0.40
    margin_m: float = 0.05


@dataclass
class PlanParams:
    proximity_m: float = 0.30
    proximity_weight: float = 2.0
    margin_cost_per_m: float = 20.0
    smooth: bool = True


@dataclass
class PlanResult:
    success: bool
    cause: str  # ok | start_outside_map | goal_outside_map | start_not_free | goal_occupied | goal_unknown |
    #             goal_footprint_blocked | blocked_by_obstacles | blocked_by_unknown
    message: str
    path_cells: list[tuple[int, int]] = field(default_factory=list)
    path_m: list[list[float]] = field(default_factory=list)
    length_m: float = 0.0
    expanded: int = 0
    map_version: int = -1
    margin_cells: int = 0  # path cells closer than radius + margin to obstacle/unknown


def _disk(radius_cells: float) -> np.ndarray:
    r = int(math.ceil(radius_cells))
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y) <= radius_cells * radius_cells + 1e-9


def blocked_grid(
    state: np.ndarray,
    res: float,
    robot: RobotSpec,
    unknown_blocks: bool = True,
    with_margin: bool = False,
) -> np.ndarray:
    r = robot.radius_m + (robot.margin_m if with_margin else 0.0)
    # An OCCUPIED cell means "an obstacle somewhere in this cell": its true surface can be half a cell
    # diagonal closer than the cell centre, so occupied cells are dilated by r + res*sqrt(2)/2 (found on the
    # procedural dev room proc_002: 0.244 m true clearance accepted for R = 0.25 m). An UNKNOWN cell may hide
    # an obstacle (validation proc_011: 0.247 m next to an unobserved cabinet corner): UNKNOWN cells adjacent to
    # an OCCUPIED cell get the same extra dilation. Other unknown cells (e.g. sparse-coverage holes in open
    # space) are dilated by r: still never traversed, but they do not close a corridor (applying the extra
    # dilation to all unknown cells blocked the passable door of dev proc_003). Outside the map is unknown.
    occ = state == OCCUPIED
    disk_extra = _disk((r + res * np.sqrt(2) / 2) / res)
    blocked = ndimage.binary_dilation(occ, structure=disk_extra)
    if unknown_blocks:
        unk = state == UNKNOWN
        near = unk & ndimage.binary_dilation(occ, structure=np.ones((3, 3), bool))
        blocked |= ndimage.binary_dilation(near, structure=disk_extra)
        blocked |= ndimage.binary_dilation(unk, structure=_disk(r / res), border_value=1)
    return blocked


def clearance_m(state: np.ndarray, res: float, unknown_blocks: bool = True) -> np.ndarray:
    obstacle = (state == OCCUPIED) | ((state == UNKNOWN) if unknown_blocks else False)
    if unknown_blocks:  # outside the map counts as unknown
        padded = np.pad(~obstacle, 1, constant_values=False)
        return ndimage.distance_transform_edt(padded)[1:-1, 1:-1] * res
    if not obstacle.any():
        return np.full(state.shape, np.inf)
    return ndimage.distance_transform_edt(~obstacle) * res


_MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def _astar(blocked, margin, clear, res, start, goal, p: PlanParams):
    nx, ny = blocked.shape
    sq2 = math.sqrt(2.0)

    def h(c):
        dx, dy = abs(c[0] - goal[0]), abs(c[1] - goal[1])
        return res * (max(dx, dy) + (sq2 - 1) * min(dx, dy))

    g = {start: 0.0}
    parent = {start: None}
    heap = [(h(start), 0.0, start)]
    expanded = 0
    closed = set()
    while heap:
        _, gc, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        closed.add(cur)
        expanded += 1
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1], expanded
        for dx, dy in _MOVES:
            nb = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nb[0] < nx and 0 <= nb[1] < ny) or blocked[nb]:
                continue
            if dx and dy and (blocked[cur[0] + dx, cur[1]] or blocked[cur[0], cur[1] + dy]):
                continue  # no corner cutting between two blocked cells
            step = res * (sq2 if dx and dy else 1.0)
            pen = p.proximity_weight * max(0.0, p.proximity_m - float(clear[nb])) * step
            if margin[nb]:
                pen += p.margin_cost_per_m * step
            ng = gc + step + pen
            if ng < g.get(nb, math.inf):
                g[nb] = ng
                parent[nb] = cur
                heapq.heappush(heap, (ng + h(nb), ng, nb))
    return None, expanded


def segment_free(blocked: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Conservative line check: samples at 1/4 cell and also tests the 4-neighbour cells crossed."""
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 4) + 1
    for t in np.linspace(0.0, 1.0, n + 1):
        x = a[0] + (b[0] - a[0]) * t
        y = a[1] + (b[1] - a[1]) * t
        for cx in {math.floor(x + 0.5 - 1e-9), math.floor(x + 0.5 + 1e-9)}:
            for cy in {math.floor(y + 0.5 - 1e-9), math.floor(y + 0.5 + 1e-9)}:
                if blocked[cx, cy]:
                    return False
    return True


def smooth_path(blocked: np.ndarray, path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(path) <= 2:
        return path
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not segment_free(blocked, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def plan(m: Map2D, start_xy, goal_xy, robot: RobotSpec, p: PlanParams | None = None) -> PlanResult:
    p = p or PlanParams()
    res = m.resolution_m
    nx, ny = m.state.shape
    s = tuple(int(v) for v in m.cell_of(np.asarray(start_xy)))
    g = tuple(int(v) for v in m.cell_of(np.asarray(goal_xy)))
    if not (0 <= s[0] < nx and 0 <= s[1] < ny):
        return PlanResult(
            False, "start_outside_map", "start lies outside the mapped area", map_version=m.version
        )
    if not (0 <= g[0] < nx and 0 <= g[1] < ny):
        return PlanResult(
            False, "goal_outside_map", "goal lies outside the mapped area", map_version=m.version
        )
    blocked = blocked_grid(m.state, res, robot, unknown_blocks=True)
    margin = blocked_grid(m.state, res, robot, unknown_blocks=True, with_margin=True) & ~blocked
    if blocked[s]:
        return PlanResult(
            False,
            "start_not_free",
            "robot footprint at the start is not entirely in observed free space",
            map_version=m.version,
        )
    if m.state[g] == OCCUPIED:
        return PlanResult(
            False, "goal_occupied", "goal cell contains an observed obstacle", map_version=m.version
        )
    if m.state[g] == UNKNOWN:
        return PlanResult(
            False, "goal_unknown", "goal cell has not been observed enough", map_version=m.version
        )
    if blocked[g]:
        why = (
            "obstacle"
            if blocked_grid(m.state, res, robot, unknown_blocks=False)[g]
            else "unknown space"
        )
        return PlanResult(
            False,
            "goal_footprint_blocked",
            f"the robot footprint at the goal would overlap {why}",
            map_version=m.version,
        )
    clear = clearance_m(m.state, res)
    path, expanded = _astar(blocked, margin, clear, res, s, g, p)
    if path is None:
        # Diagnostic only: would a path exist if unknown space were traversable?
        relaxed = blocked_grid(m.state, res, robot, unknown_blocks=False)
        no_margin = np.zeros_like(relaxed)
        alt, _ = (
            _astar(relaxed, no_margin, clearance_m(m.state, res, False), res, s, g, p)
            if not relaxed[s]
            else (None, 0)
        )
        if alt is not None:
            return PlanResult(
                False,
                "blocked_by_unknown",
                "no path through observed free space; a path may exist through unobserved space - observe more",
                expanded=expanded,
                map_version=m.version,
            )
        return PlanResult(
            False,
            "blocked_by_obstacles",
            "observed obstacles block every route for this robot footprint",
            expanded=expanded,
            map_version=m.version,
        )
    margin_cells = int(sum(bool(margin[c]) for c in path))
    if p.smooth:
        # Shortcuts may not enter the margin band unless the raw path already used those cells.
        path = smooth_path(blocked | (margin & ~_cells_mask(margin.shape, path)), path)
    pts = m.center_of(np.asarray(path))
    length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    return PlanResult(
        True,
        "ok",
        "path found in observed free space",
        [tuple(map(int, c)) for c in path],
        pts.round(4).tolist(),
        round(length, 4),
        expanded,
        m.version,
        margin_cells,
    )


def _cells_mask(shape, cells) -> np.ndarray:
    mk = np.zeros(shape, dtype=bool)
    for c in cells:
        mk[c] = True
    return mk


__all__ = [
    "FREE",
    "OCCUPIED",
    "UNKNOWN",
    "PlanParams",
    "PlanResult",
    "RobotSpec",
    "blocked_grid",
    "plan",
]
