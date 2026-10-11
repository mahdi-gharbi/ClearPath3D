import numpy as np

from clearpath3d.mapping.projection import FREE, OCCUPIED, UNKNOWN, Map2D, ProjectionParams
from clearpath3d.planning.astar import PlanParams, RobotSpec, _astar, plan, segment_free

RES = 0.05
POINT = RobotSpec(radius_m=0.0, margin_m=0.0)


def _map(state: np.ndarray) -> Map2D:
    return Map2D(
        state.astype(np.int8),
        np.zeros(2),
        RES,
        0.0,
        np.zeros(state.shape),
        np.ones(state.shape, bool),
        1,
        ProjectionParams(),
    )


def _xy(i, j):
    return ((i + 0.5) * RES, (j + 0.5) * RES)


def test_diagonal_corner_cutting_forbidden():
    s = np.full((3, 3), FREE)
    s[1, 0] = OCCUPIED
    s[0, 1] = OCCUPIED
    r = plan(_map(s), _xy(0, 0), _xy(1, 1), POINT, PlanParams(proximity_weight=0))
    assert not r.success and r.cause == "blocked_by_obstacles"


def test_path_never_crosses_unknown_and_reports_cause():
    s = np.full((20, 5), FREE)
    s[10, :] = UNKNOWN
    r = plan(_map(s), _xy(2, 2), _xy(18, 2), POINT)
    assert not r.success and r.cause == "blocked_by_unknown"
    s[10, :] = OCCUPIED
    r = plan(_map(s), _xy(2, 2), _xy(18, 2), POINT)
    assert r.cause == "blocked_by_obstacles"


def test_goal_causes():
    s = np.full((20, 20), FREE)
    s[15, 15] = UNKNOWN
    s[5, 15] = OCCUPIED
    m = _map(s)
    robot = RobotSpec(radius_m=0.1, margin_m=0.0)
    assert plan(m, _xy(2, 2), _xy(15, 15), robot).cause == "goal_unknown"
    assert plan(m, _xy(2, 2), _xy(5, 15), robot).cause == "goal_occupied"
    assert plan(m, _xy(2, 2), _xy(5, 14), robot).cause == "goal_footprint_blocked"
    assert plan(m, _xy(2, 2), (5.0, 5.0), robot).cause == "goal_outside_map"


def test_footprint_blocks_narrow_gap_for_wide_robot_only():
    s = np.full((40, 40), FREE)
    s[20, :] = OCCUPIED
    s[20, 18:22] = FREE  # 4-cell (0.20 m) gap
    m = _map(s)
    wide = RobotSpec(radius_m=0.15, margin_m=0.0)
    narrow = RobotSpec(radius_m=0.05, margin_m=0.0)
    assert plan(m, _xy(5, 20), _xy(35, 20), wide).cause == "blocked_by_obstacles"
    r = plan(m, _xy(5, 20), _xy(35, 20), narrow)
    assert r.success


def test_astar_cost_optimal_against_dijkstra():
    rng = np.random.default_rng(1)
    s = np.where(rng.random((25, 25)) < 0.25, OCCUPIED, FREE)
    s[0, 0] = s[24, 24] = FREE
    blocked = s == OCCUPIED
    margin = np.zeros_like(blocked)
    clear = np.full(s.shape, np.inf)
    p = PlanParams(proximity_weight=0.0, smooth=False)

    def cost(path):
        return sum(
            RES * np.hypot(a[0] - b[0], a[1] - b[1])
            for a, b in zip(path[:-1], path[1:], strict=True)
        )

    path, _ = _astar(blocked, margin, clear, RES, (0, 0), (24, 24), p)
    if path is None:
        return
    # Dijkstra = A* with a zero heuristic: emulate by scanning all paths' cost via a brute BFS on costs.
    import heapq

    dist = {(0, 0): 0.0}
    heap = [(0.0, (0, 0))]
    while heap:
        d, c = heapq.heappop(heap)
        if d > dist.get(c, np.inf):
            continue
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nb = (c[0] + dx, c[1] + dy)
            if not (0 <= nb[0] < 25 and 0 <= nb[1] < 25) or blocked[nb]:
                continue
            if dx and dy and (blocked[c[0] + dx, c[1]] or blocked[c[0], c[1] + dy]):
                continue
            nd = d + RES * np.hypot(dx, dy)
            if nd < dist.get(nb, np.inf) - 1e-12:
                dist[nb] = nd
                heapq.heappush(heap, (nd, nb))
    assert cost(path) == np.float64(dist[(24, 24)]) or abs(cost(path) - dist[(24, 24)]) < 1e-9


def test_smoothed_path_segments_are_collision_free():
    s = np.full((40, 40), FREE)
    s[10:30, 20] = OCCUPIED
    m = _map(s)
    robot = RobotSpec(radius_m=0.05, margin_m=0.0)
    r = plan(m, _xy(5, 25), _xy(35, 25), robot)
    assert r.success
    from clearpath3d.planning.astar import blocked_grid

    b = blocked_grid(m.state, RES, robot)
    for a, c in zip(r.path_cells[:-1], r.path_cells[1:], strict=True):
        assert segment_free(b, a, c)


def test_start_must_be_free():
    s = np.full((20, 20), FREE)
    s[5, 5] = UNKNOWN
    r = plan(_map(s), _xy(5, 5), _xy(15, 15), POINT)
    assert r.cause == "start_not_free"


def test_outside_map_counts_as_unknown():
    """Regression (held-out rooms larger than the grid): the map border must block like unknown."""
    s = np.full((20, 20), FREE)
    m = _map(s)
    robot = RobotSpec(radius_m=0.1, margin_m=0.0)
    from clearpath3d.planning.astar import blocked_grid

    b = blocked_grid(m.state, RES, robot)
    assert b[0, 10] and b[19, 10] and b[10, 0] and not b[10, 10]
    assert plan(m, _xy(10, 10), _xy(0, 10), robot).cause == "goal_footprint_blocked"
