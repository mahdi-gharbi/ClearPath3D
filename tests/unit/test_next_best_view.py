import numpy as np

from clearpath3d.contracts import Intrinsics
from clearpath3d.geometry.frames import R_BASE_OPTICAL, camera_mount, make_transform
from clearpath3d.mapping.projection import ProjectionParams, project_to_2d
from clearpath3d.mapping.voxel_evidence import EvidenceParams, VoxelEvidenceGrid
from clearpath3d.planning.astar import RobotSpec, plan
from clearpath3d.planning.next_best_view import NBVParams, predicted_gain, propose_viewpoint

INTR = Intrinsics(fx=60, fy=60, cx=31.5, cy=23.5, width=64, height=48)
ONE_RAY = np.array([[0.0, 0.0, 1.0]])  # along the optical axis


def _grid():
    return VoxelEvidenceGrid(
        EvidenceParams(resolution_m=0.1, bounds_min=(0, 0, 0), bounds_max=(3, 1, 1))
    )


def _set(g, x0, x1, value, count=1):
    for x in np.arange(x0 + 0.05, x1, 0.1):
        lin = g.index_of(np.array([[x, 0.55, 0.25]]))[0]
        g.log_odds[lin] = value
        g.obs_count[lin] = count


def _T():  # camera at (0.05, 0.55, 0.25) looking along map +x
    return make_transform(R_BASE_OPTICAL, [0.05, 0.55, 0.25])


def test_first_unknown_counts_once_and_stops():
    g = _grid()
    _set(g, 0.0, 1.0, -1.0, 2)  # free corridor, then never-observed voxels from x = 1.0
    gain = predicted_gain(g, _T(), ONE_RAY, 0.85, -0.39, 0.0, 1.0, 3.0, 0.05)
    assert gain == 1


def test_occupied_or_ambiguous_blocks_gain():
    g = _grid()
    _set(g, 0.0, 0.5, -1.0, 2)
    _set(g, 0.5, 0.6, 2.0, 3)  # occupied wall
    assert predicted_gain(g, _T(), ONE_RAY, 0.85, -0.39, 0.0, 1.0, 3.0, 0.05) == 0
    _set(g, 0.5, 0.6, 0.2, 5)  # observed but ambiguous: stops the ray, no gain
    assert predicted_gain(g, _T(), ONE_RAY, 0.85, -0.39, 0.0, 1.0, 3.0, 0.05) == 0


def test_gain_outside_traversability_slab_ignored():
    g = _grid()
    _set(g, 0.0, 1.0, -1.0, 2)
    assert (
        predicted_gain(g, _T(), ONE_RAY, 0.85, -0.39, 0.0, 0.1, 3.0, 0.05) == 0
    )  # slab below z=0.25


def _closed_room_grid():
    """A small fully observed room: floor hit everywhere, band free inside, walls occupied."""
    g = VoxelEvidenceGrid(
        EvidenceParams(resolution_m=0.05, bounds_min=(0, 0, -0.1), bounds_max=(2, 2, 1.0))
    )
    L = g.volume(g.log_odds)
    C = g.volume(g.obs_count)
    nx, ny, nz = g.shape
    zc = g.min[2] + (np.arange(nz) + 0.5) * 0.05
    floor = np.abs(zc) <= 0.05
    L[:, :, floor] = 2.0
    L[:, :, ~floor] = -2.0
    L[:3, :, :] = L[-3:, :, :] = 2.0
    L[:, :3, :] = L[:, -3:, :] = 2.0
    L[:, :, zc < -0.05] = 2.0
    C[:] = 3
    g.volume(g.hit_count)[:] = np.where(L > 0, 3, 0)  # occupied voxels were hit
    return g


def test_no_informative_candidate_in_fully_observed_closed_room():
    g = _closed_room_grid()
    m = project_to_2d(g, 0.0, ProjectionParams())
    T_bc = camera_mount(0.5, 0.2)
    r = propose_viewpoint(
        g,
        m,
        (1.0, 1.0),
        0.0,
        RobotSpec(0.2, 0.4, 0.0),
        T_bc,
        INTR,
        0.85,
        -0.39,
        NBVParams(min_gain_voxels=1, max_range_m=2.0),
    )
    assert r.status == "no_informative_candidate" and r.best is None and r.candidates


def test_no_map_is_explicit():
    r = propose_viewpoint(
        _grid(), None, (0, 0), 0.0, RobotSpec(), camera_mount(0.5, 0.2), INTR, 0.85, -0.39
    )
    assert r.status == "no_map"


def test_candidates_are_reachable_through_known_free():
    g = _closed_room_grid()
    # open an unknown pocket in the corner so some viewpoints are informative
    L = g.volume(g.log_odds)
    C = g.volume(g.obs_count)
    L[25:37, 25:37, 3:12] = 0.0
    C[25:37, 25:37, 3:12] = 0
    m = project_to_2d(g, 0.0, ProjectionParams())
    robot = RobotSpec(0.15, 0.4, 0.0)
    r = propose_viewpoint(
        g,
        m,
        (0.5, 0.5),
        0.0,
        robot,
        camera_mount(0.5, 0.2),
        INTR,
        0.85,
        -0.39,
        NBVParams(min_gain_voxels=1, max_range_m=2.0, candidate_spacing_m=0.3),
    )
    assert r.status == "ok" and r.best.gain_voxels >= 1
    for c in r.candidates[:: max(1, len(r.candidates) // 8)]:
        if (c.x, c.y) != tuple(m.center_of(m.cell_of(np.array([0.5, 0.5])))):
            assert plan(m, (0.5, 0.5), (c.x, c.y), robot).success
