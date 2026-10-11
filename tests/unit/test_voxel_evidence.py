import numpy as np
import pytest

from clearpath3d.contracts import Intrinsics, PoseSource, RGBDFrame
from clearpath3d.geometry.frames import R_BASE_OPTICAL, make_transform
from clearpath3d.mapping.projection import FREE, OCCUPIED, UNKNOWN, ProjectionParams, project_to_2d
from clearpath3d.mapping.voxel_evidence import EvidenceParams, VoxelEvidenceGrid, dda_traverse

RES = 0.1
GMIN = np.zeros(3)
SHAPE = (30, 30, 30)


def _sampled_voxels(o, e, step=1e-4):
    n = int(np.linalg.norm(e - o) / step) + 1
    pts = o + (e - o) * np.linspace(0, 1, n)[:, None]
    idx = np.floor((pts - GMIN) / RES).astype(int)
    return {tuple(i) for i in idx}


def _to_set(lin):
    return {tuple(int(v) for v in np.unravel_index(i, SHAPE)) for i in lin}


def test_dda_matches_dense_sampling_on_random_rays():
    rng = np.random.default_rng(0)
    for _ in range(40):
        o = rng.uniform(0.3, 2.7, 3)
        e = rng.uniform(0.3, 2.7, 3)
        free, hit = dda_traverse(o, e[None], GMIN, RES, SHAPE)
        sampled = _sampled_voxels(o, e)
        end = tuple(np.floor((e - GMIN) / RES).astype(int))
        assert _to_set(hit) == {end}
        assert _to_set(free) == sampled - {end}


def test_dda_axis_aligned_and_boundary_end():
    o = np.array([0.05, 0.05, 0.05])
    e = np.array([0.55, 0.05, 0.05])  # ends in voxel x=5
    free, hit = dda_traverse(o, e[None], GMIN, RES, SHAPE)
    assert _to_set(free) == {(i, 0, 0) for i in range(5)}
    assert _to_set(hit) == {(5, 0, 0)}
    # End exactly on a voxel face: the hit voxel must be the one the rest of the code assigns to that
    # point (same floor((p - min) / res) rule, incl. floating-point rounding: 0.6/0.1 = 5.999...),
    # it must not also be counted free, and the traversal must be contiguous up to it.
    g = VoxelEvidenceGrid(
        EvidenceParams(resolution_m=RES, bounds_min=(0, 0, 0), bounds_max=(3, 3, 3))
    )
    end = np.array([[0.6, 0.05, 0.05]])
    free, hit = dda_traverse(o, end, GMIN, RES, SHAPE)
    assert list(hit) == list(g.index_of(end))
    hx = _to_set(hit).pop()[0]
    assert _to_set(free) == {(i, 0, 0) for i in range(hx)}


def test_dda_zero_length_ray_frees_nothing():
    o = np.array([0.55, 0.55, 0.55])
    free, hit = dda_traverse(o, o[None], GMIN, RES, SHAPE)
    assert len(free) == 0 and _to_set(hit) == {(5, 5, 5)}


def _frame(depth: np.ndarray, T_map_cam: np.ndarray, t=0.0, fid="0") -> RGBDFrame:
    h, w = depth.shape
    intr = Intrinsics(fx=10.0, fy=10.0, cx=(w - 1) / 2, cy=(h - 1) / 2, width=w, height=h)
    valid = depth > 0
    return RGBDFrame(
        "s",
        fid,
        t,
        np.zeros((h, w, 3), np.uint8),
        depth.astype(np.float32),
        valid,
        intr,
        T_map_cam,
        PoseSource.supplied,
        "c",
    )


def _grid():
    return VoxelEvidenceGrid(
        EvidenceParams(
            resolution_m=0.1,
            bounds_min=(0, 0, 0),
            bounds_max=(3, 3, 3),
            pixel_stride=1,
            max_range_m=5.0,
        )
    )


def _looking_along_x(origin):
    # camera optical z along map +x, x right = map -y, y down = map -z
    return make_transform(R_BASE_OPTICAL, origin)


def test_single_ray_free_before_hit_nothing_behind():
    g = _grid()
    T = _looking_along_x([0.05, 1.55, 1.55])
    g.integrate_frame(_frame(np.array([[2.0]]), T))
    L = g.volume(g.log_odds)
    np.testing.assert_allclose(L[0:20, 15, 15], g.p.delta_free)  # voxels x=0..1.95 free
    assert L[20, 15, 15] == pytest.approx(g.p.delta_occ)  # hit voxel (x=2.05)
    assert np.all(L[21:, 15, 15] == 0)  # nothing behind the hit
    assert g.obs_count.sum() == 21


def test_invalid_depth_frees_nothing():
    g = _grid()
    T = _looking_along_x([0.05, 1.55, 1.55])
    g.integrate_frame(_frame(np.zeros((3, 3)), T))  # a hole everywhere
    assert np.all(g.log_odds == 0) and g.obs_count.sum() == 0


def test_correlated_rays_capped_per_frame():
    g = _grid()
    T = _looking_along_x([0.05, 1.55, 1.55])
    # 9 nearly identical rays (tiny image): the voxels they share get ONE update in this frame.
    g.integrate_frame(_frame(np.full((3, 3), 0.5), T))
    assert g.log_odds.min() >= g.p.delta_free - 1e-6
    assert g.log_odds.max() <= g.p.delta_occ + 1e-6


def test_clamping_bounds_log_odds():
    g = _grid()
    T = _looking_along_x([0.05, 1.55, 1.55])
    for k in range(20):
        g.integrate_frame(_frame(np.array([[2.0]]), T, t=k, fid=str(k)))
    assert g.log_odds.max() == pytest.approx(g.p.l_max)
    assert g.log_odds.min() == pytest.approx(g.p.l_min)


# ----------------------------------------------------------------- 2D projection semantics


def _column_grid():
    g = VoxelEvidenceGrid(
        EvidenceParams(resolution_m=0.05, bounds_min=(0, 0, -0.1), bounds_max=(0.5, 0.5, 0.6))
    )
    return g


def _set_column(g, i, j, z_lo, z_hi, value, t=0.0):
    zc = g.min[2] + (np.arange(g.shape[2]) + 0.5) * g.p.resolution_m
    for k in np.nonzero((zc >= z_lo) & (zc <= z_hi))[0]:
        lin = np.ravel_multi_index((i, j, k), g.shape)
        g.log_odds[lin] = value
        g.obs_count[lin] = 1
        g.hit_count[lin] = 1 if value > 0 else 0  # consistent: occupied evidence comes from hits
        g.last_seen_s[lin] = t


def test_floor_only_is_unknown_not_free():
    g = _column_grid()
    _set_column(g, 2, 2, -0.03, 0.03, 2.0)  # floor hit, band never observed
    m = project_to_2d(g, 0.0, ProjectionParams())
    assert m.state[2, 2] == UNKNOWN


def test_free_requires_ground_support_and_band_coverage():
    g = _column_grid()
    _set_column(g, 1, 1, 0.06, 0.40, -1.0)  # band free but no floor seen
    _set_column(g, 2, 2, -0.03, 0.03, 2.0)
    _set_column(g, 2, 2, 0.06, 0.40, -1.0)  # floor + band free
    m = project_to_2d(g, 0.0, ProjectionParams())
    assert m.state[1, 1] == UNKNOWN
    assert m.state[2, 2] == FREE


def test_obstacle_in_band_is_occupied_low_step_ignored():
    g = _column_grid()
    _set_column(g, 3, 3, -0.03, 0.03, 2.0)
    _set_column(g, 3, 3, 0.06, 0.40, -1.0)
    _set_column(g, 3, 3, 0.25, 0.30, 2.0)
    _set_column(g, 4, 4, -0.03, 0.03, 2.0)
    _set_column(g, 4, 4, 0.06, 0.40, -1.0)
    m = project_to_2d(g, 0.0, ProjectionParams())
    assert m.state[3, 3] == OCCUPIED
    assert m.state[4, 4] == FREE


def test_expiry_returns_to_unknown_never_free():
    g = _column_grid()
    _set_column(g, 2, 2, -0.03, 0.03, 2.0, t=0.0)
    _set_column(g, 2, 2, 0.06, 0.40, -1.0, t=0.0)
    _set_column(g, 3, 3, -0.03, 0.03, 2.0, t=0.0)
    _set_column(g, 3, 3, 0.25, 0.30, 2.0, t=0.0)
    p = ProjectionParams(expiry_s=5.0)
    m = project_to_2d(g, 0.0, p, now_s=100.0)
    assert m.state[2, 2] == UNKNOWN and m.state[3, 3] == UNKNOWN


def test_ground_support_survives_later_rays():
    """Regression (found with the interactive drive): support must not be lost when a later sensor ray
    updates the band voxels of a column the robot stood in, or grazes a floor voxel once."""
    from clearpath3d.geometry.frames import planar_pose

    g = _column_grid()
    g.mark_robot_footprint(planar_pose(0.25, 0.25, 0.0), 0.05, 0.4, 0.0, "f0")
    i = g.index_of(np.array([[0.25, 0.25, 0.1]])).tolist()[0]
    ij = tuple(int(v) for v in np.unravel_index(i, g.shape)[:2])
    _set_column(g, *ij, 0.06, 0.40, -1.0)  # later free rays overwrite last_update in the band
    m = project_to_2d(g, 0.0, ProjectionParams())
    assert m.ground_support[ij] and m.state[ij] == FREE
    # Floor seen once, then one grazing free update: still supported (not confirmed free).
    _set_column(g, 4, 4, -0.03, 0.03, 0.85)
    _set_column(g, 4, 4, 0.06, 0.40, -1.0)
    lin = [
        np.ravel_multi_index((4, 4, k), g.shape)
        for k in range(g.shape[2])
        if abs(g.min[2] + (k + 0.5) * 0.05) <= 0.05
    ]
    g.log_odds[lin] = 0.45  # 0.85 then one free update of -0.40
    assert project_to_2d(g, 0.0, ProjectionParams()).state[4, 4] == FREE
