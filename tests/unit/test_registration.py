import numpy as np
import open3d as o3d

from clearpath3d.geometry.frames import invert, make_transform, rot_z, transform_points
from clearpath3d.reconstruction.registration import (
    ICPParams,
    p2plane_information,
    register,
    trajectory_errors,
)


def _box_room(rng, n=20000):
    """Points on the inside of a 4x3x2.5 m box plus two inner boxes: well-constrained geometry."""
    pts = []
    for axis, val, lo, hi in [
        (0, 0, (0, 0), (3, 2.5)),
        (0, 4, (0, 0), (3, 2.5)),
        (1, 0, (0, 0), (4, 2.5)),
        (1, 3, (0, 0), (4, 2.5)),
        (2, 0, (0, 0), (4, 3)),
    ]:
        uv = rng.uniform(lo, hi, size=(n // 5, 2))
        p = np.insert(uv, axis, val, axis=1)
        pts.append(p)
    for c in ([1.0, 1.0], [2.8, 2.0]):
        uv = rng.uniform(-0.2, 0.2, size=(800, 2))
        pts.append(np.column_stack([c[0] + uv[:, 0], c[1] + 0.2 * np.ones(800), 0.4 + uv[:, 1]]))
        pts.append(np.column_stack([c[0] + 0.2 * np.ones(800), c[1] + uv[:, 0], 0.4 + uv[:, 1]]))
    return np.concatenate(pts)


def _pcd(p):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))


def test_recovers_known_transform():
    rng = np.random.default_rng(0)
    P = _box_room(rng)
    T_true = make_transform(rot_z(np.radians(8)), [0.06, -0.03, 0.01])  # maps src -> tgt
    src = transform_points(invert(T_true), P) + rng.normal(0, 0.003, P.shape)
    r = register(_pcd(src), _pcd(P), np.eye(4), ICPParams())
    err = invert(T_true) @ r.T_prev_cur
    assert r.accepted, r.reasons
    assert np.linalg.norm(err[:3, 3]) < 0.01


def test_degenerate_corridor_has_low_information_and_is_rejected():
    rng = np.random.default_rng(1)
    # Floor + one long wall only: translation along the wall (x) is unconstrained.
    floor = np.column_stack([rng.uniform(0, 6, 8000), rng.uniform(0, 2, 8000), np.zeros(8000)])
    wall = np.column_stack([rng.uniform(0, 6, 8000), np.full(8000, 2.0), rng.uniform(0, 2, 8000)])
    P = np.concatenate([floor, wall])
    info = p2plane_information(_pcd(P), _pcd(P), np.eye(4), 0.04, 0.12)
    assert info < 0.03
    r = register(_pcd(P + [0.05, 0, 0]), _pcd(P), np.eye(4), ICPParams())
    assert not r.accepted and any("weakly constrained" in s for s in r.reasons)


def test_large_jump_rejected_by_plausibility():
    rng = np.random.default_rng(2)
    P = _box_room(rng)
    T = make_transform(rot_z(np.radians(70)), [0.8, 0.0, 0.0])
    r = register(
        _pcd(transform_points(invert(T), P)), _pcd(P), np.eye(4), ICPParams(global_fallback=False)
    )
    assert not r.accepted


def test_trajectory_errors_zero_for_identical_and_first_frame_alignment():
    ref = [make_transform(rot_z(0.1 * k), [0.1 * k, 0, 0]) for k in range(5)]
    offset = make_transform(rot_z(0.5), [3.0, -1.0, 0.0])
    est = [offset @ T for T in ref]  # same trajectory in another gauge
    e = trajectory_errors(est, ref)
    assert e["ate_rmse_m"] < 1e-9 and e["rpe_trans_rmse_m"] < 1e-9
    est[3] = None  # a rejected frame is skipped, not counted as zero error
    assert trajectory_errors(est, ref)["frames_estimated"] == 4
