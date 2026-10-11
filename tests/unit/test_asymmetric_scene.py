"""Convention tests with real ray casting: a valid rotation is not necessarily the right one."""

import numpy as np

from clearpath3d.datasets.synthetic import (
    CameraSpec,
    Perturbations,
    Renderer,
    SceneObject,
    SceneSpec,
    TrajectoryPoint,
)
from clearpath3d.geometry.frames import transform_points
from clearpath3d.geometry.projection import backproject

NO_NOISE = Perturbations(depth_noise_sigma0_m=0, depth_noise_quadratic=0, dropout_prob=0)


ORIGIN = TrajectoryPoint(0, 0, 0, 0)


def _render(objects, cam, tp=ORIGIN):
    spec = SceneSpec("t", "", objects, [tp], camera=cam, perturbations=NO_NOISE)
    r = Renderer(spec)
    return r, r.render(tp, np.random.default_rng(0))


def test_fronto_parallel_wall_gives_constant_z_depth():
    cam = CameraSpec(
        width=64, height=48, mount_pitch_down_deg=0.0, mount_forward_m=0.1, mount_height_m=1.0
    )
    wall = SceneObject("wall", "wall", (3.0, -10, -10), (3.1, 10, 10), (1, 1, 1))
    _, fr = _render([wall], cam)
    assert (fr.instance == 0).all()
    # z-depth (not ray length) is constant = 3.0 - 0.1 everywhere, including image corners.
    np.testing.assert_allclose(fr.depth_clean, 2.9, atol=1e-4)


def test_left_box_appears_left_and_backprojects_left():
    cam = CameraSpec(width=96, height=72, mount_pitch_down_deg=10.0)
    floor = SceneObject("floor", "floor", (-1, -5, -0.05), (10, 5, 0.0), (0.5, 0.5, 0.5))
    left_box = SceneObject("box", "box", (2.0, 0.6, 0.0), (2.4, 1.0, 0.6), (1, 0, 0))
    r, fr = _render([floor, left_box], cam)
    vs, us = np.nonzero(fr.instance == 1)
    assert len(us) > 20
    assert us.max() < r.intr.cx, (
        "box on the robot's left (+y) must be in the left half of the image"
    )
    # Red colour lands on the same pixels (colour/depth registration).
    assert (fr.color[vs, us, 0] > fr.color[vs, us, 1]).all()
    pts, pix = backproject(fr.depth_clean, fr.instance == 1, r.intr)
    p_map = transform_points(fr.T_map_camera, pts)
    assert (p_map[:, 1] > 0.55).all() and (p_map[:, 1] < 1.05).all()
    assert (p_map[:, 0] > 1.95).all() and (p_map[:, 0] < 2.45).all()
    assert (p_map[:, 2] > -0.01).all() and (p_map[:, 2] < 0.61).all()
    # Floor points reconstruct at z = 0 in map.
    fp, _ = backproject(fr.depth_clean, fr.instance == 0, r.intr)
    np.testing.assert_allclose(transform_points(fr.T_map_camera, fp)[:, 2], 0.0, atol=1e-4)


def test_rotated_robot_still_consistent():
    # Robot at (1, 1) facing map +y; a box at map (1.2, 3) is straight ahead, slightly to the right.
    cam = CameraSpec(width=96, height=72, mount_pitch_down_deg=10.0)
    floor = SceneObject("floor", "floor", (-5, -5, -0.05), (10, 10, 0.0), (0.5, 0.5, 0.5))
    box = SceneObject("box", "box", (1.1, 2.9, 0.0), (1.5, 3.1, 0.5), (1, 0, 0))
    r, fr = _render([floor, box], cam, TrajectoryPoint(0, 1.0, 1.0, np.pi / 2))
    _, us = np.nonzero(fr.instance == 1)
    assert us.mean() > r.intr.cx  # map +x is the robot's right when facing +y
    pts, _ = backproject(fr.depth_clean, fr.instance == 1, r.intr)
    c = transform_points(fr.T_map_camera, pts).mean(axis=0)
    assert 1.05 < c[0] < 1.55 and 2.85 < c[1] < 3.15
