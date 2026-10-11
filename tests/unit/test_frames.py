import numpy as np
import pytest

from clearpath3d.geometry.frames import (
    R_BASE_OPTICAL,
    assert_rigid,
    camera_mount,
    compose,
    invert,
    is_rigid,
    make_transform,
    planar_pose,
    rot_x,
    rot_y,
    rot_z,
    transform_points,
)


def random_transform(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return make_transform(R, rng.normal(size=3) * 3)


def test_inverse_and_composition_properties():
    rng = np.random.default_rng(0)
    for _ in range(200):
        T_ab, T_bc = random_transform(rng), random_transform(rng)
        assert is_rigid(T_ab)
        np.testing.assert_allclose(compose(T_ab, invert(T_ab)), np.eye(4), atol=1e-12)
        np.testing.assert_allclose(invert(T_ab), np.linalg.inv(T_ab), atol=1e-10)
        T_ac = compose(T_ab, T_bc)
        p_c = rng.normal(size=(5, 3))
        np.testing.assert_allclose(
            transform_points(T_ac, p_c),
            transform_points(T_ab, transform_points(T_bc, p_c)),
            atol=1e-10,
        )
        np.testing.assert_allclose(invert(T_ac), compose(invert(T_bc), invert(T_ab)), atol=1e-10)


def test_rotation_validity():
    for R in (R_BASE_OPTICAL, rot_x(0.3), rot_y(-1.1), rot_z(2.0), camera_mount(0.8, 0.26)[:3, :3]):
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)


def test_reflection_rejected():
    T = np.eye(4)
    T[0, 0] = -1
    assert not is_rigid(T)
    with pytest.raises(ValueError):
        assert_rigid(T)


def test_optical_axes_in_base():
    # optical z (forward) -> base +x ; optical x (image right) -> base -y ; optical y (down) -> base -z
    np.testing.assert_allclose(R_BASE_OPTICAL @ [0, 0, 1], [1, 0, 0])
    np.testing.assert_allclose(R_BASE_OPTICAL @ [1, 0, 0], [0, -1, 0])
    np.testing.assert_allclose(R_BASE_OPTICAL @ [0, 1, 0], [0, 0, -1])


def test_point_in_front_of_camera_known_location():
    # Camera 0.8 m high, pitched down 30 deg. The optical axis hits the floor at 0.8/tan(30deg).
    T_base_cam = camera_mount(0.8, np.radians(30.0))
    dist = 0.8 / np.sin(np.radians(30.0))  # along the optical axis
    p_base = transform_points(T_base_cam, [[0.0, 0.0, dist]])[0]
    np.testing.assert_allclose(p_base, [0.8 / np.tan(np.radians(30.0)), 0.0, 0.0], atol=1e-9)
    # A point to the image right is on the robot's right (base -y).
    p_right = transform_points(T_base_cam, [[0.5, 0.0, 2.0]])[0]
    assert p_right[1] < 0


def test_planar_pose_and_translation_semantics():
    T_map_base = planar_pose(2.0, 1.0, np.pi / 2)  # facing map +y
    # translation column = base origin in map
    np.testing.assert_allclose(T_map_base[:3, 3], [2.0, 1.0, 0.0])
    # 1 m forward in base is +y in map
    np.testing.assert_allclose(
        transform_points(T_map_base, [[1, 0, 0]])[0], [2.0, 2.0, 0.0], atol=1e-12
    )
