"""ROS conversion functions (pure Python). These tests do NOT establish a working ROS integration;
that requires a real run in ROS 2 (see PROGRESS.md)."""

import numpy as np
import pytest
from clearpath3d_ros.conversions import (
    occupancy_grid_fields,
    path_poses,
    pointcloud2_fields,
    stamp_from_seconds,
    track_markers,
    transform_fields,
)
from scipy.spatial.transform import Rotation

from clearpath3d.geometry.frames import camera_mount, planar_pose, transform_points


def test_occupancy_grid_index_and_origin():
    st = np.full((4, 3), -1, dtype=np.int8)  # nx=4 (x), ny=3 (y)
    st[3, 0] = 100  # x=3, y=0
    st[0, 2] = 0  # x=0, y=2
    g = occupancy_grid_fields(st, (-0.5, -0.25), 0.05, 0.01)
    assert g["width"] == 4 and g["height"] == 3
    d = g["data"]
    assert d[0 * 4 + 3] == 100 and d[2 * 4 + 0] == 0 and d.count(-1) == 10
    assert g["origin_position"] == [-0.5, -0.25, 0.01]
    with pytest.raises(ValueError):
        occupancy_grid_fields(np.full((2, 2), 50, np.int8), (0, 0), 0.05, 0.0)


def test_pointcloud2_layout_and_rgb_packing():
    xyz = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.0]], np.float32)
    rgb = np.array([[255, 0, 16], [1, 2, 3]], np.uint8)
    f = pointcloud2_fields(xyz, rgb)
    assert f["point_step"] == 16 and f["row_step"] == 32 and len(f["data"]) == 32
    rec = np.frombuffer(f["data"], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])
    np.testing.assert_allclose(np.stack([rec["x"], rec["y"], rec["z"]], 1), xyz)
    assert rec["rgb"][0] == 0xFF0010 and rec["rgb"][1] == 0x010203
    assert [n for n, *_ in f["fields"]] == ["x", "y", "z", "rgb"]


def test_transform_fields_roundtrip():
    T = planar_pose(1.0, 2.0, 0.7) @ camera_mount(0.8, 0.26, 0.1)
    f = transform_fields(T)
    R = Rotation.from_quat(f["rotation"]).as_matrix()  # x, y, z, w
    np.testing.assert_allclose(R, T[:3, :3], atol=1e-9)
    np.testing.assert_allclose(f["translation"], T[:3, 3])
    # The published child->parent transform maps child points into the parent like T_parent_child.
    p = np.array([[0.0, 0.0, 2.0]])
    np.testing.assert_allclose(p @ R.T + f["translation"], transform_points(T, p))


def test_stamps_keep_source_time():
    assert stamp_from_seconds(0.0) == (0, 0)
    assert stamp_from_seconds(3.2) == (3, 200000000)
    assert stamp_from_seconds(1.9999999999) == (2, 0)
    with pytest.raises(ValueError):
        stamp_from_seconds(-1.0)


def test_markers_and_path():
    m = track_markers(
        [
            {
                "track_id": 3,
                "center_m": [1, 2, 0.15],
                "visible_extent_m": [0.3, 0.2, 0.0],
                "lifecycle": "confirmed",
            }
        ]
    )
    assert [x["id"] for x in m] == [6, 7] and m[0]["scale"][2] == 0.02
    poses = path_poses([[0, 0], [1, 0], [1, 1]], 0.0)
    yaw = [Rotation.from_quat(p["orientation"]).as_euler("xyz")[2] for p in poses]
    assert yaw[0] == pytest.approx(0.0) and yaw[2] == pytest.approx(np.pi / 2)
