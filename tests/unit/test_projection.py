import numpy as np
import pytest

from clearpath3d.contracts import FrameValidationError, Intrinsics, PoseSource, RGBDFrame
from clearpath3d.geometry.preprocess import (
    DepthUnitError,
    check_depth_plausibility,
    valid_depth_mask,
)
from clearpath3d.geometry.projection import (
    backproject,
    project,
    ray_distance_to_z_depth,
    z_depth_to_ray_distance,
)

INTR = Intrinsics(fx=200.0, fy=210.0, cx=79.5, cy=59.5, width=160, height=120)


def test_backproject_project_roundtrip():
    rng = np.random.default_rng(1)
    depth = rng.uniform(0.5, 6.0, size=(120, 160)).astype(np.float32)
    valid = np.ones_like(depth, dtype=bool)
    pts, pix = backproject(depth, valid, INTR)
    uv, front = project(pts, INTR)
    assert front.all()
    np.testing.assert_allclose(uv[:, 0], pix[:, 1], atol=1e-4)
    np.testing.assert_allclose(uv[:, 1], pix[:, 0], atol=1e-4)
    np.testing.assert_allclose(pts[:, 2], depth[pix[:, 0], pix[:, 1]], rtol=1e-6)


def test_principal_point_maps_to_optical_axis():
    pts, _ = backproject(
        np.full((1, 1), 2.0, np.float32),
        np.ones((1, 1), bool),
        Intrinsics(fx=100, fy=100, cx=0, cy=0, width=1, height=1),
    )
    np.testing.assert_allclose(pts[0], [0, 0, 2.0])


def test_image_axes_directions():
    depth = np.full((120, 160), 2.0, np.float32)
    valid = np.zeros_like(depth, dtype=bool)
    valid[0, 159] = True  # top-right pixel
    pts, _ = backproject(depth, valid, INTR)
    assert pts[0, 0] > 0 and pts[0, 1] < 0  # right of and above the optical axis


def test_ray_distance_conversion_roundtrip():
    z = np.full((120, 160), 3.0, np.float32)
    r = z_depth_to_ray_distance(z, INTR)
    assert r[0, 0] > 3.0 and r[60, 80] == pytest.approx(3.0, abs=1e-3)
    np.testing.assert_allclose(ray_distance_to_z_depth(r, INTR), z, rtol=1e-6)


def test_invalid_depth_never_valid():
    d = np.array([[0.0, np.nan, np.inf, -1.0, 0.1, 9.0, 2.0]])
    np.testing.assert_array_equal(valid_depth_mask(d, 0.2, 8.0), [[0, 0, 0, 0, 0, 0, 1]])
    pts, _ = backproject(
        d.astype(np.float32),
        valid_depth_mask(d, 0.2, 8.0),
        Intrinsics(fx=1, fy=1, cx=0, cy=0, width=7, height=1),
    )
    assert len(pts) == 1


def test_millimetre_depth_detected():
    with pytest.raises(DepthUnitError, match="millimetres"):
        check_depth_plausibility(np.full((10, 10), 2500.0))
    with pytest.raises(DepthUnitError):
        check_depth_plausibility(np.zeros((10, 10)))


def _frame(**kw):
    base = dict(
        sequence_id="s",
        frame_id="0",
        timestamp_s=0.0,
        color_rgb=np.zeros((120, 160, 3), np.uint8),
        depth_m=np.ones((120, 160), np.float32),
        depth_valid=np.ones((120, 160), bool),
        intrinsics=INTR,
        T_map_camera=None,
        pose_source=PoseSource.unavailable,
        calibration_id="c",
    )
    base.update(kw)
    return RGBDFrame(**base)


def test_frame_validation():
    _frame()
    with pytest.raises(FrameValidationError, match="does not match intrinsics"):
        _frame(depth_m=np.ones((100, 160), np.float32))
    with pytest.raises(FrameValidationError):
        _frame(pose_source=PoseSource.supplied)  # pose missing
    with pytest.raises(FrameValidationError):
        _frame(T_map_camera=np.eye(4))  # pose present but declared unavailable
    bad = np.ones((120, 160), np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(FrameValidationError, match="non-finite"):
        _frame(depth_m=bad)
    flipped = np.eye(4)
    flipped[2, 2] = -1
    with pytest.raises(ValueError):
        _frame(T_map_camera=flipped, pose_source=PoseSource.supplied)
