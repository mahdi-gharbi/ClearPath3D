"""Redwood sample: convention checks on real data. Skipped unless downloaded explicitly
(``clearpath3d download-redwood``); the test suite never downloads."""

from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
from scipy.spatial import cKDTree

from clearpath3d.datasets.base import SensorSequence
from clearpath3d.datasets.redwood import T_MAP_WORLD, convert, read_log
from clearpath3d.geometry.frames import compose, invert, transform_points
from clearpath3d.reconstruction.consistency import reprojection_residual
from clearpath3d.reconstruction.run import reconstruct
from clearpath3d.reconstruction.tsdf import TSDFParams

SRC = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "downloads"
    / "extract"
    / "SampleRedwoodRGBDImages"
)
pytestmark = pytest.mark.skipif(not SRC.exists(), reason="Redwood sample not downloaded")


@pytest.fixture(scope="module")
def seq(tmp_path_factory):
    return SensorSequence(convert(SRC, tmp_path_factory.mktemp("redwood")), max_depth_m=4.0)


def test_depth_units_and_calibration(seq):
    f = seq[0]
    assert f.intrinsics.width == 640 and f.intrinsics.fx == 525.0
    assert 0.5 < float(np.median(f.depth_m[f.depth_valid])) < 4.0  # mm converted to m
    assert f.T_base_camera is None and f.pose_source.value == "supplied"


def test_pose_direction_from_reprojection(seq):
    good = reprojection_residual(seq[0], seq[4])
    inv = reprojection_residual(
        seq[0], seq[4], invert(seq[0].T_map_camera), invert(seq[4].T_map_camera)
    )
    assert good["median_abs_dz_m"] < 0.01 and good["frac_below_2cm"] > 0.85
    assert inv["median_abs_dz_m"] > 0.05


def test_reconstruction_matches_open3d_reference(seq, tmp_path):
    p = TSDFParams(voxel_m=4.0 / 512, sdf_trunc_m=0.04, max_depth_m=4.0, use_keyframes=False)
    rep = reconstruct(seq, p, tmp_path)
    assert rep["tsdf"]["frames_integrated"] == 5
    ours = np.asarray(o3d.io.read_point_cloud(str(tmp_path / "tsdf_cloud.ply")).points)
    # The shipped reference was fused in the odometry.log world; bring it into our map.
    traj, odo = read_log(SRC / "trajectory.log"), read_log(SRC / "odometry.log")
    T = compose(T_MAP_WORLD, traj[0], invert(odo[0]))
    ref = transform_points(
        T, np.asarray(o3d.io.read_point_cloud(str(SRC / "example_tsdf_pcd.ply")).points)
    )
    d, _ = cKDTree(ours).query(ref)
    assert np.median(d) < 0.01 and (d < 0.02).mean() > 0.95


def test_map_is_gravity_aligned(seq, tmp_path):
    p = TSDFParams(voxel_m=0.02, sdf_trunc_m=0.06, max_depth_m=4.0, use_keyframes=False)
    reconstruct(seq, p, tmp_path)
    pc = o3d.io.read_point_cloud(str(tmp_path / "tsdf_cloud.ply"))
    o3d.utility.random.seed(0)
    (a, b, c, _), _ = pc.segment_plane(0.02, 3, 1000)
    assert abs(c) > np.cos(np.radians(3))  # dominant plane (floor) is horizontal in map
