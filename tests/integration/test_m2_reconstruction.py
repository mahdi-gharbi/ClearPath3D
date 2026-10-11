"""M2: TSDF fusion with supplied poses on a synthetic scan; extrinsic convention; determinism."""

import json

import numpy as np
import open3d as o3d
import pytest

import clearpath3d.reconstruction.tsdf as tsdf_mod
from clearpath3d.datasets.base import SensorSequence
from clearpath3d.evaluation.reconstruction import evaluate_reconstruction
from clearpath3d.geometry.frames import invert
from clearpath3d.reconstruction.consistency import reprojection_residual
from clearpath3d.reconstruction.run import reconstruct
from clearpath3d.reconstruction.tsdf import TSDFFusion, TSDFParams

LOWRES = dict(width=160, height=120)
PARAMS = TSDFParams(voxel_m=0.03, sdf_trunc_m=0.1)


@pytest.fixture(scope="module")
def scan(scene_dir):
    return scene_dir("room_scan", **LOWRES)


def test_tsdf_backend_produces_surface(scan):
    """Guards against the Open3D 0.20.0 Windows regression (integration produced nothing)."""
    seq = SensorSequence(scan)
    f = TSDFFusion(seq.intrinsics, PARAMS)
    assert f.integrate(seq[0])
    _, cloud = f.extract()
    assert len(cloud.points) > 1000


def test_reconstruction_matches_gt_and_reports_pose_source(scan, tmp_path):
    rep = reconstruct(SensorSequence(scan), PARAMS, tmp_path)
    assert rep["pose_source"] == "supplied"
    assert rep["tsdf"]["frames_integrated"] >= 20
    ev = evaluate_reconstruction(scan, tmp_path)
    assert ev["tsdf"]["accuracy"]["median_m"] < 0.01
    assert ev["tsdf"]["accuracy"]["frac_below_5cm"] > 0.97
    assert ev["tsdf"]["completeness"]["frac_below_5cm"] > 0.95
    # TSDF averages noise: more accurate than the raw point union.
    assert ev["tsdf"]["accuracy"]["p95_m"] < ev["union_baseline"]["accuracy"]["p95_m"]


def test_inverted_extrinsic_is_detected(scan, tmp_path, monkeypatch):
    """Passing T_map_camera instead of T_camera_map must give a visibly wrong reconstruction."""
    monkeypatch.setattr(tsdf_mod, "open3d_extrinsic", lambda T: T)
    reconstruct(SensorSequence(scan), PARAMS, tmp_path)
    ev = evaluate_reconstruction(scan, tmp_path)
    assert ev["tsdf"]["accuracy"]["median_m"] > 0.1


def test_open3d_extrinsic_convention():
    T_map_cam = np.eye(4)
    T_map_cam[:3, 3] = [1.0, 2.0, 0.5]
    E = tsdf_mod.open3d_extrinsic(T_map_cam)
    # The camera centre (map) must map to the camera origin.
    np.testing.assert_allclose(E @ [1.0, 2.0, 0.5, 1.0], [0, 0, 0, 1])
    np.testing.assert_allclose(E, invert(T_map_cam))


def test_two_runs_equivalent(scan, tmp_path):
    reconstruct(SensorSequence(scan), PARAMS, tmp_path / "a")
    reconstruct(SensorSequence(scan), PARAMS, tmp_path / "b")
    va = np.asarray(o3d.io.read_triangle_mesh(str(tmp_path / "a" / "tsdf_mesh.ply")).vertices)
    vb = np.asarray(o3d.io.read_triangle_mesh(str(tmp_path / "b" / "tsdf_mesh.ply")).vertices)
    assert va.shape == vb.shape
    np.testing.assert_allclose(va, vb, atol=1e-6)  # tolerance: 1 micrometre per vertex


def test_frames_without_pose_are_not_integrated(scan, tmp_path):
    seq = SensorSequence(scan, use_supplied_poses=False)
    f = TSDFFusion(seq.intrinsics, PARAMS)
    assert not any(f.integrate(seq[i]) for i in range(3))
    assert f.report.frames_rejected_no_pose == 3


def test_keyframe_selection_skips_static_frames(scan):
    seq = SensorSequence(scan)
    f = TSDFFusion(seq.intrinsics, PARAMS)
    fr = seq[0]
    assert [f.integrate(fr) for _ in range(4)] == [True, False, False, False]


def test_synthetic_poses_consistent_with_depth(scan):
    seq = SensorSequence(scan)
    good = reprojection_residual(seq[0], seq[1])
    bad = reprojection_residual(seq[0], seq[1], T_map_cam_b=invert(seq[1].T_map_camera))
    assert good["median_abs_dz_m"] < 0.01
    assert bad["median_abs_dz_m"] is None or bad["median_abs_dz_m"] > 0.05


def test_report_is_json(scan, tmp_path):
    reconstruct(SensorSequence(scan), PARAMS, tmp_path)
    rep = json.loads((tmp_path / "reconstruction_report.json").read_text())
    assert (
        "peak_working_set_mb" in rep["memory"]["after_tsdf"]
        or "rss_mb" in rep["memory"]["after_tsdf"]
    )
    assert rep["tsdf"]["integrate"]["n"] == rep["tsdf"]["frames_integrated"]
