"""M1 acceptance: one frame fully explained, checked against the simulator ground truth."""

import json

import numpy as np
import pytest

from clearpath3d.contracts import GroundStatus
from clearpath3d.datasets.base import SensorSequence
from clearpath3d.evaluation.metrics import evaluate_frame
from clearpath3d.runtime.pipeline import analyze_frame, export_analysis


def _run(scene_dir, cfg, tmp_path, name, **kw):
    root = scene_dir(name, **kw)
    seq = SensorSequence(root, cfg.depth.min_depth_m, cfg.depth.max_depth_m)
    fa = analyze_frame(seq[0], cfg)
    out = export_analysis(fa, tmp_path / name)
    return fa, evaluate_frame(root, 0, out), out


def test_room_clear_ground_and_obstacles(scene_dir, cfg, tmp_path):
    fa, rep, out = _run(scene_dir, cfg, tmp_path, "room_clear")
    s = fa.snapshot
    assert s.ground_status == GroundStatus.found
    assert s.output_frame == "map" and s.pose_status.value == "supplied"
    g = rep["ground"]
    assert g["precision"] > 0.97 and g["recall"] > 0.9
    assert g["normal_error_deg"] < 2.0 and g["height_error_at_robot_m"] < 0.03
    assert rep["detections_on_floor"] == 0
    objs = {o["object"]: o for o in rep["obstacles"]}
    for name in ("box_a", "box_b", "table"):
        assert name in objs, f"{name} should be visible in frame 0"
        assert objs[name]["iou_visible"] > 0.5, objs[name]
        assert objs[name]["center_error_visible_m"] < 0.1
    # Exports exist and are parseable.
    json.loads((out / "snapshot.json").read_text())
    assert (out / "points_labelled.ply").stat().st_size > 0


def test_dominant_wall_is_not_ground(scene_dir, cfg, tmp_path):
    fa, rep, _ = _run(scene_dir, cfg, tmp_path, "dominant_wall")
    expl = fa.explanation["ground_candidates"]
    # The largest extracted plane is the wall and it is rejected for its tilt.
    assert not expl[0]["accepted"] and expl[0]["tilt_deg"] > 80
    assert fa.snapshot.ground_status == GroundStatus.found
    g = rep["ground"]
    assert g["pred_ground_on_wall_fraction"] < 0.01
    assert g["normal_error_deg"] < 3.0 and g["height_error_at_robot_m"] < 0.03
    assert g["precision"] > 0.95


def test_no_ground_fails_explicitly(scene_dir, cfg, tmp_path):
    fa, rep, _ = _run(scene_dir, cfg, tmp_path, "no_ground")
    s = fa.snapshot
    assert s.ground_status == GroundStatus.unavailable
    assert s.ground_plane is None
    assert any("ground unavailable" in w for w in s.warnings)
    assert rep["ground"]["pred_ground_points"] == 0
    assert len(s.detections) >= 1  # the wall is still reported as an obstacle


def test_output_in_base_link_without_pose(scene_dir, cfg):
    seq = SensorSequence(scene_dir("room_clear"), use_supplied_poses=False)
    fa = analyze_frame(seq[0], cfg)
    assert fa.snapshot.output_frame == "base_link"
    assert fa.snapshot.pose_status.value == "unavailable"
    n = np.asarray(fa.snapshot.ground_plane.normal)
    assert n[2] > 0.999 and abs(fa.snapshot.ground_plane.offset) < 0.03


def test_dropout_holes_do_not_create_points(scene_dir, cfg):
    seq = SensorSequence(scene_dir("room_clear", hole_count=6, hole_size_px=30, seed=3))
    fr = seq[0]
    assert (~fr.depth_valid).sum() > 6 * 30 * 30 * 0.5
    fa = analyze_frame(fr, cfg)
    assert fr.depth_valid[fa.pixel[:, 0], fa.pixel[:, 1]].all()


def test_empty_frame_is_explicit(cfg):
    from clearpath3d.contracts import Intrinsics, PoseSource, RGBDFrame
    from clearpath3d.geometry.frames import camera_mount

    intr = Intrinsics(fx=100, fy=100, cx=31.5, cy=23.5, width=64, height=48)
    fr = RGBDFrame(
        "s",
        "0",
        0.0,
        np.zeros((48, 64, 3), np.uint8),
        np.zeros((48, 64), np.float32),
        np.zeros((48, 64), bool),
        intr,
        None,
        PoseSource.unavailable,
        "c",
        T_base_camera=camera_mount(0.8, 0.2),
    )
    fa = analyze_frame(fr, cfg)
    assert fa.snapshot.ground_status == GroundStatus.unavailable
    assert fa.snapshot.detections == []
    assert any("no valid depth" in w for w in fa.snapshot.warnings)


@pytest.mark.parametrize("name", ["room_clear"])
def test_deterministic(scene_dir, cfg, name):
    seq = SensorSequence(scene_dir(name))
    a, b = analyze_frame(seq[0], cfg), analyze_frame(seq[0], cfg)
    np.testing.assert_array_equal(a.labels, b.labels)
    assert a.snapshot.detections == b.snapshot.detections


def test_wrong_depth_scale_rejected(scene_dir, tmp_path):
    import shutil

    from clearpath3d.geometry.preprocess import DepthUnitError

    dst = tmp_path / "bad_scale"
    shutil.copytree(scene_dir("room_clear"), dst)
    m = json.loads((dst / "manifest.json").read_text())
    m["depth_scale"] = 1000.0  # metres declared as millimetres-to-metres the wrong way round
    (dst / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(DepthUnitError):
        SensorSequence(dst)[0]


def test_thin_obstacle_survives_preprocessing(scene_dir, cfg, tmp_path):
    """Regression: a 5 cm table leg must not be erased by preprocessing (SOR did this in M1)."""
    fa, rep, _ = _run(scene_dir, cfg, tmp_path, "room_clear")
    table_dets = [d for d in rep["detections"] if d["majority_category"] == "table"]
    assert table_dets
    tall = [
        d
        for d in fa.snapshot.detections
        if d.detection_id in {t["detection_id"] for t in table_dets}
        and d.max_m[2] - d.min_m[2] > 0.6
    ]
    assert tall, "table legs + top should form one obstacle spanning floor to table height"
