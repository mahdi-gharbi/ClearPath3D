"""M3 acceptance on synthetic scenes (pipeline reads sensor_export only; GT used by the evaluator)."""

import shutil

import numpy as np
import pytest

from clearpath3d.datasets.base import SensorSequence
from clearpath3d.evaluation.ground_truth import GroundTruth
from clearpath3d.evaluation.mapping import evaluate_map, path_collisions
from clearpath3d.mapping.projection import FREE, OCCUPIED, UNKNOWN
from clearpath3d.planning.astar import RobotSpec, blocked_grid, plan
from clearpath3d.runtime.replay import ReplayConfig, SequenceProcessor

ROBOT = RobotSpec(radius_m=0.25, height_m=0.40, margin_m=0.05)


def _process(root, cfg, upto=None, snapshot_at=None):
    seq = SensorSequence(root)
    proc = SequenceProcessor(
        ReplayConfig(pipeline=cfg, robot=ROBOT, tsdf_enabled=False), seq.intrinsics
    )
    mid = None
    n = len(seq) if upto is None else upto
    for i in range(n):
        proc.step(seq[i])
        if snapshot_at is not None and i + 1 == snapshot_at:
            mid = (proc.map2d(), seq[i].timestamp_s)
    return seq, proc, mid


@pytest.fixture(scope="module")
def room_scan(scene_dir, cfg):
    root = scene_dir("room_scan")
    seq, proc, _ = _process(root, cfg)
    return root, seq, proc, proc.map2d()


def test_map_safety_and_path_to_corridor(room_scan):
    root, seq, proc, m = room_scan
    gt = GroundTruth(root)
    t = seq[len(seq) - 1].timestamp_s
    ev = evaluate_map(gt, m, t, ROBOT.radius_m, blocked_grid(m.state, m.resolution_m, ROBOT))
    assert ev["false_free_cells"] == 0
    assert ev["unsafe_traversable_cells"] == 0
    assert ev["coverage_free"] > 0.5
    r = plan(m, (0.6, 2.3), (5.0, 2.45), ROBOT)
    assert r.success, r.message
    col = path_collisions(gt, r.path_m, t, ROBOT.radius_m, m.params.clearance_m, ROBOT.height_m)
    assert col["collisions"] == 0 and col["min_clearance_m"] >= ROBOT.radius_m


def test_why_explanation_cites_frames(room_scan):
    from clearpath3d.mapping.projection import explain_cell

    _, _, proc, m = room_scan
    ij = tuple(int(v) for v in m.cell_of(np.array([2.0, 2.0])))
    e = explain_cell(proc.evidence, m, ij)
    assert e["state"] == "free" and e["ground_support"]
    assert any(v["last_frame"] and v["last_update"] == "free_ray" for v in e["voxels"])


def test_narrow_door_blocks_wide_robot_only(scene_dir, cfg):
    root = scene_dir("narrow_door")
    _, proc, _ = _process(root, cfg)
    m = proc.map2d()
    door = (3.95, 2.375)
    assert m.state[tuple(m.cell_of(np.array(door)))] == FREE  # the doorway itself was observed free
    wide = plan(m, (0.6, 2.375), door, ROBOT)
    assert (
        not wide.success and wide.cause == "goal_footprint_blocked" and "obstacle" in wide.message
    )
    small = RobotSpec(radius_m=0.10, height_m=0.40, margin_m=0.02)
    r = plan(m, (0.6, 2.375), door, small)
    assert r.success
    col = path_collisions(GroundTruth(root), r.path_m, 3.2, small.radius_m, 0.06, 0.4)
    assert col["collisions"] == 0


def test_area_hidden_behind_cabinet_stays_unknown(scene_dir, cfg):
    root = scene_dir("hidden_box")
    _, proc, _ = _process(root, cfg)
    m = proc.map2d()
    for xy in ([2.4, 3.7], [2.3, 3.6], [2.5, 3.8]):
        assert m.state[tuple(m.cell_of(np.array(xy)))] == UNKNOWN
    assert plan(m, (0.6, 2.3), (2.4, 3.7), ROBOT).cause == "goal_unknown"


def test_moved_box_old_place_not_freed_without_reobservation(scene_dir, cfg):
    root = scene_dir("moving_box")
    _, proc, (m_mid, t_mid) = _process(root, cfg, snapshot_at=12)
    old = [[1.8, 1.7], [1.7, 1.62], [1.9, 1.78]]
    # Frame 12: moved away at t=1.3 s, old place not re-observed yet -> still occupied, never free.
    assert all(m_mid.state[tuple(m_mid.cell_of(np.array(xy)))] == OCCUPIED for xy in old)
    # End: re-observed through free rays -> no longer occupied; the floor under the box was never seen,
    # so it becomes unknown, not free.
    m_end = proc.map2d()
    assert all(m_end.state[tuple(m_end.cell_of(np.array(xy)))] != OCCUPIED for xy in old)
    assert all(m_end.state[tuple(m_end.cell_of(np.array(xy)))] != FREE for xy in old)
    assert m_end.state[tuple(m_end.cell_of(np.array([2.1, 0.75])))] == OCCUPIED
    kinds = {(c.kind, round(c.center_m[0], 0), round(c.center_m[1], 0)) for c in proc.changes}
    assert ("freed", 2.0, 2.0) in kinds and ("appeared", 2.0, 1.0) in kinds


def test_replay_without_ground_truth_is_identical(scene_dir, cfg, tmp_path):
    src = scene_dir("narrow_door")
    dst = tmp_path / "sensor_only"
    shutil.copytree(src, dst)
    shutil.rmtree(dst / "ground_truth")
    _, a, _ = _process(src, cfg, upto=6)
    _, b, _ = _process(dst, cfg, upto=6)
    np.testing.assert_array_equal(a.evidence.log_odds, b.evidence.log_odds)
    assert [t.center_m for t in a.tracker.snapshot()] == [t.center_m for t in b.tracker.snapshot()]


def test_sensor_dropout_never_creates_false_free(scene_dir, cfg, room_scan):
    """Same geometry and trajectory as room_scan, heavy holes + noise: less coverage, still no false free."""
    root = scene_dir("sensor_dropout")
    seq, proc, _ = _process(root, cfg)
    m = proc.map2d()
    t = seq[len(seq) - 1].timestamp_s
    ev = evaluate_map(
        GroundTruth(root), m, t, ROBOT.radius_m, blocked_grid(m.state, m.resolution_m, ROBOT)
    )
    assert ev["false_free_cells"] == 0 and ev["unsafe_traversable_cells"] == 0
    clean = room_scan[3]
    assert (m.state == FREE).sum() < (clean.state == FREE).sum()


def test_replay_survives_frames_without_pose(scene_dir, cfg):
    """Regression: a frame rejected by pose estimation (no pose) in the middle of a replay."""
    from clearpath3d.contracts import PoseSource
    from clearpath3d.reconstruction.registration import _with_pose

    seq = SensorSequence(scene_dir("narrow_door"))
    proc = SequenceProcessor(
        ReplayConfig(pipeline=cfg, robot=ROBOT, tsdf_enabled=False), seq.intrinsics
    )
    for i in range(5):
        f = seq[i]
        if i == 2:
            f = _with_pose(f, None, PoseSource.unavailable)
        snap = proc.step(f)
        if i == 2:
            assert snap.output_frame == "base_link"
            assert any("no pose" in w for w in snap.warnings)
    assert proc.map2d() is not None
