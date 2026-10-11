import numpy as np

from clearpath3d.contracts import Intrinsics, ObstacleDetection, PoseSource, RGBDFrame
from clearpath3d.geometry.frames import R_BASE_OPTICAL, make_transform
from clearpath3d.perception.tracking import Tracker, TrackerParams

INTR = Intrinsics(fx=50, fy=50, cx=31.5, cy=23.5, width=64, height=48)


def _frame(t, depth_value=0.0, fid=None):
    T = make_transform(R_BASE_OPTICAL, [0.0, 0.0, 0.5])  # camera at origin looking along map +x
    d = np.full((48, 64), depth_value, np.float32)
    return RGBDFrame(
        "s",
        fid or f"{t:.1f}",
        t,
        np.zeros((48, 64, 3), np.uint8),
        d,
        d > 0,
        INTR,
        T,
        PoseSource.supplied,
        "c",
    )


def _det(c, i=0):
    c = np.asarray(c, float)
    e = np.array([0.3, 0.3, 0.3])
    return ObstacleDetection(
        detection_id=i,
        frame_id="map",
        center_m=c.tolist(),
        visible_extent_m=e.tolist(),
        min_m=(c - e / 2).tolist(),
        max_m=(c + e / 2).tolist(),
        point_count=100,
    )


def test_constant_velocity_keeps_one_id_and_estimates_velocity():
    tr = Tracker()
    for k in range(10):
        t = 0.2 * k
        tr.step(_frame(t), [_det([2.0 + 0.5 * t, 1.0, 0.15])], [""])
    assert len(tr.tracks) == 1
    trk = tr.tracks[0]
    assert trk.lifecycle == "confirmed" and trk.hits == 10
    np.testing.assert_allclose(trk.x[3], 0.5, atol=0.1)


def test_gating_rejects_far_detection():
    tr = Tracker()
    for k in range(4):
        tr.step(_frame(0.2 * k), [_det([2.0, 1.0, 0.15])], [""])
    tr.step(_frame(1.0), [_det([3.5, 1.0, 0.15])], [""])  # 1.5 m jump: beyond gate
    ids = sorted(t.track_id for t in tr.tracks)
    assert len(ids) == 2


def test_two_targets_hungarian_assignment():
    tr = Tracker()
    for k in range(5):
        t = 0.2 * k
        a, b = [2.0, 0.5 + 0.1 * k, 0.15], [2.0, 1.5 - 0.1 * k, 0.15]
        dets = [_det(a, 0), _det(b, 1)] if k % 2 == 0 else [_det(b, 0), _det(a, 1)]
        tr.step(_frame(t), dets, ["", ""])
    ys = sorted(round(t.x[1], 1) for t in tr.tracks)
    assert len(tr.tracks) == 2 and ys == [0.9, 1.1]


def test_out_of_view_track_becomes_occluded_then_retires_after_timeout():
    p = TrackerParams(retire_after_s=1.0)
    tr = Tracker(p)
    for k in range(4):
        tr.step(_frame(0.2 * k), [_det([2.0, 1.0, 0.15])], [""])
    tr.step(_frame(0.8), [], [])  # no depth at all: cannot be "visible"
    assert tr.tracks[0].lifecycle == "occluded" and tr.tracks[0].visible_misses == 0
    tr.step(_frame(2.0), [], [])
    assert not tr.tracks and tr.retired[0].lifecycle == "retired"


def test_visible_misses_retire_faster():
    tr = Tracker(TrackerParams(max_visible_misses=2, retire_after_s=100))
    for k in range(4):
        tr.step(_frame(0.2 * k), [_det([2.0, 0.0, 0.5])], [""])
    # Depth 5 m everywhere: the predicted centre (2 m ahead) is visible and nothing is there.
    tr.step(_frame(1.0, 5.0), [], [])
    tr.step(_frame(1.2, 5.0), [], [])
    assert not tr.tracks


def test_structure_not_tracked():
    tr = Tracker()
    tr.step(_frame(0.0), [_det([2.0, 1.0, 0.5])], ["top"])
    assert not tr.tracks
