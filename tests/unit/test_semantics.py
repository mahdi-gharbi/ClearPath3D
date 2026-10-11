import numpy as np
import pytest

from clearpath3d.contracts import Intrinsics, PoseSource, RGBDFrame
from clearpath3d.geometry.frames import R_BASE_OPTICAL, make_transform
from clearpath3d.perception.semantics import (
    Detection2D,
    LabelVotes,
    SemanticFusion,
    _nearest_depth_mode,
    lift,
)

INTR = Intrinsics(fx=100, fy=100, cx=49.5, cy=39.5, width=100, height=80)


def _frame(pose=True):
    """An object at 1.5 m (centre square) in front of a wall at 4 m."""
    d = np.full((80, 100), 4.0, np.float32)
    d[30:50, 40:60] = 1.5
    T = make_transform(R_BASE_OPTICAL, [0, 0, 1.0]) if pose else None
    return RGBDFrame(
        "s",
        "f0",
        0.0,
        np.zeros((80, 100, 3), np.uint8),
        d,
        d > 0,
        INTR,
        T,
        PoseSource.supplied if pose else PoseSource.unavailable,
        "c",
    )


def _det(pad=10):
    m = np.zeros((80, 100), bool)
    m[30:50, 40:60] = True
    return Detection2D("box", 0.9, (40 - pad, 30 - pad, 60 + pad, 50 + pad), m)


def test_mask_lift_contains_only_the_object():
    L = lift(_frame(), _det(), "mask", erode_px=0)
    # optical z (depth) maps to map +x: all points at x = 1.5
    np.testing.assert_allclose(L.points_map[:, 0], 1.5, atol=1e-6)


def test_bbox_lift_leaks_background_and_depth_filter_removes_it():
    bb = lift(_frame(), _det(pad=10), "bbox")
    assert (bb.points_map[:, 0] > 3.9).mean() > 0.4  # wall pixels leak in
    f = lift(_frame(), _det(pad=10), "bbox_depth_filter")
    np.testing.assert_allclose(f.points_map[:, 0], 1.5, atol=1e-6)


def test_no_pose_no_lift_and_bad_mode_rejected():
    assert lift(_frame(pose=False), _det(), "mask") is None
    with pytest.raises(ValueError):
        lift(_frame(), _det(), "magic")


def test_nearest_depth_mode_picks_near_cluster():
    z = np.concatenate([np.full(300, 1.5), np.full(700, 4.0)])
    keep = _nearest_depth_mode(z)
    assert keep[:300].all() and not keep[300:].any()


def test_votes_need_count_and_majority():
    v = LabelVotes()
    v.add("chair", 0.9, "0")
    assert v.decide() is None  # one vote is not enough
    v.add("cat", 0.8, "1")
    assert v.decide() is None  # no clear majority
    v.add("chair", 0.9, "2")
    v.add("chair", 0.7, "3")
    assert v.decide() == "chair"


def test_fusion_attaches_by_3d_overlap_only():
    class T:
        track_id = 7
        x = np.array([1.5, 0.0, 1.0, 0, 0, 0])
        extent = np.array([0.1, 0.4, 0.4])

    fusion = SemanticFusion(min_votes=1)
    L = lift(_frame(), _det(), "mask", erode_px=0)
    fusion.update([T()], [L])
    assert fusion.label_of(7) == "box"
    far = lift(_frame(), _det(), "mask", erode_px=0)
    far.min_m, far.max_m = far.min_m + 5, far.max_m + 5
    fusion.update([T()], [far])
    assert len(fusion.unmatched) == 1
