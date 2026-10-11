"""Semantic fusion with an ORACLE detector (simulator masks): labels are right, geometry is unchanged."""

import numpy as np
import pytest

from clearpath3d.datasets.base import SensorSequence
from clearpath3d.evaluation.ground_truth import GroundTruth
from clearpath3d.evaluation.semantics_eval import OracleDetector
from clearpath3d.runtime.replay import ReplayConfig, SequenceProcessor


def test_oracle_semantics_labels_tracks_without_changing_geometry(scene_dir, cfg):
    root = scene_dir("room_scan")
    seq, gt = SensorSequence(root), GroundTruth(root)
    base = SequenceProcessor(ReplayConfig(pipeline=cfg, tsdf_enabled=False), seq.intrinsics)
    sem = SequenceProcessor(
        ReplayConfig(pipeline=cfg, tsdf_enabled=False, semantic_detector=OracleDetector(gt)),
        seq.intrinsics,
    )
    for i in range(14):
        base.step(seq[i])
        snap = sem.step(seq[i])
    np.testing.assert_array_equal(base.evidence.log_odds, sem.evidence.log_odds)
    labels = {t.track_id: t.semantic_label for t in snap.tracks}
    assert any(v is not None for v in labels.values())
    for t in sem.tracker.tracks:
        lab = sem.semantics.label_of(t.track_id)
        if lab is None:
            continue
        truth = [
            gt.objects[idxs[0]]["category"]
            for idxs in gt.physical_objects().values()
            if np.all(t.x[:3] >= gt.full_aabb(idxs, 0.0)[0] - 0.15)
            and np.all(t.x[:3] <= gt.full_aabb(idxs, 0.0)[1] + 0.15)
        ]
        assert truth and lab == truth[0], (t.track_id, lab, truth)


def test_real_detector_is_optional():
    torch = pytest.importorskip("torch")
    from clearpath3d.perception.detectors_torch import TorchvisionMaskRCNN

    assert torch.__version__ and TorchvisionMaskRCNN.name.startswith("torchvision")
