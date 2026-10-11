"""Evaluate 2D->3D semantic lifting with an ORACLE detector (simulator instance masks).

The oracle isolates the geometry of lifting (pixels -> depth -> map, background filtering) from detector
quality. It is labelled as such everywhere; it is not a detector result. Lives in ``evaluation/`` because it
reads ground truth; the pipeline only ever receives ``Detection2D`` objects.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from clearpath3d.config import PipelineConfig
from clearpath3d.datasets.base import SensorSequence
from clearpath3d.datasets.synthetic import build_scene, generate
from clearpath3d.evaluation.ground_truth import GroundTruth
from clearpath3d.geometry.frames import transform_points
from clearpath3d.geometry.projection import backproject
from clearpath3d.perception.clustering import aabb_iou
from clearpath3d.perception.semantics import LIFT_MODES, Detection2D, lift
from clearpath3d.runtime.replay import ReplayConfig, SequenceProcessor


class OracleDetector:
    """Perfect masks from the simulator; ``dilate_px`` simulates sloppy mask boundaries."""

    name = "oracle_gt_masks"

    def __init__(self, gt: GroundTruth, dilate_px: int = 0, min_pixels: int = 150):
        self.gt, self.dilate_px, self.min_pixels = gt, dilate_px, min_pixels
        self.groups = gt.physical_objects()

    def detect(self, frame) -> list[Detection2D]:
        inst = self.gt.frame(frame.frame_id).instance
        out = []
        for name, idxs in self.groups.items():
            m = np.isin(inst, idxs)
            if m.sum() < self.min_pixels:
                continue
            if self.dilate_px:
                m = ndimage.binary_dilation(m, iterations=self.dilate_px)
            vs, us = np.nonzero(m)
            label = self.gt.objects[idxs[0]]["category"]
            out.append(
                Detection2D(
                    label,
                    1.0,
                    (int(us.min()), int(vs.min()), int(us.max()) + 1, int(vs.max()) + 1),
                    m,
                    source=f"oracle:{name}",
                )
            )
        return out


def _lift_metrics(
    frame, gtf, gt: GroundTruth, det: Detection2D, mode: str, erode: int
) -> dict | None:
    L = lift(frame, det, mode, erode_px=erode)
    if L is None:
        return None
    name = det.source.split(":", 1)[1]
    idxs = gt.physical_objects()[name]
    inst = gtf.instance[L.pixels[:, 0], L.pixels[:, 1]]
    purity = float(np.isin(inst, idxs).mean())
    vis = np.isin(gtf.instance, idxs)
    p, _ = backproject(gtf.depth_clean, vis, frame.intrinsics)
    pm = transform_points(gtf.T_map_camera, p)
    return {
        "purity": purity,
        "iou_visible": aabb_iou(L.min_m, L.max_m, pm.min(0), pm.max(0)),
        "points": len(L.points_map),
    }


def run_semantics_oracle(
    cfg: PipelineConfig, out: Path | str, scene: str = "room_scan", step: int = 2
) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    root = generate(build_scene(scene), Path("data/synthetic") / scene)
    seq, gt = SensorSequence(root), GroundTruth(root)
    report: dict = {
        "scene": scene,
        "detector": "ORACLE (simulator instance masks) - not a detector result",
    }

    abl = {}
    for dil in (0, 4):
        det = OracleDetector(gt, dilate_px=dil)
        for mode in LIFT_MODES:
            for erode in (0, 2) if mode == "mask" else (0,):
                rows = []
                for i in range(0, len(seq), step):
                    fr = seq[i]
                    gtf = gt.frame(fr.frame_id)
                    for d in det.detect(fr):
                        m = _lift_metrics(fr, gtf, gt, d, mode, erode)
                        if m:
                            rows.append(m)
                key = f"mask_dilate{dil}px/{mode}" + (f"/erode{erode}px" if mode == "mask" else "")
                abl[key] = {
                    "lifted": len(rows),
                    "purity_median": round(float(np.median([r["purity"] for r in rows])), 4),
                    "purity_p10": round(float(np.percentile([r["purity"] for r in rows], 10)), 4),
                    "iou_visible_median": round(
                        float(np.median([r["iou_visible"] for r in rows])), 4
                    ),
                }
    report["lifting_ablation"] = abl

    # End-to-end: semantic votes on geometric tracks; geometry must be unchanged by semantics.
    base = SequenceProcessor(ReplayConfig(pipeline=cfg, tsdf_enabled=False), seq.intrinsics)
    sem = SequenceProcessor(
        ReplayConfig(pipeline=cfg, tsdf_enabled=False, semantic_detector=OracleDetector(gt)),
        seq.intrinsics,
    )
    for i in range(len(seq)):
        base.step(seq[i])
        snap = sem.step(seq[i])
    report["geometry_unchanged_with_semantics"] = bool(
        np.array_equal(base.evidence.log_odds, sem.evidence.log_odds)
    )
    tracks = []
    for t in sem.tracker.tracks + sem.tracker.retired:
        label = sem.semantics.label_of(t.track_id)
        c = t.x[:3]
        truth = None
        for idxs in gt.physical_objects().values():
            lo, hi = gt.full_aabb(idxs, seq[len(seq) - 1].timestamp_s)
            if np.all(c >= lo - 0.15) and np.all(c <= hi + 0.15):
                truth = gt.objects[idxs[0]]["category"]
        tracks.append(
            {"track_id": t.track_id, "lifecycle": t.lifecycle, "label": label, "truth": truth}
        )
    labelled = [t for t in tracks if t["label"]]
    report["tracks"] = {
        "total": len(tracks),
        "labelled": len(labelled),
        "correct": sum(t["label"] == t["truth"] for t in labelled),
        "wrong": sum(t["truth"] is not None and t["label"] != t["truth"] for t in labelled),
        "labelled_but_no_object": sum(t["truth"] is None for t in labelled),
        "unlabelled_on_objects": sum(t["truth"] is not None and not t["label"] for t in tracks),
        "unlabelled_structure": sum(t["truth"] is None and not t["label"] for t in tracks),
    }
    report["snapshot_labels_last_frame"] = [(t.track_id, t.semantic_label) for t in snap.tracks]
    (out / "oracle_report.json").write_text(json.dumps(report, indent=2, default=float))
    return report


def run_semantics_real(
    out: Path | str, root: Path | str = "data/redwood_sample", score_threshold: float = 0.6
) -> dict:
    """Real images (Redwood sample): pretrained Mask R-CNN, lifted with mask vs bbox, multi-view consistency.

    No semantic ground truth exists for this sample: the metric is consistency (does the same label land at
    the same 3D place across frames?), not accuracy.
    """
    import open3d as o3d

    from clearpath3d.perception.detectors_torch import TorchvisionMaskRCNN

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    seq = SensorSequence(root, max_depth_m=4.0)
    det = TorchvisionMaskRCNN(score_threshold=score_threshold)
    per_mode: dict[str, list] = {m: [] for m in LIFT_MODES}
    frames = []
    rng = np.random.default_rng(0)
    for i in range(len(seq)):
        fr = seq[i]
        dets = det.detect(fr)
        frames.append(
            {"frame_id": fr.frame_id, "detections": [(d.label, round(d.score, 3)) for d in dets]}
        )
        overlay = fr.color_rgb.astype(np.float32)
        for d in dets:
            if d.mask is not None:
                overlay[d.mask] = 0.5 * overlay[d.mask] + 0.5 * rng.integers(60, 255, 3)
            for mode in LIFT_MODES:
                L = lift(fr, d, mode)
                if L is not None:
                    per_mode[mode].append(L)
        o3d.io.write_image(
            str(out / f"redwood_{fr.frame_id}_maskrcnn.png"),
            o3d.geometry.Image(np.ascontiguousarray(overlay.astype(np.uint8))),
        )
    consistency = {}
    for mode, lifted in per_mode.items():
        groups: list[list] = []
        for L in lifted:  # greedy grouping: same label and overlapping 3D boxes
            for g in groups:
                ref = g[0]
                inter = np.clip(
                    np.minimum(ref.max_m, L.max_m) - np.maximum(ref.min_m, L.min_m), 0, None
                ).prod()
                vol = min(np.prod(ref.max_m - ref.min_m), np.prod(L.max_m - L.min_m))
                if ref.label == L.label and vol > 0 and inter / vol > 0.3:
                    g.append(L)
                    break
            else:
                groups.append([L])
        rows = []
        for g in groups:
            c = np.array([x.points_map.mean(0) for x in g])
            ext = np.array([x.max_m - x.min_m for x in g])
            rows.append(
                {
                    "label": g[0].label,
                    "frames": len({x.frame_id for x in g}),
                    "centroid_spread_m": round(float(np.linalg.norm(c.std(0))), 4),
                    "extent_m_median": np.median(ext, 0).round(3).tolist(),
                }
            )
        multi = [r for r in rows if r["frames"] >= 2]
        consistency[mode] = {
            "objects": len(rows),
            "seen_in_2plus_frames": len(multi),
            "median_centroid_spread_m": round(
                float(np.median([r["centroid_spread_m"] for r in multi])), 4
            )
            if multi
            else None,
            "groups": sorted(rows, key=lambda r: -r["frames"])[:12],
        }
    report = {
        "dataset": "Redwood sample (real, 5 frames)",
        "detector": det.name,
        "score_threshold": score_threshold,
        "poses": "supplied (dataset trajectory)",
        "frames": frames,
        "consistency": consistency,
        "note": "no semantic ground truth: consistency only, not accuracy",
    }
    (out / "redwood_report.json").write_text(json.dumps(report, indent=2, default=float))
    return report
