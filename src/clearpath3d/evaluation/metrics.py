"""Single-frame metrics against the simulator ground truth (M1)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from clearpath3d.datasets.base import SensorSequence
from clearpath3d.evaluation.ground_truth import GroundTruth
from clearpath3d.geometry.frames import transform_points
from clearpath3d.geometry.projection import backproject
from clearpath3d.perception.clustering import aabb_iou
from clearpath3d.runtime.pipeline import LABEL_GROUND

MIN_VISIBLE_PIXELS = 30


def evaluate_frame(seq_root: Path | str, frame_index: int, run_dir: Path | str) -> dict:
    seq = SensorSequence(seq_root)
    gt = GroundTruth(seq_root)
    meta = seq.frames_meta[frame_index]
    g = gt.frame(meta["frame_id"])
    run_dir = Path(run_dir)
    snap = json.loads((run_dir / "snapshot.json").read_text())
    with np.load(run_dir / "points.npz") as z:
        labels, pixel = z["labels"], z["pixel"]

    report: dict = {
        "sequence": seq.sequence_id,
        "frame_id": meta["frame_id"],
        "pose_source": snap["pose_status"],
        "output_frame": snap["output_frame"],
        "config_hash": snap["config_hash"],
        "notes": [
            "Point-level ground labels use one representative pixel per 3 cm voxel.",
            "Obstacle boxes are compared to the VISIBLE part of each object (from clean GT depth); "
            "IoU against the full object box is reported separately and is expected to be lower.",
        ],
    }

    # ---- ground
    inst = g.instance[pixel[:, 0], pixel[:, 1]] if len(pixel) else np.zeros(0, dtype=np.int32)
    gt_floor = (inst >= 0) & (gt.categories[np.clip(inst, 0, None)] == "floor")
    pred = labels == LABEL_GROUND
    tp = int((pred & gt_floor).sum())
    ground = {
        "status": snap["ground_status"],
        "gt_floor_points": int(gt_floor.sum()),
        "pred_ground_points": int(pred.sum()),
        "precision": round(tp / pred.sum(), 4) if pred.sum() else None,
        "recall": round(tp / gt_floor.sum(), 4) if gt_floor.sum() else None,
    }
    gp = snap.get("ground_plane")
    if gp is not None and snap["output_frame"] == "map":
        n = np.asarray(gp["normal"])
        ground["normal_error_deg"] = round(float(np.degrees(np.arccos(np.clip(n[2], -1, 1)))), 3)
        robot_xy = g.T_map_base[:2, 3]
        ground["height_error_at_robot_m"] = round(
            float(abs(n @ [*robot_xy, 0.0] + gp["offset"])), 4
        )
    if gp is not None:
        # Fraction of predicted ground points that actually lie on a wall: the "wall became floor" check.
        on_wall = (inst >= 0) & (gt.categories[np.clip(inst, 0, None)] == "wall")
        ground["pred_ground_on_wall_fraction"] = round(
            float((pred & on_wall).sum() / max(pred.sum(), 1)), 4
        )
    report["ground"] = ground

    # ---- obstacles (non-structural objects visible in this frame)
    vis_pts, vis_pix = backproject(g.depth_clean, g.instance >= 0, seq.intrinsics)
    vis_inst = g.instance[vis_pix[:, 0], vis_pix[:, 1]]
    T_out_cam = g.T_map_camera if snap["output_frame"] == "map" else (seq.T_base_camera)
    vis_out = transform_points(T_out_cam, vis_pts)
    dets = snap["detections"]
    obstacles = []
    for name, idxs in gt.physical_objects().items():
        m = np.isin(vis_inst, idxs)
        if m.sum() < MIN_VISIBLE_PIXELS:
            continue
        lo, hi = vis_out[m].min(axis=0), vis_out[m].max(axis=0)
        entry = {
            "object": name,
            "visible_pixels": int(m.sum()),
            "visible_box_min": lo.round(3).tolist(),
            "visible_box_max": hi.round(3).tolist(),
            "matched_detection": None,
        }
        if dets:
            ious = [aabb_iou(lo, hi, d["min_m"], d["max_m"]) for d in dets]
            j = int(np.argmax(ious))
            if ious[j] > 0:
                d = dets[j]
                entry.update(
                    matched_detection=d["detection_id"],
                    iou_visible=round(ious[j], 4),
                    center_error_visible_m=round(
                        float(np.linalg.norm(np.asarray(d["center_m"]) - (lo + hi) / 2)), 4
                    ),
                )
                if snap["output_frame"] == "map":
                    flo, fhi = gt.full_aabb(idxs, float(meta["timestamp_s"]))
                    entry["iou_full_object"] = round(aabb_iou(flo, fhi, d["min_m"], d["max_m"]), 4)
        obstacles.append(entry)
    report["obstacles"] = obstacles

    # ---- detection purity: majority GT category of each detection's points
    with np.load(run_dir / "points.npz") as z:
        cluster = z["cluster"]
    purity = []
    for d in dets:
        sel = cluster == d["detection_id"]
        cats = [gt.categories[i] if i >= 0 else "none" for i in inst[sel]]
        vals, counts = np.unique(cats, return_counts=True)
        k = int(np.argmax(counts))
        purity.append(
            {
                "detection_id": d["detection_id"],
                "majority_category": str(vals[k]),
                "majority_fraction": round(float(counts[k] / counts.sum()), 4),
            }
        )
    report["detections"] = purity
    report["detections_on_floor"] = sum(p["majority_category"] == "floor" for p in purity)
    return report
