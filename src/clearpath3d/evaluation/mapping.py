"""Map and path metrics against the simulator ground truth (evaluator side only).

GT 2D occupancy at time t: a cell is GT-occupied if its centre lies inside the xy footprint of an
object (walls, table parts, boxes; motion applied) whose z-range intersects the robot band
``[clearance, robot_height]`` above the floor. GT-free = inside the floor footprint and not occupied.
A predicted FREE cell that is GT-occupied is a *false free* — the critical error.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from clearpath3d.evaluation.ground_truth import GroundTruth
from clearpath3d.mapping.projection import FREE, OCCUPIED, UNKNOWN, Map2D


def _object_boxes(gt: GroundTruth, t: float, z_lo: float, z_hi: float):
    boxes = []
    for o in gt.objects:
        if o["category"] == "floor":
            continue
        lo, hi = gt.full_aabb([o["index"]], t)
        if hi[2] > z_lo and lo[2] < z_hi:
            boxes.append((o["name"], lo, hi))
    return boxes


def gt_occupancy(gt: GroundTruth, m: Map2D, t: float) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = m.state.shape
    ij = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij"), axis=-1).reshape(-1, 2)
    c = m.center_of(ij)
    occ = np.zeros(len(c), dtype=bool)
    p = m.params
    for _, lo, hi in _object_boxes(gt, t, p.clearance_m, p.robot_height_m):
        occ |= (c[:, 0] > lo[0]) & (c[:, 0] < hi[0]) & (c[:, 1] > lo[1]) & (c[:, 1] < hi[1])
    floor = next(o for o in gt.objects if o["category"] == "floor")
    flo, fhi = np.asarray(floor["min_m"]), np.asarray(floor["max_m"])
    inside = (c[:, 0] >= flo[0]) & (c[:, 0] <= fhi[0]) & (c[:, 1] >= flo[1]) & (c[:, 1] <= fhi[1])
    return occ.reshape(nx, ny), (inside & ~occ).reshape(nx, ny)


def _gt_clearance(gt: GroundTruth, m: Map2D, t: float) -> np.ndarray:
    nx, ny = m.state.shape
    ij = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij"), axis=-1).reshape(-1, 2)
    c = m.center_of(ij)
    d = np.full(len(c), np.inf)
    for _, lo, hi in _object_boxes(gt, t, m.params.clearance_m, m.params.robot_height_m):
        dx = np.maximum(np.maximum(lo[0] - c[:, 0], 0), c[:, 0] - hi[0])
        dy = np.maximum(np.maximum(lo[1] - c[:, 1], 0), c[:, 1] - hi[1])
        d = np.minimum(d, np.hypot(dx, dy))
    return d.reshape(nx, ny)


def evaluate_map(
    gt: GroundTruth,
    m: Map2D,
    t: float,
    robot_radius_m: float | None = None,
    blocked: np.ndarray | None = None,
) -> dict:
    occ, free = gt_occupancy(gt, m, t)
    s = m.state
    conf = {}
    for pname, pv in (("free", FREE), ("occupied", OCCUPIED), ("unknown", UNKNOWN)):
        conf[pname] = {
            "gt_free": int(((s == pv) & free).sum()),
            "gt_occupied": int(((s == pv) & occ).sum()),
        }
    pred_free = s == FREE
    safety = {}
    if robot_radius_m is not None and blocked is not None:
        unsafe = (~blocked) & (_gt_clearance(gt, m, t) < robot_radius_m)
        safety = {
            "traversable_cells": int((~blocked).sum()),
            "unsafe_traversable_cells": int(unsafe.sum()),
        }
    return {
        **safety,
        "t_s": t,
        "cells_gt_free": int(free.sum()),
        "cells_gt_occupied": int(occ.sum()),
        "confusion": conf,
        "coverage_free": round(float((pred_free & free).sum() / max(free.sum(), 1)), 4),
        "false_free_cells": int((pred_free & occ).sum()),
        "false_free_rate": round(float((pred_free & occ).sum() / max(pred_free.sum(), 1)), 5),
        "occupied_precision_within_1_cell": round(
            float(((s == OCCUPIED) & ndimage.binary_dilation(occ, np.ones((3, 3), bool))).sum())
            / max((s == OCCUPIED).sum(), 1),
            4,
        ),
        "occupied_precision": round(
            float(((s == OCCUPIED) & occ).sum() / max((s == OCCUPIED).sum(), 1)), 4
        ),
        "notes": "occupied precision is penalised by obstacle cells observed slightly outside the exact "
        "GT footprint (voxel discretisation, 5 cm); false free is the safety-relevant number",
    }


def path_collisions(
    gt: GroundTruth,
    path_m: list[list[float]],
    t: float,
    radius_m: float,
    z_lo: float,
    z_hi: float,
    step_m: float = 0.02,
) -> dict:
    """Check the robot disk along the polyline against every GT object footprint in the band."""
    if len(path_m) < 2:
        return {"collisions": 0, "min_clearance_m": None}
    pts = []
    for a, b in zip(path_m[:-1], path_m[1:], strict=True):
        a, b = np.asarray(a), np.asarray(b)
        n = max(int(np.linalg.norm(b - a) / step_m), 1)
        pts.append(a + (b - a) * np.linspace(0, 1, n + 1)[:, None])
    P = np.concatenate(pts)
    min_d, hits = np.inf, set()
    for name, lo, hi in _object_boxes(gt, t, z_lo, z_hi):
        dx = np.maximum(np.maximum(lo[0] - P[:, 0], 0), P[:, 0] - hi[0])
        dy = np.maximum(np.maximum(lo[1] - P[:, 1], 0), P[:, 1] - hi[1])
        d = np.hypot(dx, dy)
        min_d = min(min_d, float(d.min()))
        if (d < radius_m).any():
            hits.add(name)
    return {
        "collisions": len(hits),
        "collided_with": sorted(hits),
        "min_clearance_m": round(min_d, 4),
    }
