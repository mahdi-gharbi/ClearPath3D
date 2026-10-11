"""Simulated active-perception loop: propose viewpoint -> move along a known-free path -> render -> update.

Separation: the *simulator* (Renderer) uses the full scene to render the next image; the *planner*
(``propose_viewpoint``) and the map only see sensor frames. Ground truth is read by ``GroundTruth``
for metrics only.

Policies share the same candidate set (reachable, within ``max_travel_m``), the same number of views,
the same initial frames and the same scene seed:
- ``nbv``:    best score (gain, travel, rotation);
- ``random``: uniform choice among the same reachable candidates (seeded).
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from clearpath3d.config import PipelineConfig
from clearpath3d.contracts import PoseSource, RGBDFrame
from clearpath3d.datasets.synthetic import Renderer, TrajectoryPoint, build_scene, generate
from clearpath3d.evaluation.ground_truth import GroundTruth
from clearpath3d.evaluation.mapping import evaluate_map
from clearpath3d.geometry.frames import planar_pose
from clearpath3d.geometry.preprocess import valid_depth_mask
from clearpath3d.planning.astar import RobotSpec, blocked_grid, plan
from clearpath3d.planning.next_best_view import NBVParams, propose_viewpoint
from clearpath3d.runtime.replay import ReplayConfig, SequenceProcessor


def _frame_from_render(
    renderer: Renderer, tp: TrajectoryPoint, rng, seq_id: str, fid: str, cfg
) -> RGBDFrame:
    fr = renderer.render(tp, rng)
    d = fr.depth_noisy.astype(np.float64)
    valid = valid_depth_mask(d, cfg.depth.min_depth_m, cfg.depth.max_depth_m)
    return RGBDFrame(
        seq_id,
        fid,
        tp.t,
        fr.color,
        np.where(valid, d, 0).astype(np.float32),
        valid,
        renderer.intr,
        fr.T_map_camera,
        PoseSource.supplied,
        f"{seq_id}-cam-v1",
        T_base_camera=renderer.T_base_camera,
    )


def run_exploration(
    scene: str,
    policy: str,
    cfg: PipelineConfig,
    n_views: int = 6,
    n_initial: int = 6,
    seed: int = 0,
    robot: RobotSpec | None = None,
    nbv: NBVParams | None = None,
    data_root: Path | str = "data/synthetic",
    log=None,
) -> dict:
    robot = robot or RobotSpec()
    nbv = nbv or NBVParams()
    spec = build_scene(scene, seed=seed)  # seed drives the sensor noise; geometry is identical
    root = Path(data_root) / scene
    if not (root / "manifest.json").exists():
        generate(spec, root)  # GT files for the evaluator
    gt = GroundTruth(root)
    renderer = Renderer(spec)
    rng_sensor = np.random.default_rng(spec.seed)  # identical sensor noise stream for both policies
    rng_policy = np.random.default_rng(seed)
    proc = SequenceProcessor(
        ReplayConfig(pipeline=cfg, robot=robot, tsdf_enabled=False), renderer.intr
    )
    t = 0.0
    for k in range(n_initial):
        tp = spec.trajectory[k]
        t = tp.t
        proc.step(_frame_from_render(renderer, tp, rng_sensor, spec.name, f"init{k:02d}", cfg))
    pose = spec.trajectory[n_initial - 1]
    x, y, yaw = pose.x, pose.y, pose.yaw
    history = []

    def metrics():
        m = proc.map2d()
        ev = evaluate_map(gt, m, t, robot.radius_m, blocked_grid(m.state, m.resolution_m, robot))
        return {
            k: ev[k] for k in ("coverage_free", "false_free_cells", "unsafe_traversable_cells")
        } | {"unknown_cells": m.counts()["unknown"]}

    history.append({"view": 0, "x": x, "y": y, "yaw": yaw, "travel_m": 0.0, **metrics()})
    travel_total = 0.0
    for v in range(1, n_views + 1):
        m = proc.map2d()
        t0 = time.perf_counter()
        res = propose_viewpoint(
            proc.evidence,
            m,
            (x, y),
            yaw,
            robot,
            renderer.T_base_camera,
            renderer.intr,
            cfg_l_occ(proc),
            cfg_l_free(proc),
            nbv,
        )
        plan_ms = (time.perf_counter() - t0) * 1000
        if res.status != "ok":
            history.append({"view": v, "status": res.status, "message": res.message})
            break
        if policy == "nbv":
            choice = res.best
        elif policy == "random":
            choice = res.candidates[int(rng_policy.integers(len(res.candidates)))]
        else:
            raise ValueError(policy)
        pr = (
            plan(m, (x, y), (choice.x, choice.y), robot) if (choice.x, choice.y) != (x, y) else None
        )
        if pr is not None and not pr.success:
            history.append({"view": v, "status": "plan_failed", "cause": pr.cause})
            break
        length = pr.length_m if pr is not None else 0.0
        # The robot body sweeps the planned path: footprint evidence along it (as during the scan).
        if pr is not None:
            for a, b in zip(pr.path_m[:-1], pr.path_m[1:], strict=True):
                T_a = planar_pose_xy(a)
                T_b = planar_pose_xy(b)
                proc.evidence.mark_robot_footprint(
                    T_b, robot.radius_m, robot.height_m, t, f"v{v:02d}:path", T_a
                )
        travel_total += length
        t += 0.2 + length / 0.3  # nominal 0.3 m/s; used only for timestamps
        x, y, yaw = choice.x, choice.y, choice.yaw
        frame = _frame_from_render(
            renderer, TrajectoryPoint(t, x, y, yaw), rng_sensor, spec.name, f"v{v:02d}", cfg
        )
        proc.step(frame)
        history.append(
            {
                "view": v,
                "x": round(x, 3),
                "y": round(y, 3),
                "yaw": round(yaw, 3),
                "predicted_gain_voxels": choice.gain_voxels,
                "score": round(choice.score, 4),
                "travel_m": round(length, 3),
                "candidates": len(res.candidates),
                "nbv_ms": round(plan_ms, 1),
                **metrics(),
            }
        )
        if log:
            log(f"[{scene}/{policy}/seed{seed}] view {v}: {history[-1]}")
    m_end = proc.map2d()
    hidden = _object_cells_state(gt, m_end, t)
    return {
        "hidden_objects_final": hidden,
        "scene": scene,
        "policy": policy,
        "seed": seed,
        "n_views": n_views,
        "n_initial": n_initial,
        "robot": asdict(robot),
        "nbv_params": asdict(nbv),
        "pose_source": "supplied (simulator)",
        "travel_total_m": round(travel_total, 3),
        "history": history,
    }


def _object_cells_state(gt: GroundTruth, m, t: float) -> dict:
    """For each non-structural GT object: fraction of its footprint cells that are OCCUPIED / UNKNOWN."""
    from clearpath3d.mapping.projection import OCCUPIED, UNKNOWN

    out = {}
    for name, idxs in gt.physical_objects().items():
        lo, hi = gt.full_aabb(idxs, t)
        i0, j0 = m.cell_of(lo[:2])
        i1, j1 = m.cell_of(hi[:2])
        cells = m.state[max(i0, 0) : i1 + 1, max(j0, 0) : j1 + 1]
        if cells.size:
            out[name] = {
                "occupied": round(float((cells == OCCUPIED).mean()), 3),
                "unknown": round(float((cells == UNKNOWN).mean()), 3),
            }
    return out


def cfg_l_occ(proc: SequenceProcessor) -> float:
    return proc.cfg.projection.l_occ


def cfg_l_free(proc: SequenceProcessor) -> float:
    return proc.cfg.projection.l_free


def planar_pose_xy(xy) -> np.ndarray:
    return planar_pose(float(xy[0]), float(xy[1]), 0.0)
