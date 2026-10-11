"""Scenario runner: replay a synthetic sequence, plan queries, evaluate against GT, save artefacts.

The processing side (SequenceProcessor, plan) never sees GT; only the evaluation calls below do.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from clearpath3d.config import PipelineConfig
from clearpath3d.datasets.base import SensorSequence
from clearpath3d.datasets.synthetic import build_scene, generate
from clearpath3d.evaluation.ground_truth import GroundTruth
from clearpath3d.evaluation.mapping import evaluate_map, path_collisions
from clearpath3d.mapping.render import render_map, save_png
from clearpath3d.planning.astar import RobotSpec, blocked_grid, plan
from clearpath3d.runtime.profiling import hardware, memory_mb, percentiles
from clearpath3d.runtime.replay import ReplayConfig, SequenceProcessor


@dataclass
class Query:
    name: str
    start: tuple[float, float]
    goal: tuple[float, float]
    robot: RobotSpec = field(default_factory=RobotSpec)


SCENARIOS: dict[str, dict] = {
    "room_scan": {
        "queries": [
            Query("to_corridor", (0.6, 2.3), (5.0, 2.45)),
            Query("behind_table", (0.6, 2.3), (2.4, 3.65)),
        ]
    },
    "narrow_door": {
        "queries": [
            Query("door_wide_robot", (0.6, 2.375), (3.95, 2.375)),
            Query(
                "door_small_robot",
                (0.6, 2.375),
                (3.95, 2.375),
                RobotSpec(radius_m=0.10, margin_m=0.02),
            ),
            Query("corridor_wide_robot", (0.6, 2.375), (4.6, 2.375)),
        ]
    },
    "hidden_box": {"queries": [Query("to_hidden_box", (0.6, 2.3), (2.4, 3.7))]},
    # Goal = where the box is moved to: stale "free" at frame 12 (limitation made visible), occupied at the end.
    "moving_box": {
        "queries": [Query("to_new_box_place", (0.6, 2.3), (2.1, 0.75))],
        "snapshot_at": 12,
    },
}


def _evaluate(gt, m, t, robot, queries, out: Path, tag: str) -> dict:
    ev = evaluate_map(gt, m, t, robot.radius_m, blocked_grid(m.state, m.resolution_m, robot))
    plans = {}
    first_path = None
    for q in queries:
        r = plan(m, q.start, q.goal, q.robot)
        entry = {k: v for k, v in asdict(r).items() if k not in ("path_cells",)}
        entry["robot"] = asdict(q.robot)
        if r.success:
            entry["gt_check"] = path_collisions(
                gt, r.path_m, t, q.robot.radius_m, m.params.clearance_m, q.robot.height_m
            )
            first_path = first_path or r.path_m
        plans[q.name] = entry
    q0 = queries[0]
    save_png(render_map(m, first_path, q0.start, q0.goal), out / f"map_{tag}.png")
    return {"map": ev, "map_counts": m.counts(), "plans": plans}


def run_scenario(
    name: str,
    cfg: PipelineConfig,
    out_dir: Path | str,
    data_root: Path | str = "data/synthetic",
    robot: RobotSpec | None = None,
) -> dict:
    robot = robot or RobotSpec()
    spec = SCENARIOS[name]
    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    root = generate(build_scene(name), Path(data_root) / name)
    seq = SensorSequence(root, cfg.depth.min_depth_m, cfg.depth.max_depth_m)
    proc = SequenceProcessor(ReplayConfig(pipeline=cfg, robot=robot), seq.intrinsics)
    gt = GroundTruth(root)
    report: dict = {
        "scene": name,
        "pose_source": "supplied",
        "robot": asdict(robot),
        "config_hash": cfg.config_hash(),
    }
    mem0 = memory_mb()
    totals = []
    for i in range(len(seq)):
        snap = proc.step(seq[i])
        totals.append(snap.timings_ms["total"])
        if spec.get("snapshot_at") == i + 1:
            report[f"at_frame_{i + 1}"] = _evaluate(
                gt,
                proc.map2d(),
                seq[i].timestamp_s,
                robot,
                spec["queries"],
                out,
                f"frame{i + 1:02d}",
            )
    t_end = seq[len(seq) - 1].timestamp_s
    m = proc.map2d()
    report["final"] = _evaluate(gt, m, t_end, robot, spec["queries"], out, "final")
    report["changes"] = [c.to_dict() for c in proc.changes]
    report["tracks_final"] = [t.model_dump() for t in proc.tracker.snapshot()]
    report["tracks_retired"] = len(proc.tracker.retired)
    report["dynamic_track_ids"] = sorted(proc.dynamic_track_ids)
    report["floor_z_estimate_m"] = proc.floor_z
    report["timing_per_frame"] = percentiles(totals)
    report["timing_breakdown_last_frame_ms"] = snap.timings_ms
    report["memory"] = {"before": mem0, "after": memory_mb()}
    report["hardware"] = hardware()
    report["frames"] = len(seq)
    if proc.tsdf is not None:
        import open3d as o3d

        mesh, _ = proc.tsdf.extract()
        o3d.io.write_triangle_mesh(str(out / "static_tsdf_mesh.ply"), mesh)
        report["tsdf_frames_integrated"] = proc.tsdf.report.frames_integrated
    np.save(out / "map_final_state.npy", m.state)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    return report
