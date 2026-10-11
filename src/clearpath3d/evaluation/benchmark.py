"""Split-based benchmark on procedural rooms (development / validation / held-out test).

For each scene (geometry from its seed) the *pipeline* side runs exactly as in the demos; the evaluator
then reads GT for metrics. Planning query = evaluator-chosen: from the first scripted pose to a point 1 m
past the door on the scan line. The GT answer ("is the door passable for this robot?") is computed from
the layout only to score the planner's decision.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from clearpath3d.config import PipelineConfig
from clearpath3d.datasets.base import SensorSequence
from clearpath3d.datasets.procedural import build_procedural
from clearpath3d.datasets.synthetic import generate
from clearpath3d.evaluation.active import run_exploration
from clearpath3d.evaluation.ground_truth import GroundTruth
from clearpath3d.evaluation.mapping import evaluate_map, path_collisions
from clearpath3d.evaluation.metrics import evaluate_frame
from clearpath3d.planning.astar import RobotSpec, blocked_grid, plan
from clearpath3d.planning.next_best_view import NBVParams
from clearpath3d.runtime.pipeline import analyze_frame, export_analysis
from clearpath3d.runtime.replay import ReplayConfig, SequenceProcessor

SPLITS_V1 = {
    "dev": list(range(0, 10)),
    "validation": list(range(10, 20)),
    "test": list(range(100, 120)),
}


def write_splits(path: Path | str, cfg: PipelineConfig) -> dict:
    """Freeze the split manifest (and the config hash) before any test-split run."""
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    manifest = {
        "version": "v1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "clearpath3d.datasets.procedural.build_procedural(seed)",
        "splits": {k: [f"proc_{s:03d}" for s in v] for k, v in SPLITS_V1.items()},
        "config_hash_frozen": cfg.config_hash(),
        "rule": "parameters may be changed after looking at dev/validation only; the test split is run once "
        "with the frozen config and reported as is",
        "note": "seeds of different splits give disjoint geometries; no frame is shared between splits",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return manifest


def _door_passable_gt(layout: dict, robot: RobotSpec) -> bool:
    """GT-side truth for the query: door wider than the robot diameter (no margin)."""
    return layout["door_w"] > 2 * robot.radius_m


def evaluate_scene(
    seed: int, cfg: PipelineConfig, out: Path, robot: RobotSpec, nbv_views: int
) -> dict:
    spec = build_procedural(seed)
    root = generate(spec, Path("data/procedural") / spec.name)
    (root / "ground_truth" / "layout.json").write_text(json.dumps(spec.layout, indent=2))
    gt = GroundTruth(root)
    seq = SensorSequence(root, cfg.depth.min_depth_m, cfg.depth.max_depth_m)
    row: dict = {"scene": spec.name, "layout": {k: round(v, 3) for k, v in spec.layout.items()}}

    fa = analyze_frame(seq[0], cfg)
    run1 = export_analysis(fa, out / spec.name / "frame0")
    e1 = evaluate_frame(root, 0, run1)
    row["m1"] = {
        "ground_status": fa.snapshot.ground_status.value,
        **{
            k: e1["ground"].get(k)
            for k in ("precision", "recall", "normal_error_deg", "height_error_at_robot_m")
        },
        "detections_on_floor": e1["detections_on_floor"],
    }

    t0 = time.perf_counter()
    proc = SequenceProcessor(
        ReplayConfig(pipeline=cfg, robot=robot, tsdf_enabled=False), seq.intrinsics
    )
    for i in range(len(seq)):
        proc.step(seq[i])
    t_end = seq[len(seq) - 1].timestamp_s
    m = proc.map2d()
    ev = evaluate_map(gt, m, t_end, robot.radius_m, blocked_grid(m.state, m.resolution_m, robot))
    lay = spec.layout
    start = (spec.trajectory[0].x, spec.trajectory[0].y)
    goal = (lay["partition_x"] + 1.0, lay["path_y"])
    r = plan(m, start, goal, robot)
    row["m3"] = {
        k: ev[k]
        for k in (
            "coverage_free",
            "false_free_cells",
            "unsafe_traversable_cells",
            "occupied_precision_within_1_cell",
        )
    }
    row["m3"]["replay_s_per_frame"] = round((time.perf_counter() - t0) / len(seq), 3)
    row["plan"] = {
        "cause": r.cause,
        "length_m": r.length_m,
        "door_passable_gt": _door_passable_gt(lay, robot),
        "gt_collisions": path_collisions(gt, r.path_m, t_end, robot.radius_m, 0.06, robot.height_m)[
            "collisions"
        ]
        if r.success
        else None,
    }

    if nbv_views:
        n_init = len(spec.trajectory)
        res = {}
        for policy in ("nbv", "random"):
            ex = run_exploration(
                spec.name,
                policy,
                cfg,
                n_views=nbv_views,
                n_initial=n_init,
                seed=seed,
                robot=robot,
                nbv=NBVParams(),
                data_root="data/procedural",
            )
            h = [x for x in ex["history"] if "coverage_free" in x]
            res[policy] = {
                "coverage_gain": round(h[-1]["coverage_free"] - h[0]["coverage_free"], 4),
                "views": len(h) - 1,
                "travel_m": ex["travel_total_m"],
                "false_free_max": max(x["false_free_cells"] for x in h),
                "unsafe_max": max(x["unsafe_traversable_cells"] for x in h),
                "stop": next((x["status"] for x in ex["history"] if "status" in x), "budget"),
            }
        row["m5"] = res
    return row


def run_split(
    split: str,
    cfg: PipelineConfig,
    out: Path | str,
    robot: RobotSpec | None = None,
    nbv_views: int = 4,
    splits_path: Path | str = "experiments/splits/v1.json",
    log=print,
) -> dict:
    out = Path(out) / split
    out.mkdir(parents=True, exist_ok=True)
    manifest = write_splits(splits_path, cfg)
    robot = robot or RobotSpec()
    if split == "test" and cfg.config_hash() != manifest["config_hash_frozen"]:
        raise RuntimeError(
            "config changed since the splits were frozen: the test split must use the frozen config"
        )
    rows = []
    for name in manifest["splits"][split]:
        row = evaluate_scene(int(name.split("_")[1]), cfg, out, robot, nbv_views)
        rows.append(row)
        (out / f"{name}.json").write_text(json.dumps(row, indent=2, default=float))
        log(f"{name}: m3={row['m3']} plan={row['plan']} m5={row.get('m5')}")
    summary = summarize(rows)
    summary["split"] = split
    summary["config_hash"] = cfg.config_hash()
    summary["code_fingerprint"] = _code_fingerprint()
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def summarize(rows: list[dict]) -> dict:
    def arr(f):
        return np.array([f(r) for r in rows], dtype=float)

    plans = [r["plan"] for r in rows]
    passable = [p for p in plans if p["door_passable_gt"]]
    blocked = [p for p in plans if not p["door_passable_gt"]]
    s = {
        "scenes": len(rows),
        "m1_ground_found": int(sum(r["m1"]["ground_status"] == "found" for r in rows)),
        "m1_precision_median": float(np.nanmedian(arr(lambda r: r["m1"]["precision"] or np.nan))),
        "m1_recall_median": float(np.nanmedian(arr(lambda r: r["m1"]["recall"] or np.nan))),
        "m3_false_free_total": int(arr(lambda r: r["m3"]["false_free_cells"]).sum()),
        "m3_unsafe_total": int(arr(lambda r: r["m3"]["unsafe_traversable_cells"]).sum()),
        "m3_coverage_median": float(np.median(arr(lambda r: r["m3"]["coverage_free"]))),
        "plan_narrow_door_scenes": len(blocked),
        "plan_narrow_door_wrongly_ok": sum(p["cause"] == "ok" for p in blocked),
        "plan_passable_door_scenes": len(passable),
        "plan_passable_ok": sum(p["cause"] == "ok" for p in passable),
        "plan_passable_failure_causes": sorted(
            {p["cause"] for p in passable if p["cause"] != "ok"}
        ),
        "plan_gt_collisions": int(sum((p["gt_collisions"] or 0) for p in plans)),
    }
    if all("m5" in r for r in rows):
        d = arr(lambda r: r["m5"]["nbv"]["coverage_gain"] - r["m5"]["random"]["coverage_gain"])
        rng = np.random.default_rng(0)
        boot = [rng.choice(d, len(d)).mean() for _ in range(5000)]
        s["m5_nbv_minus_random_mean"] = round(float(d.mean()), 4)
        s["m5_bootstrap95_over_scenes"] = [
            round(float(np.percentile(boot, 2.5)), 4),
            round(float(np.percentile(boot, 97.5)), 4),
        ]
        s["m5_nbv_better_scenes"] = int((d > 0).sum())
        s["m5_nbv_false_free_max"] = int(arr(lambda r: r["m5"]["nbv"]["false_free_max"]).max())
        s["m5_nbv_unsafe_max"] = int(arr(lambda r: r["m5"]["nbv"]["unsafe_max"]).max())
        s["m5_random_unsafe_max"] = int(arr(lambda r: r["m5"]["random"]["unsafe_max"]).max())
    return s


def _code_fingerprint() -> str:
    src = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for p in sorted(src.rglob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]
