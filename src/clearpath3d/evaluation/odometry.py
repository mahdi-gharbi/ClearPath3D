"""M6 (optional research): ICP odometry experiment — baseline, ablations, failure case, map safety.

Reference poses: simulator GT (synthetic) or the dataset trajectory (Redwood). Alignment: first frame.
Map experiment: only the first pose is supplied (gauge anchor); all later poses are estimated.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from clearpath3d.config import PipelineConfig
from clearpath3d.datasets.base import SensorSequence
from clearpath3d.datasets.synthetic import build_scene, generate
from clearpath3d.evaluation.ground_truth import GroundTruth
from clearpath3d.evaluation.mapping import evaluate_map
from clearpath3d.planning.astar import RobotSpec, blocked_grid
from clearpath3d.reconstruction.registration import ICPOdometry, ICPParams, trajectory_errors
from clearpath3d.runtime.replay import ReplayConfig, SequenceProcessor

VARIANTS = {
    "icp_full": ICPParams(),
    "no_degeneracy_gate": ICPParams(min_p2plane_information=0),
    "no_motion_gate": ICPParams(max_deviation_from_prediction_m=0),
    "no_gates": ICPParams(min_p2plane_information=0, max_deviation_from_prediction_m=0),
    "fine_pyramid_only": ICPParams(
        voxel_sizes_m=(0.08, 0.04, 0.02), max_iterations=(30, 20, 15), corr_factor=2.0
    ),
    "point_to_point": ICPParams(method="point_to_point"),
}


def _run(seq: SensorSequence, p: ICPParams, T0: np.ndarray) -> tuple[list, ICPOdometry, float]:
    odo = ICPOdometry(
        seq.T_base_camera if seq.T_base_camera is not None else np.eye(4), p, T_map_camera_0=T0
    )
    t0 = time.perf_counter()
    est = [odo.process(seq[i]).T_map_camera for i in range(len(seq))]
    return est, odo, (time.perf_counter() - t0) * 1000 / len(seq)


def run_odometry_experiment(cfg: PipelineConfig, out: Path | str, log=print) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"protocol": __doc__.strip(), "params_default": asdict(ICPParams())}

    # 1. Ablation on the dense synthetic scan (where thresholds were chosen: not a held-out result).
    d = generate(build_scene("room_scan_dense"), Path("data/synthetic/room_scan_dense"))
    seq = SensorSequence(d, use_supplied_poses=False)
    gt = GroundTruth(d)
    ref = [gt.frame(m["frame_id"]).T_map_camera for m in seq.frames_meta]
    abl = {}
    for name, p in VARIANTS.items():
        est, odo, ms = _run(seq, p, None)
        abl[name] = {
            **trajectory_errors(est, ref),
            "ms_per_frame": round(ms, 1),
            "first_rejection": next(
                (
                    {"frame": x["frame_id"], "reasons": x.get("reasons")}
                    for x in odo.log
                    if not x["accepted"]
                ),
                None,
            ),
            "tracking_lost": odo.lost,
        }
        log(f"room_scan_dense/{name}: {abl[name]}")
    abl["baseline_static_pose"] = trajectory_errors([seq.T_base_camera] * len(seq), ref)
    report["room_scan_dense_ablation"] = abl

    # 2. Held-out checks for the chosen settings.
    lo = generate(build_scene("low_overlap"), Path("data/synthetic/low_overlap"))
    lseq = SensorSequence(lo, use_supplied_poses=False)
    _, lodo, _ = _run(lseq, ICPParams(), None)
    report["low_overlap_failure_case"] = [
        {k: x.get(k) for k in ("frame_id", "accepted", "reasons")} for x in lodo.log
    ]
    red = Path("data/redwood_sample")
    if (red / "manifest.json").exists():
        rs = SensorSequence(red, max_depth_m=4.0, use_supplied_poses=False)
        rref = [np.asarray(m["T_map_camera"]) for m in rs.frames_meta]
        rest, rodo, ms = _run(rs, ICPParams(), np.eye(4))
        report["redwood_sample"] = {
            **trajectory_errors(rest, rref),
            "ms_per_frame": round(ms, 1),
            "note": "real data, 5 frames; reference = dataset trajectory.log",
        }
    else:
        report["redwood_sample"] = "skipped: run `clearpath3d download-redwood`"

    # 3. Does the map stay safe with estimated poses?
    sup = SensorSequence(d)
    robot = RobotSpec()
    maps = {}
    for name, p in (
        ("supplied", None),
        ("icp_full", ICPParams()),
        ("no_gates", ICPParams(min_p2plane_information=0, max_deviation_from_prediction_m=0)),
    ):
        proc = SequenceProcessor(
            ReplayConfig(pipeline=cfg, robot=robot, tsdf_enabled=False), sup.intrinsics
        )
        odo = (
            None
            if p is None
            else ICPOdometry(sup.T_base_camera, p, T_map_camera_0=sup[0].T_map_camera)
        )
        used = 0
        for i in range(len(sup)):
            f = sup[i] if odo is None else odo.process(seq[i])
            used += f.T_map_camera is not None
            proc.step(f)
        m = proc.map2d()
        t = sup[len(sup) - 1].timestamp_s
        ev = evaluate_map(gt, m, t, robot.radius_m, blocked_grid(m.state, m.resolution_m, robot))
        maps[name] = {
            "frames_in_map": used,
            **{k: ev[k] for k in ("coverage_free", "false_free_cells", "unsafe_traversable_cells")},
        }
        log(f"map/{name}: {maps[name]}")
    report["map_with_estimated_poses"] = maps
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    return report
