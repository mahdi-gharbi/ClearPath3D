"""Real-data benchmark on a TUM RGB-D robot sequence (motion-capture poses, no scene geometry ground truth).

What can be measured honestly on real data:
1. odometry (ICP) against motion-capture poses: ATE (first-frame and best-fit SE(3) alignment) and RPE;
2. ground detection per frame against the floor plane fitted in the mocap world (normal angle, height);
3. map safety *relative to* the map built with mocap poses: cells the ICP-pose map calls FREE while the
   mocap-pose map calls them OCCUPIED (a proxy for false free; the mocap map itself is not ground truth);
4. TSDF surface agreement between mocap-pose and ICP-pose reconstructions.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from clearpath3d.config import PipelineConfig
from clearpath3d.datasets.base import SensorSequence
from clearpath3d.geometry.frames import invert
from clearpath3d.mapping.projection import FREE, OCCUPIED, UNKNOWN
from clearpath3d.mapping.render import render_map, save_png
from clearpath3d.planning.astar import RobotSpec, plan
from clearpath3d.reconstruction.registration import ICPOdometry, ICPParams, trajectory_errors
from clearpath3d.reconstruction.tsdf import TSDFFusion, TSDFParams
from clearpath3d.runtime.pipeline import analyze_frame
from clearpath3d.runtime.replay import ReplayConfig, SequenceProcessor


def umeyama_ate(est: list, ref: list) -> dict:
    """ATE after the best rigid alignment (no scale) of estimated to reference positions."""
    pairs = [(e[:3, 3], r[:3, 3]) for e, r in zip(est, ref, strict=True) if e is not None]
    if len(pairs) < 3:
        return {"ate_rmse_m": None}
    A, B = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    ma, mb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((B - mb).T @ (A - ma))
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ S @ Vt
    err = np.linalg.norm((A - ma) @ R.T + mb - B, axis=1)
    return {
        "alignment": "best-fit SE(3) (Umeyama, no scale)",
        "ate_rmse_m": round(float(np.sqrt(np.mean(err**2))), 4),
        "ate_max_m": round(float(err.max()), 4),
    }


def plan_demo(m, start, robot: RobotSpec):
    """Plan from ``start`` to the farthest cell reachable through observed free space (qualitative demo)."""
    from clearpath3d.planning.astar import blocked_grid
    from clearpath3d.planning.frontiers import travel_distance

    s_ij = tuple(int(v) for v in m.cell_of(np.asarray(start)))
    dist = travel_distance(blocked_grid(m.state, m.resolution_m, robot), s_ij, m.resolution_m)
    if not np.isfinite(dist).any() or not np.isfinite(dist[s_ij]):
        return np.asarray(start), plan(m, start, start, robot)
    d = np.where(np.isfinite(dist), dist, -1)
    far = m.center_of(np.array(np.unravel_index(int(np.argmax(d)), d.shape)))
    return far, plan(m, start, far, robot)


def run_real_benchmark(
    cfg: PipelineConfig, root: Path | str, out: Path | str, map_stride: int = 2, log=print
) -> dict:
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    sup = SensorSequence(root, cfg.depth.min_depth_m, cfg.depth.max_depth_m)
    raw = SensorSequence(
        root, cfg.depth.min_depth_m, cfg.depth.max_depth_m, use_supplied_poses=False
    )
    n = len(sup)
    ref = [sup[i].T_map_camera for i in range(n)]
    report: dict = {
        "sequence": manifest["sequence"],
        "frames": n,
        "license": manifest["license"],
        "reference_poses": "motion capture (TUM ground truth)",
        "protocol": __doc__.strip(),
    }

    # 1. Odometry.
    odo_res, est_by = {}, {}
    # *_range8 variants are POST-HOC: added after the first run on this sequence showed that the default
    # 4 m range (chosen for the synthetic rooms) drops 46 % of the valid depth in this hall and leaves only
    # degenerate floor geometry. 8 m is the Kinect's nominal maximum range. Reported as tuned on this sequence.
    no_gates = dict(min_p2plane_information=0, max_deviation_from_prediction_m=0)
    for name, p in (
        ("icp_full", ICPParams()),
        ("icp_full_range8_posthoc", ICPParams(max_range_m=8.0)),
        ("no_gates", ICPParams(**no_gates)),
        ("no_gates_range8_posthoc", ICPParams(max_range_m=8.0, **no_gates)),
    ):
        odo = ICPOdometry(sup.T_base_camera, p, T_map_camera_0=ref[0])
        t0 = time.perf_counter()
        est = [odo.process(raw[i]).T_map_camera for i in range(n)]
        est_by[name] = est
        reasons: dict[str, int] = {}
        for e in odo.log:
            for r in e.get("reasons") or []:
                key = r.split(" ")[0] if not r.startswith("translation weakly") else "degenerate"
                reasons[key] = reasons.get(key, 0) + 1
        odo_res[name] = {
            "first_frame_alignment": trajectory_errors(est, ref),
            **{"best_fit": umeyama_ate(est, ref)},
            "tracking_lost": odo.lost,
            "rejection_reasons": reasons,
            "ms_per_frame": round((time.perf_counter() - t0) * 1000 / n, 1),
        }
        log(f"odometry/{name}: {odo_res[name]}")
    report["odometry"] = odo_res

    # 2. Ground detection per frame vs the mocap-world floor.
    g = manifest["gravity_check"]
    n_floor, floor_z = np.asarray(g["floor_normal_world"]), g["floor_z_world_m"]
    rows = []
    for i in range(0, n, 10):
        fa = analyze_frame(sup[i], cfg)
        gp = fa.snapshot.ground_plane
        if gp is None:
            rows.append({"found": False})
            continue
        nn = np.asarray(gp.normal)
        T_mb = ref[i] @ invert(sup.T_base_camera)
        xy = T_mb[:2, 3]
        h = -(nn[0] * xy[0] + nn[1] * xy[1] + gp.offset) / nn[2]
        rows.append(
            {
                "found": True,
                "angle_deg": float(np.degrees(np.arccos(np.clip(nn @ n_floor, -1, 1)))),
                "height_err_m": float(abs(h - floor_z)),
            }
        )
    found = [r for r in rows if r["found"]]
    report["ground"] = {
        "frames_tested": len(rows),
        "found": len(found),
        "angle_deg_median": round(float(np.median([r["angle_deg"] for r in found])), 3)
        if found
        else None,
        "height_err_m_median": round(float(np.median([r["height_err_m"] for r in found])), 4)
        if found
        else None,
        "height_err_m_p95": round(float(np.percentile([r["height_err_m"] for r in found], 95)), 4)
        if found
        else None,
        "reference": "floor plane fitted in the mocap world (manifest gravity_check)",
    }
    log(f"ground: {report['ground']}")

    # 3. Maps: mocap poses vs ICP poses (same frames, stride map_stride).
    robot = RobotSpec()
    maps = {}
    for name, poses in (
        ("mocap", ref),
        ("icp_full_range8_posthoc", est_by["icp_full_range8_posthoc"]),
        ("no_gates_range8_posthoc", est_by["no_gates_range8_posthoc"]),
    ):
        from dataclasses import replace as _replace

        from clearpath3d.mapping.voxel_evidence import EvidenceParams, bounds_around

        ev = EvidenceParams()
        lo, hi = bounds_around(np.array([T[:2, 3] for T in ref]), ev.max_range_m, ev.resolution_m)
        rc = ReplayConfig(
            pipeline=cfg,
            robot=robot,
            tsdf_enabled=False,
            evidence=_replace(ev, bounds_min=lo, bounds_max=hi),
        )
        proc = SequenceProcessor(rc, sup.intrinsics)
        from clearpath3d.contracts import PoseSource
        from clearpath3d.reconstruction.registration import _with_pose

        t0 = time.perf_counter()
        for i in range(0, n, map_stride):
            f = raw[i]
            T = poses[i]
            proc.step(
                _with_pose(
                    f,
                    T,
                    PoseSource.supplied
                    if name == "mocap"
                    else (PoseSource.estimated if T is not None else PoseSource.unavailable),
                )
            )
        maps[name] = (proc.map2d(), (time.perf_counter() - t0) / len(range(0, n, map_stride)))
    m_ref = maps["mocap"][0]
    report["maps"] = {}
    for name, (m, spf) in maps.items():
        d = {"counts": m.counts(), "s_per_frame": round(spf, 3)}
        if name != "mocap":
            d["free_where_mocap_map_occupied"] = int(
                ((m.state == FREE) & (m_ref.state == OCCUPIED)).sum()
            )
            d["free_where_mocap_map_unknown"] = int(
                ((m.state == FREE) & (m_ref.state == UNKNOWN)).sum()
            )
            d["agreement_on_cells_known_in_both"] = round(
                float(
                    (m.state == m_ref.state)[(m.state != UNKNOWN) & (m_ref.state != UNKNOWN)].mean()
                ),
                4,
            )
        report["maps"][name] = d
        save_png(render_map(m), out / f"map_{name}.png")
    # A plan on the mocap-pose map: from the first pose to the farthest cell REACHABLE through observed free
    # space for this robot (the first version picked the farthest free cell, which sat next to unknown space).
    start = (ref[0] @ invert(sup.T_base_camera))[:2, 3]
    far, r = plan_demo(m_ref, start, robot)
    np.save(out / "map_mocap_state.npy", m_ref.state)
    report["plan_demo_mocap_map"] = {
        "start": np.round(start, 3).tolist(),
        "goal": np.round(far, 3).tolist(),
        "cause": r.cause,
        "length_m": r.length_m,
    }
    save_png(
        render_map(m_ref, r.path_m if r.success else None, start, far), out / "map_mocap_plan.png"
    )
    log(f"maps: {report['maps']}")

    # 4. TSDF: mocap vs ICP poses.
    clouds = {}
    for name, poses in (
        ("mocap", ref),
        ("icp_full_range8_posthoc", est_by["icp_full_range8_posthoc"]),
    ):
        fu = TSDFFusion(sup.intrinsics, TSDFParams(voxel_m=0.03, sdf_trunc_m=0.09, max_depth_m=4.0))
        from clearpath3d.contracts import PoseSource
        from clearpath3d.reconstruction.registration import _with_pose

        for i in range(0, n, 2):
            if poses[i] is not None:
                fu.integrate(_with_pose(raw[i], poses[i], PoseSource.supplied))
        _, c = fu.extract()
        clouds[name] = np.asarray(c.points)
        o3d.io.write_point_cloud(str(out / f"tsdf_{name}.ply"), c)
    dd, _ = cKDTree(clouds["mocap"]).query(clouds["icp_full_range8_posthoc"])
    report["tsdf_icp_vs_mocap"] = {
        "icp_variant": "icp_full_range8_posthoc",
        "frames_integrated_icp": int(
            sum(T is not None for T in est_by["icp_full_range8_posthoc"][::2])
        ),
        "median_m": round(float(np.median(dd)), 4),
        "p95_m": round(float(np.percentile(dd, 95)), 4),
        "frac_below_5cm": round(float((dd < 0.05).mean()), 4),
        "note": "distance from ICP-pose surface to mocap-pose surface (not to the true scene)",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    return report
