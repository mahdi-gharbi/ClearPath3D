"""M5 campaign: NBV vs random viewpoints with equal views, equal per-view travel budget and seeds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from clearpath3d.config import PipelineConfig
from clearpath3d.evaluation.active import run_exploration
from clearpath3d.planning.next_best_view import NBVParams

SETUPS = {
    # scene: (initial scripted frames, views)
    "hidden_box": (31, 10),
    "narrow_door": (17, 6),
}


def _final(h: list[dict]) -> dict:
    done = [x for x in h if "coverage_free" in x]
    return done[-1]


def run_campaign(
    cfg: PipelineConfig, out: Path | str, seeds=(0, 1, 2), scenes=None, log=print
) -> dict:
    out = Path(out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    rows = []
    for scene in scenes or SETUPS:
        n_init, n_views = SETUPS[scene]
        for seed in seeds:
            for policy in ("nbv", "random"):
                r = run_exploration(
                    scene,
                    policy,
                    cfg,
                    n_views=n_views,
                    n_initial=n_init,
                    seed=seed,
                    nbv=NBVParams(),
                    log=log,
                )
                (out / "runs" / f"{scene}_{policy}_seed{seed}.json").write_text(
                    json.dumps(r, indent=2)
                )
                h = r["history"]
                f = _final(h)
                rows.append(
                    {
                        "scene": scene,
                        "policy": policy,
                        "seed": seed,
                        "views_done": len(
                            [x for x in h if x.get("view", 0) > 0 and "coverage_free" in x]
                        ),
                        "stop_status": next(
                            (x["status"] for x in h if "status" in x), "budget_exhausted"
                        ),
                        "coverage_initial": h[0]["coverage_free"],
                        "coverage_final": f["coverage_free"],
                        "coverage_gain": round(f["coverage_free"] - h[0]["coverage_free"], 4),
                        "false_free_max": max(x.get("false_free_cells", 0) for x in h),
                        "unsafe_max": max(x.get("unsafe_traversable_cells", 0) for x in h),
                        "travel_total_m": r["travel_total_m"],
                        "gain_per_m": round(
                            (f["coverage_free"] - h[0]["coverage_free"])
                            / max(r["travel_total_m"], 1e-9),
                            4,
                        ),
                        "hidden_objects": r["hidden_objects_final"],
                    }
                )
    summary = {
        "rows": rows,
        "comparison": _compare(rows),
        "protocol": {
            "policies": "nbv = best score; random = uniform over the SAME reachable candidates (seeded)",
            "equal": "initial frames, number of views, per-view travel budget (max_travel_m), "
            "sensor-noise seed",
            "metrics": "coverage_free = observed-free cells / GT free cells (2D); false free and "
            "unsafe cells; travel",
            "planner_inputs": "observed voxel map only; simulator renders the chosen view; "
            "GT used for metrics only",
            "n": f"{len(seeds)} seeds per scene: too few for strong statistics; per-seed values are reported",
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _compare(rows: list[dict]) -> dict:
    out = {}
    for scene in sorted({r["scene"] for r in rows}):
        diffs, rng = [], np.random.default_rng(0)
        by = {(r["policy"], r["seed"]): r for r in rows if r["scene"] == scene}
        seeds = sorted({s for _, s in by})
        for s in seeds:
            if ("nbv", s) in by and ("random", s) in by:
                diffs.append(by[("nbv", s)]["coverage_gain"] - by[("random", s)]["coverage_gain"])
        d = np.asarray(diffs)
        boot = [float(rng.choice(d, len(d)).mean()) for _ in range(2000)] if len(d) else []
        out[scene] = {
            "coverage_gain_nbv_minus_random_per_seed": [round(float(x), 4) for x in d],
            "mean_difference": round(float(d.mean()), 4) if len(d) else None,
            "bootstrap_95ci_over_seeds": [
                round(float(np.percentile(boot, 2.5)), 4),
                round(float(np.percentile(boot, 97.5)), 4),
            ]
            if boot
            else None,
            "note": f"bootstrap over seeds (sequence level), n = {len(d)}",
        }
    return out
