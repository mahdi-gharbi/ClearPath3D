"""ClearPath3D command line."""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def simulate(
    scene: str = typer.Argument(
        ..., help="Scene name (room_clear, dominant_wall, no_ground, narrow_door)"
    ),
    out: Path = typer.Option(Path("data/synthetic"), help="Parent output directory"),
    width: int = typer.Option(320),
    height: int = typer.Option(240),
    seed: int = typer.Option(0),
) -> None:
    """Render a synthetic sequence into <out>/<scene>/{sensor_export,ground_truth}."""
    from clearpath3d.datasets.synthetic import build_scene, generate

    spec = build_scene(scene, width=width, height=height, seed=seed)
    path = generate(spec, out / scene)
    typer.echo(f"wrote {len(spec.trajectory)} frame(s) to {path}")


@app.command("analyze-frame")
def analyze_frame_cmd(
    sequence: Path = typer.Argument(..., help="Sequence directory (with manifest.json)"),
    frame: int = typer.Option(0, help="Frame index"),
    config: Path | None = typer.Option(None, help="Pipeline YAML config"),
    out: Path = typer.Option(Path("runs/frame"), help="Output directory"),
    no_pose: bool = typer.Option(False, help="Ignore supplied poses (output in base_link)"),
) -> None:
    """Analyse one RGB-D frame: ground, obstacles, JSON/PLY export. Reads sensor data only."""
    from clearpath3d.config import load_config
    from clearpath3d.datasets.base import SensorSequence
    from clearpath3d.runtime.pipeline import analyze_frame, export_analysis

    cfg = load_config(config)
    seq = SensorSequence(
        sequence, cfg.depth.min_depth_m, cfg.depth.max_depth_m, use_supplied_poses=not no_pose
    )
    fa = analyze_frame(seq[frame], cfg)
    export_analysis(fa, out)
    s = fa.snapshot
    typer.echo(
        f"{s.sequence_id}/{s.frame_id}: ground={s.ground_status.value} detections={len(s.detections)} "
        f"pose={s.pose_status.value} -> {out}"
    )
    for w in s.warnings:
        typer.echo(f"  warning: {w}")


@app.command("evaluate-frame")
def evaluate_frame_cmd(
    sequence: Path = typer.Argument(...),
    run: Path = typer.Argument(..., help="Directory written by analyze-frame"),
    frame: int = typer.Option(0),
) -> None:
    """Compare an analyze-frame output with the simulator ground truth (evaluator side)."""
    from clearpath3d.evaluation.metrics import evaluate_frame

    rep = evaluate_frame(sequence, frame, run)
    (run / "evaluation.json").write_text(json.dumps(rep, indent=2))
    typer.echo(json.dumps({"ground": rep["ground"], "obstacles": rep["obstacles"]}, indent=2))


@app.command("m1-demo")
def m1_demo(
    out: Path = typer.Option(Path("artifacts/demo/m1_reference")),
    data: Path = typer.Option(Path("data/synthetic")),
    config: Path = typer.Option(Path("configs/synthetic.yaml")),
) -> None:
    """Generate room_clear, dominant_wall and no_ground; analyse and evaluate frame 0 of each."""
    from clearpath3d.config import load_config
    from clearpath3d.datasets.base import SensorSequence
    from clearpath3d.datasets.synthetic import build_scene, generate
    from clearpath3d.evaluation.metrics import evaluate_frame
    from clearpath3d.runtime.pipeline import analyze_frame, export_analysis

    cfg = load_config(config)
    summary = {}
    for scene in ("room_clear", "dominant_wall", "no_ground"):
        seq_dir = generate(build_scene(scene), data / scene)
        seq = SensorSequence(seq_dir, cfg.depth.min_depth_m, cfg.depth.max_depth_m)
        fa = analyze_frame(seq[0], cfg)
        run = export_analysis(fa, out / scene)
        rep = evaluate_frame(seq_dir, 0, run)
        (run / "evaluation.json").write_text(json.dumps(rep, indent=2))
        summary[scene] = {
            "ground_status": fa.snapshot.ground_status.value,
            "ground": rep["ground"],
            "detections": len(fa.snapshot.detections),
            "obstacles": [
                {
                    k: e.get(k)
                    for k in ("object", "iou_visible", "center_error_visible_m", "iou_full_object")
                }
                for e in rep["obstacles"]
            ],
            "warnings": fa.snapshot.warnings,
            "timings_ms": fa.snapshot.timings_ms,
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    typer.echo(json.dumps(summary, indent=2))


@app.command("download-redwood")
def download_redwood(data_root: Path = typer.Option(Path("data/downloads"))) -> None:
    """Explicit network download of Open3D's SampleRedwoodRGBDImages (~11 MB), then conversion."""
    from clearpath3d.datasets.redwood import convert, download

    src = download(data_root)
    out = convert(src, Path("data/redwood_sample"))
    typer.echo(f"downloaded to {src}; converted to {out}")


@app.command()
def reconstruct(
    sequence: Path = typer.Argument(...),
    out: Path = typer.Option(Path("runs/reconstruction")),
    voxel: float = typer.Option(0.02, help="TSDF voxel size (m)"),
    trunc: float = typer.Option(0.08, help="TSDF truncation distance (m)"),
    max_depth: float = typer.Option(6.0),
    no_keyframes: bool = typer.Option(False),
) -> None:
    """TSDF fusion + point-union baseline with supplied poses."""
    from clearpath3d.datasets.base import SensorSequence
    from clearpath3d.reconstruction.run import reconstruct as run
    from clearpath3d.reconstruction.tsdf import TSDFParams

    p = TSDFParams(
        voxel_m=voxel, sdf_trunc_m=trunc, max_depth_m=max_depth, use_keyframes=not no_keyframes
    )
    rep = run(SensorSequence(sequence, max_depth_m=max_depth), p, out)
    t = rep["tsdf"]
    typer.echo(
        f"pose_source={rep['pose_source']} integrated={t['frames_integrated']}/{t['frames_seen']} "
        f"mesh_vertices={t['mesh_vertices']} -> {out}"
    )


@app.command("m2-demo")
def m2_demo(out: Path = typer.Option(Path("artifacts/demo/m2_reference"))) -> None:
    """Fuse the synthetic room_scan (and the Redwood sample if downloaded); evaluate vs GT."""
    from clearpath3d.datasets.base import SensorSequence
    from clearpath3d.datasets.redwood import convert
    from clearpath3d.datasets.synthetic import build_scene, generate
    from clearpath3d.evaluation.reconstruction import evaluate_reconstruction
    from clearpath3d.reconstruction.run import reconstruct as run
    from clearpath3d.reconstruction.tsdf import TSDFParams

    summary = {}
    d = generate(build_scene("room_scan"), Path("data/synthetic/room_scan"))
    rep = run(SensorSequence(d), TSDFParams(), out / "room_scan")
    ev = evaluate_reconstruction(d, out / "room_scan")
    summary["room_scan"] = {"report": rep, "evaluation": ev}
    red = Path("data/downloads/extract/SampleRedwoodRGBDImages")
    if red.exists():
        rd = convert(red, Path("data/redwood_sample"))
        p = TSDFParams(voxel_m=4.0 / 512, sdf_trunc_m=0.04, max_depth_m=4.0, use_keyframes=False)
        summary["redwood_sample"] = {
            "report": run(SensorSequence(rd, max_depth_m=4.0), p, out / "redwood_sample"),
            "evaluation": "no ground truth; see tests/integration/test_redwood.py for convention checks",
        }
    else:
        summary["redwood_sample"] = "skipped: run `clearpath3d download-redwood` first"
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    brief = {k: (v["evaluation"] if isinstance(v, dict) else v) for k, v in summary.items()}
    typer.echo(json.dumps(brief, indent=2))


@app.command("m3-demo")
def m3_demo(
    out: Path = typer.Option(Path("artifacts/demo/m3_reference")),
    config: Path = typer.Option(Path("configs/synthetic.yaml")),
    scenes: str = typer.Option("room_scan,narrow_door,hidden_box,moving_box"),
) -> None:
    """Replay M3 scenarios (tracking, visibility map, A*) and evaluate against the simulator GT."""
    from clearpath3d.config import load_config
    from clearpath3d.evaluation.runner import run_scenario

    cfg = load_config(config)
    summary = {}
    for name in scenes.split(","):
        rep = run_scenario(name, cfg, out)
        brief = {
            "final_map": {
                k: rep["final"]["map"].get(k)
                for k in ("coverage_free", "false_free_cells", "unsafe_traversable_cells")
            },
            "plans": {k: v["cause"] for k, v in rep["final"]["plans"].items()},
            "timing_per_frame": rep["timing_per_frame"],
        }
        for k in rep:
            if k.startswith("at_frame_"):
                brief[k] = {
                    "map": {
                        x: rep[k]["map"].get(x)
                        for x in ("false_free_cells", "unsafe_traversable_cells")
                    },
                    "plans": {q: v["cause"] for q, v in rep[k]["plans"].items()},
                }
        summary[name] = brief
        typer.echo(f"{name}: {json.dumps(brief)}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


@app.command("m5-experiment")
def m5_experiment(
    out: Path = typer.Option(Path("artifacts/demo/m5_reference")),
    config: Path = typer.Option(Path("configs/synthetic.yaml")),
    seeds: str = typer.Option("0,1,2"),
    scenes: str = typer.Option("hidden_box,narrow_door"),
) -> None:
    """Next-best-view vs random viewpoints (same candidates, views, travel budget, seeds)."""
    from clearpath3d.config import load_config
    from clearpath3d.evaluation.campaign import run_campaign

    s = run_campaign(
        load_config(config),
        out,
        tuple(int(x) for x in seeds.split(",")),
        scenes.split(","),
        log=lambda msg: typer.echo(msg[:160]),
    )
    typer.echo(json.dumps(s["comparison"], indent=2))


@app.command()
def benchmark(
    split: str = typer.Argument(..., help="dev | validation | test"),
    out: Path = typer.Option(Path("artifacts/benchmark")),
    config: Path = typer.Option(Path("configs/synthetic.yaml")),
    nbv_views: int = typer.Option(
        4, help="views per policy for the NBV vs random comparison (0 = skip)"
    ),
) -> None:
    """Procedural-room benchmark on a frozen split (experiments/splits/v1.json)."""
    from clearpath3d.config import load_config
    from clearpath3d.evaluation.benchmark import run_split

    s = run_split(
        split, load_config(config), out, nbv_views=nbv_views, log=lambda m: typer.echo(m[:220])
    )
    typer.echo(json.dumps(s, indent=2))


@app.command("m6-odometry")
def m6_odometry(
    out: Path = typer.Option(Path("artifacts/demo/m6_reference")),
    config: Path = typer.Option(Path("configs/synthetic.yaml")),
) -> None:
    """Optional research: ICP odometry ablation, failure case, real-data check, map safety."""
    from clearpath3d.config import load_config
    from clearpath3d.evaluation.odometry import run_odometry_experiment

    run_odometry_experiment(load_config(config), out, log=lambda m: typer.echo(m[:220]))


@app.command("m6-semantics")
def m6_semantics(
    out: Path = typer.Option(Path("artifacts/demo/m6_semantics")),
    config: Path = typer.Option(Path("configs/synthetic.yaml")),
) -> None:
    """Optional research: 2D->3D semantic lifting (oracle ablation on synthetic; Mask R-CNN on Redwood)."""
    from clearpath3d.config import load_config
    from clearpath3d.evaluation.semantics_eval import run_semantics_oracle, run_semantics_real

    r = run_semantics_oracle(load_config(config), out)
    typer.echo(
        json.dumps(
            {"tracks": r["tracks"], "geometry_unchanged": r["geometry_unchanged_with_semantics"]}
        )
    )
    try:
        import torch  # noqa: F401
    except ImportError:
        typer.echo(
            "real-data part skipped: install requirements-semantics.txt (torch, torchvision)"
        )
        return
    if not Path("data/redwood_sample/manifest.json").exists():
        typer.echo("real-data part skipped: run `clearpath3d download-redwood` first")
        return
    rr = run_semantics_real(out)
    typer.echo(
        json.dumps(
            {
                m: {
                    k: c[k] for k in ("objects", "seen_in_2plus_frames", "median_centroid_spread_m")
                }
                for m, c in rr["consistency"].items()
            }
        )
    )


@app.command("download-tum")
def download_tum(
    sequence: str = typer.Option("freiburg2_pioneer_360", help="TUM RGB-D sequence name"),
    data_root: Path = typer.Option(Path("data/downloads/tum")),
) -> None:
    """Explicit download (~0.7 GB) of a TUM RGB-D sequence (CC BY 4.0), extraction and conversion."""
    import subprocess
    import tarfile

    from clearpath3d.datasets.tum import convert

    group = sequence.split("_")[0]
    name = f"rgbd_dataset_{sequence}"
    url = f"https://cvg.cit.tum.de/rgbd/dataset/{group}/{name}.tgz"
    data_root.mkdir(parents=True, exist_ok=True)
    tgz = data_root / f"{name}.tgz"
    if not tgz.exists():
        subprocess.run(["curl", "-L", "--fail", "--retry", "5", "-o", str(tgz), url], check=True)
    ext = data_root / "extract"
    if not (ext / name).exists():
        with tarfile.open(tgz) as tf:
            tf.extractall(ext, filter="data")
    out = convert(ext / name, Path(f"data/tum_{sequence.split('_', 1)[1]}"), archive=tgz)
    typer.echo(f"converted to {out}")


@app.command("real-benchmark")
def real_benchmark(
    sequence: Path = typer.Option(Path("data/tum_pioneer_360")),
    out: Path = typer.Option(Path("artifacts/demo/real_tum_pioneer_360")),
    config: Path = typer.Option(Path("configs/synthetic.yaml")),
) -> None:
    """Real-data benchmark (TUM robot sequence): ICP vs mocap, ground, map agreement, TSDF."""
    from clearpath3d.config import load_config
    from clearpath3d.evaluation.real_benchmark import run_real_benchmark

    run_real_benchmark(load_config(config), sequence, out, log=lambda m: typer.echo(m[:300]))


if __name__ == "__main__":
    app()
