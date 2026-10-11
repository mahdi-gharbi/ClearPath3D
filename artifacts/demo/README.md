# Demo artefacts

`m1_reference/` is produced by `clearpath3d m1-demo` (config `configs/synthetic.yaml`, seed 0).
For each scene (`room_clear`, `dominant_wall`, `no_ground`), frame 0:

- `snapshot.json` — pipeline output (sensor data only): ground status/plane, detections, warnings, timings.
- `explanation.json` — every ground-plane candidate with the reasons it was accepted or rejected.
- `points_labelled.ply` — green = ground, coloured = obstacle clusters, grey = other.
- `points_rgb.ply` — the same points with their RGB colour.
- `evaluation.json` — evaluator-side comparison with the simulator ground truth.
- `summary.json` (top level) — compact summary of the three scenes.

Open a PLY with any point-cloud viewer (e.g. `python -c "import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('points_labelled.ply')])"`).
