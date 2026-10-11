# Architecture and decisions

## Data flow (M1, single frame)

```text
sensor_export/ ──SensorSequence──> RGBDFrame ──analyze_frame──> SceneSnapshot + labelled points
   (npz, calib, poses)    validity mask,          backproject -> base_link -> voxel -> outliers
                          unit check              -> normals -> ground (gated RANSAC) -> DBSCAN
                                                  -> boxes (map if pose supplied, else base_link)
ground_truth/ ──GroundTruth──> evaluate_frame (evaluator only) ──> evaluation.json
```

## Decision log

| # | Decision | Reason |
|---|---|---|
| D1 | Python 3.12 venv on Windows; Open3D from PyPI | Open3D publishes no 3.13 wheel; 3.12 is the newest supported. |
| D2 | CPU-only profile | MX330 (2 GB) gives no useful advantage for geometry; GPU optional later for a 2D detector. |
| D3 | ROS2 target: Jazzy on the existing WSL2 Ubuntu 24.04 | Matching official platform; not installed yet (needs sudo inside WSL, owner action). |
| D4 | `RGBDFrame` is a validated dataclass, outputs are Pydantic | NumPy arrays in Pydantic are awkward; JSON outputs benefit from schemas. |
| D5 | Simulator casts rays with unit-z directions | `t_hit` equals z-depth directly; verified by the fronto-parallel wall test. |
| D6 | Ground detection in `base_link` with the calibration as gravity prior | Works with or without a pose; height plausibility is meaningful (floor at z≈0). Limitation: assumes a level base. |
| D7 | Own NumPy RANSAC instead of `segment_plane` | Seeded and deterministic; sequential candidates with explicit rejection reasons for the "why?" view. |
| D8 | Voxel downsampling keeps one representative pixel per voxel | Keeps the point → pixel link for evidence and GT evaluation. |
| D9 | Obstacle boxes are axis-aligned in the output frame and describe visible points only | Partial views cannot give full object dimensions; evaluation reports both visible and full IoU. |
| D10 | Storage: per-frame compressed NPZ + JSON manifests | No database needed; easy to checksum. |
| D11 | Statistical outlier removal off by default on synthetic data | M1 ablation: it erased thin table legs and far floor (see PROGRESS.md). Re-evaluate on real data. |
| D12 | Open3D installed in two steps if `dash` download stalls | `open3d/__init__` imports the web visualiser (plotly, dash) unconditionally. |
| D13 | Pin `open3d==0.19.0` | 0.20.0 Windows wheel: TSDF integration (legacy and tensor) yields no surface, even on the official tutorial. Guard test added. |
| D14 | Redwood adapter rotates world y-up into map z-up once (`T_map_world`) | Measured: pose direction, z-depth and gravity axis (PROGRESS.md, M2). |
| D15 | Reconstruction completeness is measured on the *observed* GT surface | Unseen surfaces are not a reconstruction error; coverage is a mapping metric (M3). |
| D16 | Free space only from depth rays (exact voxel traversal), never from the TSDF | Missing depth or an unseen volume must stay unknown. |
| D17 | Planner: forbidden = obstacle/unknown dilated by R; R..R+margin allowed at high cost | Strict R+margin dilation made every start pose infeasible (unobserved surroundings). Reported per path. |
| D18 | Robot footprint is evidence of free space (source `robot_footprint`), swept between poses < 0.5 m apart | The body occupied that volume at supplied poses; reach R + half a cell diagonal (<= 3.5 cm over-claim). |
| D19 | Log-odds clamp [-2, 2] | Lets a moved object be freed after ~6 free observations; trade-off: less persistent occupied evidence. |
| D20 | Moved objects are detected by `appeared` change events, not by track displacement | Visible-centre shifts of partial clusters made displacement unreliable. |
| D21 | `ReplaySession` in the core, FastAPI only as a thin layer | Same session drives API, tests and the ROS node; processing on one worker thread per replay, never on the event loop. |
| D22 | Binary artefacts (int8 grid, float32 xyz) with metadata headers; SSE carries metadata only | Measured ~0.4 MB per UI refresh instead of millions of JSON points. |
| D23 | ROS: map→base_link published from supplied poses only; no `odom` frame | No odometry estimator exists; one authority per transform. |
| D24 | NBV gain counts never-observed voxels only, first along each ray | Ambiguous surface voxels never resolve and inflated the gain; unknown is not treated as transparent. |
| D25 | NBV exploration starts after the scripted scan | From an unmoved pose the camera blind zone (~1 m) leaves nothing reachable through known free space. |
| D26 | Held-out protocol: procedural rooms, frozen splits and code fingerprint, test run once | Plan §10; all thresholds before were set on the reported scenes. |
| D27 | Outside the map = unknown; occupied and obstacle-adjacent unknown cells dilated by R + half a cell diagonal | Discretisation and grid-extent safety bugs found on dev/validation. |
| D28 | Two free observations required only next to obstacles | Global rule blocked every passable door. |
| D29 | ICP odometry rejects instead of guessing (degeneracy + motion gates); tracking loss is reported | Accepting all converged results produced 203 unsafe cells. |
| D30 | Semantics are votes on geometric tracks, never a source of geometry; oracle masks only for evaluation | Plan: geometry must work without labels; isolates lifting errors from detector errors. |
| D31 | torch/torchvision kept out of the core lock (`requirements-semantics.txt`) | ~1 GB, optional feature, tight disk. |
