# PROGRESS — ClearPath3D

Target: V1.1. Milestones M0 → M5 (M6 optional). Every number below comes from a saved file
(`artifacts/demo/m1_reference/*`), measured on the machine described in the environment section.

## Status overview

| Milestone | State | Evidence |
|---|---|---|
| M0 Install & conventions | **done** | `requirements.lock`, `docs/coordinates.md`, 16 unit tests |
| M1 One frame fully explained | **done** | `artifacts/demo/m1_reference/`, 11 integration/regression tests |
| M2 Multi-view reconstruction | **done** | `artifacts/demo/m2_reference/`, 13 tests (4 need the Redwood download) |
| M3 Tracking, map, A* | **done** | `artifacts/demo/m3_reference/`, 30 tests (unit + scenario) |
| M4 API, viewer, ROS2 | **partial**: API + viewer done; ROS2 code written, **not yet run in ROS** | `artifacts/demo/m4_reference/`, 10 tests |
| M5 Next-best-view | **done** (synthetic) | `artifacts/demo/m5_reference/`, `docs/experiments.md`, 6 tests |
| Held-out evaluation (plan §10) | **done** (procedural rooms) | `experiments/splits/`, `artifacts/benchmark/`, E5 |
| M6 ICP pose estimation (optional) | **done** (local odometry, no loop closure) | `artifacts/demo/m6_reference/`, E6 |
| M6 2D semantics (optional) | **done** (oracle ablation + real-image demo, no semantic GT on real data) | `artifacts/demo/m6_semantics/`, E7, 8 tests |

## Environment audit (2026-10-09)

- Windows 10 Pro 19045, Intel i7-10510U (4 cores / 8 threads), 19.8 GB RAM.
- GPU: NVIDIA GeForce MX330, 2 GB, driver 472.19 → **CPU-only profile** (D2).
- Disk: **only ~20 GB free on C:** (97 % used). Generated data is kept small and git-ignored.
- Python 3.12.10 and 3.13.2 installed. **3.12 chosen**: Open3D publishes no 3.13 wheel (D1).
- WSL2 present with Ubuntu 24.04.5 (stopped), no `/opt/ros`. ROS2 Jazzy target = Ubuntu 24.04 → compatible, not installed.
- Node 22.14 / npm 10.9 available (for the M4 viewer).
- Repository was empty; `git init` done. **No commits**: the machine's git identity is not the owner's
  (owner: github.com/mahdi-gharbi). Changes stay uncommitted on purpose.

Installed and tested: **open3d 0.19.0** (0.20.0 rejected, see M2), numpy 2.2.6, scipy 1.18.1, pydantic 2.14.0, typer 0.27.3,
pytest 9.1.1, ruff 0.16.10 (full list: `requirements.lock`).

Install incident: a plain `pip install open3d` stalled twice while downloading `dash` (a dependency of
Open3D's web visualiser, imported unconditionally by `open3d/__init__`). Workaround: install
`open3d --no-deps`, then the core deps, then `pip install open3d` again once the network recovered.

## M0 — conventions and contracts

Delivered: `src/clearpath3d/contracts.py` (`RGBDFrame` validated dataclass; `SceneSnapshot`,
`ObstacleTrack`, `GroundPlane`, `ObstacleDetection` Pydantic), `geometry/frames.py`,
`geometry/projection.py`, `geometry/preprocess.py`, `docs/coordinates.md`, `docs/architecture.md`.

Tests (`tests/unit`): inverse/composition properties on 200 random transforms, det/orthonormality,
reflection rejected, optical→base axes, known point in front of a pitched camera, planar pose
semantics, projection/back-projection round trip, image-axis directions, ray↔z-depth conversion,
invalid depth never valid, millimetre detection, frame validation (size mismatch, pose/source
mismatch, NaN marked valid, reflected pose). Ray-cast tests: a fronto-parallel wall gives constant
z-depth (confirms `t_hit` = z-depth), a box on the robot's left appears in the left half of the
image and back-projects to map y>0, a rotated robot (yaw 90°) stays consistent.

## M1 — one frame fully explained

Delivered:
- `datasets/synthetic.py`: scenes `room_clear`, `dominant_wall`, `no_ground`, `narrow_door`
  (Open3D RaycastingScene, seed, mesh hash, trajectory, calibration, noise σ(z)=0.002+0.0015 z²,
  1 % pixel dropout, optional rectangular holes). Writes `sensor_export/` and `ground_truth/` separately.
- `datasets/base.py` `SensorSequence`: reads manifest + `sensor_export/` only; validity mask; unit check.
- `geometry/ground.py`: sequential seeded RANSAC; each candidate accepted only if tilt ≤ 12°,
  height within 0.10 m of the floor expected in `base_link`, ≥ 200 inliers and ≥ 0.25 m² observed
  area. The best accepted one wins (not the largest plane). Reasons for rejection are exported.
- `runtime/pipeline.py`: backproject → base_link → 3 cm voxels (pixel link kept) → normals →
  ground → DBSCAN obstacles → visible-extent AABBs in `map` (pose supplied) or `base_link`.
  Exports `snapshot.json`, `explanation.json`, `points_labelled.ply`, `points_rgb.ply`, `points.npz`.
- `evaluation/` (the only reader of `ground_truth/`): ground precision/recall, plane normal and
  height error, visible-box IoU / centre error, full-object IoU, detection purity.

### Commands run

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install --no-deps open3d && .venv/Scripts/python -m pip install numpy scipy pydantic typer pyyaml pytest ruff plotly
.venv/Scripts/python -m pip install open3d            # pulls the remaining (dash, ...) deps
.venv/Scripts/python -m pip install --no-deps -e .
.venv/Scripts/python -m pytest                        # 27 passed in 7.7 s
.venv/Scripts/python -m ruff check src tests          # All checks passed
.venv/Scripts/clearpath3d m1-demo                     # writes artifacts/demo/m1_reference/
.venv/Scripts/python -m pip freeze --exclude-editable > requirements.lock
```

### Results (frame 0, 320×240, seed 0, `configs/synthetic.yaml`, pose_source = supplied)

| Scene | Ground | Precision / recall | Normal err. | Height err. | Predicted ground on walls |
|---|---|---|---|---|---|
| room_clear | found | 0.992 / 0.996 | 0.12° | 2.3 mm | 0.7 % |
| dominant_wall | found | 0.996 / 0.975 | 0.37° | 9.0 mm | 0.4 % |
| no_ground | **unavailable** (explicit warning) | — | — | — | — |

`dominant_wall`: the first (largest) extracted plane is the wall (tilt ≈ 90°), rejected; the floor is the
second candidate. `no_ground`: both candidates rejected (tilt, height, inliers/area); no traversability
asserted, the wall is still reported as an obstacle.

Obstacles (visible-part boxes; the full-object IoU is lower when only part of the object is visible):

| Scene / object | IoU visible | Centre err. visible | IoU full object |
|---|---|---|---|
| room_clear / table | 0.905 | 2.6 cm | 0.459 (partly occluded/out of view) |
| room_clear / box_a | 0.763 | 2.4 cm | 0.764 |
| room_clear / box_b | 0.808 | 2.2 cm | 0.808 |
| dominant_wall / box_a | 0.830 | 2.3 cm | 0.828 |

All detections have 100 % purity (one GT category per cluster); no detection lies on the floor.
Walls are clusters too (8 detections in room_clear: 3 objects + wall segments split by occlusions).

Timings (single run, i7-10510U, not profiled statistically): total ≈ 0.6–0.8 s per frame, dominated by
the ground RANSAC (225–612 ms) and voxel downsampling (63–163 ms). No real-time claim.

### Finding: statistical outlier removal hurts thin obstacles

With SOR (20 neighbours, std 2.0) on room_clear, 1461 points were removed, the table was fragmented
(table IoU 0.016 instead of 0.905) and ground recall dropped (0.918 vs 0.996). The synthetic data has no
flying pixels, so SOR only removed legitimate sparse points (thin legs, far floor). Decision: SOR is
**off** in `configs/synthetic.yaml` and stays available for real data / ablations. Regression test:
`test_thin_obstacle_survives_preprocessing`. To be re-evaluated on real data (Redwood, M2).

Also observed (not a bug): the table top occludes a band of the partition wall behind it and the upper
part of its legs, which splits the wall into several clusters. That is real occlusion.

## M2 — multi-view reconstruction with supplied poses

Delivered:
- `reconstruction/tsdf.py`: `TSDFFusion` (Open3D `ScalableTSDFVolume`), keyframes by translation (10 cm) or
  rotation (5°), optional exclusion mask (for moving tracks in M3). **Single conversion point**
  `open3d_extrinsic(T_map_camera) = T_camera_map`.
- `reconstruction/run.py`: TSDF + point-union baseline, `reconstruction_report.json` with pose source,
  p50/p95 timings, memory (RSS and Windows peak working set), hardware.
- `reconstruction/consistency.py`: sensor-only reprojection residual between two frames.
- `datasets/redwood.py` + `clearpath3d download-redwood`: explicit download and conversion of
  `SampleRedwoodRGBDImages` (11 MB, archive SHA-256 `4f161ba0…937b`, recorded in the manifest).
- `evaluation/reconstruction.py`: accuracy (recon → GT mesh) and completeness (observed GT surface → recon).
- Synthetic scene `room_scan`: 31 frames (rotations + 2.6 m of forward motion) in the room_clear layout.

### Blocking issue found and fixed: Open3D 0.20.0
With open3d 0.20.0 (Windows wheel), `ScalableTSDFVolume.integrate` and the tensor `VoxelBlockGrid` produced
**zero points**, including on Open3D's own tutorial recipe with the Redwood sample. open3d **0.19.0** gives
149 316 points on the same recipe. Decision D13: pin `open3d==0.19.0`. Guard test:
`test_tsdf_backend_produces_surface`. M1 results re-run under 0.19.0: identical.

### Redwood conventions, established by measurement (not assumed)
| Question | Test | Result |
|---|---|---|
| Pose direction of `trajectory.log` | reproject frame 4 into frame 0 | as `T_world_camera`: median 5.6 mm; inverted: 146 mm |
| z-depth or ray distance | flatness of the largest plane | as z: RMS 2.7 mm; as ray: 5.4 mm |
| World "up" | camera axes, then fused cloud | camera y ≈ −world y → `T_map_world = RotX(+90°)`; dominant fused plane 0.2° from map z |
| Reference `example_tsdf_pcd.ply` | nearest-neighbour distance | 3 m off in the trajectory world: it was fused in the **odometry.log** world. After alignment: median 3.9 mm, 99 % of the reference within 1 cm |

No mount calibration for this hand-held sequence → `T_base_camera = null` → single-frame ground analysis is
not available for Redwood (explicit error, no guessed base frame). Licence of the sample: **not verified**
(derived from the ICL-NUIM living room); the data is not redistributed.

### Commands run
```bash
.venv/Scripts/python -m pip install "open3d==0.19.0"
.venv/Scripts/clearpath3d download-redwood      # done via open3d.data in this session
.venv/Scripts/clearpath3d m2-demo               # writes artifacts/demo/m2_reference/
.venv/Scripts/python -m pytest                  # 40 passed in 36 s
.venv/Scripts/python -m ruff check src tests    # All checks passed
```

### Results — `room_scan` (320×240, seed 0, voxel 2 cm, trunc 8 cm, pose_source = supplied)
Accuracy = distance recon → GT mesh; completeness = distance from observed GT surface → recon.

| Method | Points | Acc. median | Acc. p95 | Acc. < 2 cm | Compl. < 2 cm |
|---|---|---|---|---|---|
| TSDF point cloud | 106 132 | 2.0 mm | 11.7 mm | 97.9 % | 99.4 % |
| TSDF mesh vertices | 108 783 | 2.1 mm | 12.9 mm | 97.3 % | 99.6 % |
| Point union (2 cm voxels) | 229 687 | 6.0 mm | 43.8 mm | 73.6 % | 99.97 % |

The TSDF averages the depth noise (σ grows with z²); the union keeps it. Both are equally complete on the
observed surface. These are synthetic results with exact poses: they say nothing about odometry.

Cost (single run, i7-10510U): TSDF integrate p50 10.7 ms / p95 20.5 ms per 320×240 frame, mesh + cloud
extraction 403 ms, whole fusion 1.2 s for 31 frames; RSS +181 MB after fusion. The union baseline took 3.2 s
and pushed the peak working set to 766 MB (2.36 M raw points): the TSDF is far cheaper in memory.
Redwood sample (640×480, 5 frames, voxel 7.8 mm): integrate p50 40 ms, 141 137 points, 0.8 s total.

Determinism: two runs give identical mesh vertices (tolerance 1 µm, `test_two_runs_equivalent`).
Inverting the extrinsic makes the accuracy median > 10 cm (`test_inverted_extrinsic_is_detected`).

## M3 — tracking, visibility map, unknown space and A*

Delivered:
- `mapping/voxel_evidence.py`: 5 cm log-odds voxel grid (`log_odds`, `obs_count`, `last_seen_s`,
  `last_frame`, update source). Exact vectorised Amanatides-Woo traversal: free strictly before the hit voxel,
  occupied at the hit, nothing behind, holes touch nothing; one update per voxel per frame (occupied wins).
  `delta_occ=+0.85`, `delta_free=-0.40`, clamp `[-2, 2]`. Robot-footprint evidence (source `robot_footprint`).
- `mapping/projection.py`: 2D map for a cylindrical robot (R=0.25 m, H=0.40 m, clearance 6 cm). Exact criterion
  in the module docstring: OCCUPIED if a band voxel is occupied; FREE only with ground support AND >= 80 % of the
  band voxels observed free; UNKNOWN otherwise; expiry -> UNKNOWN, never FREE. `explain_cell` = "why?" data.
- `mapping/changes.py`: change events (`freed` / `appeared`) against each voxel's last stable state, robot band only.
- `perception/tracking.py`: constant-velocity Kalman `[x,y,z,vx,vy,vz]`, chi-square (99 %) + 0.6 m gating,
  Hungarian assignment, tentative/confirmed/occluded/retired, visibility-aware misses, velocity damping while
  unobserved; clusters cut by the top image border are treated as structure (not tracked).
- `planning/astar.py`: 8-connected A*, no corner cutting, octile heuristic (admissible: penalties >= 0),
  proximity penalty, shortcut smoothing re-checked segment by segment, explicit causes
  (`start_not_free`, `goal_unknown`, `goal_occupied`, `goal_footprint_blocked`, `blocked_by_obstacles`,
  `blocked_by_unknown` via a diagnostic re-plan that treats unknown as free).
- `runtime/replay.py` `SequenceProcessor`: analyse -> track -> evidence -> changes -> TSDF (moved objects masked).
- `evaluation/mapping.py`, `evaluation/runner.py`, CLI `clearpath3d m3-demo`; scenes `hidden_box`, `moving_box`,
  extended `narrow_door`.

### Commands run
```bash
.venv/Scripts/python -m pytest                  # 70 passed in 123 s
.venv/Scripts/python -m ruff check src tests    # All checks passed
.venv/Scripts/clearpath3d m3-demo               # artifacts/demo/m3_reference/<scene>/{report.json,map_*.png}
```

### Acceptance criteria (plan, section 9) — all met on the synthetic scenes
| Criterion | Scene | Result |
|---|---|---|
| A too-wide robot stays blocked | narrow_door (0.35 m door) | R=0.25: `goal_footprint_blocked` (obstacle); R=0.10: path 3.35 m, 0 GT collisions |
| The area behind an obstacle stays unknown | hidden_box (box behind a solid cabinet) | box cells UNKNOWN, plan -> `goal_unknown` |
| A moved box does not free its old place without re-observation | moving_box | frame 12: old place OCCUPIED; end: re-observed -> UNKNOWN (floor under the box never seen), never FREE |

### Map and planning results (320×240, seed 0, pose_source = supplied, robot R=0.25 m H=0.40 m)
| Scene | Frames | Free coverage | False free cells | Unsafe traversable cells | Occ. precision (exact / ±1 cell) |
|---|---|---|---|---|---|
| room_scan | 31 | 64.1 % | 0 | 0 | 0.556 / 0.988 |
| narrow_door | 17 | 43.9 % | 0 | 0 | 0.550 / 0.972 |
| hidden_box | 31 | 60.0 % | 0 | 0 | 0.553 / 1.000 |
| moving_box (end) | 22 | 48.1 % | 0 | 0 | 0.537 / 0.928 |
| **moving_box at frame 12** | 12 | 43.3 % | **48** | **93** | — |

"Unsafe traversable" = cells the planner would accept as robot centre (not blocked) where the robot disk
touches a GT obstacle. The exact occupied precision is low because wall faces sit on voxel boundaries (5 cm):
noise puts about half of the surface hits in the cell on the room side; within one cell it is 0.93–1.0.

Planned paths checked against GT geometry: room_scan to the corridor 4.49 m (min clearance 0.325 m > R),
behind the table 2.85 m (0.318 m), narrow door with the small robot 3.35 m (0.175 m > 0.10 m). No collision.

**Measured limitation (moving_box, frame 12):** the box's *new* place had been observed free before the move
and has not been re-observed yet, so the map still says FREE there (48 false-free cells). The planner did not
use it in this query (`blocked_by_unknown`), but the map was wrong. No map can see an unobserved change; the
mitigations are evidence expiry (-> unknown) and re-observation before acting. Reported, not hidden.

Change events in moving_box (robot band only, >= 10 voxels): `appeared` 69 voxels at (2.15, 0.80) frame 17,
13 voxels frame 18; `freed` 14 + 83 voxels at (1.8, 1.7) frames 20–21. Other scenes: 0 events.

Cost (i7-10510U, 320×240, full pipeline incl. TSDF): p50 0.57–0.71 s/frame, p95 0.87–1.14 s; the evidence
update (ray traversal, pixel stride 2) dominates (~0.2–0.35 s). RSS ≈ 0.3 GB. Not real time; not optimised yet.

### Problems found and fixed during M3 (with evidence)
1. Start pose blocked (`start_not_free`) everywhere: the area behind the robot is never observed, and dilating
   unknown by R+margin closed the start. Fix: forbidden zone = obstacle/unknown dilated by **R**; the
   R..R+margin band is allowed at a high cost (20/m) and reported (`margin_cells`). Deviation from the plan's
   "dilate by R + margin", documented in D17.
2. Robot-footprint evidence had half-cell gaps (planner reasons on cell centres) and holes between distant poses.
   Fix: sweep the footprint between consecutive poses (< 0.5 m apart) with reach R + half a cell diagonal
   (over-claim bounded by 3.5 cm, D18).
3. An old box place could not be freed (clamp 3.5 needed ~10 free observations). Clamp lowered to 2.0 (D19).
4. Change detection was frame-to-frame (missed slow transitions) and noisy on surfaces. Fix: last-stable-state
   comparison, stricter thresholds, robot band only, >= 10 voxels.
5. Track centres of partially visible walls move with the viewpoint, so "displacement > 0.3 m" flagged static
   structure as dynamic. Fix: dynamic = tracks overlapping an `appeared` change event; displacement criterion off.
6. A tracked box drifted to z = 0.47 m while unobserved (noisy velocity). Fix: velocity damping per miss.
7. `hidden_box` first version was not hidden (the camera saw the box under the open table). Replaced by a solid
   cabinet; verified UNKNOWN.

## M4 — API, viewer, replay, evidence inspector, ROS2 adapter

Delivered and verified:
- `runtime/session.py` `ReplaySession` (framework-free): one worker thread per replay, play/pause/step/reset,
  deterministic reset, versioned events, binary artefacts cached by map version, plans, viewpoints, evidence
  (`cell:i:j`, `point:x:y`, `track:id`, `change:k`).
- `apps/api` (FastAPI): endpoints of the plan (`/v1/replays`, `commands`, `snapshot`, `events` (SSE, metadata only),
  `artifacts/{map2d,points,voxels,blocked}` (binary + headers), `plans`, `viewpoints`, `evidence/{id}`);
  long work via `run_in_threadpool`, never on the event loop.
- `apps/viewer` (React 18.3.1, TypeScript 5.6, Three.js 0.169 / R3F 8.17): 3D voxels + current cloud + tracks +
  robot + path; 2D map (unknown toggle), click modes start / goal / "why?"; Plan; Next best view; change inspection;
  pose-source and ground badges; timings, warnings, transfer size. Display rotation z-up → y-up only.
- Verified in the built-in browser: room_scan played to the end, plan to the corridor drawn in 2D and 3D,
  "why?" table shows log-odds / observations / last frame / source per voxel; map counts identical to the CLI
  run (765 / 6013 / 7222); no console errors; `npm run build` OK (bundle-size warning only).
  Screenshots: `artifacts/demo/m4_reference/*.jpg`, `artifacts/demo/m5_reference/viewer_hidden_box_nbv.jpg`.
- Transfer per UI refresh measured in the viewer: ~360–460 KiB (map + occupied voxels + ~8.5 k points);
  SSE events < 4 KB each (tested).

ROS 2 (code written, **integration NOT verified**):
- `ros2_ws/src/clearpath3d_ros`: `conversions.py` (pure, 5 unit tests on Windows: OccupancyGrid indexing and
  origin, PointCloud2 layout + RGB packing, tf quaternion round trip, source stamps, markers, path),
  `replay_node.py` (PointCloud2, OccupancyGrid -1/0/100, MarkerArray, Path, tf map→base_link from supplied poses
  as the single authority, static base_link→camera_optical; QoS map/path reliable + transient_local, others
  reliable volatile depth 5), launch file, RViz2 config.
- Blocker: `sudo` in WSL needs the owner's password. Script ready:
  `ros2_ws/scripts/install_ros2_jazzy_wsl.sh` (official Jazzy steps), then `ros2_ws/scripts/verify_ros.sh`
  (colcon build, launch, topic/QoS/tf checks, rosbag2 record) writes `ros2_ws/verify_log.txt`.
  Until that log exists and is reviewed, nothing ROS-related is claimed to work.

Not done in M4: the 90–120 s demo video (needs a screen recording by the owner; the UI is ready).
Clean-clone check (2026-10-10): the files that would be committed (`git ls-files --others --exclude-standard`,
127 files) were copied to an empty directory; a fresh Python 3.12 venv installed `requirements.lock`, pytest
exited 0 (Redwood tests skipped: no download) and `m1-demo` reproduced the reference numbers exactly.
Not covered: the viewer's `npm install` and ROS in that fresh copy.

## M5 — next-best-view (V1.1 extension)

Delivered:
- `planning/frontiers.py`: frontier cells, reachable distance field (known free only).
- `planning/next_best_view.py`: candidates = reachable robot-centre cells (spacing 0.35 m, ≤ 4 m) × 8 headings;
  gain = distinct *never-observed* voxels hit first along 24×16 rays (occupied/ambiguous voxels stop rays without
  gain; unknown not transparent), restricted to the robot slab; score = gain/100 − 0.2·m − 0.1·rad; explicit
  statuses `no_map`, `no_reachable_candidate`, `no_informative_candidate`.
- `evaluation/active.py`, `evaluation/campaign.py`, CLI `clearpath3d m5-experiment`: simulated loop
  (propose → A* through known free → footprint sweep → render with the full world → update), random baseline on the
  same candidates, same seeds.
- API `GET /viewpoints` and viewer "Next best view" button.

Results (details and caveats: `docs/experiments.md`, E4): NBV beat random in 6/6 seed pairs; mean coverage-gain
difference +0.128 (hidden_box) and +0.096 (narrow_door) with less travel. The box hidden behind the cabinet was
found in 1/3 NBV runs. NBV stopped by itself (`no_informative_candidate`) after 7–9 views in hidden_box: the
"no accessible observation" case. One unsafe traversable cell appeared in 2 NBV runs of narrow_door.

Findings during M5:
1. From a pose where the robot never moved, nothing is reachable: the camera's ~1 m blind zone stays unknown and the
   planner never crosses unknown. Exploration therefore starts after the scripted scan (as in the plan's story).
2. An aligned candidate grid missed thin reachable corridors; replaced by spacing-based sampling.
3. Ambiguous surface voxels inflated the predicted gain forever; the gain now counts never-observed voxels only.
4. First implementation: ~18 s per decision; coarser rays/steps → 1–2.5 s (approximation documented in the code).

### Commands run (M4–M5)
```bash
.venv/Scripts/python -m pip install fastapi "uvicorn[standard]" httpx
cd apps/viewer && npm install && npx tsc -b && npm run build
.venv/Scripts/python -m uvicorn clearpath3d_api.main:app --app-dir apps/api --port 8000   # + npm run dev
.venv/Scripts/clearpath3d m5-experiment          # ~10 min
.venv/Scripts/python -m pytest                   # 86 passed in 130 s
.venv/Scripts/python -m ruff check src tests apps/api ros2_ws/src   # All checks passed
```

## After V1.1 — held-out evaluation, safety fixes, optional ICP (2026-10-10)

### Held-out protocol and benchmark
- `datasets/procedural.py`: seeded room generator (size, partition, door 0.30–0.45 m or 0.75–1.1 m, boxes,
  optional solid cabinet). `evaluation/benchmark.py` + `clearpath3d benchmark <split>`.
- Splits frozen in `experiments/splits/v1.json` before any test run; code fingerprint frozen in
  `experiments/splits/test_freeze.json`; the test split (20 unseen rooms) was run once with that code.
- Test results (single run): ground found 20/20 (P 0.996, R 0.990), **0 false free, 0 unsafe cells**,
  narrow doors refused 5/5, passable doors traversed 15/15, 0 GT collisions, NBV better than random in 19/20
  rooms (mean +0.106 coverage, bootstrap 95 % over rooms [0.080, 0.130]). Details: `docs/experiments.md`, E5.

### Safety bugs found on dev/validation and fixed (each with a regression test or scenario)
1. **Map grid smaller than the room**: walls outside the grid → planner saw free cells at the border (82 unsafe
   cells in proc_003). Fix: outside the map counts as unknown; grid enlarged to [-1, 8.5] × [-1, 6.5] m.
2. **Discretisation**: an occupied cell's obstacle can be half a cell diagonal closer than its centre (0.244 m
   true clearance for R = 0.25). Fix: occupied cells, and unknown cells adjacent to them, dilated by R + 3.5 cm.
3. **Grazing ray through a partly occupied voxel** made a cabinet edge cell free. Fix: cells next to an obstacle
   (any height) need two net free observations. A global two-observation rule was tried and rejected
   (0/7 passable doors on dev and validation).
4. **Frames without a pose crashed the replay** (tracker received truncation flags without detections).
   Found while integrating ICP; fixed + regression test.
- Also added: scene `sensor_dropout` (15 % dropout, 10 holes/frame, 3× noise → 0 false free, test),
  `low_overlap`, `room_scan_dense`.

Because the planner became more conservative, the M3/M5 reference numbers were regenerated
(current values: `docs/experiments.md`, E3/E4). Notable change: in `m5-experiment` the hidden box is no longer
found by NBV in any seed (was 1/3), and the NBV advantage there is smaller (+0.084 / +0.053).

### M6 (optional) — ICP pose estimation
- `reconstruction/registration.py`: multi-scale point-to-plane ICP (0.2/0.08/0.04/0.02 m), Tukey kernel,
  constant-velocity and zero-motion initial guesses, FPFH+RANSAC fallback, gates: fitness, inlier RMSE, per-frame
  motion plausibility, deviation from the motion hypotheses, and a point-to-plane degeneracy measure.
  Rejected frames carry no pose; after 10 consecutive rejections tracking is declared lost (map frozen).
- API/viewer: `pose_mode: icp` (first pose supplied as anchor) with an "estimated" badge and "tracking lost".
- Results (E6): room_scan_dense ATE 1.2 cm on 83/121 frames (lost after a degenerate segment) vs 32 cm without
  gates; Redwood real 1.5 cm; low_overlap jumps rejected. Map with estimated poses: 0 false free with gates,
  66 false free / 203 unsafe cells without. Thresholds were chosen on room_scan_dense (disclosed).
- Findings: a converged ICP with fitness 0.98 was 25 cm wrong (sliding along a wall); Open3D's information matrix
  could not detect it (translation block N·I); the point-to-plane Σnnᵀ measure did.

### Commands run (this round)
```bash
.venv/Scripts/clearpath3d benchmark dev | validation | test     # test: once, frozen code
.venv/Scripts/clearpath3d m3-demo ; .venv/Scripts/clearpath3d m5-experiment ; .venv/Scripts/clearpath3d m6-odometry
.venv/Scripts/python -m pytest                                   # 94 passed (2 min 40 s), on the frozen code
```

## M6 (optional) — 2D semantics lifted to 3D (2026-10-10)

- `perception/semantics.py`: `Detection2D`, lifting by mask / whole box / box + nearest depth mode, robust 3D
  extent, score-weighted label votes on geometric tracks (>= 2 votes, 60 % majority). Optional in the replay
  (`ReplayConfig.semantic_detector`); geometry is bit-for-bit unchanged with or without it.
- `perception/detectors_torch.py`: torchvision Mask R-CNN (lazy import; optional dependency).
- `evaluation/semantics_eval.py`: ORACLE detector from simulator masks (evaluator side only) for the lifting
  ablation; real-image run on Redwood with multi-view consistency. CLI `clearpath3d m6-semantics`.
- Dependencies: `torch 2.14.1+cpu`, `torchvision 0.29.1+cpu` in `requirements-semantics.txt` (official CPU
  index), kept OUT of `requirements.lock` so the core install stays light (~1 GB saved).
- Results (E7): exact masks purity 1.00 vs whole-box p10 0.24; oracle tracks 6 labelled / 6 correct / 0 wrong;
  real images: chair consistent across 3 frames (10 cm spread), one false "cat" on the armrest.
- Tests: 102 passed (5 min 11 s) incl. 6 semantics unit tests and 2 integration tests.
- Protocol note: this code was added AFTER the frozen test-split run (fingerprint `169668836189bf6b`). It only
  adds an optional path (off by default; geometry bit-identical when off, tested). The test split was **not**
  re-run; the reported held-out numbers belong to the frozen fingerprint.

## Interactive demo + real-data benchmark (owner request "ajouter des features", 2026-10-10)

### Interactive demo — done, tested, checked in the browser
- `runtime/session.py`: `ingest()` for simulator frames, per-step map history (`timeline`, `map_diff`),
  height-aware re-projection in `plan`. `apps/api/clearpath3d_api/sim.py`: `SimWorld` (the simulator owns the
  true world and can move objects) and `Driver` (look-before-move, re-plan at every step, SSE `drive` events).
  Viewer: R/H/margin sliders with live re-planning, move-object mode, Drive/Stop, timeline + before/after diff.
- Bugs found and fixed while building it:
  1. the final SSE event's status was overwritten by the drive outcome → clients never saw the end (now `outcome`);
  2. **the robot drove through a box placed in the corridor it had observed free**: the box was inside the camera's
     ~1 m blind zone and the driver turned while driving. Fix: turn in place to face the travel direction, observe,
     re-plan. Safety test (robot disk vs the simulator's true box) now passes: reached in 19 steps, 3 re-plans;
  3. ground support lost after one later ray ("robot stood here" read from the last update; a grazing ray through
     the floor voxel) → persistent per-column robot support + floor hit counter. Regression test added.
- The API moved to port **8010** (port 8000 is used by another local project of the owner).
- Browser check: replay → move box_b → drive → "finished: reached · replans 3 · 19 steps"; timeline diff
  (+212 occupied, +1403 free, +113 unknown vs step 30). Screenshots in `artifacts/demo/interactive_demo/`.

### Real data — TUM RGB-D `freiburg2_pioneer_360` (CC BY 4.0) — done (see `docs/experiments.md`, E9)
- Adapter `datasets/tum.py` (conventions measured), CLI `download-tum`, `real-benchmark`.
- Ground on real frames 83/83 (normal 1.2°, height 2.0 cm median). Map with mocap poses + plan 8.77 m to the farthest
  reachable cell. Depth-only ICP does not track this sequence: default 4 m range → all frames refused (degenerate);
  8 m (post hoc, disclosed) → 40 frames at 2.9 cm then refused; without gates → 4.3 m off. ICP-pose map: 0 free cells
  where the mocap map is occupied.
- Fixes: map bounds derived from the trajectory (`bounds_around`; the first run had the robot outside the fixed
  synthetic grid — archived in `experiments/real_run1_invalid_bounds/`); plan-demo goal = farthest REACHABLE cell
  (recomputed from the same map, deterministic: identical cell counts).

### Protocol note
The mapping change (ground support) alters synthetic map results after the frozen held-out test run
(fingerprint `169668836189bf6b`). The test split was **not** re-run; E5 numbers belong to that fingerprint.
M3/M5 reference artefacts were not regenerated either (they predate this change).

### Commands run
```bash
.venv/Scripts/python -m pytest                       # 108 passed in 4 min 37 s; ruff clean; viewer build OK
.venv/Scripts/clearpath3d download-tum ; .venv/Scripts/clearpath3d real-benchmark
```

## Blockers / owner decisions pending

- **ROS2 (M4)**: run `bash /mnt/c/Users/Lenovo/Documents/ClearPath3D/ros2_ws/scripts/install_ros2_jazzy_wsl.sh`
  inside WSL (asks for your sudo password; ~2–3 GB on C:). Then I can run `verify_ros.sh` myself.
- **Demo video**: to be recorded by the owner (OBS or Xbox Game Bar) with the viewer.
- **package.xml maintainer email** is a placeholder (`mahdi.gharbi@example.invalid`): give the address to use.
- **Disk space**: ~20 GB free. Fine for synthetic + Redwood sample; a longer public RGB-D sequence (M2+)
  must be chosen with its size in mind.

## Next steps

1. Owner: install ROS 2 Jazzy in WSL (`ros2_ws/scripts/install_ros2_jazzy_wsl.sh`), then I run `verify_ros.sh`.
2. Owner: demo video; package.xml maintainer email; commit the repository under the owner's account.
3. Semantics on real data with ground truth (labelled RGB-D sequence) to measure accuracy, not only consistency.
4. Evaluation breadth: other generator families (clutter, non-axis-aligned furniture), real RGB-D sequences
   with ground truth (e.g. a longer public dataset; licence and size to be checked first).
5. Performance (profiled): ray traversal ~0.2–0.35 s/frame, NBV ~1–2.5 s, ICP ~0.5–1 s/frame.
