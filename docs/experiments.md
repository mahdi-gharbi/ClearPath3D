# Experiments

All numbers come from saved reports (`artifacts/demo/*/`, `artifacts/benchmark/*/`) produced on one
laptop (Intel i7-10510U, 4 cores, 19.8 GB RAM, Windows 10, CPU only, Open3D 0.19.0). Synthetic data
unless stated. Poses are **supplied** except in E6.

## Evaluation protocol (E5)

- Procedural rooms (`datasets/procedural.py`): size, partition, door width/position, boxes and an
  optional solid cabinet vary with the seed.
- Splits frozen in `experiments/splits/v1.json` **before** any test run: dev `proc_000-009`, validation
  `proc_010-019`, test `proc_100-119` (disjoint geometries, no shared frames).
- Code and parameters were changed only after looking at dev/validation. The code fingerprint was frozen in
  `experiments/splits/test_freeze.json`; the test split was run **once** with that exact code
  (fingerprint `169668836189bf6b`, checked in `artifacts/benchmark/test/summary.json`).
- Planning query (evaluator-chosen): first scripted pose → 1 m past the door. GT answer: the door is passable
  iff it is wider than the robot diameter (0.5 m). NBV vs random: 4 views each after the scripted scan.

## E5 — Held-out benchmark (procedural rooms)

`clearpath3d benchmark {dev,validation,test}` → `artifacts/benchmark/<split>/`.

| Split | Rooms | Ground found | Ground P / R (median) | False free | Unsafe | Narrow doors refused | Passable doors traversed | GT collisions | NBV − random coverage (mean, bootstrap 95 % over rooms) | NBV better |
|---|---|---|---|---|---|---|---|---|---|---|
| dev | 10 | 10 | 0.996 / 0.992 | 0 | 0 | 3/3 | 7/7 | 0 | +0.126 [0.089, 0.160] | 10/10 |
| validation | 10 | 10 | 0.998 / 0.991 | 0 | 0 | 3/3 | 7/7 | 0 | +0.123 [0.071, 0.163] | 9/10 |
| **test (held out, single run)** | **20** | **20** | **0.996 / 0.990** | **0** | **0** | **5/5** | **15/15** | **0** | **+0.106 [0.080, 0.130]** | **19/20** |

"Unsafe" = cells the planner accepts as robot centre where the robot disk touches a GT obstacle (also tracked
during the NBV/random explorations: max 0 on all three splits). Median free coverage after the scripted scan:
0.63 (dev), 0.60 (validation), 0.63 (test).

What dev/validation revealed and changed before the test run (each documented in `PROGRESS.md`):
1. Rooms larger than the map grid had walls outside the grid → border now counts as unknown; grid enlarged.
2. Occupied cells dilated from their centres allowed 0.244 m true clearance for R = 0.25 m → extra half-cell-
   diagonal dilation for occupied cells.
3. A single grazing ray marked a cabinet-edge cell free → two free observations required next to obstacles.
   (Requiring two everywhere was tried and rejected: 0/7 passable doors on dev and validation.)
4. An unobserved cabinet corner (unknown) allowed 0.247 m → unknown cells adjacent to obstacles get the same
   extra dilation. (Applying it to all unknown cells was tried and rejected: blocked a passable door.)

Caveats: synthetic rooms from one generator family (axis-aligned boxes, flat colours, one camera model); the
benchmark does not cover clutter, slopes, glass, real sensor artefacts or estimated poses.

## E1 — Single-frame ground and obstacles (M1)

`clearpath3d m1-demo` → `artifacts/demo/m1_reference/`. Frame 0, 320×240, seed 0.

| Scene | Ground | Precision / recall | Normal err. | Height err. |
|---|---|---|---|---|
| room_clear | found | 0.992 / 0.996 | 0.12° | 2.3 mm |
| dominant_wall | found (wall rejected, tilt ≈ 90°) | 0.996 / 0.975 | 0.37° | 9.0 mm |
| no_ground | unavailable (explicit) | — | — | — |

Ablation: statistical outlier removal ON erased thin legs (table IoU 0.016 vs 0.905) → OFF by default.

## E2 — TSDF vs point union (M2)

`clearpath3d m2-demo` → `artifacts/demo/m2_reference/`. room_scan, 31 frames, voxel 2 cm.

| Method | Acc. median | Acc. p95 | Acc. < 2 cm | Compl. < 2 cm | Peak memory |
|---|---|---|---|---|---|
| TSDF | 2.0 mm | 11.7 mm | 97.9 % | 99.4 % | +181 MB RSS |
| Point union | 6.0 mm | 43.8 mm | 73.6 % | 99.97 % | 766 MB peak working set |

Real data: the Redwood sample (5 frames) checks conventions only (reprojection 5.6 mm vs 146 mm inverted;
agreement with Open3D's reference reconstruction: median 3.9 mm).

## E3 — Visibility map and planning on the named scenes (M3, current code)

`clearpath3d m3-demo` → `artifacts/demo/m3_reference/`. Robot R = 0.25 m, H = 0.40 m.

| Scene | Free coverage | False free | Unsafe | Planning |
|---|---|---|---|---|
| room_scan | 63.6 % | 0 | 0 | to corridor and behind the table: ok, 0 GT collisions |
| narrow_door | 43.1 % | 0 | 0 | R=0.25 `goal_footprint_blocked`; R=0.10 ok |
| hidden_box | 59.1 % | 0 | 0 | hidden box area UNKNOWN → `goal_unknown` |
| moving_box, end | 48.0 % | 0 | 0 | new box place → `goal_occupied` |
| moving_box, frame 12 | — | **48** | — | stale free at the box's new place (not re-observed yet) |
| sensor_dropout (test) | lower than room_scan | 0 | 0 | 15 % dropout, 10 holes/frame, 3× noise |

## E4 — Next-best-view vs random on the named scenes (M5, current code)

`clearpath3d m5-experiment` → `artifacts/demo/m5_reference/`. Same candidates, views, per-view travel budget
and sensor-noise seeds for both policies.

| Scene | Seed | NBV coverage gain | Random | NBV travel | Random travel |
|---|---|---|---|---|---|
| hidden_box (≤ 10 views) | 0 / 1 / 2 | +0.284 / +0.288 / +0.290 | +0.160 / +0.231 / +0.220 | 9.6 / 9.7 / 8.9 m | 14.1 / 20.4 / 14.5 m |
| narrow_door (6 views) | 0 / 1 / 2 | +0.213 / +0.213 / +0.212 | +0.147 / +0.151 / +0.183 | 5.2 m each | 5.8 / 7.5 / 6.2 m |

Mean difference: +0.084 (hidden_box), +0.053 (narrow_door); 0 false free and 0 unsafe cells in all runs.
**The box hidden behind the cabinet was not discovered by either policy in these runs** (with the earlier,
less conservative planner NBV found it in 1/3 seeds: the safer planner reaches fewer viewpoints).
The larger held-out comparison is E5.

## E6 — ICP odometry (optional research, M6)

`clearpath3d m6-odometry` → `artifacts/demo/m6_reference/report.json`. Local odometry (no loop closure).
ATE/RPE after first-frame alignment against simulator GT. **Thresholds were chosen on `room_scan_dense`**:
the ablation below is not a held-out result; low_overlap and Redwood were not used for tuning.

| Variant (room_scan_dense, 121 frames, 3–8°/frame) | Frames estimated | ATE RMSE | RPE trans | RPE rot |
|---|---|---|---|---|
| **ICP, all gates (default)** | 83 (lost after a degenerate segment) | **1.2 cm** | 1.4 mm | 0.04° |
| no degeneracy gate | 116 | 46.8 cm | 7.2 cm | 0.05° |
| no motion gate | 83 | 1.2 cm | 1.4 mm | 0.04° |
| no gates | 121 | 32.1 cm | 4.1 cm | 0.05° |
| fine pyramid only (0.08/0.04/0.02 m) | 95 | 1.1 cm | 1.4 mm | 0.05° |
| point-to-point | 79 | 46.0 cm | 3.3 cm | 5.2° |
| baseline: static pose | 121 | 142.8 cm | — | — |

- Degeneracy gate = smallest eigenvalue of the point-to-plane information Σnnᵀ/N (< 0.03 rejected). Open3D's
  `get_information_matrix_from_point_clouds` was not usable (translation block = N·I).
- On this dense sequence the fine pyramid alone tracks more frames; the coarse 0.2 m level was needed on the
  sparse room_scan (13°/frame turns converged to a wrong minimum without it). Default kept as chosen before
  this run; the trade-off is reported, not re-tuned.
- low_overlap (failure case): 70° turns and a 0.8 m jump are rejected (frames 2–5), the 10° turn accepted.
- Redwood sample (real data, 5 frames): ATE 1.5 cm, RPE 8 mm / 0.29° vs the dataset trajectory.

Map built with estimated poses (first pose supplied as anchor, all others estimated):

| Poses | Frames in map | Free coverage | False free | Unsafe |
|---|---|---|---|---|
| supplied | 121 | 64.7 % | 0 | 0 |
| ICP, all gates | 83 | 58.2 % | **0** | **0** |
| ICP, no gates | 121 | 60.8 % | **66** | **203** |

Accepting every converged ICP result gives a map that would put the robot into obstacles; the gates trade
coverage (and declared tracking loss) for a safe map.

## E7 — 2D semantics lifted to 3D (optional research, M6)

`clearpath3d m6-semantics` → `artifacts/demo/m6_semantics/` (`oracle_report.json`, `redwood_report.json`,
overlay PNGs). Geometry never depends on labels: with and without semantics the voxel evidence is identical
(checked bit-for-bit). A track gets a label only with >= 2 score-weighted votes and a 60 % majority.

**Lifting ablation with an ORACLE detector** (simulator instance masks — this isolates the 2D→3D geometry,
it is not a detector result). room_scan, every 2nd frame, 21 lifted detections per configuration.
Purity = share of lifted points that belong to the object; IoU against the object's visible 3D box.

| Masks | Lifting | Purity median / p10 | IoU visible (median) |
|---|---|---|---|
| exact | mask | **1.00 / 1.00** | **0.71** |
| exact | mask eroded 2 px | 1.00 / 1.00 | 0.53 |
| exact | whole 2D box (naive) | 0.75 / **0.24** | **0.09** |
| exact | box + nearest depth mode | 0.85 / 0.25 | 0.48 |
| dilated 4 px (sloppy) | mask | 0.76 / 0.59 | 0.09 |
| dilated 4 px (sloppy) | mask eroded 2 px | 0.87 / 0.74 | 0.12 |
| dilated 4 px (sloppy) | box + nearest depth mode | 0.81 / 0.23 | 0.48 |

Projecting a whole image box drags the background in (in 10 % of cases three quarters of the points are not on
the object); the depth-mode filter helps but does not remove the tail; erosion only pays off with sloppy masks.

End to end (oracle, room_scan): 18 tracks, **6 labelled, 6 correct, 0 wrong**, 5 object tracks left unlabelled
(too few votes), 7 structure tracks (walls) correctly left without a label.

**Real images, pretrained detector** (torchvision Mask R-CNN R50-FPN, COCO weights from download.pytorch.org,
score ≥ 0.6) on the 5 Redwood frames with supplied poses. No semantic ground truth exists for this sample:
the measure is multi-view consistency, plus visual inspection of the saved overlays.

- Detections: chair in frames 0, 3 (twice), 4; nothing in frame 2; **"cat" (0.82) in frame 1 is a false
  positive on the chair's armrest** (visible in `redwood_000001_maskrcnn.png`). Vote fusion would keep "chair".
- The chair falls in the same 3D place across 3 frames: centroid spread 10.1 cm (mask), 12.0 cm (whole box),
  25.5 cm (box + depth filter, which here cut the chair's thin metal frame).
- 5 frames are far too few for any accuracy claim; this only shows the real-image path runs end to end.

Bug found by the unit tests: a surface seen face-on has zero 3D thickness, so its overlap with a track box
was 0 and labels were never attached; both boxes are now padded by 5 cm before the overlap test.

## E8 — Interactive demo: simulated drive with re-planning (feature added 2026-10-10)

Viewer + API (`/world`, `/world/move`, `/drive`, `/timeline`, `/diff`). The simulator owns the true world and
renders new views; perception only receives the rendered frames (`ReplaySession.ingest`).
Screenshots: `artifacts/demo/interactive_demo/`. Tests: `tests/integration/test_demo_api.py` (5 tests).

Scenario checked in the browser and in a test: after the room_scan replay, `box_b` is moved by the simulator
onto the corridor the robot had observed free (the map is now stale), then "Drive to goal" to (1.2, 2.35).

| Driver version | Result |
|---|---|
| first version (turn while driving) | drove **through** the box: it was placed inside the camera's ~1 m blind zone and the stale map said free — caught by the safety test (robot disk vs the simulator's true box) |
| look-before-move (turn in place to face the travel direction, observe, re-plan) | reached the goal in 19 steps, **3 re-plans**, 5 look-turns, robot disk never touches the true box |

Also found with the demo and fixed: (a) the "robot stood here" ground support was read from the voxels' last
update and lost as soon as a later sensor ray passed — now a persistent per-column flag; (b) a single grazing
ray through a floor voxel removed floor support — support now needs >= 1 floor hit not confirmed free
since. Regression test `test_ground_support_survives_later_rays`.

Remaining limitation shown by the demo: the interior of the moved box keeps old free evidence (from the scan,
before the move) mixed with new hits on its faces, so its centre cell is UNKNOWN rather than OCCUPIED. It is
never FREE and the planner avoids it, but it is not a clean occupied footprint.

## E9 — Real data: TUM RGB-D `freiburg2_pioneer_360` (feature added 2026-10-10)

`clearpath3d download-tum` (0.66 GB, CC BY 4.0, cite Sturm et al., IROS 2012; archive SHA-256 in the manifest)
then `clearpath3d real-benchmark` → `artifacts/demo/real_tum_pioneer_360/`. A Kinect on a Pioneer robot turning
in a large hall; reference poses from motion capture. **There is no geometric ground truth of the scene**, so
map and reconstruction results are measured *relative to the mocap-pose run*, not against the true world.

Conventions measured in the adapter (`datasets/tum.py`): 830/1209 frames kept (the others fall
in motion-capture gaps > 50 ms; no pose is invented); floor tilt vs the mocap z axis 1.467°;
derived camera mount height 0.6212 m (std 1.5 cm),
roll/pitch std [0.738, 0.592]° → a rigid mount, as expected on a robot.

**Ground detection on real frames** (every 10th frame): found 83/83, normal error median
1.186°, height error median 2.0 cm (p95 3.5 cm)
against the floor fitted in the mocap world.

**ICP odometry vs motion capture** (`*_posthoc` = added AFTER seeing the first run, see below):

| Variant | Frames estimated | ATE (first-frame align.) | ATE (best-fit SE(3)) | Tracking lost | Rejections |
|---|---|---|---|---|---|
| ICP, all gates, range 4 m (default) | 1/830 | — | — | yes | degenerate 10 |
| ICP, all gates, range 8 m (post hoc) | 40/830 | 3.7 cm | 2.9 cm | yes | translation 10 |
| no degeneracy/motion gates, 4 m | 825/830 | 429.9 cm | 183.1 cm | no | translation 5, fitness 1, rotation 1 |
| no degeneracy/motion gates, 8 m (post hoc) | 40/830 | 3.7 cm | 2.9 cm | yes | translation 10 |

- The default 4 m range (chosen for the synthetic rooms) drops ~46 % of the valid depth in this hall (diagnostic
  on 7 sampled frames, 11–47 % per frame; not part of the saved report): what is left
  is mostly floor, a degenerate geometry, and the gate rejects every frame (correct refusal, useless result).
- 8 m (the Kinect's nominal range) tracks the first 40 frames at 2.9 cm (best-fit ATE), then ICP converges to
  implausible jumps (> 0.5 m per frame while the robot moves a few cm) and is rejected until tracking is declared lost.
  The degeneracy/motion gates are not what stops it here: the basic plausibility gate is, in both 8 m variants.
- Without the degeneracy/motion gates at 4 m, ICP "succeeds" on 825/830 frames and is **4.3 m off** (1.8 m after the
  best rigid alignment).
- Conclusion on real data: depth-only frame-to-frame ICP does not track this sequence; the gates make it refuse
  instead of producing a wrong trajectory. A real system needs wheel odometry, colour/visual features or loop closure.

**Maps** (every 2nd frame, robot R = 0.25 m; bounds now derived from the trajectory — the first run used the fixed
synthetic bounds and the robot started outside the grid, so it was discarded, see `experiments/real_run1_invalid_bounds/`):
- mocap poses: 14045 occupied, 38334 free,
  55201 unknown cells (4.82 s/frame on the larger grid).
  Far Kinect readings (3–6 m) leave a band of noisy occupied cells: conservative, not clean (`map_mocap.png`).
- ICP-pose map (gated, 8 m; only the 40 tracked frames): 346 free cells,
  **0 free where the mocap map is occupied**, agreement
  98.4 % on cells known in both.
- Plan demo on the mocap map (start → farthest cell reachable through observed free space): `ok`,
  8.7656 m (`map_mocap_plan.png`).

**TSDF**: surface fused with the 20 ICP-pose frames vs the mocap-pose surface: median
6.3 mm, p95 30.3 mm, 99.1 % within 5 cm (only the
tracked beginning of the sequence; compared with the mocap-pose surface, not with the true scene).
