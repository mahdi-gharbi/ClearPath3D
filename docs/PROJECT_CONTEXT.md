# ClearPath3D project context

Reviewed on 2026-10-11 against the local source and saved reports.

## Purpose

ClearPath3D is an RGB-D spatial scene inspector and active-perception prototype.
It identifies ground and visible obstacles, reconstructs scenes, tracks observations,
and plans geometric paths for a simulated cylindrical robot. The viewer lets users
inspect why cells are classified, choose goals, request new viewpoints, move simulated
objects, drive the simulated robot, and compare map history.

## Architecture

- `src/clearpath3d/contracts.py`: validated RGB-D inputs, frame conventions, pose provenance,
  ground status, and serializable scene outputs.
- `datasets/`: sensor-sequence readers and synthetic/procedural scene generation;
  Redwood and TUM adapters convert dataset conventions.
- `geometry/` and `perception/`: backprojection, preprocessing, gated ground RANSAC,
  DBSCAN obstacle clustering, Kalman/Hungarian tracking, and optional semantic labels.
- `runtime/pipeline.py`: single-frame analysis and its explanation evidence.
- `runtime/replay.py`: multi-frame tracking, depth-ray voxel evidence, change detection,
  and optional TSDF reconstruction and semantics.
- `mapping/`: log-odds evidence and a free/occupied/unknown map projected for the robot height.
- `planning/`: footprint-aware A*, failure explanations, reachable frontiers, and
  next-best-view scoring from observed evidence.
- `reconstruction/`: TSDF for scene presentation and optional gated ICP pose estimation.
- `runtime/session.py`: replay control, planning, events, binary artifacts, and map history.
- `evaluation/`: ground-truth access, metrics, ablations, and benchmark campaigns.
- `apps/api/`: FastAPI replay endpoints, SSE updates, simulated world edits, and driver.
- `apps/viewer/`: React, TypeScript, Vite, Three.js and React Three Fiber; 2D/3D inspection.
- `ros2_ws/`: ROS 2 adapter, launch/RViz configuration, and installation/verification scripts.

Typical flow: sensor frame -> ground/obstacle analysis -> tracking and ray evidence ->
2D map -> A* or viewpoint proposal -> replay session -> API -> viewer.
TSDF surfaces are not used as evidence of free space.

## Invariants

- `T_A_B` maps coordinates from B to A. Optical axes are right/down/forward;
  map and base_link have z up. See `coordinates.md`.
- Pipeline input is `sensor_export/`; only evaluation reads `ground_truth/`.
- Invalid depth is no measurement and must never clear space.
- Missing ground produces `unavailable`, not an assertion of traversability.
- Normal planning cannot cross unknown space and accounts for the robot footprint.
- Reports distinguish supplied, estimated, and unavailable poses.
- Semantics are optional votes on geometric tracks, not a replacement for geometry.
- Keep the core independent of FastAPI, the viewer, and ROS.

## Environment and commands

The existing Windows environment uses Python 3.12 and Open3D 0.19.0.
Core dependencies are in `requirements.lock`; semantics and video have separate files.

```powershell
.venv/Scripts/python -m pytest
.venv/Scripts/python -m ruff check src tests apps/api ros2_ws/src
.venv/Scripts/python -m uvicorn clearpath3d_api.main:app --app-dir apps/api --port 8010
```

In `apps/viewer`, use `npm.cmd run dev` or `npm.cmd run build` on this machine:
PowerShell currently blocks `npm.ps1`. Vite serves port 5173 and proxies `/v1` to 8010.

## Evidence and limits

Saved test-split results cover 20 synthetic rooms, with zero false-free/unsafe cells,
15/15 passable doors, and NBV beating random in 19/20 rooms. These results belong to
fingerprint `169668836189bf6b`, not automatically to the current source. Later mapping
changes are disclosed in `PROGRESS.md`; do not rerun the held-out split casually or
present historical metrics as fresh validation.

The TUM robot sequence has a saved real-data report; supplied motion-capture poses
provide its reference map. This is not geometric ground truth of the real scene.
ICP can refuse frames or lose tracking in degenerate geometry. No loop closure,
live physical sensor integration, or physical-robot deployment is implemented.
ROS 2 integration has not been verified in ROS. The API is a local development tool.
Some earlier milestone statements in `limitations.md` predate later features;
cross-check `PROGRESS.md`, saved reports, and source when describing current behavior.

## Git/publication observations

At review start, Git was initialized on `master` with no commits and no remote.
The non-ignored file inventory was 325 files totaling 3,388,658 bytes before this document.
Generated data, virtual environments, and viewer dependencies are already ignored.

Before publication, review `.claude/launch.json` (absolute local paths),
`apps/viewer/tsconfig.tsbuildinfo` (generated metadata), the missing standalone MIT
license file. The ROS maintainer email already uses the owner's GitHub noreply address;
the pending-placeholder note in `PROGRESS.md` is stale. Ruff reported five E501
line-length violations in `apps/api/clearpath3d_api/demo_video.py` during this review.

`CLAUDE.md` names `mahdi-gharbi` as the owner's GitHub account, but its warning about
the machine's Git identity should be checked against the actual local configuration.
Git commit author identity and GitHub authentication are separate settings.
Use the owner's verified or GitHub-provided noreply email for commits.

## Review validation

The viewer production build passed using `npm.cmd run build`, with a bundle-size warning.
The first full pytest invocation produced 77 passes, 8 failures and 23 setup errors;
the reported failures/errors involved access denied while clearing the existing
`.pytest_tmp` directory. The follow-up run passed all 35 selected tests in 189.39 seconds
using `--basetemp=.pytest_tmp_context_20261011 -p no:cacheprovider`. The original 31
failed/errored tests are covered by this selection, so all 108 tests passed across
the two invocations. Do not interpret the initial run as evidence of geometry bugs.
The existing cached test directories were not manually removed or modified.
