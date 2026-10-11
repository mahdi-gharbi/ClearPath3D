"""Replay session: one sequence, one processing worker, deterministic reset, versioned outputs.

Framework-free (no FastAPI import) so it can be driven by the API, the CLI, tests or a ROS node.
All processing runs on a single worker thread; readers get immutable copies/encoded bytes.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from clearpath3d.contracts import RGBDFrame
from clearpath3d.datasets.base import SensorSequence
from clearpath3d.geometry.frames import compose, invert
from clearpath3d.mapping.projection import Map2D, explain_cell
from clearpath3d.planning.astar import RobotSpec, blocked_grid, plan
from clearpath3d.runtime.replay import ReplayConfig, SequenceProcessor

DISPLAY_STRIDE = 3  # display cloud keeps 1 pixel in 3 along each axis


@dataclass
class Event:
    seq: int
    kind: str  # step | reset | play | pause | end | error
    data: dict = field(default_factory=dict)


class ReplaySession:
    def __init__(
        self,
        replay_id: str,
        root: Path | str,
        cfg: ReplayConfig,
        max_events: int = 1000,
        pose_mode: str = "supplied",
    ):
        """``pose_mode``: ``supplied`` (dataset poses) or ``icp`` (first pose supplied as anchor, all others
        estimated by ICP odometry; rejected frames carry no pose)."""
        if pose_mode not in ("supplied", "icp"):
            raise ValueError(f"unknown pose_mode {pose_mode!r}")
        self.replay_id = replay_id
        self.root = Path(root)
        self.cfg = cfg
        self.pose_mode = pose_mode
        d = cfg.pipeline.depth
        self.seq = SensorSequence(self.root, d.min_depth_m, d.max_depth_m)
        self._raw = SensorSequence(
            self.root, d.min_depth_m, d.max_depth_m, use_supplied_poses=False
        )
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"replay-{replay_id}")
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._events: list[Event] = []
        self._event_seq = 0
        self._max_events = max_events
        self._playing = False
        self._play_fps = 2.0
        self._cache: dict[tuple, object] = {}
        self._reset_state()

    # ------------------------------------------------------------------ state
    def _reset_state(self) -> None:
        self.proc = SequenceProcessor(self.cfg, self.seq.intrinsics)
        self.odometry = None
        if self.pose_mode == "icp":
            from clearpath3d.reconstruction.registration import ICPOdometry

            self.odometry = ICPOdometry(
                self.seq.T_base_camera, T_map_camera_0=self.seq[0].T_map_camera
            )
        self.index = 0
        self.processed = 0  # frames processed from any origin (replay or simulator drive)
        self.history: list[dict] = []  # per processed frame: map state snapshot for the timeline
        self.last_frame: RGBDFrame | None = None
        self.last_points: tuple[np.ndarray, np.ndarray] | None = None
        self._cache.clear()

    def _emit(self, kind: str, data: dict | None = None) -> None:
        with self._cond:
            self._event_seq += 1
            self._events.append(Event(self._event_seq, kind, data or {}))
            self._events = self._events[-self._max_events :]
            self._cond.notify_all()

    def emit(self, kind: str, data: dict | None = None) -> None:
        """Publish an event from outside the session (e.g. the simulator driver)."""
        self._emit(kind, data)

    def events_after(self, seq: int, timeout_s: float = 15.0) -> list[Event]:
        """Block until events newer than ``seq`` exist (or timeout); returns them."""
        with self._cond:
            self._cond.wait_for(lambda: self._event_seq > seq, timeout=timeout_s)
            return [e for e in self._events if e.seq > seq]

    @property
    def finished(self) -> bool:
        return self.index >= len(self.seq)

    # ------------------------------------------------------------------ processing (worker thread)
    def _process(self, frame: RGBDFrame, origin: str) -> dict:
        """Process one frame (caller holds the lock). ``origin``: replay | simulator_drive | simulator_look."""
        from clearpath3d.geometry.frames import transform_points
        from clearpath3d.geometry.projection import backproject

        snap = self.proc.step(frame)
        self.last_frame = frame
        step = self.processed
        self.processed += 1
        # Downsampled current-frame cloud in map (display only).
        keep = np.zeros_like(frame.depth_valid)
        keep[::DISPLAY_STRIDE, ::DISPLAY_STRIDE] = True
        valid = frame.depth_valid & keep
        pts, pix = backproject(frame.depth_m, valid, frame.intrinsics)
        if frame.T_map_camera is not None:
            pts = transform_points(frame.T_map_camera, pts)
        self.last_points = (
            pts.astype(np.float32),
            frame.color_rgb[pix[:, 0], pix[:, 1]].astype(np.uint8),
        )
        m = self.map2d()
        self.history.append(
            {
                "step": step,
                "frame_id": frame.frame_id,
                "origin": origin,
                "timestamp_s": frame.timestamp_s,
                "version": None if m is None else m.version,
                "state": None if m is None else m.state.copy(),
                "counts": None if m is None else m.counts(),
            }
        )
        return {
            "step": step,
            "origin": origin,
            "frame_index": self.index - 1 if origin == "replay" else None,
            "frame_id": snap.frame_id,
            "timestamp_s": snap.timestamp_s,
            "occupancy_version": snap.occupancy_version,
            "pose_source": snap.pose_status.value,
            "odometry": None if self.odometry is None else self.odometry.log[-1],
            "ground_status": snap.ground_status.value,
            "tracks": len(snap.tracks),
            "changes_total": len(self.proc.changes),
            "warnings": snap.warnings,
            "timings_ms": {
                k: v for k, v in snap.timings_ms.items() if not k.startswith("analyze.")
            },
        }

    def _step_one(self) -> dict | None:
        with self._lock:
            if self.finished:
                return None
            if self.odometry is None:
                frame = self.seq[self.index]
            else:
                frame = self.odometry.process(self._raw[self.index])
            self.index += 1
            data = self._process(frame, "replay")
        self._emit("step", data)
        return data

    def ingest(self, frame: RGBDFrame, origin: str = "simulator_drive") -> dict:
        """Process an external frame (e.g. rendered by a simulator after a simulated move). Worker thread."""

        def run():
            with self._lock:
                data = self._process(frame, origin)
            self._emit("step", data)
            return data

        return self._worker.submit(run).result()

    def current_pose(self) -> tuple[np.ndarray, float] | None:
        """(xy, yaw) of the robot base at the last processed frame with a pose."""
        f = self.last_frame
        if f is None or f.T_map_camera is None or f.T_base_camera is None:
            return None
        T = compose(f.T_map_camera, invert(f.T_base_camera))
        return T[:2, 3].copy(), float(np.arctan2(T[1, 0], T[0, 0]))

    def timeline(self) -> list[dict]:
        return [{k: v for k, v in h.items() if k != "state"} for h in self.history]

    def map_bytes_at(self, step: int) -> tuple[bytes, dict] | None:
        with self._lock:
            if not 0 <= step < len(self.history) or self.history[step]["state"] is None:
                return None
            st = self.history[step]["state"]
            m = self.map2d()
        nx, ny = st.shape
        return np.ascontiguousarray(st.astype(np.int8)).tobytes(), {
            "X-Nx": str(nx),
            "X-Ny": str(ny),
            "X-Origin-X": f"{m.origin_xy[0]:.6f}",
            "X-Origin-Y": f"{m.origin_xy[1]:.6f}",
            "X-Resolution": f"{m.resolution_m:.6f}",
            "X-Step": str(step),
            "X-Layout": "int8 row-major [nx][ny]; 100 occ, 0 free, -1 unknown",
        }

    def map_diff(self, a: int, b: int) -> dict:
        """Cells whose state changed between timeline steps a and b."""
        with self._lock:
            if not (0 <= a < len(self.history) and 0 <= b < len(self.history)):
                raise KeyError("step out of range")
            sa, sb = self.history[a]["state"], self.history[b]["state"]
        if sa is None or sb is None:
            raise KeyError("no map at one of the steps")
        names = {100: "occupied", 0: "free", -1: "unknown"}
        out = {}
        for x in (100, 0, -1):
            for y in (100, 0, -1):
                if x != y:
                    n = int(((sa == x) & (sb == y)).sum())
                    if n:
                        out[f"{names[x]}->{names[y]}"] = n
        return {"from_step": a, "to_step": b, "transitions": out}

    def step(self, n: int = 1) -> list[dict]:
        def run():
            out = []
            for _ in range(n):
                d = self._step_one()
                if d is None:
                    self._emit("end", {"frames": len(self.seq)})
                    break
                out.append(d)
            return out

        return self._worker.submit(run).result()

    def play(self, fps: float = 2.0) -> None:
        self._play_fps = max(0.1, fps)
        if self._playing:
            return
        self._playing = True
        self._emit("play", {"fps": self._play_fps})

        def loop():
            while self._playing:
                t0 = time.perf_counter()
                if self._step_one() is None:
                    self._playing = False
                    self._emit("end", {"frames": len(self.seq)})
                    break
                time.sleep(max(0.0, 1.0 / self._play_fps - (time.perf_counter() - t0)))

        self._worker.submit(loop)

    def pause(self) -> None:
        self._playing = False
        self._emit("pause", {"frame_index": self.index})

    def reset(self) -> None:
        self._playing = False

        def run():
            with self._lock:
                self._reset_state()

        self._worker.submit(run).result()
        self._emit("reset", {})

    def close(self) -> None:
        self._playing = False
        self._worker.shutdown(wait=True)

    # ------------------------------------------------------------------ read side
    def map2d(self) -> Map2D | None:
        with self._lock:
            key = ("map2d", self.proc.evidence.version)
            if key not in self._cache:
                self._cache = {k: v for k, v in self._cache.items() if k[0] != "map2d"}
                self._cache[key] = self.proc.map2d()
            return self._cache[key]  # type: ignore[return-value]

    def snapshot(self) -> dict:
        with self._lock:
            snap = self.proc.snapshots[-1].model_dump(mode="json") if self.proc.snapshots else None
            m = self.map2d()
            T_map_base = None
            f = self.last_frame
            if f is not None and f.T_map_camera is not None and f.T_base_camera is not None:
                T_map_base = compose(f.T_map_camera, invert(f.T_base_camera)).tolist()
            return {
                "replay_id": self.replay_id,
                "sequence_id": self.seq.sequence_id,
                "frame_index": self.index,
                "num_frames": len(self.seq),
                "playing": self._playing,
                "finished": self.finished,
                "snapshot": snap,
                "T_map_base": T_map_base,
                "T_map_camera": None
                if f is None or f.T_map_camera is None
                else f.T_map_camera.tolist(),
                "map": None
                if m is None
                else {
                    "counts": m.counts(),
                    "version": m.version,
                    "floor_z_m": m.floor_z_m,
                    "resolution_m": m.resolution_m,
                    "criterion": m.params.__dict__,
                },
                "map_unavailable_reason": None if m is not None else "no floor established yet",
                "robot": asdict(self.cfg.robot),
                "pose_mode": self.pose_mode,
                "odometry_lost": None if self.odometry is None else self.odometry.lost,
                "changes": [c.to_dict() for c in self.proc.changes[-50:]],
            }

    def map_bytes(self) -> tuple[bytes, dict] | None:
        m = self.map2d()
        if m is None:
            return None
        nx, ny = m.state.shape
        headers = {
            "X-Nx": str(nx),
            "X-Ny": str(ny),
            "X-Origin-X": f"{m.origin_xy[0]:.6f}",
            "X-Origin-Y": f"{m.origin_xy[1]:.6f}",
            "X-Resolution": f"{m.resolution_m:.6f}",
            "X-Version": str(m.version),
            "X-Layout": "int8 row-major [nx][ny]; 100 occ, 0 free, -1 unknown",
        }
        return np.ascontiguousarray(m.state.astype(np.int8)).tobytes(), headers

    def points_bytes(self) -> tuple[bytes, dict] | None:
        with self._lock:
            if self.last_points is None:
                return None
            pts, col = self.last_points
        headers = {
            "X-Count": str(len(pts)),
            "X-Frame": "map"
            if self.last_frame and self.last_frame.T_map_camera is not None
            else "camera_optical",
            "X-Layout": "float32 xyz * N, then uint8 rgb * N",
        }
        return pts.tobytes() + col.tobytes(), headers

    def occupied_voxels_bytes(self) -> tuple[bytes, dict]:
        with self._lock:
            g = self.proc.evidence
            key = ("voxels", g.version)
            if key not in self._cache:
                lin = np.nonzero(g.log_odds >= self.cfg.projection.l_occ)[0]
                self._cache[key] = g.centers(lin).astype(np.float32)
            c = self._cache[key]
        return c.tobytes(), {
            "X-Count": str(len(c)),
            "X-Voxel-Size": str(g.p.resolution_m),
            "X-Version": str(g.version),
            "X-Layout": "float32 xyz centres * N (map)",
        }

    def plan(self, start, goal, robot: RobotSpec | None = None) -> dict:
        robot = robot or self.cfg.robot
        m = self.map2d()
        if m is not None and abs(robot.height_m - self.cfg.robot.height_m) > 1e-9:
            # The free/occupied band depends on the robot height: re-project for this robot.
            from dataclasses import replace

            from clearpath3d.mapping.projection import project_to_2d

            with self._lock:
                p = replace(self.cfg.projection, robot_height_m=robot.height_m)
                m = project_to_2d(self.proc.evidence, m.floor_z_m, p)
        if m is None:
            return {
                "success": False,
                "cause": "no_map",
                "message": "no floor established: no traversability",
            }
        r = plan(m, start, goal, robot)
        d = asdict(r)
        d.pop("path_cells")
        d["robot"] = asdict(robot)
        d["note"] = "geometric simulation in observed free space; not a safety guarantee"
        return d

    def blocked_bytes(self, robot: RobotSpec | None = None) -> tuple[bytes, dict] | None:
        m = self.map2d()
        if m is None:
            return None
        b = blocked_grid(m.state, m.resolution_m, robot or self.cfg.robot)
        return np.packbits(b, axis=None).tobytes(), {
            "X-Nx": str(b.shape[0]),
            "X-Ny": str(b.shape[1]),
        }

    def viewpoints(self, top_k: int = 10) -> dict:
        from clearpath3d.planning.next_best_view import propose_viewpoint

        with self._lock:
            f = self.last_frame
            m = self.map2d()
            if f is None or f.T_map_camera is None or f.T_base_camera is None:
                return {
                    "status": "no_pose",
                    "message": "no posed frame processed yet",
                    "best": None,
                    "top": [],
                }
            T_map_base = compose(f.T_map_camera, invert(f.T_base_camera))
            yaw = float(np.arctan2(T_map_base[1, 0], T_map_base[0, 0]))
            r = propose_viewpoint(
                self.proc.evidence,
                m,
                T_map_base[:2, 3],
                yaw,
                self.cfg.robot,
                f.T_base_camera,
                f.intrinsics,
                self.cfg.projection.l_occ,
                self.cfg.projection.l_free,
            )
        top = sorted(r.candidates, key=lambda c: c.score, reverse=True)[:top_k]
        return {
            "status": r.status,
            "message": r.message,
            "best": None if r.best is None else asdict(r.best),
            "top": [asdict(c) for c in top],
            "candidates": len(r.candidates),
            "frontier_cells": r.frontier_cells,
            "params": r.params,
            "map_version": None if m is None else m.version,
            "note": "gain predicted from the observed map only (first never-observed voxel per ray)",
        }

    def evidence(self, evidence_id: str) -> dict:
        kind, _, rest = evidence_id.partition(":")
        with self._lock:
            if kind == "cell":
                m = self.map2d()
                if m is None:
                    raise KeyError("no map yet")
                i, j = (int(v) for v in rest.split(":"))
                if not (0 <= i < m.state.shape[0] and 0 <= j < m.state.shape[1]):
                    raise KeyError("cell outside map")
                return explain_cell(self.proc.evidence, m, (i, j))
            if kind == "point":
                m = self.map2d()
                if m is None:
                    raise KeyError("no map yet")
                x, y = (float(v) for v in rest.split(":"))
                i, j = (int(v) for v in m.cell_of(np.array([x, y])))
                return self.evidence(f"cell:{i}:{j}")
            if kind == "track":
                tid = int(rest)
                for t in self.proc.tracker.tracks + self.proc.tracker.retired:
                    if t.track_id == tid:
                        c = t.to_contract().model_dump()
                        c["dynamic"] = tid in self.proc.dynamic_track_ids
                        c["max_displacement_m"] = round(t.max_displacement_m, 4)
                        c["note"] = (
                            "visible_extent covers observed points only, not the full object"
                        )
                        return c
                raise KeyError(f"track {tid} not found")
            if kind == "change":
                k = int(rest)
                if not 0 <= k < len(self.proc.changes):
                    raise KeyError(f"change {k} not found")
                c = self.proc.changes[k].to_dict()
                g = self.proc.evidence
                lo, hi = np.asarray(c["min_m"]), np.asarray(c["max_m"])
                ctr = g.centers(np.arange(len(g.log_odds)))
                inside = np.all((ctr >= lo) & (ctr <= hi), axis=1)
                frames = sorted({g.frame_ids[f] for f in g.last_frame[inside] if f >= 0})
                c["voxels_now"] = {
                    "occupied": int((g.log_odds[inside] >= self.cfg.projection.l_occ).sum()),
                    "free": int((g.log_odds[inside] <= self.cfg.projection.l_free).sum()),
                }
                c["last_frames_in_region"] = frames[-10:]
                c["tracks_flagged_dynamic"] = sorted(self.proc.dynamic_track_ids)
                return c
        raise KeyError(
            f"unknown evidence kind {kind!r}; use cell:i:j, point:x:y, track:id or change:k"
        )
