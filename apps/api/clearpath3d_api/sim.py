"""Simulator side of the interactive demo (synthetic scenes only).

The simulator owns the full world (it renders new views and can move objects on request). It lives in the
API layer on purpose: perception (``clearpath3d.runtime`` and below) never imports it and only receives the
rendered ``RGBDFrame`` objects, exactly like recorded sensor data.
"""

from __future__ import annotations

import math
import threading

import numpy as np

from clearpath3d.contracts import PoseSource, RGBDFrame
from clearpath3d.datasets.synthetic import Renderer, TrajectoryPoint, build_scene
from clearpath3d.geometry.preprocess import valid_depth_mask
from clearpath3d.planning.astar import RobotSpec
from clearpath3d.runtime.session import ReplaySession

MOVABLE_CATEGORIES = ("box", "furniture", "table")


class SimWorld:
    def __init__(self, scene: str, min_depth_m: float, max_depth_m: float):
        self.spec = build_scene(scene)
        self.renderer = Renderer(self.spec)
        self.rng = np.random.default_rng(
            self.spec.seed + 10_000
        )  # fresh noise stream for new views
        self.t = self.spec.trajectory[-1].t
        self.min_depth_m, self.max_depth_m = min_depth_m, max_depth_m
        self.n_rendered = 0
        self._lock = threading.Lock()

    def _groups(self) -> dict[str, list]:
        groups: dict[str, list] = {}
        for o in self.spec.objects:
            if o.category not in MOVABLE_CATEGORIES:
                continue
            key = o.name.split("_")[0] if o.category == "table" else o.name
            groups.setdefault(key, []).append(o)
        return groups

    def objects(self) -> list[dict]:
        out = []
        for name, parts in self._groups().items():
            lo = np.min([np.asarray(p.min_m) + p.offset_at(self.t) for p in parts], axis=0)
            hi = np.max([np.asarray(p.max_m) + p.offset_at(self.t) for p in parts], axis=0)
            out.append(
                {
                    "name": name,
                    "category": parts[0].category,
                    "center_m": ((lo + hi) / 2).round(3).tolist(),
                    "size_m": (hi - lo).round(3).tolist(),
                }
            )
        return out

    def move_object(self, name: str, x: float, y: float) -> dict:
        with self._lock:
            parts = self._groups().get(name)
            if not parts:
                raise KeyError(f"no movable object {name!r}")
            lo = np.min([np.asarray(p.min_m) for p in parts], axis=0)
            hi = np.max([np.asarray(p.max_m) for p in parts], axis=0)
            c0 = (lo + hi) / 2
            self.t += 1e-3
            for p in parts:  # new absolute offset from the original layout, effective from now on
                p.motion.append((self.t, (float(x - c0[0]), float(y - c0[1]), 0.0)))
            return {"name": name, "moved_to": [x, y], "effective_from_s": round(self.t, 3)}

    def render(self, x: float, y: float, yaw: float, dt: float = 0.2) -> RGBDFrame:
        with self._lock:
            self.t += dt
            tp = TrajectoryPoint(self.t, x, y, yaw)
            fr = self.renderer.render(tp, self.rng)
            self.n_rendered += 1
            fid = f"sim{self.n_rendered:05d}"
        d = fr.depth_noisy.astype(np.float64)
        valid = valid_depth_mask(d, self.min_depth_m, self.max_depth_m)
        return RGBDFrame(
            self.spec.name,
            fid,
            tp.t,
            fr.color,
            np.where(valid, d, 0).astype(np.float32),
            valid,
            self.renderer.intr,
            fr.T_map_camera,
            PoseSource.supplied,
            f"{self.spec.name}-cam-v1",
            T_base_camera=self.renderer.T_base_camera,
        )


def _advance(path: list[list[float]], start: np.ndarray, step_m: float) -> np.ndarray:
    """Point at arc length ``step_m`` along the polyline (or its end)."""
    pts = [np.asarray(start)] + [np.asarray(p) for p in path[1:]]
    left = step_m
    for a, b in zip(pts[:-1], pts[1:], strict=True):
        seg = float(np.linalg.norm(b - a))
        if seg >= left:
            return a + (b - a) * (left / max(seg, 1e-9))
        left -= seg
    return pts[-1]


class Driver:
    """Drives the simulated robot towards a goal, re-planning on the latest map at every step."""

    def __init__(self, session: ReplaySession, world: SimWorld):
        self.session, self.world = session, world
        self.stop_flag = threading.Event()
        self.running = False

    def run(
        self,
        goal,
        robot: RobotSpec,
        step_m: float = 0.25,
        max_steps: int = 120,
        final_yaw: float | None = None,
        turn_step_rad: float = math.radians(25),
        look_turn_rad: float = math.radians(30),
    ) -> dict:
        self.running, steps, replans, last_path, turns = True, 0, 0, None, 0
        self.stop_flag.clear()
        emit = self.session.emit
        result = {"status": "running"}
        try:
            emit("drive", {"status": "started", "goal": list(goal)})
            while steps < max_steps and not self.stop_flag.is_set():
                pose = self.session.current_pose()
                if pose is None:
                    result = {"status": "no_pose", "message": "no posed frame yet"}
                    break
                xy, yaw = pose
                if np.linalg.norm(xy - np.asarray(goal)) < 0.5 * step_m:
                    result = {"status": "reached"}
                    break
                r = self.session.plan(xy.tolist(), list(goal), robot)
                if not r["success"]:
                    result = {"status": "stopped", "cause": r["cause"], "message": r["message"]}
                    break
                nxt = _advance(r["path_m"], xy, step_m)
                if (
                    last_path is not None
                    and np.linalg.norm(nxt - _advance(last_path, xy, step_m)) > 0.1
                ):
                    replans += 1  # the new plan departs from what the previous plan intended
                last_path = r["path_m"]
                heading = float(np.arctan2(nxt[1] - xy[1], nxt[0] - xy[0]))
                # Look before moving: the camera has a ~1 m blind zone in front of the robot, so turning
                # while driving can carry the robot into space last seen long ago (a box placed there since
                # was driven through in testing). Turn in place to face the travel direction first, observe,
                # then re-plan on the updated map.
                dyaw = (heading - yaw + np.pi) % (2 * np.pi) - np.pi
                if abs(dyaw) > look_turn_rad:
                    n = max(1, int(np.ceil(abs(dyaw) / turn_step_rad)))
                    for k in range(1, n + 1):
                        a = yaw + dyaw * k / n
                        emit(
                            "drive",
                            {"status": "turning", "pose": [float(xy[0]), float(xy[1]), float(a)]},
                        )
                        self.session.ingest(
                            self.world.render(float(xy[0]), float(xy[1]), float(a)),
                            "simulator_look",
                        )
                    turns += 1
                    continue
                emit(
                    "drive",
                    {
                        "status": "moving",
                        "pose": [float(nxt[0]), float(nxt[1]), heading],
                        "path_m": r["path_m"],
                        "replans": replans,
                    },
                )
                self.session.ingest(
                    self.world.render(float(nxt[0]), float(nxt[1]), heading), "simulator_drive"
                )
                steps += 1
            else:
                result = {"status": "stopped" if self.stop_flag.is_set() else "max_steps"}
            if result.get("status") == "reached" and final_yaw is not None:
                xy, yaw = self.session.current_pose()
                diff = (final_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
                n = max(1, int(np.ceil(abs(diff) / turn_step_rad)))
                for k in range(1, n + 1):
                    a = yaw + diff * k / n
                    self.session.ingest(
                        self.world.render(float(xy[0]), float(xy[1]), float(a)), "simulator_look"
                    )
            result.update({"steps": steps, "replans": replans, "look_turns": turns})
            emit(
                "drive",
                {
                    **{k: v for k, v in result.items() if k != "status"},
                    "status": "finished",
                    "outcome": result["status"],
                },
            )
            return result
        except Exception as e:  # surface failures to the UI instead of dying silently in a thread
            emit("drive", {"status": "error", "message": f"{type(e).__name__}: {e}"})
            raise
        finally:
            self.running = False
