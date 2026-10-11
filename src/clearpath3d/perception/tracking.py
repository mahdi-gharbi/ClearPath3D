"""Multi-object tracking of obstacle clusters (geometric, no identity guarantee).

State ``[x, y, z, vx, vy, vz]`` with a constant-velocity Kalman filter. Association: gating first
(Mahalanobis distance under a chi-square bound AND a Euclidean max distance), then the Hungarian
algorithm on the gated cost matrix.

Lifecycle: ``tentative`` -> ``confirmed`` after ``confirm_hits`` associations; a confirmed track that
is not detected becomes ``occluded`` (or stays confirmed with a miss) and is ``retired`` after
``retire_after_s`` without a detection. A track missed while its predicted centre was *visible*
(in the image and not hidden by nearer depth) accumulates ``visible_misses`` and retires sooner.
Disappearance is not proof that an object left the room. ``occluded`` means "not currently
observed" (hidden or out of the field of view). While unobserved, the velocity is damped each frame so
a noisy velocity estimate cannot carry the prediction away.

Detections whose cluster touches the top image border are treated as structure (walls): their
visible centre moves with the viewpoint, so they are not tracked.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from clearpath3d.contracts import ObstacleDetection, ObstacleTrack, RGBDFrame
from clearpath3d.geometry.frames import invert, transform_points
from clearpath3d.geometry.projection import project

CHI2_3DOF_99 = 11.345


@dataclass
class TrackerParams:
    process_accel_std: float = 0.5  # m/s^2, white-acceleration model
    meas_std_m: float = 0.05
    truncated_meas_scale: float = 3.0  # inflate R for clusters cut by the image side borders
    gate_chi2: float = CHI2_3DOF_99
    gate_max_dist_m: float = 0.6
    confirm_hits: int = 3
    tentative_max_misses: int = 2
    retire_after_s: float = 4.0
    max_visible_misses: int = 3
    occlusion_margin_m: float = 0.15
    max_object_extent_m: float = 1.6
    miss_velocity_damping: float = (
        0.5  # v *= damping per missed frame (limits drift while unobserved)
    )


@dataclass
class Track:
    track_id: int
    x: np.ndarray  # 6
    P: np.ndarray  # 6x6
    first_seen_s: float
    last_seen_s: float
    last_update_s: float
    extent: np.ndarray
    point_count: int
    hits: int = 1
    misses: int = 0
    visible_misses: int = 0
    lifecycle: str = "tentative"
    evidence: list[str] = field(default_factory=list)
    first_center: np.ndarray | None = None
    max_displacement_m: float = 0.0

    def to_contract(self) -> ObstacleTrack:
        return ObstacleTrack(
            track_id=self.track_id,
            first_seen_s=self.first_seen_s,
            last_seen_s=self.last_seen_s,
            center_m=self.x[:3].round(4).tolist(),
            velocity_m_s=self.x[3:].round(4).tolist(),
            covariance=self.P[:3, :3].round(6).tolist(),
            visible_extent_m=self.extent.round(4).tolist(),
            point_count=self.point_count,
            lifecycle=self.lifecycle,
            evidence_frame_ids=self.evidence[-20:],
        )


def detection_is_structure(det: ObstacleDetection, touches_top: bool, p: TrackerParams) -> bool:
    return (
        touches_top or max(det.visible_extent_m[0], det.visible_extent_m[1]) > p.max_object_extent_m
    )


class Tracker:
    def __init__(self, p: TrackerParams | None = None):
        self.p = p or TrackerParams()
        self.tracks: list[Track] = []
        self.retired: list[Track] = []
        self._next_id = 0
        self.last_t: float | None = None

    # --------------------------------------------------------------- Kalman
    def _F_Q(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        q = self.p.process_accel_std**2
        Q = np.zeros((6, 6))
        Q[:3, :3] = np.eye(3) * dt**4 / 4 * q
        Q[:3, 3:] = Q[3:, :3] = np.eye(3) * dt**3 / 2 * q
        Q[3:, 3:] = np.eye(3) * dt**2 * q
        return F, Q

    def _predict(self, dt: float) -> None:
        if dt <= 0:
            return
        F, Q = self._F_Q(dt)
        for t in self.tracks:
            t.x = F @ t.x
            t.P = F @ t.P @ F.T + Q

    @staticmethod
    def _H() -> np.ndarray:
        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)
        return H

    def _update(self, t: Track, z: np.ndarray, R: np.ndarray) -> None:
        H = self._H()
        S = H @ t.P @ H.T + R
        K = t.P @ H.T @ np.linalg.inv(S)
        t.x = t.x + K @ (z - H @ t.x)
        t.P = (np.eye(6) - K @ H) @ t.P

    # --------------------------------------------------------------- visibility
    def _visible(self, center: np.ndarray, frame: RGBDFrame) -> bool:
        if frame.T_map_camera is None:
            return False
        pc = transform_points(invert(frame.T_map_camera), center[None])[0]
        uv, front = project(pc[None], frame.intrinsics)
        if not front[0]:
            return False
        u, v = int(round(uv[0, 0])), int(round(uv[0, 1]))
        if not (0 <= u < frame.intrinsics.width and 0 <= v < frame.intrinsics.height):
            return False
        if not frame.depth_valid[v, u]:
            return False  # no measurement: cannot claim it was visible
        return frame.depth_m[v, u] >= pc[2] - self.p.occlusion_margin_m

    # --------------------------------------------------------------- step
    def step(
        self,
        frame: RGBDFrame,
        detections: list[ObstacleDetection],
        truncated: list[str],
    ) -> dict:
        """``truncated[i]`` in {'', 'side', 'top'} for each detection. Returns association info."""
        p = self.p
        t_now = frame.timestamp_s
        dt = 0.0 if self.last_t is None else t_now - self.last_t
        self.last_t = t_now
        self._predict(dt)

        dets = [
            (d, tr)
            for d, tr in zip(detections, truncated, strict=True)
            if not detection_is_structure(d, tr == "top", p)
        ]
        Z = np.array([d.center_m for d, _ in dets]).reshape(-1, 3)
        H = self._H()
        n_t, n_d = len(self.tracks), len(dets)
        cost = np.full((n_t, n_d), 1e6)
        for i, t in enumerate(self.tracks):
            for j, (_d, tr) in enumerate(dets):
                R = np.eye(3) * (p.meas_std_m * (p.truncated_meas_scale if tr else 1.0)) ** 2
                S = H @ t.P @ H.T + R
                r = Z[j] - t.x[:3]
                m2 = float(r @ np.linalg.solve(S, r))
                if m2 <= p.gate_chi2 and np.linalg.norm(r) <= p.gate_max_dist_m:
                    cost[i, j] = m2
        matches = []
        if n_t and n_d:
            rows, cols = linear_sum_assignment(cost)
            matches = [(i, j) for i, j in zip(rows, cols, strict=True) if cost[i, j] < 1e6]
        matched_t = {i for i, _ in matches}
        matched_d = {j for _, j in matches}

        for i, j in matches:
            t, (d, tr) = self.tracks[i], dets[j]
            R = np.eye(3) * (p.meas_std_m * (p.truncated_meas_scale if tr else 1.0)) ** 2
            self._update(t, Z[j], R)
            t.hits += 1
            t.misses = 0
            t.visible_misses = 0
            t.last_seen_s = t_now
            t.extent = np.asarray(d.visible_extent_m)
            t.point_count = d.point_count
            t.evidence.append(frame.frame_id)
            t.max_displacement_m = max(
                t.max_displacement_m, float(np.linalg.norm(t.x[:3] - t.first_center))
            )
            if t.lifecycle in ("tentative", "occluded") and t.hits >= p.confirm_hits:
                t.lifecycle = "confirmed"
            elif t.lifecycle == "occluded":
                t.lifecycle = "confirmed"

        for i, t in enumerate(self.tracks):
            if i in matched_t:
                continue
            t.misses += 1
            t.x[3:] *= p.miss_velocity_damping
            if self._visible(t.x[:3], frame):
                t.visible_misses += 1
            elif t.lifecycle == "confirmed":
                t.lifecycle = "occluded"

        for j, (d, _) in enumerate(dets):
            if j in matched_d:
                continue
            c = np.asarray(d.center_m, dtype=np.float64)
            P0 = np.diag([p.meas_std_m**2] * 3 + [1.0] * 3)
            self.tracks.append(
                Track(
                    self._next_id,
                    np.concatenate([c, np.zeros(3)]),
                    P0,
                    t_now,
                    t_now,
                    t_now,
                    np.asarray(d.visible_extent_m),
                    d.point_count,
                    evidence=[frame.frame_id],
                    first_center=c,
                )
            )
            self._next_id += 1

        keep = []
        for t in self.tracks:
            retire = (
                (t.lifecycle == "tentative" and t.misses > p.tentative_max_misses)
                or t_now - t.last_seen_s > p.retire_after_s
                or t.visible_misses >= p.max_visible_misses
            )
            if retire:
                t.lifecycle = "retired"
                self.retired.append(t)
            else:
                keep.append(t)
        self.tracks = keep
        return {
            "detections_tracked": n_d,
            "detections_structure": len(detections) - n_d,
            "matches": len(matches),
            "new_tracks": n_d - len(matched_d),
            "active": len(self.tracks),
        }

    def snapshot(self) -> list[ObstacleTrack]:
        return [t.to_contract() for t in self.tracks]
