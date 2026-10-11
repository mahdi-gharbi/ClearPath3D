"""Sequence processor (M3): per frame, analyse -> track -> visibility evidence -> changes -> TSDF.

Reads only ``RGBDFrame`` objects (sensor data, calibration, supplied/estimated poses).
The static TSDF is for presentation; planning uses the recent ray-based evidence map.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from clearpath3d.config import PipelineConfig
from clearpath3d.contracts import RGBDFrame, SceneSnapshot
from clearpath3d.geometry.frames import compose, invert, transform_points
from clearpath3d.geometry.projection import backproject
from clearpath3d.mapping.changes import ChangeDetector, ChangeEvent
from clearpath3d.mapping.projection import Map2D, ProjectionParams, project_to_2d
from clearpath3d.mapping.voxel_evidence import EvidenceParams, VoxelEvidenceGrid
from clearpath3d.perception.semantics import SemanticFusion, lift
from clearpath3d.perception.tracking import Tracker, TrackerParams
from clearpath3d.planning.astar import RobotSpec
from clearpath3d.reconstruction.tsdf import TSDFFusion, TSDFParams
from clearpath3d.runtime.pipeline import FrameAnalysis, analyze_frame

BORDER_FRACTION = 0.025  # representative pixels are sparse: a few pixels of tolerance


def detection_truncation(fa: FrameAnalysis, width: int, height: int) -> list[str]:
    out = []
    b = max(2, round(BORDER_FRACTION * height))
    for d in fa.snapshot.detections:
        pix = fa.pixel[fa.cluster == d.detection_id]
        if len(pix) and pix[:, 0].min() <= b:
            out.append("top")
        elif len(pix) and (pix[:, 1].min() <= b or pix[:, 1].max() >= width - 1 - b):
            out.append("side")
        else:
            out.append("")
    return out


@dataclass
class ReplayConfig:
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    evidence: EvidenceParams = field(default_factory=EvidenceParams)
    projection: ProjectionParams = field(default_factory=ProjectionParams)
    tracker: TrackerParams = field(default_factory=TrackerParams)
    tsdf: TSDFParams = field(default_factory=TSDFParams)
    robot: RobotSpec = field(default_factory=RobotSpec)
    mark_robot_footprint: bool = True
    dynamic_displacement_m: float | None = None  # visible-centre shifts make this unreliable (M3)
    # Change events need confident states (>= 2 net hits / >= 3 net free observations), >= 10 voxels,
    # and are only reported in the robot band above the floor (surface voxels flicker).
    change_l_occ: float = 1.5
    change_l_free: float = -1.1
    change_min_voxels: int = 10
    tsdf_enabled: bool = True
    # Optional semantics (perception/semantics.py): any object with ``detect(frame) -> list[Detection2D]``.
    semantic_detector: object | None = None
    semantic_lift_mode: str = "mask"


class SequenceProcessor:
    def __init__(self, cfg: ReplayConfig, intrinsics):
        self.cfg = cfg
        self.evidence = VoxelEvidenceGrid(cfg.evidence)
        self.change_detector = ChangeDetector(
            len(self.evidence.log_odds),
            self.evidence.shape,
            self.evidence.min,
            cfg.evidence.resolution_m,
            cfg.change_l_occ,
            cfg.change_l_free,
        )
        self.tracker = Tracker(cfg.tracker)
        self.semantics = SemanticFusion() if cfg.semantic_detector is not None else None
        self.tsdf = TSDFFusion(intrinsics, cfg.tsdf) if cfg.tsdf_enabled else None
        self.floor_heights: list[float] = []
        self.changes: list[ChangeEvent] = []
        self.dynamic_track_ids: set[int] = set()
        self.snapshots: list[SceneSnapshot] = []
        self.frame_count = 0
        self.last_frame: RGBDFrame | None = None
        self._prev_T_map_base: np.ndarray | None = None

    @property
    def floor_z(self) -> float | None:
        return float(np.median(self.floor_heights)) if self.floor_heights else None

    def _dynamic_mask(self, frame: RGBDFrame) -> np.ndarray | None:
        boxes = [
            (np.asarray(t.x[:3]) - t.extent / 2 - 0.05, np.asarray(t.x[:3]) + t.extent / 2 + 0.05)
            for t in self.tracker.tracks
            if t.track_id in self.dynamic_track_ids
        ]
        if not boxes or frame.T_map_camera is None:
            return None
        pts, pix = backproject(frame.depth_m, frame.depth_valid, frame.intrinsics)
        pm = transform_points(frame.T_map_camera, pts)
        inside = np.zeros(len(pm), dtype=bool)
        for lo, hi in boxes:
            inside |= np.all((pm >= lo) & (pm <= hi), axis=1)
        mask = np.zeros(frame.depth_m.shape, dtype=bool)
        mask[pix[inside, 0], pix[inside, 1]] = True
        return mask

    def step(self, frame: RGBDFrame) -> SceneSnapshot:
        timings = {}
        t0 = time.perf_counter()
        fa = analyze_frame(frame, self.cfg.pipeline)
        snap = fa.snapshot
        timings.update({f"analyze.{k}": v for k, v in snap.timings_ms.items()})

        T_map_base = None
        if frame.T_map_camera is not None and frame.T_base_camera is not None:
            T_map_base = compose(frame.T_map_camera, invert(frame.T_base_camera))
        if snap.ground_plane is not None and snap.output_frame == "map" and T_map_base is not None:
            n, d = np.asarray(snap.ground_plane.normal), snap.ground_plane.offset
            x, y = T_map_base[:2, 3]
            self.floor_heights.append(float(-(n[0] * x + n[1] * y + d) / n[2]))

        t1 = time.perf_counter()
        trunc = detection_truncation(fa, frame.intrinsics.width, frame.intrinsics.height)
        # Without a map pose, detections cannot be associated in map: none are passed (tracks coast).
        in_map = snap.output_frame == "map"
        assoc = self.tracker.step(frame, snap.detections if in_map else [], trunc if in_map else [])
        for t in self.tracker.tracks:
            dd = self.cfg.dynamic_displacement_m
            if dd is not None and t.lifecycle != "tentative" and t.max_displacement_m >= dd:
                self.dynamic_track_ids.add(t.track_id)
        timings["tracking"] = round((time.perf_counter() - t1) * 1000, 3)

        t1 = time.perf_counter()
        if frame.T_map_camera is not None:
            if self.cfg.mark_robot_footprint and T_map_base is not None:
                self.evidence.mark_robot_footprint(
                    T_map_base,
                    self.cfg.robot.radius_m,
                    self.cfg.robot.height_m,
                    frame.timestamp_s,
                    frame.frame_id,
                    self._prev_T_map_base,
                )
                self._prev_T_map_base = T_map_base
            ev = self.evidence.integrate_frame(frame)
            band = None
            if self.floor_z is not None:
                zc = (
                    self.evidence.min[2]
                    + (np.arange(self.evidence.shape[2]) + 0.5) * self.evidence.p.resolution_m
                )
                in_band = (zc - self.floor_z >= self.cfg.projection.clearance_m) & (
                    zc - self.floor_z <= self.cfg.robot.height_m + 0.5
                )
                band = np.broadcast_to(in_band, self.evidence.shape).ravel()
            new_events = self.change_detector.update(
                self.evidence.log_odds,
                frame.frame_id,
                frame.timestamp_s,
                self.cfg.change_min_voxels,
                band,
            )
            self.changes.extend(new_events)
            # A track standing where free space had been observed has moved in: exclude it from the TSDF.
            for e in new_events:
                if e.kind != "appeared":
                    continue
                for t in self.tracker.tracks:
                    lo, hi = t.x[:3] - t.extent / 2 - 0.1, t.x[:3] + t.extent / 2 + 0.1
                    if np.all(np.asarray(e.max_m) >= lo) and np.all(np.asarray(e.min_m) <= hi):
                        self.dynamic_track_ids.add(t.track_id)
        else:
            ev = {"rays": 0}
            snap.warnings.append("no pose: frame not added to the map")
        timings["evidence"] = round((time.perf_counter() - t1) * 1000, 3)

        t1 = time.perf_counter()
        if self.tsdf is not None:
            self.tsdf.integrate(frame, exclude_mask=self._dynamic_mask(frame))
        timings["tsdf"] = round((time.perf_counter() - t1) * 1000, 3)
        timings["total"] = round((time.perf_counter() - t0) * 1000, 3)

        snap.tracks = self.tracker.snapshot()
        if self.semantics is not None:
            t1 = time.perf_counter()
            if frame.T_map_camera is not None:
                lifted = [
                    L
                    for d in self.cfg.semantic_detector.detect(frame)
                    if (L := lift(frame, d, self.cfg.semantic_lift_mode)) is not None
                ]
                self.semantics.update(self.tracker.tracks, lifted)
            for tr in snap.tracks:
                tr.semantic_label = self.semantics.label_of(tr.track_id)
            timings["semantics"] = round((time.perf_counter() - t1) * 1000, 3)
        snap.occupancy_version = self.evidence.version
        snap.timings_ms = timings
        snap.stats.update({f"assoc.{k}": float(v) for k, v in assoc.items()})
        snap.stats["rays"] = float(ev["rays"])
        self.snapshots.append(snap)
        self.frame_count += 1
        self.last_frame = frame
        return snap

    def map2d(self, now_s: float | None = None) -> Map2D | None:
        """None when no floor has been established: no traversability can be asserted."""
        if self.floor_z is None:
            return None
        p = self.cfg.projection
        p.robot_height_m = self.cfg.robot.height_m
        return project_to_2d(self.evidence, self.floor_z, p, now_s)
