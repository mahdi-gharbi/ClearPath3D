"""Multi-view TSDF fusion with supplied (or, later, estimated) poses.

A TSDF reconstructs *surfaces*; it is not used to declare free space (that is the job of the
ray-based visibility map, mapping/). Open3D's ``integrate`` expects the extrinsic as the
world->camera transform; the only place that conversion happens is ``open3d_extrinsic``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from clearpath3d.contracts import Intrinsics, PoseSource, RGBDFrame
from clearpath3d.geometry.frames import invert


def open3d_extrinsic(T_map_camera: np.ndarray) -> np.ndarray:
    """Open3D integration extrinsic = ``T_camera_map`` (maps map points into the camera frame)."""
    return invert(T_map_camera)


def open3d_intrinsic(intr: Intrinsics) -> o3d.camera.PinholeCameraIntrinsic:
    return o3d.camera.PinholeCameraIntrinsic(
        intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy
    )


def _rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


@dataclass
class KeyframeSelector:
    """Accept a frame when it moved enough from the last accepted keyframe."""

    min_translation_m: float = 0.10
    min_rotation_deg: float = 5.0
    _last: np.ndarray | None = None

    def accept(self, T_map_camera: np.ndarray) -> bool:
        if self._last is None:
            self._last = T_map_camera
            return True
        delta = invert(self._last) @ T_map_camera
        moved = np.linalg.norm(delta[:3, 3]) >= self.min_translation_m
        turned = _rotation_angle_deg(delta[:3, :3]) >= self.min_rotation_deg
        if moved or turned:
            self._last = T_map_camera
            return True
        return False


@dataclass
class TSDFParams:
    voxel_m: float = 0.02
    sdf_trunc_m: float = 0.08
    max_depth_m: float = 6.0
    keyframe_translation_m: float = 0.10
    keyframe_rotation_deg: float = 5.0
    use_keyframes: bool = True


@dataclass
class FusionReport:
    pose_source: str
    frames_seen: int = 0
    frames_integrated: int = 0
    frames_rejected_no_pose: int = 0
    integrated_frame_ids: list[str] = field(default_factory=list)
    integrate_ms: list[float] = field(default_factory=list)
    extract_ms: float = 0.0
    mesh_vertices: int = 0
    mesh_triangles: int = 0
    cloud_points: int = 0


class TSDFFusion:
    def __init__(self, intr: Intrinsics, params: TSDFParams):
        self.intr = intr
        self.p = params
        self._o3d_intr = open3d_intrinsic(intr)
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=params.voxel_m,
            sdf_trunc=params.sdf_trunc_m,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        self.keyframes = KeyframeSelector(
            params.keyframe_translation_m, params.keyframe_rotation_deg
        )
        self.pose_sources: set[str] = set()
        self.report = FusionReport(pose_source="none")

    def integrate(self, frame: RGBDFrame, exclude_mask: np.ndarray | None = None) -> bool:
        """Integrate one frame. ``exclude_mask`` (bool [H,W]) removes pixels, e.g. moving tracks."""
        self.report.frames_seen += 1
        if frame.T_map_camera is None or frame.pose_source == PoseSource.unavailable:
            self.report.frames_rejected_no_pose += 1
            return False
        if self.p.use_keyframes and not self.keyframes.accept(frame.T_map_camera):
            return False
        t0 = time.perf_counter()
        depth = np.where(frame.depth_valid, frame.depth_m, 0.0).astype(np.float32)
        if exclude_mask is not None:
            depth[exclude_mask] = 0.0
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(frame.color_rgb)),
            o3d.geometry.Image(depth),
            depth_scale=1.0,  # depth is already metres
            depth_trunc=self.p.max_depth_m,
            convert_rgb_to_intensity=False,
        )
        self.volume.integrate(rgbd, self._o3d_intr, open3d_extrinsic(frame.T_map_camera))
        self.report.integrate_ms.append(round((time.perf_counter() - t0) * 1000, 3))
        self.report.frames_integrated += 1
        self.report.integrated_frame_ids.append(frame.frame_id)
        self.pose_sources.add(frame.pose_source.value)
        self.report.pose_source = "+".join(sorted(self.pose_sources))
        return True

    def extract(self) -> tuple[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud]:
        t0 = time.perf_counter()
        mesh = self.volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        cloud = self.volume.extract_point_cloud()
        self.report.extract_ms = round((time.perf_counter() - t0) * 1000, 3)
        self.report.mesh_vertices = len(mesh.vertices)
        self.report.mesh_triangles = len(mesh.triangles)
        self.report.cloud_points = len(cloud.points)
        return mesh, cloud
