"""Data contracts shared by every module.

Array-heavy objects (``RGBDFrame``) are validated dataclasses; JSON-serialisable outputs are
Pydantic models. Conventions: docs/coordinates.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from clearpath3d.geometry.frames import assert_rigid


class PoseSource(str, Enum):
    supplied = "supplied"
    estimated = "estimated"
    unavailable = "unavailable"


class GroundStatus(str, Enum):
    found = "found"
    unavailable = "unavailable"


class Intrinsics(BaseModel):
    model_config = ConfigDict(frozen=True)

    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float
    cy: float
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)


class FrameValidationError(ValueError):
    pass


@dataclass(frozen=True)
class RGBDFrame:
    sequence_id: str
    frame_id: str
    timestamp_s: float
    color_rgb: np.ndarray  # uint8 [H, W, 3]
    depth_m: np.ndarray  # float32 [H, W], z-depth in metres
    depth_valid: np.ndarray  # bool [H, W]
    intrinsics: Intrinsics
    T_map_camera: np.ndarray | None
    pose_source: PoseSource
    calibration_id: str
    T_base_camera: np.ndarray | None = None  # explicit mount calibration, if known
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        h, w = self.intrinsics.height, self.intrinsics.width
        if self.depth_m.shape != (h, w):
            raise FrameValidationError(
                f"depth shape {self.depth_m.shape} does not match intrinsics (H={h}, W={w})"
            )
        if self.color_rgb.shape != (h, w, 3) or self.color_rgb.dtype != np.uint8:
            raise FrameValidationError(
                f"color must be uint8 [H,W,3]=({h},{w},3), got {self.color_rgb.dtype}{self.color_rgb.shape}"
            )
        if self.depth_valid.shape != (h, w) or self.depth_valid.dtype != np.bool_:
            raise FrameValidationError("depth_valid must be bool [H,W]")
        if self.depth_m.dtype != np.float32:
            raise FrameValidationError("depth_m must be float32 metres")
        d = self.depth_m[self.depth_valid]
        if d.size and not (np.all(np.isfinite(d)) and np.all(d > 0)):
            raise FrameValidationError(
                "depth_valid marks non-finite or non-positive depth as valid"
            )
        if self.pose_source == PoseSource.unavailable:
            if self.T_map_camera is not None:
                raise FrameValidationError("pose_source=unavailable but T_map_camera is set")
        else:
            if self.T_map_camera is None:
                raise FrameValidationError(
                    f"pose_source={self.pose_source.value} requires T_map_camera"
                )
            assert_rigid(self.T_map_camera, "T_map_camera")
        if self.T_base_camera is not None:
            assert_rigid(self.T_base_camera, "T_base_camera")


# ---------------------------------------------------------------- pipeline outputs


class GroundPlane(BaseModel):
    """Plane ``normal . p + offset = 0`` in ``frame_id``; normal points up (away from the floor)."""

    frame_id: str
    normal: list[float]
    offset: float
    inlier_count: int
    residual_m: float  # RMS point-to-plane distance of inliers
    support_area_m2: float
    tilt_deg: float  # angle between normal and the gravity prior


class ObstacleDetection(BaseModel):
    """A single-frame cluster. Extent covers the *visible* points only, not the full object."""

    detection_id: int
    frame_id: str
    center_m: list[float]
    visible_extent_m: list[float]  # axis-aligned in ``frame_id``
    min_m: list[float]
    max_m: list[float]
    point_count: int
    semantic_label: str | None = None


class ObstacleTrack(BaseModel):
    track_id: int
    first_seen_s: float
    last_seen_s: float
    center_m: list[float]
    velocity_m_s: list[float]
    covariance: list[list[float]]
    visible_extent_m: list[float]
    point_count: int
    lifecycle: str  # tentative | confirmed | occluded | retired
    semantic_label: str | None = None
    evidence_frame_ids: list[str] = Field(default_factory=list)


class SceneSnapshot(BaseModel):
    sequence_id: str
    frame_id: str
    timestamp_s: float
    config_hash: str
    pose_status: PoseSource
    ground_status: GroundStatus
    output_frame: str  # frame in which geometry below is expressed (map or base_link)
    ground_plane: GroundPlane | None = None
    detections: list[ObstacleDetection] = Field(default_factory=list)
    tracks: list[ObstacleTrack] = Field(default_factory=list)
    occupancy_version: int = 0
    warnings: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    stats: dict[str, float] = Field(default_factory=dict)
