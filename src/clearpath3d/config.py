"""Pipeline configuration (YAML -> Pydantic). Values are starting hypotheses, not validated results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DepthConfig(_Strict):
    min_depth_m: float = 0.2
    max_depth_m: float = 8.0


class PreprocessConfig(_Strict):
    voxel_m: float = 0.03
    outlier_removal: bool = True
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    normals_radius_factor: float = 3.0  # radius = factor * voxel_m


class GroundConfig(_Strict):
    distance_threshold_m: float = 0.03
    max_tilt_deg: float = 12.0
    height_tolerance_m: float = 0.10
    min_inliers: int = 200
    min_support_area_m2: float = 0.25
    area_cell_m: float = 0.05
    max_candidates: int = 4
    ransac_iterations: int = 400
    seed: int = 0


class ObstacleConfig(_Strict):
    min_height_m: float = 0.04  # above the ground plane
    max_height_m: float = 2.0
    max_range_m: float = 6.0  # horizontal distance from the camera
    dbscan_eps_m: float = 0.08
    dbscan_min_points: int = 10
    min_cluster_points: int = 15


class PipelineConfig(_Strict):
    depth: DepthConfig = DepthConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    ground: GroundConfig = GroundConfig()
    obstacles: ObstacleConfig = ObstacleConfig()

    def config_hash(self) -> str:
        blob = json.dumps(self.model_dump(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def load_config(path: Path | str | None) -> PipelineConfig:
    if path is None:
        return PipelineConfig()
    data = yaml.safe_load(Path(path).read_text()) or {}
    return PipelineConfig(**data)
