"""Sensor-side sequence loader. Reads ``manifest.json`` and ``sensor_export/`` only.

This module must never open ``ground_truth/`` (enforced by tests/regression/test_gt_leakage.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from clearpath3d.contracts import Intrinsics, PoseSource, RGBDFrame
from clearpath3d.geometry.preprocess import check_depth_plausibility, valid_depth_mask


class SensorSequence:
    def __init__(
        self,
        root: Path | str,
        min_depth_m: float = 0.2,
        max_depth_m: float = 8.0,
        use_supplied_poses: bool = True,
    ):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        sensor = self.root / "sensor_export"
        self._sensor = sensor
        calib = json.loads((sensor / "calibration.json").read_text())
        self.intrinsics = Intrinsics(**calib["intrinsics"])
        tbc = calib.get("T_base_camera")
        self.T_base_camera = None if tbc is None else np.asarray(tbc, dtype=np.float64)
        self.calibration_id = calib["calibration_id"]
        meta = json.loads((sensor / "frames.json").read_text())
        self.sequence_id = meta["sequence_id"]
        self.frames_meta = meta["frames"]
        self.depth_scale = float(self.manifest["depth_scale"])
        if self.manifest.get("depth_convention") != "z":
            raise ValueError(
                "only z-depth sequences are supported; convert ray distance in the adapter"
            )
        self.min_depth_m, self.max_depth_m = min_depth_m, max_depth_m
        self.use_supplied_poses = use_supplied_poses

    def __len__(self) -> int:
        return len(self.frames_meta)

    def __getitem__(self, i: int) -> RGBDFrame:
        m = self.frames_meta[i]
        with np.load(self._sensor / m["file"]) as z:
            color = z["color_rgb"]
            depth = z["depth"].astype(np.float64) * self.depth_scale
        check_depth_plausibility(depth)
        valid = valid_depth_mask(depth, self.min_depth_m, self.max_depth_m)
        depth_m = np.where(valid, depth, 0.0).astype(np.float32)
        if self.use_supplied_poses and m.get("pose_source") == "supplied":
            T, src = np.asarray(m["T_map_camera"], dtype=np.float64), PoseSource.supplied
        else:
            T, src = None, PoseSource.unavailable
        return RGBDFrame(
            sequence_id=self.sequence_id,
            frame_id=m["frame_id"],
            timestamp_s=float(m["timestamp_s"]),
            color_rgb=color,
            depth_m=depth_m,
            depth_valid=valid,
            intrinsics=self.intrinsics,
            T_map_camera=T,
            pose_source=src,
            calibration_id=self.calibration_id,
            T_base_camera=self.T_base_camera,
        )
