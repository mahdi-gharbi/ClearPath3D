"""Adapter: Open3D ``SampleRedwoodRGBDImages`` -> ClearPath3D sequence layout.

Conventions were established empirically (``tests/integration/test_redwood.py`` re-checks them):

- ``trajectory.log`` matrices are camera poses ``T_world_camera`` (frame-to-frame reprojection
  error ~6 mm median; ~146 mm if inverted).
- Depth is uint16 millimetres, z-depth (planes are flatter as z-depth than as ray distance).
- The Redwood world frame has **y up** (camera y axis ~ world -y). We rotate it once into the
  ClearPath3D ``map`` (z up): ``T_map_world = RotX(+90 deg)``. Gravity alignment is inferred from
  the camera axes and checked against the dominant horizontal plane, not guaranteed.
- No mount calibration exists (hand-held): ``T_base_camera`` is null, so ground/traversability
  analysis that needs a base frame is unavailable for this dataset.

The download is an explicit command (``clearpath3d download-redwood``); tests skip if absent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from clearpath3d.geometry.frames import compose, make_transform, rot_x

T_MAP_WORLD = make_transform(rot_x(np.pi / 2), [0.0, 0.0, 0.0])
INTRINSICS_PRIMESENSE = dict(fx=525.0, fy=525.0, cx=319.5, cy=239.5, width=640, height=480)


def download(data_root: Path | str = "data/downloads") -> Path:
    d = o3d.data.SampleRedwoodRGBDImages(data_root=str(data_root))
    return Path(d.extract_dir)


def read_log(path: Path) -> list[np.ndarray]:
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    return [
        np.array([[float(x) for x in lines[i + k].split()] for k in range(1, 5)])
        for i in range(0, len(lines), 5)
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convert(extract_dir: Path | str, out_dir: Path | str, fps: float = 30.0) -> Path:
    src, out = Path(extract_dir), Path(out_dir)
    sensor = out / "sensor_export"
    (sensor / "frames").mkdir(parents=True, exist_ok=True)
    calib = json.loads((src / "camera_primesense.json").read_text())
    K = np.asarray(calib["intrinsic_matrix"], dtype=np.float64).reshape(3, 3).T  # column-major
    intr = dict(
        fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], width=calib["width"], height=calib["height"]
    )
    poses_world = read_log(src / "trajectory.log")
    pairs = [ln.split() for ln in (src / "rgbd.match").read_text().splitlines() if ln.strip()]
    if len(pairs) != len(poses_world):
        raise ValueError(f"{len(pairs)} image pairs but {len(poses_world)} poses")

    frames = []
    for i, ((dpath, cpath), T_world_cam) in enumerate(zip(pairs, poses_world, strict=True)):
        depth_mm = np.asarray(o3d.io.read_image(str(src / dpath)))
        color = np.asarray(o3d.io.read_image(str(src / cpath)))
        if depth_mm.dtype != np.uint16 or color.shape[:2] != depth_mm.shape:
            raise ValueError(f"unexpected image format for pair {i}")
        fid = f"{i:06d}"
        # Stored as uint16 mm; the manifest's depth_scale converts to metres.
        np.savez_compressed(sensor / "frames" / f"{fid}.npz", color_rgb=color, depth=depth_mm)
        T_map_camera = compose(T_MAP_WORLD, T_world_cam)
        frames.append(
            {
                "frame_id": fid,
                "timestamp_s": i / fps,  # sample has no timestamps; nominal 30 Hz index time
                "file": f"frames/{fid}.npz",
                "T_map_camera": T_map_camera.tolist(),
                "pose_source": "supplied",
                "source_files": [dpath, cpath],
            }
        )
    (sensor / "frames.json").write_text(
        json.dumps({"sequence_id": "redwood_sample", "frames": frames}, indent=2)
    )
    (sensor / "calibration.json").write_text(
        json.dumps(
            {
                "calibration_id": "redwood-primesense-default",
                "intrinsics": intr,
                "T_base_camera": None,
                "registered_color_depth": True,
            },
            indent=2,
        )
    )
    zip_path = (
        src.parent.parent / "download" / "SampleRedwoodRGBDImages" / "SampleRedwoodRGBDImages.zip"
    )
    manifest = {
        "dataset": "redwood_sample",
        "source": "open3d.data.SampleRedwoodRGBDImages (Open3D 0.20.0): 5 frames of Redwood 'living-room1'",
        "archive_sha256": _sha256(zip_path) if zip_path.exists() else None,
        "license": (
            "NOT VERIFIED by this project. Redwood living-room1 derives from the ICL-NUIM synthetic "
            "living room; check the original terms before any redistribution. Not redistributed here."
        ),
        "depth_convention": "z",
        "depth_unit": "mm",
        "depth_scale": 0.001,
        "invalid_depth_value": 0,
        "pose_convention": (
            "trajectory.log = T_world_camera; exported as T_map_camera = T_map_world @ T_world_camera"
        ),
        "pose_source": "supplied (dataset trajectory; not estimated by ClearPath3D)",
        "transformations_applied": ["T_map_world = RotX(+90deg): Redwood world y-up -> map z-up"],
        "map_gravity_aligned": "assumed from camera axes; checked against the dominant horizontal plane",
        "timestamps": "none in the sample; index / 30 Hz",
        "usage_limits": "5 frames, ~10 cm of motion: format/convention check only, not a generalisation test.",
        "num_frames": len(frames),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out
