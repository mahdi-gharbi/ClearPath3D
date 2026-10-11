"""Adapter: TUM RGB-D benchmark sequence -> ClearPath3D sequence layout.

Source conventions (TUM "file formats" page, checked 2026-10-10):
- depth PNG uint16, scaled by 5000 (z-depth, 0 = no data), already registered 1:1 to the RGB image;
- groundtruth.txt: ``timestamp tx ty tz qx qy qz qw`` = pose of the colour camera optical centre in the
  motion-capture world (i.e. ``T_world_camera``), 100 Hz;
- intrinsics: TUM recommends the ROS defaults (fx = fy = 525, cx = 319.5, cy = 239.5) without undistortion.

Done once here:
- rgb/depth association by nearest timestamp (max 20 ms); pose interpolated at the depth timestamp (linear +
  slerp) only if both bracketing ground-truth samples are within 50 ms;
- ``map`` = mocap world. Its z axis is checked to be vertical with the floor plane fitted in the data;
- ``T_base_camera`` is DERIVED (no calibration file exists): base_link = the camera's vertical projection on the
  fitted floor, x along the camera heading. Its constancy along the trajectory is measured and reported in the
  manifest (a rigid mount should give a nearly constant height/pitch/roll).
Licence: CC BY 4.0 — cite Sturm et al., "A Benchmark for the Evaluation of RGB-D SLAM Systems", IROS 2012.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation, Slerp

from clearpath3d.geometry.frames import invert, make_transform, rot_z, transform_points
from clearpath3d.geometry.projection import backproject

INTRINSICS_ROS_DEFAULT = dict(fx=525.0, fy=525.0, cx=319.5, cy=239.5, width=640, height=480)
DEPTH_SCALE = 1.0 / 5000.0


def _read_list(path: Path) -> list[tuple[float, str]]:
    out = []
    for ln in path.read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            t, f = ln.split()[:2]
            out.append((float(t), f))
    return out


def read_groundtruth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [
        ln.split() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")
    ]
    a = np.array(rows, dtype=np.float64)
    return a[:, 0], a[:, 1:]  # times, [tx ty tz qx qy qz qw]


def associate(a: list[tuple[float, str]], b: list[tuple[float, str]], max_dt: float = 0.02):
    tb = np.array([t for t, _ in b])
    out = []
    for t, f in a:
        j = int(np.argmin(np.abs(tb - t)))
        if abs(tb[j] - t) <= max_dt:
            out.append((t, f, b[j][1]))
    return out


def interpolate_pose(
    times: np.ndarray, poses: np.ndarray, t: float, max_gap: float = 0.05
) -> np.ndarray | None:
    k = int(np.searchsorted(times, t))
    if k == 0 or k >= len(times) or times[k] - times[k - 1] > max_gap:
        return None
    t0, t1 = times[k - 1], times[k]
    u = (t - t0) / (t1 - t0)
    p = (1 - u) * poses[k - 1, :3] + u * poses[k, :3]
    R = Slerp([0, 1], Rotation.from_quat(poses[k - 1 : k + 1, 3:]))([u]).as_matrix()[0]
    return make_transform(R, p)


def _floor_plane(points_world: np.ndarray, seed: int = 0) -> tuple[np.ndarray, float, int]:
    """Largest near-horizontal plane (|n_z| > cos 10 deg) among the 4 largest planes."""
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_world)).voxel_down_sample(0.03)
    o3d.utility.random.seed(seed)
    best = None
    for _ in range(4):
        if len(pc.points) < 500:
            break
        (a, b, c, d), inl = pc.segment_plane(0.02, 3, 1000)
        n = np.array([a, b, c])
        if abs(n[2]) > np.cos(np.radians(10)):
            if n[2] < 0:
                n, d = -n, -d
            if best is None or len(inl) > best[2]:
                best = (n, float(d), len(inl))
        pc = pc.select_by_index(inl, invert=True)
    if best is None:
        raise ValueError("no horizontal floor plane found: the world z axis may not be vertical")
    return best


def _base_from_camera(T_world_cam: np.ndarray, floor_z: float) -> np.ndarray:
    fwd = T_world_cam[:3, :3] @ np.array([0.0, 0.0, 1.0])
    yaw = float(np.arctan2(fwd[1], fwd[0]))
    c = T_world_cam[:3, 3]
    return make_transform(rot_z(yaw), [c[0], c[1], floor_z])


def convert(
    extract_dir: Path | str,
    out_dir: Path | str,
    stride: int = 1,
    max_frames: int | None = None,
    archive: Path | str | None = None,
) -> Path:
    src, out = Path(extract_dir), Path(out_dir)
    sensor = out / "sensor_export"
    (sensor / "frames").mkdir(parents=True, exist_ok=True)
    gt_t, gt_p = read_groundtruth(src / "groundtruth.txt")
    pairs = associate(_read_list(src / "depth.txt"), _read_list(src / "rgb.txt"))
    from clearpath3d.contracts import Intrinsics

    intr = Intrinsics(**INTRINSICS_ROS_DEFAULT)

    # 1) Floor plane in the world from the first posed frames (gravity check + derived mount calibration).
    pts = []
    for t, dpath, _ in pairs[:60:6]:
        T = interpolate_pose(gt_t, gt_p, t)
        if T is None:
            continue
        d = np.asarray(o3d.io.read_image(str(src / dpath))).astype(np.float64) * DEPTH_SCALE
        p, _ = backproject(d.astype(np.float32), (d > 0.3) & (d < 4.0), intr)
        pts.append(transform_points(T, p[::4]))
    n_floor, d_floor, inl = _floor_plane(np.concatenate(pts))
    floor_z = -d_floor / n_floor[2]
    floor_tilt_deg = float(np.degrees(np.arccos(abs(n_floor[2]))))

    frames, mounts = [], []
    used = 0
    for k, (t, dpath, cpath) in enumerate(pairs):
        if k % stride:
            continue
        T = interpolate_pose(gt_t, gt_p, t)
        if T is None:
            continue
        depth = np.asarray(o3d.io.read_image(str(src / dpath)))
        color = np.asarray(o3d.io.read_image(str(src / cpath)))
        if depth.dtype != np.uint16 or color.shape[:2] != depth.shape:
            raise ValueError(f"unexpected image format at {dpath}")
        fid = f"{used:06d}"
        np.savez_compressed(sensor / "frames" / f"{fid}.npz", color_rgb=color, depth=depth)
        T_base_cam = invert(_base_from_camera(T, floor_z)) @ T
        r = Rotation.from_matrix(T_base_cam[:3, :3]).as_euler("xyz", degrees=True)
        mounts.append([T_base_cam[2, 3], *r])
        frames.append(
            {
                "frame_id": fid,
                "timestamp_s": t,
                "file": f"frames/{fid}.npz",
                "T_map_camera": T.tolist(),
                "pose_source": "supplied",
                "source_files": [dpath, cpath],
            }
        )
        used += 1
        if max_frames and used >= max_frames:
            break
    mounts = np.array(mounts)
    T_base_camera = invert(
        _base_from_camera(np.asarray(frames[0]["T_map_camera"]), floor_z)
    ) @ np.asarray(frames[0]["T_map_camera"])
    (sensor / "frames.json").write_text(
        json.dumps({"sequence_id": src.name, "frames": frames}, indent=1)
    )
    (sensor / "calibration.json").write_text(
        json.dumps(
            {
                "calibration_id": f"{src.name}-derived-mount",
                "intrinsics": INTRINSICS_ROS_DEFAULT,
                "T_base_camera": T_base_camera.tolist(),
                "registered_color_depth": True,
            },
            indent=2,
        )
    )
    sha = None
    if archive and Path(archive).exists():
        h = hashlib.sha256()
        with open(archive, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        sha = h.hexdigest()
    manifest = {
        "dataset": "TUM RGB-D",
        "sequence": src.name,
        "archive_sha256": sha,
        "license": "CC BY 4.0 (TUM RGB-D benchmark). Cite: Sturm et al., IROS 2012. Not redistributed here.",
        "depth_convention": "z",
        "depth_unit": "1/5000 m",
        "depth_scale": DEPTH_SCALE,
        "invalid_depth_value": 0,
        "intrinsics_note": "ROS defaults as recommended by TUM; images not undistorted",
        "pose_convention": "groundtruth T_world_camera (mocap), interpolated at the depth timestamp; map = world",
        "pose_source": "supplied (motion capture ground truth)",
        "association": "depth->rgb nearest timestamp <= 20 ms; pose interpolation only within 50 ms gaps",
        "frames_converted": used,
        "stride": stride,
        "gravity_check": {
            "floor_normal_world": n_floor.round(4).tolist(),
            "floor_tilt_deg": round(floor_tilt_deg, 3),
            "floor_z_world_m": round(float(floor_z), 4),
            "floor_inliers": int(inl),
        },
        "derived_mount": {
            "camera_height_m": {
                "median": round(float(np.median(mounts[:, 0])), 4),
                "std": round(float(mounts[:, 0].std()), 4),
            },
            "roll_pitch_deg_median": np.median(mounts[:, 1:3], axis=0).round(3).tolist(),
            "roll_pitch_deg_std": mounts[:, 1:3].std(axis=0).round(3).tolist(),
            "note": "base_link derived from the first frame; spreads measure how rigid/level the mount really is",
        },
        "usage_limits": "Real data with motion-capture poses; no geometric ground truth of the scene.",
        "num_frames": used,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out
