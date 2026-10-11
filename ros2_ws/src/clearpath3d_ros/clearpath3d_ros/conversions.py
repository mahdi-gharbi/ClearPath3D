"""Pure conversions from ClearPath3D outputs to ROS 2 message *field values*.

No rclpy import here: these functions are unit-tested on any OS. The node fills real messages from
their output. ROS conventions (REP-103/105): metres, radians, right-handed, z up in ``map``;
quaternions are (x, y, z, w).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

FRAME_MAP = "map"
FRAME_BASE = "base_link"
FRAME_CAMERA = "camera_optical"


def stamp_from_seconds(t: float) -> tuple[int, int]:
    """Source timestamp (s) -> (sec, nanosec), kept unchanged (no re-basing)."""
    if t < 0:
        raise ValueError("negative timestamps are not representable in builtin_interfaces/Time")
    sec = int(np.floor(t))
    nsec = int(round((t - sec) * 1e9))
    if nsec >= 1_000_000_000:
        sec, nsec = sec + 1, nsec - 1_000_000_000
    return sec, nsec


def transform_fields(T_parent_child: np.ndarray) -> dict:
    """``T_parent_child`` (maps child points into parent) -> geometry_msgs/Transform fields.

    In tf2, the transform published with header.frame_id=parent and child_frame_id=child is exactly
    this matrix: the pose of the child frame expressed in the parent frame.
    """
    q = Rotation.from_matrix(T_parent_child[:3, :3]).as_quat()  # x, y, z, w
    t = T_parent_child[:3, 3]
    return {"translation": [float(v) for v in t], "rotation": [float(v) for v in q]}


def occupancy_grid_fields(
    state: np.ndarray, origin_xy, resolution_m: float, floor_z_m: float
) -> dict:
    """ClearPath3D map ``state[i=x, j=y]`` (-1/0/100) -> nav_msgs/OccupancyGrid fields.

    OccupancyGrid data is row-major with the x index fastest: ``data[y * width + x]``; cell (0, 0)
    is at ``info.origin``. Unknown stays -1, free 0, occupied 100.
    """
    st = np.asarray(state, dtype=np.int8)
    width, height = st.shape
    vals = set(np.unique(st).tolist())
    if not vals <= {-1, 0, 100}:
        raise ValueError(f"unexpected cell values {vals}")
    return {
        "width": int(width),
        "height": int(height),
        "resolution": float(resolution_m),
        "origin_position": [float(origin_xy[0]), float(origin_xy[1]), float(floor_z_m)],
        "origin_orientation": [0.0, 0.0, 0.0, 1.0],
        "data": st.T.ravel().tolist(),  # [y][x] -> row-major, x fastest
    }


def pointcloud2_fields(xyz: np.ndarray, rgb: np.ndarray | None = None) -> dict:
    """Points (N,3) float32 [+ (N,3) uint8 colours] -> sensor_msgs/PointCloud2 fields.

    Layout: x, y, z float32 (offsets 0, 4, 8) and, if coloured, ``rgb`` as a float32 field holding the
    packed 0x00RRGGBB uint32 (the RViz2 convention), offset 12. Little-endian, dense.
    """
    xyz = np.ascontiguousarray(xyz, dtype=np.float32).reshape(-1, 3)
    n = len(xyz)
    FLOAT32 = 7  # sensor_msgs/PointField.FLOAT32
    fields = [("x", 0, FLOAT32, 1), ("y", 4, FLOAT32, 1), ("z", 8, FLOAT32, 1)]
    if rgb is None:
        buf = xyz.tobytes()
        step = 12
    else:
        rgb = np.asarray(rgb, dtype=np.uint32).reshape(-1, 3)
        packed = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
        rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])
        rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        rec["rgb"] = packed
        buf = rec.tobytes()
        fields.append(("rgb", 12, FLOAT32, 1))
        step = 16
    return {
        "height": 1,
        "width": n,
        "fields": fields,
        "is_bigendian": False,
        "point_step": step,
        "row_step": step * n,
        "data": buf,
        "is_dense": True,
    }


def track_markers(tracks: list[dict], ns: str = "tracks") -> list[dict]:
    """Obstacle tracks -> visualization_msgs/Marker fields (CUBE for the visible extent + TEXT id)."""
    out = []
    colors = {
        "confirmed": (0.91, 0.35, 0.05, 0.5),
        "tentative": (0.6, 0.6, 0.6, 0.35),
        "occluded": (0.5, 0.5, 0.8, 0.35),
    }
    for t in tracks:
        c = colors.get(t["lifecycle"], (0.5, 0.5, 0.5, 0.3))
        ext = [max(float(v), 0.02) for v in t["visible_extent_m"]]
        out.append(
            {
                "ns": ns,
                "id": int(t["track_id"]) * 2,
                "type": "CUBE",
                "position": t["center_m"],
                "scale": ext,
                "color": c,
            }
        )
        out.append(
            {
                "ns": ns,
                "id": int(t["track_id"]) * 2 + 1,
                "type": "TEXT_VIEW_FACING",
                "position": [
                    t["center_m"][0],
                    t["center_m"][1],
                    t["center_m"][2] + ext[2] / 2 + 0.1,
                ],
                "scale": [0.0, 0.0, 0.12],
                "color": (1.0, 1.0, 1.0, 1.0),
                "text": f"#{t['track_id']} {t['lifecycle']}",
            }
        )
    return out


def path_poses(path_m: list[list[float]], z: float) -> list[dict]:
    """Planner polyline -> geometry_msgs/PoseStamped poses, heading along each segment."""
    poses = []
    P = np.asarray(path_m, dtype=np.float64)
    for k, p in enumerate(P):
        d = P[min(k + 1, len(P) - 1)] - P[max(k - 1, 0)] if len(P) > 1 else np.array([1.0, 0.0])
        yaw = float(np.arctan2(d[1], d[0]))
        q = Rotation.from_euler("z", yaw).as_quat()
        poses.append(
            {"position": [float(p[0]), float(p[1]), float(z)], "orientation": [float(v) for v in q]}
        )
    return poses
