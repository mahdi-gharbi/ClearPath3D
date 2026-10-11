"""ROS 2 replay node: runs the ClearPath3D core on a recorded sequence and publishes the results.

Topics (all in frame ``map`` unless stated):
- ``/clearpath3d/points``  sensor_msgs/PointCloud2  current frame cloud (downsampled, RGB)
- ``/clearpath3d/map``     nav_msgs/OccupancyGrid   -1 unknown, 0 free, 100 occupied
- ``/clearpath3d/tracks``  visualization_msgs/MarkerArray
- ``/clearpath3d/path``    nav_msgs/Path            plan from the first pose to the ``goal`` parameter
- tf: ``map -> base_link`` (supplied poses: the ONLY authority for this transform; no ``odom`` frame is
  invented because no odometry estimator exists), static ``base_link -> camera_optical`` (calibration).

QoS: map = reliable + transient_local (late subscribers such as RViz get the latest map);
other topics = reliable, volatile, depth 5. Stamps = source timestamps of the sequence (not wall time).
"""

from __future__ import annotations

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from clearpath3d.config import load_config
from clearpath3d.datasets.base import SensorSequence
from clearpath3d.geometry.frames import compose, invert, transform_points
from clearpath3d.geometry.projection import backproject
from clearpath3d.planning.astar import RobotSpec, plan
from clearpath3d.runtime.replay import ReplayConfig, SequenceProcessor
from clearpath3d_ros.conversions import (
    FRAME_BASE,
    FRAME_CAMERA,
    FRAME_MAP,
    occupancy_grid_fields,
    path_poses,
    pointcloud2_fields,
    stamp_from_seconds,
    track_markers,
    transform_fields,
)


def _time(t: float) -> Time:
    sec, nsec = stamp_from_seconds(t)
    return Time(sec=sec, nanosec=nsec)


def _tf(parent: str, child: str, T: np.ndarray, t: float) -> TransformStamped:
    f = transform_fields(T)
    m = TransformStamped()
    m.header.stamp = _time(t)
    m.header.frame_id = parent
    m.child_frame_id = child
    m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = f[
        "translation"
    ]
    (
        m.transform.rotation.x,
        m.transform.rotation.y,
        m.transform.rotation.z,
        m.transform.rotation.w,
    ) = f["rotation"]
    return m


class ReplayNode(Node):
    def __init__(self) -> None:
        super().__init__("clearpath3d_replay")
        self.declare_parameter("sequence", "")
        self.declare_parameter("config", "")
        self.declare_parameter("rate_hz", 2.0)
        self.declare_parameter("goal", [5.0, 2.45])
        self.declare_parameter("robot_radius_m", 0.25)
        self.declare_parameter("robot_height_m", 0.40)
        self.declare_parameter("loop", False)
        seq_path = self.get_parameter("sequence").value
        if not seq_path:
            raise RuntimeError(
                "parameter 'sequence' (sequence directory with manifest.json) is required"
            )
        cfg_path = self.get_parameter("config").value or None
        self.robot = RobotSpec(
            self.get_parameter("robot_radius_m").value, self.get_parameter("robot_height_m").value
        )
        self.cfg = ReplayConfig(
            pipeline=load_config(cfg_path), robot=self.robot, tsdf_enabled=False
        )
        self.seq = SensorSequence(
            seq_path, self.cfg.pipeline.depth.min_depth_m, self.cfg.pipeline.depth.max_depth_m
        )
        self.proc = SequenceProcessor(self.cfg, self.seq.intrinsics)
        self.index = 0
        self.start_xy = None

        rel = QoSProfile(
            depth=5, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE
        )
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub_points = self.create_publisher(PointCloud2, "/clearpath3d/points", rel)
        self.pub_map = self.create_publisher(OccupancyGrid, "/clearpath3d/map", latched)
        self.pub_tracks = self.create_publisher(MarkerArray, "/clearpath3d/tracks", rel)
        self.pub_path = self.create_publisher(Path, "/clearpath3d/path", latched)
        self.tf = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        if self.seq.T_base_camera is None:
            raise RuntimeError(
                "sequence has no T_base_camera calibration; base_link cannot be published"
            )
        self.static_tf.sendTransform(_tf(FRAME_BASE, FRAME_CAMERA, self.seq.T_base_camera, 0.0))
        self.timer = self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self._tick)
        self.get_logger().info(
            f"replaying {self.seq.sequence_id}: {len(self.seq)} frames, pose_source=supplied"
        )

    def _tick(self) -> None:
        if self.index >= len(self.seq):
            if self.get_parameter("loop").value:
                self.proc = SequenceProcessor(self.cfg, self.seq.intrinsics)
                self.index = 0
            return
        frame = self.seq[self.index]
        self.index += 1
        snap = self.proc.step(frame)
        t = frame.timestamp_s
        header = Header(stamp=_time(t), frame_id=FRAME_MAP)
        if frame.T_map_camera is None:
            self.get_logger().warn(f"frame {frame.frame_id}: no pose, nothing published in map")
            return
        T_map_base = compose(frame.T_map_camera, invert(frame.T_base_camera))
        self.tf.sendTransform(_tf(FRAME_MAP, FRAME_BASE, T_map_base, t))
        if self.start_xy is None:
            self.start_xy = T_map_base[:2, 3].copy()

        keep = np.zeros_like(frame.depth_valid)
        keep[::3, ::3] = True
        pts, pix = backproject(frame.depth_m, frame.depth_valid & keep, frame.intrinsics)
        f = pointcloud2_fields(
            transform_points(frame.T_map_camera, pts), frame.color_rgb[pix[:, 0], pix[:, 1]]
        )
        pc = PointCloud2(
            header=header,
            height=f["height"],
            width=f["width"],
            is_bigendian=f["is_bigendian"],
            point_step=f["point_step"],
            row_step=f["row_step"],
            is_dense=f["is_dense"],
        )
        pc.fields = [
            PointField(name=n, offset=o, datatype=d, count=c) for n, o, d, c in f["fields"]
        ]
        pc.data = f["data"]
        self.pub_points.publish(pc)

        m = self.proc.map2d()
        if m is not None:
            g = occupancy_grid_fields(m.state, m.origin_xy, m.resolution_m, m.floor_z_m)
            og = OccupancyGrid(header=header)
            og.info.map_load_time = header.stamp
            og.info.resolution = g["resolution"]
            og.info.width, og.info.height = g["width"], g["height"]
            (og.info.origin.position.x, og.info.origin.position.y, og.info.origin.position.z) = g[
                "origin_position"
            ]
            og.info.origin.orientation.w = 1.0
            og.data = g["data"]
            self.pub_map.publish(og)
            goal = list(self.get_parameter("goal").value)
            r = plan(m, self.start_xy, goal, self.robot)
            path = Path(header=header)
            if r.success:
                for p in path_poses(r.path_m, m.floor_z_m):
                    ps = PoseStamped(header=header)
                    ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = p["position"]
                    (
                        ps.pose.orientation.x,
                        ps.pose.orientation.y,
                        ps.pose.orientation.z,
                        ps.pose.orientation.w,
                    ) = p["orientation"]
                    path.poses.append(ps)
            self.pub_path.publish(path)
            if not r.success:
                self.get_logger().info(
                    f"frame {frame.frame_id}: no path ({r.cause})", throttle_duration_sec=5.0
                )

        arr = MarkerArray()
        clear = Marker(header=header, action=Marker.DELETEALL)
        arr.markers.append(clear)
        for d in track_markers([t.model_dump() for t in snap.tracks]):
            mk = Marker(header=header, ns=d["ns"], id=d["id"], action=Marker.ADD)
            mk.type = Marker.CUBE if d["type"] == "CUBE" else Marker.TEXT_VIEW_FACING
            mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = d["position"]
            mk.pose.orientation.w = 1.0
            mk.scale.x, mk.scale.y, mk.scale.z = d["scale"]
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = d["color"]
            mk.text = d.get("text", "")
            arr.markers.append(mk)
        self.pub_tracks.publish(arr)


def main() -> None:
    rclpy.init()
    node = ReplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
