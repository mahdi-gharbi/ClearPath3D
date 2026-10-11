"""Launch the ClearPath3D replay node and (optionally) RViz2.

ros2 launch clearpath3d_ros replay.launch.py sequence:=/path/to/data/synthetic/room_scan rviz:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("clearpath3d_ros")
    return LaunchDescription(
        [
            DeclareLaunchArgument("sequence", description="sequence directory with manifest.json"),
            DeclareLaunchArgument("config", default_value=""),
            DeclareLaunchArgument("rate_hz", default_value="2.0"),
            DeclareLaunchArgument("rviz", default_value="true"),
            Node(
                package="clearpath3d_ros",
                executable="replay_node",
                name="clearpath3d_replay",
                output="screen",
                parameters=[
                    {
                        "sequence": LaunchConfiguration("sequence"),
                        "config": LaunchConfiguration("config"),
                        "rate_hz": LaunchConfiguration("rate_hz"),
                    }
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", os.path.join(share, "rviz", "clearpath3d.rviz")],
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
        ]
    )
