#!/usr/bin/env bash
# Real ROS 2 Jazzy check of the ClearPath3D replay node, inside the clearpath3d-ros:jazzy container.
# Host (Windows, Docker Desktop, no sudo):
#   docker run --rm -v "C:/Users/Lenovo/Documents/ClearPath3D:/ws/ClearPath3D" clearpath3d-ros:jazzy \
#       bash /ws/ClearPath3D/ros2_ws/scripts/verify_ros_docker.sh
# Output: ros2_ws/verify_log.txt, ros2_ws/rviz_screenshot.png, ros2_ws/verify_bag_info.txt
set -uo pipefail
REPO=/ws/ClearPath3D
LOG="$REPO/ros2_ws/verify_log.txt"
SEQ="$REPO/data/synthetic/room_scan"
exec > >(tee "$LOG") 2>&1
set +u; source /opt/ros/jazzy/setup.bash; set -u
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
echo "== $(date -Is)  ROS_DISTRO=$ROS_DISTRO  $(python3 --version)  open3d $(python3 -c 'import open3d; print(open3d.__version__)')"

mkdir -p /tmp/ws/src && cp -r "$REPO/ros2_ws/src/clearpath3d_ros" /tmp/ws/src/
cd /tmp/ws && colcon build 2>&1 | tail -4 || { echo "BUILD FAILED"; exit 1; }
set +u; source /tmp/ws/install/setup.bash; set -u

if [ ! -f "$SEQ/manifest.json" ]; then
  python3 -c "from clearpath3d.datasets.synthetic import build_scene, generate; generate(build_scene('room_scan'), '$SEQ')"
fi

ros2 launch clearpath3d_ros replay.launch.py sequence:="$SEQ" rviz:=false rate_hz:=1.0 > /tmp/launch.log 2>&1 &
LAUNCH=$!
sleep 25
echo "== launch log (head)"; head -15 /tmp/launch.log
echo "== topics";                ros2 topic list | sort
echo "== map info (once)";       timeout 20 ros2 topic echo --once /clearpath3d/map --field info
echo "== map qos";               ros2 topic info -v /clearpath3d/map | grep -E "Reliability|Durability" | sort | uniq -c
echo "== points rate";           timeout 12 ros2 topic hz /clearpath3d/points 2>&1 | tail -4
echo "== points header+fields";  timeout 20 ros2 topic echo --once /clearpath3d/points --field header
                                 timeout 20 ros2 topic echo --once /clearpath3d/points --field fields | grep -E "name|offset" | head -8
echo "== tf map->base_link";     timeout 8 ros2 run tf2_ros tf2_echo map base_link 2>&1 | head -8
echo "== tf base_link->camera_optical"; timeout 8 ros2 run tf2_ros tf2_echo base_link camera_optical 2>&1 | head -8
echo "== markers count";         timeout 20 ros2 topic echo --once /clearpath3d/tracks --field markers 2>/dev/null | grep -c "ns:"
echo "== path poses count";      timeout 20 ros2 topic echo --once /clearpath3d/path --field poses 2>/dev/null | grep -c "position:"
echo "== rosbag2 record (10 s)"
rm -rf /tmp/cp3d_bag; timeout 10 ros2 bag record -o /tmp/cp3d_bag /clearpath3d/map /clearpath3d/points /clearpath3d/tracks /clearpath3d/path /tf /tf_static > /dev/null 2>&1
ros2 bag info /tmp/cp3d_bag | tee "$REPO/ros2_ws/verify_bag_info.txt"

echo "== RViz2 screenshot (Xvfb, software rendering)"
export DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1
Xvfb :99 -screen 0 1280x800x24 > /dev/null 2>&1 &
XV=$!
sleep 2
rviz2 -d /tmp/ws/install/clearpath3d_ros/share/clearpath3d_ros/rviz/clearpath3d.rviz > /tmp/rviz.log 2>&1 &
RV=$!
sleep 25
import -window root "$REPO/ros2_ws/rviz_screenshot.png" && echo "screenshot saved" || echo "screenshot FAILED"
tail -5 /tmp/rviz.log
kill $RV $XV $LAUNCH 2>/dev/null; wait 2>/dev/null
echo "== node log (tail)"; tail -8 /tmp/launch.log
echo "== done"
