#!/usr/bin/env bash
# Real ROS 2 check of the ClearPath3D replay node (no sudo needed once ROS Jazzy is installed).
# Writes everything to ros2_ws/verify_log.txt. Run inside WSL:
#   bash /mnt/c/Users/Lenovo/Documents/ClearPath3D/ros2_ws/scripts/verify_ros.sh
set -uo pipefail
REPO=/mnt/c/Users/Lenovo/Documents/ClearPath3D
WS="$HOME/clearpath3d_ws"           # build on the Linux filesystem (faster, no Windows permissions issues)
LOG="$REPO/ros2_ws/verify_log.txt"
SEQ="$REPO/data/synthetic/room_scan"
VENV="$HOME/clearpath3d-ros-venv"
exec > >(tee "$LOG") 2>&1
set +u; source /opt/ros/jazzy/setup.bash; set -u
source "$VENV/bin/activate"
echo "== $(date -Is)  ros: $ROS_DISTRO  python: $(python3 --version)"

mkdir -p "$WS/src"
rm -rf "$WS/src/clearpath3d_ros" && cp -r "$REPO/ros2_ws/src/clearpath3d_ros" "$WS/src/"
cd "$WS" && colcon build --symlink-install 2>&1 | tail -5 || { echo "BUILD FAILED"; exit 1; }
set +u; source "$WS/install/setup.bash"; set -u

if [ ! -f "$SEQ/manifest.json" ]; then
  python3 -c "from clearpath3d.datasets.synthetic import build_scene, generate; generate(build_scene('room_scan'), '$SEQ')"
fi

ros2 launch clearpath3d_ros replay.launch.py sequence:="$SEQ" rviz:=false rate_hz:=2.0 &
LAUNCH=$!
sleep 12
echo "== topics";            ros2 topic list | grep clearpath3d
echo "== map info (once)";   timeout 10 ros2 topic echo --once /clearpath3d/map --field info
echo "== map qos";           ros2 topic info -v /clearpath3d/map | grep -E "Durability|Reliability" | sort -u
echo "== points rate";       timeout 8 ros2 topic hz /clearpath3d/points | tail -3
echo "== points header";     timeout 10 ros2 topic echo --once /clearpath3d/points --field header
echo "== tf map->base_link"; timeout 6 ros2 run tf2_ros tf2_echo map base_link 2>&1 | head -8
echo "== tf base_link->camera_optical"; timeout 6 ros2 run tf2_ros tf2_echo base_link camera_optical 2>&1 | head -8
echo "== markers";           timeout 10 ros2 topic echo --once /clearpath3d/tracks --field markers 2>&1 | grep -c "ns:" || true
echo "== path";              timeout 10 ros2 topic echo --once /clearpath3d/path --field poses 2>&1 | grep -c "position:" || true
echo "== rosbag2";           rm -rf /tmp/cp3d_bag; timeout 8 ros2 bag record -o /tmp/cp3d_bag /clearpath3d/map /clearpath3d/points /tf /tf_static; ros2 bag info /tmp/cp3d_bag
kill $LAUNCH 2>/dev/null; wait $LAUNCH 2>/dev/null
echo "== done"
