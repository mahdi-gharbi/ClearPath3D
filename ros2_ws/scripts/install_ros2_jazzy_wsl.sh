#!/usr/bin/env bash
# Install ROS 2 Jazzy (ros-base + RViz2 + rosbag2) on WSL2 Ubuntu 24.04, then a Python venv for ClearPath3D.
# Steps follow the official ROS 2 Jazzy docs (ros2/ros2_documentation, branch jazzy:
# _Ubuntu-Set-Locale.rst, _Apt-Repositories.rst, Ubuntu-Install-Debs.rst).
# Run it yourself inside WSL (it asks for your sudo password):
#   bash /mnt/c/Users/Lenovo/Documents/ClearPath3D/ros2_ws/scripts/install_ros2_jazzy_wsl.sh
# Disk: the WSL virtual disk lives on C: (only ~20 GB free there); expect roughly 2-3 GB.
set -euo pipefail

# 1. Locale (UTF-8)
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# 2. Repositories: universe + ros2-apt-source package
sudo apt install -y software-properties-common
sudo add-apt-repository -y universe
sudo apt update && sudo apt install -y curl
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
  | grep -F "tag_name" | awk -F'"' '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb

# 3. ROS 2 Jazzy: base, dev tools, RViz2, tf2, rosbag2, PointCloud2 helpers
sudo apt update
sudo apt install -y ros-dev-tools ros-jazzy-ros-base ros-jazzy-rviz2 ros-jazzy-tf2-ros \
  ros-jazzy-rosbag2 ros-jazzy-sensor-msgs-py python3-venv python3-pip

# 4. Python venv that can see rclpy (system site packages) + the ClearPath3D core deps
VENV="$HOME/clearpath3d-ros-venv"
python3 -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install "open3d==0.19.0" "numpy<2.3" scipy "pydantic>=2.5" typer pyyaml psutil pytest
"$VENV/bin/pip" install --no-deps -e /mnt/c/Users/Lenovo/Documents/ClearPath3D

echo
echo "Done. Check with:"
echo "  source /opt/ros/jazzy/setup.bash && ros2 doctor --report | head -20"
