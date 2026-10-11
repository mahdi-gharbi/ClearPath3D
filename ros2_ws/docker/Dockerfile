# ROS 2 Jazzy environment for the ClearPath3D replay node — no sudo needed on the host (Docker Desktop).
# Base: Docker official image ros:jazzy-ros-base (Ubuntu 24.04, Python 3.12).
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-rviz2 ros-jazzy-rosbag2 ros-jazzy-rosbag2-storage-mcap ros-jazzy-sensor-msgs-py \
        ros-jazzy-tf2-ros ros-jazzy-tf2-tools python3-pip python3-colcon-common-extensions \
        xvfb x11-apps imagemagick libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ClearPath3D core dependencies (same pins as requirements.lock where they matter).
RUN pip3 install --break-system-packages --no-cache-dir \
        "open3d==0.19.0" "numpy<2.3" scipy "pydantic>=2.5" typer pyyaml psutil

WORKDIR /ws
