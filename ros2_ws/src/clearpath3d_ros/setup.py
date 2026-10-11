from glob import glob

from setuptools import setup

package_name = "clearpath3d_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Mahdi Gharbi",
    maintainer_email="137292846+mahdi-gharbi@users.noreply.github.com",
    description="ROS 2 replay adapter for ClearPath3D",
    license="MIT",
    entry_points={"console_scripts": ["replay_node = clearpath3d_ros.replay_node:main"]},
)
