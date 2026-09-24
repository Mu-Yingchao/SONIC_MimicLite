#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

source /opt/ros/noetic/setup.bash

# --workspace pins the build to this directory, so a stray .catkin_tools in a
# parent directory (e.g. $HOME) can't hijack it.
catkin config --workspace "$ROOT_DIR" --extend /opt/ros/noetic --cmake-args -DCMAKE_BUILD_TYPE=Release
catkin build --workspace "$ROOT_DIR"
