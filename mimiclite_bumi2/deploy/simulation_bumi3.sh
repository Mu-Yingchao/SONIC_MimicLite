#!/usr/bin/env bash

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export ROBOT_TYPE=bumi3

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT_DIR/devel/setup.bash"
roslaunch rl_controllers_bumi3 ac_start.launch
