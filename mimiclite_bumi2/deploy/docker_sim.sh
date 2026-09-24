#!/usr/bin/env bash
# ============================================================
#  容器外启动脚本：进入 deploy_bumi_v2 开发/仿真容器
#  用法： bash docker_sim.sh
# ============================================================
set -e

IMAGE_NAME=ros_0723_realsense:latest
WS_HOST=/home/user/simulation/deploy_bumi_v2
WS_CONT=/ws
CONTAINER_NAME=deploy_bumi_v2

# 1) 允许容器访问宿主机的 X11 显示（Gazebo/RViz 要用）
xhost +local:root || true

# Qt/rqt 在容器里需要 runtime dir，否则容易 segfault
RUNTIME_DIR=/tmp/runtime-root

# 2) 若容器已存在：启动并进入；否则新建
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo ">> 容器 ${CONTAINER_NAME} 已存在，启动并进入..."
    docker start "${CONTAINER_NAME}" >/dev/null
    docker exec "${CONTAINER_NAME}" bash -lc "mkdir -p ${RUNTIME_DIR} && chmod 700 ${RUNTIME_DIR}" >/dev/null
    exec docker exec -it \
        -e DISPLAY="${DISPLAY}" \
        -e QT_X11_NO_MITSHM=1 \
        -e XDG_RUNTIME_DIR="${RUNTIME_DIR}" \
        -w "${WS_CONT}" \
        "${CONTAINER_NAME}" bash
fi

echo ">> 创建容器 ${CONTAINER_NAME}，挂载 ${WS_HOST} -> ${WS_CONT}"
docker run -it \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --env DISPLAY="${DISPLAY}" \
    --env QT_X11_NO_MITSHM=1 \
    --env XDG_RUNTIME_DIR="${RUNTIME_DIR}" \
    --env NVIDIA_DRIVER_CAPABILITIES=all \
    --privileged \
    --network host \
    --security-opt label=disable \
    -v /dev:/dev \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --mount type=bind,source="${WS_HOST}",target="${WS_CONT}" \
    -w "${WS_CONT}" \
    "${IMAGE_NAME}" bash
