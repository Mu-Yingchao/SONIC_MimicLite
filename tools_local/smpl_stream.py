#!/usr/bin/env python3
"""SMPL 实时流：线格式 + 用 PKL 模拟 PICO 的发送端。

线格式沿用官方 SONIC 的 ZMQ pose 消息布局（gear_sonic/utils/teleop/zmq/zmq_planner_sender.py）：
``[topic][1280 字节 JSON 头][按头里字段顺序拼接的小端二进制]``，topic 为 ``smpl``。
每条消息一帧，字段与训练用 SMPL PKL 对齐：

    frame_index  i64 [1]      发送端单调递增帧号（50 Hz）
    pose_aa      f32 [72]     SMPL 轴角，前 3 维是根朝向（Y-up，与 PKL 相同）
    transl       f32 [3]      根平移（Y-up，与 PKL 相同）
    smpl_joints  f32 [24,3]   关节位置（Z-up、骨盆附近为原点，与 PKL 相同）

官方 PICO 服务（pico_manager_thread_server.py）发的 pose 消息里**没有根平移**，接 PICO 时要在
那边补一个 transl 字段（它内部已经算了骨盆位置，只是没发），并换成这里的字段名。

    # 用 PKL 按 50 Hz 模拟实时流（--loop 循环播放）
    python tools_local/smpl_stream.py send --smpl test_data/motions/smpl_pkl/walk_forward_loop_003__A022.pkl --loop
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

HEADER_SIZE = 1280  # 与官方 zmq_planner_sender.HEADER_SIZE 一致
TOPIC = b"smpl"
DEFAULT_ENDPOINT = "tcp://127.0.0.1:5560"
_DTYPES = {"f32": np.float32, "f64": np.float64, "i32": np.int32, "i64": np.int64, "bool": np.bool_}
_NAMES = {np.dtype(v): k for k, v in _DTYPES.items()}


def pack(frame_index: int, frame: dict) -> bytes:
    fields = {
        "frame_index": np.array([frame_index], dtype=np.int64),
        "pose_aa": np.asarray(frame["pose_aa"], dtype=np.float32).reshape(72),
        "transl": np.asarray(frame["transl"], dtype=np.float32).reshape(3),
        "smpl_joints": np.asarray(frame["smpl_joints"], dtype=np.float32).reshape(24, 3),
    }
    header = {
        "v": 3, "endian": "le", "count": 1,
        "fields": [{"name": k, "dtype": _NAMES[v.dtype], "shape": list(v.shape)} for k, v in fields.items()],
    }
    head = json.dumps(header, separators=(",", ":")).encode("utf-8").ljust(HEADER_SIZE, b"\x00")
    return TOPIC + head + b"".join(np.ascontiguousarray(v).astype(v.dtype.newbyteorder("<")).tobytes() for v in fields.values())


def unpack(msg: bytes) -> tuple[int, dict]:
    if not msg.startswith(TOPIC):
        raise ValueError("不是 smpl 消息")
    body = msg[len(TOPIC):]
    header = json.loads(body[:HEADER_SIZE].rstrip(b"\x00"))
    off, out = HEADER_SIZE, {}
    for f in header["fields"]:
        dt = np.dtype(_DTYPES[f["dtype"]]).newbyteorder("<")
        n = int(np.prod(f["shape"])) * dt.itemsize
        out[f["name"]] = np.frombuffer(body[off:off + n], dtype=dt).reshape(f["shape"])
        off += n
    return int(out.pop("frame_index")[0]), out


def send(args: argparse.Namespace) -> None:
    import zmq

    from sonic_offline_bridge import load_robot_motion
    from sonic_online_bridge import pkl_frames

    frames = list(pkl_frames(load_robot_motion(args.smpl)))
    sock = zmq.Context.instance().socket(zmq.PUB)
    sock.bind(args.endpoint.replace("127.0.0.1", "*"))
    print(f"发送 {args.smpl.name}：{len(frames)} 帧 @ {args.fps} Hz → {args.endpoint}"
          f"{'（循环）' if args.loop else ''}，{args.wait:.0f} 秒后开始", flush=True)
    time.sleep(args.wait)  # PUB/SUB 建连需要时间，太早发的消息会丢
    period, i = 1.0 / args.fps, 0
    t_next = time.perf_counter()
    # 循环时把水平位移接续上（Y-up 下水平是 x、z），否则每轮开头 root 会瞬移回起点
    step = np.asarray(frames[-1]["transl"], dtype=np.float64) - np.asarray(frames[0]["transl"], dtype=np.float64)
    step[1] = 0.0
    shift = np.zeros(3)
    while True:
        for fr in frames:
            sock.send(pack(i, {**fr, "transl": np.asarray(fr["transl"]) + shift}))
            i += 1
            t_next += period
            time.sleep(max(0.0, t_next - time.perf_counter()))
        if not args.loop:
            break
        shift += step
    print(f"发送完毕，共 {i} 帧", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("send", help="用 SMPL PKL 模拟实时流")
    s.add_argument("--smpl", type=Path, required=True)
    s.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    s.add_argument("--fps", type=float, default=50.0)
    s.add_argument("--loop", action="store_true", help="播完从头再来（帧号继续递增）")
    s.add_argument("--wait", type=float, default=1.0, help="绑定后等几秒再发")
    args = ap.parse_args()
    send(args)


if __name__ == "__main__":
    main()
