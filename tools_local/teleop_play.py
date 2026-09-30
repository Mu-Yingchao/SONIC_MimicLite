#!/usr/bin/env python3
"""SMPL 遥操的仿真端：收 SMPL 实时流 → 在线 SONIC 桥接 → MimicLite 在 mjlab 里实时跟踪。

不改同事的 play.py，而是在外面包一层：
  1. 让 play 加载一条很长的"占位动作"（站立姿态，默认 30 分钟），作为实时参考的存储；
  2. 每个控制步之前（挂在 play_pre_step 上）：收 ZMQ 上的 SMPL 帧 → OnlineSmplBridge 出 qpos →
     按 any4hdmi 同一套算法（前向差分 qvel + mj_forward）算出 FK → 写进占位动作对应的行；
     还没到的未来帧用最后一帧顶上（hold）；
  3. 播放游标跟随实时流：正常情况下 command.t 落后最新可用帧固定 LAG 帧（MimicLite 要看参考的
     未来 4 帧，再留 2 帧余量），落后太多就跳到最新；reset（摔倒终止等）时也从当前实时帧开始。

画面、切换、viser 等与普通 play 完全相同。只跟一个机器人：task.num_envs=1。

    # 终端 1：模拟 PICO，按 50 Hz 发 SMPL
    python tools_local/smpl_stream.py send --smpl test_data/motions/smpl_pkl/walk_forward_loop_003__A022.pkl --loop
    # 终端 2：仿真端（参数与 play.py 相同，另加本脚本的 --onnx 等）
    python tools_local/teleop_play.py --onnx test_data/policies/sonic/recon_v1_step_010000_smpl.onnx -- \\
      headless=false checkpoint_path=/home/yingchaomu/sonic_deploy/policies/mimiclite/checkpoint_40000.pt
"""

from __future__ import annotations

import argparse
import builtins
import json
import runpy
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from smpl_stream import DEFAULT_ENDPOINT, PICO_ENDPOINT, PICO_TOPIC, TOPIC, pico_frames, unpack  # noqa: E402
from sonic_online_bridge import OnlineSmplBridge  # noqa: E402

REPO = HERE.parent
# Hydra 解析不了中文路径：数据和 play.py 都走 ASCII 软链（见 play_guide.md）
DEPLOY = Path.home() / "sonic_deploy"
TRAINING = Path.home() / "sonic_aa"
DATASET = DEPLOY / "motions/any4hdmi-bumi-v2"
PLACEHOLDER_DIR = DATASET / "motions/live_stream"
MAX_FUTURE = 4       # MimicLite future_steps 的最大值（short: [-8..4]）
LAG = MAX_FUTURE + 2  # 播放游标落后最新可用帧的帧数
MAX_LAG = LAG + 25    # 落后超过这个就跳到最新（仿真跑不满实时等情况）
HOLD = 16            # 最新帧之后用 hold 填多少行
START_FRAMES = 20    # 攒够多少帧再开始
WARMUP_STEPS = 150   # 开头先原地站立跑这么多步：首次运行要编译 CUDA kernel，前几秒跑不满实时
OFFSET = 1000        # 占位动作前 OFFSET 行留给热身站立，实时帧 i 写在第 OFFSET+i 行
STALL_S = 0.3        # 超过这么久没有新帧，就认为流断了：参考平滑回到站立
TO_STAND = 100       # 回站立用多少帧（2 s；1 s 时从走路中途急停实测会摔）
RESUME_BLEND = 25    # 流恢复时从当前参考过渡到操作员姿态用多少帧（0.5 s）


def ensure_placeholder(frames: int) -> Path:
    """占位动作：站立姿态重复 frames 帧。第一次运行时生成，之后复用（FK 缓存也会复用）。"""
    path = PLACEHOLDER_DIR / "live_stream_from_g1_bumi_v2.npz"
    if path.exists() and len(np.load(path)["qpos"]) == frames:
        return PLACEHOLDER_DIR
    stand = np.load(DATASET / "motions/original_qpos/idle_loop_001__A021_from_g1_bumi_v2.npz")["qpos"][0].copy()
    stand[:2] = 0.0
    PLACEHOLDER_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, qpos=np.repeat(stand[None], frames, axis=0).astype(np.float32))
    print(f"[teleop] 生成占位动作 {path}（{frames} 帧）", flush=True)
    return PLACEHOLDER_DIR


def _qvel(model: mujoco.MjModel, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """与 any4hdmi.utils.dataset.compute_motion_qvel 同一算法的单帧版本（前向差分）。"""
    from any4hdmi.utils.dataset import compute_motion_qvel

    return compute_motion_qvel(model, np.stack([q0, q1]), 50.0)[0]


def _blend(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    """qpos 插值：位置与关节线性，root 四元数球面插值。w=0 取 a，w=1 取 b。"""
    out = (1 - w) * a + w * b
    rots = Rotation.from_quat(np.stack([a[[4, 5, 6, 3]], b[[4, 5, 6, 3]]]))
    out[3:7] = Slerp([0.0, 1.0], rots)(w).as_quat()[[3, 0, 1, 2]]
    return out


class LiveFeeder:
    def __init__(self, args: argparse.Namespace):
        import zmq

        self.bridge = OnlineSmplBridge(args.onnx, DATASET / "manifest.json", root_solve=not args.no_mink)
        man = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
        self.model = mujoco.MjModel.from_xml_path(str(DATASET / man["mjcf"]))
        self.mdata = mujoco.MjData(self.model)
        self.sock = zmq.Context.instance().socket(zmq.SUB)
        self.sock.setsockopt(zmq.RCVHWM, 10000)
        self.source = args.source
        self.sock.connect(args.endpoint)
        self.endpoint = args.endpoint
        self.sock.setsockopt(zmq.SUBSCRIBE, PICO_TOPIC if args.source == "pico" else TOPIC)
        self.zmq = zmq
        self.qpos: list[np.ndarray] = []
        self.written = -1          # 已写入真实数据的最后一帧（写第 i 帧需要第 i+1 帧来算速度）
        self.first_index = None    # 发送端第一帧的帧号（之前的发送时还没连上，丢了）
        self.expect = None
        self.last_frame = None
        self.dropped = 0
        self.received = 0
        self.jumps = 0
        self.attached = False
        self.started = False
        self.steps = 0
        self.last_rx = time.perf_counter()
        self.stalled = False
        self.resume_pending = False
        self.resume_left = 0
        self.stand = None
        self.t_report = time.perf_counter()

    # ---------- 收流 ----------
    def poll(self) -> None:
        while True:
            try:
                msg = self.sock.recv(self.zmq.NOBLOCK)
            except self.zmq.Again:
                return
            # PICO 每条消息带最近 N 帧（滑动窗口），smpl 格式每条一帧
            items = pico_frames(msg) if self.source == "pico" else [unpack(msg)]
            if self.expect is not None and items[-1][0] < self.expect - 1:
                self.expect = items[0][0]  # 整条都比已处理的旧：发送端重启，当作新的一段
            for idx, frame in items:
                if self.expect is None:
                    self.first_index = self.expect = idx
                if idx < self.expect:  # 窗口里已经处理过的帧
                    continue
                self.received += 1
                while idx > self.expect:  # 网络丢帧：用上一帧补齐，保持 50 Hz 时间轴
                    self._push(self.last_frame)
                    self.dropped += 1
                self._push(frame)
                self.last_frame = frame

    def _push(self, frame: dict) -> None:
        self.expect += 1
        if self.stalled and not self.resume_pending:
            # 断流后的新一段：在线桥接从头开始，并以机器人当前参考位置为起点
            # （操作员此间可能走动过，或发送端重新开始，SMPL 位置不再连续）
            self.bridge.reset(origin_xy=self.qpos[-1][:2])
            self.resume_pending = True
        self.last_rx = time.perf_counter()
        for _, q in self.bridge.push(frame):
            q = q.astype(np.float64)
            if self.stalled:  # 断流后恢复：从当前参考（站立）平滑过渡回操作员姿态
                self.stalled = False
                self.resume_pending = False
                self.resume_left = RESUME_BLEND
            if self.resume_left > 0 and self.qpos:
                w = 1.0 - self.resume_left / (RESUME_BLEND + 1)
                q = _blend(self.qpos[-1], q, w)
                self.resume_left -= 1
            self.qpos.append(q)

    def _feed_stall(self) -> None:
        """断流时每个控制步追加一帧合成参考：从最后姿态逐步混向站立（保持位置与朝向），之后原地站着。"""
        last = self.qpos[-1]
        if not self.stalled:
            self.stalled = True
            print(f"[teleop] SMPL 流中断超过 {STALL_S}s，参考平滑回到站立", flush=True)
            yaw = float(Rotation.from_quat(last[[4, 5, 6, 3]]).as_euler("ZYX")[0])
            self.stand = self.stand_pose.copy()
            self.stand[:2] = last[:2]
            self.stand[3:7] = Rotation.from_euler("Z", yaw).as_quat()[[3, 0, 1, 2]]
        self.qpos.append(_blend(last, self.stand, 1.0 / TO_STAND))

    def _heading_check(self) -> str:
        """坐标约定自检：最近 1 s 在走动时，行进方向相对身体朝向的夹角。
        向前走应接近 0°；接近 180° 说明根平移前后反了，接近 ±90° 说明平移与朝向的坐标轴没对齐。"""
        if len(self.qpos) < 51 or self.stalled:
            return ""
        a, b = self.qpos[-51], self.qpos[-1]
        d = b[:2] - a[:2]
        if np.linalg.norm(d) < 0.3:  # 1 s 内走不到 0.3 m：没在走，不判断
            return ""
        yaw = Rotation.from_quat(b[[4, 5, 6, 3]]).as_euler("ZYX")[0]
        ang = np.rad2deg(np.arctan2(d[1], d[0]) - yaw)
        return f"  行进方向-朝向 {(ang + 180) % 360 - 180:+.0f}°"

    # ---------- 写入占位动作 ----------
    def attach(self, env) -> None:
        base = getattr(env, "base_env", env)
        cmd = base.command_manager
        ds = cmd.dataset
        full = ds.datasets[0] if hasattr(ds, "datasets") else ds
        self.cmd, self.ds, self.full = cmd, ds, full
        self.row0 = int(full.starts[0])
        self.capacity = int(full.lengths[0])
        m = self.model
        self.body_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in full.body_names]
        self.jq = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in full.joint_names]
        self.jd = [int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in full.joint_names]
        if min(self.body_ids) < 0:
            raise RuntimeError("数据集刚体名在 MJCF 里找不到")
        # 自检：用本脚本的 FK 重算占位动作第 0 帧，必须与数据集缓存一致
        stand = np.load(PLACEHOLDER_DIR / "live_stream_from_g1_bumi_v2.npz")["qpos"][0].astype(np.float64)
        row = self._fk(stand, np.zeros(m.nv))
        # 数据集按 float16 存储，先按存储精度取整再比，映射正确时应完全相等
        got = self.full.data[self.row0]
        err = max(
            float((getattr(got, k).cpu().float() - row[k].to(getattr(got, k).dtype).float()).abs().max())
            for k in ("body_pos_w", "body_quat_w", "joint_pos")
        )
        print(f"[teleop] FK 自检：与数据集缓存最大差 {err:.2e}（存储 {got.body_pos_w.dtype}）", flush=True)
        if err > 1e-6:
            raise RuntimeError("FK 与数据集缓存不一致，刚体/关节映射有误")

        self.stand_pose = stand.copy()

        # reset 时从当前实时帧开始（而不是动作开头）
        from mimic_lite.tasks.command import assign_motion_clip

        def sample_motion(env_ids, **_):
            s = assign_motion_clip(ds, env_ids, 0)
            start = OFFSET + self.written - LAG if self.started else 0
            s.start_t = s.start_t.clone().fill_(max(0, start))
            return s

        ds.sample_motion = sample_motion
        self.attached = True

    def _fk(self, q: np.ndarray, v: np.ndarray) -> dict:
        import torch

        d = self.mdata
        d.qpos[:] = q
        d.qvel[:] = v
        mujoco.mj_forward(self.model, d)
        b = self.body_ids
        return {
            "body_pos_w": torch.as_tensor(np.asarray(d.xpos[b], dtype=np.float32)),
            "body_lin_vel_w": torch.as_tensor(np.asarray(d.cvel[b, 3:6], dtype=np.float32)),
            "body_quat_w": torch.as_tensor(np.asarray(d.xquat[b], dtype=np.float32)),
            "body_ang_vel_w": torch.as_tensor(np.asarray(d.cvel[b, 0:3], dtype=np.float32)),
            "joint_pos": torch.as_tensor(np.asarray(d.qpos[self.jq], dtype=np.float32)),
            "joint_vel": torch.as_tensor(np.asarray(d.qvel[self.jd], dtype=np.float32)),
        }

    def _write_row(self, i: int, fields: dict) -> None:
        data = self.full.data
        r = self.row0 + OFFSET + i
        for k, v in fields.items():
            buf = getattr(data, k)
            buf[r] = v.to(device=buf.device, dtype=buf.dtype)

    def write(self) -> None:
        wrote = False
        while self.written + 2 < len(self.qpos) and OFFSET + self.written + 1 < self.capacity - HOLD:
            i = self.written + 1
            self._write_row(i, self._fk(self.qpos[i], _qvel(self.model, self.qpos[i], self.qpos[i + 1])))
            self.written = i
            wrote = True
        if wrote:  # 最新帧之后先用 hold 顶上，真实帧到了会覆盖
            hold = self._fk(self.qpos[self.written], np.zeros(self.model.nv))
            for i in range(self.written + 1, self.written + 1 + HOLD):
                self._write_row(i, hold)

    # ---------- 每个控制步之前 ----------
    def tick(self, env, carry):
        if not self.attached:
            self.attach(env)
        self.steps += 1
        if not self.started:
            # 热身：机器人在占位动作开头原地站立；这期间到的实时帧照收，开始时从最新帧起播
            self.poll()
            if self.steps < WARMUP_STEPS:
                return carry
            if self.steps == WARMUP_STEPS:
                print(f"[teleop] 热身完成，等待 SMPL 流（{self.endpoint}）…", flush=True)
            if self.written < START_FRAMES:
                self.write()
                return carry
            self.started = True
            print(f"[teleop] 收到流（首帧号 {self.first_index}），开始跟踪", flush=True)
            return env.reset()  # sample_motion 已被替换：从当前实时帧开始
        self.poll()
        t = int(self.cmd.t[0]) - OFFSET
        # 断流：播放游标快追上最新帧、且一段时间没收到新帧 → 每步补一帧合成参考，让机器人平滑站好
        if time.perf_counter() - self.last_rx > STALL_S:
            while len(self.qpos) - 2 < t + LAG:  # 补到游标前方 LAG 帧，未来帧不落空
                self._feed_stall()
        self.write()
        if self.written - t > MAX_LAG:
            self.cmd.t[:] = OFFSET + self.written - LAG
            self.jumps += 1
        now = time.perf_counter()
        if now - self.t_report > 5.0:
            self.t_report = now
            print(f"[teleop] 已收 {self.received} 帧  已写 {self.written + 1}  "
                  f"播放 t={t}  落后 {self.written - t} 帧  补帧 {self.dropped}  跳帧 {self.jumps}"
                  f"{self._heading_check()}", flush=True)
        return carry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", type=Path, required=True, help="SONIC *_smpl.onnx")
    ap.add_argument("--source", choices=["smpl", "pico"], default="smpl",
                    help="smpl：tools_local/smpl_stream.py 逐帧流；pico：pico_manager_thread_server.py 的 pose 消息")
    ap.add_argument("--endpoint", default=None, help=f"默认 smpl {DEFAULT_ENDPOINT}，pico {PICO_ENDPOINT}")
    ap.add_argument("--minutes", type=float, default=30.0, help="单次会话最长时长（占位动作长度）")
    ap.add_argument("--no-mink", action="store_true", help="不做 mink 后处理（对照用）")
    ap.add_argument("play_args", nargs="*", help="`--` 之后原样传给 play.py 的 Hydra 参数")
    args = ap.parse_args()
    if args.endpoint is None:
        args.endpoint = PICO_ENDPOINT if args.source == "pico" else DEFAULT_ENDPOINT

    placeholder = ensure_placeholder(int(args.minutes * 60 * 50))
    feeder = LiveFeeder(args)

    # play.py 在 main 里才 import playback；等它被导入时把 play_pre_step 包一层
    real_import = builtins.__import__

    def hook(name, *a, **k):
        mod = real_import(name, *a, **k)
        pb = sys.modules.get("active_adaptation.envs.backends.mjlab.playback")
        if pb is not None and hasattr(pb, "play_pre_step") and not getattr(pb, "_teleop_patched", False):
            pb._teleop_patched = True
            orig = pb.play_pre_step

            def play_pre_step(env, carry, **kw):
                return orig(env, feeder.tick(env, carry), **kw)

            pb.play_pre_step = play_pre_step
        return mod

    builtins.__import__ = hook
    import os

    os.chdir(TRAINING)
    defaults = [
        "task=tracking-bumi-v2", "task/motion=bumi/v2", "+exp=ppo/train", "algo/ppo/module=huge",
        "backend=mjlab", "task.num_envs=1", "task.termination.root_pos_error.enabled=false",
        "task.command.start_from_zero=false",
        f"task.command.motion_cfgs.bumi_v2.path={placeholder}",
    ]
    sys.argv = [str(TRAINING / "projects/mimic-lite/scripts/play.py"), *defaults, *args.play_args]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
