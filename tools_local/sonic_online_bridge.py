#!/usr/bin/env python3
"""SMPL 遥操的在线桥接：逐帧吃 SMPL，因果地吐出 MimicLite 能读的参考 qpos。

和离线桥接（sonic_offline_bridge.py）的区别只有两点，其余逐位相同：

1. **因果**：每一刻只用已经收到的 SMPL 帧。
   SMPL 编码器的未来帧步长是 1，一个窗口只看 10 帧（0.18 s）。近端拼接下帧 t 用到的
   窗口起点 s 满足 s+9 ≤ t+4，所以**帧 t 最晚在收到 t+4 时就能算出来**，不需要把未来钳位，
   在线结果与离线整段计算逐位一致（除了动作末尾——离线把越界帧钳到最后一帧）。
2. **root 只来自 SMPL**：离线版的 mink 目标取自配对的机器人 PKL，遥操时没有它。
   这里用 SmplTargetEstimator 从 SMPL 估计 root 与双脚目标（2026-09-30 用 21 条配对动作标定）：
     - 水平位移：机器人 ≈ 0.59 × SMPL（跨动作一致，误差均值 1.1 cm）；
     - 双脚水平轨迹：同一比例缩放后与机器人脚踝差 0.9 cm（去掉每只脚的恒定站距差）；
     - 接触判定：与机器人真值一致 89%；
     - 朝向：SMPL 根朝向（与训练端同一换算）和机器人 yaw 差 1.9°，俯仰/侧倾差约 7°，交给 mink 调和。

延迟：SONIC 侧最多 4 帧（80 ms）；MimicLite 还要看参考的未来 4 步，整条链路约 8 帧（160 ms）。

    # 用 PKL 模拟实时流（逐帧喂），输出 qpos npz，并与离线结果对照
    python tools_local/sonic_online_bridge.py \\
      --smpl test_data/motions/smpl_pkl/walk_forward_loop_003__A022.pkl \\
      --onnx test_data/policies/sonic/recon_v1_step_010000_smpl.onnx \\
      --manifest test_data/motions/any4hdmi-bumi-v2/manifest.json \\
      --out /tmp/online.npz
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from mink_root_solve import FOOT_BODIES, RootSolver
from sonic_offline_bridge import (
    FUTURE_FRAME_STRIDE,
    NUM_DOF,
    NUM_FUTURE_FRAMES,
    PROPRIOCEPTION_DIM,
    load_joint_maps,
    load_robot_motion,
    open_session,
    rotation_to_6d,
    split_pos_vel,
)

# SMPL → 机器人的尺度与站姿（21 条配对动作标定，见模块说明）
XY_SCALE = 0.59
Z_SCALE = 0.47
ROBOT_ROOT_Z = 0.474
# 接触判定阈值：与 mink_root_solve.foot_contacts 相同，只是地面高度改为因果的历史最低
CONTACT_HEIGHT_TOL = 0.03
CONTACT_SPEED_TOL = 0.25
FPS = 50.0

_Y_TO_Z_UP = Rotation.from_rotvec([np.pi / 2.0, 0.0, 0.0])
_SMPL_BASE_INV = Rotation.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
SMPL_ANKLES = (7, 8)


def smpl_root_rotation(root_aa: np.ndarray) -> Rotation:
    """单帧 SMPL 根朝向 → 训练端的 Z-up 根朝向（与 sonic_offline_bridge.prepare_smpl_reference 同一换算）。"""
    return _Y_TO_Z_UP * Rotation.from_rotvec(root_aa) * _SMPL_BASE_INV


def _yaw(rot: Rotation) -> float:
    return float(rot.as_euler("ZYX")[0])


def _rz(yaw: float) -> Rotation:
    return Rotation.from_euler("Z", yaw)


class SmplTargetEstimator:
    """从 SMPL 逐帧估计 mink 的目标（左脚、右脚、root 位姿）和双脚接触状态。

    第 0 帧做标定：约定此刻人站立、机器人 root 在 origin，机器人双脚取重建首帧的正向运动学，
    记下"机器人脚相对 SMPL 缩放脚踝"的偏移（在身体朝向坐标系里记，人转身时偏移跟着转）。
    """

    def __init__(self, model: mujoco.MjModel, origin_xy=(0.0, 0.0), foot_lock: bool = True):
        self.model = model
        self.data = mujoco.MjData(model)
        self.foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in FOOT_BODIES]
        self.origin_xy = np.asarray(origin_xy, dtype=np.float64)
        # 脚锁：着地瞬间记下目标位置，着地期间目标不再跟随 SMPL 脚踝。
        # 不锁的话 SMPL 脚踝在支撑期的微小移动（约 0.06 m/s）会原样变成参考动作的脚底打滑。
        self.foot_lock = foot_lock
        self.locked: list[np.ndarray | None] = [None, None]
        self.calibrated = False

    def _smpl_world(self, frame: dict) -> tuple[np.ndarray, np.ndarray, Rotation]:
        """返回 (骨盆世界位置 Z-up, 双脚踝世界位置 Z-up [2,3], 根朝向)。

        注意坐标：PKL 的 smpl_joints 已是 Z-up、以骨盆为原点附近；transl 仍是 Y-up，需要转轴。
        """
        pelvis = _Y_TO_Z_UP.apply(np.asarray(frame["transl"], dtype=np.float64))
        joints = np.asarray(frame["smpl_joints"], dtype=np.float64)
        ankles = joints[list(SMPL_ANKLES)] - joints[0] + pelvis
        return pelvis, ankles, smpl_root_rotation(np.asarray(frame["pose_aa"], dtype=np.float64)[:3])

    def calibrate(self, frame: dict, q_recon0: np.ndarray) -> None:
        pelvis, ankles, rot = self._smpl_world(frame)
        self.pelvis0 = pelvis
        self.yaw0 = _yaw(rot)
        # 标定姿态只取 SMPL 的 yaw、root 放平：SMPL 骨盆倾角与机器人差约 7°，带进来会被
        # 复制进脚的朝向目标，再经 mink 反传回 root
        q = np.asarray(q_recon0, dtype=np.float64).copy()
        q[:3] = [*self.origin_xy, ROBOT_ROOT_Z]
        q[3:7] = _rz(self.yaw0).as_quat()[[3, 0, 1, 2]]
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        feet = np.array([self.data.xpos[i] for i in self.foot_ids])
        feet_quat = np.array([self.data.xquat[i] for i in self.foot_ids])  # wxyz
        scaled = self._scaled_xy(ankles)
        # 偏移记在第 0 帧的身体朝向坐标系里
        self.foot_offset_local = _rz(self.yaw0).inv().apply(
            np.concatenate([feet[:, :2] - scaled, np.zeros((2, 1))], axis=1))
        self.foot_ground_z = feet[:, 2].copy()
        # 脚的朝向目标：放平（支撑脚贴地），只保留 yaw
        self.foot_rot0 = [_rz(_yaw(Rotation.from_quat(fq[[1, 2, 3, 0]]))) for fq in feet_quat]
        self.human_ground = ankles[:, 2].copy()  # 地面高度：之后取历史最低
        self.prev_ankle_xy = scaled.copy()
        self.calibrated = True

    def _scaled_xy(self, pts: np.ndarray) -> np.ndarray:
        return self.origin_xy + XY_SCALE * (pts[..., :2] - self.pelvis0[:2])

    def step(self, frame: dict) -> tuple[np.ndarray, np.ndarray]:
        """返回 (targets [3,7] 左脚/右脚/root 的 xyz+wxyz, contact [2])。必须先 calibrate。"""
        pelvis, ankles, rot = self._smpl_world(frame)
        dyaw = _rz(_yaw(rot) - self.yaw0)
        targets = np.zeros((3, 7))
        # root：水平按比例、高度按比例围绕标定站高、朝向直接用 SMPL 根朝向
        targets[2, :2] = self._scaled_xy(pelvis)
        targets[2, 2] = ROBOT_ROOT_Z + Z_SCALE * (pelvis[2] - self.pelvis0[2])
        targets[2, 3:] = rot.as_quat()[[3, 0, 1, 2]]
        # 双脚：水平 = 缩放脚踝 + 随身体朝向旋转的站距偏移；高度 = 地面 + 缩放后的抬脚高度
        self.human_ground = np.minimum(self.human_ground, ankles[:, 2])
        scaled = self._scaled_xy(ankles)
        offset = dyaw.apply(self.foot_offset_local)[:, :2]
        lift = ankles[:, 2] - self.human_ground
        for k in range(2):
            targets[k, :2] = scaled[k] + offset[k]
            targets[k, 2] = self.foot_ground_z[k] + XY_SCALE * lift[k]
            targets[k, 3:] = (dyaw * self.foot_rot0[k]).as_quat()[[3, 0, 1, 2]]
        # 接触：离历史最低地面 3 cm 以内，且缩放后的水平速度 < 0.25 m/s（与离线判定同阈值）
        speed = np.linalg.norm(scaled - self.prev_ankle_xy, axis=1) * FPS
        self.prev_ankle_xy = scaled
        contact = (lift < CONTACT_HEIGHT_TOL) & (speed < CONTACT_SPEED_TOL)
        if self.foot_lock:
            for k in range(2):
                if not contact[k]:
                    self.locked[k] = None
                    continue
                if self.locked[k] is None:
                    self.locked[k] = np.array([*targets[k, :2], self.foot_ground_z[k]])
                targets[k, :3] = self.locked[k]
        return targets, contact


class OnlineSmplBridge:
    """push(SMPL 帧) → 返回这次新产出的 [(帧号, qpos[28])...]。"""

    def __init__(
        self,
        onnx: Path,
        manifest: Path,
        root_solve: bool = True,
        origin_xy=(0.0, 0.0),
        foot_lock: bool = True,
        tilt_cost: float = 0.0,
    ):
        """tilt_cost：root 俯仰/侧倾跟随 SMPL 的权重（yaw 固定 5）。人和机器人骨盆倾角差约 7°，
        给低权重让倾角主要由双脚 + 重建腿部关节决定。"""
        self.session = open_session(onnx, "smpl")
        self.in_name = self.session.get_inputs()[0].name
        self.perm, _ = load_joint_maps(manifest)
        man = json.loads(Path(manifest).read_text(encoding="utf-8"))
        mjcf = str(Path(manifest).parent / man["mjcf"])
        # base_link 局部系 x 前 y 左 z 上：绕 x/y 是侧倾/俯仰，绕 z 是 yaw
        self.solver = (
            RootSolver(mjcf, mode="legs", root_ori_cost=np.array([tilt_cost, tilt_cost, 5.0]))
            if root_solve else None
        )
        self.estimator = SmplTargetEstimator(mujoco.MjModel.from_xml_path(mjcf), origin_xy, foot_lock=foot_lock)
        self.proprio = np.zeros(PROPRIOCEPTION_DIM, dtype=np.float32)  # g1_kin 不读本体感知
        self.reset(origin_xy)

    def reset(self, origin_xy=(0.0, 0.0)) -> None:
        """清空所有流状态，重新从下一帧开始（断流恢复时用）。origin_xy：新一段在机器人世界里的起点。"""
        self.estimator.origin_xy = np.asarray(origin_xy, dtype=np.float64)
        self.estimator.calibrated = False
        self.estimator.locked = [None, None]
        if self.solver is not None:
            self.solver.reset()
        self.local: list[np.ndarray] = []   # 每帧局部关键点 [24,3]
        self.roots: list[Rotation] = []     # 每帧训练端根朝向
        self.frames: list[dict] = []
        self.decoded: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.next_out = 0
        self.pending_targets: deque = deque()

    def _decode(self, s: int) -> tuple[np.ndarray, np.ndarray]:
        if s not in self.decoded:
            idx = [s + k for k in range(NUM_FUTURE_FRAMES)]
            ori = rotation_to_6d(self.roots[s].inv() * Rotation.concatenate([self.roots[i] for i in idx]))
            tok = np.concatenate([np.stack([self.local[i] for i in idx]).reshape(-1), ori.reshape(-1)])
            obs = np.concatenate([tok, self.proprio]).astype(np.float32)[None]
            out = self.session.run(None, {self.in_name: obs})[0].reshape(NUM_FUTURE_FRAMES, 48)
            self.decoded[s] = split_pos_vel(out)
            self.decoded.pop(s - 3 * FUTURE_FRAME_STRIDE, None)  # 只留最近几个窗口
        return self.decoded[s]

    def _at(self, s: int, offset: float) -> np.ndarray:
        pos, _ = self._decode(s)
        k0 = int(offset // FUTURE_FRAME_STRIDE)
        a = (offset - k0 * FUTURE_FRAME_STRIDE) / FUTURE_FRAME_STRIDE
        return (1 - a) * pos[k0] + a * pos[k0 + 1]

    def _joints(self, t: int) -> np.ndarray | None:
        """帧 t 的重建关节（IsaacLab 顺序）；所需输入还没到则返回 None。与离线 near 拼接同一公式。"""
        stride = FUTURE_FRAME_STRIDE
        if t < 2 * stride:
            need, parts = NUM_FUTURE_FRAMES - 1, [(0, float(t), 1.0)]
        else:
            s = stride * ((t - stride) // stride)
            u = (t - s - stride) / stride
            need = s + NUM_FUTURE_FRAMES - 1
            parts = [(s - stride, float(t - s + stride), 1 - u), (s, float(t - s), u)]
        if need >= len(self.local):
            return None
        return sum(w * self._at(s, off) for s, off, w in parts)

    def push(self, frame: dict) -> list[tuple[int, np.ndarray]]:
        pose = np.asarray(frame["pose_aa"], dtype=np.float64)
        root = smpl_root_rotation(pose[:3])
        joints = np.asarray(frame["smpl_joints"], dtype=np.float64)
        self.local.append((root.inv().apply(joints)).astype(np.float32))
        self.roots.append(root)
        self.frames.append(frame)
        out = []
        while True:
            t = self.next_out
            jp = self._joints(t)
            if jp is None:
                break
            q = np.zeros(7 + NUM_DOF)
            q[7:] = np.asarray(jp)[self.perm]
            if not self.estimator.calibrated:
                self.estimator.calibrate(self.frames[t], q)
            targets, contact = self.estimator.step(self.frames[t])
            q[:7] = targets[2]
            if self.solver is not None:
                q = self.solver.step(q, targets, contact)
            out.append((t, q.astype(np.float32)))
            self.frames[t] = None  # 已用完，释放
            self.next_out += 1
        return out


def pkl_frames(motion: dict):
    for t in range(len(motion["pose_aa"])):
        yield {k: np.asarray(motion[k])[t] for k in ("pose_aa", "transl", "smpl_joints")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smpl", type=Path, required=True, help="SMPL PKL，逐帧喂入模拟实时流")
    ap.add_argument("--onnx", type=Path, required=True, help="*_smpl.onnx")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root-solve", choices=["legs", "none"], default="legs")
    args = ap.parse_args()

    bridge = OnlineSmplBridge(args.onnx, args.manifest, root_solve=args.root_solve == "legs")
    motion = load_robot_motion(args.smpl)
    qs, lag = [], []
    for i, fr in enumerate(pkl_frames(motion)):
        for t, q in bridge.push(fr):
            qs.append(q)
            lag.append(i - t)
    qpos = np.stack(qs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, qpos=qpos)
    print(f"输入 {len(motion['pose_aa'])} 帧，产出 {len(qpos)} 帧（末尾 {len(motion['pose_aa']) - len(qpos)} 帧需要后续输入）")
    print(f"延迟（收到第几帧时产出）: 最大 {max(lag)} 帧，平均 {np.mean(lag):.1f} 帧")


if __name__ == "__main__":
    main()
