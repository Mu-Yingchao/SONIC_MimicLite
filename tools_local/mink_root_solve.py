#!/usr/bin/env python3
"""用 mink 反解与 SONIC 重建关节自洽的 root 轨迹（借鉴 GMR）。

## 要解决的问题

喂给 MimicLite 的参考动作是 [root | 关节] 两部分：关节来自 SONIC 重建（g1_kin），
root 却只能取自源动作——SONIC 两个编码器的输入里都没有根平移（robot 只有关节角/速度
+ 根朝向差，SMPL 只有相对骨盆的局部关键点 + 根朝向），token 不含这个信息，解码器
不可能重建。于是 root 与重建关节不是同一套：root 说往前 1 米，重建步态只走 0.8 米，
脚在地上滑，play 里 ghost 与机器人越拉越远。实测 MimicLite 观测窗口（8 步）内
root 位移与关节隐含位移的不匹配为 0.030 m，而 GMR 原始数据只有 0.0001~0.013 m。

## GMR 怎么做到自洽

GMR（github.com/YanjieZe/GMR）用 mink 对整个 MuJoCo 模型做微分 IK：free joint
（root）和所有关节是同一组求解变量，一组 FrameTask 把各刚体拉向人体关键点。root 不是
"先定"或"后推"，而是和关节在同一个最小二乘里同时确定，脚部任务锁住了脚，root 为满足
它必须跟着动——自洽是联合优化的副产品。

## 这里的做法

同一套 mink 微分 IK，但**关节冻结为 SONIC 重建值**，只有 root 的 6 个自由度可动：
  - 双脚 FrameTask：拉向源动作里双脚的位姿（权重高）——脚踩在源动作踩的地方，
    root 自然与重建关节自洽，脚不打滑；
  - root FrameTask：拉向源动作的 root（权重低）——整体路线跟随源动作；
  - PostureTask：root 自由度权重 0、关节权重极大，把关节钉住；每次迭代后再把关节
    硬重置回重建值，防止数值漂移。
逐帧求解，用上一帧的解热启动。不需要重训 SONIC，不改任何网络。

源动作的"双脚/root 位姿"对 robot 链路就是真值机器人动作做正向运动学；对 SMPL 链路，
当前用配对的真值机器人动作（离线测试集都有）。纯 SMPL 输入时应换成缩放后的 SMPL 关键点，
即 GMR 原本的做法——属于后续工作。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mink
import mujoco
import numpy as np

FOOT_BODIES = ("l_ankle_roll_link", "r_ankle_roll_link")
ROOT_BODY = "base_link"
# BUMI2 MJCF 的 qpos 布局：free joint 0~6，腰 7，手臂 8~15，腿 16~27；
# 速度自由度对应 free 0~5，腰 6，手臂 7~14，腿 15~26。
LEG_QPOS = slice(16, 28)
LEG_DOF = slice(15, 27)


def foot_contacts(foot_pos: np.ndarray, fps: float = 50.0, height_tol: float = 0.03, speed_tol: float = 0.25) -> np.ndarray:
    """由源动作的脚部轨迹判定接触：[T, 2] 布尔。脚高接近该脚的最低点且水平几乎不动。"""
    h = foot_pos[:, :, 2]
    speed = np.zeros(h.shape)
    speed[1:] = np.linalg.norm(np.diff(foot_pos[:, :, :2], axis=0), axis=2) * fps
    speed[0] = speed[1]
    return (h < h.min(axis=0) + height_tol) & (speed < speed_tol)


def _body_poses(model, data, qpos: np.ndarray, bodies: list[int]) -> np.ndarray:
    """逐帧正向运动学，返回 [T, len(bodies), 7]（xyz + wxyz）。"""
    out = np.zeros((len(qpos), len(bodies), 7))
    for t, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        for k, b in enumerate(bodies):
            out[t, k, :3] = data.xpos[b]
            out[t, k, 3:] = data.xquat[b]
    return out


class RootSolver:
    """逐帧的 mink 求解器：给定一帧重建 qpos 和这一帧的目标，返回修正后的 qpos。

    离线（solve_root）和在线遥操（sonic_online_bridge）共用这一个类，保证两边算法一致。
    只依赖当前帧和上一帧的解（热启动），天然因果。

    mode="root"：关节全部冻结，只解 root。root 只有 6 个自由度，单脚支撑时正好能把
      支撑脚钉住；但双脚支撑时要同时满足两只脚（12 个约束），而重建腿部关节本身有误差、
      两脚相对位置就是错的，此时任何 root 都做不到两脚同时不动——这是冻结关节的根本局限。
    mode="legs"：上半身（腰+手臂）冻结为重建值，腿部关节也作为变量、带正则拉回重建值
      （leg_reg_cost）并受关节限位约束——即 GMR 式联合求解，只是把目标从人体关键点换成
      "源动作的支撑脚位姿"，并尽量少改 SONIC 的重建。
    两种模式都按接触状态加权：支撑脚强约束、摆动脚弱约束，避免摆动脚拖累支撑脚。
    """

    def __init__(
        self,
        mjcf: str,
        mode: str = "root",
        stance_pos_cost: float = 20.0,
        stance_ori_cost: float = 2.0,
        swing_pos_cost: float = 0.1,
        root_pos_cost: float = 1.0,
        root_ori_cost: float = 5.0,
        leg_reg_cost: float = 5.0,
        max_iter: int = 20,
        tol: float = 1e-5,
    ):
        if mode not in ("root", "legs"):
            raise ValueError(f"未知模式 {mode!r}")
        self.mode = mode
        self.stance_pos_cost, self.stance_ori_cost, self.swing_pos_cost = stance_pos_cost, stance_ori_cost, swing_pos_cost
        self.max_iter, self.tol = max_iter, tol
        self.model = mujoco.MjModel.from_xml_path(mjcf)
        self.data = mujoco.MjData(self.model)
        self.body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in (*FOOT_BODIES, ROOT_BODY)
        ]
        if min(self.body_ids) < 0:
            raise ValueError(f"MJCF 缺少刚体 {FOOT_BODIES + (ROOT_BODY,)}")

        self.cfg = mink.Configuration(self.model)
        self.feet = [
            mink.FrameTask(b, "body", position_cost=stance_pos_cost, orientation_cost=stance_ori_cost, lm_damping=1.0)
            for b in FOOT_BODIES
        ]
        self.root = mink.FrameTask(
            ROOT_BODY, "body", position_cost=root_pos_cost, orientation_cost=root_ori_cost, lm_damping=1.0
        )
        # free joint 的 6 个速度自由度权重 0 = 自由；上半身权重极大 = 冻结；
        # 腿部在 legs 模式下只加正则（拉回重建值），root 模式下同样冻结
        posture_cost = np.full(self.model.nv, 1e4)
        posture_cost[:6] = 0.0
        if mode == "legs":
            posture_cost[LEG_DOF] = leg_reg_cost
        self.posture = mink.PostureTask(self.model, cost=posture_cost)
        self.tasks = [*self.feet, self.root, self.posture]
        self.limits = [mink.ConfigurationLimit(self.model)] if mode == "legs" else None
        self.hard_reset = slice(7, 28) if mode == "root" else slice(7, 16)  # 硬重置回重建值的 qpos 段
        self.prev_root: np.ndarray | None = None
        self.iters: list[int] = []

    def reset(self) -> None:
        self.prev_root = None
        self.iters = []

    def step(self, q_recon: np.ndarray, targets: np.ndarray, contact: np.ndarray) -> np.ndarray:
        """q_recon [28]；targets [3,7]（左脚、右脚、root 的 xyz+wxyz）；contact [2] 布尔。

        首帧 root 从 root 目标起步，之后用上一帧的解热启动。
        """
        q = np.asarray(q_recon, dtype=np.float64).copy()
        q[:7] = targets[2] if self.prev_root is None else self.prev_root
        cfg = self.cfg
        cfg.update(q)
        self.posture.set_target(q)  # 正则目标 = 重建关节
        for k, task in enumerate(self.feet):
            task.set_position_cost(self.stance_pos_cost if contact[k] else self.swing_pos_cost)
            task.set_orientation_cost(self.stance_ori_cost if contact[k] else 0.0)
        for k, task in enumerate((*self.feet, self.root)):
            p = targets[k]
            task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(p[3:]), p[:3]))
        prev = cfg.q[:7].copy()
        for it in range(self.max_iter):
            v = mink.solve_ik(cfg, self.tasks, 0.02, solver="daqp", damping=1e-3, limits=self.limits)
            cfg.integrate_inplace(v, 0.02)
            qq = cfg.q.copy()
            qq[self.hard_reset] = q_recon[self.hard_reset]  # 冻结段硬重置，防数值漂移
            cfg.update(qq)
            if np.abs(cfg.q[:7] - prev).max() < self.tol:
                break
            prev = cfg.q[:7].copy()
        self.iters.append(it + 1)
        sol = cfg.q.copy()
        out = np.asarray(q_recon, dtype=np.float64).copy()
        out[:7] = sol[:7]
        if self.mode == "legs":
            out[LEG_QPOS] = sol[LEG_QPOS]
        # 四元数与上一帧同半球，避免下游插值走长路
        if self.prev_root is not None and float(np.dot(out[3:7], self.prev_root[3:7])) < 0.0:
            out[3:7] = -out[3:7]
        self.prev_root = out[:7].copy()
        return out


def solve_root(
    qpos_recon: np.ndarray,
    qpos_source: np.ndarray,
    mjcf: str,
    mode: str = "root",
    **solver_kwargs,
) -> tuple[np.ndarray, dict]:
    """离线整段求解：目标取自源动作（双脚 / root 位姿做正向运动学）。返回 (新 qpos, 求解统计)。"""
    solver = RootSolver(mjcf, mode=mode, **solver_kwargs)
    if solver.model.nq != qpos_recon.shape[1]:
        raise ValueError(f"MJCF nq={solver.model.nq} 与 qpos 维度 {qpos_recon.shape[1]} 不一致")
    n = min(len(qpos_recon), len(qpos_source))
    qpos_recon, qpos_source = qpos_recon[:n], qpos_source[:n]
    targets = _body_poses(solver.model, solver.data, qpos_source, solver.body_ids)  # 源动作的双脚 / root 位姿
    contact = foot_contacts(targets[:, :2, :3])
    out = np.stack([solver.step(qpos_recon[t], targets[t], contact[t]) for t in range(n)])
    iters = solver.iters
    stats = {
        "contact_ratio": float(contact.mean()),
        "leg_change_deg": float(np.rad2deg(np.abs(out[:, LEG_QPOS] - qpos_recon[:, LEG_QPOS])).mean()),
        "mean_iters": float(np.mean(iters)),
        "root_shift_m": float(np.linalg.norm(out[:, :3] - qpos_source[:, :3], axis=1).mean()),
    }
    return out.astype(np.float32), stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recon", type=Path, required=True, help="SONIC 重建的 qpos npz（关节来源）")
    ap.add_argument("--source", type=Path, required=True, help="源动作 qpos npz（双脚/root 目标来源）")
    ap.add_argument("--mjcf", type=Path, required=True, help="数据集自带的 FK 骨架 MJCF")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["root", "legs"], default="legs")
    args = ap.parse_args()
    q, st = solve_root(np.load(args.recon)["qpos"], np.load(args.source)["qpos"], str(args.mjcf), mode=args.mode)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, qpos=q)
    print(f"{args.recon.name}: {len(q)} 帧  平均迭代 {st['mean_iters']:.1f}  root 平均偏移 {st['root_shift_m']:.3f} m")


if __name__ == "__main__":
    main()
