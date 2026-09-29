#!/usr/bin/env python3
"""从重建的关节角反推自洽的 root 轨迹（脚不打滑约束）。

## 为什么需要这个

我们给 MimicLite 的参考动作是拼出来的：

    qpos = [真值 root | SONIC 重建的关节角]
            └─ A 的 ─┘   └──── B 的 ────┘

root 来自原始 GMR 数据，关节角来自 SONIC 重建（有 7~13° 误差）。GMR 数据里
root 和关节是同一次优化同时解出来的、带脚不打滑约束，所以自洽；换掉关节之后
这个配对就被破坏了。实测站立动作的脚底滑移从 0.001 m/s 涨到 0.122 m/s（84 倍）。

后果不只是难看。MimicLite 的终止条件 `cum_body_pos_error` 比的是
``(ref_body_pos_w - robot_body_link_pos_w)``，**世界系**。参考 root 说往前 1 米、
重建步态只走 0.8 米，机器人就越落越远直到被判定追踪失败；而策略的观测里
**没有任何 root 信息**（只有关节角 + 本体 IMU），它根本无从修正这个偏差。

官方两边都不会遇到：MimicLite 官方训练数据是 GMR 出的、天然自洽；SONIC 官方
用 g1_dyn 直接出动作，root 由物理涌现，不存在外部指定的 root。只有我们这种
"上层重建关节 + 下层追踪"的拼接架构会踩。

## 做法

标准的脚锁定（foot locking）反推：

1. 每帧用 MuJoCo 正向运动学，在 root 固定于原点的前提下算出两只脚的位置。
2. 判定当前哪只脚是支撑脚（取更低的那只）。
3. 支撑脚在世界系里应当不动，所以 root 的位移 = -(支撑脚在身体系里的位移)。
4. 逐帧积分得到 root 轨迹；换脚时以新支撑脚当前位置为基准，避免跳变。

root 朝向沿用重建输出里本来就有的朝向（g1_kin 会重建 root 朝向差值），
高度取支撑脚贴地所需的值。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np


def recover_root(
    qpos: np.ndarray, xml_path: str, foot_bodies: tuple[str, str] = ("l_ankle_roll_link", "r_ankle_roll_link")
) -> np.ndarray:
    """返回替换了 root 位置的 qpos（朝向保持不变）。"""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in foot_bodies]
    if any(i < 0 for i in foot_ids):
        raise ValueError(f"MJCF 里找不到脚部刚体 {foot_bodies}")

    n_frames = qpos.shape[0]

    # 在"root 位于原点、无偏航"的基准姿态下做 FK，得到脚相对 root 的位置与朝向。
    # 只保留 roll/pitch（由重建的 root 朝向决定身体倾斜），偏航另行积分——因为
    # 偏航正是要从支撑脚反推的量。
    feet_local = np.zeros((n_frames, 2, 3))
    feet_yaw_local = np.zeros((n_frames, 2))
    for t in range(n_frames):
        data.qpos[:28] = qpos[t]
        data.qpos[0:3] = 0.0
        # 去掉 root 偏航，只留 roll/pitch
        w, x, y, z = qpos[t, 3:7]
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        cy, sy = np.cos(-yaw / 2.0), np.sin(-yaw / 2.0)
        data.qpos[3:7] = np.array(
            [w * cy - z * sy, x * cy + y * sy, y * cy - x * sy, z * cy + w * sy]
        )
        mujoco.mj_forward(model, data)
        for k, bid in enumerate(foot_ids):
            feet_local[t, k] = data.xpos[bid]
            fw, fx, fy, fz = data.xquat[bid]
            feet_yaw_local[t, k] = np.arctan2(
                2.0 * (fw * fz + fx * fy), 1.0 - 2.0 * (fy * fy + fz * fz)
            )

    support = np.argmin(feet_local[:, :, 2], axis=1)

    # 先积分偏航：支撑脚在世界系里朝向不变，所以 root 偏航的增量等于
    # -(支撑脚在身体系里的偏航增量)。只修位置不修偏航会引入新的不一致——
    # 实测那样做 body_pos_error 从 0.113 降到 0.046，但 body_ori_error 会从
    # 0.013 涨到 0.237，整体反而更差。位置和朝向必须一起反推。
    root_yaw = np.zeros(n_frames)
    w0, x0, y0, z0 = qpos[0, 3:7]
    root_yaw[0] = np.arctan2(
        2.0 * (w0 * z0 + x0 * y0), 1.0 - 2.0 * (y0 * y0 + z0 * z0)
    )
    for t in range(1, n_frames):
        s_prev, s_now = support[t - 1], support[t]
        ref = s_now if s_prev == s_now else s_now
        d_yaw = feet_yaw_local[t, ref] - feet_yaw_local[t - 1, ref]
        d_yaw = (d_yaw + np.pi) % (2 * np.pi) - np.pi  # 绕回归一化
        root_yaw[t] = root_yaw[t - 1] - d_yaw

    # 用积分出的偏航把脚位置转到世界系，再积分 root 平移
    root_pos = np.zeros((n_frames, 3))
    root_pos[0, :2] = qpos[0, :2]

    def rot2(vec: np.ndarray, ang: float) -> np.ndarray:
        c, s = np.cos(ang), np.sin(ang)
        return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1], vec[2]])

    for t in range(1, n_frames):
        s_prev, s_now = support[t - 1], support[t]
        cur = rot2(feet_local[t, s_now], root_yaw[t])
        if s_prev == s_now:
            prev = rot2(feet_local[t - 1, s_now], root_yaw[t - 1])
            root_pos[t, :2] = root_pos[t - 1, :2] - (cur - prev)[:2]
        else:
            prev = rot2(feet_local[t - 1, s_now], root_yaw[t - 1])
            foot_world_prev = root_pos[t - 1, :2] + prev[:2]
            root_pos[t, :2] = foot_world_prev - cur[:2]

    # 高度：支撑脚始终贴地，避免积分漂移
    for t in range(n_frames):
        root_pos[t, 2] = -feet_local[t, support[t], 2]

    # 把积分出的偏航重新合成回四元数，保留原有的 roll/pitch
    out = qpos.copy()
    out[:, :3] = root_pos
    for t in range(n_frames):
        w, x, y, z = qpos[t, 3:7]
        old_yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        d = (root_yaw[t] - old_yaw) / 2.0
        c, s = np.cos(d), np.sin(d)
        out[t, 3:7] = [w * c - z * s, x * c + y * s, y * c - x * s, z * c + w * s]

    # 半球对齐，避免下游插值走长路
    for t in range(1, n_frames):
        if float(np.dot(out[t, 3:7], out[t - 1, 3:7])) < 0.0:
            out[t, 3:7] = -out[t, 3:7]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inp", type=Path, required=True, help="输入 qpos npz")
    ap.add_argument("--out", type=Path, required=True, help="输出 qpos npz")
    ap.add_argument("--mjcf", type=Path, required=True, help="数据集自带的 FK 骨架 MJCF")
    args = ap.parse_args()

    qpos = np.load(args.inp)["qpos"]
    fixed = recover_root(qpos, str(args.mjcf)).astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, qpos=fixed)
    drift = np.linalg.norm(fixed[:, :3] - qpos[:, :3], axis=1)
    print(f"{args.inp.name}: {len(qpos)} 帧  root 位置改动 平均 {drift.mean():.3f}m 最大 {drift.max():.3f}m -> {args.out}")


if __name__ == "__main__":
    main()
