#!/usr/bin/env python3
"""本地离线跑通 SONIC 上层：动作 PKL -> ONNX 编解码 -> MimicLite 能读的 qpos npz。

不需要 Isaac Lab，也不需要 MuJoCo。纯 NumPy + onnxruntime。

## 为什么可以不要仿真器

`g1_kin` 是纯运动学重建解码器。导出的 ONNX 输入是
``tokenizer(480 或 780) + proprioception(690)``，但实测把本体感知那 690 维
换成任意随机值（幅度 0.1~2.0，各重复多次），输出变化恒为 **0.00000000**，
而改 tokenizer 会带来 0.44 量级的变化——说明 g1_kin 根本不读机器人状态。
所以离线生成参考动作时本体感知填零即可，结果与带真实机器人状态时逐位相同。

这也是为什么此前必须上 Isaac Lab 的那条路可以作废：当初 `export_bridge_motion`
写在 `train_agent_trl.py` 里、要靠 `env.reset()` 拿观测，其实只有 tokenizer
那部分是真正需要的，而 tokenizer 完全可以从 PKL 直接算出来。

## 机器人朝向怎么办

tokenizer 里的朝向项是"参考朝向相对**机器人当前**朝向"的 6D 表示。离线没有
机器人，这里用参考自身的 anchor 朝向替代——这等价于 Isaac Lab 桥接导出里每个
窗口 `env.reset()` 把机器人物理姿态摆到参考起始帧的那一瞬间，即"机器人此刻
正好站在参考姿态上"。相对旋转因此退化为单位阵，与训练时窗口开头的情形一致。

## 用法

    python3 tools_local/sonic_offline_bridge.py \\
        --motion test_data/motions/robot_pkl/walk_forward_loop_003__A022.pkl \\
        --onnx   test_data/policies/sonic/model_step_017600_g1.onnx \\
        --encoder robot \\
        --out    test_data/motions/any4hdmi-bumi-v2/motions/sonic_bridge/xxx.npz \\
        --manifest test_data/motions/any4hdmi-bumi-v2/manifest.json

SMPL 链路把 --encoder 换成 smpl、--onnx 换成 *_smpl.onnx、--motion 指向
配对的 SMPL PKL 即可。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation

# 解码器一次前向覆盖的未来帧数与帧间隔（与训练配置一致：10 帧 × 0.1s = 1 秒）
NUM_FUTURE_FRAMES = 10
FUTURE_FRAME_STRIDE = 5  # robot 侧：50Hz 下每 5 帧 = 0.1s
SMPL_FUTURE_STRIDE = 1   # SMPL 侧步长是 1，与 robot 侧不同
PROPRIOCEPTION_DIM = 690
ROBOT_TOKENIZER_DIM = 480
SMPL_TOKENIZER_DIM = 780
NUM_DOF = 21


def quat_wxyz_to_rot(quat_wxyz: np.ndarray) -> Rotation:
    """scipy 用 xyzw，数据里是 wxyz。"""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    return Rotation.from_quat(q[..., [1, 2, 3, 0]])


def _centered_finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    """中心差分求一阶导，端点用单边差分。

    与 ``gear_sonic/utils/motion_lib/motion_lib_base.py`` 里同名函数的行为一致。
    这里独立实现是为了让本脚本只依赖 NumPy，不用为了一个差分去 import 整条
    motion_lib 依赖链（那会把 torch 和 Isaac Lab 相关的东西一起拖进来）。
    """
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] < 2:
        return np.zeros_like(values)
    out = np.empty_like(values)
    out[1:-1] = (values[2:] - values[:-2]) * (fps / 2.0)
    out[0] = (values[1] - values[0]) * fps
    out[-1] = (values[-1] - values[-2]) * fps
    return out


def rotation_to_6d(rot: Rotation) -> np.ndarray:
    """旋转矩阵前两列展平成 6D 表示，与训练端一致。"""
    return rot.as_matrix()[..., :2].reshape(*rot.as_matrix().shape[:-2], 6)


def load_robot_motion(path: Path) -> dict:
    """读 SONIC 训练用的机器人动作 PKL。"""
    raw = joblib.load(path)
    if isinstance(raw, dict) and len(raw) == 1 and isinstance(next(iter(raw.values())), dict):
        raw = next(iter(raw.values()))
    return raw


def build_robot_tokenizer(
    motion: dict, frame: int, num_frames: int, mujoco_to_isaaclab: list[int]
) -> np.ndarray:
    """复刻训练端 robot encoder 的 480 维输入。

    两个容易静默出错的点：

    1. 关节顺序。PKL 里的 ``dof`` 是 **MuJoCo 顺序**，而 encoder 吃的
       ``command_multi_future`` 是 **IsaacLab 顺序**，必须先重排。不转的话数值
       全是错位的，但程序不会报错——实测直接喂 MuJoCo 顺序，重建误差从 4.7°
       恶化到 9.6°，正好是"看起来能跑但结果不对"的典型。
    2. 布局。必须是 cat(joint_pos_flat, joint_vel_flat, orientation_6d_flat)——
       训练端先把两个 (F,D) 各自展平再 concat，不是按帧交错。
    """
    dof_mujoco = np.asarray(motion["dof"], dtype=np.float32)   # (T, 21) MuJoCo 顺序
    dof = dof_mujoco[:, mujoco_to_isaaclab]                    # -> IsaacLab 顺序
    root_rot = np.asarray(motion["root_rot"], dtype=np.float32)  # (T, 4) xyzw
    total = dof.shape[0]

    idx = np.clip(frame + np.arange(num_frames) * FUTURE_FRAME_STRIDE, 0, total - 1)

    joint_pos = dof[idx]
    # PKL 不带关节速度，动作库现算。用**前向差分**——对应
    # torch_humanoid_batch.py:447 的 `(dof_pos[:,1:] - dof_pos[:,:-1]) / dt`，
    # 那条才是喂 command_multi_future 的路径。
    # 注意别照抄 motion_lib_base._centered_finite_difference：那个中心差分是
    # 动力学门禁（筛异常片段）在用的，不是 tokenizer 的输入来源。实测误用中心
    # 差分会让重建误差从 5.17° 变成 5.35°。
    nxt = np.clip(idx + 1, 0, total - 1)
    joint_vel = (dof[nxt] - dof[idx]) * 50.0

    # 朝向：参考 anchor 相对机器人当前朝向。离线用参考自身首帧朝向当作机器人朝向
    # （等价于 reset 后机器人正好站在参考姿态上），因此相对旋转 = 当前帧相对首帧。
    ref_rot = Rotation.from_quat(root_rot[idx])           # 已是 xyzw
    robot_rot = Rotation.from_quat(root_rot[frame])       # 机器人 = 参考当前帧
    orientation_6d = rotation_to_6d(robot_rot.inv() * ref_rot)

    value = np.concatenate(
        (joint_pos.reshape(-1), joint_vel.reshape(-1), orientation_6d.reshape(-1))
    ).astype(np.float32)
    if value.size != ROBOT_TOKENIZER_DIM:
        raise ValueError(f"robot tokenizer 应为 {ROBOT_TOKENIZER_DIM} 维，实际 {value.size}")
    return value


def prepare_smpl_reference(motion: dict) -> tuple[np.ndarray, Rotation]:
    """把 SMPL PKL 预处理成训练端见到的形式：局部关键点 + Z-up 根朝向。

    这段换算直接照搬 ``gear_sonic/utils/mujoco_sim/bumi3_smpl_reference.py``
    里已验证过的实现，不要自己重推。其中**右乘 SMPL 基准旋转的逆**
    （``Rotation.from_quat([0.5,0.5,0.5,0.5]).inv()``）那一项很容易漏——漏掉
    以后程序照跑不误，但重建误差会从 5° 级别劣化到 21°。

    对齐的是 commands.py 里的 ``smpl_root_quat_w_multi_future``：
    左乘 Y-up→Z-up 转轴，右乘基准旋转的逆，再用每帧根朝向的逆旋转关键点。
    不能再额外转关键点轴、减 pelvis 或叠加 transl，否则输入分布就和 checkpoint
    见过的不一样了。
    """
    pose = np.asarray(motion["pose_aa"], dtype=np.float64)
    joints = np.asarray(motion["smpl_joints"], dtype=np.float64)
    root = (
        Rotation.from_rotvec([np.pi / 2.0, 0.0, 0.0])
        * Rotation.from_rotvec(pose[:, :3])
        * Rotation.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
    )
    local_joints = np.einsum("tji,tkj->tki", root.as_matrix(), joints).astype(np.float32)
    return local_joints, root


def build_smpl_tokenizer(
    local_joints: np.ndarray, root: Rotation, frame: int, num_frames: int
) -> np.ndarray:
    """复刻训练端 smpl encoder 的 780 维输入（720 局部关键点 + 60 朝向）。

    SMPL 侧的未来帧步长是 1（与 robot 侧的 5 不同），见
    bumi3_smpl_reference.SMPL_FUTURE_STRIDE。
    """
    total = local_joints.shape[0]
    idx = np.clip(frame + np.arange(num_frames) * SMPL_FUTURE_STRIDE, 0, total - 1)

    # 朝向：参考朝向相对机器人当前朝向。离线用参考自身当前帧当作机器人朝向。
    orientation_6d = rotation_to_6d(root[frame].inv() * root[idx])

    value = np.concatenate(
        (local_joints[idx].reshape(-1), orientation_6d.reshape(-1))
    ).astype(np.float32)
    if value.size != SMPL_TOKENIZER_DIM:
        raise ValueError(f"smpl tokenizer 应为 {SMPL_TOKENIZER_DIM} 维，实际 {value.size}")
    return value


def split_pos_vel(decoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """解码输出 (F, 48) -> (关节位置, 关节速度)。

    训练端是 cat(pos_flat, vel_flat) 之后再 reshape 成 (F, 2D)，所以**不能**直接
    按行切前 21 / 后 21——必须先重新摊平再按 F*D 边界切。这个坑踩过一次。
    """
    flat = decoded[:, : NUM_DOF * 2].reshape(-1)
    half = decoded.shape[0] * NUM_DOF
    return flat[:half].reshape(-1, NUM_DOF), flat[half : half * 2].reshape(-1, NUM_DOF)


def load_joint_maps(manifest_path: Path) -> tuple[list[int], list[int]]:
    """从 manifest 现读关节重排映射（不硬编码）。

    返回 (perm, mujoco_to_isaac)：
      perm            IsaacLab -> MuJoCo（解码输出 -> qpos）
      mujoco_to_isaac MuJoCo -> IsaacLab（PKL 的 dof -> encoder 输入）
    """
    man = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    src_names = man["source"]["source_joint_names"]       # IsaacLab
    tgt_names = man["source"]["target_hinge_joint_names"]  # MuJoCo
    return [src_names.index(n) for n in tgt_names], [tgt_names.index(n) for n in src_names]


def open_session(onnx_path: Path, encoder: str) -> ort.InferenceSession:
    """打开 ONNX 并校验输入维度与 encoder 匹配。"""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_dim = session.get_inputs()[0].shape[1]
    tok_dim = SMPL_TOKENIZER_DIM if encoder == "smpl" else ROBOT_TOKENIZER_DIM
    if in_dim != tok_dim + PROPRIOCEPTION_DIM:
        raise ValueError(
            f"ONNX 输入 {in_dim} 与 encoder={encoder} 期望的 "
            f"{tok_dim}+{PROPRIOCEPTION_DIM} 不符，模型和 --encoder 选错了"
        )
    return session


def reconstruct_qpos(
    session: ort.InferenceSession,
    motion: dict,
    encoder: str,
    perm: list[int],
    mujoco_to_isaac: list[int],
    root_src: dict,
    stitch: str = "near",
) -> tuple[np.ndarray, np.ndarray, int]:
    """动作 -> ONNX 编解码 -> qpos。

    返回 (qpos[T,28], 重建关节角 MuJoCo 顺序 [T,21], 四元数半球修正帧数)。
    命令行入口和 checkpoint 回归脚本共用这一个函数，保证两边算的是同一个东西。

    stitch 决定怎么把每次解码的 1 秒窗口拼成整条动作（默认 near）：
      window  每 46 帧解码一次、用满整个窗口（0~0.9s 的预测全用上）——旧做法
      near    每 5 帧解码一次、只用窗口里 0.1~0.3s 的那段，相邻窗口交叉淡化
    实测解码误差随窗口内偏移呈 U 形（21 条动作，step 17600，robot 链路）：
    偏移 0 帧 6.37°、5~10 帧 5.56~5.64°、45 帧 8.30°。window 模式把远端那段
    误差大的预测也平均进来了；near 只取最准的一段，也更接近在线部署时"每个控制步
    重新解码、只用近端预测"的真实用法。

    实测 21 条动作（step 17600）window -> near：smpl 平均 7.52° -> 5.50°（-27%），
    p95 25.5° -> 17.9°，root 位移不匹配 0.0423 -> 0.0305 m（-28%），
    关节加速度（抖动）2.61x -> 2.82x 真值，基本持平。
    """
    if stitch not in ("window", "near"):
        raise ValueError(f"未知拼接方式 {stitch!r}")
    in_name = session.get_inputs()[0].name
    smpl_local, smpl_root = prepare_smpl_reference(motion) if encoder == "smpl" else (None, None)

    # root 轨迹不过 decoder，直接取真值（g1_kin 只重建关节，这是刻意设计）
    if "root_trans_offset" not in root_src:
        raise ValueError("root 轨迹来源缺少 root_trans_offset，SMPL 链路需要 --root-motion 指向配对的机器人 PKL")
    root_pos = np.asarray(root_src["root_trans_offset"], dtype=np.float32)
    root_quat_xyzw = np.asarray(root_src["root_rot"], dtype=np.float32)
    total_frames = root_pos.shape[0]

    # 逐窗口解码再拼接：每个窗口覆盖 (NUM_FUTURE_FRAMES-1)*STRIDE+1 帧
    window_span = (NUM_FUTURE_FRAMES - 1) * FUTURE_FRAME_STRIDE + 1
    sparse_offsets = np.arange(NUM_FUTURE_FRAMES) * FUTURE_FRAME_STRIDE
    dense_offsets = np.arange(window_span)

    joint_pos_full = np.zeros((total_frames, NUM_DOF), dtype=np.float32)
    joint_vel_full = np.zeros((total_frames, NUM_DOF), dtype=np.float32)
    filled = np.zeros(total_frames, dtype=bool)

    proprio = np.zeros(PROPRIOCEPTION_DIM, dtype=np.float32)  # 实测对输出零影响

    def decode(start: int) -> tuple[np.ndarray, np.ndarray]:
        if encoder == "smpl":
            tok = build_smpl_tokenizer(smpl_local, smpl_root, start, NUM_FUTURE_FRAMES)
        else:
            tok = build_robot_tokenizer(motion, start, NUM_FUTURE_FRAMES, mujoco_to_isaac)
        obs = np.concatenate([tok, proprio]).astype(np.float32)[None]
        decoded = session.run(None, {in_name: obs})[0].reshape(NUM_FUTURE_FRAMES, 48)
        return split_pos_vel(decoded)

    if stitch == "window":
        for start in range(0, total_frames, window_span):
            pos_sparse, vel_sparse = decode(start)
            # 10 个稀疏采样点插值成连续帧
            pos_dense = np.stack(
                [np.interp(dense_offsets, sparse_offsets, pos_sparse[:, j]) for j in range(NUM_DOF)], axis=1
            )
            vel_dense = np.stack(
                [np.interp(dense_offsets, sparse_offsets, vel_sparse[:, j]) for j in range(NUM_DOF)], axis=1
            )
            end = min(start + window_span, total_frames)
            n = end - start
            joint_pos_full[start:end] = pos_dense[:n]
            joint_vel_full[start:end] = vel_dense[:n]
            filled[start:end] = True
    else:
        # 每 STRIDE(5) 帧起一个窗口。帧 t 同时落在两个窗口的"准区"里：
        #   新窗口 s   = t 往前对齐到的起点，偏移 t-s ∈ [5,10)
        #   旧窗口 s-5，偏移 ∈ [10,15)
        # 按 u=(t-s-5)/5 从旧窗口线性过渡到新窗口。t 走到 s+10 时混合值恰等于
        # "窗口 s 在偏移 10 处的预测"，而下一段开头取的也正是它——切换点连续。
        # 不做交叉淡化、每 5 帧硬切窗口的话，误差同样降，但切换点有台阶，实测关节
        # 加速度涨到真值的 ~8 倍，等于给 MimicLite 的参考注入 10Hz 抖动。
        cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        def at(s: int, offset: float) -> tuple[np.ndarray, np.ndarray]:
            if s not in cache:
                cache[s] = decode(s)
            pos_sparse, vel_sparse = cache[s]
            k0 = int(offset // FUTURE_FRAME_STRIDE)
            a = (offset - k0 * FUTURE_FRAME_STRIDE) / FUTURE_FRAME_STRIDE
            return (
                (1 - a) * pos_sparse[k0] + a * pos_sparse[k0 + 1],
                (1 - a) * vel_sparse[k0] + a * vel_sparse[k0 + 1],
            )

        for t in range(total_frames):
            if t < 2 * FUTURE_FRAME_STRIDE:
                # 开头不足两个窗口可混合，直接用窗口 0
                pos, vel = at(0, float(t))
            else:
                s = FUTURE_FRAME_STRIDE * ((t - FUTURE_FRAME_STRIDE) // FUTURE_FRAME_STRIDE)
                u = (t - s - FUTURE_FRAME_STRIDE) / FUTURE_FRAME_STRIDE
                new_pos, new_vel = at(s, float(t - s))
                old_pos, old_vel = at(s - FUTURE_FRAME_STRIDE, float(t - s + FUTURE_FRAME_STRIDE))
                pos = (1 - u) * old_pos + u * new_pos
                vel = (1 - u) * old_vel + u * new_vel
            joint_pos_full[t] = pos
            joint_vel_full[t] = vel
            filled[t] = True

    if not filled.all():
        raise RuntimeError(f"有 {int((~filled).sum())} 帧没被任何窗口覆盖")

    # root 四元数：数据里是 xyzw，qpos 要 wxyz，并强制相邻帧同半球
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]].astype(np.float32)
    flips = 0
    for i in range(1, len(root_quat_wxyz)):
        if float(np.dot(root_quat_wxyz[i], root_quat_wxyz[i - 1])) < 0.0:
            root_quat_wxyz[i] = -root_quat_wxyz[i]
            flips += 1

    joint_pos_mujoco = joint_pos_full[:, perm]
    qpos = np.concatenate([root_pos, root_quat_wxyz, joint_pos_mujoco], axis=1).astype(np.float32)
    return qpos, joint_pos_mujoco, flips


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motion", type=Path, required=True, help="SONIC 训练用的动作 PKL")
    ap.add_argument("--onnx", type=Path, required=True, help="导出的联合 ONNX（*_g1 或 *_smpl）")
    ap.add_argument("--encoder", choices=["robot", "smpl"], required=True)
    ap.add_argument("--out", type=Path, required=True, help="输出的 qpos npz")
    ap.add_argument("--manifest", type=Path, required=True, help="any4hdmi manifest.json，用于关节重排")
    ap.add_argument("--root-motion", type=Path, default=None,
                    help="SMPL 链路需要配对的机器人 PKL 提供 root 轨迹；robot 链路默认用 --motion 自身")
    ap.add_argument("--stitch", choices=["near", "window"], default="near",
                    help="窗口拼接方式，见 reconstruct_qpos 说明；window 为旧做法，仅供对照")
    ap.add_argument("--root-solve", choices=["legs", "none"], default="legs",
                    help="后处理：用 mink 让 root 与重建关节自洽（见 tools_local/mink_root_solve.py）。"
                         "legs 为默认；none 保留拼接的源动作 root，仅供对照")
    args = ap.parse_args()

    session = open_session(args.onnx, args.encoder)
    perm, mujoco_to_isaac = load_joint_maps(args.manifest)
    motion = load_robot_motion(args.motion)
    root_src = load_robot_motion(args.root_motion) if args.root_motion else motion

    qpos, joint_pos_mujoco, flips = reconstruct_qpos(
        session, motion, args.encoder, perm, mujoco_to_isaac, root_src, stitch=args.stitch
    )
    total_frames = qpos.shape[0]
    window_span = (NUM_FUTURE_FRAMES - 1) * FUTURE_FRAME_STRIDE + 1

    if args.root_solve == "legs":
        # 后处理：mink 反解与重建关节自洽的 root（腿部允许 <1° 的修正）。目标取自源动作的
        # 支撑脚与 root，即 root_src（robot 链路就是输入本身，SMPL 链路是 --root-motion）。
        # 实测参考动作脚底打滑 -52~81%、腿部关节反而更接近真值；但它不减少 play 中
        # ghost 与机器人的世界系距离——那是 MimicLite 不追世界坐标所致，喂真值也一样。
        from mink_root_solve import solve_root

        src_quat = np.asarray(root_src["root_rot"], dtype=np.float32)[:, [3, 0, 1, 2]].copy()
        for i in range(1, len(src_quat)):
            if float(np.dot(src_quat[i], src_quat[i - 1])) < 0.0:
                src_quat[i] = -src_quat[i]
        source_qpos = np.concatenate([
            np.asarray(root_src["root_trans_offset"], dtype=np.float32), src_quat,
            np.asarray(root_src["dof"], dtype=np.float32)], axis=1)
        man = json.loads(args.manifest.read_text(encoding="utf-8"))
        mjcf = args.manifest.parent / man["mjcf"]  # 数据集自带的 FK 骨架，与 any4hdmi 一致
        qpos, st = solve_root(qpos, source_qpos, str(mjcf), mode="legs")
        print(f"root 自洽后处理: 腿部平均修正 {st['leg_change_deg']:.2f}°，root 平均偏移 {st['root_shift_m']:.3f} m")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, qpos=qpos)

    print(f"编码器        : {args.encoder}")
    print(f"输入动作      : {args.motion.name}")
    print(f"帧数          : {total_frames}   窗口数: {(total_frames + window_span - 1)//window_span}")
    print(f"四元数半球修正: {flips} 帧")
    print(f"输出          : {args.out}  shape={qpos.shape}")

    # 自检：和数据自带的真值关节角比一比，给出重建误差（两边都是 MuJoCo 顺序）
    if "dof" in motion:
        gt = np.asarray(motion["dof"], dtype=np.float32)[:total_frames]
        err = np.rad2deg(np.abs(gt - joint_pos_mujoco))
        print(f"关节重建误差  : 平均 {err.mean():.3f}°  p95 {np.percentile(err,95):.3f}°  最大 {err.max():.3f}°")


if __name__ == "__main__":
    main()
