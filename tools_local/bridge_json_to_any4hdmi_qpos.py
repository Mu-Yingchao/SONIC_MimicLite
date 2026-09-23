#!/usr/bin/env python3
"""把 SONIC 桥接导出的 bumi_deploy_motion_v1 JSON 转成 any4hdmi 的 qpos npz。

用途：让 MimicLite-BUMI2 的 play.py / 训练框架能直接读我们自己从 SONIC token
重建出来的参考动作，在本地做可视化和追踪验证。

any4hdmi 的 qpos 布局（已用同事真实训练数据逐值核对过）：

    qpos[t] = [root_pos_w(3), root_quat_w_wxyz(4), joint_pos_MuJoCo顺序(21)]

关节顺序有两套，不能混：

  - 桥接导出 JSON 里的 joint_pos 用的是 IsaacLab 顺序，也就是数据集 manifest 里
    的 source_joint_names。
  - qpos 尾部 21 个值用的是 MuJoCo 顺序，也就是 manifest 里的
    target_hinge_joint_names。

本脚本默认从 manifest.json 现读这两个列表来推导重排索引，而不是写死一份
硬编码表——manifest 是数据集自己带的权威记录，机器人定义变了它会跟着变。

反向转换（原始 any4hdmi 源 npz -> qpos npz）也在这里，用 --from-raw，方便把
Noetix-9 上 /data0/bumi_v2_filtered 里的原始动作转成同格式做 A/B 对照。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_joint_permutation(manifest_path: Path) -> list[int]:
    """从数据集 manifest 推导 IsaacLab 顺序 -> MuJoCo 顺序的重排索引。"""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest["source"]
    isaaclab_order = source["source_joint_names"]
    mujoco_order = source["target_hinge_joint_names"]
    if sorted(isaaclab_order) != sorted(mujoco_order):
        raise ValueError("manifest 里两套关节名集合不一致，无法建立重排关系")
    return [isaaclab_order.index(name) for name in mujoco_order]


def enforce_quaternion_hemisphere(quat_wxyz: np.ndarray) -> tuple[np.ndarray, int]:
    """强制相邻帧四元数同半球，返回 (修正后数组, 翻转了多少帧)。

    q 和 -q 是同一个旋转，但符号在帧之间跳变会让任何 slerp/插值沿"长路"绕行，
    插出错误的中间姿态。any4hdmi 建 FK cache 时的重采样、真机部署端读 JSON 后的
    平滑都会踩这个坑，所以写文件之前必须对齐。
    """
    aligned = quat_wxyz.copy()
    flipped = 0
    for idx in range(1, aligned.shape[0]):
        if float(np.dot(aligned[idx], aligned[idx - 1])) < 0.0:
            aligned[idx] = -aligned[idx]
            flipped += 1
    return aligned, flipped


def qpos_from_bridge_json(json_path: Path, perm: list[int]) -> np.ndarray:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    if metadata.get("quaternion_order") != "wxyz":
        raise ValueError(
            f"只支持 wxyz 四元数，JSON 里写的是 {metadata.get('quaternion_order')!r}"
        )

    root_pos = np.asarray(payload["root_pos_w"], dtype=np.float32)
    root_quat = np.asarray(payload["root_quat_w"], dtype=np.float32)
    joint_pos_isaaclab = np.asarray(payload["joint_pos"], dtype=np.float32)

    if joint_pos_isaaclab.shape[1] != len(perm):
        raise ValueError(
            f"JSON 里有 {joint_pos_isaaclab.shape[1]} 个关节，manifest 期望 {len(perm)} 个"
        )

    root_quat, flipped = enforce_quaternion_hemisphere(root_quat)
    if flipped:
        print(f"  [四元数半球对齐] 修正了 {flipped} 帧的符号")

    return np.concatenate(
        [root_pos, root_quat, joint_pos_isaaclab[:, perm]], axis=1
    ).astype(np.float32)


def qpos_from_raw_npz(npz_path: Path, perm: list[int]) -> np.ndarray:
    """原始 any4hdmi 源 npz -> qpos，配方来自 manifest 的 source.qpos_from 字段：
    body_pos_w[:,0] + body_quat_w[:,0] + reordered joint_pos。
    """
    raw = np.load(npz_path)
    root_pos = raw["body_pos_w"][:, 0].astype(np.float32)
    root_quat = raw["body_quat_w"][:, 0].astype(np.float32)
    joint_pos_isaaclab = raw["joint_pos"].astype(np.float32)

    root_quat, flipped = enforce_quaternion_hemisphere(root_quat)
    if flipped:
        print(f"  [四元数半球对齐] 修正了 {flipped} 帧的符号")

    return np.concatenate(
        [root_pos, root_quat, joint_pos_isaaclab[:, perm]], axis=1
    ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="桥接导出的 JSON，或加 --from-raw 时是原始 any4hdmi npz")
    parser.add_argument("output", type=Path, help="输出的 qpos npz 路径")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="any4hdmi 数据集的 manifest.json，用来推导关节重排顺序",
    )
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help="输入是原始 any4hdmi 源 npz（含 body_pos_w/body_quat_w/joint_pos），不是桥接 JSON",
    )
    args = parser.parse_args()

    perm = load_joint_permutation(args.manifest)
    print(f"关节重排索引（IsaacLab -> MuJoCo）: {perm}")

    if args.from_raw:
        qpos = qpos_from_raw_npz(args.input, perm)
    else:
        qpos = qpos_from_bridge_json(args.input, perm)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, qpos=qpos)
    print(f"已写出 {args.output}  shape={qpos.shape}  dtype={qpos.dtype}")


if __name__ == "__main__":
    main()
