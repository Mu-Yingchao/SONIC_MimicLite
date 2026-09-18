#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""把同事已重定向好的 BUMI2 机器人动作（npz）和已对齐筛选好的 BUMI3 SMPL PKL
配对，整理成 SONIC 训练能直接读取的 motion-lib 格式。

本工具与 ``prepare_bumi3_sonic_dataset.py`` 的定位不同：BUMI3 那份脚本绑定的是
一个特定上游 ``genmo.bumi_music.v1`` ``.pt``+manifest 中间契约（30Hz、legacy
Y-up、需要在当前 MJCF 下重新做足底穿地优化），这个中间契约本身的生产过程不在
本仓库、也不是我们现在手上的数据形态。本工具的实际输入是两份已经核实过真实
结构的数据（结构核实过程见 ``sonic_mimiclite_new.md`` §8 对应记录，不是假设）：

- Robot：``.../bumi_v2_filtered/**/*_from_g1_bumi_v2.npz``，字段为
  ``joint_pos``/``joint_vel`` ``[T,21]``、``body_pos_w``/``body_quat_w``
  ``[T,22,3或4]``、``fps``。经实测确认：``joint_pos`` 是 Isaac Lab 关节顺序
  （用已知关节限位边界值 ``l_arm_roll_joint`` 下限 ``-0.0872``、
  ``r_arm_roll_joint`` 上限 ``0.0872`` 精确核对过第 9/10 列）；``body_quat_w``
  是 Isaac Lab 惯用的 scalar-first ``wxyz``；根高度范围（约 0.46～0.47m）与
  BUMI3 sonic_bumi2.yaml 沿用的参考根高一致；已经是 50Hz，不需要重采样；数据
  直接来自 Isaac Lab rollout（不是原始动作捕捉重定向），不存在需要足底穿地
  优化修正的坐标系问题。
- SMPL：``.../bumi3_smpl_filtered/smpl/*.pkl``，是 BUMI3 SONIC 训练已经在用的
  **最终态**数据（字段已经是 ``pose_aa[T,72]``/``transl[T,3]``/
  ``smpl_joints[T,24,3]``/``fps=50.0``），不需要任何转换，只需要按文件名和
  Robot 配对、核对帧数严格一致。抽样核对 97,660 条里 93,822 条（96.1%）能按
  去掉 ``_from_g1_bumi_v2`` 后缀的文件名直接配对，抽样的 ``Idle_Left_001__A017``
  两边帧数都是 3827，完全一致。

Robot 侧转换成 SONIC 目标契约（``root_trans_offset``/``pose_aa``/``dof``/
``root_rot``/``fps``）的具体做法：

- ``root_trans_offset`` = ``body_pos_w[:,0,:]``（根 body 即 ``base_link``）。
- ``root_rot``（xyzw 存盘，和 BUMI3 脚本存盘约定一致）=
  ``body_quat_w[:,0,:]`` 从 wxyz 转 xyzw。
- ``dof`` = ``joint_pos`` 按 ``BUMI2_ISAACLAB_TO_MUJOCO_DOF`` 重排到 MuJoCo
  actuator 顺序（SONIC 训练端按 MJCF actuator 顺序消费 dof）。
- ``pose_aa``：根用 root 四元数转 axis-angle；21 个关节体各自是单自由度
  hinge，局部旋转就是"关节轴 × 关节角"，用当前 BUMI2 MJCF 逐 body 解析出的
  ``joint_axis`` 和上面重排后的 ``dof`` 直接算，不依赖 ``body_quat_w`` 的其余
  21 个 body（避免额外引入父子 body 相对姿态换算的复杂度和潜在出错点）。

典型用法：

    python gear_sonic/tools/prepare_bumi2_sonic_dataset.py build \\
      --robot-root /data0/bumi_v2_filtered \\
      --smpl-root /data0/bumi3_smpl_filtered/smpl \\
      --output-root /data0/bumi2_sonic_dataset_v1 \\
      --workers 16

    python gear_sonic/tools/prepare_bumi2_sonic_dataset.py validate \\
      --output-root /data0/bumi2_sonic_dataset_v1
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any
import xml.etree.ElementTree as ET

import joblib
import numpy as np
import torch


TARGET_FPS = 50.0
ROBOT_SUFFIX = "_from_g1_bumi_v2.npz"


@dataclass(frozen=True)
class SampleRecord:
    """一段 BUMI2 训练动作的来源、配对关系。"""

    key: str
    robot_source: str
    smpl_source: str | None
    num_frames: int
    fps: float
    paired: bool


@dataclass(frozen=True)
class MjcfContract:
    """当前 BUMI2 MJCF 中与 motion-lib 顺序有关的最小稳定契约。"""

    actuator_joint_names: tuple[str, ...]
    body_joint_names: tuple[str, ...]
    body_joint_axes: tuple[tuple[float, float, float], ...]
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_mjcf(path: Path) -> MjcfContract:
    """与 prepare_bumi3_sonic_dataset.py 的 _parse_mjcf 逻辑一致，独立实现避免跨脚本耦合。"""

    tree = ET.parse(path)
    root = tree.getroot()
    world_root = root.find("worldbody/body")
    if world_root is None:
        raise ValueError(f"MJCF 缺少 worldbody 根 body: {path}")

    body_joint_names: list[str] = []
    body_joint_axes: list[tuple[float, float, float]] = []

    def visit(body: ET.Element) -> None:
        joints = [joint for joint in body.findall("joint") if joint.get("type") != "free"]
        if len(joints) > 1:
            raise ValueError(f"BUMI2 body {body.get('name')} 含多个非 free joint")
        if joints:
            joint = joints[0]
            name = joint.get("name")
            axis_text = joint.get("axis")
            if name is None or axis_text is None:
                raise ValueError(f"BUMI2 joint 缺少 name/axis: {ET.tostring(joint)}")
            axis = tuple(float(value) for value in axis_text.split())
            if len(axis) != 3 or not math.isclose(sum(v * v for v in axis), 1.0, abs_tol=1e-6):
                raise ValueError(f"BUMI2 joint {name} axis 非单位三维向量: {axis}")
            body_joint_names.append(name)
            body_joint_axes.append(axis)
        for child in body.findall("body"):
            visit(child)

    visit(world_root)
    if len(body_joint_names) != 21 or len(set(body_joint_names)) != 21:
        raise ValueError(f"BUMI2 MJCF 必须是 21 个唯一 body joint，得到 {len(body_joint_names)}")
    # 本仓库的 bumi2.xml 是纯运动学导出（来自同事 tools/generate_mjcf.py），
    # 没有 <actuator> 段——PD/执行器定义在 gear_sonic/envs/manager_env/robots/
    # bumi2.py 里单独维护，不依赖 MJCF 里的 actuator 标签。这里直接把 body
    # joint 在文件中出现的顺序（即 MuJoCo qpos/dof 的天然顺序）当作 actuator
    # 顺序使用，和 BUMI3 脚本里"actuator 顺序=MJCF actuator 段顺序"是同一件事，
    # 只是本文件没有 actuator 段可读，只能退回 body 遍历顺序。
    return MjcfContract(
        actuator_joint_names=tuple(body_joint_names),
        body_joint_names=tuple(body_joint_names),
        body_joint_axes=tuple(body_joint_axes),
        sha256=_sha256(path),
    )


def _collect_robot_records(
    robot_root: Path, smpl_root: Path, *, scan_limit: int | None = None
) -> list[SampleRecord]:
    """扫描 BUMI2 npz，按去后缀文件名和已有 SMPL PKL 做名称配对与帧数校验。

    帧数不一致的配对**不做插值/裁剪凑数**，直接降级为 robot-only（不是报错
    终止整个 build），因为这批数据不是本工具自己生产的，无法回头改上游；
    降级会被计入返回记录并在 manifest 里如实记录，不会被静默丢弃。

    ``scan_limit``：只用于 ``--limit`` 小样本抽查模式提前截断扫描本身（完整
    扫描 11 万+文件每条都要打开一次 npz/pkl 读帧数，本身就要跑很久，不能等
    扫完全量再截断转换数量），正式 build（``scan_limit=None``）忽略该参数，
    永远做全量扫描。
    """

    records: list[SampleRecord] = []
    npz_paths = sorted(robot_root.rglob(f"*{ROBOT_SUFFIX}"))
    if not npz_paths:
        raise FileNotFoundError(f"{robot_root} 下没有找到 *{ROBOT_SUFFIX} 文件")
    if scan_limit is not None:
        npz_paths = npz_paths[:scan_limit]
    for npz_path in npz_paths:
        key = npz_path.name[: -len(ROBOT_SUFFIX)]
        smpl_path = smpl_root / f"{key}.pkl"
        with np.load(npz_path, allow_pickle=True) as data:
            num_frames = int(data["joint_pos"].shape[0])
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        paired = False
        if smpl_path.is_file():
            smpl_payload = joblib.load(smpl_path)
            smpl_frames = int(np.asarray(smpl_payload["pose_aa"]).shape[0])
            smpl_fps = float(smpl_payload["fps"])
            if smpl_frames == num_frames and math.isclose(smpl_fps, TARGET_FPS, abs_tol=1e-6):
                paired = True
            # 帧数或 fps 不一致：不报错、不强行对齐，按 robot-only 处理，
            # 具体差异值不在这里打印（数量可能很大），完整统计留给 build() 的汇总。
        records.append(
            SampleRecord(
                key=key,
                robot_source=str(npz_path.resolve()),
                smpl_source=str(smpl_path.resolve()) if paired else None,
                num_frames=num_frames,
                fps=fps,
                paired=paired,
            )
        )
    duplicate_keys = {r.key for r in records if sum(1 for x in records if x.key == r.key) > 1}
    if duplicate_keys:
        raise ValueError(f"存在重复 key（不同子目录下同名文件）: {sorted(duplicate_keys)[:10]}")
    return records


def _convert_robot_job(args: tuple[SampleRecord, MjcfContract, str]) -> str:
    record, contract, output_dir_text = args
    torch.set_num_threads(1)
    source = Path(record.robot_source)
    with np.load(source, allow_pickle=True) as data:
        joint_pos = torch.as_tensor(np.asarray(data["joint_pos"]), dtype=torch.float32)
        body_pos_w = torch.as_tensor(np.asarray(data["body_pos_w"]), dtype=torch.float32)
        body_quat_w = torch.as_tensor(np.asarray(data["body_quat_w"]), dtype=torch.float32)
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])

    if joint_pos.shape != (record.num_frames, 21):
        raise ValueError(f"{source} joint_pos 形状错误: {tuple(joint_pos.shape)}")
    if body_pos_w.shape != (record.num_frames, 22, 3):
        raise ValueError(f"{source} body_pos_w 形状错误: {tuple(body_pos_w.shape)}")
    if body_quat_w.shape != (record.num_frames, 22, 4):
        raise ValueError(f"{source} body_quat_w 形状错误: {tuple(body_quat_w.shape)}")
    if not torch.isfinite(joint_pos).all() or not torch.isfinite(body_pos_w).all():
        raise ValueError(f"{source} 含 NaN/Inf")
    if not math.isclose(fps, TARGET_FPS, abs_tol=1e-6):
        raise ValueError(f"{source} fps 必须为 50，实际 {fps}")

    # joint_pos 已实测确认是 Isaac Lab 关节顺序，重排到 MJCF actuator（MuJoCo）顺序。
    # 故意不 import gear_sonic.envs.manager_env.robots.bumi2：该模块在文件顶部
    # `import isaaclab.sim`，而 isaaclab 的 pxr/USD 绑定只有在 Isaac Lab 的
    # AppLauncher 先启动一个 SimulationApp 之后才能被导入（实测确认：单独
    # `import isaacsim`/`import pxr` 都失败，报 `ModuleNotFoundError: No module
    # named 'pxr'`）。离线数据转换脚本不需要、也不应该为了读几个常量数组去启动
    # 一整个 Isaac Sim 进程。这里直接内联同一份 Isaac Lab 关节顺序表，和
    # bumi2.py 顶部注释里的表逐项相同；BUMI3 的 prepare_bumi3_sonic_dataset.py
    # 也是同样处理方式（自己独立解析 MJCF，不 import bumi3.py）。
    isaaclab_dof_names = (
        "l_leg_pitch_joint", "r_leg_pitch_joint", "waist_yaw_joint",
        "l_leg_roll_joint", "r_leg_roll_joint", "l_arm_pitch_joint", "r_arm_pitch_joint",
        "l_leg_yaw_joint", "r_leg_yaw_joint", "l_arm_roll_joint", "r_arm_roll_joint",
        "l_knee_pitch_joint", "r_knee_pitch_joint", "l_arm_yaw_joint", "r_arm_yaw_joint",
        "l_ankle_pitch_joint", "r_ankle_pitch_joint", "l_elbow_pitch_joint", "r_elbow_pitch_joint",
        "l_ankle_roll_joint", "r_ankle_roll_joint",
    )
    if len(isaaclab_dof_names) != 21 or set(isaaclab_dof_names) != set(contract.actuator_joint_names):
        raise ValueError("内联的 Isaac Lab 关节顺序表和当前 MJCF 关节集合不一致")
    # gather_idx[j] = 目标 mujoco 顺序第 j 个关节名，在源 joint_pos（isaaclab
    # 顺序）里对应的列号；dof[:, j] = joint_pos[:, gather_idx[j]]。
    isaaclab_index_of = {name: i for i, name in enumerate(isaaclab_dof_names)}
    gather_idx = torch.as_tensor(
        [isaaclab_index_of[name] for name in contract.actuator_joint_names], dtype=torch.long
    )
    dof = joint_pos[:, gather_idx]

    root_quat_wxyz = body_quat_w[:, 0, :]
    norms = torch.linalg.vector_norm(root_quat_wxyz, dim=1, keepdim=True)
    if bool((norms < 1e-8).any()):
        raise ValueError(f"{source} 含零范数 root quaternion")
    root_quat_wxyz = root_quat_wxyz / norms
    root_trans_offset = body_pos_w[:, 0, :].clone()

    from gear_sonic.trl.utils.torch_transform import quaternion_to_angle_axis

    root_axis_angle = quaternion_to_angle_axis(root_quat_wxyz)
    pose_aa = torch.zeros((record.num_frames, 22, 3), dtype=torch.float32)
    pose_aa[:, 0] = root_axis_angle
    actuator_index = {name: index for index, name in enumerate(contract.actuator_joint_names)}
    for body_index, (joint_name, axis) in enumerate(
        zip(contract.body_joint_names, contract.body_joint_axes, strict=True), start=1
    ):
        pose_aa[:, body_index] = (
            torch.tensor(axis, dtype=torch.float32) * dof[:, actuator_index[joint_name], None]
        )

    entry = {
        "root_trans_offset": root_trans_offset.numpy().astype(np.float32, copy=False),
        "pose_aa": pose_aa.numpy().astype(np.float32, copy=False),
        "dof": dof.numpy().astype(np.float32, copy=False),
        "root_rot": root_quat_wxyz[:, [1, 2, 3, 0]].numpy().astype(np.float32, copy=False),
        "fps": 50,
        "source_npz": str(source),
    }
    output = Path(output_dir_text) / f"{record.key}.pkl"
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    joblib.dump({record.key: entry}, temporary, compress=3)
    os.replace(temporary, output)
    return record.key


def _copy_smpl_job(args: tuple[SampleRecord, str]) -> str:
    """SMPL 侧已经是最终格式，直接原样复制到输出目录，不做任何数值改动。"""

    record, output_dir_text = args
    source = Path(record.smpl_source or "")
    payload = joblib.load(source)
    for field, shape_tail in (("pose_aa", (72,)), ("transl", (3,)), ("smpl_joints", (24, 3))):
        value = np.asarray(payload[field])
        if value.shape != (record.num_frames, *shape_tail) or value.dtype != np.float32:
            raise ValueError(f"{source} {field} 契约错误: {value.shape} {value.dtype}")
        if not np.isfinite(value).all():
            raise ValueError(f"{source} {field} 含 NaN/Inf")
    output = Path(output_dir_text) / f"{record.key}.pkl"
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    joblib.dump(
        {
            "pose_aa": payload["pose_aa"],
            "transl": payload["transl"],
            "smpl_joints": payload["smpl_joints"],
            "fps": 50.0,
        },
        temporary,
        compress=3,
    )
    os.replace(temporary, output)
    return record.key


def _run_jobs(label: str, worker: Any, jobs: list[Any], workers: int) -> None:
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for _ in executor.map(worker, jobs, chunksize=4):
            completed += 1
            if completed == len(jobs) or completed % 500 == 0:
                print(f"[{label}] {completed}/{len(jobs)}", flush=True)


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={repo_root}", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    built_root = output_root / "built"
    robot_final = built_root / "robot_all"
    smpl_final = built_root / "smpl_all"
    robot_staging = built_root / ".robot_all.staging"
    smpl_staging = built_root / ".smpl_all.staging"
    if robot_final.exists() or smpl_final.exists():
        raise FileExistsError("最终 robot_all/smpl_all 已存在；为防止覆盖，本工具拒绝重建")
    built_root.mkdir(parents=True, exist_ok=True)
    robot_staging.mkdir(exist_ok=True)
    smpl_staging.mkdir(exist_ok=True)

    contract = _parse_mjcf(args.mjcf.resolve())
    records = _collect_robot_records(
        args.robot_root.resolve(), args.smpl_root.resolve(), scan_limit=args.limit
    )
    paired_count = sum(r.paired for r in records)
    print(
        f"DATASET_CONTRACT robot={len(records)} paired={paired_count} "
        f"robot_only={len(records) - paired_count}"
    )
    if args.limit is not None:
        print(f"[build] --limit 生效，只扫描并处理前 {len(records)} 条用于抽样验证")

    robot_jobs = [
        (record, contract, str(robot_staging))
        for record in records
        if not (robot_staging / f"{record.key}.pkl").exists()
    ]
    _run_jobs("robot", _convert_robot_job, robot_jobs, args.workers)
    smpl_jobs = [
        (record, str(smpl_staging))
        for record in records
        if record.paired and not (smpl_staging / f"{record.key}.pkl").exists()
    ]
    _run_jobs("smpl", _copy_smpl_job, smpl_jobs, args.workers)

    if args.limit is not None:
        print("[build] --limit 模式：跳过原子发布，产物留在 staging 目录供人工核查")
        print(f"  robot staging: {robot_staging}")
        print(f"  smpl staging:  {smpl_staging}")
        return

    os.replace(robot_staging, robot_final)
    os.replace(smpl_staging, smpl_final)

    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = meta_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            row = asdict(record)
            row["robot_file"] = f"built/robot_all/{record.key}.pkl"
            row["smpl_file"] = f"built/smpl_all/{record.key}.pkl" if record.paired else None
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    provenance = {
        "contract_version": "sonic.bumi2_v1",
        "repo_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "mjcf_sha256": contract.sha256,
        "robot_root": str(args.robot_root.resolve()),
        "smpl_root": str(args.smpl_root.resolve()),
        "robot_count": len(records),
        "paired_count": paired_count,
        "robot_only_count": len(records) - paired_count,
        "target_fps": TARGET_FPS,
    }
    (meta_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("BUMI2_SONIC_DATASET_BUILD=PASS")
    print(f"最终计数：robot={len(records)}, paired={paired_count}, robot_only={len(records)-paired_count}")


def validate(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    manifest = output_root / "meta" / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    robot_dir = output_root / "built" / "robot_all"
    smpl_dir = output_root / "built" / "smpl_all"
    count = 0
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            robot_path = robot_dir / f"{row['key']}.pkl"
            if not robot_path.is_file():
                raise FileNotFoundError(robot_path)
            payload = joblib.load(robot_path)[row["key"]]
            expected_shapes = {
                "root_trans_offset": (row["num_frames"], 3),
                "pose_aa": (row["num_frames"], 22, 3),
                "dof": (row["num_frames"], 21),
                "root_rot": (row["num_frames"], 4),
            }
            for field, shape in expected_shapes.items():
                value = np.asarray(payload[field])
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"{row['key']} {field} 校验失败: {value.shape}")
            if row["paired"]:
                smpl_path = smpl_dir / f"{row['key']}.pkl"
                if not smpl_path.is_file():
                    raise FileNotFoundError(smpl_path)
                smpl_payload = joblib.load(smpl_path)
                if np.asarray(smpl_payload["pose_aa"]).shape[0] != row["num_frames"]:
                    raise ValueError(f"{row['key']} SMPL 帧数与 robot 不一致")
            count += 1
    print(f"BUMI2_SONIC_DATASET_VALIDATE=PASS ({count} 条)")


def _parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="转换、校验并发布 BUMI2 训练数据集")
    build_parser.add_argument("--robot-root", type=Path, required=True)
    build_parser.add_argument("--smpl-root", type=Path, required=True)
    build_parser.add_argument("--output-root", type=Path, required=True)
    build_parser.add_argument(
        "--mjcf",
        type=Path,
        default=repo_root / "gear_sonic/data/assets/robot_description/mjcf/bumi2.xml",
    )
    build_parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    build_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只处理前 N 条并跳过原子发布，用于小样本抽查转换是否正确",
    )
    build_parser.set_defaults(func=build)

    validate_parser = subparsers.add_parser("validate", help="重新校验已发布数据")
    validate_parser.add_argument("--output-root", type=Path, required=True)
    validate_parser.set_defaults(func=validate)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if getattr(args, "workers", 1) < 1:
        raise ValueError("--workers 必须大于 0")
    args.func(args)


if __name__ == "__main__":
    main()
