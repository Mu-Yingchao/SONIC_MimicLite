"""
Play and export policy for mimic-lite project.

This script provides the mimic-lite policy export behavior:
- export traced policy as .pt
- export ONNX as .onnx
- export deploy config as .yaml
"""

from __future__ import annotations

import atexit
import copy
import csv
import datetime
import itertools
import os
import re
import secrets
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm
import yaml

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torchrl.envs.transforms import VecNorm as TorchRLVecNorm
from torchrl.envs.utils import ExplorationType, set_exploration_type
from active_adaptation.utils.profiling import ScopedTimer

import active_adaptation as aa
from active_adaptation.learning.modules.vecnorm import VecNorm
from active_adaptation.utils.export import export_onnx
from active_adaptation.utils.helpers import EpisodeStats
from active_adaptation.utils.timerfd import Timer
from active_adaptation.utils.wandb import parse_checkpoint_path

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase

FILE_PATH = Path(__file__).resolve().parent
CONFIG_PATH = FILE_PATH.parent / "cfg"


def _get_asset_meta(asset) -> dict:
    meta = {
        "joint_names": list(getattr(asset, "joint_names", [])),
        "joint_kp": {},
        "joint_kd": {},
        "default_joint_pos": {},
    }

    cfg = getattr(asset, "cfg", None)
    init_state = getattr(cfg, "init_state", None)
    if init_state is not None and hasattr(init_state, "joint_pos"):
        joint_pos = init_state.joint_pos
        if isinstance(joint_pos, dict):
            meta["default_joint_pos"] = dict(joint_pos)

    actuators = getattr(asset, "actuators", [])
    for actuator in actuators:
        acfg = getattr(actuator, "cfg", None)
        if acfg is None:
            continue

        names = (
            getattr(acfg, "target_names_expr", None)
            or getattr(acfg, "joint_names_expr", None)
            or []
        )
        stiffness = getattr(acfg, "stiffness", None)
        damping = getattr(acfg, "damping", None)

        for joint_name in names:
            if stiffness is not None:
                meta["joint_kp"][joint_name] = float(stiffness)
            if damping is not None:
                meta["joint_kd"][joint_name] = float(damping)

    return meta


def _checkpoint_tags(checkpoint_path: str | None) -> tuple[str, str]:
    wandb_run_id = "unknown"
    checkpoint_num = "unknown"

    if checkpoint_path is None:
        return wandb_run_id, checkpoint_num

    state_dict = torch.load(checkpoint_path, weights_only=False)
    if "wandb" in state_dict and "id" in state_dict["wandb"]:
        wandb_run_id = state_dict["wandb"]["id"]

    filename = os.path.basename(checkpoint_path)
    match = re.search(r"checkpoint_(\d+)", filename)
    if match:
        checkpoint_num = match.group(1)
    elif filename.endswith("_final.pt"):
        checkpoint_num = "final"

    return wandb_run_id, checkpoint_num


def _make_render_output_path() -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = secrets.token_hex(4)
    return Path.cwd() / f"{timestamp}-{suffix}.mp4"


def _quat_wxyz_to_rpy(quat: np.ndarray) -> np.ndarray:
    """Convert normalized wxyz quaternions to roll, pitch, yaw in radians."""
    quat = quat / np.clip(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-12, None)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack((roll, pitch, yaw), axis=-1)


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _safe_column_name(name: object) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")


def _save_trajectory_log(
    path: Path,
    metadata: dict[str, np.ndarray],
    trajectory: dict[str, list[np.ndarray]],
) -> None:
    """Save full-fidelity NPZ plus a human-readable per-step CSV and summary."""
    if not trajectory.get("control_step"):
        print("Trajectory log is empty; nothing to save.")
        return

    path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.stack(samples, axis=0) for name, samples in trajectory.items()}

    root_pos_error_xyz = arrays["root_pos"] - arrays["reference_root_pos"]
    arrays["root_pos_error_xyz"] = root_pos_error_xyz
    arrays["root_pos_error"] = np.linalg.norm(root_pos_error_xyz, axis=-1)
    arrays["root_xy_error"] = np.linalg.norm(root_pos_error_xyz[..., :2], axis=-1)

    root_trajectory_error_xyz = (
        arrays["root_pos"] - arrays["root_pos"][:1]
    ) - (
        arrays["reference_root_pos"] - arrays["reference_root_pos"][:1]
    )
    arrays["root_trajectory_error_xyz"] = root_trajectory_error_xyz
    arrays["root_trajectory_error"] = np.linalg.norm(
        root_trajectory_error_xyz, axis=-1
    )
    arrays["root_trajectory_xy_error"] = np.linalg.norm(
        root_trajectory_error_xyz[..., :2], axis=-1
    )

    reference_root_rpy = _quat_wxyz_to_rpy(arrays["reference_root_quat"])
    root_rpy = _quat_wxyz_to_rpy(arrays["root_quat"])
    arrays["reference_root_rpy"] = reference_root_rpy
    arrays["root_rpy"] = root_rpy
    arrays["root_rpy_error"] = _wrap_to_pi(root_rpy - reference_root_rpy)
    quat_dot = np.abs(
        np.sum(arrays["reference_root_quat"] * arrays["root_quat"], axis=-1)
    )
    arrays["root_ori_error"] = 2.0 * np.arccos(np.clip(quat_dot, 0.0, 1.0))

    np.savez_compressed(path, **metadata, **arrays)

    body_names = [_safe_column_name(v) for v in metadata["tracking_body_names"]]
    joint_names = [_safe_column_name(v) for v in metadata["tracking_joint_names"]]
    fixed_columns = [
        "control_step", "time_s", "env_id", "motion_id", "motion_frame",
        "is_standing", "reference_root_x", "reference_root_y", "reference_root_z",
        "root_x", "root_y", "root_z", "root_pos_error_x", "root_pos_error_y",
        "root_pos_error_z", "root_pos_error_m", "root_xy_error_m",
        "root_trajectory_error_x", "root_trajectory_error_y",
        "root_trajectory_error_z", "root_trajectory_error_m",
        "root_trajectory_xy_error_m", "reference_root_roll_rad",
        "reference_root_pitch_rad", "reference_root_yaw_rad", "root_roll_rad",
        "root_pitch_rad", "root_yaw_rad", "root_roll_error_rad",
        "root_pitch_error_rad", "root_yaw_error_rad", "root_ori_error_rad",
    ]
    body_columns = [
        f"{body}_{suffix}"
        for body in body_names
        for suffix in (
            "pos_error_w_m", "pos_error_local_m", "ori_error_w_rad",
            "ori_error_local_rad", "lin_vel_error_m_s", "ang_vel_error_rad_s",
        )
    ]
    joint_columns = [
        f"{joint}_{suffix}"
        for joint in joint_names
        for suffix in ("pos_error_rad", "vel_error_rad_s")
    ]
    csv_path = path.with_suffix(".csv")
    step_dt = float(np.asarray(metadata["step_dt_s"]))
    env_ids = np.asarray(metadata["env_ids"])
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=fixed_columns + body_columns + joint_columns
        )
        writer.writeheader()
        for step_idx, control_step in enumerate(arrays["control_step"]):
            for env_idx, env_id in enumerate(env_ids):
                ref_pos = arrays["reference_root_pos"][step_idx, env_idx]
                pos = arrays["root_pos"][step_idx, env_idx]
                pos_error = arrays["root_pos_error_xyz"][step_idx, env_idx]
                trajectory_error = arrays["root_trajectory_error_xyz"][step_idx, env_idx]
                ref_rpy = arrays["reference_root_rpy"][step_idx, env_idx]
                rpy = arrays["root_rpy"][step_idx, env_idx]
                rpy_error = arrays["root_rpy_error"][step_idx, env_idx]
                row = {
                    "control_step": int(control_step),
                    "time_s": float(control_step) * step_dt,
                    "env_id": int(env_id),
                    "motion_id": int(arrays["motion_id"][step_idx, env_idx]),
                    "motion_frame": int(arrays["motion_frame"][step_idx, env_idx]),
                    "is_standing": bool(arrays["is_standing"][step_idx, env_idx]),
                    "reference_root_x": float(ref_pos[0]),
                    "reference_root_y": float(ref_pos[1]),
                    "reference_root_z": float(ref_pos[2]),
                    "root_x": float(pos[0]), "root_y": float(pos[1]),
                    "root_z": float(pos[2]),
                    "root_pos_error_x": float(pos_error[0]),
                    "root_pos_error_y": float(pos_error[1]),
                    "root_pos_error_z": float(pos_error[2]),
                    "root_pos_error_m": float(arrays["root_pos_error"][step_idx, env_idx]),
                    "root_xy_error_m": float(arrays["root_xy_error"][step_idx, env_idx]),
                    "root_trajectory_error_x": float(trajectory_error[0]),
                    "root_trajectory_error_y": float(trajectory_error[1]),
                    "root_trajectory_error_z": float(trajectory_error[2]),
                    "root_trajectory_error_m": float(
                        arrays["root_trajectory_error"][step_idx, env_idx]
                    ),
                    "root_trajectory_xy_error_m": float(
                        arrays["root_trajectory_xy_error"][step_idx, env_idx]
                    ),
                    "reference_root_roll_rad": float(ref_rpy[0]),
                    "reference_root_pitch_rad": float(ref_rpy[1]),
                    "reference_root_yaw_rad": float(ref_rpy[2]),
                    "root_roll_rad": float(rpy[0]), "root_pitch_rad": float(rpy[1]),
                    "root_yaw_rad": float(rpy[2]),
                    "root_roll_error_rad": float(rpy_error[0]),
                    "root_pitch_error_rad": float(rpy_error[1]),
                    "root_yaw_error_rad": float(rpy_error[2]),
                    "root_ori_error_rad": float(arrays["root_ori_error"][step_idx, env_idx]),
                }
                for body_idx, body in enumerate(body_names):
                    row[f"{body}_pos_error_w_m"] = float(
                        arrays["body_pos_error"][step_idx, env_idx, body_idx]
                    )
                    row[f"{body}_pos_error_local_m"] = float(
                        arrays["body_pos_error_local"][step_idx, env_idx, body_idx]
                    )
                    row[f"{body}_ori_error_w_rad"] = float(
                        arrays["body_ori_error"][step_idx, env_idx, body_idx]
                    )
                    row[f"{body}_ori_error_local_rad"] = float(
                        arrays["body_ori_error_local"][step_idx, env_idx, body_idx]
                    )
                    row[f"{body}_lin_vel_error_m_s"] = float(
                        arrays["body_lin_vel_error"][step_idx, env_idx, body_idx]
                    )
                    row[f"{body}_ang_vel_error_rad_s"] = float(
                        arrays["body_ang_vel_error"][step_idx, env_idx, body_idx]
                    )
                for joint_idx, joint in enumerate(joint_names):
                    row[f"{joint}_pos_error_rad"] = float(
                        arrays["joint_pos_error"][step_idx, env_idx, joint_idx]
                    )
                    row[f"{joint}_vel_error_rad_s"] = float(
                        arrays["joint_vel_error"][step_idx, env_idx, joint_idx]
                    )
                writer.writerow(row)

    summaries = []
    for env_idx, env_id in enumerate(env_ids):
        root_pos_error = arrays["root_pos_error"][:, env_idx]
        root_xy_error = arrays["root_xy_error"][:, env_idx]
        root_ori_error = arrays["root_ori_error"][:, env_idx]
        trajectory_error = arrays["root_trajectory_error"][:, env_idx]
        trajectory_xy_error = arrays["root_trajectory_xy_error"][:, env_idx]
        summaries.append({
            "env_id": int(env_id),
            "samples": int(len(arrays["control_step"])),
            "duration_s": float(len(arrays["control_step"]) * step_dt),
            "motion_ids": sorted({int(v) for v in arrays["motion_id"][:, env_idx]}),
            "root_pos_rmse_m": float(np.sqrt(np.mean(root_pos_error ** 2))),
            "root_xy_rmse_m": float(np.sqrt(np.mean(root_xy_error ** 2))),
            "root_height_rmse_m": float(
                np.sqrt(np.mean(arrays["root_pos_error_xyz"][:, env_idx, 2] ** 2))
            ),
            "root_ori_rmse_rad": float(np.sqrt(np.mean(root_ori_error ** 2))),
            "root_ori_rmse_deg": float(np.degrees(np.sqrt(np.mean(root_ori_error ** 2)))),
            "root_pitch_error_rmse_rad": float(
                np.sqrt(np.mean(arrays["root_rpy_error"][:, env_idx, 1] ** 2))
            ),
            "root_pitch_error_rmse_deg": float(
                np.degrees(
                    np.sqrt(np.mean(arrays["root_rpy_error"][:, env_idx, 1] ** 2))
                )
            ),
            "root_trajectory_rmse_m": float(np.sqrt(np.mean(trajectory_error ** 2))),
            "root_trajectory_xy_rmse_m": float(np.sqrt(np.mean(trajectory_xy_error ** 2))),
            "root_trajectory_final_error_m": float(trajectory_error[-1]),
            "root_trajectory_final_xy_error_m": float(trajectory_xy_error[-1]),
        })
    summary_path = path.with_suffix(".summary.yaml")
    with summary_path.open("w") as summary_file:
        yaml.safe_dump({"environments": summaries}, summary_file, sort_keys=False)

    print(f"Saved trajectory NPZ to {path}")
    print(f"Saved trajectory CSV to {csv_path}")
    print(f"Saved trajectory summary to {summary_path}")



@VecNorm.freeze()
def export_policy(cfg: DictConfig, env: "_EnvBase", policy) -> None:
    checkpoint_path = parse_checkpoint_path(cfg.checkpoint_path)
    wandb_run_id, checkpoint_num = _checkpoint_tags(checkpoint_path)

    deploy_policy = copy.deepcopy(policy.get_rollout_policy("deploy")).eval()
    fake_input = env.observation_spec[0].rand().to(env.device)

    export_dir = FILE_PATH / "exports" / str(cfg.task.name)
    export_dir.mkdir(parents=True, exist_ok=True)
    base = export_dir / f"policy-{wandb_run_id}-{checkpoint_num}"

    onnx_path = str(base.with_suffix(".onnx"))
    yaml_path = str(base.with_suffix(".yaml"))

    export_onnx(deploy_policy, fake_input, onnx_path)

    dict_cfg = OmegaConf.to_container(cfg, resolve=True)
    policy_config = {}

    obs_cfg = policy_config.setdefault("observation", {})
    for k in deploy_policy.in_keys:
        obs_cfg[k] = dict_cfg["task"]["observation"][k]

    asset = env.scene.articulations["robot"]
    asset_meta = _get_asset_meta(asset)
    policy_config["joint_names_simulation"] = asset.cfg.joint_names_simulation
    policy_config["body_names_simulation"] = asset.cfg.body_names_simulation
    policy_config["joint_kp"] = asset_meta["joint_kp"]
    policy_config["joint_kd"] = asset_meta["joint_kd"]
    policy_config["default_joint_pos"] = asset_meta["default_joint_pos"]

    # Make joint observation order explicit for sim2real consumers.
    from mimic_lite.tasks.command import RobotTracking
    from mimic_lite.tasks.actions import JointPosition
    action_manager = cast(JointPosition, env.action_manager)
    policy_config["policy_joint_names"] = action_manager.joint_names
    policy_config["action_scale"] = action_manager.action_scaling.detach().cpu().tolist()

    command = cast(RobotTracking, env.command_manager)

    motion_cfg = policy_config.setdefault("motion", {})
    from mimic_lite.tasks.multi_dataset import motion_cfgs_to_dict

    motion_cfg["motion_cfgs"] = motion_cfgs_to_dict(command.motion_cfgs)
    if len(command.motion_cfgs) == 1 and isinstance(command.motion_cfgs[0].path, str):
        motion_cfg["motion_path"] = str(command.motion_cfgs[0].path)
    motion_cfg["future_steps"] = command.future_steps.tolist()
    motion_cfg["body_names"] = command.tracking_body_names
    motion_cfg["joint_names"] = command.tracking_joint_names
    motion_cfg["root_body_name"] = command.root_body_name
    motion_cfg["anchor_body_name"] = command.anchor_body_name

    with open(yaml_path, "w") as f:
        yaml.dump(policy_config, f, sort_keys=False)

    print(f"Exported deploy config to {yaml_path}")


@hydra.main(config_path=str(CONFIG_PATH), config_name="play", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    aa.init(cfg, auto_rank=True)

    from active_adaptation.helpers import make_env_policy

    checkpoint_path = parse_checkpoint_path(cfg.get("checkpoint_path", None))
    if checkpoint_path is not None:
        cfg.checkpoint_path = checkpoint_path

    env, policy = make_env_policy(
        cfg.task,
        cfg.algo,
        seed=cfg.seed,
        headless=cfg.headless,
        device=cfg.device,
        checkpoint_path=cfg.get("checkpoint_path", None),
    )

    if cfg.get("export_policy", False):
        export_policy(cfg, env, policy)
        if cfg.get("export_only", False):
            return

    stats_keys = [
        k
        for k in env.reward_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)
    rollout_policy = policy.get_rollout_policy("eval")
    rollout_holder = {"policy": rollout_policy}

    env.base_env.eval()
    carry = env.reset()

    assert not env.base_env.training

    trajectory_log_path_cfg = cfg.get("trajectory_log_path", None)
    trajectory_log_path = (
        Path(str(trajectory_log_path_cfg)).expanduser().resolve()
        if trajectory_log_path_cfg
        else None
    )
    trajectory: dict[str, list[np.ndarray]] = {}
    trajectory_metadata: dict[str, np.ndarray] = {}
    if trajectory_log_path is not None:
        from mimic_lite.tasks.actions import JointPosition
        from mimic_lite.tasks.command import RobotTracking

        action_manager = cast(JointPosition, env.action_manager)
        command = cast(RobotTracking, env.command_manager)
        robot = env.scene.articulations["robot"]
        trajectory_env_ids = torch.arange(env.num_envs, device=env.device)
        trajectory = {
            "control_step": [],
            "motion_id": [],
            "motion_frame": [],
            "is_standing": [],
            "reference_joint_pos": [],
            "reference_joint_vel": [],
            "joint_pos": [],
            "joint_vel": [],
            "policy_action": [],
            "raw_action_prev": [],
            "applied_action_prev": [],
            "joint_pos_target_prev": [],
            "actuator_force_prev": [],
            "action_delay": [],
            "action_alpha": [],
            "reference_root_pos": [],
            "root_pos": [],
            "reference_root_quat": [],
            "root_quat": [],
            "body_pos_error": [],
            "body_pos_error_local": [],
            "body_ori_error": [],
            "body_ori_error_local": [],
            "body_lin_vel_error": [],
            "body_ang_vel_error": [],
            "joint_pos_error": [],
            "joint_vel_error": [],
        }
        trajectory_metadata = {
            "env_ids": trajectory_env_ids.cpu().numpy(),
            "action_joint_names": np.asarray(action_manager.joint_names),
            "tracking_joint_names": np.asarray(command.tracking_joint_names),
            "tracking_body_names": np.asarray(command.tracking_body_names),
            "step_dt_s": np.asarray(env.step_dt, dtype=np.float64),
            "sample_timing": np.asarray(
                "state/target/force are sampled at the start of each control step; "
                "policy_action is applied by the following env step"
            ),
        }

        def record_tensor(name: str, tensor: torch.Tensor) -> None:
            trajectory[name].append(
                tensor.index_select(0, trajectory_env_ids).detach().cpu().numpy()
            )

        trajectory_log_saved = False

        def save_trajectory_log() -> None:
            nonlocal trajectory_log_saved
            if trajectory_log_saved:
                return
            _save_trajectory_log(trajectory_log_path, trajectory_metadata, trajectory)
            trajectory_log_saved = True

        atexit.register(save_trajectory_log)

    timer = Timer(env.step_dt)
    fps_window_start = time.perf_counter()
    fps_window_frames = 0
    render_seconds = float(cfg.get("render_seconds", 0.0))
    render_enabled = render_seconds != 0.0
    if render_enabled:
        max_steps = max(1, int(render_seconds / env.step_dt))
        progress = tqdm(range(max_steps), total=max_steps, desc="Playing", unit="step")
    else:
        progress = itertools.count()
    output_path = _make_render_output_path()

    from active_adaptation.envs.backends.mjlab.playback import (
        play_post_step,
        play_pre_step,
        resolve_mjlab_viewer,
    )

    def _load_checkpoint(path: str) -> None:
        print(f"[Play]: Loading checkpoint {path}")
        state_dict = torch.load(path, weights_only=False)
        if "policy" not in state_dict:
            raise KeyError(f"No 'policy' key in checkpoint {path}")
        policy.load_state_dict(state_dict["policy"])
        rollout_holder["policy"] = policy.get_rollout_policy("eval")
        print(f"[Play]: Loaded checkpoint {path}")

    viewer = resolve_mjlab_viewer(env)
    if viewer is not None and cfg.get("checkpoint_path"):
        viewer.attach_checkpoints(cfg.checkpoint_path, on_load=_load_checkpoint)

    with (
        env.get_recorder(output_path, enabled=render_enabled) as recorder,
        torch.inference_mode(),
        set_exploration_type(ExplorationType.DETERMINISTIC),
        # set_exploration_type(ExplorationType.RANDOM),
    ):
        for i in progress:
            carry = play_pre_step(
                env, carry, on_checkpoint_load=_load_checkpoint
            )
            with ScopedTimer("inference", sync=False):
                carry = rollout_holder["policy"](carry)

            if trajectory_log_path is not None:
                trajectory["control_step"].append(np.asarray(i, dtype=np.int64))
                record_tensor("motion_id", command.motion_ids)
                record_tensor("motion_frame", command.t)
                record_tensor("is_standing", command.is_standing_env)
                record_tensor("reference_joint_pos", command.ref_joint_pos)
                record_tensor("reference_joint_vel", command.ref_joint_vel)
                record_tensor("joint_pos", command.robot_joint_pos)
                record_tensor("joint_vel", command.robot_joint_vel)
                record_tensor("policy_action", carry["action"])
                record_tensor("raw_action_prev", action_manager.action_buf[:, 0])
                record_tensor("applied_action_prev", action_manager.applied_action)
                record_tensor(
                    "joint_pos_target_prev",
                    robot.data.joint_pos_target[:, action_manager.joint_ids],
                )
                record_tensor(
                    "actuator_force_prev",
                    robot.data.actuator_force[:, action_manager.joint_ids],
                )
                record_tensor("action_delay", action_manager.delay)
                record_tensor("action_alpha", action_manager.alpha)
                record_tensor(
                    "reference_root_pos",
                    command.ref_root_pos_future_w[:, command.reward_current_step_index],
                )
                record_tensor("root_pos", command.robot_root_pos_w)
                record_tensor(
                    "reference_root_quat",
                    command.ref_root_quat_future_w[:, command.reward_current_step_index],
                )
                record_tensor("root_quat", command.robot_root_quat_w)
                record_tensor("body_pos_error", command.body_pos_error)
                record_tensor("body_pos_error_local", command.body_pos_error_local)
                record_tensor("body_ori_error", command.body_ori_error)
                record_tensor("body_ori_error_local", command.body_ori_error_local)
                record_tensor("body_lin_vel_error", command.body_lin_vel_error)
                record_tensor("body_ang_vel_error", command.body_ang_vel_error)
                record_tensor("joint_pos_error", command.joint_pos_error)
                record_tensor("joint_vel_error", command.joint_vel_error)

            with ScopedTimer("env_step", sync=False):
                td, carry = env.step_and_maybe_reset(carry)
            episode_stats.add(td)

            if len(episode_stats) >= env.num_envs:
                print("Step", i)
                for k, v in sorted(episode_stats.pop().items(True, True)):
                    print(k, torch.mean(v).item())

            if render_enabled:
                recorder.add_frame()

            fps_window_frames += 1
            window_elapsed = time.perf_counter() - fps_window_start
            if window_elapsed >= 1.0:
                if not render_enabled:
                    print(
                        f"Loop FPS: {fps_window_frames} frames in "
                        f"{window_elapsed:.2f}s"
                    )
                # ScopedTimer.print_summary(clear=True)
                fps_window_start = time.perf_counter()
                fps_window_frames = 0

            play_post_step(env, step_dt=env.step_dt, timer=timer)

    if trajectory_log_path is not None:
        save_trajectory_log()
        atexit.unregister(save_trajectory_log)

    env.close()


if __name__ == "__main__":
    main()
