#!/usr/bin/env python3
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Fix sys.path: when running as `python gear_sonic/train_agent_trl.py`, Python adds
# gear_sonic/ to sys.path[0], causing `from trl import ...` to resolve to our local
# gear_sonic/trl/ instead of the HuggingFace trl package. Replace with repo root.
import sys
import os
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
if _script_dir in sys.path:
    sys.path.remove(_script_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    import isaaclab  # noqa: F401
except ImportError:
    print(
        "\n"
        "ERROR: Isaac Lab is required for training but not installed.\n"
        "\n"
        "Isaac Lab is not a pip dependency — it must be installed separately.\n"
        "Follow the official guide:\n"
        "  https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html\n"
        "\n"
        "After installing, activate the Isaac Lab conda/venv environment\n"
        "before running this script.\n"
    )
    sys.exit(1)

import glob
import logging
import os
from pathlib import Path
import re
import sys

from filelock import FileLock
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import wandb
import yaml

from gear_sonic.trl.utils.common import (
    custom_instantiate,
    get_filtered_state_dict,
    materialize_lazy_params,
    wandb_run_exists,
)
from gear_sonic.utils.common import seeding
from gear_sonic.utils.config_utils import register_rl_resolvers
from gear_sonic.utils.obs_utils import get_group_term_obs_shape

register_rl_resolvers()


def resume_training(config):
    if config.get("checkpoint", None) is not None:
        last_existing_checkpoint = config.checkpoint
    elif config.get("experiment_dir", None) is not None:
        last_existing_checkpoint = os.path.join(config.experiment_dir, "last.pt")
    else:
        # Use experiment_dir to find the checkpoint, rather than reconstructing
        # from config.project_name which can differ from the actual filesystem path.
        experiment_dir_base = re.sub(r"-\d{8}_\d{6}$", "", config.experiment_dir)
        checkpoints = sorted(glob.glob(os.path.join(f"{experiment_dir_base}-*", "last.pt")))
        if not checkpoints:
            print(f"No checkpoint found matching {experiment_dir_base}-*/last.pt, starting fresh")
            return
        last_existing_checkpoint = checkpoints[-1]
    experiment_dir = os.path.dirname(last_existing_checkpoint)
    config.experiment_dir = experiment_dir
    config.checkpoint = last_existing_checkpoint
    print(f"Resuming training from {last_existing_checkpoint}")


def resume_checkpoint(config):
    config.checkpoint = config.checkpoint


def create_manager_env(config, device, args_cli):

    # import wandb

    from isaaclab.envs import (
        ManagerBasedRLEnv,
    )

    from gear_sonic.envs.wrapper.manager_env_wrapper import ManagerEnvWrapper

    env_instance_cfg = custom_instantiate(config.manager_env)

    # Iteratively check the difference in attribute of env_instance_cfg1 and env_instance_cfg, print out the difference
    def compare_attrs(obj1, obj2, prefix=""):
        # Only compare attributes that do not start with '__' and are not methods
        attrs1 = set(dir(obj1))
        attrs2 = set(dir(obj2))
        common_attrs = attrs1 & attrs2
        for attr in sorted(common_attrs):
            if (
                attr.startswith("__")
                or callable(getattr(obj1, attr))
                or callable(getattr(obj2, attr))
            ):
                continue
            try:
                val1 = getattr(obj1, attr)
                val2 = getattr(obj2, attr)
            except Exception:
                continue
            # Recursively compare if both are objects with __dict__ or are dicts
            if isinstance(val1, dict | DictConfig) and isinstance(val2, dict | DictConfig):
                compare_attrs(val1, val2, prefix + attr + ".")
            elif hasattr(val1, "__dict__") and hasattr(val2, "__dict__"):
                compare_attrs(val1, val2, prefix + attr + ".")
            else:
                if isinstance(val1, list):
                    val1 = tuple(val1)
                if isinstance(val2, list):
                    val2 = tuple(val2)
                if val1 != val2:
                    print(
                        f"\nDifference found at '{prefix}{attr}':\n"
                        f"  - env_instance_cfg1: {val1!r}\n"
                        f"  - env_instance_cfg : {val2!r}\n"
                    )

    env_instance_cfg.seed = config.seed
    env_instance_cfg.sim.device = device
    env_instance_cfg.config["headless"] = args_cli.headless
    env = ManagerBasedRLEnv(
        cfg=env_instance_cfg, render_mode="rgb_array" if not args_cli.headless else None
    )

    env = ManagerEnvWrapper(env, env_instance_cfg.config)
    return env


@hydra.main(config_path="config", config_name="base", version_base="1.1")
def main(config: OmegaConf):
    simulator_type = "IsaacSim"
    env_config = config.manager_env
    from transformers import HfArgumentParser
    from trl import ModelConfig, PPOConfig, ScriptArguments

    # Setup model components
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))

    if config.get("resume", False):
        resume_training(config)
    elif config.get("checkpoint", None) is not None:
        resume_checkpoint(config)

    config.algo.trl.output_dir = str(Path(config.experiment_dir))

    script_args, training_args, model_args = parser.parse_dict(config.algo.trl)

    # Add exp_name from main config to training_args
    training_args.exp_name = config.experiment_name

    from datetime import timedelta

    from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
    import torch  # noqa: E402

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=6000))
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs, kwargs],
    )

    device = str(accelerator.device)
    if device == "cuda":
        device = "cuda:0"
    config.multi_gpu = accelerator.num_processes > 1
    if config.multi_gpu:
        config.global_rank = accelerator.process_index
        config.seed += accelerator.process_index
        config.algo.config.global_rank = accelerator.process_index
        config.algo.config.world_size = accelerator.num_processes
    seeding(config.seed)

    meta_path = Path(config.experiment_dir) / "meta.yaml"
    if meta_path.exists():
        meta = yaml.safe_load(open(meta_path))
        config.wandb.wandb_id = meta["wandb_run"]
        print(f"resume wandb from run: {config.wandb.wandb_id}")

    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    if config.use_wandb and accelerator.is_main_process:
        project_name = f"{config.project_name}"
        run_name = config.experiment_dir.replace(f"{config.base_dir}/{project_name}/", "")
        wandb_dir = Path(config.wandb.wandb_dir)
        wandb_dir.mkdir(exist_ok=True, parents=True)
        wandb_group = None if config.wandb.wandb_id is not None else config.wandb.wandb_group
        logger.info(f"Saving wandb logs to {wandb_dir}")
        wandb.init(
            project=project_name,
            entity=config.wandb.wandb_entity,
            name=run_name,
            sync_tensorboard=True,
            config=unresolved_conf,
            dir=wandb_dir,
            id=config.wandb.wandb_id,
            group=wandb_group,
            resume="allow",
        )

    # Setup simulator similar to train_agent.py

    if simulator_type == "IsaacSim":
        try:
            with open("./rl/simulator/isaacsim/.isaacsim_version", encoding="utf-8") as f:
                DEFAULT_ISAACSIM_VERSION = f.read().strip()
        except FileNotFoundError:
            DEFAULT_ISAACSIM_VERSION = "4.5"

        if DEFAULT_ISAACSIM_VERSION == "4.5":
            from isaaclab.app import AppLauncher
        elif DEFAULT_ISAACSIM_VERSION == "4.2":
            logger.warning("Using IsaacSim 4.2, replacing isaaclab with omni.isaac.lab")
            from omni.isaac.lab.app import AppLauncher  # 4.2

            # from isaaclab.app import AppLauncher # not working
            # from omni.isaac.lab.app import AppLauncher

        import argparse

        parser = argparse.ArgumentParser(description="Train an RL agent with TRL.")
        AppLauncher.add_app_launcher_args(parser)

        ######################################################### ZL: fix isaacsim 4.5 rendering #########################################################
        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = env_config.config.env_spacing  # config.env_spacing
        args_cli.output_dir = config.output_dir
        # Enable cameras if enable_cameras, render_results, render_ego, or overview_camera is True
        args_cli.enable_cameras = (
            env_config.config.get("enable_cameras", False)
            or env_config.config.get("render_results", False)
            or env_config.config.get("render_ego", False)
            or env_config.config.get("overview_camera", False)
        )
        args_cli.headless = config.headless
        args_cli.multi_gpu = config.multi_gpu
        args_cli.distributed = config.multi_gpu
        args_cli.device = device

        # Base kit args (quiet logs)
        args_cli.kit_args = (
            "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
        )

        # AppLauncher can't handle multiple processes creating it at the same time so we need a lock
        # Serialize Kit startup only among this Unix user's local ranks.  A
        # machine-global filename can be left owned and non-writable by another
        # user on shared training nodes, preventing an otherwise independent run.
        _lock_path = os.environ.get(
            "ISAACLAB_APP_LAUNCHER_LOCK",
            f"/tmp/isaaclab_app_launcher_{os.getuid()}.lock",
        )
        _local_rank = int(os.environ.get("LOCAL_RANK", 0))
        with FileLock(_lock_path):
            app_launcher = AppLauncher(args_cli)

        simulation_app = app_launcher.app

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    from gear_sonic.utils.logging import HydraLoggerBridge

    # resolve=False is important otherwise overrides
    # at inference time won't work properly
    # also, I believe this must be done before instantiation

    # logging to hydra log file
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "train.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    # Setup wandb if enabled
    os.chdir(hydra.utils.get_original_cwd())

    # Save config and meta BEFORE env creation so eval jobs can postprocess
    # checkpoint configs even if training crashes during env init.
    experiment_save_dir = Path(config.experiment_dir)
    if accelerator.is_main_process:
        experiment_save_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Saving config file to {experiment_save_dir}")
        with open(experiment_save_dir / "config.yaml", "w") as file:
            OmegaConf.save(unresolved_conf, file)
        meta = {"wandb_run": wandb.run.id if wandb_run_exists() else None}
        meta["max_train_steps"] = config.algo.config.num_learning_iterations
        yaml.safe_dump(meta, open(meta_path, "w"))
        print("saved meta:", meta)

    # Initialize environment
    env_config.config.save_rendering_dir = str(Path(config.experiment_dir) / "renderings_training")
    env_config.config.experiment_dir = str(Path(config.experiment_dir))

    env = create_manager_env(config, device, args_cli)
    if config.get("replay", False):
        _save_video_path = config.get("replay_save_video", None)
        env.run_replay(
            start_time_step=-1,
            loop=config.get("replay_loop_num", True),
            save_video_path=_save_video_path,
            grid_spacing=config.get("replay_grid_spacing", 2.0),
        )
        os._exit(0)
    if config.get("vplanner_replay", False):
        vplanner_checkpoint = config.get("vplanner_checkpoint", None)
        if vplanner_checkpoint is None:
            raise ValueError("vplanner_checkpoint must be specified for vplanner_replay")
        env.run_vplanner_replay(
            checkpoint_path=vplanner_checkpoint,
            max_frames=config.get("vplanner_max_frames", 500),
            replan_interval=config.get("vplanner_replan_interval", 0),
            speed=config.get("vplanner_speed", 1.0),
            loop=config.get("vplanner_loop", True),
            save_images=config.get("vplanner_save_images", False),
            output_dir=config.get("vplanner_output_dir", None),
            dof_noise=config.get("vplanner_dof_noise", 0.0),
            dof_vel_noise=config.get("vplanner_dof_vel_noise", 0.0),
            quat_noise=config.get("vplanner_quat_noise", 0.0),
        )
        os._exit(0)

    ref_model = None
    value_model = None
    disc_model = None
    # import ipdb; ipdb.set_trace()

    if config.algo.config.get("use_new_actor_critic", False):
        module_dim_dict = getattr(config.algo.config, "module_dim", {})
        policy_backbone_kwargs = {}
        critic_backbone_kwargs = {}
        env.config["obs"]["obs_dims"]["actor_obs"] = env.env.observation_space["policy"].shape[-1]
        env.config["obs"]["obs_dims"]["critic_obs"] = env.env.observation_space["critic"].shape[-1]
        env.config["robot"]["algo_obs_dim_dict"]["actor_obs"] = env.env.observation_space[
            "policy"
        ].shape[-1]
        env.config["robot"]["algo_obs_dim_dict"]["critic_obs"] = env.env.observation_space[
            "critic"
        ].shape[-1]
        example_obs = env.reset(flatten_dict_obs=False)
        for key in env.env.observation_space:
            if key not in ["policy", "critic"]:
                group_obs_dims, group_obs_names, group_obs_total_dim = get_group_term_obs_shape(
                    example_obs, key
                )
                env.config["obs"]["group_obs_dims"][key] = group_obs_dims
                env.config["obs"]["group_obs_names"][key] = group_obs_names
                env.config["obs"]["obs_dims"][key] = group_obs_total_dim
                env.config["robot"]["algo_obs_dim_dict"][key] = group_obs_total_dim
        if config.manager_env.config.get("meta_action_dim", None) is not None:
            env.config["robot"]["actions_dim"] = config.manager_env.config.meta_action_dim
        else:
            env.config["robot"]["actions_dim"] = env.env.action_space.shape[-1]

        policy = custom_instantiate(
            config.algo.config.actor,
            env_config=env.config,
            algo_config=config.algo.config,
            module_dim_dict=module_dim_dict,
            backbone_kwargs=policy_backbone_kwargs,
            _resolve=False,
        ).to(device)

        if getattr(config.algo.config, "use_dagger", False):
            # Get teacher input key from config or default to "teacher"
            teacher_input_key = config.algo.config.get("teacher_input_key", "teacher")
            ref_model = custom_instantiate(
                config.algo.config.teacher_actor,
                env_config=env.config,
                algo_config=config.algo.config,
                module_dim_dict=module_dim_dict,
                _resolve=False,
                input_key=teacher_input_key,
            ).to(device)
        if not getattr(config.algo.config, "distill_only", False):
            value_model = custom_instantiate(
                config.algo.config.critic,
                env_config=env.config,
                algo_config=config.algo.config,
                module_dim_dict=module_dim_dict,
                backbone_kwargs=critic_backbone_kwargs,
                _resolve=False,
            ).to(device)
        if config.algo.config.get("use_amp", False):
            disc_model = custom_instantiate(
                config.algo.config.disc,
                env_config=env.config,
                algo_config=config.algo.config,
                module_dim_dict=module_dim_dict,
                _resolve=False,
            ).to(device)
    else:
        raise ValueError("No longer supported")

    materialize_lazy_params(policy, env)

    if config.algo.config.get("pretrained_model", None) is not None:
        pretrained_cfg = config.algo.config.pretrained_model
        sd_key = pretrained_cfg.get("state_dict_key", "state_dict")
        strict = pretrained_cfg.get("strict", True)
        state_dict = torch.load(pretrained_cfg.path, map_location=device, weights_only=False)[
            sd_key
        ]
        for (
            module_name,
            state_dict_key,
        ) in pretrained_cfg.module_mapping.items():
            module = eval(module_name)
            filtered_state_dict = get_filtered_state_dict(state_dict, state_dict_key)
            missing, unexpected = module.load_state_dict(filtered_state_dict, strict=strict)
            if missing:
                logger.info(f"Pretrained loading '{module_name}': missing keys: {missing}")
            if unexpected:
                logger.info(f"Pretrained loading '{module_name}': unexpected keys: {unexpected}")

    accelerator.wait_for_everyone()

    callbacks = []
    for callback in config.callbacks.values():
        callbacks.append(instantiate(callback))

    ################
    # Training
    ################
    trainer = custom_instantiate(
        config.trainer,
        args=training_args,
        config=config.algo.config,
        env=env,
        model=policy,
        disc_model=disc_model,
        value_model=value_model,
        ref_model=ref_model,
        use_ref_model=getattr(config.algo.config, "use_dagger", False),
        train_dataset=None,
        eval_dataset=None,
        callbacks=callbacks,
        checkpoint=config.checkpoint,
        resume=config.get("resume", False),
        local_seed=config.seed,
        log_dir=experiment_save_dir,
        accelerator=accelerator,
        _resolve=False,
    )

    # 诊断模式：不训练，只用已加载好权重的 policy 对单条指定动作跑一次
    # encoder→token→g1_kin decoder 前向，报告关节重建误差（角度制），用于验证
    # BUMI2 SONIC 上层训练是否已经收敛到可用于桥接的重建质量。通过
    # ++manager_env.commands.motion.filter_motion_keys=[<key>] 配合 num_envs=1
    # 保证 env 0 加载的就是指定动作。不影响任何正常训练路径。
    if config.get("inspect_g1_recon", False):
        import numpy as np

        # 实测发现：全新构造的 env 第一次 env.reset() 存在"冷启动"效应，算出来的
        # 重建误差系统性偏大 2~4 倍（实测同一个 checkpoint/同一条动作/同一个起始帧，
        # 只做一次 reset 测出 ~14°，先做一次热身 reset 再测第二次只有 ~4~6°）。
        # 具体机制未查证到 IsaacLab 源码层面的确切原因，但现象在多次独立测试里
        # 稳定复现，因此这里固定加一次热身 reset，不能省略，否则度数误差会被
        # 系统性高估。2026-09-21 用有偏差的测法得到过一版"G1 明显更好、BUMI 系列
        # 追不上"的错误结论，补上热身 reset 重测后完全反转（G1/BUMI3-10万轮/BUMI2
        # 三者实际都在 3.5~4.2° 左右，没有明显差距），详见 sonic_mimiclite_new.md
        # §3 和 BUMI3_SONIC_修改记录.md 对应条目。
        env.reset()

        obs_dict = env.reset()
        if isinstance(obs_dict, tuple):
            obs_dict = obs_dict[0]
        print("INSPECT obs_dict keys:", list(obs_dict.keys()))
        for k, v in obs_dict.items():
            if torch.is_tensor(v):
                print(f"INSPECT obs[{k}].shape = {tuple(v.shape)}")

        policy.eval()
        with torch.no_grad():
            obs_for_model = dict(obs_dict)
            if policy.running_mean_std is not None:
                obs_for_model[policy.input_key] = policy.running_mean_std(
                    obs_for_model[policy.input_key]
                )
            for key in ("actor_obs", "tokenizer"):
                if obs_for_model[key].dim() == 2:
                    obs_for_model[key] = obs_for_model[key].unsqueeze(1)
            output = policy.actor_module(
                obs_for_model, compute_aux_loss=False, return_dict=True
            )

        print("INSPECT output keys:", list(output.keys()))
        tokenizer_obs = output["tokenizer_obs"]
        decoded = output["decoded_outputs"]["g1_kin"]
        print("INSPECT tokenizer_obs keys:", list(tokenizer_obs.keys()))
        print("INSPECT decoded keys:", list(decoded.keys()))

        gt_nonflat = tokenizer_obs["command_multi_future_nonflat"]
        pred_nonflat = decoded["command_multi_future_nonflat"]
        print("INSPECT command_multi_future_nonflat gt shape:", tuple(gt_nonflat.shape))
        print("INSPECT command_multi_future_nonflat pred shape:", tuple(pred_nonflat.shape))

        num_future_frames = config.manager_env.commands.motion.num_future_frames
        robot_num_dof = env.env.scene["robot"].num_joints
        print(f"INSPECT num_future_frames={num_future_frames} robot_num_dof={robot_num_dof}")

        def split_pos_vel(nonflat):
            flat = nonflat.reshape(nonflat.shape[0], -1)
            half = num_future_frames * robot_num_dof
            pos_flat = flat[:, :half]
            vel_flat = flat[:, half:]
            pos = pos_flat.reshape(-1, num_future_frames, robot_num_dof)
            vel = vel_flat.reshape(-1, num_future_frames, robot_num_dof)
            return pos, vel

        gt_pos, gt_vel = split_pos_vel(gt_nonflat)
        pred_pos, pred_vel = split_pos_vel(pred_nonflat)

        pos_err_rad = (pred_pos - gt_pos).abs()
        pos_err_deg = torch.rad2deg(pos_err_rad)
        print(
            "INSPECT joint_pos reconstruction error (deg): "
            f"mean={pos_err_deg.mean().item():.4f} "
            f"max={pos_err_deg.max().item():.4f} "
            f"p95={torch.quantile(pos_err_deg.flatten(), 0.95).item():.4f}"
        )
        per_frame_deg = pos_err_deg.mean(dim=(0, 2))
        print(
            "INSPECT per-future-frame mean error (deg):",
            np.round(per_frame_deg.cpu().numpy(), 4).tolist(),
        )

        robot_joint_names = list(env.env.scene["robot"].joint_names)
        isaaclab_joints_declared = list(env.env.cfg.isaaclab_to_mujoco_mapping["isaaclab_joints"])
        print("INSPECT robot.joint_names           :", robot_joint_names)
        print("INSPECT isaaclab_to_mujoco[isaaclab_joints]:", isaaclab_joints_declared)
        print("INSPECT names_match:", robot_joint_names == isaaclab_joints_declared)
        motion_command = env.env.command_manager.get_term("motion")
        print(
            "INSPECT motion_ids:", motion_command.motion_ids.tolist(),
            "time_steps:", motion_command.time_steps.tolist(),
            "motion_start_time_steps:", motion_command.motion_start_time_steps.tolist(),
        )
        print("INSPECT mujoco_to_isaaclab_dof:", motion_command.mujoco_to_isaaclab_dof)
        print("INSPECT isaaclab_to_mujoco_dof:", motion_command.isaaclab_to_mujoco_dof)
        raw_dof_pos_frame0 = motion_command.motion_lib.get_dof_pos(
            motion_command.motion_ids, motion_command.motion_start_time_steps
        )
        print("INSPECT raw get_dof_pos frame0:", raw_dof_pos_frame0[0].cpu().numpy().tolist())
        print(
            "INSPECT command.joint_pos (current frame) property:",
            motion_command.joint_pos[0].cpu().numpy().tolist(),
        )
        dump_path = config.get("inspect_g1_recon_dump_path", None)
        if dump_path is not None:
            import json

            dump = {
                "robot_joint_names_isaaclab_order": robot_joint_names,
                "motion_ids": motion_command.motion_ids.tolist(),
                "time_steps": motion_command.time_steps.tolist(),
                "motion_start_time_steps": motion_command.motion_start_time_steps.tolist(),
                "gt_pos_rad": gt_pos[0].cpu().numpy().tolist(),
                "pred_pos_rad": pred_pos[0].cpu().numpy().tolist(),
            }
            with open(dump_path, "w") as f:
                json.dump(dump, f)
            print(f"INSPECT dumped raw values to {dump_path}")

        print("INSPECT_G1_RECON_DONE")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    # 桥接导出模式：把 g1_kin decoder 的 token 往返重建结果，和驱动源自带的绝对
    # root 轨迹拼在一起，导出成 MimicLite-BUMI2 能读的 bumi_deploy_motion_v1 JSON。
    #
    # g1_kin decoder 单次前向只能吐出"当前时刻 + 未来 10 帧（间隔 0.1s，覆盖 1
    # 秒）"这一个窗口，覆盖完整动作需要多次调用、把窗口拼起来。这里用"多次独立
    # env.reset()"的方式拼接：每次通过临时替换 motion_lib.sample_time_steps
    # （只在这一次 reset 期间生效，reset 完立刻还原）把该 env 的起始帧钉死在我们
    # 想要的窗口起点上，让 Isaac Lab 官方 reset 流程（包括把机器人物理姿态写回
    # 仿真器这一步）在正确的起始帧上完整走一遍——不是自己手搓改 time_steps 张量后
    # 绕开物理状态同步，这样才能保证 encoder 第二个输入特征（参考朝向相对当前
    # 机器人朝向的差值）在每个窗口开头都是有效值，不是从上一个窗口结束时残留的
    # 姿态算出来的垃圾值。root 轨迹（位置/朝向）不经过 decoder，直接从驱动源本身
    # 的真值轨迹整段读取——这是刻意的设计（g1_kin 不重建绝对根轨迹，只重建关节
    # 角度/速度，详见 sonic_mimiclite_new.md §6）。
    if config.get("export_bridge_motion", False):
        import json

        import numpy as np

        output_path = config.get("export_bridge_motion_output", None)
        if output_path is None:
            raise ValueError("export_bridge_motion requires ++export_bridge_motion_output=<path>")

        motion_command = env.env.command_manager.get_term("motion")
        robot_joint_names = list(env.env.scene["robot"].joint_names)
        num_future_frames = config.manager_env.commands.motion.num_future_frames
        robot_num_dof = env.env.scene["robot"].num_joints
        frame_skips = motion_command.frame_skips
        sparse_frame_idx = np.arange(num_future_frames) * frame_skips  # e.g. [0,5,...,45]
        window_len = int(sparse_frame_idx[-1] + 1)  # 每个窗口稠密化后的帧数，例如 46

        def split_pos_vel(nonflat):
            flat = nonflat.reshape(nonflat.shape[0], -1)
            half = num_future_frames * robot_num_dof
            pos = flat[:, :half].reshape(-1, num_future_frames, robot_num_dof)
            vel = flat[:, half:].reshape(-1, num_future_frames, robot_num_dof)
            return pos, vel

        def interp_to_dense(sparse_values, n_dense):
            dense_idx = np.arange(n_dense)
            out = np.empty((n_dense, sparse_values.shape[1]), dtype=np.float64)
            for d in range(sparse_values.shape[1]):
                out[:, d] = np.interp(dense_idx, sparse_frame_idx, sparse_values[:, d])
            return out

        policy.eval()

        # 先正常 reset 一次，只是为了读出 motion_ids、总帧数，不用这次的重建结果。
        env.reset()
        motion_ids = motion_command.motion_ids.clone()
        total_frames = int(
            motion_command.motion_lib.get_time_step_total(motion_ids)[0].item()
        )
        print(f"EXPORT total_frames={total_frames} window_len={window_len}")

        # sonic_bumi2.yaml 开了 adaptive_sampling.enable=true，真正的 reset 路径走的
        # 是 sample_motion_ids_and_time_steps（不是 sample_time_steps），必须патch
        # 这一个——第一版代码 patch 错了方法，被下面的断言当场抓出来（错误地拿到了
        # adaptive sampling 自己采样出的起始帧 60，不是我们要的 0），改成正确的方法。
        real_sample_ids_and_steps = motion_command.motion_lib.sample_motion_ids_and_time_steps
        fixed_motion_ids = motion_ids.clone()

        joint_pos_chunks = []
        joint_vel_chunks = []
        window_starts = list(range(0, total_frames, window_len))
        for w_idx, start in enumerate(window_starts):

            def _fixed_ids_and_steps(n, _start=start):
                return (
                    fixed_motion_ids[:n].clone(),
                    torch.full((n,), _start, dtype=torch.long, device=fixed_motion_ids.device),
                )

            motion_command.motion_lib.sample_motion_ids_and_time_steps = _fixed_ids_and_steps
            try:
                obs_dict = env.reset()
            finally:
                motion_command.motion_lib.sample_motion_ids_and_time_steps = (
                    real_sample_ids_and_steps
                )
            if isinstance(obs_dict, tuple):
                obs_dict = obs_dict[0]

            actual_start = int(motion_command.motion_start_time_steps[0].item())
            if actual_start != start:
                raise RuntimeError(
                    f"Window {w_idx}: requested start {start} but motion_start_time_steps "
                    f"came back as {actual_start} — sample_time_steps monkeypatch didn't "
                    "take effect as expected, aborting rather than exporting silently wrong data."
                )

            with torch.no_grad():
                obs_for_model = dict(obs_dict)
                if policy.running_mean_std is not None:
                    obs_for_model[policy.input_key] = policy.running_mean_std(
                        obs_for_model[policy.input_key]
                    )
                for key in ("actor_obs", "tokenizer"):
                    if obs_for_model[key].dim() == 2:
                        obs_for_model[key] = obs_for_model[key].unsqueeze(1)
                output = policy.actor_module(obs_for_model, compute_aux_loss=False, return_dict=True)

            decoded = output["decoded_outputs"]["g1_kin"]
            pred_pos, pred_vel = split_pos_vel(decoded["command_multi_future_nonflat"])
            pred_pos = pred_pos[0].cpu().numpy()
            pred_vel = pred_vel[0].cpu().numpy()

            n_valid = min(window_len, total_frames - start)
            dense_pos = interp_to_dense(pred_pos, window_len)[:n_valid]
            dense_vel = interp_to_dense(pred_vel, window_len)[:n_valid]
            joint_pos_chunks.append(dense_pos)
            joint_vel_chunks.append(dense_vel)
            print(f"EXPORT window {w_idx}: start={start} n_valid={n_valid}")

        joint_pos_full = np.concatenate(joint_pos_chunks, axis=0)
        joint_vel_full = np.concatenate(joint_vel_chunks, axis=0)
        assert joint_pos_full.shape[0] == total_frames, (
            f"stitched joint_pos length {joint_pos_full.shape[0]} != total_frames {total_frames}"
        )

        all_time_steps = torch.arange(total_frames, device=motion_ids.device, dtype=torch.long)
        motion_ids_full = motion_ids[0].expand(total_frames)
        root_pos_w = motion_command.motion_lib.get_root_pos_w(
            motion_ids_full, all_time_steps
        ).cpu().numpy()
        # motion_lib.get_root_quat_w() 直接返回 wxyz（实测核对：与已知真值逐位相等，
        # 不需要再做 xyzw->wxyz 转换——不要被 motion_lib_base.py 里对
        # body_quat_w_full 做的 xyzw_to_wxyz 转换误导，那是另一个派生数组）。
        root_quat_wxyz = motion_command.motion_lib.get_root_quat_w(
            motion_ids_full, all_time_steps
        ).cpu().numpy()

        deploy_json = {
            "metadata": {
                "format": "bumi_deploy_motion_v1",
                "frames_count": total_frames,
                "joints_count": robot_num_dof,
                "fps": 50.0,
                "joint_order": "deploy",
                "joint_names": robot_joint_names,
                "root_body_name": "base_link",
                "quaternion_order": "wxyz",
                "source": "sonic_g1_kin_bridge_export_multiwindow",
            },
            "joint_pos": joint_pos_full.tolist(),
            "joint_vel": joint_vel_full.tolist(),
            "root_pos_w": root_pos_w.tolist(),
            "root_quat_w": root_quat_wxyz.tolist(),
        }
        with open(output_path, "w") as f:
            json.dump(deploy_json, f)
        print(f"EXPORT_BRIDGE_MOTION_DONE frames={total_frames} windows={len(window_starts)} -> {output_path}")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    # Training loop
    trainer.train()

    if simulator_type == "IsaacSim":
        os._exit(0)


if __name__ == "__main__":

    main()
