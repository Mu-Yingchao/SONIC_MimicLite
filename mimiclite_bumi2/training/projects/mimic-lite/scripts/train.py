import torch
import hydra
import logging
import re
import time
import datetime
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig

from collections import OrderedDict
from tqdm import tqdm
from setproctitle import setproctitle

from torch.utils.tensorboard import SummaryWriter
from torchrl.envs.utils import set_exploration_type, ExplorationType
from tensordict import TensorDict

import active_adaptation as aa
from active_adaptation.utils.profiling import ScopedTimer
from active_adaptation.utils.checkpoint_cfg import (
    find_run_cfg_yaml,
    is_from_checkpoint_algo,
    load_algo_cfg_from_checkpoint,
)
from active_adaptation.learning.ppo.ppo_base import PPOBase
from active_adaptation.learning.modules.vecnorm import VecNorm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


FILE_PATH = Path(__file__).resolve().parent
CONFIG_PATH = FILE_PATH.parent / "cfg"


def _start_iteration(checkpoint_path: str | None, current_iter: int) -> int:
    return current_iter + 1 if checkpoint_path and current_iter > 0 else 0


def _scalar_metrics(info: dict) -> dict:
    metrics = {}
    for key, value in info.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            value = value.detach().item()
        elif hasattr(value, "item"):
            try:
                value = value.item()
            except (TypeError, ValueError):
                continue
        if isinstance(value, (bool, int, float, str)) or value is None:
            metrics[key] = value
    return metrics


def _restore_checkpoint_observations(
    cfg: DictConfig, checkpoint_path: str
) -> list[str]:
    """Restore observation groups that define the checkpoint network inputs."""
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    run_cfg_path = find_run_cfg_yaml(checkpoint_file)
    if run_cfg_path is None:
        raise FileNotFoundError(
            f"No cfg.yaml found next to checkpoint {checkpoint_file}"
        )

    saved_cfg = OmegaConf.load(run_cfg_path)
    OmegaConf.resolve(saved_cfg)
    saved_observations = OmegaConf.select(saved_cfg, "task.observation")
    if saved_observations is None:
        raise KeyError(f"{run_cfg_path} has no task.observation section")

    in_keys = cfg.algo.get("in_keys", [])
    if isinstance(in_keys, str):
        in_keys = [in_keys]
    restored = []
    for group_name, group_cfg in saved_observations.items():
        if any(re.fullmatch(pattern, group_name) for pattern in in_keys):
            cfg.task.observation[group_name] = OmegaConf.create(
                OmegaConf.to_container(group_cfg, resolve=True)
            )
            restored.append(group_name)
    return restored


@hydra.main(config_path=str(CONFIG_PATH), config_name="train", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    # Training needs algorithm fields such as ``train_every`` before
    # ``make_env_policy`` is called.  Resolve the from-checkpoint sentinel here
    # and store the real config back into cfg so resumed checkpoints also save
    # the complete algorithm configuration.
    if is_from_checkpoint_algo(cfg.algo):
        checkpoint_path = cfg.get("checkpoint_path", None)
        if not checkpoint_path:
            raise ValueError(
                "algo=from_checkpoint requires checkpoint_path=... for training resume"
            )
        cfg.algo = load_algo_cfg_from_checkpoint(checkpoint_path)
        if bool(cfg.get("resume_rebase_schedules", False)):
            if cfg.algo.name != "mimic_lite_ppo":
                raise ValueError(
                    "resume_rebase_schedules is only supported for ordinary PPO"
                )
            cfg.algo.resume_rebase_schedules = True
            print(
                "[Info]: PPO resume will rebase entropy/KL schedules to the "
                "new total_iters."
            )
        restored_observations = _restore_checkpoint_observations(
            cfg, checkpoint_path
        )
        print(
            "[Info]: Restored checkpoint observation groups: "
            + ", ".join(restored_observations)
        )

    warm_start_phase = cfg.get("warm_start_phase", None)
    if warm_start_phase is not None:
        if warm_start_phase not in ("train", "adapt", "finetune"):
            raise ValueError(f"Invalid warm_start_phase: {warm_start_phase}")
        if not cfg.get("checkpoint_path") or cfg.algo.name != "ppo_roa":
            raise ValueError("Three-stage warm start requires a PPO-ROA checkpoint")
        # Reuse the project's phase-specific optimization settings, retaining
        # the checkpoint's architecture and observation layout.
        recipe = OmegaConf.load(CONFIG_PATH / "exp" / "ppo_roa" / f"{warm_start_phase}.yaml")
        cfg.algo = OmegaConf.merge(cfg.algo, recipe.get("algo", {}))
        from mimic_lite_learning.ppo_roa import PPOConfig

        # Adapt uses shorter rollouts; do not inherit those into finetune.
        cfg.algo.train_every = recipe.algo.get(
            "train_every", PPOConfig(phase=warm_start_phase).train_every
        )
        cfg.algo.phase = warm_start_phase
        print(f"[Info]: Warm start phase={warm_start_phase}; iteration starts at 0")

    OmegaConf.resolve(cfg)

    aa.init(cfg, auto_rank=True)

    print(
        f"is_distributed: {aa.is_distributed()}, local_rank: {aa.get_local_rank()}/{aa.get_world_size()}"
    )

    from active_adaptation.helpers import make_env_policy
    from active_adaptation.utils.helpers import EpisodeStats

    num_envs = 1 if cfg.backend == "mujoco" else cfg.task.num_envs
    frames_per_batch = num_envs * cfg.algo.train_every
    total_iters = cfg.get("total_iters", None)
    if total_iters is None:
        total_frames = cfg.get("total_frames", -1) // aa.get_world_size()
        total_frames = total_frames // frames_per_batch * frames_per_batch
        total_iters = total_frames // frames_per_batch
    cfg.task.total_iters = total_iters

    env, policy = make_env_policy(
        cfg.task,
        cfg.algo,
        seed=cfg.seed,
        headless=cfg.headless,
        device=cfg.device,
        checkpoint_path=cfg.get("checkpoint_path", None),
    )
    policy: PPOBase
    if warm_start_phase is not None:
        # Also reset when the source and destination phases are the same.
        env.set_progress(0)
    requires_rollout_value = bool(getattr(policy, "requires_rollout_value", True))

    env.total_iters = total_iters
    policy.total_iters = total_iters

    checkpoint_interval = cfg.checkpoint_interval
    log_interval = (cfg.task.max_episode_length // cfg.algo.train_every) + 1
    logging.info(f"Log interval: {log_interval} steps")

    stats_keys = [
        k
        for k in env.reward_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)

    def save(policy, checkpoint_name: str):
        output_dir = Path(HydraConfig.get().runtime.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = output_dir / f"{checkpoint_name}.pt"
        state_dict = OrderedDict()
        state_dict["logger"] = {
            "type": "tensorboard",
            "log_dir": str(output_dir / "tb"),
        }
        state_dict["policy"] = policy.state_dict()
        state_dict["env"] = env.state_dict()
        state_dict["cfg"] = cfg

        torch.save(state_dict, ckpt_path)
        latest_link = output_dir / "checkpoint_latest.pt"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(ckpt_path.name)
        logging.info(f"Saved checkpoint to {ckpt_path}")
        return str(ckpt_path)

    assert env.training

    def should_save(i):
        if not aa.is_main_process():
            return False
        return i % checkpoint_interval == 0

    carry = env.reset()
    next_saved_keys = [
        # "command",
        # "command_",
        # "policy",
        # "priv",
        "done",
        "terminated",
        "truncated",
        "discount",
        "reward",
        "stats",
        "is_init",
        "adapt_hx",
        "episode_id",
    ]
    next_saved_keys.extend(policy.get_next_saved_keys())
    next_saved_keys = list(dict.fromkeys(next_saved_keys))

    start_iter = _start_iteration(
        cfg.get("checkpoint_path", None), int(getattr(env, "current_iter", 0))
    )
    env_frames = start_iter * frames_per_batch

    if hasattr(policy.cfg, "stages"):
        stages = policy.cfg.stages
    else:
        stages = ("",)

    tb_writer = None
    if aa.is_main_process():
        hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
        hydra_output_dir.mkdir(parents=True, exist_ok=True)
        default_run_name = (
            f"{cfg.exp_name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        setproctitle(default_run_name)

        cfg_save_path = hydra_output_dir / "cfg.yaml"
        OmegaConf.save(cfg, cfg_save_path)

        if bool(cfg.get("tensorboard", True)):
            tb_dir = hydra_output_dir / "tb"
            tb_writer = SummaryWriter(log_dir=str(tb_dir))
            tb_writer.add_text(
                "run/config",
                "```yaml\n" + OmegaConf.to_yaml(cfg) + "\n```",
                0,
            )
            logging.info(f"TensorBoard logs will be written to {tb_dir}")

    for stage in stages:
        policy.on_stage_start(stage, env)
        rollout_policy = policy.get_rollout_policy("train")

        with (
            torch.inference_mode(),
            set_exploration_type(ExplorationType.RANDOM),
            VecNorm.freeze(),
        ):
            tmp_carry = rollout_policy(carry.clone(True))
            if tmp_carry.device.type == "cuda":
                torch.cuda.empty_cache()
            tmp_td, _ = env.step_and_maybe_reset(tmp_carry.clone(False))
            tmp_td["next"] = tmp_td["next"].select(*next_saved_keys, strict=False)

        data_buf: TensorDict = (
            tmp_td.unsqueeze(-1).expand(env.num_envs, cfg.algo.train_every).clone()
        )

        progress = range(start_iter, total_iters)
        if aa.is_main_process():
            progress = tqdm(progress, desc=stage)

        for i in progress:
            if should_save(i):
                checkpoint_name = f"checkpoint_{i}"
                ckpt_path = save(policy, checkpoint_name)
                if ckpt_path is not None:
                    print(f"Latest checkpoint: {ckpt_path}")

            rollout_start = time.perf_counter()
            with ScopedTimer("rollout") as rollout_timer:
                with (
                    torch.inference_mode(),
                    set_exploration_type(ExplorationType.RANDOM),
                ):
                    if hasattr(env, "set_progress"):
                        env.set_progress(i)
                    for step in range(cfg.algo.train_every):
                        with ScopedTimer("policy_inference"):
                            carry = rollout_policy(carry)
                        td, carry = env.step_and_maybe_reset(carry)
                        td["next"] = td["next"].select(*next_saved_keys, strict=False)
                        data_buf[:, step] = td

                    if requires_rollout_value:
                        if hasattr(policy, "compute_rollout_values"):
                            policy.compute_rollout_values(data_buf, carry.copy())
                        else:
                            policy.critic(data_buf)
                            values = data_buf["state_value"]
                            last_value = policy.compute_value(carry.copy())[
                                "state_value"
                            ]
                            next_values = torch.cat(
                                [values[:, 1:], last_value.unsqueeze(1)], dim=1
                            )
                            data_buf["next", "state_value"] = torch.where(
                                data_buf["next", "done"],
                                values,
                                next_values,
                            )

            rollout_time = rollout_timer.last_time

            episode_stats.add(data_buf)
            env_frames += data_buf.numel()

            info = {}
            if i % log_interval == 0 and len(episode_stats):
                for k, v in sorted(episode_stats.pop().items(True, True)):
                    key = "train/" + ("/".join(k) if isinstance(k, tuple) else k)
                    info[key] = torch.mean(v.float()).item()

            with ScopedTimer("training") as training_timer:
                info.update(policy.train_op(data_buf))
            training_time = training_timer.last_time

            info.update(env.extra)
            info.update(env.stats_ema)

            if hasattr(policy, "step_schedule"):
                policy.step_schedule(i / total_iters)

            info["env_frames"] = env_frames * aa.get_world_size()
            info["performance/rollout_fps"] = (
                data_buf.numel() / rollout_time * aa.get_world_size()
            )
            info["performance/rollout_time"] = rollout_time
            info["performance/training_time"] = training_time
            info["performance/iter_time"] = time.perf_counter() - rollout_start

            if aa.is_main_process():
                # ScopedTimer.print_summary(clear=True, depth=5)
                # print(
                #     OmegaConf.to_yaml(
                #         {k: v for k, v in info.items() if isinstance(v, (float, int))}
                #     )
                # )
                metrics_record = _scalar_metrics(info)
                if tb_writer is not None:
                    for key, value in metrics_record.items():
                        if isinstance(value, (int, float)):
                            tb_writer.add_scalar(key, value, i)

    if aa.is_main_process():
        ckpt_path = save(policy, f"checkpoint_{total_iters}")
        # policy_eval = policy.get_rollout_policy("eval")
        # info, trajs, stats = evaluate(
        #     env, policy_eval, render=cfg.eval_render, seed=cfg.seed
        # )
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()
        print(f"Final checkpoint: {ckpt_path}")
    exit(0)


if __name__ == "__main__":
    main()
