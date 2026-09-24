#!/usr/bin/env python3
"""Export active-adaptation PPO/PPO-ROA checkpoints to deploy ONNX.

Unlike an architecture-specific exporter, this tool restores the complete
``task`` and ``algo`` configuration embedded in each checkpoint.  The policy's
own ``get_rollout_policy("deploy")`` therefore selects the correct PPO actor or
PPO-ROA student actor, matching VecNorm state, observation keys and ordering.

Run this script in the active-adaptation ``mjlab`` environment.  One or more
checkpoint paths may be supplied in the same invocation.

Names include the training run timestamp and checkpoint path identity. Re-exporting
the same checkpoint replaces its export; another checkpoint is never overwritten.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import onnx
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

import active_adaptation as aa
from active_adaptation.learning.modules.vecnorm import VecNorm
from active_adaptation.utils.export import export_onnx


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-") or "policy"


def _training_stamp(checkpoint: Path) -> str:
    """Read the run time, including for stages/03-finetune checkpoints."""
    for parent in checkpoint.parents:
        combined = re.match(
            r"^(\d{4})-(\d{2})-(\d{2})[_-](\d{2})-(\d{2})-(\d{2})(?:\D|$)",
            parent.name,
        )
        if combined:
            parts = combined.groups()
            return "".join(parts[:3]) + "_" + "".join(parts[3:])
        clock = re.match(r"^(\d{2})-(\d{2})-(\d{2})(?:\D|$)", parent.name)
        date = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", parent.parent.name)
        if clock and date:
            return "".join(date.groups()) + "_" + "".join(clock.groups())
    # Do not mislabel file mtime or export time as the training start time.
    return "run-time-unknown"


def _export_filename(checkpoint: Path, algo_name: str) -> str:
    checkpoint = checkpoint.expanduser().resolve()
    identity = hashlib.sha256(str(checkpoint).encode("utf-8")).hexdigest()[:12]
    return (
        f"{_training_stamp(checkpoint)}_{_safe_name(checkpoint.parent.name)}_"
        f"{_safe_name(checkpoint.stem)}_{_safe_name(algo_name)}_{identity}_deploy.onnx"
    )


def _check_export_owner(output_path: Path, checkpoint: Path) -> None:
    """Only overwrite an export whose sidecar identifies this exact checkpoint."""
    yaml_path = output_path.with_suffix(".yaml")
    if not output_path.exists() and not yaml_path.exists():
        return
    if yaml_path.is_file():
        with yaml_path.open(encoding="utf-8") as stream:
            metadata = yaml.safe_load(stream)
        owner = metadata.get("checkpoint") if isinstance(metadata, dict) else None
        if isinstance(owner, str) and Path(owner).expanduser().resolve() == checkpoint:
            return
    raise FileExistsError(
        f"refusing to overwrite {output_path}: existing export belongs to another "
        "checkpoint or has no verifiable checkpoint metadata"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint_cfg(checkpoint: Path) -> tuple[dict[str, Any], DictConfig]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "policy" not in state:
        raise ValueError(f"{checkpoint} is not an active-adaptation policy checkpoint")

    raw_cfg = state.get("cfg")
    if raw_cfg is None:
        cfg_path = checkpoint.parent / "cfg.yaml"
        if not cfg_path.is_file():
            raise FileNotFoundError(
                f"checkpoint has no embedded cfg and {cfg_path} does not exist"
            )
        cfg = OmegaConf.load(cfg_path)
    elif isinstance(raw_cfg, DictConfig):
        cfg = copy.deepcopy(raw_cfg)
    else:
        cfg = OmegaConf.create(raw_cfg)

    if "task" not in cfg or "algo" not in cfg:
        raise KeyError("checkpoint config must contain both task and algo sections")
    OmegaConf.set_struct(cfg, False)
    return state, cfg


def _checkpoint_kind(state: dict[str, Any], cfg: DictConfig) -> str:
    policy_keys = set(state["policy"])
    if {"encoder_student", "actor_student"}.issubset(policy_keys):
        return "ppo-roa"
    if "actor" in policy_keys:
        return "ppo"
    return str(OmegaConf.select(cfg, "algo.name") or "policy")


def _tensor_key_name(key: Any) -> str:
    if isinstance(key, str):
        return key
    if isinstance(key, tuple):
        return "_".join(str(part) for part in key)
    return str(key)


def _onnx_shape(value: onnx.ValueInfoProto) -> list[int | str | None]:
    result: list[int | str | None] = []
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            result.append(dimension.dim_param)
        else:
            result.append(None)
    return result


def _graph_interface(path: Path) -> dict[str, list[dict[str, Any]]]:
    model = onnx.load(str(path))
    return {
        "inputs": [
            {"name": value.name, "shape": _onnx_shape(value)}
            for value in model.graph.input
        ],
        "outputs": [
            {"name": value.name, "shape": _onnx_shape(value)}
            for value in model.graph.output
        ],
    }


def _container(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _asset_metadata(asset: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "joint_names_simulation": list(getattr(asset.cfg, "joint_names_simulation", [])),
        "body_names_simulation": list(getattr(asset.cfg, "body_names_simulation", [])),
        "joint_kp": {},
        "joint_kd": {},
        "default_joint_pos": {},
    }
    init_state = getattr(asset.cfg, "init_state", None)
    if init_state is not None and isinstance(getattr(init_state, "joint_pos", None), dict):
        metadata["default_joint_pos"] = dict(init_state.joint_pos)

    for actuator in getattr(asset, "actuators", []):
        actuator_cfg = getattr(actuator, "cfg", None)
        if actuator_cfg is None:
            continue
        names = (
            getattr(actuator_cfg, "target_names_expr", None)
            or getattr(actuator_cfg, "joint_names_expr", None)
            or []
        )
        for name in names:
            stiffness = getattr(actuator_cfg, "stiffness", None)
            damping = getattr(actuator_cfg, "damping", None)
            if stiffness is not None:
                metadata["joint_kp"][name] = float(stiffness)
            if damping is not None:
                metadata["joint_kd"][name] = float(damping)
    return metadata


def _build_metadata(
    checkpoint: Path,
    cfg: DictConfig,
    state: dict[str, Any],
    env: Any,
    deploy_policy: Any,
    fake_input: Any,
) -> dict[str, Any]:
    input_keys = list(deploy_policy.in_keys)
    output_keys = list(deploy_policy.out_keys)
    observation_cfg = OmegaConf.select(cfg, "task.observation")
    observation_layout: list[dict[str, Any]] = []
    for key in input_keys:
        key_name = _tensor_key_name(key)
        item: dict[str, Any] = {
            "key": key_name,
            "shape": list(fake_input[key].shape),
        }
        if observation_cfg is not None and key in observation_cfg:
            item["config"] = _container(observation_cfg[key])
        observation_layout.append(item)

    action_manager = env.action_manager
    asset = env.scene.articulations["robot"]
    metadata: dict[str, Any] = {
        "format": "active_adaptation_deploy_onnx_v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_kind": _checkpoint_kind(state, cfg),
        "task_name": str(OmegaConf.select(cfg, "task.name") or "unknown"),
        "algo_name": str(OmegaConf.select(cfg, "algo.name") or "unknown"),
        "algo_phase": OmegaConf.select(cfg, "algo.phase"),
        "observation_order": [_tensor_key_name(key) for key in input_keys],
        "observation_layout": observation_layout,
        "output_order": [_tensor_key_name(key) for key in output_keys],
        "policy_joint_names": list(getattr(action_manager, "joint_names", [])),
        "action_scale": getattr(action_manager, "action_scaling").detach().cpu().tolist(),
        "asset": _asset_metadata(asset),
    }

    command = getattr(env, "command_manager", None)
    if command is not None:
        for output_name, attribute in (
            ("future_steps", "future_steps"),
            ("tracking_body_names", "tracking_body_names"),
            ("tracking_joint_names", "tracking_joint_names"),
            ("root_body_name", "root_body_name"),
            ("anchor_body_name", "anchor_body_name"),
        ):
            value = getattr(command, attribute, None)
            if value is not None:
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().tolist()
                elif isinstance(value, tuple):
                    value = list(value)
                metadata[output_name] = value
    return metadata


def _validate_export(
    onnx_path: Path,
    deploy_policy: Any,
    joint_names: list[str],
) -> dict[str, list[dict[str, Any]]]:
    graph = _graph_interface(onnx_path)
    expected_inputs = [_tensor_key_name(key) for key in deploy_policy.in_keys]
    actual_inputs = [item["name"] for item in graph["inputs"]]
    if actual_inputs != expected_inputs:
        raise RuntimeError(
            "ONNX observation order mismatch: "
            f"expected={expected_inputs}, actual={actual_inputs}"
        )

    action_output = next(
        (item for item in graph["outputs"] if item["name"] == "action"), None
    )
    if action_output is None:
        raise RuntimeError(f"ONNX has no action output: {graph['outputs']}")
    if action_output["shape"][-1:] != [len(joint_names)]:
        raise RuntimeError(
            f"action shape {action_output['shape']} does not match "
            f"{len(joint_names)} policy joints"
        )
    return graph


@VecNorm.freeze()
def export_checkpoint(
    checkpoint: Path,
    output_path: Path | None,
    *,
    backend: str,
    device: str,
    force: bool,
    initialize_framework: bool = True,
) -> tuple[Path, Path]:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    state, cfg = _load_checkpoint_cfg(checkpoint)
    kind = _checkpoint_kind(state, cfg)
    algo_name = str(OmegaConf.select(cfg, "algo.name") or kind)

    if output_path is None:
        output_path = (
            checkpoint.parent
            / "exported"
            / _export_filename(checkpoint, algo_name)
        )
    output_path = output_path.expanduser().resolve()
    if output_path.suffix.lower() != ".onnx":
        raise ValueError(f"output path must end in .onnx: {output_path}")
    yaml_path = output_path.with_suffix(".yaml")
    # Retain force for API compatibility, but never let it bypass ownership.
    _check_export_owner(output_path, checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg.backend = backend
    cfg.device = device
    cfg.headless = True
    cfg.checkpoint_path = str(checkpoint)
    cfg.task.num_envs = 1
    if OmegaConf.select(cfg, "algo.compile") is not None:
        cfg.algo.compile = False
    if OmegaConf.select(cfg, "algo.compile_rollout") is not None:
        cfg.algo.compile_rollout = False
    OmegaConf.resolve(cfg)
    if initialize_framework:
        aa.init(cfg, auto_rank=True)

    from active_adaptation.helpers import make_env_policy

    env, policy = make_env_policy(
        cfg.task,
        cfg.algo,
        seed=int(cfg.seed),
        headless=True,
        device=str(cfg.device),
        checkpoint_path=str(checkpoint),
    )
    deploy_policy = copy.deepcopy(policy.get_rollout_policy("deploy")).eval()
    fake_input = env.observation_spec[0].rand().to(env.device)
    metadata = _build_metadata(
        checkpoint, cfg, state, env, deploy_policy, fake_input
    )

    try:
        export_onnx(deploy_policy, fake_input, str(output_path), meta=metadata)
        graph = _validate_export(
            output_path, deploy_policy, metadata["policy_joint_names"]
        )
        metadata["onnx_interface"] = graph
        metadata["onnx_sha256"] = _sha256(output_path)
        with yaml_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(metadata, stream, sort_keys=False, allow_unicode=True)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    print(
        f"[OK] {kind}: {checkpoint.name} -> {output_path}\n"
        f"     inputs={metadata['observation_order']} "
        f"outputs={metadata['output_order']} "
        f"joints={len(metadata['policy_joint_names'])}"
    )
    return output_path, yaml_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional common output directory. Default: exported/ beside each checkpoint.",
    )
    parser.add_argument(
        "--backend", choices=("mjlab", "mujoco"), default="mjlab"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--force", action="store_true",
        help="Compatibility flag; same-checkpoint exports are replaced automatically. "
        "Never permits overwriting another checkpoint's export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.checkpoints) > 1 and args.output_dir is None:
        outputs: list[Path | None] = [None] * len(args.checkpoints)
    elif args.output_dir is None:
        outputs = [None]
    else:
        output_dir = args.output_dir.expanduser().resolve()
        outputs = []
        for checkpoint in args.checkpoints:
            state, cfg = _load_checkpoint_cfg(checkpoint.expanduser().resolve())
            algo_name = str(OmegaConf.select(cfg, "algo.name") or "policy")
            filename = _export_filename(checkpoint, algo_name)
            outputs.append(output_dir / filename)
            del state

    results = []
    for index, (checkpoint, output) in enumerate(
        zip(args.checkpoints, outputs, strict=True)
    ):
        onnx_path, yaml_path = export_checkpoint(
            checkpoint,
            output,
            backend=args.backend,
            device=args.device,
            force=args.force,
            initialize_framework=index == 0,
        )
        results.append({"onnx": str(onnx_path), "yaml": str(yaml_path)})
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
