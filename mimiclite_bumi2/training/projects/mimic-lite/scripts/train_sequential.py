#!/usr/bin/env python3
from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import active_adaptation as aa

FILE_PATH = Path(__file__).resolve().parent
CONFIG_PATH = FILE_PATH.parent / "cfg"
ROOT_PATH = FILE_PATH.parents[2]
ORCHESTRATOR_KEYS = {
    "stages",
    "nproc_per_node",
    "script",
    "random_suffix_length",
    "tag",
    "warm_start_checkpoint",
    "stage_iters",
    "dry_run",
}


def _resolve_script_path(script: str) -> str:
    script_path = Path(script)
    if not script_path.is_absolute():
        script_path = ROOT_PATH / script_path
    return str(script_path.resolve())


def _sanitize_tag(tag: str) -> str:
    return tag.replace("/", "_").replace(" ", "_")


def _stage_run_dir(stage_name: str, index: int, base_dir: Path | None = None) -> str:
    if base_dir is None:
        base_dir = Path.cwd()
    stage_slug = _sanitize_tag(stage_name)
    stage_dir = (base_dir / "stages" / f"{index + 1:02d}-{stage_slug}").resolve()
    return str(stage_dir)


def _collect_cli_overrides() -> list[str]:
    cli_overrides = []
    script_name = Path(__file__).name
    for arg in sys.argv[1:]:
        if arg.startswith("hydra.") or script_name in arg:
            continue
        top_level_key = arg.lstrip("+").split("=", 1)[0].split(".", 1)[0]
        if top_level_key in ORCHESTRATOR_KEYS:
            continue
        cli_overrides.append(arg)
    return cli_overrides


def _normalize_stage(stage, index: int) -> dict[str, object]:
    if isinstance(stage, DictConfig):
        return {
            "name": str(stage.get("name", f"stage-{index + 1}")),
            "overrides": [str(item) for item in stage.get("overrides", [])],
            "load_checkpoint_from_previous": bool(
                stage.get("load_checkpoint_from_previous", index > 0)
            ),
        }
    stage_name = str(stage)
    return {
        "name": stage_name,
        "overrides": [f"algo={stage_name}"],
        "load_checkpoint_from_previous": index > 0,
    }


def _to_stage_items(stages_cfg) -> list[dict[str, object]]:
    stages = list(stages_cfg)
    if not stages:
        raise ValueError("No stages configured.")
    return [_normalize_stage(stage, index) for index, stage in enumerate(stages)]


def _has_future_checkpoint_consumer(
    stages: list[dict[str, object]], current_index: int
) -> bool:
    return any(
        bool(stage["load_checkpoint_from_previous"])
        for stage in stages[current_index + 1 :]
    )


def _run_command(command: list[str]) -> None:
    print(f">>> {shlex.join(command)}", flush=True)
    process = subprocess.Popen(command, cwd=ROOT_PATH)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


@hydra.main(
    config_path=str(CONFIG_PATH), config_name="train_sequential", version_base=None
)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    cli_overrides = _collect_cli_overrides()
    stages = _to_stage_items(cfg.stages)
    script_path = _resolve_script_path(str(cfg.script))
    sequential_run_dir = Path.cwd()
    warm_start = cfg.get("warm_start_checkpoint")
    if warm_start:
        warm_start = str(Path(hydra.utils.to_absolute_path(warm_start)).resolve())
        if not Path(warm_start).is_file():
            raise FileNotFoundError(warm_start)
        if [stage["name"] for stage in stages] != ["train", "adapt", "finetune"]:
            raise ValueError("Warm start requires stages: train, adapt, finetune")
        for phase in ("train", "adapt", "finetune"):
            value = cfg.stage_iters[phase]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"stage_iters.{phase} must be a positive integer")
        # These choices come from the checkpoint and phase recipes, never from
        # a default module preset or a shared total_iters override.
        for ov in cli_overrides:
            key = ov.lstrip("+").split("=", 1)[0]
            if key in ("algo", "exp", "checkpoint_path", "total_iters", "warm_start_phase") or key.startswith(("algo/", "algo.")):
                raise ValueError(f"Warm start controls {key}; use stage_iters for phase budgets")

    print("=" * 80)
    print("Detected command-line overrides applied to all stages:")
    if cli_overrides:
        for ov in cli_overrides:
            print(f"  - {ov}")
    else:
        print("  - None")
    print("=" * 80)

    previous_checkpoint_path = None

    for i, stage in enumerate(stages):
        stage_name_value = str(stage["name"])
        stage_specific_overrides = list(stage["overrides"])
        load_from_previous = bool(stage["load_checkpoint_from_previous"])

        print("\n" + "=" * 80)
        print(f"Preparing stage {i + 1}/{len(stages)}: {stage_name_value}")
        print("=" * 80)

        stage_run_dir = _stage_run_dir(
            stage_name_value,
            i,
            base_dir=sequential_run_dir,
        )
        stage_overrides = cli_overrides.copy()
        if warm_start:
            source_checkpoint = warm_start if i == 0 else previous_checkpoint_path
            stage_overrides.extend([
                "algo=from_checkpoint",
                f"+warm_start_phase={stage_name_value}",
                f"total_iters={cfg.stage_iters[stage_name_value]}",
                f"checkpoint_path={source_checkpoint}",
            ])
        else:
            stage_overrides.extend(stage_specific_overrides)
        stage_overrides.append(f"hydra.run.dir={stage_run_dir}")

        if not warm_start and previous_checkpoint_path and load_from_previous:
            stage_overrides.append(f"checkpoint_path={previous_checkpoint_path}")
            print(f"Loading checkpoint from previous stage: {previous_checkpoint_path}")

        print(f"Stage run dir: {stage_run_dir}")

        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={cfg.nproc_per_node}",
            script_path,
            *stage_overrides,
        ]

        print(f"{'Previewing' if cfg.dry_run else 'Starting'} stage '{stage_name_value}'")
        if cfg.dry_run:
            print(f">>> {shlex.join(command)}", flush=True)
        else:
            _run_command(command)
            if _has_future_checkpoint_consumer(stages, i):
                latest = Path(stage_run_dir) / "checkpoint_latest.pt"
                if not latest.is_file():
                    raise FileNotFoundError(f"Stage produced no checkpoint: {latest}")
        if not cfg.dry_run:
            print(f"Child process for stage '{stage_name_value}' finished")

        if _has_future_checkpoint_consumer(stages, i):
            previous_checkpoint_path = str(Path(stage_run_dir) / "checkpoint_latest.pt")
            print(
                "Recorded local checkpoint for downstream stages: "
                f"{previous_checkpoint_path}"
            )
        else:
            previous_checkpoint_path = None

        print(f"{'Previewed' if cfg.dry_run else 'Completed'} stage {i + 1}/{len(stages)}: {stage_name_value}")
        print("=" * 80)

    print("\nPreview finished; no training launched" if cfg.dry_run else "\nAll training stages finished")


if __name__ == "__main__":
    main()
