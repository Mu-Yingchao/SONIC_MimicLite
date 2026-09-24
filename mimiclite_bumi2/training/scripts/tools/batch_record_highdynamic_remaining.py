#!/usr/bin/env python3
"""Record one reference cycle and one PPO-ROA rollout for the other 24 motions."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np


ROOT = Path("/home/user/zcx/active-adaptation")
PYTHON = ROOT / "venv/mjlab/.venv/bin/python"
PLAY = ROOT / "projects/mimic-lite/scripts/play.py"
MOTION_ROOT = Path("/data2/zcx/datasets/any4hdmi-bumi-v2/motions/score_highdynamic")
CHECKPOINT = Path(
    "/data2/zcx/outputs/2026-09-08/ppo-roa-highdynamic-only/"
    "stages/03-finetune/checkpoint_1000.pt"
)
OUT = ROOT / "outputs/highdynamic_remaining24_mujoco_onepass"
RUNS = OUT / "runs"
POLICY = OUT / "policy"
REFERENCE = OUT / "reference"
STEP_DT = 0.02
COMPLETED = {
    "bumi2_前手翻_30fps",
    "bumi2_前空翻_30fps",
    "bumi2_后手翻_30fps",
    "bumi2_垫布韦伯斯特_30fps",
    "bumi2_左侧手翻_30fps",
    "bumi2_韦伯斯特_30fps",
}


def env() -> dict[str, str]:
    result = os.environ.copy()
    result.update(
        CUDA_VISIBLE_DEVICES="0",
        MUJOCO_GL="egl",
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        HF_HUB_DISABLE_XET="1",
    )
    return result


def command(motion: Path, run_dir: Path, trajectory: Path, steps: int, replay: bool) -> list[str]:
    relative = motion.relative_to(MOTION_ROOT.parent).as_posix()
    args = [
        str(PYTHON), "-u", str(PLAY),
        "task=tracking-bumi-v2",
        "task/motion=bumi/v2",
        "+exp=ppo_roa/finetune",
        "algo/ppo_roa/module=huge",
        "backend=mjlab",
        "device=cuda",
        "task.num_envs=1",
        "task.command.start_from_zero=true",
        "task.termination.root_pos_error.enabled=false",
        "~task.termination.body_pos_error",
        f"task.command.motion_cfgs.bumi_v2.path={MOTION_ROOT}",
        f'+task.command.motion_cfgs.bumi_v2.filenames=["{relative}"]',
        "+task/patches=teacher_future_t16",
        f"checkpoint_path={CHECKPOINT}",
        "headless=true",
        f"render_seconds={steps * STEP_DT}",
        f"+trajectory_log_path={trajectory}",
        f"hydra.run.dir={run_dir}",
        "hydra.job.chdir=true",
    ]
    if replay:
        args.append("+task.command.replay_motion=true")
    return args


def run_one(motion: Path, run_dir: Path, steps: int, replay: bool) -> tuple[int, float, Path | None, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    trajectory = run_dir / "trajectory.npz"
    started = time.monotonic()
    with (run_dir / "console.log").open("w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(
                command(motion, run_dir, trajectory, steps, replay),
                cwd=ROOT,
                env=env(),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=240,
                check=False,
            )
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = 124
            log.write("\nBatch timeout after 240 seconds.\n")
    videos = sorted(run_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    video = videos[-1] if videos and videos[-1].stat().st_size else None
    return code, time.monotonic() - started, video, trajectory


def cycle_steps(trajectory: Path, fallback: int) -> tuple[int, int, int]:
    if not trajectory.is_file():
        return fallback, 0, -1
    with np.load(trajectory) as data:
        frames = np.asarray(data["motion_frame"])[:, 0].astype(np.int64)
    wraps = np.flatnonzero(np.diff(frames) < 0)
    steps = int(wraps[0] + 1) if len(wraps) else int(len(frames))
    return max(1, steps), int(len(frames)), int(frames.max()) if len(frames) else -1


def write_manifest(rows: list[dict[str, object]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "index", "motion", "stem", "source_frames", "cycle_steps",
            "reference_returncode", "policy_returncode", "reference_video",
            "policy_video", "reference_recorded_steps", "reference_max_frame",
            "reference_elapsed_s", "policy_elapsed_s", "status",
        ]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    for directory in (RUNS, POLICY, REFERENCE):
        directory.mkdir(parents=True, exist_ok=True)
    motions = [p for p in sorted(MOTION_ROOT.rglob("*.npz")) if p.stem not in COMPLETED]
    if len(motions) != 24:
        raise RuntimeError(f"Expected 24 remaining motions, found {len(motions)}")
    rows: list[dict[str, object]] = []
    for index, motion in enumerate(motions, 1):
        result_file = RUNS / f"{index:02d}" / "result.json"
        if result_file.is_file():
            row = json.loads(result_file.read_text(encoding="utf-8"))
            if Path(str(row.get("policy_video", ""))).is_file() and Path(str(row.get("reference_video", ""))).is_file():
                row["status"] = "skipped_existing"
                rows.append(row)
                write_manifest(rows)
                print(f"[{index:02d}/24] SKIP {motion.stem}", flush=True)
                continue

        with np.load(motion) as data:
            source_frames = int(data["qpos"].shape[0])
        base = RUNS / f"{index:02d}"
        print(f"[{index:02d}/24] REF    {motion.stem}", flush=True)
        ref_code, ref_elapsed, ref_raw, ref_traj = run_one(
            motion, base / "reference", source_frames, True
        )
        steps, ref_recorded, ref_max = cycle_steps(ref_traj, source_frames)

        print(f"[{index:02d}/24] POLICY {motion.stem} ({steps} steps)", flush=True)
        pol_code, pol_elapsed, pol_raw, _ = run_one(
            motion, base / "policy", steps, False
        )
        ref_dest = REFERENCE / f"{index:02d}_reference.mp4"
        pol_dest = POLICY / f"{index:02d}_policy.mp4"
        if ref_raw:
            shutil.copy2(ref_raw, ref_dest)
        if pol_raw:
            shutil.copy2(pol_raw, pol_dest)
        status = "recorded" if ref_dest.is_file() and pol_dest.is_file() else "failed"
        row: dict[str, object] = {
            "index": index,
            "motion": motion.relative_to(MOTION_ROOT.parent).as_posix(),
            "stem": motion.stem,
            "source_frames": source_frames,
            "cycle_steps": steps,
            "reference_returncode": ref_code,
            "policy_returncode": pol_code,
            "reference_video": str(ref_dest) if ref_dest.is_file() else "",
            "policy_video": str(pol_dest) if pol_dest.is_file() else "",
            "reference_recorded_steps": ref_recorded,
            "reference_max_frame": ref_max,
            "reference_elapsed_s": round(ref_elapsed, 2),
            "policy_elapsed_s": round(pol_elapsed, 2),
            "status": status,
        }
        result_file.parent.mkdir(parents=True, exist_ok=True)
        result_file.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(row)
        write_manifest(rows)
        print(f"[{index:02d}/24] {status.upper()} {motion.stem}", flush=True)
    print(f"DONE: {OUT}", flush=True)


if __name__ == "__main__":
    main()
