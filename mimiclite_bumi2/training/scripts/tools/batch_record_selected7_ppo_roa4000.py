#!/usr/bin/env python3
"""Record seven selected score1 motions with PPO-ROA stage-3 checkpoint 4000."""

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
MOTION_ROOT = Path("/data2/zcx/datasets/any4hdmi-bumi-v2/motions")
CHECKPOINT = Path(
    "/data2/zcx/outputs/ppo-roa-bumi-full-20260901/"
    "stages/03-finetune/checkpoint_4000.pt"
)
OUT = ROOT / "outputs/selected7_score1_mujoco_ppo_roa4000_onepass"
RUNS = OUT / "runs"
VIDEOS = OUT / "videos"
STEP_DT = 0.02
RELATIVE_MOTIONS = [
    "score1/220705/Loop_Backward_Walk_001__A017_from_g1_bumi_v2.npz",
    "score1/220705/Loop_Forward_Jog_001__A017_from_g1_bumi_v2.npz",
    "score1/220705/Sideway_Walk_Left_001__A018_from_g1_bumi_v2.npz",
    "score1/220705/Turn_Start_Walk_0315_002__A018_from_g1_bumi_v2.npz",
    "score1/210707/dancecards1_CD_normal_003__A008_from_g1_bumi_v2.npz",
    "score1/221011/walk_ff_loop_360_001__A052_from_g1_bumi_v2.npz",
    "score1/210531/walk_forward_amateur_001__A001_from_g1_bumi_v2.npz",
]


def analyse(log_path: Path, expected: int) -> dict[str, object]:
    if not log_path.is_file():
        return {"recorded_steps": 0, "reset_count": 0, "finite": False}
    with np.load(log_path) as data:
        frames = np.asarray(data["motion_frame"])[:, 0].astype(np.int64)
        finite = all(
            bool(np.isfinite(data[key]).all())
            for key in ("root_pos", "root_quat", "joint_pos", "joint_vel", "policy_action")
        )
    return {
        "recorded_steps": int(len(frames)),
        "reset_count": int(np.count_nonzero(np.diff(frames) < 0)),
        "finite": finite,
        "last_frame": int(frames[-1]) if len(frames) else -1,
        "max_frame": int(frames.max()) if len(frames) else -1,
        "full_duration": bool(len(frames) == expected),
    }


def save(rows: list[dict[str, object]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "index", "motion", "source_frames", "duration_s", "recorded_steps",
        "reset_count", "finite", "last_frame", "max_frame", "full_duration",
        "returncode", "elapsed_s", "status", "video", "trajectory_npz",
    ]
    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    VIDEOS.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES="0", MUJOCO_GL="egl", HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_XET="1",
    )
    rows: list[dict[str, object]] = []
    for index, relative in enumerate(RELATIVE_MOTIONS, 1):
        motion = MOTION_ROOT / relative
        if not motion.is_file():
            raise FileNotFoundError(motion)
        with np.load(motion) as data:
            steps = int(data["qpos"].shape[0])
        run_dir = RUNS / f"{index:02d}"
        result_file = run_dir / "result.json"
        if result_file.is_file():
            row = json.loads(result_file.read_text(encoding="utf-8"))
            if Path(str(row.get("video", ""))).is_file():
                row["status"] = "skipped_existing"
                rows.append(row)
                save(rows)
                print(f"[{index}/7] SKIP {motion.stem}", flush=True)
                continue
        run_dir.mkdir(parents=True, exist_ok=True)
        trajectory = run_dir / "trajectory.npz"
        target = VIDEOS / f"{index:02d}_{motion.stem}_onepass.mp4"
        cmd = [
            str(PYTHON), "-u", str(PLAY),
            "task=tracking-bumi-v2", "task/motion=bumi/v2",
            "+exp=ppo_roa/finetune", "algo/ppo_roa/module=huge",
            "backend=mjlab", "device=cuda", "task.num_envs=1",
            "task.command.start_from_zero=true",
            "task.termination.root_pos_error.enabled=false",
            "~task.termination.body_pos_error",
            f"task.command.motion_cfgs.bumi_v2.path={MOTION_ROOT}",
            f'+task.command.motion_cfgs.bumi_v2.filenames=["{relative}"]',
            "+task/patches=teacher_future_t16",
            f"checkpoint_path={CHECKPOINT}", "headless=true",
            f"render_seconds={steps * STEP_DT}",
            f"+trajectory_log_path={trajectory}",
            f"hydra.run.dir={run_dir}", "hydra.job.chdir=true",
        ]
        print(f"[{index}/7] START {motion.stem}: {steps} steps ({steps * STEP_DT:.2f}s)", flush=True)
        started = time.monotonic()
        with (run_dir / "console.log").open("w", encoding="utf-8") as log:
            try:
                proc = subprocess.run(
                    cmd, cwd=ROOT, env=environment, stdout=log,
                    stderr=subprocess.STDOUT, timeout=300, check=False,
                )
                code = proc.returncode
            except subprocess.TimeoutExpired:
                code = 124
                log.write("\nBatch timeout after 300 seconds.\n")
        elapsed = time.monotonic() - started
        candidates = sorted(run_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        if candidates and candidates[-1].stat().st_size:
            shutil.copy2(candidates[-1], target)
        metrics = analyse(trajectory, steps)
        status = "recorded" if code == 0 and target.is_file() else "failed"
        row: dict[str, object] = {
            "index": index, "motion": relative, "source_frames": steps,
            "duration_s": round(steps * STEP_DT, 2), **metrics,
            "returncode": code, "elapsed_s": round(elapsed, 2), "status": status,
            "video": str(target) if target.is_file() else "",
            "trajectory_npz": str(trajectory) if trajectory.is_file() else "",
        }
        result_file.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(row)
        save(rows)
        print(f"[{index}/7] {status.upper()} {motion.stem}", flush=True)
    print(f"DONE: {OUT}", flush=True)


if __name__ == "__main__":
    main()
