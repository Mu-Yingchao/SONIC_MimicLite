#!/usr/bin/env python3
"""Record one PPO-ROA pass for each of the other 24 high-dynamic motions."""

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
OUT = ROOT / "outputs/highdynamic_remaining24_mujoco_ppo_roa_finetune1000_onepass"
RUNS = OUT / "runs"
VIDEOS = OUT / "videos"
STEP_DT = 0.02
ALREADY_RECORDED = {
    "bumi2_前手翻_30fps", "bumi2_前空翻_30fps", "bumi2_后手翻_30fps",
    "bumi2_垫布韦伯斯特_30fps", "bumi2_左侧手翻_30fps", "bumi2_韦伯斯特_30fps",
}


def save_manifest(rows: list[dict[str, object]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "index", "motion", "stem", "source_frames", "recorded_steps",
        "last_frame", "max_frame", "early_reset_count", "completed",
        "returncode", "elapsed_s", "video", "trajectory_npz", "status",
    ]
    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def assess(path: Path, expected: int) -> dict[str, object]:
    if not path.is_file():
        return {"recorded_steps": 0, "last_frame": -1, "max_frame": -1,
                "early_reset_count": 0, "completed": False}
    with np.load(path) as data:
        frames = np.asarray(data["motion_frame"])[:, 0].astype(np.int64)
        finite = all(
            np.isfinite(data[key]).all()
            for key in ("root_pos", "root_quat", "joint_pos", "joint_vel", "policy_action")
        )
    resets = np.flatnonzero(np.diff(frames) < 0)
    # A wrap on the final transition is natural; earlier wraps indicate reset/failure.
    early = [int(i) for i in resets if i + 1 < expected - 1]
    completed = bool(finite and len(frames) == expected and not early)
    return {
        "recorded_steps": int(len(frames)),
        "last_frame": int(frames[-1]) if len(frames) else -1,
        "max_frame": int(frames.max()) if len(frames) else -1,
        "early_reset_count": len(early),
        "completed": completed,
    }


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    VIDEOS.mkdir(parents=True, exist_ok=True)
    motions = [p for p in sorted(MOTION_ROOT.rglob("*.npz")) if p.stem not in ALREADY_RECORDED]
    if len(motions) != 24:
        raise RuntimeError(f"Expected 24 remaining motions, found {len(motions)}")
    rows: list[dict[str, object]] = []
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES="0", MUJOCO_GL="egl", HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_XET="1",
    )
    for index, motion in enumerate(motions, 1):
        run_dir = RUNS / f"{index:02d}"
        result_file = run_dir / "result.json"
        if result_file.is_file():
            row = json.loads(result_file.read_text(encoding="utf-8"))
            if Path(str(row.get("video", ""))).is_file():
                row["status"] = "skipped_existing"
                rows.append(row)
                save_manifest(rows)
                print(f"[{index:02d}/24] SKIP {motion.stem}", flush=True)
                continue
        run_dir.mkdir(parents=True, exist_ok=True)
        trajectory = run_dir / "trajectory.npz"
        with np.load(motion) as data:
            steps = int(data["qpos"].shape[0])
        relative = motion.relative_to(MOTION_ROOT.parent).as_posix()
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
            "+task/patches=teacher_future_t16", f"checkpoint_path={CHECKPOINT}",
            "headless=true", f"render_seconds={steps * STEP_DT}",
            f"+trajectory_log_path={trajectory}", f"hydra.run.dir={run_dir}",
            "hydra.job.chdir=true",
        ]
        print(f"[{index:02d}/24] START {motion.stem} ({steps} steps)", flush=True)
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
        raw = sorted(run_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        target = VIDEOS / f"{index:02d}_{motion.stem}_onepass.mp4"
        if raw and raw[-1].stat().st_size:
            shutil.copy2(raw[-1], target)
        metrics = assess(trajectory, steps)
        status = "completed" if code == 0 and target.is_file() and metrics["completed"] else "recorded_with_reset"
        row: dict[str, object] = {
            "index": index, "motion": relative, "stem": motion.stem,
            "source_frames": steps, **metrics, "returncode": code,
            "elapsed_s": round(elapsed, 2),
            "video": str(target) if target.is_file() else "",
            "trajectory_npz": str(trajectory) if trajectory.is_file() else "",
            "status": status,
        }
        result_file.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(row)
        save_manifest(rows)
        print(f"[{index:02d}/24] {status.upper()} {motion.stem}", flush=True)
    print(f"DONE: {OUT}", flush=True)


if __name__ == "__main__":
    main()
