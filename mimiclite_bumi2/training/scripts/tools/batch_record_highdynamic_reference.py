#!/usr/bin/env python3
"""Record reference replay for the six motions used by the comparison videos."""

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
MOTION_ROOT = Path(
    "/data2/zcx/datasets/any4hdmi-bumi-v2/motions/score_highdynamic"
)
CHECKPOINT = Path(
    "/data2/zcx/outputs/2026-09-08/ppo-roa-highdynamic-only/"
    "stages/03-finetune/checkpoint_1000.pt"
)
OUT = ROOT / "outputs/highdynamic_completed6_mujoco_reference_onepass"
RUNS = OUT / "runs"
SUCCESS = OUT / "success"
FAILED = OUT / "failed"
STEP_DT = 0.02
COMPLETED_MOTION_STEMS = {
    "bumi2_前手翻_30fps",
    "bumi2_前空翻_30fps",
    "bumi2_后手翻_30fps",
    "bumi2_垫布韦伯斯特_30fps",
    "bumi2_左侧手翻_30fps",
    "bumi2_韦伯斯特_30fps",
}
RUNTIME_FRAME_OVERRIDES = {
    # These cached motions are trimmed/resampled relative to their source qpos.
    "bumi2_前空翻_30fps": 70,
    "bumi2_垫布韦伯斯特_30fps": 103,
    "bumi2_左侧手翻_30fps": 90,
}


def analyse(
    log_path: Path,
    source_frames: int,
    expected_steps: int,
    record_seconds: float,
) -> dict[str, object]:
    result: dict[str, object] = {
        "source_frames": source_frames,
        "expected_steps": expected_steps,
        "record_seconds": record_seconds,
    }
    if not log_path.is_file():
        return {**result, "passed": False, "reason": "trajectory log missing"}

    with np.load(log_path) as data:
        frames = np.asarray(data["motion_frame"])[:, 0].astype(np.int64)
        finite = True
        for key in (
            "root_pos",
            "root_quat",
            "joint_pos",
            "joint_vel",
            "policy_action",
        ):
            finite = finite and bool(np.isfinite(data[key]).all())

    resets = np.flatnonzero(np.diff(frames) < 0)
    early_resets = [
        int(i)
        for i in resets
        if int(frames[i]) < max(0, source_frames - 3)
    ]
    completed_cycles = sum(
        int(frames[i]) >= max(0, source_frames - 3) for i in resets
    )
    reached_end = len(frames) == expected_steps
    passed = finite and reached_end and not early_resets
    if not finite:
        reason = "non-finite trajectory"
    elif not reached_end:
        reason = f"incomplete recording: {len(frames)}/{expected_steps} steps"
    elif early_resets:
        reason = f"early termination at steps {early_resets[:8]}"
    else:
        reason = "passed one complete motion"
    return {
        **result,
        "passed": passed,
        "reason": reason,
        "recorded_steps": int(len(frames)),
        "finite": finite,
        "reset_count": int(len(resets)),
        "early_reset_count": int(len(early_resets)),
        "completed_cycles": int(completed_cycles),
        "first_frame": int(frames[0]) if len(frames) else None,
        "last_frame": int(frames[-1]) if len(frames) else None,
        "max_frame": int(frames.max()) if len(frames) else None,
    }


def write_manifest(rows: list[dict[str, object]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "motion",
        "status",
        "passed",
        "reason",
        "video",
        "trajectory_npz",
        "source_frames",
        "recorded_steps",
        "completed_cycles",
        "early_reset_count",
        "returncode",
        "elapsed_s",
    ]
    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    for directory in (RUNS, SUCCESS, FAILED):
        directory.mkdir(parents=True, exist_ok=True)
    motions = [
        path
        for path in sorted(MOTION_ROOT.rglob("*.npz"))
        if path.stem in COMPLETED_MOTION_STEMS
    ]
    if len(motions) != len(COMPLETED_MOTION_STEMS):
        found = {path.stem for path in motions}
        raise RuntimeError(
            f"Missing selected motions: {sorted(COMPLETED_MOTION_STEMS - found)}"
        )
    rows: list[dict[str, object]] = []

    for index, motion in enumerate(motions, 1):
        stem = motion.stem
        # Hydra's override lexer rejects Unicode in trajectory_log_path and
        # hydra.run.dir. Keep internal run paths ASCII; final videos retain
        # the original motion name.
        run_dir = RUNS / (
            f"{index:02d}-{stem}" if stem.isascii() else f"{index:02d}-motion"
        )
        trajectory = run_dir / "trajectory.npz"
        result_path = run_dir / "result.json"
        run_dir.mkdir(parents=True, exist_ok=True)

        previous = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.is_file()
            else None
        )
        previous_video = Path(str(previous.get("video", ""))) if previous else None
        if (
            previous
            and previous.get("recorded_steps") == previous.get("expected_steps")
            and previous_video
            and previous_video.is_file()
        ):
            previous["status"] = "skipped_existing"
            rows.append(previous)
            write_manifest(rows)
            print(f"[{index:02d}/{len(motions)}] SKIP {stem}", flush=True)
            continue

        with np.load(motion) as source:
            source_frames = int(source["qpos"].shape[0])
        expected_steps = RUNTIME_FRAME_OVERRIDES.get(stem, source_frames)
        record_seconds = expected_steps * STEP_DT
        relative_motion = motion.relative_to(MOTION_ROOT.parent).as_posix()
        command = [
            str(PYTHON),
            "-u",
            str(PLAY),
            "task=tracking-bumi-v2",
            "task/motion=bumi/v2",
            "+exp=ppo_roa/finetune",
            "algo/ppo_roa/module=huge",
            "backend=mjlab",
            "device=cuda",
            "task.num_envs=1",
            "task.command.start_from_zero=true",
            "+task.command.replay_motion=true",
            "task.termination.root_pos_error.enabled=false",
            "~task.termination.body_pos_error",
            "+task.command.viz.ghost_color=[1.0,0.1,0.1,0.85]",
            f"task.command.motion_cfgs.bumi_v2.path={MOTION_ROOT}",
            f'+task.command.motion_cfgs.bumi_v2.filenames=["{relative_motion}"]',
            "+task/patches=teacher_future_t16",
            f"checkpoint_path={CHECKPOINT}",
            "headless=true",
            f"render_seconds={record_seconds}",
            f"+trajectory_log_path={trajectory}",
            f"hydra.run.dir={run_dir}",
            "hydra.job.chdir=true",
        ]
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "MUJOCO_GL": "egl",
                "HF_HUB_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "HF_HUB_DISABLE_XET": "1",
            }
        )
        print(f"[{index:02d}/{len(motions)}] START {stem}", flush=True)
        started = time.monotonic()
        with (run_dir / "console.log").open("w", encoding="utf-8") as console:
            try:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=console,
                    stderr=subprocess.STDOUT,
                    timeout=180,
                    check=False,
                )
                returncode = completed.returncode
            except subprocess.TimeoutExpired:
                returncode = 124
                console.write("\nBatch timeout after 180 seconds.\n")
        elapsed = time.monotonic() - started
        assessment = analyse(
            trajectory,
            source_frames,
            expected_steps,
            record_seconds,
        )
        video_candidates = sorted(run_dir.glob("*.mp4"))
        valid_video = bool(video_candidates and video_candidates[-1].stat().st_size > 0)
        passed = bool(assessment["passed"] and returncode == 0 and valid_video)
        target_dir = SUCCESS if passed else FAILED
        target_video = target_dir / f"{index:02d}_{stem}_onepass.mp4"
        if valid_video:
            shutil.copy2(video_candidates[-1], target_video)
        row: dict[str, object] = {
            "motion": relative_motion,
            "status": "passed" if passed else "failed",
            **assessment,
            "passed": passed,
            "returncode": returncode,
            "elapsed_s": round(elapsed, 2),
            "video": str(target_video) if valid_video else "",
            "trajectory_npz": str(trajectory) if trajectory.is_file() else "",
        }
        if returncode != 0:
            row["reason"] = f"play returncode={returncode}; {row['reason']}"
        elif not valid_video:
            row["reason"] = f"video missing; {row['reason']}"
        result_path.write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rows.append(row)
        write_manifest(rows)
        print(
            f"[{index:02d}/{len(motions)}] {row['status'].upper()} "
            f"{stem}: {row['reason']} ({elapsed:.1f}s)",
            flush=True,
        )

    passed_count = sum(bool(row.get("passed")) for row in rows)
    print(f"DONE passed={passed_count}/{len(rows)} output={OUT}", flush=True)


if __name__ == "__main__":
    main()
