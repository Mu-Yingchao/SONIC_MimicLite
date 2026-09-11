#!/usr/bin/env python3
"""Audit paired BUMI Robot/SMPL PKL fields without importing Isaac Sim."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import joblib
import numpy as np


def _shape(value: object) -> tuple[int, ...]:
    return tuple(np.asarray(value).shape)


def _audit_one(item: tuple[str, str]) -> tuple[str, int, str | None]:
    robot_path_text, smpl_path_text = item
    robot_path = Path(robot_path_text)
    smpl_path = Path(smpl_path_text)
    key = robot_path.stem
    try:
        wrapped = joblib.load(robot_path)
        if not isinstance(wrapped, dict) or list(wrapped) != [key]:
            raise ValueError("robot outer key does not equal filename stem")
        robot = wrapped[key]
        smpl = joblib.load(smpl_path)
        robot_frames = _shape(robot["dof"])[0]
        expected_robot = {
            "root_trans_offset": (robot_frames, 3),
            "pose_aa": (robot_frames, 22, 3),
            "dof": (robot_frames, 21),
            "root_rot": (robot_frames, 4),
        }
        expected_smpl = {
            "pose_aa": (robot_frames, 72),
            "transl": (robot_frames, 3),
            "smpl_joints": (robot_frames, 24, 3),
        }
        if robot_frames < 2:
            raise ValueError(f"too few frames: {robot_frames}")
        if float(robot["fps"]) != 50.0 or float(smpl["fps"]) != 50.0:
            raise ValueError(f"fps mismatch: robot={robot['fps']} smpl={smpl['fps']}")
        for name, expected in expected_robot.items():
            if _shape(robot[name]) != expected:
                raise ValueError(f"robot {name} shape {_shape(robot[name])} != {expected}")
            if not np.isfinite(np.asarray(robot[name])).all():
                raise ValueError(f"robot {name} contains NaN/Inf")
        for name, expected in expected_smpl.items():
            if _shape(smpl[name]) != expected:
                raise ValueError(f"smpl {name} shape {_shape(smpl[name])} != {expected}")
            if not np.isfinite(np.asarray(smpl[name])).all():
                raise ValueError(f"smpl {name} contains NaN/Inf")
        quat_norm = np.linalg.norm(np.asarray(robot["root_rot"]), axis=1)
        if not np.allclose(quat_norm, 1.0, atol=1e-3):
            raise ValueError(f"robot root quaternion norm range {quat_norm.min()}..{quat_norm.max()}")
        return key, robot_frames, None
    except Exception as exc:  # report every bad pair instead of stopping at the first one
        return key, 0, f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-dir", type=Path, required=True)
    parser.add_argument("--smpl-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--expected-pairs", type=int)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    robots = sorted(args.robot_dir.rglob("*.pkl"))
    robot_names = [path.stem for path in robots]
    if len(robot_names) != len(set(robot_names)):
        raise ValueError("duplicate Robot filename stems")
    smpls = {path.stem: path for path in args.smpl_dir.rglob("*.pkl")}
    if len(smpls) != len(list(args.smpl_dir.rglob("*.pkl"))):
        raise ValueError("duplicate SMPL filename stems")
    if set(robot_names) != set(smpls):
        raise ValueError(
            f"pair key mismatch: robot_only={len(set(robot_names) - set(smpls))} "
            f"smpl_only={len(set(smpls) - set(robot_names))}"
        )
    if args.expected_pairs is not None and len(robots) != args.expected_pairs:
        raise ValueError(f"pair count {len(robots)} != {args.expected_pairs}")

    items = [(str(path), str(smpls[path.stem])) for path in robots]
    failures: list[dict[str, str]] = []
    frame_counts: list[int] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (key, frames, error) in enumerate(
            pool.map(_audit_one, items, chunksize=64), start=1
        ):
            if error is None:
                frame_counts.append(frames)
            else:
                failures.append({"key": key, "error": error})
            if index % 5000 == 0:
                print(f"AUDIT_PROGRESS {index}/{len(items)} failures={len(failures)}", flush=True)

    report = {
        "pairs": len(items),
        "passed": len(frame_counts),
        "failed": len(failures),
        "min_frames": min(frame_counts, default=None),
        "max_frames": max(frame_counts, default=None),
        "total_frames": sum(frame_counts),
        "failures": failures,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n")
    if failures:
        print(rendered)
        raise SystemExit(1)
    print(
        "BUMI_DATASET_AUDIT=PASS "
        f"pairs={len(items)} total_frames={sum(frame_counts)} "
        f"frame_range={min(frame_counts)}..{max(frame_counts)}"
    )


if __name__ == "__main__":
    main()
