#!/usr/bin/env python3
"""Convert a Bumi motion NPZ to the compact deploy JSON/NPZ format.

Two input schemas are detected automatically:

1. Any4HDMI Bumi V2: ``qpos[T, 28]`` in MuJoCo model order.
2. Isaac Lab motion: ``joint_pos``, ``joint_vel`` and ``body_quat_w``.

The default output uses the 21-joint Isaac Lab/policy order consumed by
``scripts/sim2sim_mimic.py``.  It preserves both the legacy waist quaternion
and the root pose required to construct active-adaptation ``command`` inputs.
Quaternion data stays in wxyz order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np


DEFAULT_FPS = 50.0

MUJOCO_JOINT_NAMES = (
    "waist_yaw_joint",
    "l_arm_pitch_joint",
    "l_arm_roll_joint",
    "l_arm_yaw_joint",
    "l_elbow_pitch_joint",
    "r_arm_pitch_joint",
    "r_arm_roll_joint",
    "r_arm_yaw_joint",
    "r_elbow_pitch_joint",
    "l_leg_pitch_joint",
    "l_leg_roll_joint",
    "l_leg_yaw_joint",
    "l_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
    "r_leg_pitch_joint",
    "r_leg_roll_joint",
    "r_leg_yaw_joint",
    "r_knee_pitch_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
)

DEPLOY_JOINT_NAMES = (
    "l_leg_pitch_joint",
    "r_leg_pitch_joint",
    "waist_yaw_joint",
    "l_leg_roll_joint",
    "r_leg_roll_joint",
    "l_arm_pitch_joint",
    "r_arm_pitch_joint",
    "l_leg_yaw_joint",
    "r_leg_yaw_joint",
    "l_arm_roll_joint",
    "r_arm_roll_joint",
    "l_knee_pitch_joint",
    "r_knee_pitch_joint",
    "l_arm_yaw_joint",
    "r_arm_yaw_joint",
    "l_ankle_pitch_joint",
    "r_ankle_pitch_joint",
    "l_elbow_pitch_joint",
    "r_elbow_pitch_joint",
    "l_ankle_roll_joint",
    "r_ankle_roll_joint",
)


def _as_names(value: np.ndarray | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    names = tuple(str(item) for item in np.asarray(value).reshape(-1).tolist())
    return names or None


def _normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norms < 1.0e-8):
        raise ValueError("body_quat_w contains a zero-length quaternion")
    result = quaternions / norms
    for frame in range(1, len(result)):
        if np.dot(result[frame - 1], result[frame]) < 0.0:
            result[frame] *= -1.0
    return result


def _quat_mul_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _waist_quat_from_qpos(qpos: np.ndarray) -> np.ndarray:
    root_quat_w = _normalize_quaternions(qpos[:, 3:7])
    half_yaw = 0.5 * qpos[:, 7].astype(np.float64)
    waist_quat_b = np.column_stack(
        (
            np.cos(half_yaw),
            np.zeros_like(half_yaw),
            np.zeros_like(half_yaw),
            np.sin(half_yaw),
        )
    )
    return _normalize_quaternions(_quat_mul_wxyz(root_quat_w, waist_quat_b))


def _joint_velocity(joint_pos: np.ndarray, fps: float) -> np.ndarray:
    if len(joint_pos) < 2:
        return np.zeros_like(joint_pos, dtype=np.float64)
    continuous_pos = np.unwrap(joint_pos.astype(np.float64), axis=0)
    # NumPy's first-order edge rule matches torch.gradient used by the source
    # motion converters and avoids a noisy quadratic extrapolation at frame 0.
    return np.gradient(continuous_pos, 1.0 / fps, axis=0, edge_order=1)


def _resolve_fps(npz: np.lib.npyio.NpzFile, requested_fps: float | None) -> float:
    if requested_fps is not None:
        fps = float(requested_fps)
    elif "fps" in npz.files:
        values = np.asarray(npz["fps"]).reshape(-1)
        if len(values) != 1:
            raise ValueError(f"expected scalar fps, got shape {npz['fps'].shape}")
        fps = float(values[0])
    else:
        fps = DEFAULT_FPS
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be positive and finite, got {fps}")
    return fps


def _reorder_joints(
    values: np.ndarray,
    source_names: tuple[str, ...],
    target_order: Literal["deploy", "source"],
) -> tuple[np.ndarray, tuple[str, ...]]:
    if values.ndim != 2 or values.shape[1] != len(source_names):
        raise ValueError(
            f"joint array shape {values.shape} does not match {len(source_names)} names"
        )
    if target_order == "source":
        return values, source_names
    if set(source_names) != set(DEPLOY_JOINT_NAMES):
        missing = sorted(set(DEPLOY_JOINT_NAMES) - set(source_names))
        extra = sorted(set(source_names) - set(DEPLOY_JOINT_NAMES))
        raise ValueError(
            "cannot convert joints to Bumi deploy order; "
            f"missing={missing}, extra={extra}"
        )
    indices = [source_names.index(name) for name in DEPLOY_JOINT_NAMES]
    return values[:, indices], DEPLOY_JOINT_NAMES


def _select_body_quaternion(
    body_quat_w: np.ndarray,
    body_names: tuple[str, ...] | None,
    body_name: str | None,
    body_index: int,
) -> tuple[np.ndarray, str]:
    if body_quat_w.ndim == 2 and body_quat_w.shape[1] == 4:
        return _normalize_quaternions(body_quat_w), body_name or "preselected"
    if body_quat_w.ndim != 3 or body_quat_w.shape[2] != 4:
        raise ValueError(
            f"expected body_quat_w shape (T, 4) or (T, B, 4), got {body_quat_w.shape}"
        )
    selected_index = body_index
    selected_name = f"body_index_{selected_index}"
    if body_name is not None and body_names is not None:
        if body_name not in body_names:
            raise ValueError(f"body {body_name!r} not found in body_names")
        selected_index = body_names.index(body_name)
        selected_name = body_name
    elif body_names is not None:
        if selected_index < 0 or selected_index >= len(body_names):
            raise IndexError(f"body index {selected_index} is out of range")
        selected_name = body_names[selected_index]
    if selected_index < 0 or selected_index >= body_quat_w.shape[1]:
        raise IndexError(
            f"body index {selected_index} is out of range for {body_quat_w.shape}"
        )
    return _normalize_quaternions(body_quat_w[:, selected_index]), selected_name


def _select_root_position(
    body_pos_w: np.ndarray,
    body_names: tuple[str, ...] | None,
) -> np.ndarray:
    if body_pos_w.ndim == 2 and body_pos_w.shape[1] == 3:
        return body_pos_w.astype(np.float64)
    if body_pos_w.ndim != 3 or body_pos_w.shape[2] != 3:
        raise ValueError(
            f"expected body_pos_w shape (T, 3) or (T, B, 3), got {body_pos_w.shape}"
        )
    root_index = body_names.index("base_link") if body_names and "base_link" in body_names else 0
    if root_index >= body_pos_w.shape[1]:
        raise IndexError(f"root body index {root_index} is out of range for {body_pos_w.shape}")
    return body_pos_w[:, root_index].astype(np.float64)


def _select_root_quaternion(
    body_quat_w: np.ndarray,
    body_names: tuple[str, ...] | None,
) -> np.ndarray:
    if body_quat_w.ndim == 2 and body_quat_w.shape[1] == 4:
        return _normalize_quaternions(body_quat_w)
    if body_quat_w.ndim != 3 or body_quat_w.shape[2] != 4:
        raise ValueError(
            f"expected body_quat_w shape (T, 4) or (T, B, 4), got {body_quat_w.shape}"
        )
    root_index = body_names.index("base_link") if body_names and "base_link" in body_names else 0
    if root_index >= body_quat_w.shape[1]:
        raise IndexError(f"root body index {root_index} is out of range for {body_quat_w.shape}")
    return _normalize_quaternions(body_quat_w[:, root_index])


def _load_motion(
    input_file: Path,
    requested_fps: float | None,
    source_order: Literal["auto", "mujoco", "deploy"],
    target_order: Literal["deploy", "source"],
    body_name: str | None,
    body_index: int,
    qpos_anchor: Literal["waist", "root"],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    tuple[str, ...],
    dict[str, str],
]:
    with np.load(input_file, allow_pickle=False) as motion:
        fps = _resolve_fps(motion, requested_fps)
        files = set(motion.files)

        if "qpos" in files:
            qpos = np.asarray(motion["qpos"])
            if qpos.ndim != 2 or qpos.shape[1] != 28:
                raise ValueError(f"expected Bumi V2 qpos shape (T, 28), got {qpos.shape}")
            source_names = MUJOCO_JOINT_NAMES
            joint_pos = qpos[:, 7:]
            joint_pos, output_names = _reorder_joints(
                joint_pos, source_names, target_order
            )
            joint_vel = _joint_velocity(joint_pos, fps)
            root_pos_w = qpos[:, :3].astype(np.float64)
            root_quat_w = _normalize_quaternions(qpos[:, 3:7])
            if qpos_anchor == "root":
                body_quat_w = root_quat_w.copy()
                anchor_name = "base_link"
            else:
                body_quat_w = _waist_quat_from_qpos(qpos)
                anchor_name = "waist_yaw_link"
            schema = "any4hdmi_qpos"
        elif {"joint_pos", "body_quat_w"}.issubset(files):
            joint_pos = np.asarray(motion["joint_pos"])
            source_names = _as_names(motion["joint_names"] if "joint_names" in files else None)
            if source_order == "mujoco":
                source_names = MUJOCO_JOINT_NAMES
            elif source_order == "deploy":
                source_names = DEPLOY_JOINT_NAMES
            elif source_names is None:
                if joint_pos.ndim == 2 and joint_pos.shape[1] == len(DEPLOY_JOINT_NAMES):
                    source_names = DEPLOY_JOINT_NAMES
                else:
                    raise ValueError(
                        "input has no joint_names; pass --source-order for this schema"
                    )
            joint_pos, output_names = _reorder_joints(
                joint_pos, source_names, target_order
            )
            if "joint_vel" in files:
                joint_vel_raw = np.asarray(motion["joint_vel"])
                joint_vel, velocity_names = _reorder_joints(
                    joint_vel_raw, source_names, target_order
                )
                if velocity_names != output_names:
                    raise AssertionError("position and velocity joint orders differ")
            else:
                joint_vel = _joint_velocity(joint_pos, fps)
            body_names = _as_names(motion["body_names"] if "body_names" in files else None)
            full_body_quat_w = np.asarray(motion["body_quat_w"])
            body_quat_w, anchor_name = _select_body_quaternion(
                full_body_quat_w,
                body_names,
                body_name,
                body_index,
            )
            if "root_pos_w" in files:
                root_pos_w = np.asarray(motion["root_pos_w"], dtype=np.float64)
            elif "body_pos_w" in files:
                root_pos_w = _select_root_position(
                    np.asarray(motion["body_pos_w"]), body_names
                )
            else:
                raise KeyError(
                    "expanded motion has no root_pos_w or body_pos_w; "
                    "root trajectory is required by the new PPO/PPO-ROA command input"
                )
            if "root_quat_w" in files:
                root_quat_w = _normalize_quaternions(motion["root_quat_w"])
            elif full_body_quat_w.ndim == 3:
                root_quat_w = _select_root_quaternion(full_body_quat_w, body_names)
            elif anchor_name == "base_link":
                root_quat_w = body_quat_w.copy()
            else:
                raise KeyError(
                    "preselected body_quat_w is not identified as base_link and "
                    "root_quat_w is missing"
                )
            schema = "expanded_motion"
        else:
            raise KeyError(
                f"unsupported NPZ keys {sorted(files)}; expected qpos, or joint_pos + body_quat_w"
            )

    if root_pos_w.ndim != 2 or root_pos_w.shape[1] != 3:
        raise ValueError(f"expected root_pos_w shape (T, 3), got {root_pos_w.shape}")
    if root_quat_w.ndim != 2 or root_quat_w.shape[1] != 4:
        raise ValueError(f"expected root_quat_w shape (T, 4), got {root_quat_w.shape}")
    arrays = (joint_pos, joint_vel, body_quat_w, root_pos_w, root_quat_w)
    if any(len(array) != len(joint_pos) for array in arrays):
        raise ValueError("joint and body arrays have different frame counts")
    if len(joint_pos) == 0:
        raise ValueError("motion has no frames")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("motion contains NaN or infinity")
    return (
        joint_pos.astype(np.float32),
        joint_vel.astype(np.float32),
        body_quat_w.astype(np.float32),
        root_pos_w.astype(np.float32),
        root_quat_w.astype(np.float32),
        fps,
        output_names,
        {"input_schema": schema, "body_quat_name": anchor_name},
    )


def convert(
    input_file: Path,
    output_json: Path,
    *,
    fps: float | None = None,
    source_order: Literal["auto", "mujoco", "deploy"] = "auto",
    target_order: Literal["deploy", "source"] = "deploy",
    body_name: str | None = None,
    body_index: int = 3,
    qpos_anchor: Literal["waist", "root"] = "waist",
    write_npz: bool = True,
) -> tuple[Path, Path | None]:
    input_file = input_file.expanduser().resolve()
    if not input_file.is_file():
        raise FileNotFoundError(input_file)
    (
        joint_pos,
        joint_vel,
        body_quat_w,
        root_pos_w,
        root_quat_w,
        fps,
        joint_names,
        details,
    ) = _load_motion(
        input_file,
        fps,
        source_order,
        target_order,
        body_name,
        body_index,
        qpos_anchor,
    )

    output_json = output_json.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "format": "bumi_deploy_motion_v1",
            "original_file": str(input_file),
            "frames_count": int(joint_pos.shape[0]),
            "joints_count": int(joint_pos.shape[1]),
            "fps": float(fps),
            "joint_order": target_order,
            "joint_names": list(joint_names),
            "body_quat_name": details["body_quat_name"],
            "root_body_name": "base_link",
            "quaternion_order": "wxyz",
            "input_schema": details["input_schema"],
        },
        "joint_pos": joint_pos.tolist(),
        "joint_vel": joint_vel.tolist(),
        "body_quat_w": body_quat_w.tolist(),
        "root_pos_w": root_pos_w.tolist(),
        "root_quat_w": root_quat_w.tolist(),
    }
    with output_json.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")

    output_npz = output_json.with_suffix(".npz") if write_npz else None
    if output_npz is not None:
        np.savez_compressed(
            output_npz,
            fps=np.asarray([fps], dtype=np.float32),
            joint_names=np.asarray(joint_names),
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_quat_w=body_quat_w,
            root_pos_w=root_pos_w,
            root_quat_w=root_quat_w,
        )
    return output_json, output_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path)
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Default: <input_stem>_deploy.json beside the input.",
    )
    parser.add_argument("--fps", type=float, help="Override/inject the input FPS.")
    parser.add_argument(
        "--source-order",
        choices=("auto", "mujoco", "deploy"),
        default="auto",
        help="Used for expanded files without reliable joint_names.",
    )
    parser.add_argument(
        "--target-order", choices=("deploy", "source"), default="deploy"
    )
    parser.add_argument(
        "--body-name",
        help="Select this body when body_names and full body_quat_w are present.",
    )
    parser.add_argument(
        "--body-index",
        type=int,
        default=3,
        help="Fallback index for full body_quat_w without body_names (default: 3).",
    )
    parser.add_argument(
        "--qpos-anchor",
        choices=("waist", "root"),
        default="waist",
        help="Quaternion exported from qpos input (default: waist).",
    )
    parser.add_argument("--json-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_json = args.output_json or args.input_file.with_name(
        f"{args.input_file.stem}_deploy.json"
    )
    json_path, npz_path = convert(
        args.input_file,
        output_json,
        fps=args.fps,
        source_order=args.source_order,
        target_order=args.target_order,
        body_name=args.body_name,
        body_index=args.body_index,
        qpos_anchor=args.qpos_anchor,
        write_npz=not args.json_only,
    )
    print(f"[OK] JSON: {json_path}")
    if npz_path is not None:
        print(f"[OK] NPZ:  {npz_path}")


if __name__ == "__main__":
    main()
