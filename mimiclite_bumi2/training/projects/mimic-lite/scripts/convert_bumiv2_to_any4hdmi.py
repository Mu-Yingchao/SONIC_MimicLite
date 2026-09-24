#!/usr/bin/env python3
"""Convert BUMI rich-FK NPZ or deploy CSV files to any4hdmi qpos.

The source is read-only. Output files contain only ``qpos`` and preserve the
source score directory hierarchy below ``motions``. Deploy CSV files are
expected to contain root position, root quaternion in xyzw order, and the 21
BUMI joints. The input rate is read from ``_*fps`` in the filename and motions
are resampled to 50 Hz. Writes are atomic, and valid existing output files are
skipped so an interrupted run can resume.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mujoco
import numpy as np
from any4hdmi.core.format import write_manifest
from any4hdmi.utils.mjcf import qpos_names_from_model
from tqdm import tqdm

DEFAULT_SRC_ROOT = Path("/data2/zcx/datasets/bumiv2")
DEFAULT_DST_ROOT = Path("/data2/zcx/datasets/any4hdmi-bumi-v2")
DEFAULT_MJCF = Path(
    "/home/user/zcx/active-adaptation/active_adaptation/assets/BUMI/"
    "BM2-V2.0/mjcf/bumi_v2_0810_rl.xml"
)
DEFAULT_MESHES = Path(
    "/home/user/zcx/active-adaptation/active_adaptation/assets/BUMI/"
    "BM2-V2.0/meshes"
)
QPOS_DIM = 28
SOURCE_FPS = 50.0
CSV_ROOT_COLUMNS = (
    "root_pos_x",
    "root_pos_y",
    "root_pos_z",
    "root_rot_x",
    "root_rot_y",
    "root_rot_z",
    "root_rot_w",
)
CSV_JOINT_NAMES = {
    "leg.pitch.l": "l_leg_pitch_joint",
    "leg.roll.l": "l_leg_roll_joint",
    "leg.yaw.l": "l_leg_yaw_joint",
    "knee.pitch.l": "l_knee_pitch_joint",
    "ankle.pitch.l": "l_ankle_pitch_joint",
    "ankle.roll.l": "l_ankle_roll_joint",
    "leg.pitch.r": "r_leg_pitch_joint",
    "leg.roll.r": "r_leg_roll_joint",
    "leg.yaw.r": "r_leg_yaw_joint",
    "knee.pitch.r": "r_knee_pitch_joint",
    "ankle.pitch.r": "r_ankle_pitch_joint",
    "ankle.roll.r": "r_ankle_roll_joint",
    "waist.yaw": "waist_yaw_joint",
    "arm.pitch.l": "l_arm_pitch_joint",
    "arm.roll.l": "l_arm_roll_joint",
    "arm.yaw.l": "l_arm_yaw_joint",
    "elbow.pitch.l": "l_elbow_pitch_joint",
    "arm.pitch.r": "r_arm_pitch_joint",
    "arm.roll.r": "r_arm_roll_joint",
    "arm.yaw.r": "r_arm_yaw_joint",
    "elbow.pitch.r": "r_elbow_pitch_joint",
}
_FPS_SUFFIX = re.compile(r"_(?P<fps>\d+(?:\.\d+)?)fps$")

# Column order in the rich-FK source archive's joint_pos array.
SOURCE_JOINT_NAMES = (
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


def _hinge_joint_names(model: mujoco.MjModel) -> list[str]:
    names: list[str] = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name is None:
                raise RuntimeError(f"MJCF hinge joint {joint_id} has no name")
            names.append(str(name))
    return names


def _source_to_mjcf_perm(mjcf_joint_names: list[str]) -> tuple[int, ...]:
    if len(mjcf_joint_names) != len(SOURCE_JOINT_NAMES):
        raise RuntimeError(
            f"Expected {len(SOURCE_JOINT_NAMES)} hinge joints, "
            f"got {len(mjcf_joint_names)}"
        )
    source_names = set(SOURCE_JOINT_NAMES)
    mjcf_names = set(mjcf_joint_names)
    if source_names != mjcf_names:
        raise RuntimeError(
            "Source/MJCF joint name mismatch: "
            f"only_source={sorted(source_names - mjcf_names)}, "
            f"only_mjcf={sorted(mjcf_names - source_names)}"
        )
    source_index = {name: index for index, name in enumerate(SOURCE_JOINT_NAMES)}
    return tuple(source_index[name] for name in mjcf_joint_names)


def _valid_existing_output(path: Path) -> tuple[bool, int]:
    if not path.is_file():
        return False, 0
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"qpos"}:
                return False, 0
            qpos = archive["qpos"]
            if qpos.ndim != 2 or qpos.shape[1] != QPOS_DIM or qpos.shape[0] < 1:
                return False, 0
            if qpos.dtype != np.float32 or not np.isfinite(qpos).all():
                return False, 0
            return True, int(qpos.shape[0])
    except (OSError, ValueError, KeyError):
        return False, 0


def _normalize_quaternions(quat: np.ndarray, src_path: Path) -> np.ndarray:
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError(f"Zero-length root quaternion in {src_path}")
    quat = quat / norms
    for frame in range(1, len(quat)):
        if np.dot(quat[frame - 1], quat[frame]) < 0.0:
            quat[frame] *= -1.0
    return quat


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, blend: np.ndarray) -> np.ndarray:
    dot = np.sum(q0 * q1, axis=1)
    flip = dot < 0.0
    q1 = q1.copy()
    q1[flip] *= -1.0
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    result = np.empty_like(q0)
    linear = dot > 0.9995
    if np.any(linear):
        b = blend[linear, None]
        result[linear] = q0[linear] * (1.0 - b) + q1[linear] * b
    spherical = ~linear
    if np.any(spherical):
        theta = np.arccos(dot[spherical])
        sin_theta = np.sin(theta)
        b = blend[spherical]
        w0 = np.sin((1.0 - b) * theta) / sin_theta
        w1 = np.sin(b * theta) / sin_theta
        result[spherical] = q0[spherical] * w0[:, None] + q1[spherical] * w1[:, None]
    return result / np.linalg.norm(result, axis=1, keepdims=True)


def _resample_csv_motion(
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    joint_pos: np.ndarray,
    input_fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = len(root_pos)
    if frames < 2:
        return root_pos, root_quat, joint_pos
    if abs(input_fps - SOURCE_FPS) <= 1e-6:
        return root_pos, root_quat, joint_pos
    duration = (frames - 1) / input_fps
    output_frames = int(round(duration * SOURCE_FPS)) + 1
    times = np.linspace(0.0, duration, output_frames, dtype=np.float64)
    source_frame = np.minimum(times * input_fps, frames - 1)
    index_0 = np.floor(source_frame).astype(np.int64)
    index_1 = np.minimum(index_0 + 1, frames - 1)
    blend = source_frame - index_0
    root_pos_out = root_pos[index_0] * (1.0 - blend[:, None]) + root_pos[index_1] * blend[:, None]
    joint_pos_out = (
        joint_pos[index_0] * (1.0 - blend[:, None]) + joint_pos[index_1] * blend[:, None]
    )
    root_quat_out = _quat_slerp(root_quat[index_0], root_quat[index_1], blend)
    return root_pos_out, root_quat_out, joint_pos_out


def _qpos_from_csv(src_path: Path, mjcf_joint_names: tuple[str, ...]) -> np.ndarray:
    with src_path.open(encoding="utf-8-sig") as handle:
        header = tuple(part.strip() for part in handle.readline().strip().split(","))
    if header[:7] != CSV_ROOT_COLUMNS:
        raise ValueError(f"Unexpected root columns in {src_path}: {header[:7]}")
    try:
        source_joint_names = tuple(CSV_JOINT_NAMES[name] for name in header[7:])
    except KeyError as error:
        raise ValueError(f"Unknown BUMI CSV joint column {error.args[0]!r} in {src_path}") from error
    if len(source_joint_names) != len(SOURCE_JOINT_NAMES) or set(source_joint_names) != set(
        SOURCE_JOINT_NAMES
    ):
        raise ValueError(f"CSV joint columns do not match the 21 BUMI joints in {src_path}")
    match = _FPS_SUFFIX.search(src_path.stem)
    if match is None:
        raise ValueError(f"CSV filename must end in _<fps>fps: {src_path.name}")
    input_fps = float(match.group("fps"))
    if input_fps <= 0.0:
        raise ValueError(f"Invalid input FPS {input_fps} in {src_path.name}")
    motion = np.loadtxt(src_path, delimiter=",", skiprows=1, dtype=np.float64)
    if motion.ndim != 2 or motion.shape[1] != QPOS_DIM or motion.shape[0] < 1:
        raise ValueError(f"Expected CSV shape (T, {QPOS_DIM}), got {motion.shape} in {src_path}")
    if not np.isfinite(motion).all():
        raise ValueError(f"Non-finite CSV values in {src_path}")
    root_pos = motion[:, :3]
    root_quat = _normalize_quaternions(motion[:, [6, 3, 4, 5]], src_path)  # xyzw -> wxyz
    source_index = {name: index for index, name in enumerate(source_joint_names)}
    joint_perm = tuple(source_index[name] for name in mjcf_joint_names)
    joint_pos = motion[:, 7:][:, joint_perm]
    root_pos, root_quat, joint_pos = _resample_csv_motion(
        root_pos, root_quat, joint_pos, input_fps
    )
    qpos = np.concatenate((root_pos, root_quat, joint_pos), axis=1)
    return np.asarray(qpos, dtype=np.float32)


def _qpos_from_rich_npz(src_path: Path, perm: tuple[int, ...]) -> np.ndarray:
    with np.load(src_path, allow_pickle=False) as archive:
        required = {"fps", "joint_pos", "body_pos_w", "body_quat_w"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Missing {sorted(missing)} in {src_path}")
        joint_pos = np.asarray(archive["joint_pos"], dtype=np.float32)
        body_pos_w = np.asarray(archive["body_pos_w"], dtype=np.float32)
        body_quat_w = np.asarray(archive["body_quat_w"], dtype=np.float32)
        fps = float(np.asarray(archive["fps"]).reshape(-1)[0])

    if joint_pos.ndim != 2 or joint_pos.shape[1] != len(SOURCE_JOINT_NAMES):
        raise ValueError(
            f"Expected joint_pos (T, {len(SOURCE_JOINT_NAMES)}), "
            f"got {joint_pos.shape} in {src_path}"
        )
    if body_pos_w.ndim != 3 or body_pos_w.shape[1:] != (22, 3):
        raise ValueError(f"Expected body_pos_w (T, 22, 3), got {body_pos_w.shape} in {src_path}")
    if body_quat_w.ndim != 3 or body_quat_w.shape[1:] != (22, 4):
        raise ValueError(
            f"Expected body_quat_w (T, 22, 4), got {body_quat_w.shape} in {src_path}"
        )
    frames = int(joint_pos.shape[0])
    if frames < 1 or body_pos_w.shape[0] != frames or body_quat_w.shape[0] != frames:
        raise ValueError(f"Inconsistent or empty frame dimensions in {src_path}")
    if abs(fps - SOURCE_FPS) > 1e-4:
        raise ValueError(f"Expected fps={SOURCE_FPS}, got {fps} in {src_path}")

    qpos = np.empty((frames, QPOS_DIM), dtype=np.float32)
    qpos[:, 0:3] = body_pos_w[:, 0]
    qpos[:, 3:7] = body_quat_w[:, 0]  # Source and MuJoCo both use wxyz.
    qpos[:, 7:] = joint_pos[:, perm]
    return qpos


def _convert_one(
    job: tuple[str, str, tuple[int, ...], tuple[str, ...]],
) -> tuple[str, int]:
    src_string, dst_string, perm, mjcf_joint_names = job
    src_path = Path(src_string)
    dst_path = Path(dst_string)

    valid, frames = _valid_existing_output(dst_path)
    if valid:
        return "skipped", frames

    if src_path.suffix.lower() == ".csv":
        qpos = _qpos_from_csv(src_path, mjcf_joint_names)
    elif src_path.suffix.lower() == ".npz":
        qpos = _qpos_from_rich_npz(src_path, perm)
    else:
        raise ValueError(f"Unsupported BUMI source format: {src_path}")
    frames = int(qpos.shape[0])
    if not np.isfinite(qpos).all():
        raise ValueError(f"Non-finite qpos in {src_path}")
    root_quat_norm = np.linalg.norm(qpos[:, 3:7], axis=1)
    max_quat_error = float(np.max(np.abs(root_quat_norm - 1.0)))
    if max_quat_error > 1e-3:
        raise ValueError(
            f"Root quaternion norm error {max_quat_error:.6g} exceeds 1e-3 in {src_path}"
        )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_name(f".{dst_path.name}.tmp.{os.getpid()}")
    try:
        with tmp_path.open("wb") as handle:
            np.savez(handle, qpos=qpos)
        os.replace(tmp_path, dst_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return "converted", frames


def _prepare_shared_assets(dst_root: Path, mjcf_src: Path, mesh_src: Path) -> Path:
    mjcf_dir = dst_root / "mjcf"
    mjcf_dir.mkdir(parents=True, exist_ok=True)
    mjcf_dst = mjcf_dir / mjcf_src.name
    if mjcf_dst.exists():
        if mjcf_dst.read_bytes() != mjcf_src.read_bytes():
            raise RuntimeError(f"Refusing to replace different MJCF: {mjcf_dst}")
    else:
        shutil.copy2(mjcf_src, mjcf_dst)

    meshes_dst = dst_root / "meshes"
    if meshes_dst.is_symlink():
        if meshes_dst.resolve() != mesh_src.resolve():
            raise RuntimeError(f"Existing mesh symlink points elsewhere: {meshes_dst}")
    elif meshes_dst.exists():
        raise RuntimeError(f"Refusing to replace existing non-symlink: {meshes_dst}")
    else:
        meshes_dst.symlink_to(mesh_src)
    return mjcf_dst


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, default=DEFAULT_SRC_ROOT)
    parser.add_argument("--dst-root", type=Path, default=DEFAULT_DST_ROOT)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--meshes", type=Path, default=DEFAULT_MESHES)
    parser.add_argument("--scores", nargs="+", default=["score1", "score2"])
    parser.add_argument("--workers", type=int, default=min(16, max(1, (os.cpu_count() or 8) // 2)))
    parser.add_argument("--limit", type=int, default=0, help="Convert at most N files total (0=all)")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--no-write-manifest",
        action="store_true",
        help="Keep an existing dataset manifest unchanged when appending selected scores.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.workers < 1 or args.batch_size < 1 or args.limit < 0:
        raise ValueError("workers and batch-size must be positive; limit must be non-negative")

    src_root = args.src_root.expanduser().resolve()
    dst_root = args.dst_root.expanduser().resolve()
    mjcf_src = args.mjcf.expanduser().resolve()
    mesh_src = args.meshes.expanduser().resolve()
    if not src_root.is_dir():
        raise FileNotFoundError(f"BUMI source dataset not found: {src_root}")
    if not mjcf_src.is_file():
        raise FileNotFoundError(f"BUMI MJCF not found: {mjcf_src}")
    if not mesh_src.is_dir():
        raise FileNotFoundError(f"BUMI meshes not found: {mesh_src}")
    if src_root == dst_root or src_root in dst_root.parents or dst_root in src_root.parents:
        raise RuntimeError("Source and destination must be separate directory trees")

    mjcf_dst = _prepare_shared_assets(dst_root, mjcf_src, mesh_src)
    model = mujoco.MjModel.from_xml_path(str(mjcf_dst))
    if model.nq != QPOS_DIM:
        raise RuntimeError(f"Expected model.nq={QPOS_DIM}, got {model.nq}")
    hinge_names = _hinge_joint_names(model)
    perm = _source_to_mjcf_perm(hinge_names)
    qpos_names = list(qpos_names_from_model(model))
    if len(qpos_names) != QPOS_DIM:
        raise RuntimeError(f"Expected {QPOS_DIM} qpos names, got {len(qpos_names)}")
    if qpos_names[:7] != [
        "root_tx", "root_ty", "root_tz", "root_qw", "root_qx", "root_qy", "root_qz"
    ]:
        raise RuntimeError(f"Unexpected floating-base qpos names: {qpos_names[:7]}")
    if qpos_names[7:] != hinge_names:
        raise RuntimeError("qpos hinge names do not match MJCF hinge order")

    jobs: list[tuple[str, str, tuple[int, ...], tuple[str, ...]]] = []
    score_counts: dict[str, int] = {}
    for score in args.scores:
        src_score = src_root / score
        if not src_score.is_dir():
            raise FileNotFoundError(f"Score directory not found: {src_score}")
        src_files = sorted((*src_score.rglob("*.npz"), *src_score.rglob("*.csv")))
        if not src_files:
            raise RuntimeError(f"No NPZ or CSV files under {src_score}")
        score_counts[score] = len(src_files)
        output_paths: set[Path] = set()
        for src_path in src_files:
            rel_path = src_path.relative_to(src_score)
            dst_path = dst_root / "motions" / score / rel_path.with_suffix(".npz")
            if dst_path in output_paths:
                raise RuntimeError(f"Multiple sources map to one output: {dst_path}")
            output_paths.add(dst_path)
            jobs.append((str(src_path), str(dst_path), perm, tuple(hinge_names)))
    if args.limit:
        jobs = jobs[: args.limit]

    print(
        f"BUMI conversion: source={src_root} destination={dst_root} "
        f"files={len(jobs)} workers={args.workers}",
        flush=True,
    )
    print(f"discovered_by_score={score_counts}", flush=True)
    print(f"source_joint_order={list(SOURCE_JOINT_NAMES)}", flush=True)
    print(f"target_joint_order={hinge_names}", flush=True)

    total_frames = 0
    converted = 0
    skipped = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        with tqdm(total=len(jobs), desc="convert BUMI", unit="file") as progress:
            for start in range(0, len(jobs), args.batch_size):
                futures = [pool.submit(_convert_one, job) for job in jobs[start : start + args.batch_size]]
                for future in as_completed(futures):
                    status, frames = future.result()
                    total_frames += frames
                    converted += status == "converted"
                    skipped += status == "skipped"
                    progress.update()

    if not args.no_write_manifest:
        mjcf_sha256 = hashlib.sha256(mjcf_src.read_bytes()).hexdigest()
        write_manifest(
            dst_root,
            dataset_name="bumi_v2",
            mjcf=mjcf_dst,
            timestep=1.0 / SOURCE_FPS,
            qpos_names=qpos_names,
            num_motions=len(jobs),
            source={
                "src_root": str(src_root),
                "scores": list(args.scores),
                "discovered_by_score": score_counts,
                "fps": SOURCE_FPS,
                "root_body": "base_link",
                "root_quaternion_order": "wxyz",
                "qpos_from": "rich FK NPZ or deploy CSV reordered into MuJoCo qpos",
                "source_joint_names": list(SOURCE_JOINT_NAMES),
                "target_hinge_joint_names": hinge_names,
                "source_mjcf": str(mjcf_src),
                "source_mjcf_sha256": mjcf_sha256,
            },
            total_hours=total_frames / SOURCE_FPS / 3600.0,
        )
    print(
        f"DONE motions={len(jobs)} converted={converted} skipped={skipped} "
        f"frames={total_frames} hours={total_frames / SOURCE_FPS / 3600.0:.6f} "
        f"manifest={'preserved' if args.no_write_manifest else dst_root / 'manifest.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
