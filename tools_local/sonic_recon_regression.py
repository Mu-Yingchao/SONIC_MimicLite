#!/usr/bin/env python3
"""SONIC 上层重建质量的 checkpoint 回归：每个 checkpoint 跑一遍固定测试集，看训练是否真在改善目标指标。

为什么不能只看训练 loss：训练端的 aux_g1_recon 是在随机采样的训练窗口上算的均值，
既混了 robot/smpl 两个 encoder，也不区分动作难度；而我们真正关心的是"喂给 MimicLite
的参考动作离原始 PKL 有多远"，而且要分链路、分难度看——高动态动作重建误差是静态的
3 倍，平均数会把这种差异抹平。所以这里用固定的 21 条测试动作、走和部署完全相同的
ONNX 路径（tools_local/sonic_offline_bridge.reconstruct_qpos），逐 checkpoint 出数。

每个 checkpoint 产出：
  - 每条动作 × 每条链路（robot/smpl）的关节重建误差（平均 / p95，单位度）
  - root 位移自洽度：8 步窗口（= MimicLite 观测窗口 0.16s）内，真值 root 的位移与
    重建关节隐含的位移之差（米）。这就是 play 里 ghost 和机器人越拉越远的来源；
    同一指标在 GMR 原始数据上只有 0.0001~0.013 m。

用法：
  评估一个 checkpoint
    python tools_local/sonic_recon_regression.py evaluate --step 18000 \\
        --g1-onnx .../model_step_018000_g1.onnx --smpl-onnx .../model_step_018000_smpl.onnx
  看趋势与告警
    python tools_local/sonic_recon_regression.py report
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools_local"))
from sonic_offline_bridge import (  # noqa: E402
    load_joint_maps,
    load_robot_motion,
    open_session,
    reconstruct_qpos,
)

DATA = REPO / "test_data" / "motions"
MANIFEST = DATA / "any4hdmi-bumi-v2" / "manifest.json"
MJCF = DATA / "any4hdmi-bumi-v2" / "mjcf" / "bumi_v2_0810_rl.xml"
OUT_DIR = REPO / "test_data" / "regression"
PER_MOTION_CSV = OUT_DIR / "per_motion.csv"
SUMMARY_CSV = OUT_DIR / "summary.csv"
TRAIN_CSV = OUT_DIR / "train_scalars.csv"


def set_out_dir(path: Path) -> None:
    """每个训练 run 用独立子目录：新 run 的 step 从 0 重新计数，和旧 run 混写会撞号。"""
    global OUT_DIR, PER_MOTION_CSV, SUMMARY_CSV, TRAIN_CSV
    OUT_DIR = Path(path)
    PER_MOTION_CSV = OUT_DIR / "per_motion.csv"
    SUMMARY_CSV = OUT_DIR / "summary.csv"
    TRAIN_CSV = OUT_DIR / "train_scalars.csv"

WINDOW = 8  # MimicLite command 观测窗口的步数

# 按动作类别分层（不是按误差分），用来看训练对哪类动作见效、哪类停滞
TIERS = {
    "静态日常": ("idle_loop", "looking_around", "looking_in_the_mirror", "clap_", "fixing_something"),
    "行走": ("walk_",),
    "跑跳": ("jog_", "jump_ff", "jump_and_land"),
    "表演舞蹈": ("dancing_", "body_check", "crowd_cheer", "rage_"),
    "高动态": ("high_jump", "reach_jump"),
}
TIER_ORDER = list(TIERS)

# 告警阈值：主指标 = SMPL 链路全体平均误差（项目目标就是 SMPL 这条）
REGRESS_TOL = 0.03   # 比历史最好差 3% 以上 -> 回退
STALL_TOL = 0.01     # 最近几次相邻改善都不足 1% -> 停滞
STALL_WINDOW = 3


def tier_of(name: str) -> str:
    for tier, prefixes in TIERS.items():
        if name.startswith(prefixes):
            return tier
    return "其他"


def _feet_model():
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    feet = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        for n in ("l_ankle_roll_link", "r_ankle_roll_link")
    ]
    return mujoco, model, data, feet


def displacement_mismatch(qpos: np.ndarray, fk) -> float:
    """8 步窗口内真值 root 位移 与 关节隐含位移（支撑脚锁定）之差的均值，米。"""
    mujoco, model, data, feet = fk
    n = len(qpos)
    feet_local = np.zeros((n, 2, 3))
    for t in range(n):
        data.qpos[:28] = qpos[t]
        data.qpos[0:3] = 0.0
        mujoco.mj_forward(model, data)
        for k, bid in enumerate(feet):
            feet_local[t, k] = data.xpos[bid]
    support = np.argmin(feet_local[:, :, 2], axis=1)
    joint_step = np.zeros((n, 2))
    same = support[1:] == support[:-1]
    idx = np.arange(1, n)[same]
    joint_step[idx] = -(feet_local[idx, support[idx], :2] - feet_local[idx - 1, support[idx], :2])
    root_step = np.diff(qpos[:, :2], axis=0)
    joint_step = joint_step[1:]
    mism = [
        np.linalg.norm(root_step[i : i + WINDOW].sum(0) - joint_step[i : i + WINDOW].sum(0))
        for i in range(0, len(root_step) - WINDOW, WINDOW)
    ]
    return float(np.mean(mism)) if mism else float("nan")


def evaluate(step: int, g1_onnx: Path, smpl_onnx: Path) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    perm, mujoco_to_isaac = load_joint_maps(MANIFEST)
    sessions = {"robot": open_session(g1_onnx, "robot"), "smpl": open_session(smpl_onnx, "smpl")}
    fk = _feet_model()

    rows = []
    for robot_pkl in sorted((DATA / "robot_pkl").glob("*.pkl")):
        name = robot_pkl.stem
        smpl_pkl = DATA / "smpl_pkl" / robot_pkl.name
        if not smpl_pkl.exists():
            continue
        robot_motion = load_robot_motion(robot_pkl)
        gt = np.asarray(robot_motion["dof"], dtype=np.float32)
        for chain in ("robot", "smpl"):
            src = robot_motion if chain == "robot" else load_robot_motion(smpl_pkl)
            qpos, joints, _ = reconstruct_qpos(
                sessions[chain], src, chain, perm, mujoco_to_isaac, robot_motion
            )
            err = np.rad2deg(np.abs(gt[: len(joints)] - joints))
            rows.append(
                {
                    "step": step,
                    "motion": name,
                    "tier": tier_of(name),
                    "chain": chain,
                    "frames": len(joints),
                    "mean_deg": round(float(err.mean()), 4),
                    "p95_deg": round(float(np.percentile(err, 95)), 4),
                    "disp_mismatch_m": round(displacement_mismatch(qpos, fk), 5),
                }
            )

    # 同一 step 重跑时覆盖旧行，保证幂等
    _rewrite_csv(PER_MOTION_CSV, rows, key_step=step)

    summary = []
    for chain in ("robot", "smpl"):
        sub = [r for r in rows if r["chain"] == chain]
        rec = {
            "step": step,
            "chain": chain,
            "mean_deg": round(float(np.mean([r["mean_deg"] for r in sub])), 4),
            "p95_deg": round(float(np.mean([r["p95_deg"] for r in sub])), 4),
            "disp_mismatch_m": round(float(np.nanmean([r["disp_mismatch_m"] for r in sub])), 5),
        }
        for tier in TIER_ORDER:
            vals = [r["mean_deg"] for r in sub if r["tier"] == tier]
            rec[tier] = round(float(np.mean(vals)), 4) if vals else ""
        summary.append(rec)
    _rewrite_csv(SUMMARY_CSV, summary, key_step=step)

    for rec in summary:
        print(
            f"step {step:>6}  {rec['chain']:<5}  平均 {rec['mean_deg']:.3f}°  "
            f"p95 {rec['p95_deg']:.3f}°  位移不匹配 {rec['disp_mismatch_m']:.4f}m"
        )


def _rewrite_csv(path: Path, new_rows: list[dict], key_step: int) -> None:
    old = []
    if path.exists():
        with path.open(encoding="utf-8") as f:
            old = [r for r in csv.DictReader(f) if int(r["step"]) != key_step]
    fields = list(new_rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(old + new_rows, key=lambda r: (int(r["step"]), r.get("chain", ""), r.get("motion", ""))):
            w.writerow({k: r.get(k, "") for k in fields})


def _load_summary() -> dict[str, list[dict]]:
    if not SUMMARY_CSV.exists():
        return {}
    with SUMMARY_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_chain: dict[str, list[dict]] = {}
    for r in rows:
        by_chain.setdefault(r["chain"], []).append(r)
    for chain in by_chain:
        by_chain[chain].sort(key=lambda r: int(r["step"]))
    return by_chain


def _load_train() -> dict[int, dict]:
    if not TRAIN_CSV.exists():
        return {}
    with TRAIN_CSV.open(encoding="utf-8") as f:
        return {int(r["step"]): r for r in csv.DictReader(f)}


def _nearest_train(train: dict[int, dict], step: int) -> dict | None:
    if not train:
        return None
    k = min(train, key=lambda s: abs(s - step))
    return train[k] if abs(k - step) <= 250 else None


def _baseline_best(baseline_dir: Path | None) -> dict[str, tuple[int, float]]:
    """读另一个 run 的 summary，返回各链路的 (step, 最好平均误差)。"""
    if baseline_dir is None or not (Path(baseline_dir) / "summary.csv").exists():
        return {}
    best: dict[str, tuple[int, float]] = {}
    with (Path(baseline_dir) / "summary.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v = float(r["mean_deg"])
            if r["chain"] not in best or v < best[r["chain"]][1]:
                best[r["chain"]] = (int(r["step"]), v)
    return best


def report(baseline_dir: Path | None = None) -> int:
    by_chain = _load_summary()
    baseline = _baseline_best(baseline_dir)
    if not by_chain:
        print("还没有任何回归结果。")
        return 0
    train = _load_train()

    print("=" * 96)
    print(f"SONIC 上层重建质量回归（固定 21 条测试动作，走部署同款 ONNX 路径）  run: {OUT_DIR.name}")
    if baseline:
        print(f"基线 = {Path(baseline_dir).name} 的最好成绩：" + "，".join(
            f"{c} {v:.3f}°(step {s})" for c, (s, v) in sorted(baseline.items())))
    print("训练侧 loss 列只用于看同一 run 内的趋势；改过 loss 口径的 run 之间数值不可比")
    print("=" * 96)
    for chain in ("smpl", "robot"):
        rows = by_chain.get(chain, [])
        if not rows:
            continue
        print(f"\n[{chain} 链路]  关节重建误差（度，越低越好）")
        head = f"{'step':>7} {'平均':>7} {'p95':>7} {'位移不匹配m':>11} " + " ".join(f"{t:>6}" for t in TIER_ORDER)
        if chain == "smpl":
            head += f" {'train:smpl_latent':>18}"
        else:
            head += f" {'train:g1_recon':>15}"
        print(head)
        for r in rows:
            line = (
                f"{int(r['step']):>7} {float(r['mean_deg']):>7.3f} {float(r['p95_deg']):>7.3f} "
                f"{float(r['disp_mismatch_m']):>11.4f} "
                + " ".join(f"{float(r[t]):>6.2f}" if r.get(t) else f"{'-':>6}" for t in TIER_ORDER)
            )
            tr = _nearest_train(train, int(r["step"]))
            if tr:
                key = "aux_g1_smpl_latent" if chain == "smpl" else "aux_g1_recon"
                if tr.get(key):
                    line += f" {float(tr[key]):>18.5f}" if chain == "smpl" else f" {float(tr[key]):>15.5f}"
            print(line)

    # 告警：主指标 SMPL 链路平均误差
    rows = by_chain.get("smpl", [])
    status = 0
    print("\n" + "-" * 96)
    if len(rows) >= 2:
        vals = [float(r["mean_deg"]) for r in rows]
        steps = [int(r["step"]) for r in rows]
        best_i = int(np.argmin(vals))
        latest = vals[-1]
        first = vals[0]
        print(
            f"主指标 SMPL 平均误差：首个 {first:.3f}°（step {steps[0]}） -> 最新 {latest:.3f}°（step {steps[-1]}），"
            f"累计改善 {100 * (first - latest) / first:+.1f}%；历史最好 {vals[best_i]:.3f}°（step {steps[best_i]}）"
        )
        if latest > vals[best_i] * (1 + REGRESS_TOL):
            print(f"⚠ 回退：最新比历史最好差 {100 * (latest / vals[best_i] - 1):.1f}%（阈值 {REGRESS_TOL:.0%}）")
            status = 2
        if len(vals) >= STALL_WINDOW + 1:
            recent = vals[-(STALL_WINDOW + 1):]
            gains = [(recent[i] - recent[i + 1]) / recent[i] for i in range(STALL_WINDOW)]
            if all(g < STALL_TOL for g in gains):
                print(
                    f"⚠ 停滞：最近 {STALL_WINDOW} 次相邻改善分别为 "
                    + ", ".join(f"{100 * g:+.2f}%" for g in gains)
                    + f"，均不足 {STALL_TOL:.0%}"
                )
                status = max(status, 1)
        if "smpl" in baseline:
            bs, bv = baseline["smpl"]
            print(f"对比基线（{Path(baseline_dir).name} 最好 {bv:.3f}°）：最新 {latest:.3f}°，"
                  f"{'优于' if latest < bv else '差于'}基线 {100 * abs(bv - latest) / bv:.1f}%")
        if status == 0:
            if len(vals) < STALL_WINDOW + 1:
                # 点太少时既不能说"正常"也不能说"停滞"，如实说明，避免给出虚假的安心
                print(f"· 未见回退；数据点只有 {len(vals)} 个，至少 {STALL_WINDOW + 1} 个才能判断是否停滞")
            else:
                print("✓ 趋势正常：没有回退，也没有停滞")
    else:
        print("结果不足两个点，还不能判断趋势")
    return status


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--step", type=int, required=True)
    ev.add_argument("--g1-onnx", type=Path, required=True)
    ev.add_argument("--smpl-onnx", type=Path, required=True)
    ev.add_argument("--out-dir", type=Path, default=OUT_DIR)
    rp = sub.add_parser("report")
    rp.add_argument("--out-dir", type=Path, default=OUT_DIR)
    rp.add_argument("--baseline-dir", type=Path, default=None, help="另一个 run 的回归目录，报告里作为基线")
    args = ap.parse_args()
    set_out_dir(args.out_dir)
    if args.cmd == "evaluate":
        evaluate(args.step, args.g1_onnx, args.smpl_onnx)
    else:
        sys.exit(report(args.baseline_dir))


if __name__ == "__main__":
    main()
