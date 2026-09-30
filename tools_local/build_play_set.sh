#!/usr/bin/env bash
# 用指定 SONIC checkpoint 批量重建 play 用的参考动作（21 条配对测试动作 × 4 组）。
#
#   bash tools_local/build_play_set.sh <g1.onnx> <smpl.onnx>
#
# 输出到 test_data/motions/any4hdmi-bumi-v2/motions/ 下（整目录替换）：
#   sonic_smpl_mink/  SMPL → SONIC → mink（项目目标链路）
#   sonic_smpl/       SMPL → SONIC，root 直接拼源动作（对照）
#   sonic_robot/      robot PKL → SONIC → mink
#   sonic_smpl_online/ SMPL → 在线桥接（逐帧、因果、root 只来自 SMPL）——遥操等价链路
# 每个目录写一份 SOURCE.txt 记录来自哪个 checkpoint。original_qpos/ 与 SONIC 无关，不动。
set -euo pipefail
cd "$(dirname "$0")/.."

G1_ONNX="${1:?用法: build_play_set.sh <g1.onnx> <smpl.onnx>}"
SMPL_ONNX="${2:?用法: build_play_set.sh <g1.onnx> <smpl.onnx>}"
PY="mimiclite_bumi2/training/venv/mjlab/.venv/bin/python"
DATA="test_data/motions"
M="$DATA/any4hdmi-bumi-v2"
unset VIRTUAL_ENV

for sub in sonic_smpl_mink sonic_smpl sonic_robot; do
  tmp="$M/motions/.${sub}.tmp"
  rm -rf "$tmp"; mkdir -p "$tmp"
  for pkl in "$DATA"/robot_pkl/*.pkl; do
    name=$(basename "$pkl" .pkl)
    case "$sub" in
      sonic_smpl_mink) args=(--motion "$DATA/smpl_pkl/$name.pkl" --root-motion "$pkl" --encoder smpl --onnx "$SMPL_ONNX") ;;
      sonic_smpl)      args=(--motion "$DATA/smpl_pkl/$name.pkl" --root-motion "$pkl" --encoder smpl --onnx "$SMPL_ONNX" --root-solve none) ;;
      sonic_robot)     args=(--motion "$pkl" --encoder robot --onnx "$G1_ONNX") ;;
    esac
    "$PY" tools_local/sonic_offline_bridge.py "${args[@]}" \
      --manifest "$M/manifest.json" --out "$tmp/${name}_from_g1_bumi_v2.npz" >/dev/null
  done
  printf 'g1:    %s\nsmpl:  %s\nbuilt: %s\n' "$G1_ONNX" "$SMPL_ONNX" "$(date '+%F %T')" > "$tmp/SOURCE.txt"
  rm -rf "${M:?}/motions/$sub"; mv "$tmp" "$M/motions/$sub"
  echo "$sub: $(ls "$M/motions/$sub"/*.npz | wc -l) 条"
done

sub=sonic_smpl_online; tmp="$M/motions/.${sub}.tmp"
rm -rf "$tmp"; mkdir -p "$tmp"
for pkl in "$DATA"/smpl_pkl/*.pkl; do
  name=$(basename "$pkl" .pkl)
  "$PY" tools_local/sonic_online_bridge.py --smpl "$pkl" --onnx "$SMPL_ONNX" \
    --manifest "$M/manifest.json" --out "$tmp/${name}_from_g1_bumi_v2.npz" >/dev/null
done
printf 'smpl:  %s\nbuilt: %s\n在线桥接，不用配对 robot PKL\n' "$SMPL_ONNX" "$(date '+%F %T')" > "$tmp/SOURCE.txt"
rm -rf "${M:?}/motions/$sub"; mv "$tmp" "$M/motions/$sub"
echo "$sub: $(ls "$M/motions/$sub"/*.npz | wc -l) 条"
