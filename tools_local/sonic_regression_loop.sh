#!/usr/bin/env bash
# SONIC 上层 checkpoint 回归驱动（本地控制面执行）。
#
# 分工：服务器只做需要 Isaac Lab 的 ONNX 导出；评估在本地做（本地 mjlab venv 有
# onnxruntime，服务器训练环境没有，也不在正在训练的环境里装包——可能牵动 numpy 版本）。
# 幂等：已评估过的 step 自动跳过，本地关机后再跑会从断点继续。
#
# 用法：
#   bash tools_local/sonic_regression_loop.sh                      # 处理所有新 checkpoint 一次
#   bash tools_local/sonic_regression_loop.sh --steps "8000 10000"  # 只处理指定 step（回填历史）
#   bash tools_local/sonic_regression_loop.sh --loop 1800           # 每 30 分钟检查一次，常驻
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
KEY=".local/keys/noetix9.pem"
HOST="root@14.103.42.170"
# 续训写回原 run 目录：每 2000 步一个编号 checkpoint，last.pt 每 500 步覆盖一次
RUN="/data0/bumi2_sonic_runs/TRL_BUMI2_Track/sonic_bumi2_v2-20260920_155335"
EXP="$RUN/exported"
MIN_FREE_MB="${MIN_FREE_MB:-8000}"
LOCAL_ONNX="test_data/policies/sonic/regression"
PY="mimiclite_bumi2/training/venv/mjlab/.venv/bin/python"
LOG_DIR="test_data/regression"
mkdir -p "$LOCAL_ONNX" "$LOG_DIR"
unset VIRTUAL_ENV

remote() {
  ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30 \
    "$HOST" "$@" 2>&1 | grep -v "remote port forwarding"
}

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

server_steps() {
  remote "ls $RUN/model_step_*.pt 2>/dev/null" | grep -oE 'model_step_[0-9]+\.pt' \
    | grep -oE '[0-9]+' | sed 's/^0*//' | sort -n
}

evaluated_steps() {
  [ -f "$LOG_DIR/summary.csv" ] || return 0
  awk -F, 'NR>1 {print $1}' "$LOG_DIR/summary.csv" | sort -un
}

export_on_server() {
  local s6="$1"
  if remote "test -f $EXP/model_step_${s6}_g1.onnx && test -f $EXP/model_step_${s6}_smpl.onnx && echo HAVE" | grep -q HAVE; then
    return 0
  fi
  # 导出与训练并行，跑在 cuda:0（accelerate 单进程默认设备）。不能用
  # CUDA_VISIBLE_DEVICES 把它挪到别的卡：Isaac Sim 的 GPU Foundation 走 Vulkan 自己枚举
  # 设备，和 CUDA 的屏蔽编号对不上，实测直接 "No device could be created" 后段错误。
  # 所以只能与训练 rank0 共享 GPU0，先确认显存余量，防止把训练挤 OOM。
  local free_mb
  free_mb=$(remote "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0" | tr -dc '0-9')
  if [ "${free_mb:-0}" -lt "$MIN_FREE_MB" ]; then
    say "step $s6：GPU0 仅剩 ${free_mb}MB（需要 ≥${MIN_FREE_MB}MB），本轮跳过导出，避免挤爆训练"
    return 1
  fi
  say "step $s6：服务器导出 ONNX（GPU0 剩 ${free_mb}MB，与训练并行）"
  # 单独导出 decoder 那一步对 g1_kin 必然报 KeyError: 'token'（包装器只给了 token_flattened），
  # 但它排在最后、所需的 _g1/_smpl 两个联合模型此时已写好，而且这个单独 decoder 我们用不到。
  # 所以这里不看退出码，只认两个文件是否产出。
  remote "cd /root/SONIC_MimicLite && mkdir -p /data0/.tmp_isaaclab_bumi2_export && \
    setsid env OMNI_KIT_ACCEPT_EULA=YES \
      TMPDIR=/data0/.tmp_isaaclab_bumi2_export OMP_NUM_THREADS=4 \
    nohup /data0/sonic_mimiclite_env/bin/python gear_sonic/eval_agent_trl.py \
      checkpoint=$RUN/model_step_${s6}.pt ++num_envs=1 ++headless=true \
      ++export_onnx_only=true ++export_decoder_name=g1_kin \
      > /data0/bumi2_sonic_runs/export_${s6}.log 2>&1 < /dev/null & disown" >/dev/null
  local waited=0
  while [ $waited -lt 2400 ]; do
    sleep 30; waited=$((waited + 30))
    # 正则写成 [e]val_...：能匹配真正的导出进程，但匹配不到执行这条 pgrep 的 bash -c
    # 自己（它的命令行里是字面量 "[e]val_..."）。直接写 eval_... 会永远匹配到自身。
    if ! remote "pgrep -f '[e]val_agent_trl.py checkpoint=$RUN/model_step_${s6}.pt' >/dev/null && echo RUN" | grep -q RUN; then
      break
    fi
  done
  if remote "test -f $EXP/model_step_${s6}_g1.onnx && test -f $EXP/model_step_${s6}_smpl.onnx && echo HAVE" | grep -q HAVE; then
    return 0
  fi
  say "step $s6：导出失败，日志尾部："
  remote "grep -E 'Traceback|Error' -A4 /data0/bumi2_sonic_runs/export_${s6}.log | tail -15"
  return 1
}

pull_train_scalars() {
  # 训练侧三个 aux loss，按 250 步聚合后写到本地，报告里与实测误差对照
  remote "/data0/sonic_mimiclite_env/bin/python - <<'PYEOF'
import glob
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
tags = {'loss/aux_g1_recon_avg': 'aux_g1_recon',
        'loss/aux_g1_smpl_latent_avg': 'aux_g1_smpl_latent',
        'loss/aux_reencoded_smpl_g1_latent_avg': 'aux_reencoded_smpl_g1_latent'}
acc = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob('$RUN/tensorboard/events.out.*')):
    ea = EventAccumulator(f, size_guidance={'scalars': 0}); ea.Reload()
    have = set(ea.Tags()['scalars'])
    for tag, col in tags.items():
        if tag in have:
            for e in ea.Scalars(tag):
                acc[(e.step // 250) * 250 + 250][col].append(e.value)
print('step,' + ','.join(tags.values()))
for s in sorted(acc):
    print(str(s) + ',' + ','.join(
        ('%.6f' % (sum(acc[s][c]) / len(acc[s][c]))) if acc[s][c] else '' for c in tags.values()))
PYEOF" > "$LOG_DIR/train_scalars.csv.tmp" && grep -q '^step,' "$LOG_DIR/train_scalars.csv.tmp" \
    && mv "$LOG_DIR/train_scalars.csv.tmp" "$LOG_DIR/train_scalars.csv" \
    || { say "训练侧 loss 拉取失败（不影响实测回归）"; rm -f "$LOG_DIR/train_scalars.csv.tmp"; }
}

run_once() {
  local want="${1:-}"
  local todo
  if [ -n "$want" ]; then
    todo="$want"
  else
    # 不用 comm：它要求字典序，而 step 按数值排（8000 与 10000 字典序会颠倒）
    todo=$(server_steps | grep -vxF -f <(evaluated_steps; echo "__none__"))
  fi
  for step in $todo; do
    local s6; s6=$(printf "%06d" "$step")
    if evaluated_steps | grep -qx "$step" && [ -z "$want" ]; then continue; fi
    export_on_server "$s6" || continue
    for kind in g1 smpl; do
      scp -q -i "$KEY" -o StrictHostKeyChecking=no \
        "$HOST:$EXP/model_step_${s6}_${kind}.onnx" "$LOCAL_ONNX/" 2>/dev/null
    done
    say "step $s6：本地评估"
    "$PY" tools_local/sonic_recon_regression.py evaluate --step "$step" \
      --g1-onnx "$LOCAL_ONNX/model_step_${s6}_g1.onnx" \
      --smpl-onnx "$LOCAL_ONNX/model_step_${s6}_smpl.onnx" || say "step $s6：评估失败"
  done
  pull_train_scalars
  "$PY" tools_local/sonic_recon_regression.py report
  local status=$?
  if [ $status -ne 0 ]; then
    date '+%F %T' > "$LOG_DIR/ALERT"
    "$PY" tools_local/sonic_recon_regression.py report >> "$LOG_DIR/ALERT" 2>&1
    say "⚠ 趋势告警已写入 $LOG_DIR/ALERT"
  else
    rm -f "$LOG_DIR/ALERT"
  fi
}

STEPS=""
LOOP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --steps) STEPS="$2"; shift 2 ;;
    --loop) LOOP="$2"; shift 2 ;;
    *) echo "未知参数 $1"; exit 2 ;;
  esac
done

if [ "$LOOP" -gt 0 ]; then
  while true; do run_once "$STEPS"; STEPS=""; sleep "$LOOP"; done
else
  run_once "$STEPS"
fi
