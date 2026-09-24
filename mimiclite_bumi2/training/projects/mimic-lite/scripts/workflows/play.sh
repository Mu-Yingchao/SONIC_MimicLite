#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SH=""
if [[ -f "$SCRIPT_DIR/_interactive_common.sh" ]]; then
    COMMON_SH="$SCRIPT_DIR/_interactive_common.sh"
elif [[ -f "$SCRIPT_DIR/../_interactive_common.sh" ]]; then
    COMMON_SH="$SCRIPT_DIR/../_interactive_common.sh"
else
    echo "找不到 _interactive_common.sh" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$COMMON_SH"

aa_ensure_uv
cd "$ROOT"

aa_header "MimicLite Play"

aa_list_play_checkpoints() {
    local recipe="$1"
    local task="$2"
    local checkpoint checkpoint_dir checkpoint_lower overrides
    while IFS= read -r checkpoint; do
        if [[ "$recipe" == "ppo_roa" ]]; then
            [[ "$checkpoint" == */stages/03-finetune/* ]] || continue
        else
            [[ "$checkpoint" == */stages/* ]] && continue
            checkpoint_lower="${checkpoint,,}"
            [[ "$checkpoint_lower" == *ppo_roa* ]] && continue
            [[ "$checkpoint_lower" == *ppo-roa* ]] && continue
        fi

        checkpoint_dir="$(dirname "$checkpoint")"
        overrides="$checkpoint_dir/.hydra/overrides.yaml"
        if [[ -f "$overrides" ]]; then
            grep -Fq -- "task=$task" "$overrides" || continue
        elif [[ "$task" == "tracking-bumi-v2" ]]; then
            checkpoint_lower="${checkpoint,,}"
            [[ "$checkpoint_lower" == *bumi* ]] || continue
        else
            continue
        fi
        echo "$checkpoint"
    done < <(aa_list_checkpoints)
}

mapfile -t TASKS < <(aa_list_tasks)
mapfile -t MOTIONS < <(aa_list_motions)

aa_pick BACKEND "仿真后端" "mjlab" mjlab isaaclab
aa_pick TASK "任务 (task)" "tracking-bumi-v2" "${TASKS[@]}"
aa_pick MOTION "动作集 (task/motion)" "bumi/v2_mixed" "${MOTIONS[@]}"

MOTION_WEIGHT_ARGS=()
MOTION_WEIGHT_KEYS=()
MOTION_WEIGHT_LABELS=()
MOTION_WEIGHT_DEFAULTS=()
case "$MOTION" in
"bumi/v2_mixed")
    MOTION_WEIGHT_KEYS=(bumi_v2_score1 bumi_v2_highdynamic bumi_v2_dance3)
    MOTION_WEIGHT_LABELS=(score1 score_highdynamic score_highdynamic_dance3)
    MOTION_WEIGHT_DEFAULTS=(0.8 0.1 0.1)
    ;;
esac
if ((${#MOTION_WEIGHT_KEYS[@]} > 0)); then
    MOTION_WEIGHT_VALUES=()
    for ((i = 0; i < ${#MOTION_WEIGHT_KEYS[@]}; i++)); do
        aa_ask MOTION_WEIGHT "${MOTION_WEIGHT_LABELS[$i]} 采样权重" "${MOTION_WEIGHT_DEFAULTS[$i]}"
        weight="$MOTION_WEIGHT"
        if ! [[ "$weight" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] \
            || ! awk -v value="$weight" 'BEGIN { exit !(value > 0) }'; then
            echo "动作集权重必须是大于 0 的数字: $weight" >&2
            exit 1
        fi
        MOTION_WEIGHT_VALUES+=("$weight")
    done
    TOTAL_WEIGHT="$(printf '%s\n' "${MOTION_WEIGHT_VALUES[@]}" | awk '{ total += $1 } END { print total }')"
    NORMALIZED_WEIGHTS=""
    for ((i = 0; i < ${#MOTION_WEIGHT_KEYS[@]}; i++)); do
        PERCENT="$(awk -v value="${MOTION_WEIGHT_VALUES[$i]}" -v total="$TOTAL_WEIGHT" \
            'BEGIN { printf "%.2f", 100 * value / total }')"
        [[ -z "$NORMALIZED_WEIGHTS" ]] || NORMALIZED_WEIGHTS+=", "
        NORMALIZED_WEIGHTS+="${MOTION_WEIGHT_LABELS[$i]}=${PERCENT}%"
        MOTION_WEIGHT_ARGS+=(
            "task.command.motion_cfgs.${MOTION_WEIGHT_KEYS[$i]}.weight=${MOTION_WEIGHT_VALUES[$i]}"
        )
    done
    echo "归一化采样比例: $NORMALIZED_WEIGHTS"
fi

aa_pick RECIPE "策略配方" "ppo" ppo ppo_roa

if [[ "$RECIPE" == "ppo_roa" ]]; then
    mapfile -t MODULES < <(aa_list_modules ppo_roa)
    DEFAULT_MODULE="huge"
    ALGO_MODULE_KEY="algo/ppo_roa/module"
    EXP="+exp=ppo_roa/finetune"
else
    mapfile -t MODULES < <(aa_list_modules ppo)
    DEFAULT_MODULE="huge"
    ALGO_MODULE_KEY="algo/ppo/module"
    EXP="+exp=ppo/train"
fi
aa_pick MODULE "网络规模 (module)" "$DEFAULT_MODULE" "${MODULES[@]}"

mapfile -t CKPTS < <(aa_list_play_checkpoints "$RECIPE" "$TASK")
CKPT_CHOICES=()
if ((${#CKPTS[@]} > 0)); then
    for p in "${CKPTS[@]}"; do
        CKPT_CHOICES+=("$(aa_relpath "$p")")
        if ((${#CKPT_CHOICES[@]} >= 20)); then
            break
        fi
    done
fi
CKPT_CHOICES+=("手动输入路径")

DEFAULT_CKPT="手动输入路径"
# The list is already restricted to the selected task. Prefer the newest
# concrete checkpoint over checkpoint_latest.pt so the selected iteration is clear.
for p in "${CKPT_CHOICES[@]}"; do
    p_base="${p##*/}"
    if [[ "$p_base" =~ ^checkpoint_[0-9]+\.pt$ ]]; then
        DEFAULT_CKPT="$p"
        break
    fi
done

aa_pick CKPT_SEL \
    "训练日志 / Checkpoint（PPO-ROA 只显示 03-finetune）" \
    "$DEFAULT_CKPT" \
    "${CKPT_CHOICES[@]}"
if [[ "$CKPT_SEL" == "手动输入路径" ]]; then
    aa_ask CKPT_PATH "输入 checkpoint 路径" ""
    if [[ -z "$CKPT_PATH" ]]; then
        echo "必须指定 checkpoint" >&2
        exit 1
    fi
else
    CKPT_PATH="$CKPT_SEL"
fi
if [[ "$CKPT_PATH" != /* ]]; then
    CKPT_PATH="$ROOT/$CKPT_PATH"
fi
if [[ ! -e "$CKPT_PATH" ]]; then
    echo "checkpoint 不存在: $CKPT_PATH" >&2
    exit 1
fi
CKPT_PATH="$(readlink -f "$CKPT_PATH")"

aa_ask NUM_ENVS "并行环境数" "4"

RECORD_LOG="n"
TRAJECTORY_LOG_PATH=""
LOG_SECONDS="0"
if [[ "$NUM_ENVS" == "1" ]]; then
    aa_yes_no RECORD_LOG "记录误差日志" "y"
    if [[ "$RECORD_LOG" == "y" ]]; then
        aa_ask LOG_SECONDS "记录秒数（0=手动停止）" "60"
        if ! [[ "$LOG_SECONDS" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
            echo "记录秒数必须是非负数字: $LOG_SECONDS" >&2
            exit 1
        fi
        LOG_STAMP="$(date +%Y%m%d-%H%M%S)"
        TASK_TAG="${TASK//\//_}"
        MOTION_TAG="${MOTION//\//_}"
        CKPT_TAG="$(basename "${CKPT_PATH%.pt}")"
        TRAJECTORY_LOG_PATH="$ROOT/outputs/play_logs/${LOG_STAMP}-${TASK_TAG}-${MOTION_TAG}-${RECIPE}-${CKPT_TAG}.npz"
    fi
fi

HEADLESS_DEFAULT="n"
if [[ -n "${SSH_CONNECTION:-}" ]]; then
    HEADLESS_DEFAULT="y"
fi
aa_yes_no HEADLESS "无窗口 (headless)" "$HEADLESS_DEFAULT"

aa_show_gpus || true
IDLE_GPUS="$(aa_idle_gpus || true)"
PLAY_GPU="${IDLE_GPUS%%,*}"
if [[ -z "$PLAY_GPU" ]]; then
    PLAY_GPU="0"
fi
aa_ask GPU "使用的 GPU（逗号或区间）" "$PLAY_GPU"
GPU="$(aa_expand_gpus "$GPU")"
echo "使用 GPU: $GPU"

aa_yes_no HF_OFFLINE "Hugging Face 离线 (HF_HUB_OFFLINE=1)" "y"
aa_yes_no EXPORT_ONNX "生成 ONNX" "n"
aa_ask EXTRA "额外 Hydra 覆盖（直接回车跳过）" ""
EXTRA="${EXTRA#"${EXTRA%%[![:space:]]*}"}"
EXTRA="${EXTRA%"${EXTRA##*[![:space:]]}"}"
if [[ "$EXTRA" == "." || "$EXTRA" == "-" ]]; then
    EXTRA=""
fi

VENV="$(aa_venv_for_backend "$BACKEND")"
if [[ ! -d "$VENV" ]]; then
    echo "找不到 uv project: $VENV" >&2
    exit 1
fi

CMD=(env "CUDA_VISIBLE_DEVICES=$GPU")
if [[ "$BACKEND" == "mjlab" && "$RECORD_LOG" == "y" ]] \
    && awk -v value="$LOG_SECONDS" 'BEGIN { exit !(value > 0) }'; then
    # Viser is browser-based and does not need X11. MuJoCo video capture does:
    # select EGL before Python imports mujoco so remote recording works without DISPLAY.
    CMD+=("MUJOCO_GL=egl" "PYOPENGL_PLATFORM=egl")
fi
CMD+=(
    uv --project "$VENV" run
    "$MIMIC_SCRIPTS/play.py"
    "task=$TASK"
    "task/motion=$MOTION"
    "$EXP"
    "$ALGO_MODULE_KEY=$MODULE"
    "backend=$BACKEND"
    "task.num_envs=$NUM_ENVS"
    "task.termination.root_pos_error.enabled=false"
    "~task.termination.body_pos_error"
    "checkpoint_path=$CKPT_PATH"
)

if [[ "$HEADLESS" == "y" ]]; then
    CMD+=("headless=true")
else
    CMD+=("headless=false")
fi
if [[ "$RECIPE" == "ppo_roa" ]]; then
    CMD+=("+task/patches=teacher_future_t16")
fi
if [[ "$RECORD_LOG" == "y" ]]; then
    CMD+=("+trajectory_log_path=$TRAJECTORY_LOG_PATH")
    if awk -v value="$LOG_SECONDS" 'BEGIN { exit !(value > 0) }'; then
        CMD+=("render_seconds=$LOG_SECONDS")
    fi
fi
if ((${#MOTION_WEIGHT_ARGS[@]} > 0)); then
    CMD+=("${MOTION_WEIGHT_ARGS[@]}")
fi
if [[ -n "$EXTRA" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=($EXTRA)
    for tok in "${EXTRA_ARR[@]}"; do
        if [[ "$tok" == "." || "$tok" == "-" ]]; then
            continue
        fi
        if [[ "$tok" != *=* && "$tok" != +* ]]; then
            echo "忽略无效 Hydra 覆盖: $tok" >&2
            continue
        fi
        CMD+=("$tok")
    done
fi

HF_EXPORTS=()
if [[ "$HF_OFFLINE" == "y" ]]; then
    HF_EXPORTS+=(HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1)
fi
FULL_CMD=("${CMD[@]}")
if ((${#HF_EXPORTS[@]} > 0)); then
    FULL_CMD=(env "${HF_EXPORTS[@]}" "${CMD[@]}")
fi

EXPORT_DIR="$MIMIC_SCRIPTS/exports/BumiV2TrackBase"
EXPORT_CMD=()
if [[ "$EXPORT_ONNX" == "y" ]]; then
    EXPORT_CMD=(
        env "CUDA_VISIBLE_DEVICES=$GPU"
        uv --project "$VENV" run
        "$ROOT/scripts/tools/export_checkpoint_onnx.py"
        "$CKPT_PATH"
        --output-dir "$EXPORT_DIR"
        --backend mjlab
        --device cuda
    )
    if ((${#HF_EXPORTS[@]} > 0)); then
        EXPORT_CMD=(env "${HF_EXPORTS[@]}" "${EXPORT_CMD[@]}")
    fi
fi

echo
echo "checkpoint: $CKPT_PATH"
echo "venv:       $VENV"
if [[ "$RECORD_LOG" == "y" ]]; then
    echo "轨迹日志:   $TRAJECTORY_LOG_PATH"
    echo "             ${TRAJECTORY_LOG_PATH%.npz}.csv"
    echo "             ${TRAJECTORY_LOG_PATH%.npz}.summary.yaml"
fi
if [[ "$EXPORT_ONNX" == "y" ]]; then
    echo "ONNX:       $EXPORT_DIR"
fi
if ! aa_confirm_run "${FULL_CMD[@]}"; then
    echo "已取消"
    exit 0
fi

if [[ "$EXPORT_ONNX" == "y" ]]; then
    mkdir -p "$EXPORT_DIR"
    "${EXPORT_CMD[@]}"
fi

"${FULL_CMD[@]}"
