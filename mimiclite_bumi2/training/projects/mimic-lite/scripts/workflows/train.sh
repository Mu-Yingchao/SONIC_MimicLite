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

aa_header "MimicLite 训练"

aa_list_resume_checkpoints() {
    local recipe="$1"
    local checkpoint
    while IFS= read -r checkpoint; do
        if [[ "$recipe" == "ppo_roa_sequential" ]]; then
            [[ "$checkpoint" == */stages/03-finetune/* ]] || continue
        else
            [[ "$checkpoint" == */stages/* ]] && continue
            [[ "${checkpoint,,}" == *ppo_roa* ]] && continue
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

aa_pick RECIPE "训练配方" "ppo" \
    ppo \
    ppo_roa_sequential

if [[ "$RECIPE" == "ppo_roa_sequential" ]]; then
    mapfile -t MODULES < <(aa_list_modules ppo_roa)
    DEFAULT_MODULE="huge"
    ALGO_MODULE_KEY="algo/ppo_roa/module"
else
    mapfile -t MODULES < <(aa_list_modules ppo)
    DEFAULT_MODULE="huge"
    ALGO_MODULE_KEY="algo/ppo/module"
fi
if [[ "$RECIPE" == "ppo_roa_sequential" ]]; then
    aa_pick TRAIN_MODE "训练模式：new 从零 / resume 续当前阶段 / warm_start 旧权重三阶段" "new" new resume warm_start
else
    aa_pick TRAIN_MODE "训练模式" "new" new resume
fi
if [[ "$TRAIN_MODE" == "new" ]]; then
    aa_pick MODULE "网络规模 (module)" "$DEFAULT_MODULE" "${MODULES[@]}"
else
    MODULE="from_checkpoint"
fi

RESUME_CKPT=""
RESUME_ITER=""
if [[ "$TRAIN_MODE" != "new" ]]; then
    mapfile -t RESUME_CKPTS < <(aa_list_resume_checkpoints "$RECIPE")
    CKPT_CHOICES=()
    for checkpoint in "${RESUME_CKPTS[@]}"; do
        CKPT_CHOICES+=("$(aa_relpath "$checkpoint")")
        if ((${#CKPT_CHOICES[@]} >= 30)); then
            break
        fi
    done
    CKPT_CHOICES+=("手动输入路径")

    DEFAULT_CKPT="手动输入路径"
    if ((${#RESUME_CKPTS[@]} > 0)); then
        DEFAULT_CKPT="$(aa_relpath "${RESUME_CKPTS[0]}")"
    fi
    aa_pick CKPT_SEL \
        "Resume checkpoint（PPO-ROA 只显示 03-finetune）" \
        "$DEFAULT_CKPT" \
        "${CKPT_CHOICES[@]}"
    if [[ "$CKPT_SEL" == "手动输入路径" ]]; then
        aa_ask RESUME_CKPT "输入 checkpoint 路径" ""
        if [[ -z "$RESUME_CKPT" ]]; then
            echo "resume 必须指定 checkpoint" >&2
            exit 1
        fi
    else
        RESUME_CKPT="$CKPT_SEL"
    fi
    if [[ "$RESUME_CKPT" != /* ]]; then
        RESUME_CKPT="$ROOT/$RESUME_CKPT"
    fi
    if [[ ! -e "$RESUME_CKPT" ]]; then
        echo "checkpoint 不存在: $RESUME_CKPT" >&2
        exit 1
    fi
    RESUME_CKPT="$(readlink -f "$RESUME_CKPT")"
    if [[ "${RESUME_CKPT##*/}" =~ ^checkpoint_([0-9]+)\.pt$ ]]; then
        RESUME_ITER="${BASH_REMATCH[1]}"
    fi
fi

aa_show_gpus || true
IDLE_GPUS="$(aa_idle_gpus || true)"
if [[ -z "$IDLE_GPUS" ]]; then
    IDLE_GPUS="0"
    echo
    echo "没有明显空闲 GPU，默认 0。占用中的卡请不要再选。"
fi
aa_ask GPUS "GPU 编号（逗号或区间，如 1-7）" "$IDLE_GPUS"
GPUS="$(aa_expand_gpus "$GPUS")"
echo "使用 GPU: $GPUS"

aa_yes_no HF_OFFLINE "Hugging Face 离线 (HF_HUB_OFFLINE=1)" "y"
aa_ask NUM_ENVS "每卡环境数 (task.num_envs)" "8192"
STAGE_ARGS=()
if [[ "$TRAIN_MODE" == "warm_start" ]]; then
    aa_ask TRAIN_ITERS "Teacher 训练轮数" "2000"
    aa_ask ADAPT_ITERS "Adapt 训练轮数" "2000"
    aa_ask FINETUNE_ITERS "Student finetune 轮数" "2000"
    for count in "$TRAIN_ITERS" "$ADAPT_ITERS" "$FINETUNE_ITERS"; do
        if ! [[ "$count" =~ ^[1-9][0-9]*$ ]]; then
            echo "各阶段轮数必须是正整数" >&2
            exit 1
        fi
    done
    STAGE_ARGS=("stage_iters.train=$TRAIN_ITERS" "stage_iters.adapt=$ADAPT_ITERS" "stage_iters.finetune=$FINETUNE_ITERS")
    TOTAL_ITERS="$((TRAIN_ITERS + ADAPT_ITERS + FINETUNE_ITERS))"
else
TOTAL_ITERS_DEFAULT="4000"
if [[ "$TRAIN_MODE" == "resume" ]]; then
    TOTAL_ITERS_DEFAULT="40000"
fi
aa_ask TOTAL_ITERS "目标总 iteration (total_iters)" "$TOTAL_ITERS_DEFAULT"
if ! [[ "$TOTAL_ITERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "total_iters 必须是正整数: $TOTAL_ITERS" >&2
    exit 1
fi
if [[ -n "$RESUME_ITER" ]] && (( TOTAL_ITERS <= RESUME_ITER )); then
    echo "total_iters ($TOTAL_ITERS) 必须大于 checkpoint iteration ($RESUME_ITER)" >&2
    exit 1
fi
fi
aa_ask EXTRA "额外 Hydra 覆盖（直接回车跳过，例如 total_iters=1000）" ""
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

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NPROC="${#GPU_ARR[@]}"

ALGO_ARGS=()
if [[ "$TRAIN_MODE" != "new" ]]; then
    ALGO_ARGS+=("algo=from_checkpoint")
else
    ALGO_ARGS+=("$ALGO_MODULE_KEY=$MODULE")
fi

CMD=()
if [[ "$TRAIN_MODE" == "warm_start" ]]; then
    CMD=(
        env "CUDA_VISIBLE_DEVICES=$GPUS"
        uv --project "$VENV" run
        "$MIMIC_SCRIPTS/train_sequential.py"
        "task=$TASK" "task/motion=$MOTION"
        "+task/patches=teacher_future_t16"
        "backend=$BACKEND" "task.num_envs=$NUM_ENVS"
        "nproc_per_node=$NPROC"
        "warm_start_checkpoint=$RESUME_CKPT"
        "${STAGE_ARGS[@]}"
    )
elif [[ "$RECIPE" == "ppo_roa_sequential" && "$TRAIN_MODE" == "new" ]]; then
    CMD=(
        env "CUDA_VISIBLE_DEVICES=$GPUS"
        uv --project "$VENV" run
        "$MIMIC_SCRIPTS/train_sequential.py"
        "task=$TASK"
        "task/motion=$MOTION"
        "+task/patches=teacher_future_t16"
        "${ALGO_ARGS[@]}"
        "backend=$BACKEND"
        "task.num_envs=$NUM_ENVS"
        "total_iters=$TOTAL_ITERS"
        "nproc_per_node=$NPROC"
    )
else
    EXP_ARGS=()
    PATCH_ARGS=()
    if [[ "$RECIPE" == "ppo_roa_sequential" ]]; then
        PATCH_ARGS+=("+task/patches=teacher_future_t16")
    fi
    if [[ "$TRAIN_MODE" == "new" ]]; then
        if [[ "$RECIPE" == "ppo_roa_sequential" ]]; then
            EXP_ARGS+=("+exp=ppo_roa/finetune")
        else
            EXP_ARGS+=("+exp=ppo/train")
        fi
    fi
    CMD=(
        bash "$ROOT/scripts/launch_ddp.sh" "$GPUS"
        "$MIMIC_SCRIPTS/train.py" "$VENV"
        "task=$TASK"
        "task/motion=$MOTION"
        "${PATCH_ARGS[@]}"
        "${EXP_ARGS[@]}"
        "${ALGO_ARGS[@]}"
        "backend=$BACKEND"
        "task.num_envs=$NUM_ENVS"
        "total_iters=$TOTAL_ITERS"
    )
fi

if [[ "$TRAIN_MODE" == "resume" ]]; then
    CMD+=("checkpoint_path=$RESUME_CKPT")
    if [[ "$RECIPE" == "ppo" ]]; then
        CMD+=("+resume_rebase_schedules=true")
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

aa_pick RUN_MODE "运行方式" "tmux" tmux foreground
STAMP="$(date +%Y%m%d_%H%M%S)"
SESSION="mimic_train_${STAMP}"
LOG_DIR="$OUTPUTS_DIR/logs"
LOG_FILE="$LOG_DIR/${SESSION}.log"

echo
echo "venv:    $VENV"
echo "log:     $LOG_FILE"
echo "mode:    $TRAIN_MODE"
if [[ "$TRAIN_MODE" != "new" ]]; then
    echo "checkpoint: $RESUME_CKPT"
fi
if [[ "$TRAIN_MODE" == "warm_start" ]]; then
    echo "轮数: train=$TRAIN_ITERS / adapt=$ADAPT_ITERS / finetune=$FINETUNE_ITERS（各阶段从 0 开始）"
fi
if [[ "$RUN_MODE" == "tmux" ]]; then
    echo "tmux:    $SESSION"
fi

if ! aa_confirm_run "${FULL_CMD[@]}"; then
    echo "已取消"
    exit 0
fi

mkdir -p "$LOG_DIR"
{
    echo "launch_start=$(date -Is)"
    echo "host=$(hostname) gpus=$GPUS task=$TASK motion=$MOTION recipe=$RECIPE mode=$TRAIN_MODE module=$MODULE iters=$TOTAL_ITERS backend=$BACKEND logger=tensorboard resume=$RESUME_CKPT"
    printf 'cmd: %q ' "${FULL_CMD[@]}"
    echo
} | tee "$LOG_FILE"

if [[ "$RUN_MODE" == "tmux" ]]; then
    TMUX_LINE="cd $(printf '%q' "$ROOT") && $(printf '%q ' "${FULL_CMD[@]}") 2>&1 | tee -a $(printf '%q' "$LOG_FILE"); echo; echo DONE; exec bash"
    tmux new-session -d -s "$SESSION" bash -lc "$TMUX_LINE"
    echo "已在 tmux 启动: $SESSION"
    echo "  tmux attach -t $SESSION"
    echo "  日志: $LOG_FILE"
else
    "${FULL_CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
fi
