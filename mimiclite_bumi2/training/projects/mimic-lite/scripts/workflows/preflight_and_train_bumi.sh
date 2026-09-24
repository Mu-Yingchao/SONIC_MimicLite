#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
SCORE1="/data2/zcx/datasets/any4hdmi-bumi-v2/motions/score1"
SCORE3="/data2/zcx/datasets/any4hdmi-bumi-v2/motions/score3"
OUTPUTS="/data2/zcx/outputs/bumi_preflight_then_train"

GPUS="0,1,2,3,4,5,6,7"
SCREEN_ENVS=8192
SCREEN_STEPS=12
SCREEN_ACTION_STD=0.0
SCREEN_RETRIES=3
TRAIN_ENVS=8192
TRAIN_ITERS=4000
MAX_RESTARTS=100
TMUX_MODE=0
SESSION=""
EXTRA=()

usage() {
    cat <<'EOF'
Usage: preflight_and_train_bumi.sh [options] [-- training Hydra overrides...]
  --tmux                  run the full workflow in a new tmux session
  --session NAME          set tmux name and imply --tmux
  --gpus IDS              default: 0,1,2,3,4,5,6,7
  --screen-envs N         screening environments per GPU, default: 8192
  --screen-steps N        simulation steps per reset frame, default: 12
  --screen-action-std X   action stddev, default: 0.0 (zero action)
  --screen-retries N      resume failed screening up to N times, default: 3
  --train-envs N          training environments per GPU, default: 8192
  --train-iters N         formal PPO iterations, default: 4000
  --max-restarts N        formal-training fallback restarts, default: 100
  -h, --help              show help

The preflight scans every frame as a BUMI reset state. It never edits motion
contents. Complete .npz files reported as non-finite are moved from score1 to
score3, then formal BUMI V2 PPO training starts automatically.
EOF
}

while (($#)); do
    case "$1" in
        --tmux) TMUX_MODE=1; shift ;;
        --session) SESSION="$2"; TMUX_MODE=1; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --screen-envs) SCREEN_ENVS="$2"; shift 2 ;;
        --screen-steps) SCREEN_STEPS="$2"; shift 2 ;;
        --screen-action-std) SCREEN_ACTION_STD="$2"; shift 2 ;;
        --screen-retries) SCREEN_RETRIES="$2"; shift 2 ;;
        --train-envs) TRAIN_ENVS="$2"; shift 2 ;;
        --train-iters) TRAIN_ITERS="$2"; shift 2 ;;
        --max-restarts) MAX_RESTARTS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA+=("$@"); break ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for value in "$SCREEN_ENVS" "$SCREEN_STEPS" "$SCREEN_RETRIES" "$TRAIN_ENVS" "$TRAIN_ITERS"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Expected positive integer: $value" >&2; exit 2; }
done
[[ "$MAX_RESTARTS" =~ ^[0-9]+$ ]] || { echo "Invalid --max-restarts" >&2; exit 2; }
[[ "$SCREEN_ACTION_STD" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "Invalid --screen-action-std" >&2; exit 2; }
[[ "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "Invalid --gpus" >&2; exit 2; }

if ((TMUX_MODE)); then
    [[ -z "${BUMI_PREFLIGHT_INNER:-}" ]] || { echo "Nested --tmux is not allowed" >&2; exit 2; }
    [[ -n "$SESSION" ]] || SESSION="bumi_preflight_$(date +%Y%m%d_%H%M%S)"
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "Session exists: $SESSION" >&2; exit 2; }
    inner=(env BUMI_PREFLIGHT_INNER=1 bash "$0"
        --gpus "$GPUS" --screen-envs "$SCREEN_ENVS"
        --screen-steps "$SCREEN_STEPS" --screen-action-std "$SCREEN_ACTION_STD"
        --screen-retries "$SCREEN_RETRIES" --train-envs "$TRAIN_ENVS"
        --train-iters "$TRAIN_ITERS"
        --max-restarts "$MAX_RESTARTS")
    (("${#EXTRA[@]}" == 0)) || inner+=(-- "${EXTRA[@]}")
    tmux new-session -d -s "$SESSION" "${inner[@]}"
    echo "Started: $SESSION"
    echo "Attach: tmux attach -t $SESSION"
    exit 0
fi

mkdir -p "$SCORE3" "$OUTPUTS"
SCORE1_REAL="$(realpath -e "$SCORE1")"
SCORE3_REAL="$(realpath -e "$SCORE3")"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$OUTPUTS/$RUN_ID"
SCREEN_DIR="$RUN_DIR/screen"
MANIFEST="$RUN_DIR/quarantine.tsv"
mkdir -p "$SCREEN_DIR"
printf 'timestamp\tmotion\tsource\n' > "$MANIFEST"
exec > >(tee -a "$RUN_DIR/controller.log") 2>&1

echo "run=$RUN_ID"
echo "score1=$SCORE1_REAL score3=$SCORE3_REAL"
echo "screen: gpus=$GPUS envs_per_gpu=$SCREEN_ENVS steps=$SCREEN_STEPS action_std=$SCREEN_ACTION_STD"
echo "train: envs_per_gpu=$TRAIN_ENVS iterations=$TRAIN_ITERS logger=tensorboard"

screen_status=1
for ((attempt=1; attempt<=SCREEN_RETRIES; attempt++)); do
    echo
    echo "Screen attempt $attempt/$SCREEN_RETRIES start $(date -Is)"
    set +e
    env HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1 \
        bash "$ROOT/scripts/launch_ddp.sh" "$GPUS" \
        "$ROOT/projects/mimic-lite/scripts/screen_bumi_motions.py" \
        "$ROOT/venv/mjlab" \
        task=tracking-bumi-v2 task/motion=bumi/v2 +exp=ppo/train \
        algo/ppo/module=huge backend=mjlab \
        "task.num_envs=$SCREEN_ENVS" \
        "+screen.output_dir=$SCREEN_DIR" \
        "+screen.steps=$SCREEN_STEPS" \
        "+screen.action_std=$SCREEN_ACTION_STD" \
        +screen.resume=true \
        "hydra.run.dir=$RUN_DIR/screen_hydra"
    screen_status=$?
    set -e
    echo "Screen attempt $attempt exit=$screen_status $(date -Is)"
    ((screen_status == 0)) && break
    echo "Screen failed; completed ranks will be skipped and unfinished ranks resume."
done
((screen_status == 0)) || { echo "Screening failed after $SCREEN_RETRIES attempts; formal training not started." >&2; exit "$screen_status"; }

BAD_ALL="$RUN_DIR/bad_paths.txt"
find "$SCREEN_DIR" -maxdepth 1 -type f -name 'bad_paths.rank_*.txt' -print0 \
    | xargs -0 -r cat | sort -u > "$BAD_ALL"

moved=0
while IFS= read -r src; do
    [[ -n "$src" ]] || continue
    [[ "$src" == "$SCORE1/"*.npz ]] || { echo "Refused outside score1: $src" >&2; exit 1; }
    [[ -f "$src" ]] || { echo "Missing screening candidate: $src" >&2; exit 1; }
    real="$(realpath -e "$src")"
    [[ "$real" == "$SCORE1_REAL/"* ]] || { echo "Refused resolved path: $real" >&2; exit 1; }
    rel="${real#"$SCORE1_REAL/"}"
    dst="$SCORE3/$rel"
    [[ ! -e "$dst" ]] || { echo "Refused overwrite: $dst" >&2; exit 1; }
    mkdir -p "$(dirname "$dst")"
    mv -- "$real" "$dst"
    printf '%s\t%s\tpreflight\n' "$(date -Is)" "$rel" >> "$MANIFEST"
    echo "Quarantined after preflight: $rel"
    moved=$((moved + 1))
done < "$BAD_ALL"

echo "Preflight complete: candidates=$moved"
printf 'score1 motions after preflight: '
find "$SCORE1" -type f -name '*.npz' | wc -l
printf 'score3 motions after preflight: '
find "$SCORE3" -type f -name '*.npz' | wc -l

echo
echo "Formal BUMI V2 PPO training start $(date -Is)"
train_cmd=(bash "$ROOT/projects/mimic-lite/scripts/workflows/train_bumi_auto_quarantine.sh"
    --gpus "$GPUS" --num-envs "$TRAIN_ENVS" --iters "$TRAIN_ITERS"
    --max-restarts "$MAX_RESTARTS")
train_overrides=(
    '~task.randomization.perturb_body_materials'
    '~task.randomization.perturb_body_mass'
    '~task.randomization.perturb_body_com'
    '~task.randomization.actuator_params'
)
train_overrides+=("${EXTRA[@]}")
train_cmd+=(-- "${train_overrides[@]}")
printf 'Command:'; printf ' %q' "${train_cmd[@]}"; echo
"${train_cmd[@]}"
echo "Full workflow completed successfully $(date -Is)"
