#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
SCORE1="/data2/zcx/datasets/any4hdmi-bumi-v2/motions/score1"
SCORE3="/data2/zcx/datasets/any4hdmi-bumi-v2/motions/score3"
OUTPUTS="/data2/zcx/outputs/bumi_auto_quarantine"

GPUS="0,1,2,3,4,5,6,7"
NUM_ENVS=8192
ITERS=4000
MAX_RESTARTS=100
TMUX_MODE=0
SESSION=""
DRY_LOG=""
EXTRA=()

usage() {
    cat <<'EOF'
Usage: train_bumi_auto_quarantine.sh [options] [-- Hydra overrides...]
  --tmux                  run the controller in a new tmux session
  --session NAME          set tmux name and imply --tmux
  --gpus IDS              default: 0,1,2,3,4,5,6,7
  --num-envs N            environments per GPU, default: 8192
  --iters N               iterations per attempt, default: 4000
  --max-restarts N        default: 100
  --dry-run-log FILE      parse a log without moving files
  -h, --help              show help

On a failed attempt, only paths reported by "Non-finite env row" and located
under score1 are moved intact to score3. The next attempt starts from scratch.
EOF
}

while (($#)); do
    case "$1" in
        --tmux) TMUX_MODE=1; shift ;;
        --session) SESSION="$2"; TMUX_MODE=1; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --num-envs) NUM_ENVS="$2"; shift 2 ;;
        --iters) ITERS="$2"; shift 2 ;;
        --max-restarts) MAX_RESTARTS="$2"; shift 2 ;;
        --dry-run-log) DRY_LOG="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA+=("$@"); break ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$NUM_ENVS" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid --num-envs" >&2; exit 2; }
[[ "$ITERS" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid --iters" >&2; exit 2; }
[[ "$MAX_RESTARTS" =~ ^[0-9]+$ ]] || { echo "Invalid --max-restarts" >&2; exit 2; }

if ((TMUX_MODE)); then
    [[ -z "${BUMI_QUARANTINE_INNER:-}" ]] || { echo "Nested --tmux is not allowed" >&2; exit 2; }
    [[ -n "$SESSION" ]] || SESSION="bumi_auto_$(date +%Y%m%d_%H%M%S)"
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "Session exists: $SESSION" >&2; exit 2; }
    inner=(env BUMI_QUARANTINE_INNER=1 bash "$0"
        --gpus "$GPUS" --num-envs "$NUM_ENVS" --iters "$ITERS"
        --max-restarts "$MAX_RESTARTS")
    [[ -z "$DRY_LOG" ]] || inner+=(--dry-run-log "$DRY_LOG")
    (("${#EXTRA[@]}" == 0)) || inner+=(-- "${EXTRA[@]}")
    tmux new-session -d -s "$SESSION" "${inner[@]}"
    echo "Started: $SESSION"
    echo "Attach: tmux attach -t $SESSION"
    exit 0
fi

mkdir -p "$SCORE3"
SCORE1_REAL="$(realpath -e "$SCORE1")"
SCORE3_REAL="$(realpath -e "$SCORE3")"

extract_paths() {
    grep -oE 'motion_path=[^[:space:]]+\.npz' "$1" 2>/dev/null |
        sed 's/^motion_path=//' | sort -u || true
}

quarantine_log() {
    local log="$1" dry="$2" src real rel dst
    MOVED=0
    while IFS= read -r src; do
        [[ -n "$src" ]] || continue
        [[ "$src" == "$SCORE1/"*.npz ]] || { echo "Refused outside score1: $src" >&2; continue; }
        rel="${src#"$SCORE1"/}"
        dst="$SCORE3/$rel"
        if [[ ! -e "$src" ]]; then
            [[ -f "$dst" ]] && echo "Already quarantined: $rel" || echo "Missing: $src" >&2
            continue
        fi
        real="$(realpath -e "$src")"
        [[ "$real" == "$SCORE1_REAL/"* ]] || { echo "Refused resolved path: $real" >&2; continue; }
        [[ ! -e "$dst" ]] || { echo "Refused overwrite: $dst" >&2; continue; }
        if ((dry)); then
            echo "Would quarantine: $rel"
        else
            mkdir -p "$(dirname "$dst")"
            mv -- "$real" "$dst"
            printf '%s\t%s\t%s\n' "$(date -Is)" "$rel" "$log" >> "$MANIFEST"
            echo "Quarantined: $rel"
        fi
        MOVED=$((MOVED + 1))
    done < <(extract_paths "$log")
}

if [[ -n "$DRY_LOG" ]]; then
    [[ -f "$DRY_LOG" ]] || { echo "Missing log: $DRY_LOG" >&2; exit 2; }
    MANIFEST=/dev/null
    quarantine_log "$DRY_LOG" 1
    echo "Dry-run candidates: $MOVED"
    exit 0
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$OUTPUTS/$RUN_ID"
MANIFEST="$RUN_DIR/quarantine.tsv"
mkdir -p "$RUN_DIR"
printf 'timestamp\tmotion\tattempt_log\n' > "$MANIFEST"
exec > >(tee -a "$RUN_DIR/controller.log") 2>&1

STOP=0
trap 'STOP=1' INT TERM
echo "run=$RUN_ID score1=$SCORE1_REAL score3=$SCORE3_REAL"
echo "gpus=$GPUS num_envs=$NUM_ENVS iters=$ITERS logger=tensorboard"

for ((attempt=1; attempt<=MAX_RESTARTS+1; attempt++)); do
    attempt_dir="$RUN_DIR/attempt_$(printf '%03d' "$attempt")"
    log="$attempt_dir/train.log"
    mkdir -p "$attempt_dir"
    cmd=(env HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1
        bash "$ROOT/scripts/launch_ddp.sh" "$GPUS"
        "$ROOT/projects/mimic-lite/scripts/train.py" "$ROOT/venv/mjlab"
        task=tracking-bumi-v2 task/motion=bumi/v2 +exp=ppo/train
        algo/ppo/module=huge backend=mjlab "task.num_envs=$NUM_ENVS"
        "total_iters=$ITERS" "hydra.run.dir=$attempt_dir")
    cmd+=("${EXTRA[@]}")
    echo
    echo "Attempt $attempt start $(date -Is)"
    echo "TensorBoard log dir: $attempt_dir/tb"
    printf 'Command:'; printf ' %q' "${cmd[@]}"; echo
    set +e
    "${cmd[@]}" 2>&1 | tee "$log"
    status=${PIPESTATUS[0]}
    set -e
    echo "Attempt $attempt exit=$status $(date -Is)"
    if ((STOP)) || [[ "$status" -eq 130 || "$status" -eq 143 ]]; then
        echo "Stopped by user; no quarantine or restart."
        exit "$status"
    fi
    [[ "$status" -ne 0 ]] || { echo "Training completed successfully."; exit 0; }
    quarantine_log "$log" 0
    echo "Newly quarantined: $MOVED"
    ((MOVED > 0)) || { echo "No new bad motion; stopping." >&2; exit "$status"; }
    ((attempt <= MAX_RESTARTS)) || { echo "Restart limit reached." >&2; exit "$status"; }
    echo "Restarting from scratch."
done
