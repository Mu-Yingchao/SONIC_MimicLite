#!/usr/bin/env bash
# Shared helpers for MimicLite interactive train/play scripts.

aa_find_repo_root() {
    local d
    d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    while [[ "$d" != "/" ]]; do
        if [[ -d "$d/active_adaptation" && -f "$d/pyproject.toml" ]]; then
            echo "$d"
            return 0
        fi
        d="$(dirname "$d")"
    done
    echo "找不到 active-adaptation 仓库根目录" >&2
    return 1
}

ROOT="$(aa_find_repo_root)"
MIMIC_DIR="$ROOT/projects/mimic-lite"
MIMIC_SCRIPTS="$MIMIC_DIR/scripts"
CFG_DIR="$MIMIC_DIR/cfg"
OUTPUTS_DIR="$ROOT/outputs"

if [[ -f "$HOME/.local/bin/env" ]]; then
    # shellcheck source=/dev/null
    source "$HOME/.local/bin/env"
fi

aa_header() {
    echo
    echo "======== $* ========"
}

aa_ask() {
    local __var="$1"
    local __prompt="$2"
    local __default="${3:-}"
    local __reply
    if [[ -n "$__default" ]]; then
        read -r -p "$__prompt [$__default]: " __reply || exit 1
    else
        read -r -p "$__prompt: " __reply || exit 1
    fi
    if [[ -z "$__reply" ]]; then
        __reply="$__default"
    fi
    printf -v "$__var" '%s' "$__reply"
}

aa_yes_no() {
    local __var="$1"
    local __prompt="$2"
    local __default="${3:-y}"
    local __hint="y/n"
    local __reply
    if [[ "$__default" == "y" ]]; then
        __hint="Y/n"
    else
        __hint="y/N"
    fi
    read -r -p "$__prompt [$__hint]: " __reply || exit 1
    if [[ -z "$__reply" ]]; then
        __reply="$__default"
    fi
    __reply="$(echo "$__reply" | tr '[:upper:]' '[:lower:]')"
    if [[ "$__reply" == "y" || "$__reply" == "yes" ]]; then
        printf -v "$__var" '%s' "y"
    else
        printf -v "$__var" '%s' "n"
    fi
}

aa_pick() {
    local __var="$1"
    local __prompt="$2"
    local __default="$3"
    shift 3
    local __items=("$@")
    local __i __mark __choice __idx __default_idx=""

    echo
    echo "$__prompt"
    for __i in "${!__items[@]}"; do
        __mark=""
        if [[ "${__items[$__i]}" == "$__default" ]]; then
            __mark="  <- 默认"
            __default_idx="$((__i + 1))"
        fi
        printf "  %2d) %s%s\n" "$((__i + 1))" "${__items[$__i]}" "$__mark"
    done

    local __hint="$__default"
    if [[ -n "$__default_idx" ]]; then
        __hint="$__default_idx"
    fi
    read -r -p "选择 [${__hint}]: " __choice || exit 1
    if [[ -z "$__choice" ]]; then
        printf -v "$__var" '%s' "$__default"
        return 0
    fi
    if [[ "$__choice" =~ ^[0-9]+$ ]]; then
        __idx=$((__choice - 1))
        if (( __idx < 0 || __idx >= ${#__items[@]} )); then
            echo "无效编号: $__choice" >&2
            exit 1
        fi
        printf -v "$__var" '%s' "${__items[$__idx]}"
        return 0
    fi
    printf -v "$__var" '%s' "$__choice"
}

aa_list_yaml_stems() {
    local dir="$1"
    local prefix="${2:-}"
    local path stem rel
    [[ -d "$dir" ]] || return 0
    while IFS= read -r path; do
        rel="${path#"$dir"/}"
        stem="${rel%.yaml}"
        if [[ -n "$prefix" ]]; then
            echo "${prefix}${stem}"
        else
            echo "$stem"
        fi
    done < <(find "$dir" -type f -name '*.yaml' | sort)
}

aa_list_tasks() {
    find "$CFG_DIR/task" -maxdepth 1 -type f -name '*.yaml' -printf '%f\n' \
        | sed 's/\.yaml$//' | sort
}

aa_list_motions() {
    aa_list_yaml_stems "$CFG_DIR/task/motion"
}

aa_list_modules() {
    local family="$1"
    aa_list_yaml_stems "$CFG_DIR/algo/${family}/module"
}

aa_venv_for_backend() {
    local backend="$1"
    if [[ "$backend" == "isaaclab" ]]; then
        echo "$ROOT/venv/isaaclab"
    else
        echo "$ROOT/venv/mjlab"
    fi
}

aa_show_gpus() {
    echo
    echo "当前 GPU:"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "  未找到 nvidia-smi"
        return 1
    fi
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv
}

aa_idle_gpus() {
    local ids=()
    local line idx used
    while IFS=',' read -r idx used _; do
        idx="${idx// /}"
        used="${used// /}"
        used="${used%%.*}"
        if [[ -n "$used" && "$used" -lt 500 ]]; then
            ids+=("$idx")
        fi
    done < <(
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null
    )
    local IFS=','
    echo "${ids[*]}"
}

# Expand "1-7" or "0,2-4,7" into "1,2,3,4,5,6,7" / "0,2,3,4,7".
aa_expand_gpus() {
    local spec="${1// /}"
    local parts part a b i
    local out=()
    if [[ -z "$spec" ]]; then
        echo ""
        return 0
    fi
    IFS=',' read -ra parts <<< "$spec"
    for part in "${parts[@]}"; do
        if [[ "$part" =~ ^[0-9]+-[0-9]+$ ]]; then
            a="${part%-*}"
            b="${part#*-}"
            if (( a > b )); then
                echo "无效 GPU 区间: $part" >&2
                exit 1
            fi
            for ((i = a; i <= b; i++)); do
                out+=("$i")
            done
        elif [[ "$part" =~ ^[0-9]+$ ]]; then
            out+=("$part")
        else
            echo "无效 GPU 编号: $part（请用 1,2,3 或 1-7）" >&2
            exit 1
        fi
    done
    local IFS=','
    echo "${out[*]}"
}

aa_list_checkpoints() {
    [[ -d "$OUTPUTS_DIR" ]] || return 0
    # Follow OUTPUTS_DIR when outputs is stored elsewhere through a symlink,
    # but do not follow nested W&B artifact links or list their duplicates.
    find -H "$OUTPUTS_DIR" \( -type f -o -type l \) \
        \( -name 'checkpoint_*.pt' -o -name 'checkpoint_latest.pt' \) \
        ! -name 'checkpoint_temp.pt' ! -path '*/wandb/*' -printf '%T@\t%p\n' 2>/dev/null \
        | sort -nr \
        | awk -F '\t' '{print $2}'
}

aa_relpath() {
    local path="$1"
    if [[ "$path" == "$ROOT/"* ]]; then
        echo "${path#"$ROOT"/}"
    else
        echo "$path"
    fi
}

aa_confirm_run() {
    local __ok
    echo
    echo "即将执行:"
    printf '  '
    printf '%q ' "$@"
    echo
    echo
    aa_yes_no __ok "确认启动?" "y"
    [[ "$__ok" == "y" ]]
}

aa_ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        return 0
    fi
    if [[ -x "$HOME/.local/bin/uv" ]]; then
        export PATH="$HOME/.local/bin:$PATH"
        return 0
    fi
    echo "未找到 uv，请先安装" >&2
    exit 1
}
