#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${MIMICLITE_CLUSTER_CONFIG:-$REPO_ROOT/.local/mimiclite_cluster.env}"

usage() {
  cat <<'EOF'
Usage: tools_local/mimiclite_cluster.sh COMMAND [ARGS]

Workstation commands:
  status                         Inspect both nodes without changing them
  preflight                      Validate GPUs, network, disk, Python and data
  bootstrap-direct              Initial direct upload; refuses existing targets
  sync-code                      Fast-forward both nodes from GitHub main
  verify-code [EXPECTED_SHA]     Verify local, GitHub and both nodes use one SHA
  launch-nccl RUN_ID             Run a two-node, 16-rank NCCL smoke test
  launch-smoke RUN_ID            Run a five-iteration training smoke test
  launch-train RUN_ID            Start a fresh 16-GPU training run in tmux
  training-status RUN_ID         Inspect ranks, GPUs, logs and checkpoints

Internal command (invoked over SSH):
  remote-node RANK RUN_ID ENVS ITERATIONS ROBOT_DIR SMPL_DIR
  remote-nccl RANK
EOF
}

load_config() {
  [[ -f "$CONFIG_FILE" ]] || {
    echo "Missing config: $CONFIG_FILE" >&2
    echo "Copy tools_local/mimiclite_cluster.env.example to it first." >&2
    exit 2
  }
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
  : "${NODE0_SSH:?}" "${NODE0_PORT:?}" "${NODE0_KEY:?}"
  : "${NODE1_SSH:?}" "${NODE1_PORT:?}" "${NODE1_KEY:?}"
  : "${LOCAL_PUSH_REPO:?}" "${GITHUB_REPO:?}" "${REMOTE_REPO:?}" "${REMOTE_PYTHON:?}" "${RUN_ROOT:?}"
  : "${MASTER_ADDR:?}" "${MASTER_PORT:?}" "${NCCL_MASTER_PORT:?}"
  : "${ROBOT_MOTION_DIR:?}" "${SMPL_MOTION_DIR:?}"
  NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-=eth0}"
  NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  NCCL_IB_HCA="${NCCL_IB_HCA:-}"
  REMOTE_OWNER="${REMOTE_OWNER:-root:root}"
}

validate_run_id() {
  [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Unsafe RUN_ID: $1" >&2; exit 2; }
}

node_value() {
  local node="$1" field="$2"
  case "$node:$field" in
    node0:host) printf '%s' "$NODE0_SSH" ;; node0:port) printf '%s' "$NODE0_PORT" ;;
    node0:key) printf '%s' "$NODE0_KEY" ;; node0:ip) printf '%s' "$NODE0_PRIVATE_IP" ;;
    node1:host) printf '%s' "$NODE1_SSH" ;; node1:port) printf '%s' "$NODE1_PORT" ;;
    node1:key) printf '%s' "$NODE1_KEY" ;; node1:ip) printf '%s' "$NODE1_PRIVATE_IP" ;;
    *) echo "Unknown node field: $node:$field" >&2; exit 2 ;;
  esac
}

check_key() {
  local key="$1" mode
  [[ -f "$key" ]] || { echo "Missing SSH key: $key" >&2; exit 2; }
  mode="$(stat -c '%a' "$key")"
  (( (8#$mode & 8#077) == 0 )) || {
    echo "SSH key is accessible by group/others ($mode): $key" >&2
    echo "Run: chmod 600 '$key'" >&2
    exit 2
  }
}

ssh_node() {
  local node="$1" host port key
  shift
  host="$(node_value "$node" host)"; port="$(node_value "$node" port)"; key="$(node_value "$node" key)"
  check_key "$key"
  ssh -F /dev/null -i "$key" -p "$port" -o BatchMode=yes \
    -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$host" "$@"
}

rsync_node() {
  local node="$1" source="$2" destination="$3" host port key ssh_transport
  host="$(node_value "$node" host)"; port="$(node_value "$node" port)"; key="$(node_value "$node" key)"
  check_key "$key"
  printf -v ssh_transport 'ssh -F /dev/null -i %q -p %q -o BatchMode=yes -o StrictHostKeyChecking=accept-new' "$key" "$port"
  rsync -a --partial -e "$ssh_transport" "$source" "$host:$destination"
}

rsync_tracked_node() {
  local node="$1" destination="$2" host port key ssh_transport
  host="$(node_value "$node" host)"; port="$(node_value "$node" port)"; key="$(node_value "$node" key)"
  check_key "$key"
  printf -v ssh_transport 'ssh -F /dev/null -i %q -p %q -o BatchMode=yes -o StrictHostKeyChecking=accept-new' "$key" "$port"
  git -C "$REPO_ROOT" ls-files -z | rsync -a --partial --from0 --files-from=- \
    -e "$ssh_transport" "$REPO_ROOT/" "$host:$destination"
}

status() {
  local node
  for node in node0 node1; do
    echo "[$node]"
    ssh_node "$node" "hostname; ip -br -4 addr show; git -C '$REMOTE_REPO' rev-parse --short HEAD 2>/dev/null || true; nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader; df -h /; tmux list-sessions 2>/dev/null || true"
  done
}

preflight() {
  local node expected_ip command
  for node in node0 node1; do
    expected_ip="$(node_value "$node" ip)"
    printf -v command 'set -eu; test "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -eq 8; ip -4 addr show dev eth0 | grep -Fq %q; test -x %q; test -d %q; test -d %q; test "$(df -Pk / | awk "NR==2 {print \\$4}")" -gt 52428800; echo PREFLIGHT_OK' "$expected_ip" "$REMOTE_PYTHON" "$ROBOT_MOTION_DIR" "$SMPL_MOTION_DIR"
    echo "[$node]"
    ssh_node "$node" "$command"
  done
}

bootstrap_direct() {
  local node
  [[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no)" ]] || {
    echo "Tracked worktree must be clean before bootstrap." >&2; exit 1;
  }
  [[ "$(git -C "$REPO_ROOT" remote get-url origin)" == "$LOCAL_PUSH_REPO" ]] || {
    echo "origin must be $LOCAL_PUSH_REPO before bootstrap." >&2; exit 1;
  }
  for node in node0 node1; do
    ssh_node "$node" "test ! -e '$REMOTE_REPO' || { echo 'Refusing existing target: $REMOTE_REPO' >&2; exit 1; }; mkdir -p '$REMOTE_REPO'"
    rsync_tracked_node "$node" "$REMOTE_REPO/"
    rsync_node "$node" "$REPO_ROOT/.git/" "$REMOTE_REPO/.git/"
    ssh_node "$node" "chown -R '$REMOTE_OWNER' '$REMOTE_REPO' && git -C '$REMOTE_REPO' remote set-url origin '$GITHUB_REPO'"
  done
}

sync_code() {
  local node command
  for node in node0 node1; do
    printf -v command 'set -eu; cd %q; test -z "$(git status --porcelain --untracked-files=no)"; test "$(git remote get-url origin)" = %q; git fetch origin main; git merge --ff-only origin/main; git rev-parse HEAD' "$REMOTE_REPO" "$GITHUB_REPO"
    echo "[$node]"
    ssh_node "$node" "$command"
  done
}

verify_code() {
  local expected="${1:-$(git -C "$REPO_ROOT" rev-parse HEAD)}" github_sha node actual dirty
  [[ "$(git -C "$REPO_ROOT" remote get-url origin)" == "$LOCAL_PUSH_REPO" ]] || { echo "Local origin mismatch" >&2; exit 1; }
  [[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no)" ]] || { echo "Local tracked worktree is dirty" >&2; exit 1; }
  github_sha="$(git -C "$REPO_ROOT" ls-remote "$GITHUB_REPO" refs/heads/main | awk '{print $1}')"
  [[ "$github_sha" == "$expected" ]] || { echo "GitHub SHA mismatch: ${github_sha:-missing} != $expected" >&2; exit 1; }
  echo "local+github CODE_OK $expected"
  for node in node0 node1; do
    actual="$(ssh_node "$node" "git -C '$REMOTE_REPO' rev-parse HEAD")"
    dirty="$(ssh_node "$node" "git -C '$REMOTE_REPO' status --porcelain --untracked-files=no")"
    [[ "$actual" == "$expected" ]] || { echo "$node SHA mismatch: $actual != $expected" >&2; exit 1; }
    [[ -z "$dirty" ]] || { echo "$node tracked worktree is dirty" >&2; exit 1; }
    echo "$node CODE_OK $actual"
  done
}

remote_env_prefix() {
  printf 'REMOTE_REPO=%q REMOTE_PYTHON=%q RUN_ROOT=%q MASTER_ADDR=%q MASTER_PORT=%q NCCL_MASTER_PORT=%q NCCL_SOCKET_IFNAME=%q NCCL_IB_DISABLE=%q NCCL_IB_HCA=%q OMNI_KIT_ACCEPT_EULA=%q' \
    "$REMOTE_REPO" "$REMOTE_PYTHON" "$RUN_ROOT" "$MASTER_ADDR" "$MASTER_PORT" "$NCCL_MASTER_PORT" "$NCCL_SOCKET_IFNAME" "$NCCL_IB_DISABLE" "$NCCL_IB_HCA" "$OMNI_KIT_ACCEPT_EULA"
}

remote_start() {
  local node="$1" rank="$2" run_id="$3" envs="$4" iterations="$5" robot_dir="$6" smpl_dir="$7"
  local session="SONIC_MimicLite_${run_id}_node${rank}" inner command env_prefix
  env_prefix="$(remote_env_prefix)"
  printf -v inner 'exec env %s bash tools_local/mimiclite_cluster.sh remote-node %q %q %q %q %q %q > %q 2>&1 < /dev/null' \
    "$env_prefix" "$rank" "$run_id" "$envs" "$iterations" "$robot_dir" "$smpl_dir" "$RUN_ROOT/$run_id/node${rank}.log"
  printf -v command 'cd %q && mkdir -p %q && ! tmux has-session -t %q 2>/dev/null && tmux new-session -d -s %q bash -lc %q && echo STARTED_TMUX:%s' \
    "$REMOTE_REPO" "$RUN_ROOT/$run_id" "$session" "$session" "$inner" "$session"
  ssh_node "$node" "$command"
}

launch_training() {
  local mode="$1" run_id="$2" envs iterations robot_dir smpl_dir
  validate_run_id "$run_id"
  if [[ "$mode" == smoke ]]; then
    envs="${SMOKE_ENVS_PER_GPU:-16}"; iterations="${SMOKE_ITERATIONS:-5}"
    robot_dir="${SMOKE_ROBOT_MOTION_DIR:-$ROBOT_MOTION_DIR}"; smpl_dir="${SMOKE_SMPL_MOTION_DIR:-$SMPL_MOTION_DIR}"
  else
    envs="${TRAIN_ENVS_PER_GPU:-4096}"; iterations="${TRAIN_ITERATIONS:-100000}"
    robot_dir="$ROBOT_MOTION_DIR"; smpl_dir="$SMPL_MOTION_DIR"
  fi
  verify_code
  preflight
  remote_start node1 1 "$run_id" "$envs" "$iterations" "$robot_dir" "$smpl_dir"
  remote_start node0 0 "$run_id" "$envs" "$iterations" "$robot_dir" "$smpl_dir"
  echo "Started $mode: run=$run_id world=16 envs/GPU=$envs iterations=$iterations"
}

remote_nccl_start() {
  local node="$1" rank="$2" run_id="$3" env_prefix inner command
  env_prefix="$(remote_env_prefix)"
  printf -v inner 'exec env %s bash tools_local/mimiclite_cluster.sh remote-nccl %q > %q 2>&1 < /dev/null' "$env_prefix" "$rank" "$RUN_ROOT/$run_id/nccl_node${rank}.log"
  printf -v command 'cd %q && mkdir -p %q && setsid -f bash -lc %q && echo STARTED' "$REMOTE_REPO" "$RUN_ROOT/$run_id" "$inner"
  ssh_node "$node" "$command"
}

launch_nccl() {
  local run_id="$1"
  validate_run_id "$run_id"; verify_code
  remote_nccl_start node1 1 "$run_id"; remote_nccl_start node0 0 "$run_id"
  echo "Started NCCL smoke: $run_id"
}

training_status() {
  local run_id="$1" node rank command
  validate_run_id "$run_id"
  for node in node0 node1; do
    [[ "$node" == node0 ]] && rank=0 || rank=1
    printf -v command 'run=%q; log="$run/node%s.log"; test -f "$log" || { echo "Missing log: $log"; exit 1; }; stat -c "log_bytes=%%s log_modified=%%y" "$log"; printf "training_ranks="; pgrep -fc "accelerate.*train_agent_trl.py" || true; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; grep -E "Traceback|CUDA out of memory|NCCL.*(error|Error)|ProcessExitedException" "$log" | tail -5 || true' "$RUN_ROOT/$run_id" "$rank"
    echo "[$node / node$rank]"; ssh_node "$node" "$command"
  done
  ssh_node node0 "grep -E 'Learning iteration|Mean rewards:|Total timesteps:' '$RUN_ROOT/$run_id/node0.log' | tail -6; find '$RUN_ROOT/$run_id' -maxdepth 1 -type f -name '*.pt' -printf '%f %s bytes\n' | sort | tail -10"
}

remote_nccl() {
  local rank="$1"
  cd "$REMOTE_REPO"
  export NCCL_SOCKET_IFNAME NCCL_IB_DISABLE
  [[ -z "$NCCL_IB_HCA" ]] || export NCCL_IB_HCA
  exec "$REMOTE_PYTHON" -m torch.distributed.run --nnodes=2 --nproc-per-node=8 \
    --node-rank="$rank" --master-addr="$MASTER_ADDR" --master-port="$NCCL_MASTER_PORT" \
    tools_local/nccl_smoke.py
}

remote_node() {
  local rank="$1" run_id="$2" envs="$3" iterations="$4" robot_dir="$5" smpl_dir="$6"
  local experiment_dir="$RUN_ROOT/$run_id"
  cd "$REMOTE_REPO"
  [[ "$OMNI_KIT_ACCEPT_EULA" == YES ]] || { echo "NVIDIA Omniverse EULA not accepted" >&2; exit 1; }
  [[ -x "$REMOTE_PYTHON" && -d "$robot_dir" && -d "$smpl_dir" ]] || { echo "Python or dataset missing" >&2; exit 1; }
  export TMPDIR="$RUN_ROOT/.tmp/node$rank" NCCL_SOCKET_IFNAME NCCL_IB_DISABLE OMNI_KIT_ACCEPT_EULA
  [[ -z "$NCCL_IB_HCA" ]] || export NCCL_IB_HCA
  mkdir -p "$TMPDIR" "$experiment_dir"
  printf 'sha=%s\nrank=%s\nstarted_utc=%s\n' "$(git rev-parse HEAD)" "$rank" "$(date -u +%FT%TZ)" > "$experiment_dir/node${rank}.metadata"
  exec "$REMOTE_PYTHON" -m accelerate.commands.launch --multi_gpu \
    --num_machines=2 --num_processes=16 --machine_rank="$rank" \
    --main_process_ip="$MASTER_ADDR" --main_process_port="$MASTER_PORT" \
    gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release \
    +resume=false checkpoint=null auto_load_latest=false use_wandb=false headless=True \
    "experiment_dir=$experiment_dir" "num_envs=$envs" \
    "++algo.config.num_learning_iterations=$iterations" \
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=$robot_dir" \
    "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=$smpl_dir"
}

case "${1:-}" in
  status) load_config; status ;;
  preflight) load_config; preflight ;;
  bootstrap-direct) load_config; bootstrap_direct ;;
  sync-code) load_config; sync_code ;;
  verify-code) load_config; verify_code "${2:-}" ;;
  launch-nccl) [[ $# -eq 2 ]] || { usage; exit 2; }; load_config; launch_nccl "$2" ;;
  launch-smoke) [[ $# -eq 2 ]] || { usage; exit 2; }; load_config; launch_training smoke "$2" ;;
  launch-train) [[ $# -eq 2 ]] || { usage; exit 2; }; load_config; launch_training train "$2" ;;
  training-status) [[ $# -eq 2 ]] || { usage; exit 2; }; load_config; training_status "$2" ;;
  remote-node) [[ $# -eq 7 ]] || { usage; exit 2; }; remote_node "$2" "$3" "$4" "$5" "$6" "$7" ;;
  remote-nccl) [[ $# -eq 2 ]] || { usage; exit 2; }; remote_nccl "$2" ;;
  *) usage; exit 2 ;;
esac
