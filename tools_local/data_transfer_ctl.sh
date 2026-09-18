#!/bin/bash
# BUMI2 npz / 人体 BVH 数据传输控制脚本。
#
# 设计目的：让用户不依赖 Claude 会话本身，就能随时暂停/继续/查看这两路
# 传输，且暂停不会丢数据——依赖 rsync --partial 的断点续传，停止后重新
# start 会跳过已完整下载的文件，只续传未完成的部分。
#
# 密钥和分片清单放在被 Git 忽略的 .local/data_transfer/，不放在会话临时
# 目录（/tmp 下的 scratchpad 会随 IDE/终端重启被清空，之前已经因此断过一次）。
#
# 用法：
#   bash tools_local/data_transfer_ctl.sh status   查看两路传输的进程数/进度
#   bash tools_local/data_transfer_ctl.sh stop      停止两路传输（不丢数据，可重新 start 续传）
#   bash tools_local/data_transfer_ctl.sh start     启动/续传两路传输（已在跑则跳过）

set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="$REPO_ROOT/.local/data_transfer"
KEY="$CFG/transfer_key"

BUMI2_DEST="/home/yingchaomu/下载/bumi_v2_data_cache/bumi_v2_filtered"
BUMI2_SRC="user@112.65.216.193:/data0/yc/bumi_v2_data/bumi_v2_filtered"
BUMI2_PORT=50032
BUMI2_PARALLEL=3

BVH_DEST="/home/yingchaomu/下载/bvh_data_cache/soma_uniform"
BVH_SRC="muyingchao@117.161.121.54:/data/muyingchao/soma_uniform"
BVH_PORT=22115
BVH_PARALLEL=4

# 已对齐/筛选好的 BUMI3 SONIC 训练用 SMPL PKL（97660 条，22G），和 BVH 同一台
# 服务器，并发数刻意压低，避免和 BVH 传输抢占同一条链路。
SMPL_DEST="/home/yingchaomu/下载/bumi3_smpl_filtered_cache/smpl"
SMPL_SRC="muyingchao@117.161.121.54:/data/ouqin/datasets/bumi3/train/smpl"
SMPL_PORT=22115
SMPL_PARALLEL=3

MARKER_TAG="sonic_mimiclite_data_transfer"  # 出现在命令行里，用于精确匹配/停止本脚本启的进程

NOETIX9_KEY="/home/yingchaomu/下载/SONIC_MimicLite/.local/keys/noetix9.pem"
NOETIX9_SSH="root@14.103.42.170"
NOETIX9_PORT=22

run_one() {
  local name="$1" chunk_glob="$2" dest="$3" parallel="$4" src="$5" port="$6"
  if pgrep -f "$MARKER_TAG:$name" > /dev/null 2>&1; then
    echo "[$name] 已在运行，跳过"
    return
  fi
  mkdir -p "$dest"
  setsid nohup bash -c "
    : $MARKER_TAG:$name
    ls $CFG/${chunk_glob} | xargs -P $parallel -I{} \
      nice -n 19 ionice -c3 rsync -a --partial \
        -e 'ssh -i $KEY -p $port -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6' \
        --files-from={} '$src/' '$dest/'
  " > "$CFG/${name}.log" 2>&1 < /dev/null &
  disown
  echo "[$name] 已启动"
}

run_push() {
  local name="$1" chunk_glob="$2" local_dir="$3" remote_dir="$4" parallel="$5"
  if pgrep -f "$MARKER_TAG:push_$name" > /dev/null 2>&1; then
    echo "[push_$name] 已在运行，跳过"
    return
  fi
  # Noetix-9 的 sshd 用默认 MaxStartups（10:30:100），短时间内并发建立太多新
  # SSH 连接会被概率性拒绝；rsync 拿到该错误会以退出码 255 结束，xargs 遇到
  # 255 会直接中止整批而不是跳过重试。这里用 for 循环做最多 3 次重试，单个
  # chunk 偶发连接失败不再拖垮整个 push 任务。
  setsid nohup bash -c "
    : $MARKER_TAG:push_$name
    ls $CFG/${chunk_glob} | xargs -P $parallel -I{} bash -c '
      for attempt in 1 2 3; do
        nice -n 19 ionice -c3 rsync -a --partial \
          -e \"ssh -i $NOETIX9_KEY -p $NOETIX9_PORT -o ProxyCommand=none -o ServerAliveInterval=30 -o ServerAliveCountMax=6\" \
          --files-from={} \"$local_dir/\" \"$NOETIX9_SSH:$remote_dir/\" && break
        sleep \$((attempt * 3))
      done
    '
  " > "$CFG/push_${name}.log" 2>&1 < /dev/null &
  disown
  echo "[push_$name] 已启动"
}

status_push_one() {
  local name="$1" remote_dir="$2" total_human="$3"
  local n
  n=$(pgrep -fc "$MARKER_TAG:push_$name" 2>/dev/null)
  n="${n:-0}"
  echo "[push_$name] 运行中的推送进程数: $n"
  local remote_stat
  remote_stat=$(ssh -i "$NOETIX9_KEY" -p "$NOETIX9_PORT" -o ProxyCommand=none "$NOETIX9_SSH" \
    "du -sh '$remote_dir' 2>/dev/null; find '$remote_dir' -type f 2>/dev/null | wc -l" 2>/dev/null)
  echo "  Noetix-9 已有: $(echo "$remote_stat" | sed -n '1p') / 目标约 $total_human，文件数 $(echo "$remote_stat" | sed -n '2p')"
}

status_one() {
  local name="$1" dest="$2" total_human="$3"
  local n
  n=$(pgrep -fc "$MARKER_TAG:$name" 2>/dev/null)
  n="${n:-0}"
  echo "[$name] 运行中的传输进程数: $n"
  if [ -d "$dest" ]; then
    echo "  已下载: $(du -sh "$dest" 2>/dev/null | cut -f1) / 目标约 $total_human"
    echo "  已下载文件数: $(find "$dest" -type f 2>/dev/null | wc -l)"
  else
    echo "  目录尚不存在"
  fi
}

case "${1:-}" in
  start)
    run_one bumi2 "chunk_[0-9][0-9]" "$BUMI2_DEST" "$BUMI2_PARALLEL" "$BUMI2_SRC" "$BUMI2_PORT"
    run_one bvh "bvh_chunk_[0-9][0-9]" "$BVH_DEST" "$BVH_PARALLEL" "$BVH_SRC" "$BVH_PORT"
    run_one smpl "smpl_chunk_[0-9][0-9]" "$SMPL_DEST" "$SMPL_PARALLEL" "$SMPL_SRC" "$SMPL_PORT"
    ;;
  stop)
    pkill -f "$MARKER_TAG:bumi2" 2>/dev/null
    pkill -f "$MARKER_TAG:bvh" 2>/dev/null
    pkill -f "$MARKER_TAG:smpl" 2>/dev/null
    sleep 1
    echo "已发送停止信号（rsync --partial 断点已保留，重新 start 会续传）"
    ;;
  status)
    status_one bumi2 "$BUMI2_DEST" "51G"
    status_one bvh "$BVH_DEST" "277G"
    status_one smpl "$SMPL_DEST" "22G"
    ;;
  push)
    # 三路合计并发压到 7（Noetix-9 sshd 默认 MaxStartups 10:30:100，超过 10
    # 个并发新连接会开始被概率性拒绝，这里刻意留出安全边际，不是越大越快）。
    run_push bumi2 "chunk_[0-9][0-9]" "$BUMI2_DEST" "/data0/bumi_v2_filtered" 2
    run_push bvh "bvh_chunk_[0-9][0-9]" "$BVH_DEST" "/data0/soma_uniform" 3
    run_push smpl "smpl_chunk_[0-9][0-9]" "$SMPL_DEST" "/data0/bumi3_smpl_filtered/smpl" 2
    ;;
  push-stop)
    pkill -f "$MARKER_TAG:push_bumi2" 2>/dev/null
    pkill -f "$MARKER_TAG:push_bvh" 2>/dev/null
    pkill -f "$MARKER_TAG:push_smpl" 2>/dev/null
    sleep 1
    echo "已发送停止信号（rsync --partial 断点已保留，重新 push 会续传）"
    ;;
  push-status)
    status_push_one bumi2 "/data0/bumi_v2_filtered" "51G"
    status_push_one bvh "/data0/soma_uniform" "277G"
    status_push_one smpl "/data0/bumi3_smpl_filtered/smpl" "22G"
    ;;
  *)
    echo "用法: $0 {start|stop|status|push|push-stop|push-status}"
    exit 1
    ;;
esac
