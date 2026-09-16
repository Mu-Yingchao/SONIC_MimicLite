# SONIC_MimicLite 项目状态记录

维护目的：本仓库的集群控制闭环（本地 VSCode/Codex 控制两台服务器 16 卡训练）刚完成初始化，
细节容易随时间遗忘。本文件只记录**已核实的事实**和**明确的下一步**，供后续会话直接续接。
架构和操作规范以 [`docs/source/getting_started/codex_local_control_multi_node_training.md`](docs/source/getting_started/codex_local_control_multi_node_training.md)
为准，本文件不重复搬运其内容。

> 本项目是 SONIC → **BUMI3** 机型迁移，与全局 `~/下载/CLAUDE.md` 描述的 SONIC_N3 项目
> （远程 `yuanjie-server`、目标机型 N3）是两个不同项目，互不影响，不要混用其路径、
> 服务器别名或结论。

## 1. 项目三端身份

| 端 | 地址/路径 |
|---|---|
| GitHub | `git@github.com:Mu-Yingchao/SONIC_MimicLite.git`（push）/ `https://github.com/Mu-Yingchao/SONIC_MimicLite.git`（服务器只读 pull） |
| 本地（控制面 + 部署端） | `/home/yingchaomu/下载/SONIC_MimicLite` |
| Noetix-0（world rank 0，8×RTX4090D） | `118.196.95.17`（内网 `172.31.0.32`），`/root/SONIC_MimicLite`，密钥 `Noetix-2-7.pem` |
| Noetix-9（8×RTX4090D） | `14.103.42.170`（内网 `172.31.0.2`），`/root/SONIC_MimicLite`，密钥 `noetix-8.pem` |
| Run 目录 | `/root/SONIC_MimicLite_runs/<RUN_ID>` |

原始私钥留在 `/home/yingchaomu/下载/Noetix-2-7.pem`、`/home/yingchaomu/下载/noetix-8.pem`；
仓库内实际引用的是被 Git 忽略的受限副本 `.local/keys/noetix0.pem`、`.local/keys/noetix9.pem`
（均 600 权限）。真实集群配置在 `.local/mimiclite_cluster.env`（同样 Git 忽略），模板见
`tools_local/mimiclite_cluster.env.example`。

## 2. 2026-09-16 实测核实结果

以下每一项都是本次会话实际连接/执行后得到的结果，不是转述：

- **三端 SHA 一致**：`bash tools_local/mimiclite_cluster.sh verify-code` 输出
  ```
  local+github CODE_OK a0bbc02dbae8872cdd42334a3e6390d86b340390
  node0 CODE_OK a0bbc02dbae8872cdd42334a3e6390d86b340390
  node1 CODE_OK a0bbc02dbae8872cdd42334a3e6390d86b340390
  ```
- **两节点 GPU 空闲**：`nvidia-smi` 显示两台各 8 张 RTX 4090 D，显存占用 2 MiB、利用率 0%，
  当前没有 SONIC_MimicLite 的训练/NCCL 任务在跑。
- **两节点均缺 Python 环境**：`/root/SONIC_MimicLite/.venv/bin/python` 在 node0、node1 均
  不存在（`VENV_MISSING`）——这是启动 NCCL smoke 和训练前的第一个硬阻塞。
- **两节点均无数据**：`/root/SONIC_MimicLite_data` 在两台机器都不存在（`NO_DATA_DIR`），
  Robot/SMPL 训练数据尚未上传/同步。
- **两节点 Git LFS 已装**：`git-lfs/3.0.2` 均可用。
- **两节点 tracked 工作树干净**：`git status --short` 均无输出。
- **磁盘空间**：node0 `/` 2.0T 用 684G（余 1.3T，充裕）；node1 `/` 1008G 用 879G（**仅余
  约 88G**，上传数据/建环境前必须重新核对，避免装环境或传数据时撑爆）。
- **两节点各有大量与本项目无关的历史 tmux 会话**（例如 node0 的 `jump1/2/3`、
  `lys_n2*`、`mimic_1`；node1 的 `bumi-stage*`、`bumi_night*` 等），均不是
  `SONIC_MimicLite_<RUN_ID>_nodeN` 命名格式，**属于其他旧项目遗留，禁止误触/清理**。

## 3. 训练前仍未满足的前置条件（按顺序）

1. 在两台服务器 `/root/SONIC_MimicLite` 下创建 `.venv` 并安装训练依赖
   （`REMOTE_PYTHON=/root/SONIC_MimicLite/.venv/bin/python`，来自 `.local/mimiclite_cluster.env`）。
2. 把 Robot/SMPL 训练数据同步到两节点，落盘到 `.env` 中约定的
   `ROBOT_MOTION_DIR` / `SMPL_MOTION_DIR`（`/root/SONIC_MimicLite_data/{robot,smpl}_filtered`），
   并生成/核对两端一致的 SHA-256 manifest。
3. 需要用户明确接受 NVIDIA Omniverse EULA 后，才能把
   `.local/mimiclite_cluster.env` 中的 `OMNI_KIT_ACCEPT_EULA` 从 `NO` 改成 `YES`——
   在此之前不得运行 NCCL smoke 和训练 smoke（脚本会检查该开关）。
4. 满足以上三项后，依次执行低风险到正式任务：
   ```bash
   cd /home/yingchaomu/下载/SONIC_MimicLite
   bash tools_local/mimiclite_cluster.sh verify-code
   bash tools_local/mimiclite_cluster.sh launch-nccl   nccl_<YYYYMMDD_HHMMSS>
   bash tools_local/mimiclite_cluster.sh launch-smoke  smoke_<YYYYMMDD_HHMMSS>
   bash tools_local/mimiclite_cluster.sh launch-train  <RUN_ID>
   ```

## 4. 已完成、不需要重做的事

- GitHub 仓库已建立且为唯一代码真源；本地 `origin` 已指向
  `git@github.com:Mu-Yingchao/SONIC_MimicLite.git`。
- 两台服务器已通过 `bootstrap-direct` 直传初始化到 `/root/SONIC_MimicLite`，
  权属已修正为 `root:root`，此后不再需要直传通道，只走
  `本地 commit → GitHub push → 服务器 sync-code` 闭环。
- 两台服务器均已安装并初始化 Git LFS。
- 私钥已复制到 `.local/keys/`（Git 忽略），权限 600。
- 集群控制脚本、无密钥配置模板、NCCL smoke 脚本均已就绪：
  - [`tools_local/mimiclite_cluster.sh`](tools_local/mimiclite_cluster.sh)
  - [`tools_local/mimiclite_cluster.env.example`](tools_local/mimiclite_cluster.env.example)
  - [`tools_local/nccl_smoke.py`](tools_local/nccl_smoke.py)
- GitHub 曾对历史中的 `picoSetUp.mp4`（75.14 MB）给出大文件警告，但推送已成功，
  不是当前阻塞项；后续新增大文件应优先用 Git LFS 或排除到 `.gitignore`。

## 5. 关键命令速查

```bash
cd /home/yingchaomu/下载/SONIC_MimicLite
bash tools_local/mimiclite_cluster.sh status          # 网络/GPU/磁盘/tmux 会话总览
bash tools_local/mimiclite_cluster.sh verify-code      # 四端 SHA 一致性核验
bash tools_local/mimiclite_cluster.sh sync-code        # 服务器 fast-forward 到 GitHub main
bash tools_local/mimiclite_cluster.sh sync-code-bundle # GitHub 不可达时的增量 bundle 后备同步
bash tools_local/mimiclite_cluster.sh training-status <RUN_ID>
```

## 6. 更新规则

每次集群拓扑、阻塞项或核实结果发生变化时，直接更新本文件对应章节并注明日期，
不要另建同类快照文件。涉及具体模型/训练/部署代码改动的记录仍写入
`BUMI3_SONIC_修改记录.md`（`agent.md` 规定的闭环），本文件只管"集群控制与同步状态"这一层。
