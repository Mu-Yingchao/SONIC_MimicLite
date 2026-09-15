# SONIC_MimicLite：本地 VSCode/Codex 控制双机 16 卡训练与部署

本文将 `sonic_bumi_full` 的工作方式落地到 `SONIC_MimicLite`。本地开发机是代码控制
中枢、训练运维入口和部署测试端；Noetix-0 和 Noetix-9 只承担训练计算。

文档不保存服务器密码、私钥正文、访问令牌或其他凭据。

## 1. 架构与职责

```text
本地 VSCode/Codex（控制面 + 部署端）
  ├─ 修改、测试、commit、push
  ├─ SSH 发起集群检查、NCCL smoke 和 tmux 训练
  ├─ SSH 隧道查看 TensorBoard
  └─ 下载 ONNX，在本地 MuJoCo/真机部署
                  │
                  ▼
GitHub main（代码唯一真源）
          │                    │
          ▼                    ▼
训练节点 0 / world rank 0     训练节点 1 / 其他 ranks
  ├─ 8 GPU                    ├─ 8 GPU
  ├─ 日志、checkpoint、ONNX    ├─ 非主 rank 日志
  └──────── NCCL/RDMA 梯度同步 ─┘
```

三者不能混淆：

- GitHub 只同步可审查、可复现的代码、配置、脚本和小型测试文件。
- 数据集、环境、日志、checkpoint、ONNX 和缓存通过专门的复制/备份流程管理。
- 多机训练的梯度流量在服务器内网直接传输，不经过本地开发机或 GitHub。
- 训练进入 `tmux` 后，本地关机、VSCode 关闭或公网 SSH 断开不会终止训练。

## 2. 不可破坏的项目原则

1. Git commit SHA 是唯一代码版本标识，GitHub `main` 是唯一代码真源。
2. 只在本地开发机修改 tracked 文件；禁止直接编辑服务器工作树。
3. 修改前先 `git fetch/pull --ff-only`，修改后测试、commit、push，再让全部服务器
   fast-forward 到同一 SHA。
4. 启动任务前必须验证所有训练节点 SHA 相同、tracked worktree 干净。
5. 每个训练任务使用唯一 `RUN_ID`，不得覆盖其他实验目录。
6. 正式从零训练必须显式关闭 resume、旧 checkpoint 和自动续训。
7. 训练必须由 `tmux`、systemd 或调度器托管，不能依赖一个长期保持的 SSH 前台会话。
8. 密钥和机器地址放在被 Git 忽略的本机配置；密码和私钥正文永不写入仓库。
9. 大型产物不进普通 Git，但必须另行备份；“不提交 Git”不等于“只保留一份”。
10. 每次交付都给出：commit SHA、测试结果、训练 RUN_ID、checkpoint step 和产物校验和。

## 3. SONIC_MimicLite 的真实映射

项目名称在三端统一为 `SONIC_MimicLite`：

```text
GitHub:    git@github.com:Mu-Yingchao/SONIC_MimicLite.git (本地 push)
           https://github.com/Mu-Yingchao/SONIC_MimicLite.git (服务器只读 pull)
本地:      /home/yingchaomu/下载/SONIC_MimicLite
Noetix-0: /root/SONIC_MimicLite  (118.196.95.17 / 172.31.0.32)
Noetix-9: /root/SONIC_MimicLite  (14.103.42.170 / 172.31.0.2)
Run:       /root/SONIC_MimicLite_runs/<RUN_ID>
```

已实测两台服务器各有 8 张 RTX 4090 D，每张 GPU 一个 DDP rank，总计 16 ranks。
world rank 0 位于 Noetix-0，负责正式日志、TensorBoard event 和 checkpoint。跨机网络使用
`eth0`，rendezvous 地址为 `172.31.0.32`。Noetix-9 根盘当前仅余约 93 GB，上传数据和
创建环境前必须重新检查空间。

关键文件：

```text
tools_local/mimiclite_cluster.sh            集群统一入口
tools_local/mimiclite_cluster.env.example   无密钥配置模板
.local/mimiclite_cluster.env                真实本机配置，Git 忽略
.local/keys/noetix0.pem                     Noetix-0 密钥的 600 权限副本
.local/keys/noetix9.pem                     Noetix-9 密钥的 600 权限副本
tools_local/nccl_smoke.py                    跨机通信验收
docs/source/user_guide/training.md           SONIC 训练入口与参数说明
gear_sonic_deploy/deploy.sh                  本地部署入口
```

原始密钥位于 `/home/yingchaomu/下载/Noetix-2-7.pem` 和
`/home/yingchaomu/下载/noetix-8.pem`。配置只引用 `.local/keys/` 中的受限副本，不改动或
提交原始 PEM。

## 4. 新项目初始化模板

### 4.1 建立唯一仓库

在 GitHub 新建与项目同名的空仓库。本地项目设置唯一 `origin`：

```bash
git init
git branch -M main
git remote add origin git@github.com:<OWNER>/<PROJECT>.git
git add <明确需要纳入版本管理的文件>
git commit -m "chore: initialize project"
git push -u origin main
```

正常情况下服务器只 clone 同一个仓库，不从旧项目复制一份再继续改：

```bash
git clone git@github.com:Mu-Yingchao/SONIC_MimicLite.git /root/SONIC_MimicLite
```

本项目首次初始化明确采用本地直传：先把干净的 `main` 推送到 GitHub，再用下面命令把
同一工作树和 Git 元数据上传到两台空服务器目录。该命令若发现目标已存在会立即拒绝，
不会覆盖服务器内容；上传后只将该新目录的归属修正为 `root:root`：

```bash
git remote set-url origin git@github.com:Mu-Yingchao/SONIC_MimicLite.git
git push -u origin main
bash tools_local/mimiclite_cluster.sh bootstrap-direct
bash tools_local/mimiclite_cluster.sh verify-code
```

首次直传完成后，直接上传通道即退出日常流程。以后的所有 tracked 改动都必须走
`本地 commit -> GitHub push -> 服务器 sync-code`。

### 4.2 Git 应包含和排除的内容

应提交：源码、训练/部署配置、环境锁文件、安装脚本、集群脚本、文档、小型确定性测试
fixture、数据 manifest 及校验工具。

应忽略：

```gitignore
.local/
.venv*/
data/
models/
runs/
outputs/
logs/
checkpoints/
wandb/
tensorboard/
*.pt
*.pth
*.onnx
*.ckpt
*.engine
__pycache__/
```

如果模型本身是必须随代码发布且体积可控的产品资产，应显式采用 Git LFS、Release 或对象
存储，而不是悄悄取消忽略。训练原始 checkpoint 通常不适合进入源码仓库。

### 4.3 机器私有配置

仓库提交不含秘密的配置模板，真实配置和密钥副本放在 `.local/`：

```bash
mkdir -p .local/keys
cp tools_local/mimiclite_cluster.env.example .local/mimiclite_cluster.env
install -m 600 /home/yingchaomu/下载/Noetix-2-7.pem .local/keys/noetix0.pem
install -m 600 /home/yingchaomu/下载/noetix-8.pem .local/keys/noetix9.pem
chmod 600 .local/mimiclite_cluster.env
```

配置至少包含：每个节点 SSH endpoint/port、私钥路径、服务器仓库和 Python 路径、
RUN_ROOT、数据集路径、节点内网 `MASTER_ADDR`、rendezvous 端口、NCCL 网卡/HCA，以及
项目要求的许可证接受开关。只记录私钥路径，不记录私钥内容或密码。

## 5. 每次代码修改的标准闭环

### 5.1 修改前

```bash
cd <LOCAL_REPO>
git status --short
git fetch origin
git rev-list --left-right --count HEAD...origin/main
git pull --ff-only
```

若工作树有不属于当前任务的修改，不覆盖、不 reset，先判断所有权和冲突。若本地与远端
分叉，先处理分叉，不能把 `--force` 当作日常同步方法。

### 5.2 修改、验证、推送

```bash
# 编辑后运行与改动风险相称的测试
git diff --check
git status --short
git add <明确文件列表>
git commit -m "<type>: <具体修改>"
git push origin main
```

不得使用 `git add .` 无检查地混入模型、数据或用户的无关修改。

### 5.3 两台服务器快进同步

在每台服务器执行：

```bash
cd <REMOTE_REPO>
git status --short
git pull --ff-only
git rev-parse HEAD
```

公网 GitHub 暂时不可达时，可以从本地生成增量 `git bundle`，复制到服务器后 fetch；
仍然必须先确认服务器 tracked worktree 干净、HEAD 是预期旧 SHA，再做 `merge --ff-only`。
bundle 是传输替代方案，不是新的代码真源，GitHub 最终仍应包含该提交。

当前项目最终验收：

```bash
cd /home/yingchaomu/下载/SONIC_MimicLite
bash tools_local/mimiclite_cluster.sh verify-code
```

只有本地、GitHub、Noetix-0、Noetix-9 都解析到同一完整 SHA，代码同步才完成。

## 6. 数据与环境一致性

Git SHA 相同并不代表实验可复现。正式训练前还要检查：

- Python、PyTorch、CUDA、驱动、Isaac Sim/Isaac Lab 和项目包版本。
- 两个节点的数据目录、文件数量、相对路径和内容哈希。
- Robot/SMPL 等配对数据的 key、帧率、帧数、shape、dtype、NaN/Inf。
- 资产文件及其运行时转换缓存是否来自同一版本。
- 每台机器可用磁盘、共享内存和临时目录权限。

推荐生成按相对路径排序的 SHA-256 manifest，并比较 manifest 自身摘要。只比较文件数量或
目录大小不足以证明两份数据相同。数据复制应使用 `rsync --partial --append-verify`、对象
存储或专用数据版本系统；不要提交数万条 PKL 到 Git。

## 7. 双机训练启动门槛

从低风险到正式任务依次执行：

1. 单节点、单 GPU、小环境 smoke。
2. 双节点 NCCL all-reduce smoke，确认 16 ranks、网卡、RDMA HCA 和端口。
3. 双节点小环境训练 smoke，检查有限 loss/reward、日志和 checkpoint。
4. 正式 16 GPU 训练。

当前项目命令：

```bash
cd /home/yingchaomu/下载/SONIC_MimicLite
bash tools_local/mimiclite_cluster.sh verify-code
bash tools_local/mimiclite_cluster.sh launch-nccl nccl_<YYYYMMDD_HHMMSS>
bash tools_local/mimiclite_cluster.sh launch-smoke smoke_<YYYYMMDD_HHMMSS>
bash tools_local/mimiclite_cluster.sh launch-train <RUN_ID>
```

集群脚本先启动 node rank 1，使其等待 rendezvous，再启动 rank 0。训练核心形式为：

```bash
python -m accelerate.commands.launch \
  --multi_gpu \
  --num_machines=<NODES> \
  --num_processes=<TOTAL_GPUS> \
  --machine_rank=<NODE_RANK> \
  --main_process_ip=<MASTER_ADDR> \
  --main_process_port=<MASTER_PORT> \
  <TRAIN_ENTRY_AND_ARGS>
```

两台 8 GPU 服务器与一台物理 16 GPU 服务器的主要区别是前者需要处理跨机网络、RDMA、
rendezvous、两份环境/数据和节点故障；算法层面仍是一个 16-rank DDP world。

## 8. tmux、断网与运行版本

每个节点都用独立且可预测的 tmux session：

```text
<project>_<RUN_ID>_node0
<project>_<RUN_ID>_node1
```

脚本必须把 stdin 重定向到 `/dev/null`，日志写到 `RUN_ROOT/RUN_ID/nodeN.log`。断开本地
SSH 只会停止查看，不会停止 tmux 中的进程。

需要区分两种版本事实：

- 仓库当前 HEAD：本地和服务器日常保持一致。
- 某次训练的启动 SHA：任务启动时实际加载的代码版本，应写入 run metadata。

正在运行的 Python 不会因为服务器仓库后来 `git pull` 就自动切换全部已导入代码，但动态
读取的配置/资产可能受影响。更严格的新项目应为每个 RUN_ID 建立固定 SHA 的只读
`git worktree` 或容器镜像，并让训练从该快照启动；控制仓库仍可继续同步新提交。未经明确
授权，不重启正在运行的正式任务来应用新代码。

## 9. 训练监控

### 9.1 一条命令检查全部节点

```bash
bash tools_local/mimiclite_cluster.sh training-status <RUN_ID>
```

健康标准：每台节点显示预期本地 rank 数；所有 GPU 有合理显存占用和动态利用率；rank 0
iteration/timestep 持续增长；没有 traceback、OOM 或 NCCL error。非 rank-0 节点日志长期
停在数据加载末尾可能是正常现象，不能只凭“日志没刷新”认定该节点没有训练。

### 9.2 文本日志

```bash
ssh <NODE0> 'tail -f <RUN_ROOT>/<RUN_ID>/node0.log'
```

`Ctrl+C` 只退出 `tail`。不要对训练 PID 发送信号。

### 9.3 TensorBoard

TensorBoard 在 rank-0 节点监听 loopback，使用 SSH 本地端口转发：

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L <LOCAL_PORT>:127.0.0.1:6006 <NODE0>
```

浏览器打开 `http://127.0.0.1:<LOCAL_PORT>`。若端口占用，先用 `ss -ltnp` 和
`curl -I` 判断现有隧道是否可用，再选择新端口；不要反复启动失效隧道。

## 10. checkpoint、导出物与本地部署

建议职责固定如下：

```text
rank 0 RUN_ROOT/RUN_ID/       checkpoint 和原始 TensorBoard
rank 0 RUN_ROOT/RUN_ID/exported/   导出的 ONNX/TensorRT 等
本地 models/deployment/<mode>/     部署副本
对象存储/备份盘                   重要里程碑和 manifest
```

只从完整、稳定、带 step 编号的 checkpoint 导出，不读取正在被覆盖的 `last.pt`。导出应在
训练完成后进行；若必须训练中导出，需要确认显存和任务隔离，不得让正式训练 OOM。

当前仓库保留了 BUMI 专用的 ONNX 下载脚本，它不应用于 MimicLite 产物。MimicLite
的导出文件名和观测配置确定后，应新增专用下载脚本。手工下载时仍必须先写 `.part`、
检查非空后原子改名并输出 SHA-256。

模型由 `.gitignore` 排除，因此不会因一次普通 push 被上传 GitHub。部署入口为
`gear_sonic_deploy/deploy.sh`，应先在本地 MuJoCo 验证，再进入真机。

## 11. 故障处理边界

- SSH 断开但 tmux/ranks/GPU 正常：无需重启训练。
- 单个 rank 退出：整个同步 DDP world 通常无法继续，保存证据后按项目恢复策略处理。
- 服务器 worktree dirty：停止同步，先识别修改来源，禁止 `reset --hard` 擦除未知工作。
- 两端 SHA 不同：禁止启动正式训练。
- NCCL smoke 失败：先查内网地址、端口、防火墙、网卡和 HCA，不能用正式训练试错。
- 数据 manifest 不同：重新增量同步并复验，不能假设缺少文件会自动跳过。
- checkpoint/ONNX 未进 Git：这是预期行为；检查 run 目录、备份和下载校验和。
- 本地部署异常：先固定 policy SHA、motion、代码 SHA 和环境，再区分策略问题与部署问题。

## 12. 交给 Codex 的新项目启动说明模板

可把下面内容连同新项目路径发给 Codex：

```text
本地 VSCode 项目是唯一控制中枢和部署端，GitHub main 是代码唯一真源，多台 GPU
服务器只作为训练节点。请先审计仓库、环境、网络、数据和已有修改，不要使用任何旧项目
路径或环境。

建立以下闭环：
1. 所有 tracked 代码只在本地修改、测试、commit、push。
2. 所有服务器只 fast-forward 到 GitHub 同一 SHA；每次启动前验证 SHA 和 clean 状态。
3. 服务器地址与密钥路径放进被 Git 忽略的 .local 配置，禁止提交秘密。
4. 数据、环境、日志、checkpoint、ONNX 不进普通 Git；分别做 manifest、同步和备份。
5. 先完成单卡 smoke、跨机 NCCL smoke、全 ranks 小训练，再开始正式训练。
6. 正式训练必须在 tmux/调度器中运行，每个 RUN_ID 唯一，记录启动 SHA 和完整命令。
7. rank 0 保存日志/checkpoint/TensorBoard；从稳定 checkpoint 导出策略并带 SHA-256
   下载到本地部署目录。
8. 未经我授权，不停止、重启或覆盖正式训练，不删除未知文件，不直接修改服务器源码。
9. 每次修改后同步本地、GitHub、所有服务器，并报告完整 SHA、测试和运行状态。

请先给出并实现：.gitignore、无秘密 env.example、集群管理脚本、NCCL smoke、数据审计、
训练状态检查、tmux 启动、TensorBoard 隧道说明、策略导出/下载脚本和部署文档。
```

这份模板定义的是工作流和安全边界。新项目的 GPU 数量、每卡环境数、训练入口、数据格式、
NCCL 网卡和导出方法必须经过该项目自身验证，不能从 `sonic_bumi_full` 机械照搬。
