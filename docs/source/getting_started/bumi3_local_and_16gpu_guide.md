# BUMI3 本地部署与双机 16 GPU 训练指南

本文只描述当前仓库的 BUMI3 原生 SONIC 路径。旧 `gear_sonic_deploy` 的 G1 C++
DDS/CSV 三终端流程不是当前 BUMI3 sim2sim 入口，不能混用按键和观测约定。

## 1. 代码一致性规则

Git commit 是唯一代码版本标识。算法、配置、资产或部署代码只能先在开发机修改、测试、
提交并推送，再让两台服务器快进到同一个 commit；禁止直接在服务器工作树修改源码。
数据、环境、日志和 checkpoint 放在仓库外，不提交 Git。每次启动前执行：

```bash
tools_local/bumi_cluster.sh verify-code
```

当前临时首次分发可用 Git bundle；长期唯一远端固定为
`git@github.com:Mu-Yingchao/sonic_bumi_full.git`。开发机和两台服务器的 `origin` 已指向
该地址，但必须先在 GitHub 创建同名空仓库才能首次 push。真实服务器地址放在被忽略的
`.local/sonic_bumi_cluster.env`，不得记录密码或私钥正文。

当前三份代码路径统一为：

- 本地：`/home/yingchaomu/下载/sonic_bumi_full`
- GPU14：`/data/ouqin/sonic_bumi_full`
- GPU15：`/data/ouqin/sonic_bumi_full`

两台服务器的 `/data` 都是本地 NVMe 文件系统，不是内存盘；GPU14、GPU15 分别约有
2.4 TiB、2.6 TiB 可用空间。旧 `/data/muyingchao/SONIC_BUMI` 和旧本地
`bumi_local_deploy_bundle` 都不是本训练任务的源码、数据或 Python 导入来源。

## 2. 本地 MuJoCo sim2sim

安装（已有 Isaac Lab 环境时可直接复用其 Python）：

```bash
cd /home/yingchaomu/下载/sonic_bumi_full
python -m venv .venv_sim
source .venv_sim/bin/activate
python -m pip install --upgrade pip
python -m pip install -e 'gear_sonic[sim]'
python gear_sonic/tools/validate_bumi3_sim2sim.py --skip-smoke
```

Robot Encoder（`1170 -> 21`）：

```bash
source .venv_sim/bin/activate
python gear_sonic/scripts/run_bumi3_sim2sim.py \
  --encoder robot \
  --policy /绝对路径/model_step_030000_g1.onnx \
  --motion /绝对路径/robot_motion.pkl
```

SMPL Encoder（`1470 -> 21`），推荐传入同名 Robot 动作用于一致的初始状态和红色影子：

```bash
python gear_sonic/scripts/run_bumi3_sim2sim.py \
  --encoder smpl \
  --policy /绝对路径/model_step_030000_smpl.onnx \
  --motion /绝对路径/smpl_motion.pkl \
  --robot-motion /绝对路径/robot_motion.pkl
```

窗口中只需：先单击窗口，按 `T` 播放，按 `P` 切换清单动作。没有 `]`、`9`，不需要
两个或三个终端，也不需要先把 PKL 转 CSV。白色是不透明的 MuJoCo policy 机器人，红色
半透明机器人是 Robot 参考影子。无窗口验收命令：

```bash
python gear_sonic/scripts/run_bumi3_sim2sim.py \
  --encoder robot --policy /绝对路径/model_g1.onnx \
  --motion /绝对路径/robot.pkl \
  --headless --no-real-time --duration 10
```

批量动作使用 `--dataset /绝对路径/dataset.json`；详细清单格式、首帧等待语义、SMPL
字段契约和已知限制见 `bumi3_sim2sim.md`。

当前仓库中可直接测试的本地资产（均位于被 Git 忽略的运行数据目录）是：

```text
data/bumi3_sim2sim_test/robot/wave_R_001__A428.pkl
data/bumi3_sim2sim_test/smpl/wave_R_001__A428.pkl
models/deployment/robot/model_step_030000.onnx
models/deployment/robot/model_step_050000.onnx
models/deployment/smpl/model_step_030000.onnx
models/deployment/smpl/model_step_050000.onnx
```

因此 Robot 与 SMPL 的 30000-step 测试可分别执行：

```bash
cd /home/yingchaomu/下载/sonic_bumi_full
source .venv_sim/bin/activate
python gear_sonic/scripts/run_bumi3_sim2sim.py \
  --encoder robot \
  --policy models/deployment/robot/model_step_030000.onnx \
  --motion data/bumi3_sim2sim_test/robot/wave_R_001__A428.pkl

python gear_sonic/scripts/run_bumi3_sim2sim.py \
  --encoder smpl \
  --policy models/deployment/smpl/model_step_030000.onnx \
  --motion data/bumi3_sim2sim_test/smpl/wave_R_001__A428.pkl \
  --robot-motion data/bumi3_sim2sim_test/robot/wave_R_001__A428.pkl
```

## 3. Isaac Lab play 与 ONNX 导出

`.pt` 的 play/eval 使用训练环境，必须把 checkpoint 配套配置中的服务器数据路径覆盖为
本机绝对路径：

```bash
python gear_sonic/eval_agent_trl.py \
  checkpoint=/绝对路径/model_step_030000.pt \
  ++num_envs=1 ++headless=false \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/绝对路径/robot目录 \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=/绝对路径/smpl目录
```

只导出联合 ONNX 时增加 `++headless=true ++export_onnx_only=true`。导出的
`*_g1.onnx` 用于 Robot reference，`*_smpl.onnx` 用于离线 SMPL 以及后续的
SMPL/PICO 5 点实时参考。不要交叉使用两种输入维度。

## 4. 双机 16 GPU 训练策略

每张 GPU 一个进程，2 台 × 8 GPU 共 16 个 Accelerate/DDP rank。两台机器没有共享盘，
因此必须各自保存同路径、同内容的数据与环境；world rank 0 位于 GPU14，只有它写正式
checkpoint。`num_envs` 是每张 GPU 的环境数。SONIC 论文使用 4096/GPU，因此
128 GPU 正式训练为 524,288 个并行环境；本项目正式配置也保持 4096/GPU：

- 正式 16 GPU 使用 `4096/GPU`，全局 `65536`，用更大的 PPO batch 获取论文所述的
  多 GPU 优化稳定性收益；
- `2048/GPU`（全局 `32768`）只作为与原 8 卡 × 4096 保持相同全局 batch 的可选
  强扩展速度对照，不能当成正式扩展实验；
- 先做 5 iteration/64 env 每卡的双机 smoke，再启动 100000 iteration 正式训练。

初始化本机配置：

```bash
mkdir -p .local
cp tools_local/bumi_cluster.env.example .local/sonic_bumi_cluster.env
# 编辑地址、SSH key、远端 Python、Robot/SMPL 数据目录；不要填写密码。
bash tools_local/bumi_cluster.sh status
bash tools_local/bumi_cluster.sh verify-code
bash tools_local/bumi_cluster.sh launch-nccl bumi3_nccl_smoke_YYYYMMDD_HHMMSS
```

环境、数据、NCCL 双机 collective 和 NVIDIA Isaac Sim EULA均验证后：

```bash
bash tools_local/bumi_cluster.sh launch-smoke bumi3_16gpu_smoke_YYYYMMDD_HHMMSS
# 检查两台 node0.log/node1.log、16 个 rank、PPO step、有限指标和 GPU 利用率后：
bash tools_local/bumi_cluster.sh launch-train bumi3_16gpu_scratch_100k_YYYYMMDD_HHMMSS
```

正式启动参数固定包含 `resume=false checkpoint=null auto_load_latest=false`，因此不会加载
旧策略。默认正式配置每卡 4096 env；若要做保持 32768 全局环境的速度对照，在本机
私有配置中另设 `TRAIN_ENVS_PER_GPU=2048`，并使用新的 run ID，不能覆盖正式实验。

日志跟踪（本地终端执行）与 TensorBoard：

```bash
# GPU14/world-rank 0：训练指标、checkpoint 写入端
ssh -i ~/.ssh/id_ed25519_sonic_bumi -p 22115 ouqin@117.161.121.54 \
  'tail -f /data/ouqin/runs/RUN_ID/node0.log'

# GPU15：非零 rank 的初始化和报错日志
ssh -i ~/.ssh/id_ed25519_sonic_bumi -p 22116 ouqin@117.161.121.54 \
  'tail -f /data/ouqin/runs/RUN_ID/node1.log'

# 建立隧道并在 GPU14 启动 TensorBoard；浏览器打开 http://127.0.0.1:6006
ssh -i ~/.ssh/id_ed25519_sonic_bumi -p 22115 \
  -L 6006:127.0.0.1:6006 ouqin@117.161.121.54 \
  '/data/ouqin/envs/sonic_bumi/bin/python -m tensorboard.main \
   --logdir /data/ouqin/runs/RUN_ID/tensorboard --host 127.0.0.1 --port 6006'
```

`last.pt` 每 50 step 更新一次，长期 `model_step_*.pt` 每 2000 step 保存一次；只有
world rank 0 写这些文件。

如果当前终端本来就在 GPU14 上，不要再次使用外网 SSH 地址或本地工作站私钥，直接执行：

```bash
tail -f /data/ouqin/runs/RUN_ID/node0.log
```

训练 checkpoint 与导出策略的存放/传输规则：

- GPU14/world-rank 0 保存：`/data/ouqin/runs/RUN_ID/last.pt` 和
  `model_step_XXXXXX.pt`；GPU15 不重复保存 checkpoint。
- ONNX 不在训练中自动导出。选定 checkpoint 后在 GPU14 运行
  `eval_agent_trl.py ... ++export_onnx_only=true`，结果写入
  `/data/ouqin/runs/RUN_ID/exported/`，包括 `*_g1.onnx` 和 `*_smpl.onnx`。
- 在本地开发机运行以下命令，把已导出的策略分别下载到 Robot/SMPL 部署目录：

```bash
cd /home/yingchaomu/下载/sonic_bumi_full
bash tools_local/fetch_bumi_onnx.sh RUN_ID STEP
```

checkpoint/ONNX 是大体积、可再生的实验产物，不进入普通 Git 历史；服务器 run 目录是训练
原件，本地 `models/deployment/{robot,smpl}` 是部署副本。重要里程碑应另做带校验和的备份或
对象存储归档，而不能只保留一份；“不提交 Git”不等于“不备份”。

## 5. 启动门槛

正式训练前必须同时满足：两台 commit 完全一致且 tracked worktree 干净；Python、PyTorch、
CUDA、Isaac Sim、Isaac Lab 和 `gear_sonic` 版本一致；Robot/SMPL 文件集合和内容哈希一致；
动作是一一配对的 50 FPS 当前训练集；单机 BUMI 环境 smoke 通过；跨机 NCCL all-reduce
通过。任一项不满足时，不用正式 16 卡任务来“试错”。

PICO 遥操是 SMPL encoder 的实时输入生产端，不是当前离线 sim2sim 的第三种 policy。
验收顺序应是：离线 Robot → 离线 SMPL（同名配对初始化）→ SMPL/PICO 5 点实时流 →
真机低风险分级测试。

## 6. 2026-09-11 首次建群记录

- 两台节点均为 8 × RTX 4090，Isaac Lab 固定提交
  `37ddf626`（2.3.2）。
- 环境固定为 Python 3.11、PyTorch 2.7.0+cu128、Isaac Sim 5.1.0.0、
  Isaac Lab 0.54.2、Accelerate 1.14.0；两端导入路径已确认指向 `/data/ouqin`。
- 当前候选训练副本为 97,660 对同名 Robot/SMPL；SMPL 源软链接已解引用。两端
  195,320 个 PKL 的 SHA-256 manifest 摘要一致。该一致性只证明复制无误，正式启动前
  仍需由当前 MotionLib 做坐标契约检查。独立全量字段审计已通过：所有配对均为 50 FPS、
  帧数一致、字段形状正确、无 NaN/Inf，合计 33,655,845 帧，单条 30～9007 帧。
- 互联为 `bond4`，RDMA HCA 为 `mlx5_bond_0:1`。首次自动选择网络的 16-rank NCCL
  all-reduce 已通过；64 MiB、20 次测试的慢端用时 0.254643 秒，测试口径吞吐
  4.909 GiB/s。显式固定 HCA 后再次通过，慢端 0.253931 秒、4.923 GiB/s，正式配置
  使用 `NCCL_SOCKET_IFNAME==bond4` 与 `NCCL_IB_HCA==mlx5_bond_0:1`。
- 不把历史 checkpoint 放入新 run；正式命令保持 `resume=false`、`checkpoint=null`、
  `auto_load_latest=false`。
- 2026-09-11，项目所有者明确接受 NVIDIA Omniverse EULA，并授权两台训练节点设置
  `OMNI_KIT_ACCEPT_EULA=YES`；运维脚本在未明确设置为 `YES` 时拒绝启动 Isaac 训练。
- Isaac Kit 启动锁按 Unix UID 隔离，避免共享服务器上其他用户遗留的全局 `/tmp` 锁
  阻止当前用户启动；同一节点的8个本地rank仍共用同一把锁并保持串行启动。
- Isaac Lab 日志临时目录和 BUMI URDF 生成的 USD 缓存分别按用户、节点/rank 隔离，避免
  共享服务器旧账号目录权限以及多 rank 并发转换冲突。GPU15 缺少的 `libGLU.so.1` 放在
  `/data/ouqin/lib` 并通过私有集群配置加入运行库路径，不修改系统目录。
- 单卡 16 env/2 iteration smoke 已通过；双机 16 GPU、64 env/GPU、5 iteration smoke
  已通过，共 122,880 timestep。正式任务
  `sonic_bumi3_16gpu_4096_scratch_100k_20260911_150534` 已从零启动：16 GPU、4096
  env/GPU、全局 65,536 env、24 rollout step；前 9 次更新无 OOM、NCCL 或非有限指标错误。
- 当前正式任务由 `setsid` 启动，Accelerate 主进程的父进程和 session 已脱离 SSH，因此
  本地终端关闭或公网 SSH 断开不会停止训练。后续新任务由运维脚本创建独立 `tmux` session；
  `status` 会同时显示 session 和训练进程。两节点内部训练网络若中断，NCCL 作业仍会失败，
  需要从最近 checkpoint 重新启动，tmux 不能恢复同一次 collective。

不启动 Isaac Sim 的全量配对字段审计：

```bash
/data/ouqin/envs/sonic_bumi/bin/python tools_local/audit_bumi_dataset.py \
  --robot-dir /data/ouqin/datasets/bumi3/train/robot \
  --smpl-dir /data/ouqin/datasets/bumi3/train/smpl \
  --expected-pairs 97660 --workers 16 \
  --report /data/ouqin/datasets/bumi3/audit.json
```
