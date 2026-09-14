# BUMI3 部署与 Play 操作手册

本文只适用于当前唯一项目 `/home/yingchaomu/下载/sonic_bumi_full`。旧目录
`bumi_local_deploy_bundle` 不参与以下任何流程。

## 1. 本地资产与环境

```bash
cd /home/yingchaomu/下载/sonic_bumi_full
test -x .venv_sim/bin/python
ls -lh models/deployment/robot/model_step_064000_robot.onnx
ls -lh models/deployment/smpl/model_step_064000_smpl.onnx
```

本机普通 Python 启动 MuJoCo 会落到 CPU 的 `llvmpipe`。所有带窗口的 MuJoCo
命令必须使用 `tools_local/*_nvidia.sh` 启动器。

当前 sim-to-sim 动力学在加载 XML 后会将全部 21 个驱动关节的
`armature` 统一覆盖为 `0.01`；浮动根的 6 个自由度保持 `0`。地面和机器人
碰撞使用 `condim=6`、`friction="1 0.05 0.01"`、elliptic 摩擦锥、
`impratio=10` 和 `solref="0.01 1"`。这些参数对 Robot、SMPL 和 PICO 三条
MuJoCo 路径同时生效，无需额外命令行参数。

## 2. 离线 Robot sim-to-sim

### 单个 Robot 动作

```bash
cd /home/yingchaomu/下载/sonic_bumi_full

tools_local/run_bumi3_sim2sim_nvidia.sh \
  --encoder robot \
  --policy models/deployment/robot/model_step_064000_robot.onnx \
  --motion data/bumi3_sim2sim_test/robot/wave_R_001__A428.pkl
```

### 24 条快速清单

```bash
tools_local/run_bumi3_sim2sim_nvidia.sh \
  --encoder robot \
  --policy models/deployment/robot/model_step_064000_robot.onnx \
  --dataset data/bumi3_sim2sim_test/original_split_5217_seen_by_current_run/dataset_smoke_24.json
```

把清单改成 `dataset_categories_172.json` 可覆盖 172 类；改成
`dataset_all_5217.json` 可遍历全部 5217 对。窗口按 `T` 播放，按 `P`
切换下一条。

## 3. 离线 SMPL sim-to-sim

### 单个配对 SMPL 动作

```bash
cd /home/yingchaomu/下载/sonic_bumi_full

tools_local/run_bumi3_sim2sim_nvidia.sh \
  --encoder smpl \
  --policy models/deployment/smpl/model_step_064000_smpl.onnx \
  --motion data/bumi3_sim2sim_test/smpl/wave_R_001__A428.pkl \
  --robot-motion data/bumi3_sim2sim_test/robot/wave_R_001__A428.pkl
```

Robot 配对文件只负责一致的初始状态和红色参考影子；1470 维 policy 输入的参考部分
来自 SMPL。

### 24 条快速清单

```bash
tools_local/run_bumi3_sim2sim_nvidia.sh \
  --encoder smpl \
  --policy models/deployment/smpl/model_step_064000_smpl.onnx \
  --dataset data/bumi3_sim2sim_test/original_split_5217_seen_by_current_run/dataset_smoke_24.json
```

## 4. Isaac Lab play

`play` 使用训练端 PyTorch checkpoint（`*.pt`）和 Isaac Lab，不使用 ONNX，
也不能在轻量 `.venv_sim` 中运行。当前本机还没有 Isaac Lab 环境和 checkpoint，
因此现在不能直接在本机打开 Isaac Sim viewer。

训练仍在使用 GPU14/GPU15 的全部 16 张卡；训练结束前不要在服务器额外启动 play
或导出任务。训练结束后，先把稳定 checkpoint 传回本机：

```bash
cd /home/yingchaomu/下载/sonic_bumi_full
mkdir -p models/checkpoints

scp -i ~/.ssh/id_ed25519_sonic_bumi -P 22115 \
  ouqin@117.161.121.54:/data/ouqin/runs/sonic_bumi3_16gpu_4096_scratch_100k_20260911_150534/model_step_064000.pt \
  models/checkpoints/
```

安装并激活与服务器一致的 Isaac Lab 2.3.2 环境后，不要激活 `.venv_sim`，执行
Robot Encoder play：

```bash
cd /home/yingchaomu/下载/sonic_bumi_full
source /绝对路径/isaaclab环境/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES

python gear_sonic/eval_agent_trl.py \
  checkpoint="$PWD/models/checkpoints/model_step_064000.pt" \
  ++headless=false ++num_envs=1 ++use_encoder=g1 \
  ++manager_env.observations.policy.enable_corruption=false \
  ++manager_env.observations.tokenizer.enable_corruption=false \
  ++manager_env.commands.motion.start_from_first_frame=true \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file="$PWD/data/bumi3_sim2sim_test/robot" \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file="$PWD/data/bumi3_sim2sim_test/smpl" \
  '++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=[wave_R_001__A428]'
```

SMPL Encoder play 只把 `++use_encoder=g1` 改成：

```bash
++use_encoder=smpl
```

关闭 Isaac Sim 或在终端按 `Ctrl+C` 退出。这里没有 MuJoCo 的 `T/P` 按键流程；
环境从动作首帧直接执行。若 checkpoint 步数更新，文件名必须在下载和运行命令中同时修改。

## 5. SMPL + PICO 五点遥操 sim-to-sim

这条链路不使用 planner/hybrid encoder，直接使用
`model_step_064000_smpl.onnx`。PICO 端将头、双手、双脚五点人体跟踪解算成
SMPL，并发布 10 个连续 50 Hz 帧；BUMI3 端以约 180 ms 缓冲延迟获得训练所需的
10 帧窗口，生成 780 维 SMPL token，与 690 维本体历史组成 1470 维输入。

### 一次性安装 PICO 环境

首次安装或重建 `.venv_teleop` 时执行：

```bash
cd /home/yingchaomu/下载/sonic_bumi_full
bash install_scripts/install_pico.sh
```

只运行 PICO 发布端、不需要在这个环境中安装 MuJoCo/Unitree 依赖时，可使用更快的配置：

```bash
SKIP_SIM_AND_UNITREE=1 bash install_scripts/install_pico.sh
```

安装后先验证 XRoboToolkit。输出 `XRT_IMPORT_OK` 才继续：

```bash
source .venv_teleop/bin/activate
python -c "import xrobotoolkit_sdk; print('XRT_IMPORT_OK')"
```

确认 PICO、两个手柄和两个脚踝 tracker 已完成校准，PICO 与 PC 位于同一网络，
XRoboToolkit PICO 应用已连接 PC service。

### 终端 1：PICO → SMPL → ZMQ

```bash
cd /home/yingchaomu/下载/sonic_bumi_full
source .venv_teleop/bin/activate

python gear_sonic/scripts/pico_manager_thread_server.py \
  --input-source xrt \
  --port 5556 \
  --target_fps 50 \
  --num_frames_to_send 10
```

本流程故意不加 `--manager`，因为 BUMI3 当前不使用官方 G1 planner 状态机；程序会
直接持续发布 `pose`。`torch.jit.script is deprecated` 是无害的 FutureWarning。发布端会先
启动本机 `/opt/apps/roboticsservice`；PICO 未连接或身体数据尚未就绪时应继续等待。看到
`ZMQ socket bound to port 5556` 和周期 FPS 后保持运行，再启动终端 2。

### 终端 2：SMPL ONNX → BUMI3 MuJoCo

```bash
cd /home/yingchaomu/下载/sonic_bumi_full

tools_local/run_bumi3_pico_sim2sim_nvidia.sh \
  --policy models/deployment/smpl/model_step_064000_smpl.onnx \
  --zmq-url tcp://127.0.0.1:5556 \
  --startup-timeout 300 \
  --stream-timeout 0.5
```

看到 `BUMI3_PICO_CONNECTED` 后 MuJoCo 窗口打开。先保持与机器人初始姿态尽量一致，
单击 MuJoCo 窗口并按 `T`，才会从冻结的首个窗口切入实时追踪。关闭窗口或在终端
按 `Ctrl+C` 退出。PICO 流超过 0.5 秒没有更新时，控制端会报 stale 并停止，而不是
无限复用旧姿态。

若终端 2 报 `no PICO pose received`，说明 5556 尚无有效 pose；先检查终端 1，不能
通过重复启动终端 2 修复。若终端 1 报 `XRoboToolkit SDK import failed`，重新运行上面的
安装命令及导入验证。`--startup-timeout 300` 只给连接和校准更多时间，不会掩盖
`--stream-timeout 0.5` 的运行中断流保护。

验收顺序：

1. 终端 1 稳定输出接近 50 FPS。
2. 终端 2 输出 `required_frames=10`、`policy_input_dim=1470`。
3. 按 `T` 前机器人保持初始参考；按 `T` 后缓慢移动手臂、下蹲和抬腿。
4. 先只做小幅动作；离线 SMPL 稳定通过前不要进行快速动作或真机测试。

当前实时 MuJoCo 窗口不显示离线 Robot 红色影子，因为实时输入没有 Robot 重定向
轨迹；白色机器人就是策略实际输出。
