# SONIC_MimicLite 交接指南

（2026-09-30）

## 1. 项目是什么

在 BUMI2 机器人上做"任意动作输入 → 机器人跟踪执行"。整条链路分两层：

- **上层 SONIC**：负责把动作翻译成机器人的参考轨迹。
  - 输入：SMPL 人体动作或机器人动作 PKL。
  - 编码器把输入压成 64 维 token，`g1_kin` 解码器再把 token 还原成机器人参考轨迹。
  - 这一层在 Noetix-9 上训练。
- **下层 MimicLite-BUMI2**：负责在物理仿真或真机上把参考轨迹跟出来。
  - 用同事训好的 `checkpoint_40000.pt`，冻结不训。

最终目标是离线 SMPL 跟踪，之后做 PICO 实时遥操。

架构图：`media/sonic_mimiclite_bridge.png`，源文件是同目录的 `.svg`。

## 2. 当前状态

| 环节 | 状态 |
|---|---|
| 上层训练 | 第二轮进行中（`sonic_bumi2_recon_v1`）。目前最好是第 10000 步：SMPL 链路重建误差 4.09°，第一轮最好是 5.47° |
| 离线全链路 | 已通。SMPL/Robot PKL → SONIC → mink 后处理 → MimicLite，在本地 mjlab 里跑 21 条测试动作，零失败 |
| 端到端误差 | 机器人与原始动作差 5.1°。喂原始真值时是 3.5°，差距全部来自 SONIC 重建误差 |
| PICO 实时遥操 | 未做。方案见 `sonic_mimiclite_new.md` §7.1 |
| 真机 | 未做 |

看 play 时，ghost 和机器人会慢慢拉开约 0.25 m。**这是正常现象**：MimicLite 不追世界坐标，喂原始真值也一样，官方配置同样把它当截断而不是失败。判断跟踪好坏要看姿态，不看两者隔多远。

## 3. 机器与分工

| 机器 | 作用 | 代码路径 |
|---|---|---|
| 本地（1×4090） | 改代码、控制训练、play、回归评估 | `/home/yingchaomu/下载/SONIC_MimicLite` |
| Noetix-9（8×4090D，`ssh Noetix-9`） | **只训练上层 SONIC** | `/root/SONIC_MimicLite` |
| GitHub | 中转 | `github.com/Mu-Yingchao/SONIC_MimicLite`（main） |

**硬规则：**
- 本地、GitHub、Noetix-9 三处代码始终保持一致。流程是：本地改 → push → 服务器 `git pull --ff-only`。
- 不在服务器上直接改代码。
- play 只在本地跑。

## 4. 关键路径（Noetix-9）

| 内容 | 路径 |
|---|---|
| Python 环境 | `/data0/sonic_mimiclite_env` |
| 训练数据 Robot | `/data0/bumi2_sonic_dataset_v1/built/robot_all`（114,995 条） |
| 训练数据 SMPL | `/data0/bumi2_sonic_dataset_v1/built/smpl_all`（93,822 条，与 Robot 配对） |
| 原始 any4hdmi npz | `/data0/bumi_v2_filtered` |
| 第二轮输出 | `/data0/bumi2_sonic_runs/TRL_BUMI2_Track/sonic_bumi2_recon_v1-20260929_200625/` |
| 第一轮输出（基线） | `/data0/bumi2_sonic_runs/TRL_BUMI2_Track/sonic_bumi2_v2-20260920_155335/` |

训练启动命令（第二轮，从第一轮第 20000 步的权重热启动）：

```bash
cd /root/SONIC_MimicLite
OMNI_KIT_ACCEPT_EULA=YES TMPDIR=/data0/.tmp_isaaclab_bumi2 OMP_NUM_THREADS=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/data0/sonic_mimiclite_env/bin/python -m accelerate.commands.launch --num_processes=8 \
  gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_bumi2_recon \
  +resume=false checkpoint=/data0/bumi2_sonic_runs/TRL_BUMI2_Track/sonic_bumi2_v2-20260920_155335/model_step_020000.pt \
  auto_load_latest=false headless=True num_envs=3072 base_dir=/data0/bumi2_sonic_runs experiment_name=sonic_bumi2_recon_v1 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/data0/bumi2_sonic_dataset_v1/built/robot_all \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=/data0/bumi2_sonic_dataset_v1/built/smpl_all \
  2>&1 | tee /data0/bumi2_sonic_runs/recon_v1_launch.log
```

`num_envs` 最多到 3072，设成 4096 会 OOM。

## 5. 关键路径（本地）

| 内容 | 路径 |
|---|---|
| MimicLite 训练框架（play 用，同事代码） | `mimiclite_bumi2/training/` |
| MimicLite 真机部署（ROS1，未用过） | `mimiclite_bumi2/deploy/` |
| 下层策略 | `test_data/policies/mimiclite/checkpoint_40000.pt` |
| 上层 ONNX（当前用的） | `test_data/policies/sonic/recon_v1_step_010000_{g1,smpl}.onnx` |
| 回归自动拉回的各步 ONNX | `test_data/policies/sonic/regression/<run>/` |
| 测试动作（21 条配对） | `test_data/motions/{robot_pkl,smpl_pkl}/` |
| play 参考动作 | `test_data/motions/any4hdmi-bumi-v2/motions/{sonic_smpl_mink,sonic_smpl,sonic_robot,original_qpos}/` |
| 回归结果 | `test_data/regression/<run>/summary.csv` |

另外要建两个 ASCII 软链（Hydra 解析不了中文路径）：`~/sonic_aa` 指向 `mimiclite_bumi2/training`，`~/sonic_deploy` 指向 `test_data`。

**不在 Git 里、需要单独拿的东西**（`*.pt`、`*.onnx` 都被 gitignore 了）：

| 缺什么 | 从哪里拿 |
|---|---|
| `checkpoint_40000.pt` | 找我拷，或者从同事服务器 `ssh -p 50019 user@112.65.216.193` 拿 |
| SONIC ONNX | Noetix-9 各 run 目录下的 `exported/`，或者跑一次回归循环自动拉回 |
| Noetix-9 SSH 私钥 | 找我要，放到 `.local/keys/noetix9.pem`（回归脚本从这里读）；`~/.ssh/config` 里也要配 `Host Noetix-9` |
| mjlab 环境 | 按 play_guide.md 里"环境重建"一节装 |

## 6. 日常操作

| 要做什么 | 怎么做 |
|---|---|
| **看效果（play）** | 按 `docs/source/getting_started/play_guide.md` 执行，浏览器打开 http://127.0.0.1:8080，右侧 Mimic 面板可切换动作 |
| **换上层 checkpoint** | `bash tools_local/build_play_set.sh <g1.onnx> <smpl.onnx>`，约 1 分钟，**完成后必须重启 play** |
| **看训练趋势** | 本地回归循环常驻运行：每 30 分钟检查新 checkpoint，自动导出、评估、出报告，结果在 `test_data/regression/<run>/summary.csv`。出现 `ALERT` 文件表示在退步或停滞 |
| **手动跑一次回归** | `bash tools_local/sonic_regression_loop.sh --run <run名> --baseline sonic_bumi2_v2-20260920_155335` |
| **看训练日志** | `ssh Noetix-9 tail -f /data0/bumi2_sonic_runs/recon_v1_launch.log` |

回归循环的启动命令：

```bash
bash tools_local/sonic_regression_loop.sh --loop 1800 --run sonic_bumi2_recon_v1-20260929_200625 --baseline sonic_bumi2_v2-20260920_155335
```

## 7. 核心代码

| 文件 | 作用 |
|---|---|
| `tools_local/sonic_offline_bridge.py` | **桥接主程序**。PKL → SONIC ONNX → qpos npz；解码后做近端交叉淡化拼接，默认跑 mink |
| `tools_local/mink_root_solve.py` | 用 mink 反解 root，消除参考动作的脚底打滑（借鉴 GMR） |
| `tools_local/build_play_set.sh` | 用指定 checkpoint 批量重建 play 数据 |
| `tools_local/sonic_recon_regression.py` | 回归评估：`evaluate` 算重建误差，`report` 出报告和趋势判断 |
| `tools_local/sonic_regression_loop.sh` | 常驻回归循环：服务器导出 ONNX → 拉回本地评估 |
| `gear_sonic/trl/losses/token_losses.py` | `G1ReconLossWeighted`，第二轮的重建损失，按分量加权 |
| `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_bumi2_recon.yaml` | 第二轮实验配置 |
| `gear_sonic/tools/prepare_bumi2_sonic_dataset.py` | 把 any4hdmi npz 转成 SONIC 训练数据 |

## 8. 文档地图

**现行文档，按这个顺序读：**

| 文档 | 用途 |
|---|---|
| `HANDOVER.md`（本文） | 入口 |
| `sonic_mimiclite_new.md` | **定案方案**：架构、root 问题的结论、两轮训练记录、对照实验数据、已知限制、遥操方案、下一步。最重要的一篇 |
| `media/sonic_mimiclite_bridge.png` | 架构图，对应上面文档 §9 |
| `docs/source/getting_started/play_guide.md` | 本地 play 操作手册：命令、切换动作和链路、判读指标、环境重建、故障排查 |
| `docs/source/getting_started/bumi2_sonic_dataset.md` | 上层训练集的来源、对齐与配对规则、构建和验证命令、与 MimicLite 训练集的差异 |
| `docs/source/getting_started/codex_local_control_multi_node_training.md` | 本地控制服务器的工作规范：同步流程、训练监控、checkpoint 管理 |
| `SONIC_MimicLite_修改记录.md` | 全部修改的流水账，含验证证据和**被推翻的错误结论**。查"某处为什么这样改"时按日期搜 |

**背景和参考**，不用通读：

| 文档 | 内容 |
|---|---|
| `sonic+mimiclite.md` | 早期方案讨论（两层怎么结合），决策过程；结论已并入 `sonic_mimiclite_new.md` |
| `sonic_motion_dataset_alignment_playbook.md` | 准备 SONIC 训练数据（BVH 对齐、筛选）的经验，换新机型时有用 |

**过时文档，属于 BUMI3 时期，BUMI3 已放弃，不要按它们操作**：
- `deploy.md`
- `PROJECT_STATUS.md`
- `agent.md`
- `docs/source/getting_started/bumi3_sim2sim.md`

## 9. 容易踩的坑

- **误差数字怎么比**
  - 第二轮的 `aux_g1_recon` loss 是加权口径，不能和第一轮的曲线比。训练效果**只看回归结果**。
  - 早期单窗口测出的 3.7° 这类数字，和现在 21 条全长回归的数字不可比。
- **Isaac Sim 与 GPU**：不能用 `CUDA_VISIBLE_DEVICES` 挪卡，否则 segfault。服务器上导出 ONNX 只能和训练的 rank0 共用 GPU0。
- **root 平移**：SONIC 的 token 里本来就没有 root 平移信息，root 只能取自源动作，再用 mink 修正。这是结构决定的，训练再久也解决不了。
- **play 的数据加载**：只在启动时加载一次，换了数据必须重启。所有路径参数必须写全，`~` 不会展开。
- **进程管理**：`pkill`/`pgrep -f` 会匹配到自己，要用 `[s]cripts/play.py` 这种写法，或者直接按 PID 杀。

## 10. 下一步

1. **盯第二轮训练**：看回归曲线。停滞时考虑调大 `g1_smpl_latent` 系数；出了更好的 checkpoint 就换进 play。
2. **遥操第一步**（纯本地）：桥接改成因果模式，扫一遍前瞻延迟，量化精度损失。详见 `sonic_mimiclite_new.md` §7.1。
3. **`g1_dyn` 对照**：把同训的动作解码器导出，直接在仿真里跑，作为兜底路径。
4. 仿真指标达标后上真机：先编译 `mimiclite_bumi2/deploy` 的 ROS1 包。
