# SONIC_MimicLite 定案方案（BUMI2）

本文件是当前**已拍板、正在执行**的方案文档，只记录结论和执行状态，不重复推演过程。
完整的方案演进、三份候选思路的对比评估、MimicLite-BUMI2/BUMI3 两份同事代码的解压
审计细节，都在 [sonic+mimiclite.md](sonic+mimiclite.md)，本文件不重复搬运，只在
结论变化时引用对应章节。集群控制闭环（GitHub 同步、服务器状态）见
[PROJECT_STATUS.md](PROJECT_STATUS.md)；本仓库的代码修改审计记录见
[BUMI3_SONIC_修改记录.md](BUMI3_SONIC_修改记录.md)（文件名是历史遗留，BUMI2 相关修改
也记在这份文件里，2026-09-17 的条目）。

> 与全局 `~/下载/CLAUDE.md` 描述的 SONIC_N3 项目无关，参见
> [PROJECT_STATUS.md](PROJECT_STATUS.md) 顶部说明。

## 1. 目标机型：BUMI2（从 BUMI3 切换而来）

2026-09-17 拍板：融合方案的落地目标机型从 BUMI3 切换为 **BUMI2**。

原因（详见 sonic+mimiclite.md §9）：解压核实了两位同事的 MimicLite 代码后发现，
BUMI2 上的 MimicLite（A 同事）已经训出效果好的追踪策略（`BumiV2TrackBase`，能做
前后空翻、侧手翻等高动态动作）；BUMI3 上的 MimicLite（B 同事）经代码审计确认是一个
同事本人仍在攻坚、尚未解决的 PPO 稳定性问题（actor std 裁剪、checkpoint 维度扩展
热启动、动作率惩罚等实质性工程改动仍在进行中），不具备可直接借用的下层策略。

BUMI2 与 BUMI3 同尺寸、换了电机，关节/刚体命名和排列顺序完全一致，但执行器数值
（力矩、速度、KP/KD、armature）不同，不能混用（详见 §4 和
`gear_sonic/envs/manager_env/robots/bumi2.py` 顶部注释）。

## 2. 架构：不变

桥接架构本身**不因目标机型切换而改变**，链路图见
[media/sonic_mimiclite_bridge.svg](media/sonic_mimiclite_bridge.svg) /
[media/sonic_mimiclite_bridge.png](media/sonic_mimiclite_bridge.png)（已更新为 BUMI2
版本，见 §7）：

```text
Robot(BUMI2) 参考 ─┐
SMPL/GENMO ─────────┤        共享 64 维 token（FSQ）
PICO 五点重建 ───────┘              │
                        ┌───────────┴───────────┐
              Robot encoder            SMPL encoder
                        └───────────┬───────────┘
                                     ▼
                    g1_kin 式参考动作 decoder（新训练）
                    token → 显式 BUMI2 参考动作
                                     ▼
                      当作合法 motion-lib 帧
                    走 MimicLite-BUMI2 的 ObsRef
                                     ▼
              冻结的 MimicLite-BUMI2 追踪 Actor（同事已验证）
                                     ▼
                        关节目标 + PD → BUMI2
```

- **SONIC 上层（Robot/SMPL encoder + FSQ token + g1_kin 式解码器）**：BUMI2 没有现成的
  SONIC checkpoint，需要**新训练**——这是本文件 §3 的核心工作，不是冻结复用。
- **MimicLite 下层（BUMI2 Actor）**：直接冻结复用同事已验证的 `BumiV2TrackBase` 策略。
- 两层之间的桥接 decoder 训练方法，以及"为什么解码到显式参考动作而不是策略私有输入"
  的论证，见 sonic+mimiclite.md §1、§8。

## 3. 训练方案：方案 A（复用 SONIC 流程，不改训练框架）——已执行完成，结论见下

结论（详见 sonic+mimiclite.md §7 末尾的算力讨论）：

- **不需要训练 `g1_dyn`**（重型 RL 动作解码器）——这个角色由冻结的 MimicLite-BUMI2
  Actor 承担。真正需要训练的只有 Robot/SMPL encoder + FSQ token + `g1_kin` 式重建
  解码器。
- **照抄 `sonic_bumi3.yaml` 的训练流程，不改一行训练框架代码**，只把机器人换成
  BUMI2、数据换成 BUMI2+SMPL/BVH，正常起 IsaacLab + PPO。
- **不追求 100k 轮训出完整策略**，只看 `g1_recon`/`g1_smpl_latent`/
  `reencoded_smpl_g1_latent` 这几个 aux loss 何时收敛就提前停。

**实测结果（2026-09-20~21，8 卡，`num_envs=3072`）**：

- 训练过程中发现并修复了一个共享代码 `torch_humanoid_batch.py` 的关节错位 bug（详见
  `BUMI3_SONIC_修改记录.md` 2026-09-20 条目）——第一版训练（约 44 小时、22000 轮）
  因为这个 bug 全部作废重训，第二版（修复后）才是有效结果。
- `g1_recon` 从第 500 轮左右就已经走平，一直稳定到第 17618 轮（停止训练时）都没有
  再明显下降——判定为已收敛，在第 17618 轮停止训练。全程约 30 GPU-hours（8 卡 ×
  约 3.8 小时实际训练时长，不含环境搭建和排错时间）。
- **实测重建误差**（关节角度，逐帧对比编码前后的差异）：这里有过一次重要的方法论
  纠错，如实记录——第一次测的时候只做了一次 `env.reset()`，命中了 IsaacLab
  一个"首次 reset 冷启动"的坑（同一个 checkpoint/同一条动作/同一个起始帧，只做
  一次 reset 会比"先热身 reset 一次、再测第二次"系统性偏高 2~4 倍，机制未查证到
  IsaacLab 源码层面的确切原因，但现象在多次独立测试里稳定复现）。修正后（每次
  测量前先加一次热身 reset）的真实结果：

  | | mean(°) |
  |---|---|
  | G1 官方发布 checkpoint（成熟/发布级别） | 4.17 |
  | 我们的 BUMI3 训练 3 万轮（训练目标 10 万轮的中间点） | 9.25 |
  | 我们的 BUMI3 训练满 10 万轮（真正训练终点） | 3.56 |
  | 我们的 BUMI2 第 17618 轮（本次训练终点） | 3.73 |

  **修正后结论与之前完全相反**：BUMI2 和 BUMI3-10万轮几乎打平，而且两者都略好于
  G1 官方版本——不存在"G1 明显更强、BUMI 系列追不上"的差距。BUMI3 从 3 万轮到
  10 万轮的提升依然明显（9.25°→3.56°），说明训练量在起作用；BUMI2 只训 17618
  轮就已经摸到 BUMI3 练满全程的水平，这次训练的重建质量是可信的、达标的。
- 如果提前停之后成本仍然太高，可以考虑写独立的纯监督训练脚本（跳过 IsaacLab PPO rollout，
  详见 sonic+mimiclite.md §7 的"方案 B"）——**这次没有用到**，方案 A 直接跑通了。

配置文件：[gear_sonic/config/exp/manager/universal_token/all_modes/sonic_bumi2.yaml](gear_sonic/config/exp/manager/universal_token/all_modes/sonic_bumi2.yaml)。
最终 checkpoint：Noetix-9 `/data0/bumi2_sonic_runs/TRL_BUMI2_Track/sonic_bumi2_v2-20260920_155335/last.pt`
（第 17618 轮）。

## 4. 算力与集群

- **只用 Noetix-9 单节点 8 卡**，不做跨机 16 卡联训——已验证足够，没有用到跨机。
  Noetix-0 留给后续同步复现 A 同事 MimicLite-BUMI2 基线（不同框架、不占用 SONIC
  训练资源）；目前 Noetix-0 上只装了 `uv`，`active-adaptation` 代码同步过去了但
  `uv sync` 还没跑，环境未真正建起来（半成品状态，见 §8）。
- 算力量级：**实测**用 `num_envs=3072/卡`（不是原计划的 4096——第一次尝试 4096 时
  OOM，排查后发现瓶颈是 `ppo_trainer.py` 一处被注释掉的周期性显存清理代码，修复后
  `num_envs=3072` 稳定跑了 30+ 小时零 OOM，详见 `BUMI3_SONIC_修改记录.md`
  2026-09-19 条目）。8 卡跑 17618 轮实际耗时约 30 小时（含两次 OOM 排查、一次 bug
  修复重训的总耗时，不是纯训练时间）。
- Noetix-9 磁盘：`/data0`（2TB）用于存放训练数据，训练结束后 BUMI2 数据集
  （`/data0/bumi2_sonic_dataset_v1/`）+ 训练 run 目录共占用约几十 GB，仍有 1.5TB+
  余量。
- 两台服务器的 SONIC 训练用 Python/Isaac Lab 环境已装好：Noetix-9
  `/data0/sonic_mimiclite_env`（`uv` + Python 3.10.12 + torch 2.7.0+cu128 +
  isaacsim 4.5.0 + IsaacLab v2.3.2，详见 `BUMI3_SONIC_修改记录.md` 2026-09-19 条目）。

## 5. 数据——已完成

| 数据集 | 来源 | 规模 | 状态（2026-09-21） |
|---|---|---|---|
| BUMI2 Robot 动作（npz） | `112.65.216.193:/data0/yc/bumi_v2_data/bumi_v2_filtered` | 51GB，114,995 文件 | ✅ 已推送 Noetix-9:/data0，已转换为 SONIC 契约 |
| 人体 BVH 动作 | `yuanjie-server(muyingchao):/data/muyingchao/soma_uniform` | 277GB，142,220 文件 | ✅ 已推送 Noetix-9:/data0（最终未直接使用，见下） |
| BUMI3 已对齐筛选 SMPL PKL | `muyingchao@RTX4090-gpu-014:/data/ouqin/datasets/bumi3/train/smpl` | 22GB，97,660 文件 | ✅ 已推送，直接复用（跳过重新对齐 BVH） |

**实际采用的数据方案**：BVH 传了但没有重新跑 BVH→SMPL 对齐流程——直接复用已经对齐
筛选好的 BUMI3 SMPL PKL 语料（按 `sonic_motion_dataset_alignment_playbook.md` §0
记录的"SMPL 数据可跨机型 symlink 复用"先例）。BUMI2 Robot npz 用
`gear_sonic/tools/prepare_bumi2_sonic_dataset.py`（新写，独立于 BUMI3 脚本）转换成
SONIC 目标契约，按文件名和 SMPL PKL 配对。**最终构建结果**：
`robot=114995, paired=93822, robot_only=21173`，输出在 Noetix-9
`/data0/bumi2_sonic_dataset_v1/built/{robot_all,smpl_all}`，`validate` 全量核对通过。
完整过程见 `BUMI3_SONIC_修改记录.md` 2026-09-19 条目。

## 6. 代码迁移进度（`feature/bumi2-sonic-migration` 分支）——已完成并实测通过

完整改动清单和验证结果见 `BUMI3_SONIC_修改记录.md` 2026-09-17、2026-09-19、
2026-09-20 三个条目，这里只列状态：

**已完成并在真实 IsaacLab 环境中验证**：
- BUMI2 资产迁移：URDF/MJCF/22 个 mesh 文件，来自同事 `BM2-V2.0` 权威定义。
  `bumi2.xml` 后来补了一个缺失的 `<actuator>` 段（`torch_humanoid_batch.py` 解析
  MJCF 需要）。
- `gear_sonic/envs/manager_env/robots/bumi2.py`：关节/刚体顺序映射、`BUMI2_CFG`
  执行器参数、`BUMI2_ACTION_SCALE`——**均已通过 21 关节逐个数值核对**（与同事真实
  部署 JSON 字节级匹配）。
- 四处注册：`robots/__init__.py`、`modular_tracking_env_cfg.py`、`order_converter.py`
  （新增 `Bumi2Converter`）、`mdp/commands.py`（下肢索引断言）。
- 训练入口 `sonic_bumi2.yaml`：已实测跑通 8 卡训练至收敛（第 17618 轮停止）。
- Noetix-9 Isaac Lab 训练环境搭建完成（`uv` + Python 3.10 + isaacsim 4.5.0 +
  IsaacLab v2.3.2）。
- **修复了一个共享代码 bug**（`torch_humanoid_batch.py` 的 `body_to_joint` 构造，
  详见 `BUMI3_SONIC_修改记录.md` 2026-09-20 条目）——BUMI2 是第一个真正触发这个隐藏
  bug 的机型，已验证对 BUMI3/G1 零回归。
- 修复了一个训练显存泄漏 bug（`ppo_trainer.py` 被注释掉的周期性 empty_cache，详见
  2026-09-19 条目）。

**仍未完成——这是当前阶段的核心工作，见 §8**：
- `g1_kin` 式桥接 decoder 的输出格式验证/适配：已确认 `g1_kin` **不重建绝对根轨迹**
  （只重建关节角度/速度 + 相对于当前仿真机器人姿态的朝向差值，根轨迹要从驱动源直接
  拿），已调研清楚 MimicLite-BUMI2 的 `command`（240 维）/`policy`（399 维）输入契约
  和 `bumi_deploy_motion_v1` 部署 JSON 格式，**但实际的转换代码还没写**。
- 未接入同事已训好的 MimicLite-BUMI2 checkpoint（`checkpoint_40000.pt`，还在压缩包里）。
- BUMI2 的 sim2sim 验证环境未搭好：Noetix-0 上只同步了 `active-adaptation` 代码、
  装了 `uv`，`uv sync` 还没跑，是半成品状态。

## 7. 架构图更新说明

`media/sonic_mimiclite_bridge.svg`/`.png` 已从最初的 BUMI3 版本改为 BUMI2 版本，两处
关键修改：

1. 原图"SONIC 上层"标注"原样冻结"——这是 BUMI3 语境下的说法（当时已有训好的 BUMI3
   SONIC checkpoint 可以直接冻结）。BUMI2 语境下这个前提不成立，SONIC 上层需要**新
   训练**，图上已改成"新训练（复用 SONIC-BUMI3 框架与经验）"，避免误导后续读者以为
   这一层不需要投入算力。
2. "MimicLite 下层"和"机器人本体"两处标注从泛指改为明确写 **BUMI2**，避免和 BUMI3
   混淆。

## 8. 下一步（按顺序）——§1~5 已完成，当前从 §6 开始

1. ~~数据传输~~ ✅
2. ~~Noetix-9 Isaac Lab 环境~~ ✅
3. ~~BUMI2 数据准备脚本~~ ✅
4. ~~1-env smoke~~ ✅
5. ~~8 卡训练至收敛~~ ✅（第 17618 轮，详见 §3）

**当前阶段（按顺序）**：

6. **从训练 checkpoint 隔离出推理用的 Robot encoder + FSQ token + `g1_kin` decoder**。
   已有诊断用的临时代码（`train_agent_trl.py` 里 `inspect_g1_recon` 分支，配置驱动、
   默认关闭），需要整理成正式的、可复用的独立推理模块，不依赖完整 PPO 训练环境
   （目前每次调用仍需起一个真实 IsaacLab env 来正确构造 `UniversalTokenModule`，
   这一步本身也值得评估是否能简化）。
7. **写离线桥接转换脚本**：驱动源（SMPL 动作片段或 BUMI2 Robot 动作）自带的绝对
   root 轨迹（root_pos_w/root_quat_w，直接复用，不经过 decoder）+ `g1_kin` decoder
   解码出的关节角度序列（token 往返重建）→ 拼成 `bumi_deploy_motion_v1` 格式的 JSON
   （`metadata.joint_names` 必须严格等于 `bumi_ac.yaml` 里声明的 21 个关节名/顺序，
   `fps=50.0`，`root_quat_w`/`body_quat_w` 为 wxyz）。这是"整个方案最终能否成立的
   关键验证点"这句话里真正要写的代码，之前一直没写。
8. **接入同事已训好的 MimicLite-BUMI2 checkpoint**：从 `MimicLite_bumi2.tar.gz` 解出
   `checkpoint_40000.pt`，用 `scripts/tools/export_checkpoint_onnx.py` 跑一次 ONNX
   导出（当前压缩包里没有对应 `checkpoint_40000` 的现成 onnx 产物，需要自己导出）。
9. **搭建 sim2sim 验证环境**，两条候选路径（已调研但都没实际搭完，需要选一条走完）：
   - **复用同事 mjlab 框架**（`active-adaptation`，已同步到 Noetix-0，`uv sync`
     未跑）：最贴近真实训练时的物理引擎，但要接入我们自己的驱动源需要写一个替代
     `any4hdmi` 数据加载接口的适配层。
   - **复用同事的 ROS 部署包**（`MimicLite_bumi2_deploy.zip`，`legged_rl` ROS1
     catkin 工作区 + `AcController.cpp`，读取 `bumi_deploy_motion_v1` JSON 文件）：
     不需要写 any4hdmi 适配层（直接读 JSON），但需要装 ROS1/catkin/drake/onnxruntime
     （C++ 源码编译），且原生跑在 Gazebo 而不是 mjlab。
10. 跑通 sim2sim，验证 MimicLite-BUMI2 策略能不能按照我们桥接出来的参考动作实际
    追踪；如果追不上，回头判断是 §6 重建质量不够（修正测量方法后实测约 3.7° 平均
    关节角度误差，已经接近 BUMI3 练满 10 万轮和 G1 官方版本的水平，详见 §3），
    还是桥接格式转换本身有问题。
