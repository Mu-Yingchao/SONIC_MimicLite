# SONIC_MimicLite 定案方案（BUMI2）

本文件记录**已拍板、正在执行**的方案：结论、现状和下一步，不重复推演过程。

| 想看什么 | 去哪里 |
|---|---|
| 方案演进、候选思路对比、同事代码审计 | [sonic+mimiclite.md](sonic+mimiclite.md) |
| 每次改动的完整证据链（含踩坑与纠错） | [SONIC_MimicLite_修改记录.md](SONIC_MimicLite_修改记录.md) |
| 本地 play 命令、切换数据、看 viser | [docs/source/getting_started/play_guide.md](docs/source/getting_started/play_guide.md) |
| 三端同步与集群操作规范 | [docs/source/getting_started/codex_local_control_multi_node_training.md](docs/source/getting_started/codex_local_control_multi_node_training.md) |

---

## 0. 现状一览（2026-09-29）

| 项 | 状态 |
|---|---|
| 上层 SONIC（BUMI2） | 第一轮训练到 20716 步停；**第二轮加权微调运行中**（Noetix-9，tmux `sonic_bumi2_recon`） |
| 离线桥接 PKL → ONNX → 参考动作 | ✅ 纯本地，无需 Isaac Lab |
| 下层 MimicLite-BUMI2 接入 | ✅ `checkpoint_40000.pt`，本地 mjlab + viser 可视化 |
| **SMPL → SONIC → MimicLite（项目目标链路）** | ✅ 离线端到端验证，追踪成功率 0.839，与 robot 链路相同 |
| checkpoint 回归 | ✅ 每 2000 步自动评估 21 条测试动作，回退/停滞自动告警 |
| play 中参考动作漂移 | ⚠ 已定位（root 与重建关节不自洽），修复待做 |
| 实时 PICO 遥操 | ✗ 未打通，障碍在 MimicLite 侧接口（§7） |
| 真机 | ✗ 未验证 |

---

## 1. 目标机型：BUMI2

2026-09-17 拍板从 BUMI3 切到 BUMI2。原因：BUMI2 上的 MimicLite（A 同事）已训出能做
前后空翻、侧手翻的追踪策略 `BumiV2TrackBase`；BUMI3 上的 MimicLite 仍在攻坚 PPO 稳定性，
没有可借用的下层策略（审计细节见 sonic+mimiclite.md §9）。

BUMI2 与 BUMI3 同尺寸、关节/刚体命名和顺序一致，但执行器数值（力矩、速度、KP/KD、
armature）不同，不能混用。仓库已移除全部 BUMI3 代码与数据（2026-09-24）。

---

## 2. 架构

![架构图](media/sonic_mimiclite_bridge.png)

源文件 [media/sonic_mimiclite_bridge.svg](media/sonic_mimiclite_bridge.svg)。

```text
Robot PKL ─┐                                    ┌─ g1_kin 解码器 ──► 关节角（本项目的产品）
SMPL PKL ──┼─► Robot / SMPL encoder ─► 共享 token ┤
PICO（未通）┘                        2×32 FSQ    └─ g1_dyn 解码器 ──► 动作（PPO 训练必需，不部署）
                                                                │
                     关节角（近端交叉淡化拼接） + root 轨迹（取自源动作，不经解码器）
                                                                ▼
                     qpos npz / bumi_deploy_motion_v1 JSON（参考动作）
                                                                ▼
             MimicLite-BUMI2 Actor（冻结）：观测 639 = policy 399 + command 240
                                                                ▼
                                 21 维关节目标 + PD ─► BUMI2（mjlab / 真机）
```

### 2.1 上层就是完整的官方 SONIC 架构

实际训练配置里**两个解码器都有、都在训**：

| 解码器 | 输入 | 输出 | 训练信号 | 用途 |
|---|---|---|---|---|
| `g1_dyn` | token + 本体感知 | 21 维动作 | PPO | 训练必需，**不部署** |
| `g1_kin` | token | 未来 10 帧关节角/速度 + 根朝向差 | aux 重建 loss | **本项目的产品** |

> 更正：本文件旧版写的"不需要训练 `g1_dyn`"不准确。准确说法是**不需要部署** `g1_dyn`。

### 2.2 为什么 `g1_dyn` 也必须训——同训符合项目目标

1. **框架上离不开**：SONIC 训练是 PPO，必须有出动作的头去仿真里采 rollout。去掉它就没有
   RL 目标，得另写纯监督训练器（sonic+mimiclite.md §7 的"方案 B"）。
2. **`g1_kin` 的训练输入依赖它**：tokenizer 里的根朝向项是"参考相对**仿真中机器人当前**
   朝向"，机器人要由 `g1_dyn` 控制着站在那里；`g1_dyn` 越差，这部分输入越偏离真实分布。
3. **官方 SONIC 的核心设计**：token 经控制目标训练，编码的是"可执行的动作"而非单纯姿态。
4. **白送一个实时控制器**：`g1_dyn` 本身就是 BUMI2 上可直接部署的 SONIC 策略，天然支持
   实时输入，可作为 PICO 遥操的基线和备选（§7）。

唯一的偏差不在"训了 `g1_dyn`"，而在**权重没随定位调整**：`g1_kin` 是产品，却按官方正则项
的权重在训（§3.2），第二轮训练就是修这个。

### 2.3 两层之间的接口：root 是缺口

| | SONIC 上层 | MimicLite 下层 |
|---|---|---|
| 策略本体观测 | 重力、角速度、关节角/速度、历史动作——**无 root** | 同类五项，各 7 步历史——**无 root** |
| 参考侧 | encoder 输入只含根**朝向**差，**无位置**；`g1_kin` 不输出绝对位置 | command 里 **72/240 维是参考 root**：位置（相对参考自身 anchor）24 + 朝向（相对机器人本体）48 |

两边都是相对量，不需要机器人知道自己在世界的哪里，真机可行。但 SONIC 不产出 root 位置，
MimicLite 又要，所以 root 目前取自源动作 PKL 的真值轨迹——**与重建关节不是同一套**，
这是 §6 漂移问题的根源。

### 2.4 离线桥接（`tools_local/sonic_offline_bridge.py`）

- **纯本地 ONNX**：`g1_kin` 对本体感知 690 维完全不敏感（随机扰动下输出变化恒为 0），
  离线填零即可，不需要 Isaac Lab，也不需要仿真机器人。
- **近端交叉淡化拼接**：解码误差随窗口内偏移呈 U 形（偏移 0 帧 6.37°、5~10 帧 5.56°、
  45 帧 8.30°）。每 5 帧解码、只用 0.1~0.3 s 那段、相邻窗口交叉淡化：误差 -27%、
  位移不匹配 -28%、抖动持平。旧的整窗拼接保留为 `--stitch window` 仅供对照。
- 三个会静默出错的坑（代码注释里有详细说明）：PKL 的 `dof` 是 MuJoCo 顺序而 encoder 吃
  IsaacLab 顺序；关节速度用前向差分；SMPL 根朝向要右乘基准旋转的逆。

导出的 ONNX：`_g1.onnx` 1170→[10,48]、`_smpl.onnx` 1470→[10,48]、`_encoder.onnx` 1263→[64]。

---

## 3. 上层训练

配置：[sonic_bumi2.yaml](gear_sonic/config/exp/manager/universal_token/all_modes/sonic_bumi2.yaml)（第一轮）、
[sonic_bumi2_recon.yaml](gear_sonic/config/exp/manager/universal_token/all_modes/sonic_bumi2_recon.yaml)（第二轮）。
Noetix-9 单节点 8 卡、`num_envs=3072`、约 10 步/分钟。

### 3.1 第一轮（`sonic_bumi2_v2-20260920_155335`，0 → 20716 步）

回归实测（21 条测试动作、近端拼接、SMPL 链路关节角误差）：

| step | 2000 | 4000 | 6000 | 8000 | 10000 | 12000 | 14000 | 16000 | 18000 | 20000 |
|---|---|---|---|---|---|---|---|---|---|---|
| SMPL | 9.15° | 7.81° | 7.18° | 6.76° | 6.32° | 6.18° | 5.74° | 5.54° | 5.52° | **5.47°** |

**一直在改善**（累计 -40%），只是在放缓（最后 4000 步 -1.3%）。robot 链路同步（20000 步 5.47°）。

> 更正两次误判：
> - 旧版本文件写"`g1_recon` 第 500 轮走平、判定收敛、17618 轮停"——错。
> - 2026-09-29 我据训练侧 `g1_recon` 平在 0.55~0.60 报告"17000 轮没进步"，用户据此批准停训——也错。
>
> 两次都是被训练侧 loss 骗了：它 97% 是关节速度误差（§3.2），速度那部分确实没学会、一直平着，
> 把关节角的持续改善完全淹没了。这是回归管线的价值所在。

### 3.2 审计：`g1_recon` 被关节速度主导

step 17600 拆解（本地复算整体 MSE 0.5718，与训练侧 0.56~0.58 吻合）：

| 分量 | 目标 RMS | 误差 RMS | 占 `g1_recon` |
|---|---|---|---|
| 关节角 | 0.530 | 0.197 | **3.0%** |
| 关节速度 | 1.326 | 1.125 | **96.9%** |
| 根朝向 6D | 0.577 | 0.088 | 0.2% |

MimicLite 读关节角、不读关节速度——解码器的力气几乎全花在下游不用的量上。叠加 `g1_recon`
系数只有 0.01（官方把它当正则项的权重），没随"`g1_kin` 即产品"调整。

其余审计项（不需改）：`frame_mask` 与 `command_multi_future_nonflat` 的行布局错位是上游潜在
bug，但 BUMI2 未开可变帧数、mask 恒为空；`smpl_latent` 绝对值缓慢上升，但 SMPL 链路精度与
robot 持平；学习率由 PPO 的 KL 自适应、已压到下限附近，Adam 下影响有限。

### 3.3 第二轮（`sonic_bumi2_recon_v1-20260929_200625`，运行中）

- **只换 aux loss**：`G1ReconLossWeighted`，关节角 1.0 / 速度 0.01 / 朝向 1.0，系数 0.01→0.1。
  关节角元素梯度约为原来 23 倍、速度约 1/4，aux 总量级不变，不打破 PPO 平衡。Hydra 组合后
  与第一轮配置 diff 只有这 3 处。
- **热启动微调**：从 `model_step_020000.pt`（回归成绩已知 5.47°）`+resume=false` 只加载权重，
  优化器/学习率/计数器重新开始，写新目录。没选原地 resume（会带旧 loss 下的 Adam 动量，学习率
  卡在下限），也没选从 0 训（要 28 小时以上才能回到现有 PPO 水平）。
- 启动确认：加权 `g1_recon` 首步 0.085（随机初始化约 0.6），权重确已加载；704 步时 0.0525。
- **判据**：回归以第一轮最好成绩 5.47° 为基线。若约 4000 步后关节角误差不再下降、且离目标
  仍远，说明 token 空间在旧目标下已丢失精确关节角信息，改用新目标从 0 训。
- 注意：新 run 的 `loss/aux_g1_recon_avg` 是加权口径，不能与第一轮曲线比，看回归。

### 3.4 早期单窗口基准（仅作参考）

2026-09-21 用 `inspect_g1_recon`（单窗口、加热身 reset）测过：G1 官方 4.17°、BUMI3 10 万轮
3.56°、BUMI2 17618 步 3.73°。结论"BUMI2 重建水平与官方同级"成立，但**数值与 §3.1 不可比**：
那是单条动作单个窗口，§3.1 是 21 条动作全长、含高动态动作。对外引用以 §3.1 为准。

---

## 4. 算力与分工

| 节点 | 角色 |
|---|---|
| Noetix-9（8×4090D） | 上层 SONIC 训练 + ONNX 导出。环境 `/data0/sonic_mimiclite_env`（Isaac Sim 4.5 + IsaacLab 2.3.2） |
| 本地（1×4090） | 控制面 + 部署端：代码修改、MimicLite play（`mimiclite_bumi2/training/venv/mjlab`）、回归评估 |
| Noetix-0 | 当前不参与；访问 GitHub 需代理，未同步到最新 |

`num_envs=3072/卡`：4096 会 OOM，根因是 `ppo_trainer.py` 被注释掉的周期性显存清理，修复后
稳定（2026-09-19 修改记录）。导出 ONNX 可与训练并行，但只能与训练 rank0 共享 GPU0——Isaac Sim
不能用 `CUDA_VISIBLE_DEVICES` 挪卡（2026-09-29 修改记录）。

---

## 5. 数据

| 数据 | 位置 | 规模 |
|---|---|---|
| 训练集 Robot | Noetix-9 `/data0/bumi2_sonic_dataset_v1/built/robot_all` | 114,995 条 PKL |
| 训练集 SMPL（配对） | Noetix-9 `/data0/bumi2_sonic_dataset_v1/built/smpl_all` | 93,822 条 PKL |
| any4hdmi 原始 npz | Noetix-9 `/data0/bumi_v2_filtered` | 51 GB |
| **本地测试集** | `test_data/motions/{robot_pkl,smpl_pkl}` | 21 条配对，覆盖静态/行走/跑跳/表演/高动态 |
| 本地 play 参考 | `test_data/motions/any4hdmi-bumi-v2/motions/{sonic_robot,sonic_smpl}` | 各 21 条 qpos |

SMPL 直接复用已对齐筛选的 BUMI3 SMPL 语料（可跨机型复用），BUMI2 Robot npz 用
`gear_sonic/tools/prepare_bumi2_sonic_dataset.py` 转换并按文件名配对。

---

## 6. 下层 MimicLite-BUMI2 与本地验证

- 代码：`mimiclite_bumi2/training`（同事训练框架，play 用）、`mimiclite_bumi2/deploy`（ROS1 真机）。
- 策略：`test_data/policies/mimiclite/checkpoint_40000.pt`（同事另有 45000，暂用 40000）。
- 观测 639 = `policy` 399（纯本体历史）+ `command` 240（见 §2.3），与 checkpoint 权重第一层逐位吻合。
- 仿真与训练同为 mjlab（MuJoCo），**没有跨引擎 sim2sim gap**；sim2real 未验证。

**追踪结果**（同 6 条动作、8 环境）：robot 与 SMPL 两链路 `motion_timeout` 均为 0.839，
局部系身体位置误差平均 2.9 cm——**MimicLite 追姿势追得很好**。

**漂移问题**（play 里 ghost 与机器人越拉越远）：

| 指标 | 数值 |
|---|---|
| 8 步窗口（= MimicLite 观测窗口）root 位移 vs 关节隐含位移的不匹配 | 我们 0.030 m；GMR 原始数据 0.0001~0.013 m |
| 累积 root 位移偏差 | 约 1.8 m |
| 近端拼接后的改善 | 世界系偏差 -9%（p95 -26%），累积偏差仅 -4% |

根因：root 取自源动作，关节是重建的，两者要求的位移量不一致。GMR 能做到自洽，是因为它用
`mink` 把 root（free joint）和关节放在**同一个 IK 里联合求解**。提升关节精度只能缓解，消不掉。
我先前写的手工脚锁定反推（`tools_local/recover_root_from_joints.py`）实测负收益，保留作失败记录。

---

## 7. 已知限制

| 限制 | 原因 | 方向 |
|---|---|---|
| play 漂移 | root 与重建关节不自洽（§6） | 用 `mink` 以重建关节为目标反解 root |
| 实时 PICO 遥操 | MimicLite 数据集在环境构造时一次性加载；真机 `AcController.cpp` 只读 JSON 文件 | 改造两侧接口为滚动参考；或直接用 `g1_dyn`（官方 SONIC 实时路径）|
| 高动态动作误差大 | 高动态约为静态的 2 倍 | 第二轮训练观察分层误差 |
| 真机 | 未编译、未运行 ROS1 部署包 | 仿真指标达标后再做 |

---

## 8. 下一步

1. **盯第二轮训练**：回归每 2000 步出点，对比基线 5.47°；按 §3.3 判据决定继续、调系数或从 0 训。
2. **root 自洽**：用 `mink` 反解与重建关节自洽的 root，目标把 8 步窗口位移不匹配压到 GMR 量级，
   消除 play 漂移；在 play 里用追踪统计验证。
3. **`g1_dyn` 基线**：把第二轮的 `g1_dyn` 导出，在 sim2sim 里直接跑，作为"官方 SONIC 单网络"
   对照组，也为实时遥操评估备选路径。
4. 仿真指标达标后：真机部署（ROS1 + ONNX，需先编译同事部署包）。

---

## 9. 架构图说明

2026-09-29 按实测现状重画 [media/sonic_mimiclite_bridge.svg](media/sonic_mimiclite_bridge.svg)，
主要修正：

1. 旧图把解码器画成"新增桥接层——唯一要训练的部分"。实际它就是 SONIC 自己的 `g1_kin`，
   与编码器一起训练；新图补上了同训但不部署的 `g1_dyn`。
2. 旧图写解码器输出 root。实际 root 取自源动作、不经解码器，新图用单独的蓝色虚线标出，
   并标注由此导致的已知问题。
3. 补充离线桥接（纯 ONNX、近端拼接）、MimicLite 观测的真实构成（639 维）、回归评测和
   运行载体（本地 mjlab 已验证 / 真机未验证），PICO 标为未打通。
