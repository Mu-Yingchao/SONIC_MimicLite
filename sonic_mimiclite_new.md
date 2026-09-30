# SONIC_MimicLite 定案方案（BUMI2）

本文件记录**已拍板、正在执行**的方案：结论、现状和下一步，不重复推演过程。

| 想看什么 | 去哪里 |
|---|---|
| 方案演进、候选思路对比、同事代码审计 | [sonic+mimiclite.md](sonic+mimiclite.md) |
| 每次改动的完整证据链（含踩坑与纠错） | [SONIC_MimicLite_修改记录.md](SONIC_MimicLite_修改记录.md) |
| 本地 play 命令、切换数据、看 viser | [docs/source/getting_started/play_guide.md](docs/source/getting_started/play_guide.md) |
| 三端同步与集群操作规范 | [docs/source/getting_started/codex_local_control_multi_node_training.md](docs/source/getting_started/codex_local_control_multi_node_training.md) |
| 上层训练集的来源、对齐、构建 | [docs/source/getting_started/bumi2_sonic_dataset.md](docs/source/getting_started/bumi2_sonic_dataset.md) |
| 新人入口（路径、日常操作、文档地图） | [HANDOVER.md](HANDOVER.md) |
| 全部代码改动清单 | 本文 §10 |

---

## 0. 现状一览（2026-09-30）

| 项 | 状态 |
|---|---|
| 上层 SONIC（BUMI2） | 第一轮训练到 20716 步停；**第二轮加权微调运行中**，目前最好 step 10000，SMPL 4.09°（§3.3） |
| 离线桥接 PKL → ONNX → 参考动作 | ✅ 纯本地，无需 Isaac Lab |
| 下层 MimicLite-BUMI2 接入 | ✅ `checkpoint_40000.pt`，本地 mjlab + viser 可视化 |
| **SMPL → SONIC → MimicLite（项目目标链路）** | ✅ 离线端到端零失败；机器人最终动作 vs 原始动作：第二轮 step 8000 + mink **5.13°**（第一轮 6.3°，喂真值 3.49°） |
| 本地 play 数据 | 已换为第二轮 step 10000 生成（`tools_local/build_play_set.sh`，来源见各目录 `SOURCE.txt`） |
| checkpoint 回归 | ✅ 每 2000 步自动评估 21 条测试动作，回退/停滞自动告警（§3.4） |
| 参考动作脚底打滑 | ✅ mink 后处理（root 与重建关节自洽），打滑 -52~81% |
| play 中 ghost 与机器人的世界系漂移 | ℹ 约 0.25 m，**喂真值也一样**——MimicLite 不追世界坐标，非 SONIC 所致 |
| SMPL 遥操 | ✅ 仿真端打通：ZMQ 实时流 → 在线桥接 → MimicLite，延迟约 200 ms，零失败（§7.1）；✗ 未接 PICO、未上真机 |
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
        关节角（近端交叉淡化拼接） + root（mink 以源动作支撑脚为目标反解，与关节自洽）
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
MimicLite 又要。直接取源动作的 root 会与重建关节不是同一套（脚在参考里打滑），所以桥接用
mink 反解一个与重建关节自洽的 root（§2.5）。

**为什么不让 SONIC 直接输出 root**：两个编码器的输入里都没有根平移（robot 只有关节角/速度 +
根朝向差；SMPL 只有相对骨盆的局部关键点 + 根朝向，PKL 里的 `transl` 从没喂给编码器），token
不含这个信息，解码器无从重建。要改就得改编码器输入和 `g1_kin` 输出、从 0 重训、偏离官方
SONIC；而且即使改了，root 和关节仍是两个独立回归输出，**没有任何约束保证脚不打滑**。只有
运动学约束能保证自洽——这正是 GMR 的做法。

### 2.4 离线桥接（`tools_local/sonic_offline_bridge.py`）

- **纯本地 ONNX**：`g1_kin` 对本体感知 690 维完全不敏感（随机扰动下输出变化恒为 0），
  离线填零即可，不需要 Isaac Lab，也不需要仿真机器人。
- **近端交叉淡化拼接**：解码误差随窗口内偏移呈 U 形（偏移 0 帧 6.37°、5~10 帧 5.56°、
  45 帧 8.30°）。每 5 帧解码、只用 0.1~0.3 s 那段、相邻窗口交叉淡化：误差 -27%、
  位移不匹配 -28%、抖动持平。旧的整窗拼接保留为 `--stitch window` 仅供对照。
- 三个会静默出错的坑（代码注释里有详细说明）：PKL 的 `dof` 是 MuJoCo 顺序而 encoder 吃
  IsaacLab 顺序；关节速度用前向差分；SMPL 根朝向要右乘基准旋转的逆。

导出的 ONNX：`_g1.onnx` 1170→[10,48]、`_smpl.onnx` 1470→[10,48]、`_encoder.onnx` 1263→[64]。

### 2.5 mink 后处理（`tools_local/mink_root_solve.py`，桥接默认开启）

借鉴 GMR（github.com/YanjieZe/GMR）：它用 mink 把 free joint（root）和全部关节放进同一个微分
IK 联合求解，自洽是联合优化的副产品。我们的版本：上半身冻结为 SONIC 重建值，腿部带正则
（拉回重建值）并受关节限位约束，root 自由；按源动作的接触状态加权——支撑脚强约束到源动作
的脚位，摆动脚弱约束，root 弱跟随源动作路线。

| 4 条代表动作 | 拼接 root | mink（只动 root） | **mink（root+腿）** | GMR 真值 |
|---|---|---|---|---|
| 参考脚底打滑 m/s | 0.14~0.48 | 0.09~0.27 | **0.03~0.22** | 0.001~0.19 |
| 腿部关节 vs 真值 | 2.2~6.8° | 不变 | **改善 0.15~0.5°** | 0 |

只动 root 不够：root 只有 6 个自由度，双脚支撑时要满足两只脚 12 个约束，而重建腿部本身有
误差、两脚相对位置就是错的。放开腿部后平均只修正 0.48°（最大 0.76°），且修正方向朝真值。
每帧 < 1.1 ms，可实时。纯 SMPL 输入（无配对 robot 动作）时，目标应换成缩放后的 SMPL 关键点，
即 GMR 原做法，属后续工作。

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
- 改动的文件：
  - `gear_sonic/trl/losses/token_losses.py`：新增 `G1ReconLossWeighted`；
  - `gear_sonic/config/aux_losses/terms/g1_recon_weighted.yaml`：分量权重；
  - `gear_sonic/config/aux_losses/universal_token/g1_recon_weighted_and_smpl_latent.yaml`：系数 0.1；
  - `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_bumi2_recon.yaml`：继承 `sonic_bumi2`，只替换 aux loss。
  - 启动命令见 [HANDOVER.md](HANDOVER.md) §4。

**回归结果**（21 条，关节角平均误差）：

| step | 2000 | 4000 | 6000 | 8000 | 10000 |
|---|---|---|---|---|---|
| SMPL | 4.71° | 4.54° | 4.28° | 4.22° | **4.09°** |
| robot | 4.40° | 4.05° | 3.97° | 3.94° | **3.89°** |

第一轮最好成绩 5.47°，第二轮开局即低于它，step 10000 比它好 25.3%，趋势仍在下降、未停滞。play 目前用 step 10000。

step 8000 + mink 的误差构成（`ref_diag`，重建 vs 原始动作，4.12°）：
- **分部位**：腿 3.08°、手臂 5.97°、腰 1.81°。
- **系统偏差**：占 27.6%。
- **幅度**：只有原始的 0.891，有向均值收缩的倾向。
- **时序**：滞后 22 ms，抖动是原始的 1.45 倍，物理违例可忽略。

试过留一法做幅度校准，误差反而变差 2.7%。说明这种收缩是逐动作不同的，没法事后统一校正，只能靠继续训练。

### 3.4 checkpoint 回归管线

训练侧的 loss 两次误导过判断（§3.1），所以训练效果只看回归。

| 文件 | 作用 |
|---|---|
| `tools_local/sonic_recon_regression.py` | `evaluate`：21 条动作 × robot/SMPL 两条链路，近端拼接，按类别统计关节角误差；`report`：出趋势、对比基线，回退或停滞时返回非零 |
| `tools_local/sonic_regression_loop.sh` | 常驻循环：发现新 checkpoint → 在 Noetix-9 GPU0 导出 ONNX（先确认空闲显存 ≥ 8 GB）→ 拉回本地评估 → 出报告 |

产物：
- 结果：`test_data/regression/<run>/{summary.csv,per_motion.csv,train_scalars.csv}`，出问题时还有 `ALERT`。
- ONNX：`test_data/policies/sonic/regression/<run>/`。

两轮训练的 step 都从 0 计数，所以结果和 ONNX 都按 run 分目录存放，互不覆盖。

```bash
bash tools_local/sonic_regression_loop.sh --loop 1800 \
  --run sonic_bumi2_recon_v1-20260929_200625 --baseline sonic_bumi2_v2-20260920_155335
```

### 3.5 早期单窗口基准（仅作参考）

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
| 本地 play 参考 | `test_data/motions/any4hdmi-bumi-v2/motions/{sonic_smpl_mink,sonic_smpl,sonic_robot,original_qpos}` | 各 21 条 qpos；SONIC 三组由 `tools_local/build_play_set.sh` 一键重建，来源 checkpoint 见各目录 `SOURCE.txt` |

SMPL 直接复用已对齐筛选的 BUMI3 SMPL 语料（可跨机型复用），BUMI2 Robot npz 用
`gear_sonic/tools/prepare_bumi2_sonic_dataset.py` 转换并按文件名配对。完整来历、配对规则、
构建和验证命令见 [bumi2_sonic_dataset.md](docs/source/getting_started/bumi2_sonic_dataset.md)。

2026-09-30 核实出与 MimicLite 训练集的两处差异：
- SONIC 用了 score1+score2，MimicLite 只用 score1；
- 同事的平地筛选没有同步过来，训练集里仍有 788 条楼梯、斜坡、障碍类动作。

当前这轮不受影响，下一版数据集应剔除这 788 条。

---

## 6. 下层 MimicLite-BUMI2 与本地验证

- 代码：`mimiclite_bumi2/training`（同事训练框架，play 用）、`mimiclite_bumi2/deploy`（ROS1 真机）。
- 策略：`test_data/policies/mimiclite/checkpoint_40000.pt`（同事另有 45000，暂用 40000）。
- 观测 639 = `policy` 399（纯本体历史）+ `command` 240（见 §2.3），与 checkpoint 权重第一层逐位吻合。
- 仿真与训练同为 mjlab（MuJoCo），**没有跨引擎 sim2sim gap**；sim2real 未验证。

**三组对照**（2026-09-29，同 21 条动作、8 环境、seed 42、各 136 秒；**关键是加了真值组**）：

| 喂给 MimicLite 的参考 | 机器人 vs 原始动作 | 机器人 vs 参考 | 局部身体误差 | ghost 距离（p95） | 失败/结束 |
|---|---|---|---|---|---|
| **真值**（不经 SONIC） | **3.49°** | 3.48° | 2.24 cm | **0.248 m**（0.85） | 0/42 |
| SONIC，拼接 root | 6.34° | 3.29° | 2.36 cm | 0.254 m（0.83） | 0/42 |
| SONIC + mink | 6.27° | 3.21° | 2.25 cm | 0.287 m（0.95） | 0/42 |
| 第二轮 step 8000 + mink（2026-09-30 补测） | **5.13°** | — | — | 0.254 m | 0 失败 |

"真值"组就是官方输入：它由 robot PKL 生成，与同事 any4hdmi 的 GMR npz 逐位一致（差 1.8e-7）。
表中前三组用的是第一轮 step 17600。

结论：

1. **MimicLite 追踪得很好，与参考来源无关**：对参考的误差都约 3.3°、局部误差约 2.3 cm、零失败。
2. **最终动作与原始动作的差距（6.3° vs 3.5°）完全来自 SONIC 重建误差**——要改进的是上层训练。
3. **play 里 ghost 与机器人越拉越远不是 SONIC 造成的**：喂真值也平均相距 0.25 m。MimicLite
   观测只含相对量（参考 root 相对参考自身 anchor），没有"你已落后多少"的反馈，只追姿态和位移
   节奏、不追世界坐标，偏差累积。这与"真机不需要知道绝对位置"是同一个设计取舍，对真机无害；
   只是仿真里按世界坐标画的 ghost 会显得在漂。要看追踪质量，看"机器人 vs 参考"与局部误差，不看 ghost 距离。
   **官方也是这样定性的**（2026-09-30 查同事代码）：
   - `root_pos_error`（世界系偏离 > 0.4 m）配置为 `is_timeout: true`，按截断处理而不是失败；
   - `scripts/eval.py` 把它和 `motion_timeout` 一起算成 success；
   - 官方 README 的 play 命令也关掉了它。

   所以继续训练 SONIC 消除不了这部分漂移。
4. mink 改善的是**参考动作本身的物理合理性**（ghost 不再打滑），端到端略好（6.34→6.27°），
   不减少世界系漂移。

> 更正：此前我把 ghost 漂移归因于 root 与重建关节不自洽，并据此提出 mink 方案——但从没测过
> 真值对照组。补测后该归因不成立。之前的"16% 追踪失败"是旧的整窗拼接参考下测的，
> 换成近端拼接后三组均零失败，该数字作废。手工脚锁定工具（`recover_root_from_joints.py`）
> 实测负收益，保留作失败记录，不再使用。

---

## 7. 已知限制

| 限制 | 原因 | 方向 |
|---|---|---|
| play 中 ghost 世界系漂移 | MimicLite 不追世界坐标（喂真值也有，§6） | 非缺陷；如需在仿真里对齐观察，可改可视化方式 |
| PICO / 真机遥操 | 仿真端已通（§7.1）；PICO 消息缺根平移；真机 `AcController.cpp` 只读 JSON 文件 | 补 PICO 字段；真机加流式参考输入（照搬 `teleop_play.py` 逻辑） |
| 最终动作偏离原始动作 | SONIC 重建误差（6.3° vs 真值 3.5°，§6） | 第二轮训练（§3.3），高动态约为静态 2 倍 |
| mink 目标依赖配对 robot 动作 | 当前用源 robot 动作的支撑脚作目标 | 纯 SMPL 输入时换成缩放后的 SMPL 关键点（GMR 原做法） |
| 真机 | 未编译、未运行 ROS1 部署包 | 仿真指标达标后再做 |

### 7.1 SMPL 遥操（2026-09-30 仿真端打通）

链路：SMPL 帧（ZMQ，50 Hz）→ 在线 SONIC 桥接 → mink（目标只来自 SMPL）→ FK → MimicLite 实时跟踪。
用法见 [play_guide.md](docs/source/getting_started/play_guide.md)"遥操"一节。

**前瞻延迟比原先估计的小得多**（更正：旧版本节写"编码器要 0.9 s 未来"是 robot 编码器的数字）。
SMPL 编码器未来帧步长是 1，一个窗口只看 10 帧；近端拼接下帧 t 最晚在收到 t+4 时算出。
在线结果与离线整段计算的关节**逐位一致**，不需要把未来钳位。

| 环节 | 延迟 |
|---|---|
| SONIC 桥接 | 稳态 ≤ 4 帧（开头攒第一个窗口 9 帧） |
| qvel 前向差分 | 1 帧 |
| 播放游标（MimicLite 看未来 4 帧 + 2 帧余量） | 6 帧 |
| **合计：SMPL 帧到达 → 机器人跟到这一帧** | **约 10 帧 = 200 ms** |

计算量：ONNX + mink 每帧 0.5～0.8 ms（CPU）。

**root 只来自 SMPL**（遥操没有配对的 robot PKL），用 21 条配对动作标定：
- **水平位置**：机器人 ≈ 0.59 × SMPL，误差 1.1 cm。
- **双脚轨迹**：同比例缩放后差 0.9 cm。
- **接触判定**：一致 89%。
- **朝向**：yaw 差 1.9°。

mink 的目标改为 SMPL 推出的双脚和 root，并做了三处调整，逐步排查后得到：
- **脚锁**：着地期间目标不动。
- **root 倾角不约束**：人和机器人骨盆倾角差约 7°，约束了反而把误差带进来。
- **脚的朝向目标放平**：只保留 yaw。

统一用真值接触帧衡量脚滑：

| 21 条动作 | root 水平误差 | root 朝向误差 | 参考脚滑 m/s |
|---|---|---|---|
| 离线（配对 robot root） | 2.2 cm | 0.8° | 0.047 |
| 在线，不加 mink | 1.4 cm | 7.7° | 0.186 |
| **在线 + mink（默认）** | 3.3 cm | 4.8° | **0.079** |

MimicLite 跟踪（21 条、8 环境、135 s）：

| 参考 | 机器人 vs 原始动作 | 失败 |
|---|---|---|
| 真值 | 3.49° | 0/42 |
| 离线 step 10000 + mink | 5.03° | 0/42 |
| **在线遥操链路**（`sonic_smpl_online`） | **5.54°** | **0/43** |

真实 ZMQ 流端到端（walk_forward_loop，headless）：机器人 vs 原始动作 4.46°；同一动作用离线模拟在线是 4.26°。

连续测试：走 3 遍、断流 3 次、恢复 2 次，40 s 零摔倒。播放落后稳定在 2～6 帧，没有跳帧。

**实现要点**（`tools_local/teleop_play.py`，不改同事代码，在 play.py 外面包一层）：
- **占位动作**：play 加载一条 30 分钟的站立占位动作，实时帧逐帧算 FK 后写进去。
  - FK 算法与 any4hdmi 相同：前向差分 qvel + `mj_forward`。
  - 启动时有自检：按 float16 存储精度比对，与数据集缓存完全一致。
- **热身**：先原地站 150 步，等 CUDA kernel 编译完再开始，否则开头跑不满实时，会触发跳帧。
- **reset 对齐实时帧**：摔倒等 reset 时从当前实时帧开始。
- **断流与恢复**：
  - 断流超过 0.3 s，参考在 2 s 内平滑回到站立。1 s 时从走路中途急停实测会摔。
  - 恢复时桥接重置，以当前位置为新起点，0.5 s 过渡。

**还没做的**：
1. **接 PICO**：官方 `pico_manager_thread_server.py` 的 pose 消息里**没有根平移**（内部算了骨盆位置但没发），
   要补一个 `transl` 字段，并按 `tools_local/smpl_stream.py` 的字段名发 `smpl` topic。
   另外要核对 PICO 的 `smpl_joints` 与训练 PKL 是否同一坐标约定（PKL 是 Z-up、骨盆为原点附近）。
2. **真机**：同事 `AcController.cpp` 只读 JSON 文件，要加一个流式参考输入（ROS topic），
   逻辑照搬 `teleop_play.py`（缓冲、游标、断流回站立）。
3. **SMPL→机器人标定**：0.59 等系数是用 21 条动作的平均体型标定的，换操作员后要在开始时按身高标定。
4. 备选：直接部署 `g1_dyn`（官方 SONIC 原生实时路径，已同训）。

---

## 8. 下一步

1. **盯第二轮训练**：回归每 2000 步出点。
   - 已确认在持续改善（§3.3）。
   - 如果停滞，或者 SMPL 链路与 robot 链路的差距（目前 0.28°）继续拉大，就调大 `g1_smpl_latent` 系数。
   - 出了更好的 checkpoint，用 `build_play_set.sh` 换进 play。
2. ~~root 自洽（mink）~~ ✅ 已接入桥接。它修的是参考动作打滑，不是 ghost 漂移（§6）。
   后续：纯 SMPL 输入时改用 SMPL 关键点作目标。
3. **遥操**：仿真端已通（§7.1）。下一步接 PICO（补根平移字段、核对坐标约定、按操作员身高标定），再做真机流式输入。
4. **`g1_dyn` 基线**：把第二轮的 `g1_dyn` 导出，在 sim2sim 里直接跑。作为"官方 SONIC 单网络"
   对照组，也作为实时遥操的备选路径。
5. **下一版数据集**：剔除 788 条非平地动作，并向同事确认 score2 的含义（§5）。
6. 仿真指标达标后：真机部署（ROS1 + ONNX，需先编译同事部署包）。

---

## 9. 架构图说明

2026-09-29 按实测现状重画 [media/sonic_mimiclite_bridge.svg](media/sonic_mimiclite_bridge.svg)，
主要修正：

0. 2026-09-29 晚再次更新：原红色"已知问题：root 不自洽导致漂移"改为已完成的 mink 后处理；
   另在 MimicLite 一栏标出真正的已知现象——ghost 世界系漂移（喂真值也有）。
1. 旧图把解码器画成"新增桥接层——唯一要训练的部分"。实际它就是 SONIC 自己的 `g1_kin`，
   与编码器一起训练；新图补上了同训但不部署的 `g1_dyn`。
2. 旧图写解码器输出 root。实际 root 取自源动作、不经解码器，新图用单独的蓝色虚线标出，
   并标注由此导致的已知问题。
3. 补充离线桥接（纯 ONNX、近端拼接）、MimicLite 观测的真实构成（639 维）、回归评测和
   运行载体（本地 mjlab 已验证 / 真机未验证），PICO 标为未打通。

---

## 10. 代码改动清单

从 2026-09-17 切换到 BUMI2 起的全部改动，按用途分组。详细理由和验证证据按提交号到
`SONIC_MimicLite_修改记录.md` 里查同日期的条目。

### 10.1 BUMI2 机型接入 SONIC（`c19b365`，2026-09-17）

| 文件 | 改动 |
|---|---|
| `gear_sonic/envs/manager_env/robots/bumi2.py` | 新增 BUMI2 机器人配置（执行器、PD、action scale 取自同事的实测值） |
| `gear_sonic/data/assets/robot_description/{urdf/bumi2,mjcf/bumi2.xml}` | 机器人资产；MJCF 补了 `<actuator>` 段（训练端按 actuator 顺序读 dof） |
| `gear_sonic/envs/manager_env/robots/__init__.py`、`trl/utils/order_converter.py`、`envs/manager_env/mdp/commands.py`、`envs/manager_env/modular_tracking_env_cfg.py` | 注册 BUMI2、关节顺序映射（只新增分支，不影响 G1 和其他机型） |
| `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_bumi2.yaml` | 第一轮训练配置 |

### 10.2 训练稳定性与正确性修复

| 提交 | 文件 | 问题 → 修复 |
|---|---|---|
| `7e244a2` 09-19 | `gear_sonic/trl/trainer/ppo_trainer.py` | 8 卡训练显存按 iteration 累积，最终 OOM。原来周期性 `gc`+`empty_cache` 的调用被注释掉了，恢复之后稳定在 `num_envs=3072` |
| `c400a31` 09-20 | `gear_sonic/utils/motion_lib/torch_humanoid_batch.py` | **关节错位**：根节点的 free joint 被误算成可控关节，21 个关节的训练目标整体错开一位。之前约 44 小时的训练全部作废重训。离线数据集不受影响 |

### 10.3 导出与桥接

| 提交 | 文件 | 改动 |
|---|---|---|
| `6220195` `a48f624` `17d4f9a` 09-21～23 | `gear_sonic/train_agent_trl.py` | 新增 `export_bridge_motion` 导出模式，改为多窗口拼接，并修复重建误差测量的冷启动偏差；导出四元数做**半球对齐**（相邻帧符号跳变会让下游插值绕远路）；新增 `inspect_g1_recon` 诊断 |
| `880ab1c` 09-28 | `gear_sonic/train_agent_trl.py` | 支持 `++use_encoder` 固定编码器。原来每次 reset 随机选编码器，拼出来的动作来源混杂。同时发现此前的"全链路打通"只覆盖了 robot 链路，SMPL 链路是这之后才验证的 |
| `442b5d3` 09-28 | `gear_sonic/eval_agent_trl.py` | ONNX 导出的解码器名可配置（`export_decoder_name`） |
| `c2ba831` 起，`4065a77` 加近端拼接 | `tools_local/sonic_offline_bridge.py` | 纯本地 ONNX 桥接：PKL → token → `g1_kin` → qpos npz；近端交叉淡化拼接（§2.4） |
| `d0e98ae` 09-29 | `tools_local/mink_root_solve.py`，桥接加 `--root-solve` | mink 反解 root（§2.5）。默认 `legs` 模式：上半身冻结、腿部正则权重 5，支撑脚位置权重 20、朝向权重 2，摆动脚 0.1，root 位置 1、朝向 5，求解器 daqp |
| `17d4f9a` 09-23 | `tools_local/bridge_json_to_any4hdmi_qpos.py` | 桥接 JSON 或原始 npz → MimicLite 可读的 any4hdmi qpos（`--from-raw` 生成真值组） |
| `3d124db` 09-30 | `tools_local/build_play_set.sh` | 用指定 checkpoint 一键重建三组 play 数据 |

### 10.4 第二轮训练（`7c0a243`，09-29）

见 §3.3：`G1ReconLossWeighted`，加上三个配置文件。

### 10.5 回归管线（`4065a77` 起）

见 §3.4：`tools_local/sonic_recon_regression.py`、`tools_local/sonic_regression_loop.sh`。

### 10.6 数据

`gear_sonic/tools/prepare_bumi2_sonic_dataset.py`（09-19）：见 [bumi2_sonic_dataset.md](docs/source/getting_started/bumi2_sonic_dataset.md)。

### 10.7 仓库重组（`24f3fd7`，09-24）

- 同事的 MimicLite-BUMI2 代码收进 `mimiclite_bumi2/`；
- 移除全部 BUMI3 代码和资产；
- 新建 `test_data/` 目录。

### 10.8 SMPL 遥操（2026-09-30）

| 文件 | 作用 |
|---|---|
| `tools_local/mink_root_solve.py` | 抽出逐帧的 `RootSolver` 类，离线和在线共用；离线结果与重构前逐位一致 |
| `tools_local/sonic_online_bridge.py` | `OnlineSmplBridge`：逐帧、因果的 SONIC 桥接（关节与离线逐位一致）；`SmplTargetEstimator`：只用 SMPL 估计 mink 目标（缩放、脚锁、接触判定） |
| `tools_local/smpl_stream.py` | SMPL 流的线格式（沿用官方 ZMQ pose 消息布局，topic `smpl`），以及用 PKL 模拟 PICO 的发送端 |
| `tools_local/teleop_play.py` | 仿真端：在 play.py 外包一层，把实时参考写进占位动作，负责游标同步、热身、断流回站立 |
| `tools_local/build_play_set.sh` | 增加第四组 `sonic_smpl_online` |
| mjlab 环境 | 新装 `pyzmq` |

### 10.9 试过但放弃

`tools_local/recover_root_from_joints.py`：手工锁脚来恢复 root。实测效果是负的，已被 mink 取代，只作为失败记录保留。
