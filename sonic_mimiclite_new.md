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

## 3. 训练方案：方案 A（复用 SONIC 流程，不改训练框架）

结论（详见 sonic+mimiclite.md §7 末尾的算力讨论）：

- **不需要训练 `g1_dyn`**（重型 RL 动作解码器）——这个角色由冻结的 MimicLite-BUMI2
  Actor 承担。真正需要训练的只有 Robot/SMPL encoder + FSQ token + `g1_kin` 式重建
  解码器。
- **照抄 `sonic_bumi3.yaml` 的训练流程，不改一行训练框架代码**，只把机器人换成
  BUMI2、数据换成 BUMI2+SMPL/BVH，正常起 IsaacLab + PPO。
- **不追求 100k 轮训出完整策略**，只看 `g1_recon`/`g1_smpl_latent`/
  `reencoded_smpl_g1_latent` 这几个 aux loss 何时收敛就提前停——具体停在多少轮尚未
  实测，第一次训练需要盯着 loss 曲线现场判断，不能把某个具体数字当成已验证的承诺。
- 如果提前停之后成本仍然太高，再考虑写独立的纯监督训练脚本（跳过 IsaacLab PPO rollout，
  详见 sonic+mimiclite.md §7 的"方案 B"），但现在不做，先用零框架改动的方案 A 探路。

配置文件：[gear_sonic/config/exp/manager/universal_token/all_modes/sonic_bumi2.yaml](gear_sonic/config/exp/manager/universal_token/all_modes/sonic_bumi2.yaml)。

## 4. 算力与集群

- **只用 Noetix-9 单节点 8 卡**，不做跨机 16 卡联训。理由：这次训练目标是 aux loss
  收敛而非策略最优，不需要靠堆算力换更快收敛；跨机需要先过从未跑过的 NCCL smoke，
  多引入一层风险，对这次目标不值得。Noetix-0 留给后续同步复现 A 同事 MimicLite-BUMI2
  基线（不同框架、不占用 SONIC 训练资源）。
- 算力量级：复用 BUMI3 的 `num_envs=4096/卡` 规模，8 卡实测跑满 100k 轮约 576+
  GPU-hours（3 天）；本次目标是远早于 100k 轮的 aux loss 收敛点，具体 GPU-hours
  待第一次训练实测后回填到本文件。
- Noetix-9 磁盘：新挂载的 `/data0`（2TB，此前完全未使用的空盘，2026-09-17 新分区
  格式化挂载，详见 PROJECT_STATUS.md）用于存放训练数据；Noetix-0 同样挂了一份 `/data0`
  作为后续用途预留。
- 两台服务器目前仍缺 SONIC 训练用的 Python/Isaac Lab 环境（`.venv`），这是启动训练前
  必须先补的前置项，不是本文件已完成的部分。

## 5. 数据

| 数据集 | 来源 | 规模 | 状态（2026-09-17） |
|---|---|---|---|
| BUMI2 Robot 动作（npz） | `112.65.216.193:/data0/yc/bumi_v2_data/bumi_v2_filtered` | 51GB，114,995 文件 | 传输中，本地缓存 + 同步推送 Noetix-9:/data0 |
| 人体 BVH 动作 | `yuanjie-server(muyingchao):/data/muyingchao/soma_uniform` | 277GB，142,220 文件 | 传输中，预计需要约 19～23 小时（详见传输小节） |

两份数据都先落本地 `/home/yingchaomu/下载/{bumi_v2_data_cache,bvh_data_cache}/` 再同步
推送 Noetix-9（服务器出网受限，无法直连数据源，只能本地中转，详见 sonic+mimiclite.md
之前对话记录）。传输采用按子目录拆分的多路并行 rsync + `--partial` 断点续传。

数据传输完成后，下一步是仿照 `gear_sonic/tools/prepare_bumi3_sonic_dataset.py` 写
BUMI2 版本的数据准备脚本（重定向到 BUMI2 MJCF 关节顺序、50Hz 校验、SMPL/BVH 配对、
质量筛选），**这一步还没开始**，不要假设数据落地后就能直接训练。

## 6. 代码迁移进度（`feature/bumi2-sonic-migration` 分支）

完整改动清单和验证结果见 BUMI3_SONIC_修改记录.md 2026-09-17 条目，这里只列状态：

**已完成**（已本地验证，未上服务器实测）：
- BUMI2 资产迁移：URDF/MJCF/22 个 mesh 文件，来自同事 tarball 里已验证好效果的
  `BM2-V2.0` 权威定义，路径约定与 BUMI3 一致。
- `gear_sonic/envs/manager_env/robots/bumi2.py`：关节/刚体顺序映射（与 BUMI3 完全一致，
  已用纯 Python 逻辑核实）、`BUMI2_CFG` 执行器参数（与 BUMI3 明显不同，来自同事权威配置）、
  `BUMI2_ACTION_SCALE`（实测值，不是公式推导，公式对不上的原因未查证）。
- 四处注册：`robots/__init__.py`、`modular_tracking_env_cfg.py`、`order_converter.py`
  （新增 `Bumi2Converter`）、`mdp/commands.py`（下肢索引断言）。
- 新训练入口 `sonic_bumi2.yaml`。
- 静态验证：Python 编译通过、YAML 解析通过、URDF/MJCF XML 合法、mesh 引用完整、
  DOF/Body 映射逻辑核实。

**未完成、不要假设已经能跑**：
- 未在真实 Isaac Lab 环境里实例化过 `BUMI2_CFG`（本机和两台服务器目前都没有 Isaac Lab
  运行环境）。
- 数据准备脚本（§5 末尾）未写。
- 未跑过任何 1-env smoke、训练 smoke 或正式训练。
- `g1_kin` 式桥接 decoder 的重新定向/微调（sonic+mimiclite.md §8.2 的开放问题：BUMI2
  版本的 `g1_kin` 输出格式是否需要额外适配 MimicLite-BUMI2 的 ObsRef 输入）还没做，
  要等 SONIC 侧训练先跑起来、拿到真实 checkpoint 之后才能验证。
- BUMI2 的 MuJoCo sim2sim/部署工具（对应 BUMI3 的 `bumi3_sim2sim.py` 等）未写，推迟到
  桥接验证阶段。

## 7. 架构图更新说明

`media/sonic_mimiclite_bridge.svg`/`.png` 已从最初的 BUMI3 版本改为 BUMI2 版本，两处
关键修改：

1. 原图"SONIC 上层"标注"原样冻结"——这是 BUMI3 语境下的说法（当时已有训好的 BUMI3
   SONIC checkpoint 可以直接冻结）。BUMI2 语境下这个前提不成立，SONIC 上层需要**新
   训练**，图上已改成"新训练（复用 SONIC-BUMI3 框架与经验）"，避免误导后续读者以为
   这一层不需要投入算力。
2. "MimicLite 下层"和"机器人本体"两处标注从泛指改为明确写 **BUMI2**，避免和 BUMI3
   混淆。

## 8. 下一步（按顺序）

1. 数据传输完成并核对 Noetix-9 端文件数/大小一致。
2. 在 Noetix-9 上装好 SONIC 训练用的 Python/Isaac Lab 环境（`.venv`）。
3. 写 BUMI2 数据准备脚本，把 §5 的原始数据转成 SONIC motion-lib 格式。
4. 在 Isaac Lab 里实际实例化 `BUMI2_CFG`，跑通 1-env reset/step smoke，把 §6"未完成"
   里这一项转正。
5. 用 `sonic_bumi2.yaml` 启动 8 卡训练，盯 aux loss 曲线，找到实际收敛点并回填到 §4。
6. 训练收敛后，验证/微调 `g1_kin` 式桥接 decoder 是否能直接喂给 MimicLite-BUMI2 的
   ObsRef，这是整个方案最终能否成立的关键验证点。
