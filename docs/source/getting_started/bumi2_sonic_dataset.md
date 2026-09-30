# BUMI2 SONIC 训练数据集：来源、对齐与整理

上层 SONIC 训练集 `sonic_bumi2_dataset_v1` 的完整来历、构建命令和验证方法。

## 1. 总览

```text
同事的 BUMI2 机器人动作                          BUMI3 时期已对齐筛选的 SMPL
/data2/zcx/datasets/bumiv2（同事服务器）          /data/ouqin/datasets/bumi3/train/smpl（同事服务器）
   │ 同事分档：score1 / score2                         │ 与机型无关，直接复用
   ▼                                                  ▼
Noetix-9 /data0/bumi_v2_filtered                Noetix-9 /data0/bumi3_smpl_filtered/smpl
  score1 94,224 条 + score2 20,771 条（51 GB）       97,660 条 PKL（22 GB）
   │                                                  │
   └──────── gear_sonic/tools/prepare_bumi2_sonic_dataset.py build ────────┘
             Robot：转成 SONIC 契约        SMPL：按文件名配对，帧数必须严格相等
                              ▼
          Noetix-9 /data0/bumi2_sonic_dataset_v1/
            built/robot_all   114,995 条（8.5 GB）
            built/smpl_all     93,822 条（16 GB，与 robot 同名配对）
            meta/manifest.jsonl、meta/provenance.json
```

最终规模：配对 93,822 条，只有 robot 的 21,173 条。只有 robot 的动作 SMPL 编码器看不到。

## 2. Robot 侧

### 2.1 来源

- **生产方**：同事做的 BUMI2 动作数据，文件名以 `*_from_g1_bumi_v2.npz` 结尾。
- **分档**：按目录分成 `score1/`、`score2/` 两档，下面按采集批次（`210531/`、`220713/` 等）再分目录。
- **字段**：`joint_pos`、`joint_vel` `[T,21]`，`body_pos_w` `[T,22,3]`，`body_quat_w` `[T,22,4]`，`fps`。

同事在这批数据上还做了两类筛选（脚本在 `mimiclite_bumi2/training/projects/mimic-lite/scripts/`）：

| 脚本 | 作用 |
|---|---|
| `screen_bumi_motions.py` | 在 mjlab 里把每一帧都当 reset 状态模拟几步，找出会产生非有限数值的动作 |
| `filter_flat_ground_motions.py` | 按动作名剔除非平地（楼梯、斜坡等）和依赖道具的动作，从 score1 移到 score3 |
| `convert_bumiv2_to_any4hdmi.py` | 转成 MimicLite 训练用的 any4hdmi qpos 格式（`/data2/zcx/datasets/any4hdmi-bumi-v2`） |

> **与 MimicLite 训练集的两处差异**（2026-09-30 核实）：
> 1. MimicLite 只用 score1 训练，**SONIC 用的是 score1+score2 全部**。score1/score2 的分档标准没有写进代码，需要问同事。
> 2. 同事的平地筛选（移到 score3）是在 any4hdmi 目录上做的，**没有同步到 `bumi_v2_filtered`**：
>    SONIC 训练集里仍有 788 条名字含 stair/ladder/ramp/slope/obstacle/hurdle 的动作。
>    SONIC 是在平地上训练 PPO 的，这些动作跟不住，会拉低 `g1_dyn` 的数据效率；对 `g1_kin` 重建影响不大。
>    下一版数据集建议按 `filter_flat_ground_motions.py` 的同一规则剔除。

### 2.2 核实过的格式事实（实测，不是假设）

| 项 | 结论 | 怎么核实的 |
|---|---|---|
| `joint_pos` 关节顺序 | Isaac Lab 顺序 | 第 9/10 列恰好卡在 `l_arm_roll_joint` 下限 −0.0872、`r_arm_roll_joint` 上限 0.0872 |
| 四元数 | wxyz（scalar-first） | Isaac Lab 惯例，转换后与 FK 逐位一致 |
| 帧率 | 已是 50 Hz | 不用重采样 |
| 根高度 | 0.46～0.47 m | 与 BUMI2 参考根高一致 |
| 穿地 | 不存在 | 数据来自 Isaac Lab rollout，不需要 BUMI3 那套 Root-Z 足底优化 |

### 2.3 转成 SONIC 契约

| SONIC 字段 | 算法 |
|---|---|
| `root_trans_offset` | `body_pos_w[:,0,:]`（base_link） |
| `root_rot` | `body_quat_w[:,0,:]`，wxyz → xyzw 存盘 |
| `dof` | `joint_pos` 按 Isaac Lab → MuJoCo 顺序重排（顺序表内联在脚本里） |
| `pose_aa` | 根：四元数转 axis-angle；21 个单自由度关节：MJCF 关节轴 × 关节角 |
| `fps` | 50 |

- **为什么不直接 import `bumi2.py`**：`pxr` 必须等 Isaac Sim 启动后才能 import，离线脚本没法用，所以关节顺序直接内联在脚本里。
- **为什么 `pose_aa` 不用 `body_quat_w`**：其余 21 个 body 要换算父子相对姿态，容易出错；直接用关节轴 × 关节角更可靠。

## 3. SMPL 侧

- **来源**：BUMI3 项目阶段已经对齐、筛选好的最终态 PKL，字段为 `pose_aa` `[T,72]`、`transl` `[T,3]`、`smpl_joints` `[T,24,3]`、`fps=50`。
- **可以跨机型复用**：SMPL 是人体动作，与机型无关，不做任何转换。
- **对齐和筛选的方法**：BVH ↔ robot 时间对齐检查、阈值校准、分档等，见 `sonic_motion_dataset_alignment_playbook.md`。那是 BUMI3 时期总结的通用方法，换新机型、从 BVH 重新做数据时照着用。
- **配对规则**：
  1. robot 文件名去掉 `_from_g1_bumi_v2` 后缀，就是 SMPL 文件名。
  2. **两边帧数必须严格相等**。不等的不插值、不截断，直接降级为只有 robot。
  3. 结果：97,660 条 SMPL 里配上 93,822 条（96.1%）。

## 4. 构建与验证命令（Noetix-9）

```bash
cd /root/SONIC_MimicLite
PY=/data0/sonic_mimiclite_env/bin/python

# 构建（16 进程）
$PY gear_sonic/tools/prepare_bumi2_sonic_dataset.py build \
  --robot-root /data0/bumi_v2_filtered \
  --smpl-root /data0/bumi3_smpl_filtered/smpl \
  --output-root /data0/bumi2_sonic_dataset_v1 \
  --workers 16
# 成功标志：BUMI2_SONIC_DATASET_BUILD=PASS  robot=114995 paired=93822 robot_only=21173

# 全量复核（重新读源文件逐条比对）
$PY gear_sonic/tools/prepare_bumi2_sonic_dataset.py validate \
  --output-root /data0/bumi2_sonic_dataset_v1
# 成功标志：BUMI2_SONIC_DATASET_VALIDATE=PASS (114995 条)
```

产物说明：
- `meta/manifest.jsonl`：每条动作一行，记录来源文件、帧数、是否配对。
- `meta/provenance.json`：记录 MJCF 的 sha256、仓库 commit、各项计数。

**已验证**：
- 20 条抽查逐值比对：根位置、四元数、21 关节重排都与源 npz 逐位一致；`pose_aa` 每个关节的轴角模长等于 `|dof|`。
- 与同事 any4hdmi 版本交叉核对：同一条动作（`walk_forward_loop_003__A022`）我们算出的 qpos 与同事 any4hdmi npz 差 1.8e-7。**两边训练数据同源一致。**

训练时通过命令行参数指定数据集，不写进配置文件：

```text
++manager_env.commands.motion.motion_lib_cfg.motion_file=/data0/bumi2_sonic_dataset_v1/built/robot_all
++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=/data0/bumi2_sonic_dataset_v1/built/smpl_all
```

## 5. 本地测试集（从训练集里抽的）

| 路径 | 内容 |
|---|---|
| `test_data/motions/robot_pkl/`、`smpl_pkl/` | 21 条配对动作，直接从 `built/` 拷来 |
| `test_data/motions/any4hdmi-bumi-v2/motions/original_qpos/` | 同 21 条的真值 qpos，不经 SONIC，给 play 做对照 |

21 条按类型挑选，覆盖静态日常、行走、跑跳、表演舞蹈、高动态五类，用于回归评估和 play。它们**也在训练集里**，所以测出的是拟合精度，不是泛化精度。

真值 qpos 的生成命令：

```bash
tools_local/bridge_json_to_any4hdmi_qpos.py <源npz> <输出> --manifest <manifest.json> --from-raw
```

## 6. 相关记录

- 构建过程、环境补丁、验证输出：`SONIC_MimicLite_修改记录.md` 中 2026-09-19 的条目。
- 数据准备通用方法（BVH 对齐、筛选、分档）：`sonic_motion_dataset_alignment_playbook.md`。
- 转换脚本本体，docstring 写了格式核实过程：`gear_sonic/tools/prepare_bumi2_sonic_dataset.py`。
