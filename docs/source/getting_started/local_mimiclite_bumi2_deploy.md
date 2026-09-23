# 本地 MimicLite-BUMI2 部署端：跑通 SONIC 桥接动作并可视化

本地开发机同时是**控制端**（发起 Noetix 服务器上的 SONIC 训练）和**部署端**
（跑 MimicLite-BUMI2 策略、可视化、导出真机产物）。

**不在同事的训练服务器上跑 play。** 同事服务器只作为只读参考来源
（取 manifest、取数据集自带的 MJCF），验证和可视化一律在本地做。

文档不保存任何密码、私钥正文或访问令牌。

## 1. 三个角色与它们的目录

```text
/home/yingchaomu/下载/SONIC_MimicLite/            我们的仓库（控制面，唯一代码真源）
  ├─ gear_sonic/train_agent_trl.py                 含桥接导出模式 export_bridge_motion
  ├─ tools_local/bridge_json_to_any4hdmi_qpos.py   桥接 JSON -> any4hdmi qpos 转换
  └─ tools_local/mimiclite_cluster.sh              Noetix 集群入口

/home/yingchaomu/下载/mimiclite_review/bumi2/active-adaptation/   同事训练框架（只读参考）
  ├─ projects/mimic-lite/scripts/play.py           播放入口
  ├─ active_adaptation/assets/BUMI/BM2-V2.0/       机器人资产（仿真环境加载这份）
  ├─ outputs/.../checkpoint_40000.pt               同事训练好的 BUMI2 追踪策略
  └─ venv/mjlab/.venv/                             本地建的运行环境（见 §2）

/home/yingchaomu/下载/SONIC_MimicLite_deploy/     部署端工作目录（数据，不进 Git）
  ├─ any4hdmi-bumi-v2/                             给 play.py 读的 any4hdmi 数据集
  │   ├─ manifest.json                             数据集权威元信息
  │   ├─ mjcf/bumi_v2_0810_rl.xml                  【FK 骨架版】sha256 d512a7e1…
  │   ├─ meshes/
  │   └─ motions/
  │       ├─ sonic_bridge/                         我们从 SONIC token 重建的动作
  │       └─ original_qpos/                        Noetix-9 原始动作真值（A/B 对照）
  └─ raw_from_noetix9/                             原始源格式 npz（不能放进 motions/）
```

### 两份同名 MJCF，角色不同，放反了 FK 会算错

BUMI2 在 MimicLite 侧有两份都叫 `bumi_v2_0810_rl.xml` 但内容不同的文件：

| 角色 | sha256 前缀 | 所在位置 | 内容特征 |
|---|---|---|---|
| 仿真环境加载的机器人 | `14cc4382` | `active-adaptation/active_adaptation/assets/BUMI/BM2-V2.0/mjcf/` | 实为 `bumi_v2_0904_rl` 版，带 armature、胶囊碰撞体、contact exclude |
| 数据集 FK 用的骨架 | `d512a7e1` | `SONIC_MimicLite_deploy/any4hdmi-bumi-v2/mjcf/` | manifest 里 `source_mjcf_sha256` 记的那版，圆柱/盒碰撞体、无 armature |

两份的关节名、轴向、限位完全一致，差异只在碰撞几何和 armature。同事服务器上
就是这个分工，不是配置错误。**本地必须照搬**。

`raw_from_noetix9/` 里的原始 npz **不能**放进 `motions/` 下：any4hdmi 用
`rglob("*.npz")` 扫描 `motions/`，扫到没有 `qpos` 键的文件会直接报错。

## 2. 一次性环境搭建

```bash
AA=/home/yingchaomu/下载/mimiclite_review/bumi2/active-adaptation
mkdir -p "$AA/venv/mjlab"
cp "$AA/projects/mimic-lite/pyproject-mjlab.toml" "$AA/venv/mjlab/pyproject.toml"
cd "$AA/venv/mjlab"
uv sync
```

`uv sync` 会自己拉 CPython 3.12、torch 2.11 和整套 CUDA 依赖，**首次要几十分钟**，
大部分时间在下载。

### 必须手动补装 warp 和 mjlab

`pyproject-mjlab.toml` 声明依赖 `active_adaptation[mjlab]`，但主
`pyproject.toml` 里**根本没定义 `mjlab` 这个 extra**（只有 `render`）。uv 不会
报错，这个 extra 会静默解析成空集，于是 `warp` 和 `mjlab` 两个包都装不上。

`uv sync` 显示成功之后必须补这两步，否则 `play.py` 会在 import 阶段就挂掉：

```bash
cd "$AA/venv/mjlab"
uv pip install warp-lang
uv pip install mjlab
```

### 验证环境

```bash
cd "$AA/venv/mjlab"
.venv/bin/python -c "
import warp, mjlab, active_adaptation, mimic_lite, any4hdmi, torch
print('warp', warp.__version__)
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('全部 import 通过')
"
```

`cuda` 必须是 `True`，mjlab 的物理后端（mujoco_warp）要跑在 GPU 上。

## 3. 完整链路：SONIC token → 本地可视化

```text
[Noetix-9]  SONIC 训练出的 decoder
     │  export_bridge_motion（多窗口拼接）
     ▼
  bumi_deploy_motion_v1 JSON          ← 真机 ROS 部署端读的也是这个格式
     │  tools_local/bridge_json_to_any4hdmi_qpos.py
     ▼
  any4hdmi qpos npz
     │  play.py + checkpoint_40000.pt
     ▼
[本地]  MimicLite-BUMI2 策略追踪 + 可视化
```

### 3.1 在 Noetix-9 上导出桥接动作

```bash
cd /home/yingchaomu/下载/SONIC_MimicLite
ssh -i .local/keys/noetix9.pem root@14.103.42.170
# 服务器上（在 SONIC 训练环境里）：
#   ++export_bridge_motion=true
#   ++export_bridge_motion_output=/data0/bridge_export_<动作名>.json
```

导出的 JSON 写 root 四元数之前会做**半球对齐**（逐帧检查与前一帧点积，为负取反）。
不做这一步的话，`motion_lib` 返回的符号在帧之间会跳变——姿态本身不错（`q` 和
`-q` 是同一个旋转），但任何插值/slerp 在跳变处会沿"长路"绕一圈，插出完全错误
的中间姿态。踩坑的地方包括 any4hdmi 建 FK cache 时的重采样，以及真机
`AcController.cpp` 读 JSON 之后的平滑。

把 JSON 拉回本地：

```bash
cd /home/yingchaomu/下载/SONIC_MimicLite
scp -i .local/keys/noetix9.pem \
  root@14.103.42.170:/data0/bridge_export_<动作名>.json \
  /home/yingchaomu/下载/SONIC_MimicLite_deploy/
```

### 3.2 转成 any4hdmi qpos 格式

```bash
cd /home/yingchaomu/下载/SONIC_MimicLite
DEPLOY=/home/yingchaomu/下载/SONIC_MimicLite_deploy

python3 tools_local/bridge_json_to_any4hdmi_qpos.py \
  "$DEPLOY/bridge_export_<动作名>.json" \
  "$DEPLOY/any4hdmi-bumi-v2/motions/sonic_bridge/<动作名>_from_g1_bumi_v2.npz" \
  --manifest "$DEPLOY/any4hdmi-bumi-v2/manifest.json"
```

关节重排顺序（IsaacLab -> MuJoCo）**从 manifest 现读**，不写死硬编码表。
文件名结尾保持 `_from_g1_bumi_v2.npz`，与数据集既有命名约定一致。

### 3.3 生成真值对照（可选，但强烈建议）

Noetix-9 上有 any4hdmi 的原始源数据，可以取同一条动作的真值做 A/B：

```bash
DEPLOY=/home/yingchaomu/下载/SONIC_MimicLite_deploy
cd /home/yingchaomu/下载/SONIC_MimicLite

# 找原始文件
ssh -i .local/keys/noetix9.pem root@14.103.42.170 \
  'find /data0/bumi_v2_filtered -name "<动作名>_from_g1_bumi_v2.npz"'

# 拉回本地（注意放 raw_from_noetix9/，不能放 motions/ 下）
scp -i .local/keys/noetix9.pem \
  root@14.103.42.170:/data0/bumi_v2_filtered/score1/<日期>/<动作名>_from_g1_bumi_v2.npz \
  "$DEPLOY/raw_from_noetix9/"

# 转成 qpos（--from-raw 走 body_pos_w[:,0]+body_quat_w[:,0]+reordered joint_pos 配方）
python3 tools_local/bridge_json_to_any4hdmi_qpos.py \
  "$DEPLOY/raw_from_noetix9/<动作名>_from_g1_bumi_v2.npz" \
  "$DEPLOY/any4hdmi-bumi-v2/motions/original_qpos/<动作名>_from_g1_bumi_v2.npz" \
  --manifest "$DEPLOY/any4hdmi-bumi-v2/manifest.json" --from-raw
```

### 3.4 精度对比

```bash
python3 - << 'PYEOF'
import numpy as np
D="/home/yingchaomu/下载/SONIC_MimicLite_deploy/any4hdmi-bumi-v2/motions"
NAME="<动作名>_from_g1_bumi_v2.npz"
gt=np.load(f"{D}/original_qpos/{NAME}")["qpos"]
ou=np.load(f"{D}/sonic_bridge/sonic_bridge_{NAME}")["qpos"]
n=min(len(gt),len(ou)); gt,ou=gt[:n],ou[:n]
print("root 位置 最大差(m): %.8f" % np.abs(gt[:,:3]-ou[:,:3]).max())
print("root 朝向 最大差:    %.8f" % np.abs(gt[:,3:7]-ou[:,3:7]).max())
jd=np.rad2deg(np.abs(gt[:,7:]-ou[:,7:]))
print("关节角 平均/p95/最大(度): %.3f / %.3f / %.3f"%(jd.mean(),np.percentile(jd,95),jd.max()))
# 四元数连续性自检：两边都应该是 0
for nm,q in (("真值",gt[:,3:7]),("我们",ou[:,3:7])):
    print(f"{nm} 相邻帧符号跳变: {(np.einsum('ij,ij->i',q[:-1],q[1:])<0).sum()}/{len(q)-1}")
PYEOF
```

root 位置应当**逐位相等**（不过 decoder，直接取自驱动源真值），root 朝向应在
`1e-7` 量级（float32 精度内）。关节角的差就是 SONIC 的真实重建误差。
相邻帧符号跳变两边都必须是 0。

## 4. 本地播放与可视化

```bash
AA=/home/yingchaomu/下载/mimiclite_review/bumi2/active-adaptation
DEPLOY=/home/yingchaomu/下载/SONIC_MimicLite_deploy
CKPT="$AA/outputs/2026-09-15/15-56-35-BumiV2TrackBase-from_checkpoint/checkpoint_40000.pt"
cd "$AA"
```

### 4.1 交互式可视化（主要用法）

```bash
venv/mjlab/.venv/bin/python projects/mimic-lite/scripts/play.py \
  task=tracking-bumi-v2 \
  task/motion=bumi/v2 \
  +exp=ppo/train \
  algo/ppo/module=huge \
  backend=mjlab \
  headless=false \
  task.num_envs=2 \
  task.termination.root_pos_error.enabled=false \
  task.command.motion_cfgs.bumi_v2.path="$DEPLOY/any4hdmi-bumi-v2/motions/sonic_bridge" \
  checkpoint_path="$CKPT"
```

`headless=false` 打开 mjlab 交互式 viewer，支持暂停、重置、调速和热加载
checkpoint。

参数说明：

| 参数 | 作用 |
|---|---|
| `task/motion=bumi/v2` | 载入 BUMI2 动作配置的骨架，`path` 随后被下面一行覆盖 |
| `task.command.motion_cfgs.bumi_v2.path` | **指向我们自己的动作目录**，这是链路的接入点 |
| `algo/ppo/module=huge` | 必须与 checkpoint 训练时的网络规模一致，否则权重对不上 |
| `task.termination.root_pos_error.enabled=false` | 关掉 root 位置误差终止，避免刚起步就被打断 |
| `task.num_envs=2` | 本地可视化用 1~2 个环境就够 |

### 4.2 录成 MP4

```bash
venv/mjlab/.venv/bin/python projects/mimic-lite/scripts/play.py \
  task=tracking-bumi-v2 task/motion=bumi/v2 +exp=ppo/train \
  algo/ppo/module=huge backend=mjlab headless=true task.num_envs=1 \
  task.termination.root_pos_error.enabled=false \
  task.command.motion_cfgs.bumi_v2.path="$DEPLOY/any4hdmi-bumi-v2/motions/sonic_bridge" \
  checkpoint_path="$CKPT" \
  render_seconds=10
```

MP4 写到**当前工作目录**，文件名形如 `20260923-213000-a1b2c3d4.mp4`。

### 4.3 A/B 对照：跑真值动作

把 `path` 换成 `original_qpos` 目录即可，其余参数完全不变：

```bash
  task.command.motion_cfgs.bumi_v2.path="$DEPLOY/any4hdmi-bumi-v2/motions/original_qpos"
```

### 4.4 怎么判断追踪是否成功

看日志里的终止统计，**不要只看 `success`**：

```text
('stats', 'termination', 'motion_timeout')    1.0   <- 正常播完
('stats', 'termination', 'body_pos_error')    0.0   <- 追踪失败会 >0
('stats', 'termination', 'body_ori_error')    0.0
('stats', 'termination', 'root_ori_error')    0.0
```

`motion_timeout=1.0` 且三个 `*_error` 都是 `0.0`，说明策略把整条参考动作跟完了。
`('stats','success')` 是个更严格的单集判定阈值，真实训练数据里也经常是 `0.0`，
**不能**拿它当"追踪失败"的依据。

`('stats','tracking_metrics',*)` 是 episode 累积误差，用来横向比较我们的动作和
真值动作，绝对值本身没有固定合格线。

## 5. Noetix-9 上的 BUMI2 数据集路径速查

```text
/data0/bumi2_sonic_dataset_v1/            SONIC 训练用数据集（PKL）
  ├─ built/robot_all/    114995 个 .pkl   8.5G
  ├─ built/smpl_all/      93822 个 .pkl    16G
  └─ meta/
      ├─ manifest.jsonl   47MB，每行一条动作的来源与帧数
      └─ provenance.json  契约版本、MJCF 校验和、配对数量、target_fps

/data0/bumi_v2_filtered/                  any4hdmi 原始源 npz（51G）
  ├─ score1/<日期>/*_from_g1_bumi_v2.npz   122 个日期目录
  └─ score2/<日期>/*_from_g1_bumi_v2.npz
```

两者的关系：`bumi2_sonic_dataset_v1` 是从 `bumi_v2_filtered` 转换来的，
`meta/manifest.jsonl` 每行的 `robot_source` 字段记着原始文件路径。

注意两种 npz 格式不要混：

| 目录 | npz 内容 | 能否直接给 play.py |
|---|---|---|
| `/data0/bumi_v2_filtered/` | `joint_pos`/`joint_vel`/`body_pos_w`/`body_quat_w`/… | **不能**，要先 `--from-raw` 转成 qpos |
| 转换后的 `motions/*/` | 单个 `qpos` 键，`(T,28)` | 可以 |

查某条动作的来源：

```bash
ssh -i .local/keys/noetix9.pem root@14.103.42.170 \
  'grep -m1 "\"key\": \"<动作名>\"" /data0/bumi2_sonic_dataset_v1/meta/manifest.jsonl'
```

## 6. 故障排查

| 现象 | 原因与处理 |
|---|---|
| `ModuleNotFoundError: warp` 或 `mjlab` | `uv sync` 静默跳过了 `[mjlab]` extra，按 §2 手动补装 |
| `No qpos motions found under …` | `path` 指的目录里没有 `.npz`，或指到了数据集根而不是 motions 子目录 |
| `KeyError: qpos is not a file in the archive` | 把原始源格式 npz 放进 `motions/` 了，移到 `raw_from_noetix9/` |
| `Could not find manifest.json above …` | 动作目录必须在带 `manifest.json` 的数据集根之下，any4hdmi 靠逐级向上找来定位数据集 |
| 策略权重加载报 shape 不匹配 | `algo/ppo/module=huge` 漏了或写错，必须与训练时一致 |
| 机器人姿态诡异、抖动 | 先查两份 MJCF 有没有放反（§1 表格），再查四元数相邻帧符号跳变是否为 0 |
| `torch.cuda.is_available()` 是 False | mjlab 物理后端要 GPU，检查驱动与 torch 的 CUDA 版本 |
| 首次跑很慢、卡在 `Building FK cache` | 正常，FK cache 按数据集内容哈希缓存，同一批动作第二次跑会直接命中 |

## 7. 与集群训练文档的关系

本文只覆盖**部署端**。代码同步、集群训练启动、TensorBoard 隧道等控制面操作，
见 `codex_local_control_multi_node_training.md`。两份文档共用同一套原则：
GitHub 是代码唯一真源，服务器只 fast-forward，数据和模型不进普通 Git。
