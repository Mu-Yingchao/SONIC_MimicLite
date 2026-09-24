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

整个搭建过程有**四个坑**，每个都会让你以为已经装好了、实际跑不起来。按顺序做。

### 2.0 先建 ASCII 软链接

Hydra 的 override 语法解析器**处理不了路径里的非 ASCII 字符**，
`task.command.motion_cfgs.bumi_v2.path=/home/yingchaomu/下载/...` 会直接报
`LexerNoViableAltException`，箭头正指在中文字符上。加引号能绕过一层，但下一个
含中文的参数（比如 `checkpoint_path`）照样报。建两个软链接一劳永逸：

```bash
ln -sfn /home/yingchaomu/下载/SONIC_MimicLite_deploy            /home/yingchaomu/sonic_deploy
ln -sfn /home/yingchaomu/下载/mimiclite_review/bumi2/active-adaptation /home/yingchaomu/sonic_aa
```

本文后面所有命令都走这两个 ASCII 路径。

### 2.1 创建环境（务必配国内镜像）

```bash
AA=/home/yingchaomu/sonic_aa
mkdir -p "$AA/venv/mjlab"
cp "$AA/projects/mimic-lite/pyproject-mjlab.toml" "$AA/venv/mjlab/pyproject.toml"
cd "$AA/venv/mjlab"

# 关键：绕开本机 Clash 代理，走清华镜像
export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
export no_proxy="${no_proxy},pypi.tuna.tsinghua.edu.cn"
export NO_PROXY="${NO_PROXY},pypi.tuna.tsinghua.edu.cn"

uv sync
```

**为什么必须配镜像**：本机有全局 `HTTPS_PROXY=http://127.0.0.1:7897`（Clash），
uv 会把所有 PyPI 下载塞进这一条代理连接。实测同一个 CUDA wheel：

| 路线 | 吞吐 | 下载剩余 2.5GB 预计 |
|---|---|---|
| Clash 代理 | 0.023 MB/s | **约 30 小时** |
| 清华 TUNA 直连 | **31.29 MB/s** | 约 80 秒 |

只把镜像域名加进 `no_proxy`、不整个清掉代理变量，是因为 `mjhub` 和 `any4hdmi`
两个依赖从 GitHub 拉源码，那部分仍然要走代理。

排查手法记一下：**不要只看缓存目录大小或日志有没有新行**——停滞和慢下载看起来
一模一样。要么用 `ps -o etime,time` 对比"墙上时间 vs 累计 CPU 时间"，要么像这里
一样对临时目录做两点采样算真实速率。

### 2.2 手动补装 warp 和 mjlab（且必须指定 --python）

`pyproject-mjlab.toml` 声明依赖 `active_adaptation[mjlab]`，但主
`pyproject.toml` 里**根本没定义 `mjlab` 这个 extra**（只有 `render`）。uv 不会
报错，这个 extra 会静默解析成空集，于是 `warp` 和 `mjlab` 两个包都装不上。

补装时**必须显式写 `--python`**：如果当前 shell 里有 `VIRTUAL_ENV` 指向别的
环境（比如仓库自带的 `.venv_sim`），`uv pip install` 会装进那个环境而不是刚建的
`.venv`，并且照样报告安装成功，只有最后 import 时才暴露。

```bash
cd /home/yingchaomu/sonic_aa/venv/mjlab
unset VIRTUAL_ENV
uv pip install --python .venv/bin/python warp-lang mjlab
```

### 2.3 重建项目注册表

框架靠 `<active-adaptation>/.cache/projects.json` 决定加载哪些项目和 learning
插件。**同事打包的 tar 里没有这个文件**，缺了它 `mimic_lite_learning` 不会被
导入，`play.py` 会报 `Could not find 'algo/mimic_lite_ppo'`。

```bash
cd /home/yingchaomu/sonic_aa
unset VIRTUAL_ENV
venv/mjlab/.venv/bin/aa-project discover --enabled

# discover --enabled 会把所有项目都启用，但 facet/metamorph/mimic 依赖 IsaacLab，
# mjlab 环境里没有，不关掉会在 import 阶段报 ModuleNotFoundError: No module named 'isaaclab'
for p in facet metamorph mimic; do
  venv/mjlab/.venv/bin/aa-project disable "$p"
done
```

正确状态应该只有 `mimic_lite` 的 environment 和 learning 两项是 `enabled=true`。

### 2.4 验证环境

```bash
cd /home/yingchaomu/sonic_aa
unset VIRTUAL_ENV
venv/mjlab/.venv/bin/python -c "
import warp, mjlab, active_adaptation, mimic_lite, any4hdmi, torch
print('warp', warp.__version__)
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
"
```

实测通过的版本：`warp 1.17.0`、`torch 2.11.0+cu128`、`cuda True`、RTX 4090。
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

所有路径都用 §2.0 建的 ASCII 软链接，否则 Hydra 解析不了。

```bash
cd /home/yingchaomu/sonic_aa
unset VIRTUAL_ENV
CKPT=/home/yingchaomu/sonic_aa/outputs/2026-09-15/15-56-35-BumiV2TrackBase-from_checkpoint/checkpoint_40000.pt
MOTION=/home/yingchaomu/sonic_deploy/any4hdmi-bumi-v2/motions/sonic_bridge
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
  task.command.motion_cfgs.bumi_v2.path=$MOTION \
  checkpoint_path=$CKPT
```

`headless=false` 打开 mjlab 交互式 viewer。**它是 viser 网页版，不弹原生窗口**——
启动后浏览器打开 <http://127.0.0.1:8080>（以终端实际打印的端口为准）即可看到
实时画面，支持暂停、重置、调速和热加载 checkpoint。

逐步操作见 `local_play_quickstart.md`。

参数说明：

| 参数 | 作用 |
|---|---|
| `task/motion=bumi/v2` | 载入 BUMI2 动作配置的骨架，`path` 随后被下面一行覆盖 |
| `task.command.motion_cfgs.bumi_v2.path` | **指向我们自己的动作目录**，这是链路的接入点 |
| `algo/ppo/module=huge` | 必须与 checkpoint 训练时的网络规模一致，否则权重对不上 |
| `task.termination.root_pos_error.enabled=false` | 关掉 root 位置误差终止，避免刚起步就被打断 |
| `task.num_envs=2` | 本地可视化用 1~2 个环境就够 |

### 4.2 录成 MP4

MP4 写到**当前工作目录**，所以先 `cd` 到想放视频的地方，再用绝对路径调 play.py：

```bash
mkdir -p /home/yingchaomu/sonic_deploy/renders
cd /home/yingchaomu/sonic_deploy/renders
unset VIRTUAL_ENV

/home/yingchaomu/sonic_aa/venv/mjlab/.venv/bin/python \
  /home/yingchaomu/sonic_aa/projects/mimic-lite/scripts/play.py \
  task=tracking-bumi-v2 task/motion=bumi/v2 +exp=ppo/train \
  algo/ppo/module=huge backend=mjlab headless=true task.num_envs=1 \
  task.termination.root_pos_error.enabled=false \
  task.command.motion_cfgs.bumi_v2.path=$MOTION \
  checkpoint_path=$CKPT \
  render_seconds=9
```

文件名形如 `20260923-225806-0e7ad876.mp4`。

**注意**：`render_seconds=9` 对应 450 个仿真步，实测渲染约 80 秒（比实时慢）。
视频文件在开始时就会被创建但长度只有 48 字节，**必须等进程真正退出**再看，
不能一发现 `.mp4` 存在就以为录完了。判断完成用 `pgrep -f scripts/play.py`
是否还在，或者检查文件大小是否 > 10KB。

### 4.3 A/B 对照：跑真值动作

把 `path` 换成 `original_qpos` 目录即可，其余参数完全不变：

```bash
  task.command.motion_cfgs.bumi_v2.path=/home/yingchaomu/sonic_deploy/any4hdmi-bumi-v2/motions/original_qpos
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

### 4.5 本地实测基线（2026-09-23）

用 `checkpoint_40000.pt` 跑我们自己桥接导出的 `walk_forward_loop_003__A022`
（434 帧），`task.num_envs=2`：

```text
('stats', 'episode_len')                      434.0    <- 与动作真实长度精确一致
('stats', 'termination', 'motion_timeout')    1.0
('stats', 'termination', 'body_pos_error')    0.0
('stats', 'termination', 'body_ori_error')    0.0
('stats', 'termination', 'root_ori_error')    0.0
```

两个并行环境都正常播完，零次因追踪误差被打断。与此前在同事服务器上跑同一条
动作的结果对照，本地复现良好：

| tracking_metrics | 本地 | 同事服务器 |
|---|---|---|
| body_pos | 9.46 | 9.44 / 9.77 |
| body_ori | 43.11 | 43.33 / 43.36 |
| joint_pos | 20.45 | 20.08 / 20.74 |
| root_ori | 26.22 / 27.58 | 24.54 / 27.22 |
| root_pos | 133.03 / 164.93 | 128.63 / 147.06 |

差异在同一条动作不同起始相位的正常波动范围内，说明本地环境与同事训练环境
行为一致，可以放心用本地做后续验证。

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
| `LexerNoViableAltException`，箭头指在中文字符上 | Hydra 解析不了非 ASCII 路径，用 §2.0 的 ASCII 软链接 |
| `uv sync` 慢到几十小时 | 走了 Clash 代理，按 §2.1 配 TUNA 镜像 + `no_proxy` |
| `ModuleNotFoundError: warp` 或 `mjlab`（但安装时报告成功） | 装进了 `VIRTUAL_ENV` 指向的别的环境，按 §2.2 加 `--python` 重装 |
| `Could not find 'algo/mimic_lite_ppo'` | 缺 `.cache/projects.json`，按 §2.3 用 `aa-project discover --enabled` 重建 |
| `ModuleNotFoundError: No module named 'isaaclab'` | `discover --enabled` 把 facet/metamorph/mimic 也启用了，按 §2.3 关掉 |
| MP4 只有 48 字节 | 渲染还没结束就去看了，等 `pgrep -f scripts/play.py` 为空再检查 |
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
