# BUMI2 本地 play 指南

在本地用 MimicLite-BUMI2 策略播放参考动作，浏览器看效果。

## 目录

```text
SONIC_MimicLite/
├── gear_sonic/              SONIC 上层（在 Noetix-9 训练）
├── mimiclite_bumi2/
│   ├── training/            同事训练框架，play.py 在这里
│   └── deploy/              同事 ROS1 真机代码（尚未使用）
└── test_data/
    ├── policies/mimiclite/  checkpoint_40000.pt
    └── motions/
        ├── robot_pkl/       6 条 SONIC 格式机器人动作
        ├── smpl_pkl/        6 条配对 SMPL 动作
        └── any4hdmi-bumi-v2/motions/
            ├── sonic_smpl_mink/ SMPL 链路 + mink 后处理（推荐）
            ├── sonic_smpl/      SMPL 链路，root 直接拼源动作（对照）
            ├── sonic_robot/     robot 链路
            └── original_qpos/   原始真值，不经 SONIC（对照基准）
```

两个 ASCII 软链接（Hydra 解析不了中文路径，必须用）：

```bash
ln -sfn ~/下载/SONIC_MimicLite/mimiclite_bumi2/training ~/sonic_aa
ln -sfn ~/下载/SONIC_MimicLite/test_data               ~/sonic_deploy
```

## 跑 play

```bash
cd ~/sonic_aa
unset VIRTUAL_ENV

venv/mjlab/.venv/bin/python projects/mimic-lite/scripts/play.py \
  task=tracking-bumi-v2 task/motion=bumi/v2 +exp=ppo/train \
  algo/ppo/module=huge backend=mjlab headless=false task.num_envs=2 \
  task.termination.root_pos_error.enabled=false \
  task.command.motion_cfgs.bumi_v2.path=~/sonic_deploy/motions/any4hdmi-bumi-v2/motions/sonic_smpl_mink \
  checkpoint_path=~/sonic_deploy/policies/mimiclite/checkpoint_40000.pt
```

`~` 在 Hydra 参数里不展开，实际使用时写全 `/home/yingchaomu/sonic_deploy/...`。

### 看画面

**画面在浏览器，不弹窗。** 启动后终端打印：

```text
╭────── viser (listening *:8080) ───────╮
│   HTTP      │ http://localhost:8080   │
```

打开 <http://127.0.0.1:8080>。用 `127.0.0.1` 不用 `localhost`，避免浏览器代理拦截。
端口被占时 viser 会自动换，以终端打印的为准。

首次启动编译 CUDA kernel 要 **40～60 秒**才出画面，不是卡死。
正常后终端持续刷 `Loop FPS: 50 frames in 1.00s`。

## 切换数据

改 `task.command.motion_cfgs.bumi_v2.path` 一个参数，其余不动：

| 看什么 | path 指向 |
|---|---|
| SMPL → SONIC → mink（项目目标链路，推荐） | `.../motions/sonic_smpl_mink` |
| SMPL → SONIC，root 直接拼源动作 | `.../motions/sonic_smpl` |
| Robot → SONIC | `.../motions/sonic_robot` |
| 原始真值，不经 SONIC（对照基准） | `.../motions/original_qpos` |

每个目录 21 条动作，覆盖静态/行走/跑跳/表演/高动态。

### 看 play 时注意：ghost 会"漂"，这是正常的

viser 里的半透明 ghost 按参考动作的**世界坐标**画，机器人的世界位置由物理仿真决定。MimicLite
的观测只有相对量、不追世界坐标，所以两者会逐渐拉开——**喂原始真值也一样**（实测平均 0.25 m）。
判断追踪好坏看姿态是否一致（终端的 `body_pos_error` / `body_ori_error` 终止统计），不看 ghost
离机器人多远。ghost 自身脚底打滑才是参考动作的问题，mink 后处理就是修这个的。

官方也是这么定性的：`root_pos_error`（世界系 root 偏离 >0.4 m）配置为 `is_timeout: true`（截断而非失败），
`scripts/eval.py` 把它和 `motion_timeout` 一起算作 success，官方 README 的 play 命令同样关掉它。

目录里所有 `.npz` 都会被自动扫到，加新动作直接放进去即可。

## 换策略

改 `checkpoint_path`。必须同时保证 `algo/ppo/module=huge` 与该 checkpoint
训练时的网络规模一致，否则权重 shape 对不上。

## 录 MP4

视频写到**当前工作目录**：

```bash
mkdir -p ~/sonic_deploy/renders && cd ~/sonic_deploy/renders
unset VIRTUAL_ENV

/home/yingchaomu/sonic_aa/venv/mjlab/.venv/bin/python \
  /home/yingchaomu/sonic_aa/projects/mimic-lite/scripts/play.py \
  task=tracking-bumi-v2 task/motion=bumi/v2 +exp=ppo/train \
  algo/ppo/module=huge backend=mjlab headless=true task.num_envs=1 \
  task.termination.root_pos_error.enabled=false \
  task.command.motion_cfgs.bumi_v2.path=/home/yingchaomu/sonic_deploy/motions/any4hdmi-bumi-v2/motions/sonic_smpl_mink \
  checkpoint_path=/home/yingchaomu/sonic_deploy/policies/mimiclite/checkpoint_40000.pt \
  render_seconds=9
```

`render_seconds=9` ≈ 450 步，约 80 秒。**mp4 一开始就创建但只有 48 字节**，
必须等进程退出：`pgrep -f scripts/play.py` 无输出才算完。

## 判断追踪好坏

盯终端这四行：

```text
('stats', 'episode_len')                      434.0   应等于动作帧数
('stats', 'termination', 'motion_timeout')    1.0     正常播完
('stats', 'termination', 'body_pos_error')    0.0     >0 就是追踪失败
('stats', 'termination', 'body_ori_error')    0.0
('stats', 'termination', 'root_ori_error')    0.0
```

`motion_timeout=1.0` + 三个 `*_error` 全 `0.0` = 跟完了。

`('stats','success')` **不是**合格线，真实训练数据也常是 `0.0`。
`tracking_metrics` 是累积误差，只用来横向比较，没有固定及格分。

已验证基线（我们的重建动作，434 帧）：
`body_pos 9.46 / body_ori 43.11 / joint_pos 20.45`

## 加新动作

任意 SONIC 训练用的 robot/smpl PKL 都能直接走全链路，纯本地、不需要服务器：

```bash
cd ~/下载/SONIC_MimicLite
unset VIRTUAL_ENV
M=test_data/motions/any4hdmi-bumi-v2
mimiclite_bumi2/training/venv/mjlab/.venv/bin/python tools_local/sonic_offline_bridge.py \
  --motion   test_data/motions/smpl_pkl/<名字>.pkl \
  --root-motion test_data/motions/robot_pkl/<名字>.pkl \
  --encoder smpl --onnx test_data/policies/sonic/model_step_017600_smpl.onnx \
  --manifest $M/manifest.json \
  --out $M/motions/sonic_smpl_mink/<名字>_from_g1_bumi_v2.npz
```

- robot 链路：`--motion` 指 robot PKL、`--encoder robot`、`--onnx ..._g1.onnx`，不需要 `--root-motion`
- 默认带 mink 后处理（`--root-solve legs`）；`--root-solve none` 得到 root 直接拼源动作的对照版
- 真值对照：`tools_local/bridge_json_to_any4hdmi_qpos.py <原始npz> <输出> --manifest ... --from-raw`
- 放进目录就会被 play 自动扫到，命令不用改

## 环境重建

只在换机器或环境损坏时需要。

```bash
AA=~/下载/SONIC_MimicLite/mimiclite_bumi2/training
mkdir -p $AA/venv/mjlab && cp $AA/projects/mimic-lite/pyproject-mjlab.toml $AA/venv/mjlab/pyproject.toml
cd $AA/venv/mjlab

# 必须配镜像：本机全局 HTTPS_PROXY 指向 Clash，实测下载 0.023 MB/s（要 30 小时），
# 走 TUNA 直连 31 MB/s（80 秒）。GitHub 依赖仍需代理，所以只把镜像域名加进 no_proxy。
export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
export no_proxy="${no_proxy},pypi.tuna.tsinghua.edu.cn"
uv sync

# 主 pyproject 没定义 mjlab 这个 extra，uv 会静默跳过，warp/mjlab 装不上。
# 必须显式 --python，否则会装进 VIRTUAL_ENV 指向的其它环境且报告成功。
unset VIRTUAL_ENV
uv pip install --python .venv/bin/python warp-lang mjlab
# 桥接的 mink 后处理（root 自洽）需要 mink 与 QP 求解器
uv pip install --python .venv/bin/python mink "qpsolvers[daqp]"

# 项目注册表（.cache/projects.json）不在 Git 里，需现场生成。
# discover --enabled 会连带启用三个依赖 IsaacLab 的项目，必须关掉。
cd $AA
venv/mjlab/.venv/bin/aa-project discover --enabled
for p in facet metamorph mimic; do venv/mjlab/.venv/bin/aa-project disable $p; done
```

移动过 venv 目录的话，`.venv/bin/` 里的 shebang 和 site-packages 里的
editable finder 都存着绝对路径，要整体替换旧路径才能用。

## 故障排查

| 现象 | 处理 |
|---|---|
| `LexerNoViableAltException` | 路径含中文，改用 `~/sonic_aa` / `~/sonic_deploy` |
| `headless=false` 没弹窗 | 正常，开浏览器 http://127.0.0.1:8080 |
| 启动后长时间没画面 | 正常，首次编译 CUDA kernel 40～60 秒 |
| `Could not find 'algo/mimic_lite_ppo'` | 缺注册表，见"环境重建"最后一段 |
| `No module named 'isaaclab'` | facet/metamorph/mimic 没 disable |
| `No module named 'warp'` | 装错 venv，用 `--python` 重装 |
| `错误的解释器` | venv 被移动过，修 `.venv/bin/` 里的 shebang |
| `KeyError: qpos` | 往 motions/ 放了原始格式 npz，需先转换 |
| mp4 只有 48 字节 | 没录完，等 `pgrep -f scripts/play.py` 无输出 |
