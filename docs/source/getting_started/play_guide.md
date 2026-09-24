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
            ├── sonic_bridge/    我们 SONIC 重建的（qpos）
            └── original_qpos/   原始真值（qpos）
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
  task.command.motion_cfgs.bumi_v2.path=~/sonic_deploy/motions/any4hdmi-bumi-v2/motions/sonic_bridge \
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
| 我们 SONIC 重建的动作 | `.../motions/sonic_bridge` |
| 原始真值（A/B 对照） | `.../motions/original_qpos` |

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
  task.command.motion_cfgs.bumi_v2.path=/home/yingchaomu/sonic_deploy/motions/any4hdmi-bumi-v2/motions/sonic_bridge \
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

```bash
# Noetix-9 上 export_bridge_motion 导出 JSON，拉回本地后：
cd ~/下载/SONIC_MimicLite
python3 tools_local/bridge_json_to_any4hdmi_qpos.py \
  <桥接导出.json> \
  test_data/motions/any4hdmi-bumi-v2/motions/sonic_bridge/<名字>_from_g1_bumi_v2.npz \
  --manifest test_data/motions/any4hdmi-bumi-v2/manifest.json
```

原始 any4hdmi npz 转真值对照加 `--from-raw`。

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
