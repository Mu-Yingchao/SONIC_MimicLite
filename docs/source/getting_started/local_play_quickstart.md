# 本地可视化 play 速查

直接复制粘贴就能跑。完整背景、环境搭建和踩坑说明见
`local_mimiclite_bumi2_deploy.md`，本文只管"现在就要看效果"。

## 0. 先搞清楚在用哪套代码

可视化 play 跑的是同事的**训练框架**（`active-adaptation` + mjlab 仿真），
**不是**真机 ROS1 部署代码。两者别混：

| 用途 | 代码位置 | 本文是否涉及 |
|---|---|---|
| 仿真可视化 play | `~/sonic_aa`（= `mimiclite_review/bumi2/active-adaptation`） | ✅ 就是它 |
| 真机 ROS1 部署 | `mimiclite_deploy_review/deploy_bumi_v2` | ❌ 另一条链路 |

`~/sonic_deploy` 是**数据目录**（动作 npz、数据集 manifest、渲染视频），
不含任何可执行部署代码。

## 1. 准备（每个新终端都要做一次）

```bash
cd /home/yingchaomu/sonic_aa
unset VIRTUAL_ENV

PY=/home/yingchaomu/sonic_aa/venv/mjlab/.venv/bin/python
PLAY=/home/yingchaomu/sonic_aa/projects/mimic-lite/scripts/play.py
CKPT=/home/yingchaomu/sonic_aa/outputs/2026-09-15/15-56-35-BumiV2TrackBase-from_checkpoint/checkpoint_40000.pt

# 我们 SONIC 重建的动作
OURS=/home/yingchaomu/sonic_deploy/any4hdmi-bumi-v2/motions/sonic_bridge
# 原始真值动作（A/B 对照用）
TRUTH=/home/yingchaomu/sonic_deploy/any4hdmi-bumi-v2/motions/original_qpos
```

`unset VIRTUAL_ENV` 不能省——你的仓库里有个 `.venv_sim`（Python 3.10）会干扰。

路径必须走 `~/sonic_aa` / `~/sonic_deploy` 这两个 ASCII 软链接，Hydra 的
参数解析器处理不了 `/home/yingchaomu/下载/...` 里的中文字符。

## 2. 交互式看（推荐，能暂停/调速）

```bash
$PY $PLAY \
  task=tracking-bumi-v2 task/motion=bumi/v2 +exp=ppo/train \
  algo/ppo/module=huge backend=mjlab headless=false task.num_envs=2 \
  task.termination.root_pos_error.enabled=false \
  task.command.motion_cfgs.bumi_v2.path=$OURS \
  checkpoint_path=$CKPT
```

### 画面在浏览器里，不是弹窗

mjlab 的 viewer 是 **viser 网页版**，不会弹原生窗口。启动后终端会打印：

```text
╭────── viser (listening *:8080) ───────╮
│   HTTP      │ http://localhost:8080   │
│   Websocket │ ws://localhost:8080     │
╰───────────────────────────────────────╯
```

**浏览器打开 <http://127.0.0.1:8080> 就能看到实时画面。** 页面里支持暂停、
重置、调速、热加载 checkpoint。

几个注意点：

- 首次启动要编译 CUDA kernel，**等 40～60 秒**才出画面，不是卡死。
- 正常跑起来后终端会持续刷 `Loop FPS: 50 frames in 1.00s`，即实时速度。
- 用 `127.0.0.1` 而不是 `localhost`，避免浏览器全局代理把请求转走。
- 端口被占时 viser 会自动换一个，**以终端实际打印的为准**。
- 确认服务在线：`curl -s -o /dev/null --noproxy '*' -w "%{http_code}\n" http://127.0.0.1:8080/`
  返回 `200` 即正常。

看真值动作做对照——只改 `$OURS` 为 `$TRUTH`，其它一模一样：

```bash
$PY $PLAY \
  task=tracking-bumi-v2 task/motion=bumi/v2 +exp=ppo/train \
  algo/ppo/module=huge backend=mjlab headless=false task.num_envs=2 \
  task.termination.root_pos_error.enabled=false \
  task.command.motion_cfgs.bumi_v2.path=$TRUTH \
  checkpoint_path=$CKPT
```

## 3. 录 MP4（要分享或反复看）

视频写到**当前工作目录**，所以先 `cd` 过去：

```bash
mkdir -p /home/yingchaomu/sonic_deploy/renders
cd /home/yingchaomu/sonic_deploy/renders

$PY $PLAY \
  task=tracking-bumi-v2 task/motion=bumi/v2 +exp=ppo/train \
  algo/ppo/module=huge backend=mjlab headless=true task.num_envs=1 \
  task.termination.root_pos_error.enabled=false \
  task.command.motion_cfgs.bumi_v2.path=$OURS \
  checkpoint_path=$CKPT \
  render_seconds=9
```

`render_seconds=9` ≈ 450 个仿真步，实测渲染约 80 秒。

**别被空文件骗了**：mp4 一开始就会被创建但只有 48 字节，必须等进程真正退出。
判断完成：

```bash
pgrep -f scripts/play.py   # 没输出才是跑完了
ls -lh /home/yingchaomu/sonic_deploy/renders/*.mp4   # 应该有几百 KB
```

## 4. 怎么看结果是好是坏

终端会刷出统计，**盯 termination 这四行**：

```text
('stats', 'episode_len')                      434.0   <- 应等于动作帧数
('stats', 'termination', 'motion_timeout')    1.0     <- 正常播完
('stats', 'termination', 'body_pos_error')    0.0     <- 追踪失败会 >0
('stats', 'termination', 'body_ori_error')    0.0
('stats', 'termination', 'root_ori_error')    0.0
```

`motion_timeout=1.0` 且三个 `*_error` 全 `0.0` = 策略把整条动作跟完了。

**`('stats','success')` 不能当合格线**——它是个很严格的单集阈值，同事真实训练
数据跑出来也经常是 `0.0`。真正的失败信号是上面那三个 `*_error`。

`tracking_metrics` 是累积误差，只用来横向比（比如我们的动作 vs 真值动作），
绝对值没有固定及格分。

### 已有基线（2026-09-23 实测，我们的重建动作）

```text
episode_len 434.0 | motion_timeout 1.0 | 三个 error 全 0.0
tracking_metrics: body_pos 9.46  body_ori 43.11  joint_pos 20.45
```

## 5. 换成别的动作

部署端现在只有一条动作的两个版本。要加新动作：

```bash
# 1) 在 Noetix-9 上用 export_bridge_motion 导出 JSON，拉回本地
# 2) 转成 any4hdmi qpos 格式
cd /home/yingchaomu/下载/SONIC_MimicLite
python3 tools_local/bridge_json_to_any4hdmi_qpos.py \
  <桥接导出的.json> \
  /home/yingchaomu/sonic_deploy/any4hdmi-bumi-v2/motions/sonic_bridge/<名字>_from_g1_bumi_v2.npz \
  --manifest /home/yingchaomu/sonic_deploy/any4hdmi-bumi-v2/manifest.json
```

放进 `motions/sonic_bridge/` 就会被自动扫到，play 命令不用改。
详见 `local_mimiclite_bumi2_deploy.md` §3。

## 6. 出问题先看这里

| 现象 | 处理 |
|---|---|
| `LexerNoViableAltException` | 路径里有中文，改用 `~/sonic_aa` / `~/sonic_deploy` |
| `Could not find 'algo/mimic_lite_ppo'` | 缺注册表，跑 `venv/mjlab/.venv/bin/aa-project discover --enabled` 再 disable 掉 facet/metamorph/mimic |
| `No module named 'isaaclab'` | 同上，那三个项目没关掉 |
| `No module named 'warp'` | 装进了别的 venv，`uv pip install --python .venv/bin/python warp-lang mjlab` |
| `headless=false` 没弹窗 | 正常，mjlab 用 viser 网页版，浏览器开 http://127.0.0.1:8080 |
| 启动后一直没画面 | 正常，首次编译 CUDA kernel 要 40～60 秒 |
| 浏览器打不开 8080 | 端口可能被换了，看终端里 viser 实际打印的地址；或浏览器代理拦了，改用 `127.0.0.1` |
| mp4 只有 48 字节 | 还没录完，等 `pgrep -f scripts/play.py` 没输出 |
| `KeyError: qpos` | 往 `motions/` 里放了原始格式 npz，它得先转换 |
