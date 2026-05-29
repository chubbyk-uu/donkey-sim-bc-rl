# Donkey Simulator BC/RL 中文说明

本项目记录了 Donkey Simulator 上的行为克隆（BC）和强化学习（RL）实验。运行方式是
在 WSL2 中启动 Python 训练/评估脚本，连接 Windows 侧正在运行的 Donkey Simulator。

```text
camera image -> (crop) -> encoder -> SAC policy -> steer / throttle
```

详细实验过程、失败分支和设计原因见：[docs/experiment-log.md](docs/experiment-log.md)。

**当前状态（2026-05-29）：已到达一个稳定里程碑。** 项目已有可运行的 BC 基线、最早
single generated-road VAE+SAC 基线、fixed-light loop 部署模型、Mountain DINOv2
部署模型，以及 random light + tree shadows 的 domain-randomization 实验记录。
fixed-light loop 上最强模型仍是 track-specific VAE；需要光照/外观泛化时，DINOv2
是目前最好的 frozen encoder。剩余主要问题是随机树影下的鲁棒感知：frozen encoder
只能做到部分成功，下一步真正有价值的方向是 task-adapted encoder，例如 fine-tuned
DINOv2、depth 或 segmentation，而不是继续微调 reward。

---

## 快速开始

### 1. Python 环境

```bash
conda create -n rl311 python=3.11
conda activate rl311
pip install "stable-baselines3[extra]" gymnasium numpy pillow opencv-python tqdm tensorboard
```

PyTorch 请根据本机 CUDA 版本，从 PyTorch 官方安装页面选择对应命令安装。

### 2. gym-donkeycar

不要安装 PyPI 上的旧包；它使用旧 Gym API，和本项目不兼容：

```bash
# 错误：
pip install gym-donkeycar

# 正确：从 upstream 安装
pip install git+https://github.com/tawnkramer/gym-donkeycar
```

也可以 clone 到本地做 editable install：

```bash
git clone https://github.com/tawnkramer/gym-donkeycar ../gym-donkeycar
pip install -e ../gym-donkeycar
```

### 3. 连接 Simulator

Windows Donkey Simulator 和 WSL2 Python 客户端通过 TCP 9091 通信。Windows host IP
每次启动可能变化，可从默认路由中读取：

```bash
export DONKEY_SIM_HOST="$(ip route | awk '/default/ {print $3}')"
export DONKEY_SIM_PORT=9091
```

建议写入 `~/.bashrc`。Simulator 启动后，用下面命令验证端口：

```bash
nc -vz "$DONKEY_SIM_HOST" 9091
```

### 4. 模型文件

模型文件使用 Git LFS 管理。clone 后执行：

```bash
git lfs pull
```

### 5. 跑一个 smoke test

打开 simulator 的 `generated_track` 场景后，运行 fixed-light loop 主部署模型：

```bash
python rl/eval_loop_vae_sac.py \
  --encoder vae \
  --vae-model models/vae_loop_cones_fixedlight_v1/best.pt \
  --model models/rl_loop_vae_sac_fixedlight_v3_h80/sac_loop_vae_90000_steps.zip \
  --episodes 3 --max-episode-steps 2000
```

---

## 当前部署模型

### 推荐模型

| 场景 | 模型 | Encoder | Eval | Speed | CTE | 说明 |
|---|---|---|:---:|---:|---:|---|
| Fixed-light loop (`donkey-generated-track-v0`) | `rl_loop_vae_sac_fixedlight_v3_h80` 90k | VAE | 3/3 trunc @ 2000 | 2.800 | 0.315 | loop 主部署 |
| Fixed-light loop，长 cap 评估 | `rl_loop_vae_v15` 112k | VAE | 5/5 trunc @ 3000 | 2.797 | 0.356 | co-primary / long-eval 部署 |
| Mountain (`donkey-mountain-track-v0`) | `rl_dinov2_mountain_v2` 40k | DINOv2-S | 5/5 trunc @ 2000 | 2.093 | 0.577 | mountain 主部署 |

### 研究/备份模型

| 场景 | 模型 | Encoder | Eval | 说明 |
|---|---|---|:---:|---|
| Loop，光照鲁棒 frozen encoder | `rl_loop_dinov2_v8` 30k | DINOv2-S | 3/3 trunc @ 2000 | 不需要 VAE 数据采集；比 VAE 慢，但对 random light 更稳 |
| Loop，ResNet 备份 | `rl_loop_vae_sac_resnet_v4_notrees` 50k | ResNet18 | 3/3 trunc @ 3000 | 历史备份；评估必须带 `--encoder-crop-top 40` |
| Loop，random light + trees/shadows | `rl_loop_dinov2_randomtree_v2` 50k | DINOv2-S | 约 50% random-layout trunc | 当前最好的 frozen-encoder domain-randomized 分支 |
| Single generated road | `rl_vae_sac_raffin_v1/final_model.zip` | VAE | 1390+ steps 接近路线终点 | 最早 Raffin-style 非闭环道路基线 |
| Behavioral cloning | `bc_nvidia_slow_006_flip`, `bc_official_categorical_curve_aug_balanced_v1` | CNN | 历史基线 | 有参考价值，但整体弱于 RL |

### 场景选择

| 目标 | 优先使用 | 原因 |
|---|---|---|
| fixed-light loop 上最快最稳 | VAE loop deployment | 固定光照下，track-specific VAE 速度最快、CTE 最居中 |
| 不想采 VAE 数据 / 有光照变化 | DINOv2 loop backup | frozen DINOv2 对 random light 明显比 VAE 稳 |
| random light + random tree shadows | DINOv2 domain-randomized model | 目前最佳 frozen encoder，但仍只是部分解决 |
| Mountain track | DINOv2 mountain v2 40k | right-lane CTE target 下稳定 5/5 truncate |
| 最早 single generated road | Raffin VAE+SAC final model | 历史基线；非闭环路线通常不到 3000 step 就到终点 |
| BC 对照实验 | 先看 regression CNN | regression 比 categorical 更稳，但两者都需要 steering scale，且弱于 RL |

---

## 常用评估命令

### Fixed-light loop VAE 主模型

```bash
python rl/eval_loop_vae_sac.py \
  --encoder vae \
  --vae-model models/vae_loop_cones_fixedlight_v1/best.pt \
  --model models/rl_loop_vae_sac_fixedlight_v3_h80/sac_loop_vae_90000_steps.zip \
  --episodes 3 --max-episode-steps 2000
```

### DINOv2 loop 备份

不需要 VAE 数据采集，对光照变化更鲁棒：

```bash
python rl/eval_loop_vae_sac.py \
  --encoder dinov2_vits14 \
  --model models/rl_loop_dinov2_v8/sac_dinov2_vits14_30000_steps.zip \
  --episodes 3 --max-episode-steps 2000
```

### ResNet18 loop 备份

```bash
python rl/eval_loop_vae_sac.py \
  --encoder resnet18 --encoder-crop-top 40 \
  --model models/rl_loop_vae_sac_resnet_v4_notrees/sac_resnet18_50000_steps.zip \
  --episodes 3 --max-episode-steps 2000
```

注意：这个 v4 ResNet checkpoint 训练时使用了历史设置 `--encoder-crop-top 40`，评估时
必须保持一致。新的 ResNet 实验默认应使用 `--encoder-crop-top 0`。

### Mountain 主模型

```bash
python rl/eval_loop_vae_sac.py \
  --env-id donkey-mountain-track-v0 \
  --encoder dinov2_vits14 \
  --cte-target 3.5 --max-cte-error 2.5 \
  --max-throttle 0.7 \
  --model models/rl_dinov2_mountain_v2/sac_dinov2_vits14_40000_steps.zip \
  --episodes 5 --max-episode-steps 2000
```

Mountain 场景的 spawn CTE 约为 3.54（右车道中心），黄色中线是 CTE=0。使用
`--cte-target 3.5` 可以让 reward 和 termination 都按 `abs(cte - 3.5)` 计算。
上坡路段需要 `--max-throttle 0.7`。

### Single generated-road 基线

这是最早的 Raffin-style VAE+SAC 基线。它跑的是固定 generated road，不是闭环道路；
成功运行通常会在 3000 step 前到达路线末端附近。

```bash
python rl/eval_vae_sac.py \
  --model models/rl_vae_sac_raffin_v1/final_model.zip \
  --vae-model models/vae_raffin_v1/best.pt \
  --episodes 3
```

### Behavioral Cloning 基线

Regression CNN：

```bash
python bc/eval_bc.py \
  --model models/bc_nvidia_slow_006_flip/best.pt \
  --steering-scale 1.8 \
  --episodes 3
```

Official-style categorical：

```bash
python bc/eval_bc_official_categorical.py \
  --model models/bc_official_categorical_curve_aug_balanced_v1/best.pt \
  --steering-scale 1.4 \
  --episodes 3
```

两个 BC 模型原始 steering 输出都偏小，闭环评估时需要额外 scale：regression 约 1.8 倍，
categorical 约 1.4 倍。观察上 regression 更稳定，categorical 弯道响应更尖锐但一致性
较差。两者都明显弱于 RL policy。

BC 数据主要在：

```text
data/slow_data_raw/                  6 个 generated-road 慢速驾驶 tub
data/curated_cornering_v*_clean/      增强弯道数据子集
```

完整 BC 过程见 [experiment log §1](docs/experiment-log.md#1-behavioral-cloning)。

---

## 训练

### Loop Track — VAE + SAC（推荐）

VAE 路线要求 simulator 固定光照：采集、SAC 训练、评估都要关闭 `randomlight`。当前
fixed-light loop VAE 使用 80k 帧右车道数据，按 PID lateral offset 分布采集：

```text
cte  0.0: 30k
cte +0.5:  8k    cte -0.5:  8k
cte +1.0:  8k    cte -1.0:  8k
cte +1.5:  7k    cte -1.5:  7k
cte +2.0:  2k    cte -2.0:  2k
```

每个 bucket 单独采集，然后合成一个 manifest 和 memmap cache：

```bash
# 1. 采图。这里是一个 bucket 示例；其他 cte target/count 按上表重复。
python tools/collect_sim_frames.py \
  --env-id donkey-generated-track-v0 \
  --action-mode cte-pid \
  --cte-target 0.0 \
  --frames 30000 \
  --output-dir data/vae_raw/generated_track_loop_fixedlight_cte_0_30k

# 2. 从所有 fixed-light bucket 生成 train/val manifest
python tools/prepare_vae_dataset.py \
  --source data/vae_raw/generated_track_loop_fixedlight_cte_0_30k \
  --source data/vae_raw/generated_track_loop_fixedlight_cte_p05_8k \
  --source data/vae_raw/generated_track_loop_fixedlight_cte_m05_8k \
  --source data/vae_raw/generated_track_loop_fixedlight_cte_p10_8k \
  --source data/vae_raw/generated_track_loop_fixedlight_cte_m10_8k \
  --source data/vae_raw/generated_track_loop_fixedlight_cte_p15_7k \
  --source data/vae_raw/generated_track_loop_fixedlight_cte_m15_7k \
  --source data/vae_raw/generated_track_loop_fixedlight_cte_p20_2k \
  --source data/vae_raw/generated_track_loop_fixedlight_cte_m20_2k \
  --output-dir data/vae_loop_cones_fixedlight_v1 \
  --dedupe

# 3. 构建 train_vae.py 使用的 cropped uint8 memmap cache
python tools/build_vae_cache.py \
  --manifest data/vae_loop_cones_fixedlight_v1/manifest.jsonl \
  --output-dir data/vae/cache_loop_cones_fixedlight_v1

# 4. 训练 VAE encoder
python rl/train_vae.py \
  --cache-dir data/vae/cache_loop_cones_fixedlight_v1 \
  --output-dir models/vae_loop_cones_fixedlight_v1 \
  --epochs 20

# 5. 训练 SAC（已验证 v3_h80 recipe：safe_v2 reward, hidden=80）
python rl/train_loop_vae_sac.py \
  --encoder vae \
  --vae-model models/vae_loop_cones_fixedlight_v1/best.pt \
  --output-dir models/rl_loop_new \
  --hidden-size 80 \
  --batch-size 64 \
  --gradient-steps-min 50 \
  --gradient-steps-cap 2000 \
  --alive-reward 1.5 \
  --speed-reward-weight 0.15 \
  --cte-speed-penalty-weight 0.25 \
  --reward-crash -10 --crash-speed-weight 5 \
  --min-throttle 0.2 --max-throttle 0.7 \
  --max-cte-error 2.0 --max-episode-steps 3000 \
  --timesteps 100000 --save-replay-buffer --device cuda
```

不要默认使用 final checkpoint。SAC 经常出现 10-20k 周期的 peak/valley/recovery，
应对每个 10k checkpoint 做 deterministic eval，再选择最好的部署模型。当前 CLI 默认是
`--batch-size 128 --gradient-steps-cap 1000`，适合新实验，但尚未干净复现已部署的
v3_h80 结果。

### Loop Track — DINOv2 / ResNet（不需要 VAE 采集）

```bash
python rl/train_loop_vae_sac.py \
  --encoder dinov2_vits14 \
  --output-dir models/rl_loop_dinov2_new \
  --hidden-size 256 \
  --batch-size 128 \
  --gradient-steps-min 50 \
  --gradient-steps-cap 1000 \
  --timesteps 60000 --save-replay-buffer --device cuda
```

把 `--encoder dinov2_vits14` 换成 `resnet18` 可以跑 ResNet 版本。DINOv2 对
`randomlight` 更鲁棒；VAE 不鲁棒。

如果目标是 random light + trees/shadows，需要在 simulator 中开启随机光照和随机树影，
并在训练时加 scene reload domain randomization：

```text
--scene-reload-alpha 3 --scene-reload-kmin 200
```

评估时使用：

```text
--scene-reload-every 1
```

这样每个 eval episode 都是一个新的随机 layout。DINOv2 在这种条件下能达到约 50% 的
unseen random-layout truncate。冻结 ResNet18 在同样设置下训练到 20k 仍不能完成一圈，
recent episode 基本低于 200 steps；随机树影鲁棒性明显弱于 DINOv2。

### Mountain Track — DINOv2

Mountain 需要两阶段训练：先用 permissive reward 冷启动学会开，再 resume 到更严格的
CTE reward。直接从严格 reward 冷启动会卡在第一个弯。

```bash
# Stage 1: cold start（permissive reward）
python rl/train_loop_vae_sac.py \
  --env-id donkey-mountain-track-v0 \
  --encoder dinov2_vits14 --hidden-size 256 \
  --batch-size 128 --buffer-size 50000 \
  --cte-target 3.5 --max-cte-error 2.5 \
  --min-throttle 0.2 --max-throttle 0.7 \
  --alive-reward 1.5 --speed-reward-weight 0.15 \
  --reward-crash -20 --crash-speed-weight 10 \
  --cte-speed-penalty-weight 0.25 \
  --gradient-steps-cap 1000 --gradient-steps-min 200 \
  --output-dir models/rl_dinov2_mountain_new_v1 \
  --timesteps 60000 --save-replay-buffer --device cuda

# Stage 2: 从 cold-start 最佳 checkpoint resume，使用更严格 CTE penalty
python rl/train_loop_vae_sac.py \
  --env-id donkey-mountain-track-v0 \
  --encoder dinov2_vits14 --hidden-size 256 \
  --batch-size 128 --buffer-size 50000 \
  --cte-target 3.5 --max-cte-error 2.5 \
  --min-throttle 0.2 --max-throttle 0.7 \
  --alive-reward 1.5 --speed-reward-weight 0.12 \
  --reward-crash -20 --crash-speed-weight 10 \
  --cte-speed-penalty-weight 0.30 \
  --gradient-steps-cap 1000 --gradient-steps-min 200 \
  --learning-rate 2e-4 --override-learning-rate \
  --resume-model models/rl_dinov2_mountain_new_v1/<best_checkpoint>.zip \
  --resume-replay-buffer models/rl_dinov2_mountain_new_v1/<best_checkpoint>_replay_buffer.pkl \
  --output-dir models/rl_dinov2_mountain_new_v2 \
  --timesteps 40000 --save-replay-buffer --device cuda
```

---

## 仓库结构

```text
rl/
  train_vae.py                        VAE encoder 训练
  train_vae_sac.py                    shared env / reward / encoder 代码
  train_loop_vae_sac.py               loop + mountain SAC 训练入口
  eval_loop_vae_sac.py                loop + mountain 评估入口，输出 weave 指标：
                                      |dsteer|, cte_std, steer/cte oscillation period
  eval_paired_randomized.py           在同一批 random layouts 上比较模型，避免 layout luck

bc/
  train_bc.py                         BC regression CNN 训练
  eval_bc.py                          BC regression CNN 评估
  train_bc_official_categorical.py    BC categorical 训练
  eval_bc_official_categorical.py     BC categorical 评估

tools/
  collect_sim_frames.py               从 simulator 采集 VAE 图像
  prepare_vae_dataset.py              生成 VAE manifest
  build_vae_cache.py                  生成 cropped uint8 memmap cache
  inspect_loop_replay_throttle.py     从 replay buffer 分析每个 episode 的 throttle/CTE
  test_scene_reload.py                测试 scene reload 是否重新生成 tree/light

docs/
  experiment-log.md                   完整实验历史和设计记录
  session_*.md                        每次 session 的工作笔记
```

---

## 模型库存

所有模型文件都通过 Git LFS 管理。Replay buffer (`.pkl`) 不进入仓库；如需重新生成，
需要训练或 resume 时使用 `--save-replay-buffer`。

### Loop Track

| 目录 | Checkpoint | Encoder | Eval | 说明 |
|---|---|---|:---:|---|
| `rl_loop_vae_sac_fixedlight_v3_h80` | `sac_loop_vae_90000_steps.zip` | VAE | 3/3 | **Primary** |
| `rl_loop_vae_v15` | `sac_loop_vae_112488_steps.zip` | VAE | 5/5 | Co-primary，lap bonus reward |
| `rl_loop_vae_v15` | `sac_loop_vae_132488_steps.zip` | VAE | 2/5 | 仅作参考，圈速快但不稳定 |
| `rl_loop_vae_sac_fixedlight_v2_h64` | `sac_loop_vae_100000_steps.zip` | VAE | 3/3 | Secondary，hidden=64，更居中 |
| `rl_loop_dinov2_v8` | `sac_dinov2_vits14_30000_steps.zip` | DINOv2-S | 3/3 | 光照鲁棒备份 |
| `rl_loop_dinov2_randomtree_v2` | `sac_dinov2_vits14_50000_steps.zip` | DINOv2-S | ~50% rand | random light + trees/shadows domain-randomized 分支 |
| `rl_loop_dinov2_randomtree_crop40_v1` | `sac_dinov2_vits14_130000_steps.zip`, `170000_steps.zip` | DINOv2-S | artifact | crop40 randomtree probe，未提升为部署模型 |
| `rl_loop_vae_sac_resnet_v4_notrees` | `sac_resnet18_50000_steps.zip` | ResNet18 | 3/3 | 备份模型，评估需要 `--encoder-crop-top 40` |

所有 loop VAE 模型共享 encoder：

```text
models/vae_loop_cones_fixedlight_v1/best.pt
```

新的 `--encoder vae` 训练 checkpoint 会沿用历史前缀：

```text
sac_loop_vae_*_steps.zip
```

### Mountain Track

| 目录 | Checkpoint | Encoder | Eval | 说明 |
|---|---|---|:---:|---|
| `rl_dinov2_mountain_v2` | `sac_dinov2_vits14_40000_steps.zip` | DINOv2-S | 5/5 | **Primary** |
| `rl_dinov2_mountain_v2` | `sac_dinov2_vits14_50000_steps.zip` | DINOv2-S | 5/5 | Backup，CTE 更高 |
| `rl_dinov2_mountain_v3` | `sac_dinov2_vits14_40000_steps.zip` | DINOv2-S | 5/5 | 单阶段 cold-start 替代验证 |
| `rl_dinov2_mountain_v1` | `sac_dinov2_vits14_*.zip` | DINOv2-S | - | cold-start branch，为 v2 可复现性保留 |
| `rl_loop_resnet_mountain_v1` | `sac_resnet18_90000_steps.zip` | ResNet18 | 1/3 | 历史参考，已被 DINOv2 pipeline 替代 |

### 其他

| 目录 | 说明 |
|---|---|
| `vae_loop_cones_fixedlight_v1` | fixed-light loop VAE encoder，所有 loop VAE 模型需要它 |
| `vae_raffin_v1` | generated-road VAE encoder，仅用于 single-road baseline |
| `rl_vae_sac_raffin_v1` | generated-road SAC policy，仅用于 single-road baseline |
| `bc_nvidia_slow_006_flip` | 最好的 BC regression 模型，评估需要约 1.8x steering scale |
| `bc_official_categorical_curve_aug_balanced_v1` | 最好的 BC categorical 模型，评估需要约 1.4x steering scale |
