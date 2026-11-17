# Donkey Simulator BC/RL 中文说明

本项目记录了 Donkey Simulator 上的行为克隆（BC）和强化学习（RL）实验。运行方式是
在 WSL2 中启动 Python 训练/评估脚本，连接 Windows 侧正在运行的 Donkey Simulator。

```text
camera image -> (crop) -> encoder -> SAC policy -> steer / throttle
```

详细实验过程、失败分支和设计原因见：[docs/experiment-log.md](docs/experiment-log.md)。

**最新进展（2026-11-17）：** 冻结的纯视觉学生仅使用相机图像和自身指令历史，完成
**24/24 个新随机布局，共 48,000 步、216 圈**；平均绝对 CTE 0.1912、最大 1.6282，
平均速度 2.5168 m/s，无越界或运行时中止。10 个布局出现绝对 CTE ≥1 的偏移，均恢复并跑满。
输入、动作和图像回放审计通过。保留为候选基线；24 次全完成仍不保证可靠性，也未证明优于旧策略，
未替换部署模型。该批次与之前的冒烟和 4 布局筛查单独统计。
详见实验记录 Stage 3.18 和 `logs/visual_student_eval_24layouts_v1/`。

**加长验证（2026-11-17）：** 原版纯视觉学生又完成 **10/10 个新随机光照／树影布局，每局
3,000 步、14 圈，共 30,000 步、140 圈**，无出界或运行时中止。平均绝对 CTE 0.18467、
最大 1.42320（终止阈值 2.0），平均速度 2.54026 m/s。模型及控制参数未变；这批结果与
此前 2,000 步验证分开统计。详见 Stage 3.23 和 `logs/visual_student_eval_10layouts_3000_v1/`。

**实验候选（2026-11-17）：** 已用 6 个新训练布局的 3,000 个决策补充训练，另留 2 个新布局、
1,000 个决策作开发验证，完成一次弯道微调。新开发集弯道转向 MAE 降低 37.8%，原开发集
整体 MAE 上升 1.4%；全部预先记录的离线条件通过，但原开发集转向变化方向不一致次数
从 114 增至 126。随后完成 4 个新布局的同场景闭环对照，两者均 4/4 完成；候选启动阶段
平均绝对 CTE 降低 14.3%，巡航上升 7.6%，整局上升 4.4%，未通过预设改善条件，**不升级**。
上面的 24/24 成绩仍属于原学生，未替换基线。两者继续只用相机和自身指令历史。
详见 Stage 3.20–3.21 和 `logs/visual_student_bend_paired_4layouts_v1/`。

离线交叉回放与路段对齐（Stage 3.22）已完成：候选在两类轨迹上的后续圈动作误差均更高；
完整后续圈 1..8 按 24 个路段等权比较，候选 CTE 平均高 0.01203，四个布局同向。
本轮未启动仿真或训练；暂不做动态信息探针，下一项提案是原版对原版的 A/A 重复性校准。
记录见 `logs/visual_student_cross_replay_v1/`。

14 次大偏移的离线诊断已完成：10 次峰值集中在同一弯道，偏移前学生相对教师的转向
MAE 为 0.0912，单步换成真值几何后仍为 0.0728，下一步优先检查学生控制响应。
这些固定轨迹上的对照尚不能确定物理原因或证明放宽转向限制有效；本阶段未重训、
未运行模拟器或修改运行代码。详见 Stage 3.19 和 `logs/student_excursion_diagnosis_v1/`。

**基线进展（2026-11-13）：已完成大样本随机布局基线、视觉特征探针与真值状态上限验证。** 项目已有可运行的 BC 基线、最早
single generated-road VAE+SAC 基线、fixed-light loop 部署模型、Mountain DINOv2
部署模型，以及 random light + tree shadows 的 domain-randomization 实验记录。
fixed-light loop 上最强模型仍是 track-specific VAE；需要光照/外观泛化时，DINOv2
是目前最好的 frozen encoder。在标准模拟器设置（Time scale 2, FPS limit 120）下，
`rl_loop_dinov2_randomtree_v2` 50k 在 40 个随机布局上平均跑出 6.08 圈/episode（完赛率 31/40 = 78%）。
离线特征探针表明，DINOv2 4x4 池化 patch 特征相比 CLS token 将 CTE 预测误差减半；使用模拟器真值状态（`--encoder sim_state`）训练的诊断策略圈速比视觉策略快 32%（9.18s vs 13.47s，平均 9.42 圈，21/24 完赛），但仍在起点直道存在控制侧引起的转向摆头（weave）。详见
[实验记录 §11–§13](docs/experiment-log.md#11-runtime-revalidation-and-control-diagnostics-2026-11-12)。

---

## 快速开始

### 1. Python 环境

本项目在本机实际使用的环境是 **`rlpy312`（Python 3.12）**。Shell 历史确认
曾在激活该环境后运行 `rl/eval_loop_vae_sac.py`。原来的 `rl311` / Python 3.11
示例已由下面的环境说明替代；项目仍需使用时，应保留 `rlpy312`。

本机训练或评估前，直接激活已有环境：

```bash
conda activate rlpy312
```

在新机器上初始化时，再创建环境并安装依赖：

```bash
conda create -n rlpy312 python=3.12
conda activate rlpy312
# 先按本机 CUDA 版本安装 PyTorch（见下），再执行：
pip install -r requirements.txt
```

`requirements.txt` 固定了当前可用环境的版本，包括第 2 步中 upstream gym-donkeycar 的 commit。

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

**这两个模型文件不在仓库和 Git LFS 中**（从未提交过，当前工作目录里也没有），
运行此命令前需要先找回或重新训练。

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
  --batch-size 64 --buffer-size 60000 \
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
  --output-dir models/rl_dinov2_mountain_new_v2 \
  --timesteps 40000 --save-replay-buffer --device cuda
```

---

## 评估记录与续训校验

两个 loop 评估入口默认在 `logs/` 下新建目录，保存 `metadata.json`、`results.json`、
`episodes.csv` 和起始画面。元数据记录编码器实际权重的哈希，`source/` 保留运行源码。
`--output-dir` 可指定一个尚不存在的目录；
`--save-telemetry` 额外保存逐步动作、CTE、速度、圈数、终止原因和时间戳。
`--capture-tail-steps 120` 会在内存中保留每次结束前120帧（20Hz下约6秒），
同时适用于失败和跑满的episode，并自动启用遥测。结束后写出PNG、GIF和按步号索引。
捕获画面是实际编码器输入的副本，避免读取稍后到达的实时相机画面。
遥测补充位姿、速度向量及客户端策略、编码器、整步耗时。
映射后的转向指令没有Unity执行确认；整步耗时包含等待遥测和编码，不等于网络RTT。

分析已完成的一轮评估：

```bash
python tools/analyze_control_failures.py logs/<run_directory>
```

输出每次运行的时序图、画面拼图（有捕获时）和JSON/CSV统计。
`tools/verify_policy_alignment.py logs/<run_directory>` 离线重放失败 episode 的尾部画面，
检查策略能否复现记录的动作；加 `--brightness-gains 0.9 1.1` 可做离线 RGB 增益敏感性探针。
若要检查同一布局内的重复性，可通过 `--models` 给同一checkpoint指定两个不同标签；
这些标签代表重复运行，不是两个不同训练模型。

无穷大的振荡周期保存为 JSON `null`。异常会保留已完成结果，并标记失败。
Paired 评估从首个布局开始 reload，reload 超时直接失败，不复用旧布局计数；
`--order-seed` 控制模型测试顺序。布局编号仅标识本轮加载，不能用于重建随机场景。

Paired 汇总的最后一段给出布局间的不确定性。主指标是每个 episode 的圈数
（步数上限固定，出界即停），带 bootstrap 95% 区间；其后是每个模型完成率的 95% Wilson 区间、
平均 episode 步数的 bootstrap 区间，以及每对模型的逐布局圈数差和
“只有一方完成”的布局上的精确符号检验。
同样的数字保存在 `results.json` 的 `statistics` 里；
`python tools/summarize_paired_eval.py logs/<run_directory>` 可对旧的或中途失败的运行重新计算。
12 个布局时 8/12 的区间约为 39%–86%，比较模型前请用 30 个以上布局。

```bash
python rl/eval_loop_vae_sac.py --episodes 5 --max-episode-steps 3000 \
  --output-dir logs/loop_fixedlight_check --save-telemetry

python rl/eval_paired_randomized.py --encoder dinov2_vits14 \
  --model-dir models/rl_loop_dinov2_randomtree_v2 --steps 50000 \
  --layouts 12 --max-episode-steps 2000 --save-telemetry
```

Loop 评估默认使用 VAE 90k 主模型。Simulator 的光照/树木设置仍需匹配模型。
Simulator 菜单里的 Time scale 和 FPS limit 也会影响结果：训练和评估统一使用
Time scale 2 + FPS limit 120（约快 1.9 倍，见 experiment-log §12）。评估的 metadata
和训练的 run config 会把 simulator 保存的设置记录为 `simulator_saved_settings`。
新的 loop 训练 checkpoint 内保存编码器、裁剪和动作配置，评估/续训不兼容时会拒绝运行；
历史模型没有这些元数据，会提示无法完成校验。预训练编码器目前只标识模型家族，
尚未固定其上游源码与权重版本。

Loop 训练在 `run_config.json` 中保存请求配置和实际配置（续训到同一目录时写入
`run_config_from_<steps>_steps.json`，并追加而不是覆盖 `monitor.csv`）。续训的 `--timesteps` 是
**追加步数**。SAC 算法参数继承 checkpoint；显式传入不同的 batch、buffer、网络宽度等
会报错，避免静默忽略。学习率通过 `--override-learning-rate` 修改，并持久保存在模型中。
动态梯度步数仅用于按 episode 训练。

续训时，replay buffer 攒够 `--resume-warmup-steps` 条样本（默认 2000，0 为关闭）之前
不做 SAC 更新，期间由加载的策略驾驶。否则不带 buffer 续训时，第一个很短的 episode
就会被重复采样 `--gradient-steps-min` 次。该默认值只是起点，尚未在模拟器中调过。

使用 `--save-replay-buffer` 时，策略检查点按计划保存，buffer 推迟到下一个 episode
结束时保存，避免控制环在 episode 中途停顿。buffer 文件沿用检查点的步数编号，
并包含该 episode 剩余的样本。

新保存的 replay buffer 配有 `.pkl.json` 配置和哈希记录。不允许在奖励、终止条件、
编码器或动作配置变化后复用旧 buffer；例如上面的 Mountain 第二阶段改用空 buffer。
历史实验曾混合新旧奖励，因此修正后的续训方案需要重新验证，不能视作历史结果的精确复现。
历史 buffer 无元数据时，可用 `--allow-legacy-replay-buffer` 明确接受无法校验的兼容性；
该选项不能绕过已知的配置冲突。这些续训校验适用于 `train_loop_vae_sac.py`，
最早的 generated-road `train_vae_sac.py` 入口仍为 legacy trainer。

在 `rlpy312` 中运行无需模拟器的回归测试：

```bash
python -m pytest -q
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
  experiment.py                       参数/检查点/replay 校验，以及评估记录
                                      （JSON/CSV、遥测、尾部画面捕获）
  track_state.py                      --encoder sim_state：用 simulator 真值（车在路上的位置、
                                      车速、前方道路）作观测；仅用于诊断
  eval_vae_sac.py                     legacy single generated-road 评估
  vae.py, vae_dataset.py              ConvVAE 模型与 memmap 数据集

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
  analyze_control_failures.py         从评估记录生成时序图、画面拼图和统计
  verify_policy_alignment.py          离线把捕获画面重放过 encoder + policy
  summarize_paired_eval.py            paired 评估结果的置信区间 / 配对检验
  probe_state_from_features.py        在未见过的布局上，从编码器特征（CLS / patch token）
                                      线性回归 CTE 和横向漂移；使用评估捕获的画面
  fit_grid_projection.py              为 --encoder dinov2_vits14_grid 拟合 PCA 投影
  fit_track_centerline.py             从遥测数据恢复赛道中心线（供 sim_state 使用）
  probe_teacher_action.py             离线探针：检验图像特征 + 运动状态能否预测真值 teacher 的动作
  probe_depth_seg.py                  在 sim 画面上跑深度 + Cityscapes 分割探针
  extract_cornering_segments.py       BC：从 tub 中截取弯道片段
  diag_official_categorical.py        BC：categorical 模型诊断

tests/                                无需模拟器的回归测试（python -m pytest -q）

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
| `rl_loop_sim_state_ts2_sdiff01_v1` | `sac_sim_state_60000_steps.zip` | `sim_state` | **24/24** rand | **真值状态教师模型**（`--max-steering-diff 0.1`，`--track-centerline models/track_centerline/generated_track.npz`）：24 个随机树影/光照布局 100% 跑满（216 圈，`mean_abs_cte=0.158`，`max_abs_cte=0.770`；见 `docs/experiment-log.md` §13） |

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
| `vae_raffin_v1` | generated-road VAE encoder，仅用于 single-road baseline。**不在仓库/LFS 中。** |
| `rl_vae_sac_raffin_v1` | generated-road SAC policy，仅用于 single-road baseline。**不在仓库/LFS 中。** |
| `bc_nvidia_slow_006_flip` | 最好的 BC regression 模型，评估需要约 1.8x steering scale |
| `bc_official_categorical_curve_aug_balanced_v1` | 最好的 BC categorical 模型，评估需要约 1.4x steering scale |
