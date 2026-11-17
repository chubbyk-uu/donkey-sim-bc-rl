# Donkey Simulator BC/RL 中文说明

本项目研究模拟器上的行为克隆、强化学习，以及从真值教师蒸馏得到的纯视觉学生。
Python 客户端在 WSL2 中运行，连接 Windows Donkey Simulator。

[English](README.md) · [文档导航](docs/README.md) · [当前状态](docs/status.md)

## 赛道与主要模型

项目覆盖三类模拟器环境，各分支针对相应赛道训练；最新学生目前只验证了生成闭环赛道。

| 赛道 | 环境 ID | 已做工作 |
| --- | --- | --- |
| 生成式开放道路 | `donkey-generated-roads-v0` | 人工示教 BC、早期 VAE + SAC、路口与跨道路泛化测试 |
| 生成式闭环赛道 | `donkey-generated-track-v0` | 固定光照 VAE + SAC、DINOv2／ResNet、随机光照树影训练、教师与学生 |
| Mountain 山路闭环 | `donkey-mountain-track-v0` | 为上下坡和右车道单独训练 DINOv2 + SAC，也做过 ResNet 尝试 |

保留的主要模型与历史成绩：

| 场景／用途 | 模型及检查点 | 成绩／使用要点 |
| --- | --- | --- |
| 随机光照、树影闭环 | `visual_student_no_motion_v1` + `dinov2_geometry_15d` | 当前纯视觉基线；24/24 跑满 2,000 步，另 10/10 跑满 3,000 步 |
| 固定光照闭环 | `rl_loop_vae_sac_fixedlight_v3_h80` 90k + `vae_loop_cones_fixedlight_v1` | VAE 主模型；后续复测 5/5 跑满 3,000 步，速度 2.826 m/s，平均绝对 CTE 0.311 |
| 固定光照闭环，另一保留版本 | `rl_loop_vae_v15` **112488** + 同一 VAE | 长程评估保留模型；5/5 跑满 3,000 步；后来的 132488 仅供参考 |
| Mountain 山路 | `rl_dinov2_mountain_v2` **40k** | 山路主模型；5/5 跑满 2,000 步，速度 2.093 m/s，平均车道相对绝对 CTE 0.577 |
| 随机光照树影闭环，旧视觉 SAC | `rl_loop_dinov2_randomtree_v2` 50k | 旧 RL 基线；标准设置下复测 31/40 跑满 2,000 步 |
| 光照变化闭环 | `rl_loop_dinov2_v8` 30k | 冻结 DINOv2 + SAC 备份；通过单纯光照变化测试，对树影的鲁棒性不足 |
| 固定光照闭环，ResNet 备份 | `rl_loop_vae_sac_resnet_v4_notrees` 50k | 保留的视觉 SAC 分支；需加 `--encoder-crop-top 40` |
| 开放道路 BC，回归 | `bc_nvidia_slow_006_flip` | Nvidia 风格 CNN；转向倍率 1.8 时，3/3 跑满 2,000 步 |
| 开放道路 BC，分类 | `bc_official_categorical_curve_aug_balanced_v1` | 11 档转向分类、类别均衡；使用转向倍率 1.4、平滑系数 0.1 |
| 真值状态教师 | `rl_loop_sim_state_ts2_sdiff01_v1` 60k | 用于蒸馏和诊断的真值 SAC 参考模型 |

这些是不同赛道、外观、终止条件和步数上限下的历史批次，不能直接当作统一排行榜。
开放道路 BC 的 CTE 边界不同于闭环；山路误差相对 `cte_target=3.5` 计算。
完整权重路径、备份检查点和参数要求见[模型目录](docs/guides/models.md)。

早期开放道路的 **Raffin VAE + SAC** 分支在训练中能接近路线终点，但对另一条生成道路
的路口泛化不稳定。其策略和 VAE 权重已不在 Git/LFS 或工作目录中，
[历史记录](docs/archive/experiments/02-generated-road-and-early-loop.md)仍保留；
上表中的 BC、闭环和山路模型均已入库。

结果来源：[BC](docs/archive/experiments/01-behavioral-cloning.md)、
[固定光照闭环](docs/archive/experiments/03-fixed-light-loop.md)、
[v15 复测](docs/archive/sessions/session_2026-05-27_loop_followup.md)、
[山路](docs/archive/experiments/04-mountain.md)、
[编码器对比](docs/archive/experiments/05-encoders-and-randomization.md)、
[11 月基线复测](docs/archive/experiments/07-vision-and-runtime-probes.md)。

## 各方法的关系

BC 从人工示教学习动作；视觉 SAC 用赛道专用 VAE 或冻结的 DINOv2／ResNet 编码图像，
通过模拟器奖励学习控制。真值教师使用模拟器状态，再向纯视觉学生提供离线动作标签和
可复用的演员网络权重。学生从教师初始化，而不是从旧视觉 SAC 演员网络初始化。
输入和训练细节见[学生结构与来源](docs/guides/student.md)。

## 当前重点：纯视觉学生

**随机光照闭环的学生阶段已完成，保留 `visual_student_no_motion_v1`。**
学生只用图像推算的道路几何及自身已发出的指令历史，不输入真实速度、位置、CTE、地图，
也没有使用轮编码器或 IMU。

| 独立评测批次 | 完成数 | 每局步数 | 总圈数 | 平均 / 最大绝对 CTE |
| --- | --- | --- | --- | --- |
| 24 个新随机外观布局 | 24/24 | 2,000 | 216 | 0.1912 / 1.6282 |
| 另外 10 个新布局 | 10/10 | 3,000 | 140 | 0.1847 / 1.4232 |

两批均无出界终止或运行时中止。出界终止条件为 `abs(CTE) > 2.0`，CTE 是模拟器单位，
不是米。加长批次平均速度 2.540 m/s，评估约 13 分半。随机布局改变光照、树木和阴影，
底层赛道几何固定；这些结果不代表已验证陌生赛道。

弯道微调候选**不升级**：首段改善未转化为整局配对 CTE 改善。后续训练、A/A 校准、
传感器扩展均未排期。详见[状态与决策](docs/status.md)和[入库的评测摘要](docs/results/README.md)。

## 快速运行

在仓库根目录执行。现有环境为 `rlpy312`（Python 3.12）；新机器先完成
[安装与连接](docs/guides/setup.md)，包括 PyTorch、锁定版本的 upstream SDK 和 Git LFS。

```bash
conda activate rlpy312
git lfs pull
export DONKEY_SIM_HOST="$(ip route | awk '/default/ {print $3}')"
export DONKEY_SIM_PORT=9091
```

### 纯视觉学生：随机光照闭环

Windows 模拟器使用生成闭环赛道，设置为 **timeScale 2、FPS 120、随机光照开、树木开、
锥桶关、useSeed 关**。元数据记录的是保存的菜单值，无法独立确认运行中的引擎状态。

```bash
python rl/eval_visual_student.py \
  --student models/visual_student_no_motion_v1/student.pt \
  --geometry-head models/dinov2_geometry_15d/proj15_lags_ridge.pt \
  --host "$DONKEY_SIM_HOST" --port "$DONKEY_SIM_PORT" \
  --episodes 3 --max-episode-steps 2000 \
  --output-dir "logs/student_smoke_$(date +%Y%m%d_%H%M%S)"
```

每次使用新输出目录。学生入口默认启用低抖动同步、TCP_NODELAY 和 100 ms 观测等待超时。
评分、终止与日志可读真值，学生决策网络不读取。首次运行可能下载／缓存 DINOv2。
加长复测使用 `--episodes 10 --max-episode-steps 3000`。学生和几何头两份 `.pt` 均不可缺少，
旧 SAC `.zip` 不能替代学生模型。

### VAE + SAC：固定光照闭环

关闭随机光照和树木，锥桶外观应匹配固定光照检查点。VAE 编码器与 SAC 策略都需要。
运行保留的主模型：

```bash
python rl/eval_loop_vae_sac.py \
  --env-id donkey-generated-track-v0 \
  --encoder vae --vae-model models/vae_loop_cones_fixedlight_v1/best.pt \
  --model models/rl_loop_vae_sac_fixedlight_v3_h80/sac_loop_vae_90000_steps.zip \
  --episodes 3 --max-episode-steps 2000 \
  --output-dir "logs/fixedlight_smoke_$(date +%Y%m%d_%H%M%S)"
```

使用 v15 长程评估版本时，将 `--model` 换成
`models/rl_loop_vae_v15/sac_loop_vae_112488_steps.zip`，并改为
`--episodes 5 --max-episode-steps 3000`；VAE 和外观设置相同。

### DINOv2 + SAC：Mountain 山路

打开 Mountain 场景。右车道参考值为 CTE 3.5，上坡使用保留的油门范围 0.2–0.7。
这个模型使用 DINOv2，无需 VAE：

```bash
python rl/eval_loop_vae_sac.py \
  --env-id donkey-mountain-track-v0 --encoder dinov2_vits14 \
  --model models/rl_dinov2_mountain_v2/sac_dinov2_vits14_40000_steps.zip \
  --cte-target 3.5 --max-cte-error 2.5 --min-throttle 0.2 --max-throttle 0.7 \
  --episodes 5 --max-episode-steps 2000 \
  --output-dir "logs/mountain_smoke_$(date +%Y%m%d_%H%M%S)"
```

BC 使用独立入口，见[回归与分类运行命令](docs/archive/experiments/01-behavioral-cloning.md)。
其它视觉 SAC 分支见[各模型的命令和设置](docs/guides/models.md#legacy-evaluation-commands)。

## 按需阅读

| 需求 | 文档 |
| --- | --- |
| 安装环境、连接模拟器、取得模型 | [安装与连接](docs/guides/setup.md) |
| 旧视觉模型、教师与学生的关系 | [学生结构与训练来源](docs/guides/student.md) |
| 固定光照 VAE、Mountain、备份与诊断模型 | [模型目录与命令](docs/guides/models.md) |
| 评估、同布局对照、结果文件说明 | [评估指南](docs/guides/evaluation.md) |
| 同步、超时保护及其范围 | [运行时保护](docs/guides/runtime.md) |
| 历史训练配方与正确续训 | [训练指南](docs/guides/training.md) |
| 历史实验、失败分支和旧会话 | [实验索引](docs/experiment-log.md)、[归档](docs/archive/README.md) |

## 仓库与验证

`rl/` 存放控制器、编码器、环境和评估入口；`bc/` 是早期示教基线；`tools/` 是采集、
训练与诊断；`tests/` 是不连接模拟器的测试；`models/` 保存 Git/LFS 权重；
`docs/` 保存当前指南、精简证据和历史归档。

```bash
python -m pytest tests -q
```

当前基线通过 88 项测试。原始相机数据、回放缓冲区、特征缓存及完整 `logs/` 不随 Git
分发；离线训练需恢复数据，推理可使用已入库权重。DINOv2 骨干与 upstream 源码另外下载，
运行元数据会记录实际张量哈希。
