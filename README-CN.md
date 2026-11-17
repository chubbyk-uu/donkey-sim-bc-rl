# Donkey Simulator BC/RL 中文说明

本项目研究模拟器上的行为克隆、强化学习，以及从真值教师蒸馏得到的纯视觉学生。
Python 客户端在 WSL2 中运行，连接 Windows Donkey Simulator。

[English](README.md) · [文档导航](docs/README.md) · [当前状态](docs/status.md)

## 当前可用基线

**这一阶段已完成，保留 `visual_student_no_motion_v1`。** 学生只用图像推算的道路几何
及自身已发出的指令历史，不输入真实速度、位置、CTE、地图，也没有使用轮编码器或 IMU。

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
