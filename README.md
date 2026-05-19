# Donkey Simulator BC/RL 实验

这个仓库用于在 WSL2 中训练和评估 Donkey Simulator 自动驾驶模型。当前主线是
`generated_road` 场景上的行为克隆（Behavior Cloning, BC），强化学习路线暂未正式开始。

Windows 端运行 Donkey Simulator，WSL2 端运行训练和评估：

```text
WSL2 Python -> 127.0.0.1:9091 -> Windows Donkey Simulator
```

## 当前结论

当前最好模型是 official-style categorical steering：

```text
model: models/bc_official_categorical_curve_aug_balanced_v1/best.pt
scene: generated_road
env id: donkey-generated-roads-v0
decode: argmax
steering_scale: 1.4
steer_smoothing: 0.1
throttle_max: 1.0
```

在随机 `generated_road` 上的最近闭环结果：

```text
1.4 / 0.1:
  episodes = 3
  steps_mean = 2000.0
  steps_min = 2000
  mean_abs_cte = 1.754
  max_abs_cte = 5.325

1.5 / 0.2:
  episodes = 3
  steps_mean = 2000.0
  steps_min = 2000
  mean_abs_cte = 1.901
  max_abs_cte = 5.375

1.4 / 0.2:
  episodes = 3
  steps_mean = 2000.0
  steps_min = 2000
  mean_abs_cte = 2.299
  max_abs_cte = 5.376
```

当前推荐先用 `1.4 / 0.1`。如果肉眼觉得动作太跳，可以退到 `1.5 / 0.2`
或 `1.4 / 0.2`。

固定地图泛化暂时不好：

```text
warren:          119 steps, failed
generated_track: 293 steps, failed
mini_monaco:     508 steps, failed
```

结论是当前模型主要适配 `generated_road` 的视觉分布。要跑其它固定地图，需要录对应地图数据并微调。

## 环境

已验证环境：

```text
OS: Windows + WSL2
Python: 3.11
Conda env: donkey
Simulator: SDSandbox / donkey_sim v25.10.06
gym-donkeycar: 1.3.1
gymnasium: 1.2.3
torch: 2.11.0+cu128
GPU: NVIDIA GeForce RTX 4080 Laptop GPU
```

不要安装 `donkeycar[pc]` 到这个环境。它属于另一套 DonkeyCar/TensorFlow 工作流，
会降级 `numpy`、`protobuf` 等包，和当前 Gymnasium 路线冲突。

## 目录结构

```text
bc/
  train_bc.py                         regression CNN baseline
  eval_bc.py                          regression CNN 闭环评估
  train_bc_categorical.py             旧 21-bin soft-label categorical 实验
  eval_bc_categorical.py              旧 categorical 评估
  train_bc_official_categorical.py    当前主线训练脚本
  eval_bc_official_categorical.py     当前主线评估脚本
  train_bc_gru.py                     暂停的 GRU 实验
  eval_bc_gru.py                      GRU 评估
  inspect_dataset.py                  数据质量检查
  smoke_test.py                       sim 连接/遥测/控制 smoke test

tools/
  extract_cornering_segments.py       从原始 tub 抽取大弯片段
  diag_official_categorical.py        official categorical 诊断

data/
  slow_data_raw/slow_data/            当前 6 条正常慢速 generated_road 数据
  curated_cornering_v1_clean/         第一批清洗后的大弯片段
  curated_cornering_v2_clean/         第二批清洗后的大弯片段
  Cornering data.zip                  第一批大弯原始 zip 备份
  cornorraw.zip                       第二批大弯原始 zip 备份

models/
  bc_nvidia_slow_006_flip/            regression baseline，用于初始化
  bc_official_categorical_curve_aug_balanced_v1/ 当前最好模型
```

`data/` 和 `models/` 被 `.gitignore` 忽略，不应直接提交到普通 Git 仓库。

## 数据

当前训练数据由三部分组成：

```text
data/slow_data_raw/slow_data/road1..road6
data/curated_cornering_v1_clean/corner_*
data/curated_cornering_v2_clean/corner_*
```

合并后约：

```text
samples: 78170
slow normal frames: 72999
curated corner frames: 5171
|angle| >= 0.4: 1242
|angle| >= 0.5: 332
|angle| >= 0.6: 133
|angle| >= 0.7: 42
```

录制建议：

- 使用 `generated_road -> manual driving -> w Rec`。
- 慢速、稳定驾驶，避免撞车、停住后乱打方向。
- 可以按自己的 racing line 切内线，不必严格贴黄线中心。
- 风格要一致。
- 每条路线可以混入少量恢复驾驶。
- 普通训练数据尽量避开 generated road 的交叉/闪烁路线；交叉/闪烁先当 OOD。

检查数据：

```bash
python bc/inspect_dataset.py \
  data/slow_data_raw/slow_data/road1 \
  data/slow_data_raw/slow_data/road2 \
  --sample-images 3
```

重新抽取大弯片段时，先从 zip 解压 raw 数据，再运行：

```bash
python tools/extract_cornering_segments.py \
  --input-root data/cornorraw_raw \
  --output-root data/curated_cornering_v2_clean \
  --trigger-abs-angle 0.4 \
  --max-throttle 0.25 \
  --margin-frames 30 \
  --merge-gap-frames 15 \
  --min-segment-frames 20 \
  --min-trigger-frames 1
```

## 当前主线训练

训练 official-style categorical steering：

```bash
python bc/train_bc_official_categorical.py \
  --data-dir \
    data/slow_data_raw/slow_data/road1 \
    data/slow_data_raw/slow_data/road2 \
    data/slow_data_raw/slow_data/road3 \
    data/slow_data_raw/slow_data/road4 \
    data/slow_data_raw/slow_data/road5 \
    data/slow_data_raw/slow_data/road6 \
    data/curated_cornering_v1_clean/corner_* \
    data/curated_cornering_v2_clean/corner_* \
  --output-dir models/bc_official_categorical_curve_aug_balanced_v1 \
  --steering-bins 11 \
  --steering-min -0.733 \
  --steering-max 0.733 \
  --throttle-bins 8 \
  --throttle-min 0.0 \
  --throttle-max 0.35 \
  --sampler steer-balanced \
  --sampler-weight-max 5.0 \
  --throttle-loss-weight 0.2 \
  --flip-prob 0.5 \
  --learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --batch-size 256 \
  --epochs 200 \
  --patience 15 \
  --num-workers 4 \
  --init-from-regression models/bc_nvidia_slow_006_flip/best.pt
```

可选 memmap 缓存：

```bash
  --memmap-cache-dir data/cache/official_curve_aug_probe
```

缓存会占约 4.2GB，可以随时删除，训练时会重建。

这次训练结果：

```text
best_epoch = 200
best_val_loss = 0.533785
val_steer_acc = 0.8733
val_throttle_acc = 0.6047
steer_mae = 0.0330
steer_rmse = 0.0586
```

## 当前主线评估

Windows simulator 启动后运行：

```bash
python bc/eval_bc_official_categorical.py \
  --env-id donkey-generated-roads-v0 \
  --model models/bc_official_categorical_curve_aug_balanced_v1/best.pt \
  --episodes 3 \
  --max-episode-steps 2000 \
  --exit-scene-between-episodes \
  --scene-reload-delay 3 \
  --steering-scale 1.4 \
  --steering-limit 1.0 \
  --steer-smoothing 0.1 \
  --throttle-min 0.0 \
  --throttle-max 1.0 \
  --sleep 0.02
```

参数含义：

```text
steering_scale: 放大 categorical bin center 输出
steer_smoothing: EMA 平滑，越小反应越快，越大越稳但更迟钝
throttle_max: 评估时允许模型 throttle head 自由输出到训练范围外的上限
exit_scene_between_episodes: 每轮退出场景，强制刷新 generated_road
```

当前 generated road 的 Unity 侧生成不能被 `--seed` 稳定复现。脚本已经支持
`--seed` 和 `--same-seed-each-episode`，也会打印 `obs_checksum`，但实测同一个
seed 加载出的 generated road 仍不一致。参数比较应跑更多随机 episode，看均值和失败率。

## 历史 baseline

Regression CNN + flip baseline 仍保留：

```text
model: models/bc_nvidia_slow_006_flip/best.pt
best val_loss: 0.002038
best epoch: 113
用途:
  1. 作为 regression baseline
  2. 初始化 official categorical 的 CNN encoder/trunk
```

旧评估默认曾是：

```text
steering_scale = 1.8
steering_limit = 0.8
throttle_max = 0.35
```

GRU 路线暂停。离线 loss 很低，但闭环时会产生异常转向尖峰，不作为当前 baseline。

旧 21-bin soft-label categorical 路线也暂停。它用 expectation 解码时容易压尾；
argmax 又受软标签和细 bin 影响，最终不如 official-style hard-bin 方案。

## 指标解释

评估时主要看：

```text
steps: 每轮坚持多少 step
terminated: 是否出界/失败
reward: simulator reward
mean_abs_cte: 平均中心线偏移
max_abs_cte: 最大中心线偏移
```

`CTE` 是 cross-track error。`abs(cte)` 越大越靠边，当前 simulator 通常在
`abs(cte) ~= 8` 附近失败。切内线驾驶时 CTE 不必追求 0，优先级是：

```text
1. 不出界
2. 能通过急弯
3. 尽量减少长时间贴边
```

## 后续工作

短期：

1. 用 `1.4/0.1` 和 `1.5/0.2` 各跑更多随机 generated road，统计失败率。
2. 对失败路线截图/标注，区分正常急弯、交叉闪烁、视觉 OOD。
3. 继续补录 generated road 中模型失败类型的数据，尤其偏离恢复和大弯。

中期：

1. 如果想跑 `warren`、`mini_monaco`、`generated_track`，需要录这些地图的数据并 fine-tune。
2. 尝试预测 future steering 或加入 last_action，减少当前单帧策略的滞后。
3. 做 holdout route 验证，避免随机 val split 过于乐观。

长期：

1. 采集图像数据训练 AE/VAE。
2. 用 latent state 做 SAC 或 TQC。
3. 用 BC 模型作为 RL warm start 或教师策略，对比闭环表现。

## 清理策略

已清理：

```text
data/cache/                         可重建 memmap
data/cornering_data_raw/             可由 zip 恢复
data/cornorraw_raw/                  可由 zip 恢复
data/curated_cornering_v2/           未清洗重复数据
data/generated_road_bc_002/          旧数据
models/*probe*                       试跑模型
models/旧 categorical/GRU/no-flip    失败或过时实验
__pycache__/
```

保留：

```text
data/slow_data_raw/
data/curated_cornering_v1_clean/
data/curated_cornering_v2_clean/
data/*.zip
models/bc_nvidia_slow_006_flip/
models/bc_official_categorical_curve_aug_balanced_v1/
```

## 参考

- Gymnasium Donkey environment: https://github.com/tawnkramer/gym-donkeycar
- Donkey Car autopilot workflow: https://docs.donkeycar.com/guide/train_autopilot/
- Donkey Car Keras model descriptions: https://docs.donkeycar.com/parts/keras/
- Donkey Simulator workflow: https://docs.donkeycar.com/guide/deep_learning/simulator/
