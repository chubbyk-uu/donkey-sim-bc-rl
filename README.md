# Donkey Simulator 行为克隆与强化学习实验

这个仓库用于在 WSL2 中训练 Donkey Simulator 自动驾驶模型。当前主线是行为克隆
（Behavior Cloning, BC）：在 Windows 端运行模拟器和录制数据，在 WSL2 端用
Gymnasium、PyTorch 和 CUDA 训练/评估模型。

当前连接方式：

```text
WSL2 Python -> 127.0.0.1:9091 -> Windows Donkey Simulator
```

## 当前状态

当前可用 baseline：

```text
模型: 单帧 CNN 行为克隆 + 水平翻转增强
权重: models/bc_nvidia_slow_006_flip/best.pt
数据: 6 条 generated_road 慢速人工驾驶路线
默认评估参数:
  throttle_max = 0.35
  steering_scale = 1.8
  steering_limit = 0.8
```

单帧 CNN 原始输出仍需要固定的转向/油门控制校准。这不是基于路线信息的规则控制，
没有使用 CTE、位置、道路几何或未来路线。加入水平翻转增强后，模型对左右偏移的
恢复能力更好，当前不再需要旧 baseline 使用的 `steering_scale=2.4`。

GRU 时序路线暂时暂停。它的离线验证 loss 很低，但 simulator 闭环时真实历史帧会
触发异常大转向；把当前帧重复输入给同一个 GRU 后反而稳定，说明问题在时序动态
学习上。相关脚本保留用于复盘和后续实验。

## 环境需求

推荐环境：

```text
OS: Windows + WSL2
Python: 3.11
Conda env: donkey
GPU: NVIDIA GPU
CUDA: PyTorch CUDA 12.8 build
Simulator: SDSandbox / donkey_sim v25.10.06
```

当前已验证包版本：

```text
Python 3.11.15
gym-donkeycar 1.3.1
gymnasium 1.2.3
stable-baselines3 2.8.0
torch 2.11.0+cu128
GPU: NVIDIA GeForce RTX 4080 Laptop GPU
```

不要在这个环境里安装 `donkeycar[pc]`。它属于另一套 DonkeyCar/TensorFlow
工作流，会拉低 `numpy`、`protobuf` 等包版本，和当前 Gymnasium 版本路线冲突。

## 模拟器设置

Windows 端启动 Donkey Simulator，确认端口：

```text
port: 9091
portPrivateAPI: 9092
```

当前主要使用：

```text
scene: generated_road
env id: donkey-generated-roads-v0
```

注意：

- `generated_road` 是当前模拟器里能手动录制数据的场景。
- WSL2 可以通过 `127.0.0.1:9091` 连接 Windows 模拟器。
- 每次加载 generated road 都可能生成不同路线。
- 随机路线偶尔会出现跑道交叉和画面闪烁。当前训练数据刻意避开了这种情况，所以
  交叉/闪烁失败应标记为 OOD，不应混进普通路线成功率。

## 数据采集

在 Windows simulator 中进入：

```text
generated_road -> manual driving -> w Rec
```

录制建议：

- 慢速、稳定驾驶。
- 可以按自己的 racing line 切内线，不必严格沿黄线中心。
- 风格要一致，不要一会儿沿中心线，一会儿极端切弯。
- 避免撞车、冲出跑道、停住后大幅乱打方向。
- 每条路线可以加入少量恢复驾驶，例如轻微偏左/偏右后回到路面。
- 当前训练数据刻意避开 generated road 的交叉/闪烁路线。

Windows 路径示例：

```text
D:\WSL\slow_data\road1
D:\WSL\slow_data\road2
...
```

复制到 WSL 本地后再训练，避免 `/mnt/d/...` 小文件读取过慢：

```bash
mkdir -p data/slow_data_raw/slow_data
cp -r /mnt/d/WSL/slow_data/road1 data/slow_data_raw/slow_data/road1
```

如果是 zip：

```bash
unzip /mnt/d/WSL/slow_data.zip -d data/slow_data_raw
```

## 数据检查

检查一个或多个数据目录：

```bash
python inspect_dataset.py \
  data/slow_data_raw/slow_data/road1 \
  data/slow_data_raw/slow_data/road2 \
  --sample-images 3
```

`inspect_dataset.py` 默认会输出 `21 bins / [-0.7, 0.7]` 的 angle 直方图，用于检查
categorical steering 的数据分布。

当前工作数据：

```text
data/slow_data_raw/slow_data/road1
data/slow_data_raw/slow_data/road2
data/slow_data_raw/slow_data/road3
data/slow_data_raw/slow_data/road4
data/slow_data_raw/slow_data/road5
data/slow_data_raw/slow_data/road6

records/images: 72999 total
image shape: (160, 120), RGB
overall throttle mean: about 0.095
overall throttle p95: about 0.17
```

## 训练单帧 CNN baseline

当前主力训练命令：

```bash
python train_bc.py \
  --data-dir \
    data/slow_data_raw/slow_data/road1 \
    data/slow_data_raw/slow_data/road2 \
    data/slow_data_raw/slow_data/road3 \
    data/slow_data_raw/slow_data/road4 \
    data/slow_data_raw/slow_data/road5 \
    data/slow_data_raw/slow_data/road6 \
  --drop-head 0 \
  --drop-tail 0 \
  --output-dir models/bc_nvidia_slow_006_flip \
  --history 1 \
  --frame-stride 1 \
  --flip-prob 0.5 \
  --batch-size 128 \
  --num-workers 4 \
  --epochs 160 \
  --patience 10 \
  --learning-rate 0.001 \
  --val-split 0.2
```

当前结果：

```text
best val_loss: 0.002038
best epoch: 113
early stop epoch: 123
model: models/bc_nvidia_slow_006_flip/best.pt
```

历史 no-flip baseline：

```text
best val_loss: 0.001056
best epoch: 118
model: models/bc_nvidia_slow_006_random_split/best.pt
```

说明：这里的验证集是随机切分，不等同于整条路线泛化能力。最终判断仍以 simulator
闭环评估为准。

## 评估单帧 CNN

启动 Windows simulator 后运行：

```bash
python eval_bc.py \
  --model models/bc_nvidia_slow_006_flip/best.pt \
  --env-id donkey-generated-roads-v0 \
  --host 127.0.0.1 \
  --port 9091 \
  --episodes 3 \
  --max-episode-steps 2000 \
  --recreate-env-each-episode \
  --exit-scene-between-episodes \
  --scene-reload-delay 2.0 \
  --sleep 0.0 \
  --throttle-max 0.35 \
  --steering-scale 1.8 \
  --steering-limit 0.8 \
  --device cuda
```

参数含义：

```text
throttle_max: 限制最大油门
steering_scale: 放大模型转向输出，补偿 understeer
steering_limit: 限制最终发送给模拟器的最大转向
exit_scene_between_episodes: 每轮结束后退出场景，强制刷新 generated road
```

当前评估结论：

```text
flip model, 1.8 / 0.8 / 0.35:
  当前默认参数。
  3 条随机 generated road 都跑满 2000 steps。
  eval summary:
    steps_mean = 2000.0
    reward_mean = 1748.77
    mean_abs_cte = 2.190
    max_abs_cte = 5.637
  人工观察没有明显压两侧白线或冲到白线外。

flip model, 2.4 / 0.8 / 0.35:
  转向过度，3 条路线分别只坚持 1665 / 991 / 538 steps。

no-flip model:
2.0 / 0.65 / 0.35:
  更保守，稳定性好，但更容易贴边。

2.4 / 0.8 / 0.35:
  旧默认参数。正常路线更居中，急弯能力更强。
  generated road 交叉/闪烁仍属于 OOD。

1.8 / 0.8 / 0.4:
  观感更快，部分路线更居中，但安全裕度略低。
```

其中格式为：

```text
steering_scale / steering_limit / throttle_max
```

## 评估指标

主要看：

```text
survival_steps: 每轮坚持了多少 step
terminated: 是否出界/失败
reward: simulator reward
mean_abs_cte: 平均偏离中心线距离
max_abs_cte: 最大偏离中心线距离
```

`CTE` 是 cross-track error，表示相对道路中心线的偏移。`cte=0` 接近中心线，
`abs(cte)` 越大越靠近路边。当前 simulator 里失败通常在 `abs(cte) ~= 8`
附近出现。

如果驾驶风格是切内线，CTE 大不一定代表错误。当前目标优先级是：

```text
1. 不出界
2. 能通过急弯
3. 尽量减少长时间贴边
```

## Categorical steering 实验状态

相关文件：

```text
train_bc_categorical.py
eval_bc_categorical.py
```

设计要点：

```text
分箱: 21 bins / [-0.7, 0.7]，bin 宽 ≈ 0.0667
标签: linear soft label（相邻两 bin 线性插值，按 bin centers 计算 pos）
增强: 水平翻转（image 镜像 + angle 取反），仅训练 split
头部: 双 head，steer 21-bin 分类 + throttle 标量回归
损失: 加权 soft CE + 0.1 * MSE(throttle)，class_weight 用 soft label 加权
推理: 期望解码 sum(softmax * bin_centers)，不要 argmax
```

已修复的工程 bug：

```text
- compute_class_weights 现在接收 flip_prob，不再隐式假设 0.5
- DataLoader 加 worker_init_fn 让 random 模块在多 worker 下正确分叉
- tub_summaries 的 angle 统计现在和 sample_indices 一致（filter 之后再 append）
- resume 校验 checkpoint 的 bin_centers 与当前配置一致，不只 num_bins
```

试训发现的问题：

```text
- 默认配置（lr=1e-3, class_weight_max=8.0）训 12 epoch 卡在 val_loss ≈ 1.91
- 关闭 class weight（min=max=1.0）训 10 epoch 仍卡在 ≈ 1.91
- 经验边际分布 p* 的熵 H(p*) ≈ 1.5~2.0，几乎匹配
→ 模型坍缩到"输出经验边际分布，忽略图像"
```

确认可解的两个方向（各 10 epoch 试训均能持续下降）：

```text
方向 A: 降学习率到 1e-4
  CNN 有时间从零发育出图像可分特征
  命令: --learning-rate 1e-4 --class-weight-min 1.0 --class-weight-max 1.0

方向 B: 用 regression baseline 的 best.pt 初始化 CNN encoder + trunk
  跳过 CNN 冷启动；只随机初始化两个 head
  命令: --init-from-regression models/bc_nvidia_slow_006_flip/best.pt
  （脚本里已实现 init_from_regression_checkpoint，映射 features/trunk）
```

结论：categorical 路线本身可行，CNN 冷启动是当前阶段的真正瓶颈，
class weight 不是必需。下次继续做完整训练并和 regression baseline 对比。

## GRU 实验状态

相关文件：

```text
train_bc_gru.py
eval_bc_gru.py
```

测试过的 GRU 模型：

```text
sequence_length = 8
frame_stride = 1
CNN encoder + GRU hidden_size 256 + dense head
```

离线结果：

```text
best val_loss: 9.41e-05
best epoch: 114
```

但闭环测试发现：

- 真实 8 帧历史输入会在轻弯/近直线时触发异常大转向。
- 把当前帧重复 8 次输入给同一个 GRU，尖峰消失并且 survival steps 提升。
- 说明当前 GRU 学到的时序动态不可靠。

结论：GRU 路线暂时暂停，不作为当前 baseline。后续如果继续时序模型，应优先复现
DonkeyCar 官方 LSTM/RNN 风格模型，或尝试 bounded/categorical 输出，而不是继续
当前 GRU。

## 后续工作

下次继续（categorical 收尾）：

1. 全量训练方向 A（lr=1e-4, 不加 class weight，从零训）到收敛或早停。
2. 全量训练方向 B（lr=1e-3，加 `--init-from-regression models/bc_nvidia_slow_006_flip/best.pt`，
   从 regression baseline 初始化 CNN encoder + trunk）到收敛或早停。
3. 对比两次训练的 best val_loss、收敛 epoch、diagnostics.json：
   - 关键指标：`pred_abs_percentiles` 是否接近 `true_abs_percentiles`
   - p99(pred) 接近 0.349 才算尾部学到了
4. 选其中更优的做实车评估，和 regression + flip baseline 做 A/B 对比。

短期：

1. 做更多随机路线评估，并人工标注 normal / crossing-flicker / other anomaly。
2. 继续录制正常非交叉路线，增加轻微恢复驾驶样本。
3. 如果 categorical 的 pred 尾部仍被压缩，再考虑温和 class weight（max≈3）
   或 WeightedRandomSampler；不建议把 class_weight_max 一路调高。

中期：

1. 做整条路线 holdout 验证，避免随机切分验证集过于乐观。
2. 尝试更系统的控制校准：不同速度下的 steering gain。
3. GRU 路线暂停，相关脚本保留用于复盘。后续如果重启时序模型，优先复现
   DonkeyCar 官方 LSTM/RNN 风格，或直接试 categorical 输出的时序模型。

长期：

1. 收集图像数据训练 AE/VAE。
2. 用 latent state 训练 SAC 或 TQC。
3. 将 RL 策略和 BC baseline 做闭环对比。

## 文件说明

```text
smoke_test.py      检查 Gymnasium 环境连接、图像、遥测和控制
inspect_dataset.py 检查 Donkey tub 数据质量
train_bc.py        单帧/帧堆叠 CNN 行为克隆训练
eval_bc.py         单帧 CNN 闭环评估和控制校准
train_bc_categorical.py  categorical steering 行为克隆训练
eval_bc_categorical.py   categorical steering 闭环评估
train_bc_gru.py    GRU 行为克隆实验训练
eval_bc_gru.py     GRU 闭环评估和尖峰诊断
```

## 不上传的数据

以下目录被 `.gitignore` 忽略，不应进入普通 Git 仓库：

```text
data/
models/
runs/
logs/
*.pt
*.pth
```

数据和模型如果需要共享，建议使用 Git LFS、Release artifact、网盘或单独的数据说明。

## 参考

- Donkey Car official autopilot workflow:
  https://docs.donkeycar.com/guide/train_autopilot/
- Donkey Car command documentation:
  https://docs.donkeycar.com/utility/donkey/
- Donkey Car Keras model descriptions:
  https://docs.donkeycar.com/parts/keras/
- Donkey Simulator workflow:
  https://docs.donkeycar.com/guide/deep_learning/simulator/
- Gymnasium Donkey environment:
  https://github.com/tawnkramer/gym-donkeycar
