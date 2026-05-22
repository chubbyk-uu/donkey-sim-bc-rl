# Donkey Simulator BC/RL 实验

WSL2 中训练和评估 Donkey Simulator 自动驾驶模型。**当前主线是 Raffin-style VAE + SAC**；
BC categorical 留作历史 baseline 和对照。

Windows 端运行 Donkey Simulator，WSL2 端运行训练和评估：

```text
WSL2 Python -> 127.0.0.1:9091 -> Windows Donkey Simulator
```

## 当前结论

### RL 主线 (VAE + SAC)

完整 pipeline：

```text
image (120x160) -> crop margin_top=40 -> VAE encoder (frozen) -> latent z (512)
[z; last 20 actions (steer, throttle) flat] -> MLP SAC policy
```

#### VAE (`models/vae_raffin_v1/best.pt`)

```text
architecture: 4-stride conv encoder + symmetric decoder
input: 80x160 RGB
z_size: 512, beta: 1.0, kl_tolerance: 0.5
train data: 142634 frames from generated_road
epochs: 10
best val: loss 295.96, r_loss 39.90, kl_loss 256.07 (≈ kl floor 0.5 * 512)
```

#### SAC v1 (`models/rl_vae_sac_raffin_v1/final_model.zip`)

主要超参（已全部是脚本默认值）：

```text
env id:              donkey-generated-roads-v0
max_steering:        1.0
max_steering_diff:   0.15
min_throttle / max:  0.4 / 0.6
n_command_history:   20 (stored raw, matching Raffin reference)
max_cte_error:       2.0
reward:              alive(1.0) + 0.1 * (thr/max_thr); crash = -10 - 5 * norm_thr
train_freq:          (1, "episode")
gradient_steps:      -1               # = episode_timesteps
gradient_steps_cap:  600              # 每次训练 ≤ 600 次 update
max_episode_steps:   3000             # TimeLimit 截断长 episode
learning_starts:     300
buffer_size:         30000
batch_size:          64
ent_coef:            auto_0.1
```

> **Note**：这次 SAC v1 baseline 实际跑的时候 **没传** `--gradient-steps-cap`（当时默认是
> `None`，即无 cap）。所以那次跑的有效 cap 是 3000（max_episode_steps），不是 600。
> 现在脚本默认已改为 600 ，下一次跑就是预期 schedule。两者训练曲线接近，差异在 ~5%
> n_updates 范围内（28781 vs 22678），不影响这次 baseline 的结论。

#### 训练 schedule 设计（关键）

**核心规则**：每次训练的 update 次数 = `min(episode_length, 600)`。
即 `gradient_steps=-1`（动态取 ep_len）+ `gradient_steps_cap=600`（封顶 600）。

```text
ep_len = 36   ->  36 次 update
ep_len = 300  ->  300 次 update
ep_len = 600  ->  600 次 update
ep_len = 1500 ->  600 次 update     (capped)
ep_len = 3000 ->  600 次 update     (capped; ep 同时被 TimeLimit 截断)
```

设计意图：

- **早期** ep_len 短 → 少更新，避免在乱开数据上过拟
- **晚期** ep_len 长 → 600 次封顶，避免单次训练几十秒卡顿
- 在 ep_len ≤ 600 区间内保持 1:1 ratio，driving:training wall-time 约 **3:1**
  （~10Hz sim + ~30ms/update）

要禁用 cap（纯 1:1）：传 `--gradient-steps-cap 0`。要换成 Raffin 原版固定 600（无论 ep 多长）：
传 `--gradient-steps 600 --gradient-steps-cap 0`。

#### 调参教训（已写入 memory）

1. **schedule 必须 episode-based**。step-based 的 `train_freq` 会落到 episode 中间触发训练；
   SB3 做 gradient updates 时 sim 仍在跑，车带着上一帧 throttle 继续冲，next env.step()
   时已经 cte 超界。episode 模式下 env.reset() 把车 teleport 回起点 + brake 后才训练。
2. **gradient_steps = -1（正比于 ep_len）** 比固定值好。早期 ep_len ≈ 36 用固定 300 会出现
   4s 驾驶 + 10s 训练（1:3，体感卡）。动态正比让早期"少更新"晚期"多更新"。
3. **command_history 存 raw 值**（未归一化），与 Raffin 上游一致。
4. **max_cte_error = 2.0** 是 Raffin 默认。车 reset 在 cte=0（右车道中心），不会立刻终止。

#### 训练曲线（30000 步 / 58 分钟 / RTX 4080 Laptop）

```text
step 1245:   ep_len 39   |steer| 0.85 |delta| 0.30     # 早期乱打方向
step 7677:   ep_len 113  |steer| 0.19 ent 0.019         # 蛇形消除
step 16756:  ep_len 182  rew 185
step 25161:  ep_len 252  rew 260
step 30464:  ep_len 303  rew 316  max episode 1436 步
```

#### 评估结果

固定训练 track 是非环形道路。训练后期最后 10 个 episode 的长度为：

```text
1415, 1306, 1426, 1436, 342, 1402, 1071, 1394, 1421, 1417
```

其中至少 7 个 episode 达到 1390+ step，确认已跑完整条路；342 step 是明显失败样本。

切到新生成的 procedural road（含交叉口）后 7 次评估：

```text
ep 1: 408    crash
ep 2: 1375   crash (蒙对所有交叉口)
ep 3: 1380   crash
ep 4: 542    crash
ep 5: 414    crash
ep 6: 542    crash
ep 7: 1375   crash
mean steps: 862, ≥1000 steps: 3/7
```

失败模式是 **交叉口随机猜方向**（约 50% 蒙对）。VAE 训练数据未包含交叉口 → 交叉口处 latent OOD；
即便 VAE 修好，cte-based reward 在交叉口本质 ambiguous。要做交叉口需要：补 VAE 数据 → 重训 VAE
→ 重训 SAC → waypoint conditioning。

#### Resume 注意

本次启动没加 `--save-replay-buffer`，buffer 已丢。从 `final_model.zip` 直接 resume 仍可，
SAC critic 前 1-2k 步会轻度退化。今后启动**默认就该加**：

```bash
--save-replay-buffer --save-final-replay-buffer
```

#### 继续训练实验（2026-05-22）

原 `0.4~0.6 / max_steering_diff=0.15` 模型已经接近当前 generated_road 设置下的局部上限。
继续训练和单独放宽 steering diff 都没有带来明确提升。

`models/rl_vae_sac_raffin_diff020_v1/`（已删除）：

```text
resume:              models/rl_vae_sac_raffin_v1/final_model.zip
throttle:            0.4 / 0.6
max_steering_diff:   0.2
train steps:         16169
train lens:          1453, 1370, 1408, 1413, 1391, 823, 1371, 1383, 1385, 1408, 1360, 1404
>=1390:              6/12
>=1000:              11/12
eval:                202, 1393, 1386
```

结论：`diff=0.2` 有两集接近跑完，但出现 202 步早崩，不比 `diff=0.15` 主模型稳定。

`models/rl_vae_sac_raffin_resume_010k_v1/`：

```text
resume:              models/rl_vae_sac_raffin_v1/final_model.zip
throttle:            0.4 / 0.6
max_steering_diff:   0.15
train steps:         10585
train lens:          1406, 1405, 635, 1387, 1393, 1395, 165, 1405, 1394
>=1390:              6/9
>=1000:              7/9
```

结论：成功率和原模型接近，但仍有 635 / 165 这类短失败；继续同配置训练的边际收益很小。
当前问题更像是 generated_road 非环形、终点信号缺失、随机交叉/闪烁 OOD、VAE 只覆盖 generated_road
视觉域共同造成的局部瓶颈，不是单纯训练步数不足。

### 下一条 RL 主线：generated_track 专用 VAE + SAC

generated_track 和 generated_road 视觉环境差异很大，**不要把两个域的图像混在同一个 VAE 里训练**。
如果 RL 切到 generated_track，应单独采集 generated_track 图像，训练专用 VAE，再从零训练该赛道的 SAC。

采图脚本：

```bash
python tools/collect_sim_frames.py \
    --output-dir data/vae_raw/generated_track_pid_kp7_kd20_t012
```

`tools/collect_sim_frames.py` 的默认采图配置已经设为当前手调可用参数：

```text
env_id:              donkey-generated-track-v0
action_mode:         cte-pid
frames:              30000
max_episode_steps:   3000
pid_kp / kd / ki:    7 / 20 / 0
throttle:            0.12
steer_limit:         1.0
```

注意：

- 这个脚本会通过 gym-donkeycar 建立 WSL 客户端车，并用外部 CTE PID 控制车；sim UI 里的内置 PID
  没有通过 gym TCP API 暴露给 WSL。
- 采图只用于 VAE，动作质量不需要达到 BC 训练标准；关键是覆盖 generated_track 的道路、弯道、边界和偏离视角。
- 先录 30000 帧（约 25-30 分钟）。如果 generated_track VAE 重建或 SAC 学习不稳，再追加到 50000 帧。
- 之后用 `tools/prepare_vae_dataset.py` 和 `tools/build_vae_cache.py` 对该目录单独建 manifest/cache，
  训练 `models/vae_generated_track_v1/`，再训练 `models/rl_vae_sac_generated_track_v1/`。

### BC baseline（历史）

历史主线，留作对照。**闭环最稳的是 regression CNN baseline** `models/bc_nvidia_slow_006_flip/best.pt`
（实测比 categorical 路线更稳）：

```text
val_loss = 0.002038
eval:  steering_scale=1.8, steering_limit=0.8, throttle_max=0.35
```

categorical 路线 `models/bc_official_categorical_curve_aug_balanced_v1/best.pt`（decode=argmax）
有完整的闭环 sweep 数据，但实测不如 regression 稳：

```text
1.4 / 0.1:  steps_mean 2000, mean_abs_cte 1.754   # categorical 内最好的组合
1.5 / 0.2:  steps_mean 2000, mean_abs_cte 1.901
1.4 / 0.2:  steps_mean 2000, mean_abs_cte 2.299
```

两条路线跨地图泛化都差（warren / generated_track / mini_monaco 全 fail < 600 步），
主要适配 generated_road 的视觉分布。

## 环境

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

不要安装 `donkeycar[pc]`，它属于另一套 TensorFlow 工作流，会降级 numpy/protobuf 等包。

## 目录结构

```text
rl/                                  RL 主线
  vae.py                              Raffin ConvVAE 架构 + loss
  vae_dataset.py                      memmap dataset 读取
  train_vae.py                        VAE 训练
  train_vae_sac.py                    frozen VAE + SAC 训练（主线入口）
  eval_vae_sac.py                     SAC 闭环评估

bc/                                  BC 历史 baseline
  train_bc.py                         regression CNN baseline
  eval_bc.py                          regression CNN 闭环评估
  train_bc_official_categorical.py    BC 最优 categorical 训练
  eval_bc_official_categorical.py     categorical 评估
  inspect_dataset.py                  数据质量检查
  smoke_test.py                       sim 连接/遥测 smoke test

tools/
  build_vae_cache.py                  uint8 memmap cache 构建
  prepare_vae_dataset.py              扫描图像生成 train/val manifest
  collect_sim_frames.py               WSL 客户端车自动采集 VAE 图像
  extract_cornering_segments.py       从原始 tub 抽取大弯片段（BC 用）
  diag_official_categorical.py        BC categorical 诊断

data/
  vae/                                VAE 训练 cache（manifest + uint8 memmap）
  slow_data_raw/slow_data/            BC 训练用 6 条正常慢速 generated_road
  curated_cornering_v1_clean/         BC 训练用第一批清洗大弯片段
  curated_cornering_v2_clean/         BC 训练用第二批清洗大弯片段
  *.zip                               大弯原始备份

models/
  vae_raffin_v1/                      VAE checkpoint（RL 主线）
  rl_vae_sac_raffin_v1/               SAC checkpoint（RL 主线）
  bc_nvidia_slow_006_flip/            regression baseline，BC categorical 初始化用
  bc_official_categorical_curve_aug_balanced_v1/  BC 最好模型
```

`data/` 和 `models/` 被 `.gitignore` 忽略，不提交。

## VAE 训练（主线）

1. 准备 manifest（扫描原始图像，分 train / val）：

```bash
python tools/prepare_vae_dataset.py
```

2. 构建 uint8 memmap cache（占几个 GB，可重建）：

```bash
python tools/build_vae_cache.py
```

3. 训练 VAE：

```bash
python rl/train_vae.py \
    --cache-dir data/vae/cache_raffin \
    --output-dir models/vae_raffin_v1 \
    --epochs 10
```

## SAC 训练（主线）

Windows simulator 启动后：

```bash
python rl/train_vae_sac.py \
    --timesteps 30000 \
    --save-replay-buffer --save-final-replay-buffer \
    --device cuda
```

默认 schedule 已是 episode-based + `gradient_steps=-1`，跑出来就是 baseline 的设置，
schedule 相关 flag 不用传。

Resume：

```bash
python rl/train_vae_sac.py \
    --resume-model models/rl_vae_sac_raffin_v1/final_model.zip \
    --resume-replay-buffer models/rl_vae_sac_raffin_v1/final_replay_buffer.pkl \
    --timesteps 30000 \
    --save-replay-buffer --save-final-replay-buffer \
    --device cuda
```

闭环评估：

```bash
python rl/eval_vae_sac.py \
    --model models/rl_vae_sac_raffin_v1/final_model.zip \
    --episodes 10
```

要测泛化：关掉 simulator 窗口再开（强制 generated_road 重新生成），再跑 eval。

## BC 训练（历史 baseline）

### 数据

```text
data/slow_data_raw/slow_data/road1..road6        (72999 frames)
data/curated_cornering_v1_clean/corner_*
data/curated_cornering_v2_clean/corner_*         (5171 curated corner frames)
```

合并后约 78170 samples（|angle|≥0.4: 1242，≥0.6: 133，≥0.7: 42）。

录制建议：

- `generated_road -> manual driving -> w Rec`
- 慢速、稳定驾驶
- 可切内线，不必严格贴黄线（cte=0 是右车道中心而非黄线）
- 每条路线可混入少量恢复驾驶
- 避开交叉 / 闪烁路线，先当 OOD

检查数据：

```bash
python bc/inspect_dataset.py \
  data/slow_data_raw/slow_data/road1 \
  data/slow_data_raw/slow_data/road2 \
  --sample-images 3
```

重新抽取大弯片段（先从 zip 解压 raw 数据）：

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

### 训练 regression CNN（闭环最稳）

```bash
python bc/train_bc.py \
  --data-dir \
    data/slow_data_raw/slow_data/road1 \
    data/slow_data_raw/slow_data/road2 \
    data/slow_data_raw/slow_data/road3 \
    data/slow_data_raw/slow_data/road4 \
    data/slow_data_raw/slow_data/road5 \
    data/slow_data_raw/slow_data/road6 \
    data/curated_cornering_v1_clean/corner_* \
    data/curated_cornering_v2_clean/corner_* \
  --output-dir models/bc_nvidia_slow_006_flip \
  --flip-prob 0.5 \
  --epochs 200
```

结果：`val_loss = 0.002038, best_epoch = 113`。同一份数据也用来初始化 categorical 路线的 CNN encoder。

### 训练 categorical（替代尝试）

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
  --steering-min -0.733 --steering-max 0.733 \
  --throttle-bins 8 \
  --throttle-min 0.0 --throttle-max 0.35 \
  --sampler steer-balanced \
  --sampler-weight-max 5.0 \
  --throttle-loss-weight 0.2 \
  --flip-prob 0.5 \
  --learning-rate 1e-4 --weight-decay 1e-4 \
  --batch-size 256 --epochs 200 --patience 15 \
  --num-workers 4 \
  --init-from-regression models/bc_nvidia_slow_006_flip/best.pt
```

可选 memmap 缓存：`--memmap-cache-dir data/cache/official_curve_aug_probe`（占约 4.2GB，
可随时删，训练时会重建）。

结果：

```text
best_epoch    = 200
val_loss      = 0.5338
val_steer_acc = 0.8733
steer_mae     = 0.0330
```

### 评估 regression（闭环最稳）

```bash
python bc/eval_bc.py \
  --env-id donkey-generated-roads-v0 \
  --model models/bc_nvidia_slow_006_flip/best.pt \
  --episodes 3 \
  --max-episode-steps 2000 \
  --exit-scene-between-episodes \
  --scene-reload-delay 3 \
  --steering-scale 1.8 \
  --steering-limit 0.8 \
  --throttle-max 0.35 \
  --sleep 0.02
```

### 评估 categorical（替代尝试）

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
  --throttle-min 0.0 --throttle-max 1.0 \
  --sleep 0.02
```

参数说明：

```text
steering_scale: 放大模型输出（categorical 也对 bin center 起作用）
steer_smoothing: EMA 平滑（小=快、大=稳）
throttle_max: 评估时允许 throttle head 输出到训练范围外的上限
exit_scene_between_episodes: 每轮 exit 场景，强制刷新 generated_road
```

Unity 侧 generated_road 生成不能被 `--seed` 稳定复现（脚本有 `--seed` 和
`--same-seed-each-episode` 也打印 `obs_checksum`，但实测同 seed 加载的路仍不一致）。
参数比较应跑多个随机 episode，看均值和失败率。

### BC 其他历史路线

- **GRU**（已删脚本）：离线 loss 低但闭环异常转向尖峰。
- **旧 21-bin soft-label categorical**（已删脚本）：expectation 解码压尾，argmax 受软标签影响，
  不如 11-bin hard-bin。
- **9-bin + sampler_weight_max=6**：val 全面优于 11-bin v1（loss 0.4566 vs 0.5338，acc 0.8981 vs 0.8733），
  但闭环更差（fail @ 743 / 636 / 1147）。原因：dead zone 拓宽 + sampler clip 把中等角度向激进偏 + OOD
  恢复"晚一点猛一点"。**结论：val_loss / val_steer_acc 不能代理闭环性能**，超参选择不应单看 val。
- **RL: BC CNN feature init SAC**（已删脚本 `rl/train_bc_feature_sac.py` + `rl/donkey_sac_env.py`）：
  曾尝试用 BC regression 的 CNN trunk 初始化 SAC 图像 encoder，效果不如 frozen VAE pipeline。

## 指标解释

评估时主要看：

```text
steps: 每轮坚持多少 step（越长越好）
terminated: 是否出界 / 失败
reward: simulator reward
mean_abs_cte: 平均中心线偏移
max_abs_cte: 最大中心线偏移
```

cte=0 是右车道中心（非两车道黄线）。simulator 通常在 abs(cte)~=8 失败。不必强求 cte=0，优先级：

```text
1. 不出界
2. 能通过急弯
3. 尽量减少长时间贴边
```

## 后续工作

短期：

1. 采集 generated_track 专用 VAE 图像：`tools/collect_sim_frames.py` 默认 PID 参数录 30000 帧。
2. 用 generated_track 图像单独训练 `vae_generated_track_v1`，不要混 generated_road 图像。
3. 用新 VAE 从零训练 `rl_vae_sac_generated_track_v1`，优先利用环形道避免 generated_road 的终点/存活奖励冲突。

中期：

1. 对比 generated_track SAC 和 generated_road SAC 的稳定性、蛇形程度、速度上限。
2. 在 generated_track VAE latent 上尝试 TQC。
3. 如果回到 generated_road，再考虑 completion proxy / waypoint conditioning 解决非环形终点和交叉口问题。

长期：

1. 跨地图泛化（warren / mini_monaco 等分别采图，按域训练或 domain randomization）。
2. 多进程并行采样，提高 RL 采样效率。
3. 闭环 TQC、SAC-fD（demo-augmented）等更高级方案。

## 清理策略

已清理：

```text
data/cache/                         可重建 memmap
data/cornering_data_raw/             可由 zip 恢复
data/cornorraw_raw/                  可由 zip 恢复
data/curated_cornering_v2/           未清洗重复数据
data/generated_road_bc_002/          旧数据
models/*probe*                       试跑模型
models/旧 categorical / GRU / no-flip 失败或过时实验
__pycache__/
```

保留：

```text
data/slow_data_raw/
data/curated_cornering_v1_clean/
data/curated_cornering_v2_clean/
data/vae/
data/*.zip
models/bc_nvidia_slow_006_flip/
models/bc_official_categorical_curve_aug_balanced_v1/
models/vae_raffin_v1/
models/rl_vae_sac_raffin_v1/
```

## 参考

- Gymnasium Donkey environment: https://github.com/tawnkramer/gym-donkeycar
- Raffin Donkey RL article: https://medium.com/data-science/learning-to-drive-smoothly-in-minutes-450a7cdb35f4
- Raffin learning-to-drive-in-5-minutes: https://github.com/araffin/learning-to-drive-in-5-minutes
- Donkey Car docs: https://docs.donkeycar.com/
