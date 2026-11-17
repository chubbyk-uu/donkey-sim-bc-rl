# Behavioral cloning

> Historical record. Results, “current” choices and “next” plans below belong to the dated experiments, not the active work queue. See [current status](../../status.md) and [documentation index](../../README.md). Commands assume the repository root; raw `logs/` and `data/` artifacts are local-only unless explicitly noted.

## 1. Behavioral Cloning

BC had two serious branches: continuous regression and official-style categorical
classification. Both branches learned useful generated-road driving, but both had the
same closed-loop symptom: raw steering output was too small and needed an evaluation
time steering gain. Regression was generally more stable by observation, categorical
had cleaner diagnostics and more systematic class balancing, and both were ultimately
weaker than RL.

### 1.1 Regression CNN

The regression branch uses `bc/train_bc.py` and `bc/eval_bc.py`.

Model:

```text
models/bc_nvidia_slow_006_flip/best.pt
```

Architecture:

```text
Nvidia-style CNN
input:       RGB image, optional frame stack
preprocess: crop first 10 image rows inside model
output:      continuous [steering, throttle]
loss:        MSE on steering/throttle
```

Training data:

```text
data/slow_data_raw/slow_data/road1..road6
records/images: about 72,999
style: slow, stable manual driving on generated_road
augmentation: horizontal flip, flip_prob=0.5
```

Training config:

```text
history:        1
frame_stride:   1
batch_size:     128
epochs:         160
patience:       10
learning_rate:  0.001
val_split:      0.2
best_epoch:     113
best val_loss:  0.002038
```

Closed-loop evaluation needed steering scaling because raw regression steering was
under-amplified:

```text
recommended regression eval:
steering_scale = 1.8
steering_limit = 0.8
throttle_max   = 0.35
```

Historical eval command:

```bash
python bc/eval_bc.py \
  --model models/bc_nvidia_slow_006_flip/best.pt \
  --env-id donkey-generated-roads-v0 \
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

Historical result:

```text
flip model, steering_scale / steering_limit / throttle_max = 1.8 / 0.8 / 0.35
3 random generated_road episodes reached 2000/2000 steps
reward_mean:   1748.77
mean_abs_cte:  2.190
max_abs_cte:   5.637
```

Other scale sweeps:

```text
flip model, 2.4 / 0.8 / 0.35:
  over-steered; episodes ended at 1665 / 991 / 538 steps.

no-flip model, 2.0 / 0.65 / 0.35:
  stable but more prone to edge-following.

no-flip model, 2.4 / 0.8 / 0.35:
  older default; stronger cornering, but less forgiving.

flip model, 1.8 / 0.8 / 0.4:
  faster-looking, sometimes more centered, but with less safety margin.
```

Regression lessons:

- Random validation loss was optimistic. The no-flip model had lower val loss, but the
  flip model was better in closed loop.
- Horizontal flip improved recovery symmetry.
- The policy was still a calibrated controller, not pure raw network output: `eval_bc.py`
  multiplies steering by `--steering-scale`.
- In practice this branch felt more stable than categorical, but still lacked robust
  recovery/generalization compared with RL.

### 1.2 Official-Style Categorical

The categorical branch uses `bc/train_bc_official_categorical.py` and
`bc/eval_bc_official_categorical.py`.

Final model:

```text
models/bc_official_categorical_curve_aug_balanced_v1/best.pt
```

Architecture:

```text
Nvidia-style CNN trunk
steering head: categorical class over hard bins
throttle head: categorical class over hard bins
decode: argmax class center
```

Training data:

```text
data/slow_data_raw/slow_data/road1..road6
data/curated_cornering_v1_clean/corner_*
data/curated_cornering_v2_clean/corner_*
total samples: about 78,170
curated corner frames: about 5,171
```

Intermediate model:

```text
models/bc_official_categorical_9bin_sampler6_300/best.pt
```

That branch used 9 steering bins over `[-0.7, 0.7]`, 8 throttle bins over
`[0.0, 0.35]`, steer-balanced sampling with max weight 6, and regression
initialization. It improved over the earlier soft-label categorical attempt, but the
final 11-bin `curve_aug_balanced_v1` setup was kept because it had slightly better
angle MAE/RMSE and better explicit corner augmentation:

```text
9-bin diagnostics:
  steer_mae:  0.0363
  steer_rmse: 0.0599
  pred |angle| p95: 0.311
  true |angle| p95: 0.287

11-bin curve_aug diagnostics:
  steer_mae:  0.0330
  steer_rmse: 0.0586
  pred |angle| p95: 0.267
  true |angle| p95: 0.287
```

Even when validation diagnostics looked reasonable, closed-loop driving still needed a
scale factor. This is the important BC lesson: offline angle error and class counts did
not remove the need for controller calibration.

Final categorical config:

```text
steering_bins:        11
steering_min/max:     -0.733 / 0.733
throttle_bins:        8
throttle_min/max:     0.0 / 0.35
flip_prob:            0.5
sampler:              steer-balanced
sampler_weight_max:   5.0
throttle_loss_weight: 0.2
learning_rate:        1e-4
weight_decay:         1e-4
batch_size:           256
init_from_regression: models/bc_nvidia_slow_006_flip/best.pt
best_epoch:           200
best_val_loss:        0.533785
val_steer_acc:        0.8733
val_throttle_acc:     0.6047
steer_mae:            0.0330
steer_rmse:           0.0586
```

Like regression, categorical also needed steering amplification in closed loop:

```text
recommended categorical eval:
steering_scale = 1.4
steer_smoothing = 0.1
steering_limit = 1.0
```

Historical eval command:

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

Closed-loop scale sweep:

```text
1.4 / 0.1:
  steps_mean = 2000.0
  steps_min  = 2000
  mean_abs_cte = 1.754
  max_abs_cte  = 5.325

1.5 / 0.2:
  steps_mean = 2000.0
  steps_min  = 2000
  mean_abs_cte = 1.901
  max_abs_cte  = 5.375

1.4 / 0.2:
  steps_mean = 2000.0
  steps_min  = 2000
  mean_abs_cte = 2.299
  max_abs_cte  = 5.376
```

Here the notation is:

```text
steering_scale / steer_smoothing
```

Categorical development notes:

- The earlier 21-bin soft-label categorical branch collapsed toward the empirical
  marginal distribution. Its validation loss looked plausible, but predictions ignored
  image detail too much.
- Hard-bin official-style categorical plus argmax decode worked better than expectation
  decode for closed-loop control.
- Lower learning rate (`1e-4`) and initializing the trunk from the regression checkpoint
  were important. From-scratch categorical CNN training was much less reliable.
- Steering-balanced sampling and curated corner segments improved tail coverage, but raw
  output still needed scale.
- Cross-map generalization was poor: historical tests on `warren`, `generated_track`,
  and `mini_monaco` all failed in under 600 steps. This was a generated-road visual
  distribution model, not a general driving model.

### 1.3 BC Summary

Practical ranking:

```text
best RL loop policy      >  single-road VAE+SAC  >  BC regression  >  BC categorical
```

For BC alone, regression felt more stable than categorical in closed-loop simulator
driving, even though categorical had better tooling for class distribution diagnostics.
Both BC routes exposed the same core limitation: supervised imitation from narrow
generated-road data does not give enough recovery behavior, and raw network output
under-steers unless manually scaled.
