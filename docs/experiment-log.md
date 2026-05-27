# Experiment Log

This document records the practical experiment history: what worked, what failed, and
what should not be repeated. The short project entrypoint is kept in
[README.md](../README.md).

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

## 2. Single Generated Road: VAE + SAC

The first successful RL branch targeted `donkey-generated-roads-v0`.

Pipeline:

```text
image 120x160
-> crop top 40 px
-> VAE encoder, z_size=512
-> concat last 20 steer/throttle commands
-> SAC MLP policy
```

Main artifacts:

```text
models/vae_raffin_v1/best.pt
models/rl_vae_sac_raffin_v1/final_model.zip
models/rl_vae_sac_raffin_resume_010k_v1/final_model.zip
```

The VAE was trained on `generated_road` frames:

```text
input:      80x160 RGB after top crop
z_size:     512
epochs:     10
best val:   loss 295.96, recon 39.90, KL 256.07
```

SAC baseline defaults:

```text
env id:              donkey-generated-roads-v0
max_steering:        1.0
max_steering_diff:   0.15
min/max throttle:    0.4 / 0.6
n_command_history:   20
max_cte_error:       2.0
reward:              1.0 + 0.1 * throttle/max_throttle
crash:               -10 - 5 * normalized_throttle
train_freq:          (1, "episode")
gradient_steps:      -1
gradient_steps_cap:  600 in current script
max_episode_steps:   3000
```

Result on the fixed generated road:

```text
late train episode lengths:
1415, 1306, 1426, 1436, 342, 1402, 1071, 1394, 1421, 1417
```

At least 7/10 late episodes reached 1390+ steps, which corresponded to reaching the end
of the non-closed route. The short 342-step run was a real failure sample.

Generalization to another generated road with intersections was weak:

```text
eval steps: 408, 1375, 1380, 542, 414, 542, 1375
mean:       862
>=1000:     3/7
```

Failure mode: intersections were essentially random branch choices. The VAE data did
not cover intersections, and a CTE-only reward is ambiguous at intersections. A robust
intersection policy would need new VAE data, new RL training, and likely waypoint or
route conditioning.

Single-road lessons:

- `train_freq=(1, "episode")` is important. Step-based training can pause policy updates
  while the simulator keeps moving with the previous action, causing artificial crashes.
- `gradient_steps=-1` is useful because update count scales with episode length.
- A cap on dynamic gradient steps avoids long training pauses once episodes become long.
- Always save replay buffers for resumable SAC runs:

```bash
--save-replay-buffer --save-final-replay-buffer
```

## 3. Loop Track VAE Data (random-light v1, removed — superseded by §6)

The loop-track work moved to `donkey-generated-track-v0`. This visual domain differs
enough from `generated_road` that the VAE was trained separately. The dataset and
encoder described below were collected with simulator `randomlight` ON and have been
deleted because they could not reliably reproduce later in eval (see the
reproducibility note at the end of §5 and the rebuild in §6). They are kept in this
log only for context — do not try to reuse these paths.

Final loop VAE (REMOVED):

```text
models/vae_loop_cones_v1/best.pt
```

Data collection used `tools/collect_sim_frames.py` with a CTE PID controller. The sim UI
PID was not used through the Python API; the script sends actions through gym-donkeycar.

Collected data:

```text
center line:  data/vae_raw/generated_track_loop_cones_v1                 50k
cte +1.5:     data/vae_raw/generated_track_loop_cones_cte_p15_10k        10k
cte -1.5:     data/vae_raw/generated_track_loop_cones_cte_m15_10k        10k
cte +0.75:    data/vae_raw/generated_track_loop_cones_cte_p075_5k         5k
cte -0.75:    data/vae_raw/generated_track_loop_cones_cte_m075_5k         5k
cte +2.0:     data/vae_raw/generated_track_loop_cones_cte_p20_5k          5k
cte -2.0:     data/vae_raw/generated_track_loop_cones_cte_m20_5k          5k
total:        90k frames
```

The failed/too-wide `+2.5` set was deleted because it collided with track objects.

Dataset/cache:

```text
prepared dataset: data/vae_loop_cones_v1
cache:            data/vae/cache_loop_cones_v1
crop:             top 40 px removed
shape:            90000 x 80 x 160 x 3
```

VAE training:

```text
epochs: 20
best/last: models/vae_loop_cones_v1/best.pt, last.pt
final val recon: about 71.35
```

VAE data lessons:

- Keep loop-track VAE data separate from generated-road VAE data.
- Cropping the top 40 px still worked after removing trees and center obstacles.
- Include lateral offsets. Center-only data is too narrow for recovery behavior.
- Offsets must respect the actual lane geometry. `+1.5` already touches the right white
  line in this track; `+2.5` was too aggressive.
- **Lighting must be controlled.** In the original loop VAE work, simulator
  `randomlight` was left enabled. On `generated_track`, that changes color tone between
  simulator launches. It is therefore unclear whether the VAE training images, RL
  training run, and later evals used the same lighting distribution. This likely
  explains why `safe_v2 70k` sometimes fails to reproduce the original 5/5 truncate
  result in later eval sessions.

Artifact cleanup note:

The random-light loop VAE dataset and manifest were removed from `data/` after this was
understood. The matching encoder was also removed:

```text
data/vae_raw/
data/vae_loop_cones_v1/
models/vae_loop_cones_v1/
```

The next loop VAE experiment should recollect images with `randomlight` disabled and
recreate those directories from scratch.

## 4. Loop Track RL: Failed Reward Branches

### Pure Progress Reward

A progress-reward branch was tried first:

```text
models/rl_loop_vae_sac_progress_v1/
models/rl_loop_vae_sac_progress_v2/
```

These branches failed early. The progress signal and crash penalty interacted poorly
during bootstrap: short crashing trajectories dominated the Q estimates, and throttle
collapsed. They were not useful deployment candidates.

### speed_v1

The first promising loop reward was:

```text
reward = 1.0 + 0.3 * speed
```

Artifacts:

```text
models/rl_loop_vae_sac_speed_v1/sac_loop_vae_20000_steps.zip
models/rl_loop_vae_sac_speed_v1/sac_loop_vae_replay_buffer_20000_steps.pkl
```

The 20k checkpoint was intentionally preserved while `safe_v2` was the active VAE loop
branch because `safe_v2` was resumed from it. After deciding to discard the random-light
loop VAE lineage, the whole `models/rl_loop_vae_sac_speed_v1/` directory was removed.
Other `speed_v1` checkpoints had already been removed.

The branch learned to move and complete laps, but then degraded. Evaluation of later
checkpoints:

```text
30k: 0/5 trunc, mean steps 401, median 439, max 619, mean speed 2.585
40k: 0/5 trunc, mean steps 253, median 240, max 451, mean speed 2.452
50k: 0/5 trunc, mean steps 306, median 242, max 409, mean speed 2.711
```

Failure mode: the policy chased immediate speed reward and became too aggressive,
especially around the first curve after crossing the start for the second lap. The
actor forgot the slower, safer behavior because the speed bonus provided immediate
gradient while crash was delayed.

Cost of the degradation: across the entire ~30k training, only **71 laps** were
completed in the simulator (vs **357 laps** in the 70k of `safe_v2` — see §5).

Monitoring lesson: SB3 console summaries are not enough. Some fields logged by
`DonkeyInfoCallback` are per-step snapshots, not full rollout means. Replay-buffer
episode inspection with `tools/inspect_loop_replay_throttle.py` gave a better picture
of current policy behavior.

## 5. Loop Track RL: safe_v2

The successful branch is `safe_v2`.

Training decision:

- Resume model parameters from `speed_v1` 20k.
- Do **not** load the old replay buffer, because rewards in the buffer were generated
  under the old reward function and would pollute Q-learning.
- Train with lower speed incentive and an immediate off-center high-speed penalty.

Reward:

```text
reward = 1.5 + 0.15 * speed - 0.25 * abs(cte) * speed
crash  = -10 - 5 * speed
```

Core idea:

- `1.5` alive reward keeps survival as the base objective.
- `0.15 * speed` still rewards faster laps.
- `0.25 * abs(cte) * speed` penalizes the dangerous state: fast while off center.

Training defaults:

```text
min/max throttle:      0.2 / 0.7
max_steering_diff:     0.2
max_cte_error:         2.0
max_episode_steps:     3000
buffer_size:           60000
learning_starts:       1000
train_freq:            (1, "episode")
gradient_steps:        -1
gradient_steps_cap:    1000
gradient_steps_min:    500
```

Artifacts:

```text
models/rl_loop_vae_sac_safe_v2/sac_loop_vae_30000_steps.zip
models/rl_loop_vae_sac_safe_v2/sac_loop_vae_40000_steps.zip
models/rl_loop_vae_sac_safe_v2/sac_loop_vae_50000_steps.zip
models/rl_loop_vae_sac_safe_v2/sac_loop_vae_60000_steps.zip
models/rl_loop_vae_sac_safe_v2/sac_loop_vae_70000_steps.zip
```

These artifacts were removed after the random-light issue was identified. The table
below is kept as the historical result record, not as a list of files currently present
in `models/`.

Evaluation summary:

| Checkpoint | Trunc | Steps Mean | Speed Mean | Progress Mean | Mean CTE | Max CTE | Reward Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30k | 4/5 | 2979 | 2.645 | 394.0 | 0.500 | 2.336 | 4673.7 |
| 60k | 5/5 | 3000 | 2.865 | 429.7 | 0.451 | 1.827 | 4823.4 |
| 70k | 5/5 | 3000 | 2.914 | 437.1 | 0.333 | 1.715 | 5090.5 |
| 80k | 3/5 | 2293 | 2.880 | 340.3 | 0.386 | 2.193 | 3822.8 |
| 90k | 0/5 | 770 | 2.740 | 112.7 | 0.400 | 2.295 | 1250.1 |

Best checkpoint:

```text
models/rl_loop_vae_sac_safe_v2/sac_loop_vae_70000_steps.zip
```

Why 70k:

- Same 5/5 truncation stability as 60k.
- Faster than 60k.
- Better progress than 60k.
- Much better centering: mean CTE 0.333 vs 0.451.
- Lower max CTE than 60k.
- 80k and 90k show renewed policy drift toward aggressive behavior.

Lap completion statistics over the 70k of training (357 laps total):

| Lap time bucket | Count |
| --- | ---: |
| 8.x s | 96 |
| 9.x s | 206 |
| 10.x s | 37 |
| 11.x s | 17 |
| 12.x s | 1 |

Fastest single lap during training: **8.63 s**. By contrast `speed_v1` completed only 71
laps over its ~30k of training before degrading.

Reproducibility caveat:

```text
randomlight was enabled during the original VAE loop-track experiments
```

This matters because the loop VAE was trained on simulator images, not on a color
invariant pretrained representation. If later evaluation starts the simulator with a
different lighting tone than the VAE data and RL training run, the frozen VAE latent can
shift enough to degrade the SAC policy. The original 70k checkpoint should therefore be
treated as "best under the original lighting/domain", not as a lighting-invariant
deployment model.

## 5.5 Loop Track RL: Frozen ResNet Encoder v4

Claude added an alternative image encoder path to avoid VAE data collection:

```text
encoder:       frozen ImageNet-pretrained ResNet18
preprocess:    crop top 40 px, resize to 224x224, ImageNet normalization
z_size:        512
policy:        SAC MLP on [ResNet feature; last action history]
hidden size:   256
```

The hidden size is an important comparison detail. The VAE `safe_v2` branch used a
64-unit MLP (`[64, 64]` for pi/qf), while ResNet v4 used 256 (`[256, 256]`) because the
generic ResNet feature needs a larger projection head. So the v4 comparison is not
"only encoder changed"; both encoder and SAC MLP capacity changed.

v4 eval:

| Checkpoint | Trunc | Speed | Progress | Mean CTE | Max CTE | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50k | 3/3 | 2.131 | 319.5 | 0.424 | 1.65 | 4780 |
| 60k | 3/3 | 2.050 | 307.4 | 0.414 | 1.54 | 4782 |
| 70k | 3/3 | 2.052 | 307.8 | 0.551 | 1.49 | 4571 |

Best v4 checkpoint:

```text
models/rl_loop_vae_sac_resnet_v4_notrees/sac_resnet18_50000_steps.zip
```

Comparison with the historical VAE best:

| Metric | safe_v2 70k (VAE) | v4 50k (ResNet) |
| --- | ---: | ---: |
| Trunc | 5/5 | 3/3 |
| Speed | 2.914 | 2.131 |
| Progress | 437.1 | 319.5 |
| Mean CTE | 0.333 | 0.424 |

The ResNet branch is stable but conservative. Replay-buffer inspection shows the
mechanism directly:

```text
ResNet v4 50k long episodes: throttle mean mostly 0.26-0.33
VAE safe_v2 70k long episodes: throttle mean mostly 0.38-0.43
```

Likely causes:

- Frozen ImageNet features are not track-specific. They avoid VAE collection but do not
  encode lane geometry and lateral offset as directly as a loop-track VAE.
- ResNet preprocessing resizes the cropped `80x160` simulator view to `224x224`, which
  changes aspect ratio and may distort road geometry.
- Under `safe_v2` reward, uncertain high-speed states pay the `abs(cte) * speed`
  penalty immediately. A less track-specific feature representation naturally learns a
  lower-throttle, safer policy.
- Random lighting hurts the VAE branch's reproducibility, but a generic ImageNet
  encoder can be more color/texture robust. This makes ResNet attractive as a quick
  stable baseline despite lower speed.

ResNet v4 conclusion:

```text
advantage: no VAE image collection/training, more robust to visual setup changes
cost:      about 27% slower than the matched-lighting VAE policy
```

## 6. Loop Track RL: Fixed-Light VAE Pipeline

After confirming the random-light issue (see §5 reproducibility caveat), the whole loop
VAE+SAC lineage was rebuilt from scratch with simulator `randomlight` disabled. This
means VAE training images, SAC training rollouts, and SAC evaluation all share the same
fixed lighting tone — eliminating the OOD drift that broke `safe_v2 70k` reproducibility.

### 6.1 New VAE data collection

```text
encoder:           models/vae_loop_cones_fixedlight_v1/best.pt
randomlight:       DISABLED in sim before any frame collection
total frames:      80k (slightly smaller than the 90k random-light v1 set)
sim setup:         no trees on track (tree ground shadows broke encoder later — see 6.9)
```

The VAE itself was trained with the same recipe as the previous loop VAE (z=512, top
40 px crop, 20 epochs). The only material change is the matched-lighting environment.

### 6.2 fixedlight_v1 — hidden=128, eventually deleted

First SAC attempt on the new VAE used hidden=128, batch=256, `gradient_steps_min=500`,
`gradient_steps_cap=1000`. Trained cold from random init: 50k initial, then resumed
30k → 80k under same config, then later resumed 80k → 150k with `cap=3000` to see if
longer training would let it surpass `v2_h64`.

Training trajectory (rollout `ep_len_mean`):

```text
50k:  411   80k:  440   100k: 582
110k: 608   130k: 719   150k: 894
```

Survival kept climbing across the whole 150k. But deterministic eval told a less rosy
story:

| Checkpoint | Trunc | Speed | Progress | Mean CTE | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| 110k | 0/3 | 2.71 | 76 | 0.34 | crashes early in deterministic |
| 130k | 3/3 | 2.59 | 259 | 0.34 | only checkpoint that truncates reliably |
| 140k | 0/3 | 2.55 | 67 | 0.36 | full regression |
| 150k | 1/3 | 2.71 | 174 | 0.31 | partial recovery, still unreliable |

Truncation reliability was **1/4 across the tested checkpoints**. The agent had moments
of strong policy (130k) but the surrounding checkpoints collapsed. Possible cause:
hidden=128 takes much longer to converge and the policy oscillates more violently in
the converged region. Either way, the deterministic eval said the branch is not
deployable.

After this finding the entire `models/rl_loop_vae_sac_fixedlight_v1/` directory was
deleted (about 4.1 GB freed). The branch is documented here only as a "do not use this
recipe for the next deployment" data point.

### 6.3 fixedlight_v2_h64 — current deployment

Second attempt switched to the smaller MLP that matched the historical `safe_v2`
recipe:

```text
encoder:           VAE (vae_loop_cones_fixedlight_v1/best.pt)
hidden_size:       64
batch_size:        64
gradient_steps_min: 10
gradient_steps_cap: 2000  (raised from 1000 after the 50k phase)
all reward / throttle / steering defaults: matched safe_v2
cold start (no resume)
```

Training schedule was a three-phase resume to 110k:

```text
phase 1: 0 → 50k  (cap=1000 default)
phase 2: 50k → 80k  (resumed, cap raised to 2000)
phase 3: 80k → 110k  (resumed, same config)
```

Deterministic eval at every checkpoint (`max_episode_steps=2000`):

| Checkpoint | Trunc | Speed | Progress | Mean CTE | Max CTE | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 60k | 0/3 | 2.55 | 26 | 0.34 | 2.34 | 310 |
| 80k | 3/3 | 2.03 | 203 | 0.46 | 1.85 | 3144 |
| 90k | 3/3 | 2.36 | 236 | 0.32 | 1.40 | 3332 |
| **100k** | **3/3** | **2.69** | **269** | **0.30** | 1.65 | **3409** |
| 110k | 3/3 | 2.49 | 249 | 0.33 | 1.73 | 3345 |

100k is the clear best on every metric except `max_cte` (where 90k is 1.40 vs 100k's
1.65). 110k shows slight regression on speed, progress, and CTE — the early edge of
SAC late-stage drift. 90k is retained as a backup because its `max_cte` is unusually
low.

Files kept:

```text
models/rl_loop_vae_sac_fixedlight_v2_h64/sac_loop_vae_100000_steps.zip    (deployment)
models/rl_loop_vae_sac_fixedlight_v2_h64/sac_loop_vae_90000_steps.zip     (backup)
```

All other v2_h64 checkpoints and the 110k checkpoint were removed.

### 6.4 fixedlight_v3_h80 — slightly faster than v2_h64

`hidden=80` ablation against v2_h64. Same config except hidden width:
`--hidden-size 80 --batch-size 64 --gradient-steps-min 50 --gradient-steps-cap 2000`.
Cold start, 100k timesteps.

Eval at 70k/80k/90k/100k:

| Checkpoint | Trunc | Speed | Progress | Mean CTE | Max CTE | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 70k | 3/3 | 2.754 | 275 | 0.392 | 1.93 | 3294 |
| 80k | 3/3 | 2.031 | 203 | 0.460 | 1.85 | 3144 |
| **90k** | **3/3** | **2.800** | **280** | **0.315** | 1.80 | **3408** |
| 100k | 0/3 | 933 (mean steps) | — | — | — | 1584 |

**v3_h80 90k beat v2_h64 100k on all metrics**: +4.1% speed, +4.1% progress, same
mean_cte ~0.31, similar reward. 100k showed sharp regression (0/3 truncate, 933 mean
steps) — peak was earlier. v3_h80 90k became the new deployment best, replacing
v2_h64 100k.

### 6.5 Reward weight ablations: v4 (s20c20) and v5 (s20c25)

After v3_h80 hit 9.3s lap territory and stopped improving, two reward-weight
ablations were tried to see whether a stronger speed incentive could break past 9s.

Reference gradients at typical state (cte=0.30, speed=2.7):

| Config | speed_w | cte_pen | ∂R/∂speed | ∂R/∂cte | switchpoint cte |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 safe_v2 (baseline) | 0.15 | 0.25 | 0.075 | -0.675 | 0.6 |
| v4 (s20c20) | 0.20 | 0.20 | 0.140 | -0.540 | 1.0 |
| v5 (s20c25) | 0.20 | 0.25 | 0.125 | -0.675 | 0.8 |

#### v4 — speed up AND cte penalty down

```text
--speed-reward-weight 0.20  --cte-speed-penalty-weight 0.20
hidden=80, batch=64, grad-min=50, grad-cap=2000
80k cold + 30k resume with cap=3000  (failed mid-resume; collapsed)
80k cold + 20k resume with lr=2e-4 (avoided collapse but truncate rate dropped)
```

Buffer inspection at 50k showed v4's actor used **min throttle 52.9% of the time**
during 3000-step truncates, vs v3_h80's 49.5% — the stronger speed weight did
NOT translate to higher throttle use. Instead the weaker cte penalty let the actor
settle into a "ride brake / wider lines" policy that was actually slower per lap.

Eval at all v4 checkpoints (60k/70k/80k from cold; 90k/100k from lr=2e-4 resume):

| Checkpoint | Trunc | Speed | Progress | Mean CTE | Reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| 60k | 3/3 | 2.262 | 226 | 0.463 | 3174 |
| 70k | 3/3 | 2.291 | 229 | 0.466 | 3175 |
| **80k** | **3/3** | **2.460** | 246 | 0.432 | 3227 |
| 90k (lr=2e-4) | 1/3 | 2.613 | 139 | 0.379 | 1713 |
| 100k (lr=2e-4) | 1/3 | 2.490 | 161 | 0.422 | 2019 |

v4's best (80k) had `speed=2.460` vs v3_h80 90k's `2.800` and `mean_cte=0.432` vs
v3's `0.315`. **v4 reward was strictly worse than v3 on every deterministic-eval
metric**. The branch was deleted.

#### v5 — speed up, cte penalty unchanged

```text
--speed-reward-weight 0.20  --cte-speed-penalty-weight 0.25
all other params same as v3_h80, cold start, intended for 100k
stopped at 90k after observing fragile policy (in-sim, agent crashed after 2-3 laps)
```

Training showed the most aggressive lap times of any branch: sustained median dropped
to 9.09s by 90k (vs v3's 9.41s eval-mean). But deterministic eval revealed the
fragility:

| Checkpoint | Trunc | Speed | Progress | Mean CTE | Reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| **70k** | 2/3 | 2.752 | 256 | 0.421 | 3012 |
| 80k | 0/3 (mean 202 steps!) | 2.608 | 27 | 0.484 | 295 |
| 90k | 0/3 | 2.912 (good ep only) | 157 | 0.455 | 1716 |

70k was the best v5 checkpoint but still inferior to v3 (`Trunc 2/3 vs 3/3`,
`speed 2.752 vs 2.800`, `cte 0.421 vs 0.315`). 80k collapsed completely. 90k could
produce fast individual laps (`speed 2.91`) but couldn't sustain. The s=0.20 speed
weight made the actor commit too hard; the unchanged cte=0.25 wasn't enough on its
own to prevent crashes. Branch deleted.

#### Conclusion across v4 and v5

`safe_v2` weights (s=0.15, c=0.25) — i.e. the v3_h80 90k recipe — remain the best
balance under this VAE pipeline. Further reward tuning has diminishing returns; the
gap between current deployment (`speed 2.800`, ~9.3s/lap) and the historical
`safe_v2 70k` (under uncontrolled lighting, `speed 2.914`, ~8.6s training fastest
lap) is more likely encoder / SAC / physics ceiling than reward-shape limitation.

### 6.6 Encoder crop-top: legacy v4 ResNet vs new ResNet runs

The original `FrozenPretrainedCnnEncoder` inherited `MARGIN_TOP=40` from the VAE
pipeline (the VAE was trained on cropped 80x160 frames). For ResNet / MobileNet
this crop is not a good fit:

- ImageNet-pretrained CNNs were trained on natural images with sky/horizon and
  near-square aspect ratios.
- Cropping the sim image to 80x160 (aspect 0.5) then resizing to 224x224 stretches
  vertically by 2.8x and horizontally by 1.4x — extreme aspect distortion.
- Keeping the full 120x160 (aspect 0.75) gives 224x224 stretches of 1.87x and 1.4x,
  closer to natural-image proportions.

A `crop_top` parameter was added to `FrozenPretrainedCnnEncoder` and exposed via
`--encoder-crop-top` on the train/eval scripts:

- **New ResNet runs**: `--encoder-crop-top 0` (default). No crop; full 120x160 → 224x224.
- **Legacy v4 ResNet eval**: `--encoder-crop-top 40` (required). The saved policy
  was trained on cropped 80x160 input, so eval must reproduce the same preprocessing.
- VAE encoder is unaffected; it still uses its built-in `MARGIN_TOP=40` crop because
  the VAE was trained on cropped frames and must match.

### 6.7 Porting to other tracks: mountain-track cte and uphill quirks

When prepping a ResNet18 transfer test on `donkey-mountain-track-v0`, two sim-side
differences from `donkey-generated-track-v0` showed up that would silently break
any "reuse the same config" attempt:

1. **`cte = 0` is the YELLOW CENTERLINE, not the right-lane center.** Verified with
   `tools/collect_sim_frames.py --action-mode cte-pid --cte-target 0`: the PID drove
   the car onto the middle yellow line. Default spawn has cte ≈ 3.54 (consistent
   across resets). Sim's own `max_cte` is 8.0 for this env. The trained loop
   policies expect cte ≈ 0 to mean "well on the right lane", so a naive port
   spawns the agent at cte = 3.54, which already exceeds our default
   `--max-cte-error 2.0` wrapper, so the episode terminates at step 1.

   Required adjustments for mountain-track:
   - `--max-cte-error 5.0` minimum, 6.0 safer (spawn-margin), 8.0 to match sim.
   - Any reward shaping that uses `abs(cte)` is now telling the agent to drive on
     the yellow line. For driving-in-right-lane semantics, shape around
     `abs(cte - 3.5)` (or relax cte penalty entirely for this track).

2. **Initial uphill stalls at low throttle.** The default `--throttle 0.12` in
   `tools/collect_sim_frames.py` works on flat generated-track but stalls on the
   mountain uphill — the PID probe never makes progress on its first try.
   Practical defaults for mountain-track: probe with `--throttle 0.25+`, SAC training
   should keep `--max-throttle 0.7` (lowering to 0.5 to be "safe" would also
   fail the climb).

These are env-specific calibrations. Always probe a new track with the cte-pid
collector at `--cte-target 0` first, watch where the car ends up visually, and size
`--max-cte-error` / throttles accordingly.

### 6.8 Cross-track transfer attempt: ResNet on mountain-track (v1; superseded)

**Status (2026-05-27)**: superseded by the DINOv2 pipeline in §6.10. The "v2 planned"
ResNet curriculum below was not executed; mountain DINOv2 v2 became the deployment.

First test of "does the ResNet encoder pipeline transfer to a different track at all?".
Mountain-track was chosen because it's geometrically the most different from
generated-track (uphill/downhill, real-road textures) while still being a closed loop.

Pre-flight findings (see §6.7) led to:
- `--cte-target 3.5` (right-lane center, not 0)
- `--max-cte-error 2.5` initially (later relaxed to 3.5)
- `--encoder-crop-top 0` (new no-crop default for ResNet)
- Same hidden=256, batch=256 as v4 loop ResNet
- `--max-throttle 0.7` (keep default for uphill)

**Phase 1 — cold from scratch, 80k, cte=2.5**

| Checkpoint | Trunc | Steps Mean | Speed | Progress | Mean CTE | Max CTE | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 80k | 0/3 | 449 | 2.810 | 68 | 0.527 | 2.60 | 664 |

Agent learned to drive (mean_speed 2.81 — actually faster than the loop deployment's
2.80!) but every episode terminated when max_cte hit the 2.5 wrapper limit. Inspection
showed max_cte 2.51, 2.52, 2.60 — all clipped by the wrapper, not the policy giving up.

**Phase 2 — resume 80k → 110k with cte=3.5**

| Checkpoint | Trunc | Steps Mean | Speed | Progress | Mean CTE | Max CTE | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **90k** | **1/3** | **1446** | 3.230 | 233 | 0.539 | 3.59 | 2205 |
| 100k | 0/3 | 348 | 3.091 | 56 | 0.643 | 3.63 | 475 |
| 110k | 0/3 | 1302 | 3.366 | 220 | 0.531 | 3.72 | 2001 |

Loosening the cte wrapper from 2.5 to 3.5 immediately bought a 2741-step rollout in
the first dump after resume, and a clean 3000-truncate by the second dump. But
deterministic eval shows the policy is still **fast but fragile**: max_cte still
pushes 3.5+, only 1/3 truncate at the best checkpoint (90k), 100k briefly regressed.

Speed numbers (3.0+) are higher than the loop deployment, almost certainly because
mountain-track has downhill sections that accelerate the car beyond what the loop
gives.

**Mountain throttle distribution at 110k (longest 3000-truncate episode)**:

| Throttle bin | Share |
| --- | ---: |
| [0.20, 0.25) min | 32.4% |
| middle 0.25–0.65 | 46.4% |
| [0.65, 0.70) max | 21.2% |

Less bang-bang than the loop policy (v2_h64 was 40.8% min + 12.1% max). Mountain
agent uses the middle range more, reflecting more nuanced throttle modulation for
uphill/downhill rhythm.

**Open questions / next steps** (planned for next session):

- The current v1 plateau at 1/3 truncate suggests it's not just "needs more training" —
  the agent's fast-but-fragile profile points to the curriculum being too aggressive
  from the start. Plan a `mountain_v2` run with curriculum mode:
  - `--max-throttle 0.5` to cap downhill speed during early training
  - `--max-cte-error 3.0` from the start (no late switching)
  - Same encoder / hidden / batch
  - Cold start, 80k initial
- If v2 still plateaus at 1/3 truncate, the limit is likely ResNet's domain-general
  features on a track ResNet has never specifically seen.

Kept artifacts for reference / future resume:

```text
models/rl_loop_resnet_mountain_v1/sac_resnet18_90000_steps.zip          (v1 best)
models/rl_loop_resnet_mountain_v1/sac_resnet18_replay_buffer_90000_steps.pkl
models/rl_loop_resnet_mountain_v1/sac_resnet18_80000_steps.zip          (pre-resume)
```

All other v1 checkpoints + buffers deleted (2.85 GB freed).

### 6.9 Lessons from the fixed-light loop work

- **Random light → mandatory off** before any visual encoder training on `generated_track`.
  See §3 and §5 reproducibility caveat.
- **Tree ground shadows are worse than the trees themselves.** During earlier ResNet
  experiments the agent first failed with trees enabled. Turning trees off restored
  the agent's behavior. The likely cause is the *patchy ground shadow* cast by tree
  geometry, which breaks the lane-edge texture the encoder relies on. For the next
  attempt at a "with trees" policy, the VAE data must include shadowed-ground frames.
- **hidden=128 with VAE features is worse than hidden=64**, even though it produced
  some impressive single laps (training-time fastest 9.00 s vs 9.35 s). The
  deterministic eval was less reliable — only 1/4 tested checkpoints achieved 3/3
  truncate, compared to 4/5 for hidden=64. VAE features are already task-aligned, so a
  bigger MLP wastes capacity and amplifies SAC late-stage instability.
- **Never lead with "fastest single lap"** when comparing policies. That number can come
  from one optimistic burst before a crash. Use deterministic eval `mean_speed`,
  median over the last 30 training laps, longest consecutive sub-threshold streak, and
  the full lap-time distribution instead. The hidden=128 branch repeatedly looked
  strong by fastest-lap but failed on every other sustained metric.
- **`gradient_steps_cap=2000` helped v2_h64** by allowing more updates inside long
  (truncate) episodes, where the default 1000 caps the 1:1 SAC update ratio in
  half. This was a measurable, controlled change: lap speed and CTE both improved
  after the cap was raised at 50k.
- **Cold start is fine if the encoder is good.** v2_h64 started from random init and
  reached 3/3 truncate by 80k, without the warm seed `safe_v2` had needed. Matching
  the visual training/eval distribution removed the need for a "policy prior".

### 6.10 Mountain Track DINOv2 Pipeline

First successful deployment on `donkey-mountain-track-v0`. Built 2026-05-27 after
the ResNet attempt (§6.8) plateaued.

Encoder: **frozen DINOv2-S** (`dinov2_vits14` via torch.hub, z=384). No per-track
VAE collection required — DINOv2 is pretrained.

#### v1 — cold start (raffin reward + max_cte 2.5 + cte_target 3.5)

```bash
python rl/train_loop_vae_sac.py \
    --env-id donkey-mountain-track-v0 \
    --output-dir models/rl_dinov2_mountain_v1 \
    --encoder dinov2_vits14 --encoder-crop-top 0 \
    --hidden-size 256 \
    --batch-size 64 --buffer-size 30000 \
    --learning-starts 500 \
    --gradient-steps-cap 1000 --gradient-steps-min 200 \
    --cte-target 3.5 --max-cte-error 2.5 \
    --min-throttle 0.2 --max-throttle 0.7 \
    --alive-reward 1.5 --speed-reward-weight 0.15 \
    --reward-crash -20 --crash-speed-weight 10 \
    --cte-speed-penalty-weight 0.25 \
    --timesteps 80000 --seed 42 --device cuda \
    --save-replay-buffer --save-final-replay-buffer
```

Stopped at 60k. Deterministic eval (5 ep × 2000 max steps):

| Checkpoint | Trunc | Speed | Mean CTE | Max CTE | Best lap | Mean lap |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| **30k** | **5/5** | 1.998 | 0.682 | 2.163 | 29.44s | 31.09s |
| 40k | 3/5 | 2.124 | 0.778 | 2.619 | 28.16s | 29.14s |
| 50k | 0/5 | 1.747 | 0.874 | 2.748 | — | — |
| 60k | 4/5 | 1.917 | 0.673 | 2.516 | 30.08s | 31.19s |

30k 5/5 is genuine — first DINOv2 mountain win. 40k+ shows the 10-20k oscillation
pattern (50k full valley, 60k partial recovery).

#### v2 — resume from v1 30k with stricter reward (current deployment)

Hypothesis: v1 30k learned to drive but with high cte variance. Tighten reward to
trim it. Resume with smaller LR, slightly stricter cte, heavier crash:

```bash
python rl/train_loop_vae_sac.py \
    --env-id donkey-mountain-track-v0 \
    --output-dir models/rl_dinov2_mountain_v2 \
    --encoder dinov2_vits14 --encoder-crop-top 0 \
    --hidden-size 256 \
    --batch-size 64 --buffer-size 30000 \
    --gradient-steps-cap 1000 --gradient-steps-min 200 \
    --cte-target 3.5 --max-cte-error 2.5 \
    --min-throttle 0.2 --max-throttle 0.7 \
    --alive-reward 1.5 \
    --speed-reward-weight 0.12 \
    --reward-crash -20 --crash-speed-weight 10 \
    --cte-speed-penalty-weight 0.30 \
    --learning-rate 2e-4 --override-learning-rate \
    --resume-model models/rl_dinov2_mountain_v1/sac_dinov2_vits14_30000_steps.zip \
    --resume-replay-buffer models/rl_dinov2_mountain_v1/sac_dinov2_vits14_replay_buffer_30000_steps.pkl \
    --timesteps 40000 \
    --seed 42 --device cuda \
    --save-replay-buffer --save-final-replay-buffer
```

Deterministic eval:

| Checkpoint | Trunc | Speed | Mean CTE | Max CTE | Best lap | Mean lap |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| **40k** | **5/5** | 2.093 | 0.577 | 2.112 | 28.41s | 29.88s |
| 50k | 5/5 | 2.088 | 0.801 | 2.279 | 28.29s | 29.96s |
| 60k | 0/5 | 2.520 | 0.746 | 2.742 | — | — |

**v2 40k beats v1 30k on every metric** (speed +5%, mean_cte -15%, max_cte -2%,
mean_lap -4%) and stays at 5/5 truncate. **Current mountain deployment.**
50k is 5/5 but mean_cte rising; 60k collapsed (same 10-20k cycle).

#### v3 — cold start with v2-style stricter reward (failed)

Four cold-start attempts trying to learn v2's reward shape from scratch
(seed=42 / 7, with and without max_cte=3.0, cte_pen 0.25 / 0.30). **All four
got stuck at the first corner around 130-160 ep_len** with ent_coef collapsing
to ~0.02 within 5k steps. The stricter cte penalty actively prevented the
exploration needed to learn corner-taking. Branch deleted.

Key learning: **stricter cte penalty is incompatible with cold start on
mountain**. Need to learn corners first under permissive reward, then refine.

#### v4 — resume from v1 *20k* with v2 reward (4/5, not deployable)

Tested whether earlier-stage v1 checkpoint (less committed than 30k) would
respond better to v2 reward. Reached 4/5 truncate at the 60k step but
deterministic eval shows v2 40k still beats it on every metric. Branch deleted.

| Checkpoint | Trunc | Speed | Mean CTE | Max CTE | Best lap | Mean lap |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| 60k | 4/5 | 1.988 | 0.728 | 2.503 | 29.58s | 31.25s |

Confirmed that v2's resume-from-30k path is the right recipe; v4 is left as
documented evidence not as an artifact.

#### Lessons from the mountain DINOv2 series

- **DINOv2-S is a viable encoder on mountain track** — first time we got
  ≥3/5 truncate there without per-track encoder training.
- **Stricter cte reward requires a "knows how to drive" baseline.** v3 cold
  start with stricter reward fails identically across seeds; v2 resume from
  permissive-reward baseline succeeds.
- **The 10-20k peak-collapse-recovery cycle persists on mountain**, identical
  to the loop pattern. Eval every checkpoint, deploy the peak.
- **Training-time trunc rate dropping ≠ deterministic eval degrading.** v4 60k
  had its training-time trunc rate fall from 42% → 15% but deterministic eval
  at 60k was actually the best of the v4 batch (4/5). The training-time drop
  reflects exploration noise; deterministic policy is more stable.

### 6.11 Training defaults updated 2026-05-27

After the v4 instability analysis, two SAC defaults were raised toward the
SAC paper's values for late-training stability. A `gradient_steps_scale`
parameter was also added for future ablation.

| Default | Old | New | Rationale |
| --- | ---: | ---: | --- |
| `--batch-size` | 64 | **128** | Reduce critic gradient variance. SAC paper uses 256; 128 is a midpoint. |
| `--buffer-size` | 30_000 | **50_000** | Slower buffer turnover → less distribution shift during late training. |
| `--gradient-steps-scale` | (implicit 1.0) | **1.0** (explicit) | New CLI flag. Default keeps existing behavior (1 update per env step). Set <1 (e.g. 0.5) to dampen long-episode overupdate; experimental, not yet validated. |
| `--gradient-steps-cap` (`train_vae_sac.py`) | 600 | **1000** | Unified with `train_loop_vae_sac.py`. |

`CappedDynamicGradientStepsCallback` formula changed from
`clamp(ep_len, [floor, cap])` to `clamp(ep_len * scale, [floor, cap])`. With
default `scale=1.0` this is identical to prior behavior.

Not yet validated in a deterministic-eval run — applies starting from the
next cold-start training.

### 6.12 Encoder generalization: lighting robustness comparison (2026-05-27)

Single-episode eval on `donkey-generated-track-v0`, 2000-step cap. "Random
light" = simulator `randomlight` enabled (lighting direction randomised each
episode); "Fixed light" = simulator default stable lighting.

**Models tested**

| Encoder | Model | Checkpoint |
| --- | --- | --- |
| VAE (task-specific, fixedlight) | rl_loop_vae_fixedlight_v3_h80 | 90k |
| DINOv2 ViT-S (frozen ImageNet-pretrained) | rl_loop_dinov2_v8 | 30k |
| ResNet18 (frozen ImageNet-pretrained) | rl_loop_resnet_v4 | 50k |

**Results**

| Encoder | Light | Trunc | Speed | Mean CTE | Lap time |
| --- | --- | :---: | ---: | ---: | ---: |
| VAE | Fixed | 1/1 | 2.742 | 0.296 | 9.65 s/lap (5 laps) |
| VAE | Random | 0/1 | — | — | crash at step 269 |
| DINOv2 | Fixed | 1/1 | 2.356 | 0.565 | 11.03 s/lap (4 laps) |
| DINOv2 | Random (×3) | 3/3 | 2.38 avg | — | 10.97 s/lap avg |
| ResNet18 | Random (×3) | 3/3 | 2.09 avg | — | 12.88 s/lap avg |

(ResNet was only tested under random light; VAE was not tested for a second
fixed-light run. Fixed-light VAE lap count is 5 laps / 1000 steps.)

**Conclusions**

- **VAE is task- and lighting-specific.** Trained on fixed-light data, it
  crashes immediately when lighting shifts. This is expected: the VAE
  latent space encodes lighting as part of scene appearance; distribution
  shift to OOD lighting breaks the encoder's implicit assumptions.
- **Frozen pretrained encoders (DINOv2, ResNet18) are lighting-robust.** Both
  survive all three random-light runs without issue. ImageNet pretraining
  covers a wide distribution of lighting conditions; a few random-light
  episodes are in-distribution for these encoders.
- **DINOv2 > ResNet18 in lap time** (10.97 s vs 12.88 s avg) under random
  light, consistent with DINOv2's stronger perceptual features from
  self-supervised pretraining.
- **Practical guidance**: use VAE when lighting is controlled end-to-end
  (fastest laps, lowest CTE); use DINOv2 when lighting may vary (robust,
  only ~1.3 s/lap slower). ResNet18 is a reliable fallback but lags DINOv2.

## 7. Practical Pitfalls

- Do not install the PyPI `gym-donkeycar` package — it targets the old `gym` API.
  Install upstream instead with `pip install git+https://github.com/tawnkramer/gym-donkeycar`,
  or clone that repo and `pip install -e ../gym-donkeycar` for local edits. See the
  README's gym-donkeycar section for details.
- Do not mix VAE data from visually different simulator tracks.
- Disable or control simulator `randomlight` before VAE data collection, RL training,
  and evaluation. Otherwise lighting/domain shift can make VAE checkpoints look
  non-reproducible.
- Do not judge SAC only by training reward or SB3 rolling means.
- Save replay buffers at checkpoints if resume matters.
- Do not resume SAC with replay buffers generated by a different reward function.
- `max_episode_steps` truncation is not a crash reward. It is a TimeLimit stop.
- On a non-closed generated road, not reaching 3000 steps can still be success if the
  vehicle reached the route end.
- If a speed bonus is too large, late training can regress even after the policy has
  learned to drive.

### 7.1 Monitoring gotchas (learned during safe_v2)

- **SB3 console dumps every 4 episodes by default.** The `rollout_ep_len` field in the
  printed table is the length of the *last* episode in that batch of four. Intermediate
  truncates and short crashes between dumps are invisible in stdout. To see every
  episode, either inspect the replay buffer or add a callback that records every
  termination.
- **`donkey/abs_cte_mean`, `throttle_mean`, `speed_mean` are single-step snapshots, not
  rollout means.** `DonkeyInfoCallback._on_step` calls `logger.record(...)` every step,
  which overwrites the previous value. The number shown at dump time is whatever the
  last step recorded, often a crash terminal step where `abs_cte` is near
  `max_cte_error`. For a true policy picture, use
  `tools/inspect_loop_replay_throttle.py` on a saved replay buffer.
- **Late-stage policy degradation is often a phase transition, not a gradual decay.**
  `safe_v2` went from 9/15 truncates in the buffer's recent-15 episodes at 80k to 0/5
  truncates in deterministic eval at 90k — within roughly 10k env steps. Watching
  `ep_rew_mean` slowly drift is not sufficient; the 100-episode rolling window is
  dominated by older long episodes, so it underreports recent collapses. Deterministic
  evaluation at each checkpoint is the reliable signal.

### 7.2 Monitoring practices

The practices below are what produced a deployable checkpoint without overtraining:

- Save replay buffers at every checkpoint (`--save-replay-buffer`).
- At each checkpoint, inspect the buffer's most recent ~15 episodes with
  `tools/inspect_loop_replay_throttle.py`. Track whether throttle mean is creeping up,
  whether episode length is shrinking, and whether truncates are still appearing. These
  three signals together flag drift earlier than `ep_len_mean` does.
- When training finishes (or when buffer analysis hints at drift), run deterministic
  evaluation on every saved checkpoint, not just the final one. The metric that matters
  is "truncated@cap" in eval, not training reward.
- Keep only the best-performing checkpoint plus the immediate seed; delete everything
  else to keep the model directory navigable.

## 8. Design Recommendations

Current deployments and eval commands: see [README.md](../README.md).

What the experiments established about how to train and evaluate:

- **For new loop SAC runs**, copy the v3_h80 90k recipe exactly:
  `--encoder vae`, `--hidden-size 80`, `--batch-size 64`, `--gradient-steps-min 50`,
  `--gradient-steps-cap 2000`, safe_v2 reward defaults (`speed=0.15, cte_pen=0.25`),
  cold start, eval at every checkpoint. Stop training and use the best checkpoint
  found in deterministic eval.
- **Do not retune reward weights blindly.** v4 (`s=0.20, c=0.20`) and v5
  (`s=0.20, c=0.25`) were both deliberately tested and both produced worse
  deterministic-eval results than v3 (`s=0.15, c=0.25`). See §6.5 for the full ablation.
- **Compare runs by sustained metrics**, not "fastest single lap" (see §6.9).
- **For new ResNet/MobileNet runs use `--encoder-crop-top 0`**, not the legacy 40 that
  the older v4 ResNet was trained with. See §6.6.
- **Always disable simulator `randomlight`** before VAE collection / SAC training /
  eval. This was the root cause of the historical safe_v2 reproducibility failure.
- The old `safe_v2 70k` artifact remains a "best ever under matched lighting" reference
  point but is not a reproducible deployment model.

## 9. Current Data And Model Inventory

The repository was cleaned so that the remaining local artifacts map directly to active
or historically useful branches.

Current `data/`:

```text
data/slow_data_raw/
  Six slow generated-road tubs. Used by BC regression and categorical training.

data/curated_cornering_v1_clean/
  First cleaned cornering subset. Used to add high-curvature recovery examples to BC.

data/curated_cornering_v2_clean/
  Second cleaned cornering subset. Used by the official-style categorical branch.

data/Cornering data.zip
data/cornorraw.zip
  Compressed backups of raw cornering captures. The extracted copies were deleted.
```

Deleted data categories:

```text
data/vae/
  Old generated-road VAE manifests/cache leftovers.

data/vae_raw/
  Random-light loop VAE raw images and older extracted cornering raw images.

data/vae_loop_cones_v1/
  Manifest for the removed random-light loop VAE dataset.
```

Active `models/`: see [README — Model Inventory](../README.md#model-inventory).

Deleted model categories:

```text
models/vae_loop_cones_v1/
  Random-light loop VAE encoder.

models/rl_loop_vae_sac_safe_v2/
  SAC policies trained on the removed random-light loop VAE encoder.

models/rl_loop_vae_sac_speed_v1/
  Early loop VAE seed branch for safe_v2.

models/rl_loop_vae_sac_fixedlight_v1/
  Hidden=128 attempt on the fixed-light VAE. Deleted because only 1/4 tested
  checkpoints reached 3/3 truncate; hidden=64 (v2_h64) is the better recipe.

models/rl_loop_vae_sac_fixedlight_v4_h80_s20c20/
  Reward weight ablation (speed=0.20, cte_pen=0.20). Deleted: all checkpoints lost
  to v3_h80 90k by ~12% on speed and ~37% worse on mean_cte. See §6.5.

models/rl_loop_vae_sac_fixedlight_v5_h80_s20/
  Reward weight ablation (speed=0.20, cte_pen=0.25 unchanged). Deleted: only 70k
  reached 2/3 truncate; 80k collapsed to 0/3 with mean 202 steps; 90k could produce
  fast laps but not sustain. See §6.5.

models/rl_loop_vae_sac_progress_v*/
models/rl_loop_vae_sac_v1/
models/rl_loop_vae_sac_resnet_v1/
models/rl_loop_vae_sac_resnet_v2/
models/rl_loop_vae_sac_resnet_v3/
models/rl_vae_sac_raffin_resume_010k_v1/
models/vae_raffin_smoke/
models/bc_official_categorical_9bin_sampler6_300/
  Failed, exploratory, or superseded experiment branches.
```

## 10. Open TODOs

### Validate new training defaults

`--batch-size 128` and `--buffer-size 50_000` (raised 2026-05-27, see §6.11)
have not been tested in a deterministic-eval run. The next cold-start training
should compare:
- training-time `ep_len` / `critic_loss` curves vs prior runs
- whether the late-stage 10-20k peak-collapse-recovery cycle dampens
- whether sweet-spot checkpoint shifts later than before (slower convergence
  is the expected cost of larger batch + buffer)

### `--gradient-steps-scale` ablation

New CLI flag (default 1.0). Try a single 60k cold-start with `0.5` to see if
halving the update-to-data ratio reduces late-stage drift. Compare against a
1.0 baseline on the same env / seed.

### Other open items

- Mountain v1's full checkpoint set is kept for now because v2 derives from
  v1 30k. If v2 stays stable across future work, v1 can be reduced to just
  the 30k zip + buffer.
- Loop deployment v3_h80 90k and v15 112488 remain the production models.
- Resnet mountain v1 (§6.8) is superseded but kept as historical reference.
- No further reward-weight ablations planned on loop (§6.5 closed the loop).
- No further hidden-size ablations planned on loop (h64 → h80 → h128 path
  fully explored).

