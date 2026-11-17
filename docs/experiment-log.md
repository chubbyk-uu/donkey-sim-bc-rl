# Experiment Log

This document records the practical experiment history: what worked, what failed, and
what should not be repeated. The short project entrypoint is kept in
[README.md](../README.md).

## Reading Guide

This log is chronological enough to preserve context, but the project now has several
parallel lines. Use this guide instead of reading straight through on a first pass:

```text
BC baselines                                  §1
Original single generated-road VAE+SAC        §2
Removed random-light loop VAE history         §3-5
Current fixed-light loop VAE pipeline         §6.1-6.6 and §6.9
Mountain DINOv2 pipeline                      §6.7, §6.10, §6.13
Encoder lighting/generalization comparison    §6.12
Random light + tree-shadow domain randomization §6.14-6.15
Operational pitfalls and monitoring rules     §7
Current recommendations                       §8
Data/model inventory and TODOs                §9-10
Runtime revalidation / captured failures      §11
Laps baseline, state probe, simulator speed-up §12
Ground-truth state policy against vision, paired evaluation §13
```

README is the source of truth for current deployment commands and the short model
inventory. This log explains why those choices were made and records deleted or
failed branches so they are not repeated.

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
gradient_steps_cap:  600 in this historical run; current script default is 1000
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

### 6.3 fixedlight_v2_h64 — former deployment / secondary kept model

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
models/rl_loop_vae_sac_fixedlight_v2_h64/sac_loop_vae_100000_steps.zip    (former deployment / secondary)
models/rl_loop_vae_sac_fixedlight_v2_h64/sac_loop_vae_90000_steps.zip     (backup)
```

All other v2_h64 checkpoints and the 110k checkpoint were removed.

Status update: v2_h64 100k was later replaced by v3_h80 90k as the README primary
loop deployment, but the 100k/90k v2_h64 checkpoints remain useful secondary
references.

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

Current fixed-light loop status: README treats **v3_h80 90k** as the primary
fixed-light loop deployment and **v15 112488** as the long-eval co-primary. The v15
details live in `docs/session_2026-05-26_log.md` because that branch was developed in
the later DINOv2/v15 session; model inventory keeps both artifacts visible.

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
| ResNet18 (frozen ImageNet-pretrained) | rl_loop_vae_sac_resnet_v4_notrees | 50k |

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

### 6.13 New-defaults validation: mountain DINOv2 cold start (2026-05-28)

First run exercising the raised defaults (§6.11). Used mountain DINOv2 (faster to
train than loop h80) as the test vehicle: a single cold start with the permissive
v1-recipe reward (`speed=0.15, cte=0.25`, `reward_crash=-20, crash_speed=10`).

Config tested vs the actual §6.11 defaults — **note the confound**:

| Knob | This run | §6.11 default |
| --- | ---: | ---: |
| `--batch-size` | **256** | 128 |
| `--buffer-size` | 50_000 | 50_000 |
| `--gradient-steps-scale` | **0.5** | 1.0 |
| `--gradient-steps-cap` | 1000 | 1000 |

Batch is double the default and scale=0.5 is the *ablation* value (§10), not the
default. So this run conflates the §6.11 validation with the `gradient-steps-scale`
ablation at a non-default batch — neither question is answered cleanly.

Deterministic eval (5 ep × 2000 max steps; 30-60k from cold start, 70-90k from a
resume that overshot to 90k — see session log 2026-05-28 §3):

| Checkpoint | Trunc | Speed | Mean CTE | Max CTE | Best lap | Mean lap |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| 30k | 0/5 | 2.04 | 0.59 | 2.76 | 25.90s | 27.80s |
| **40k** | **5/5** | 2.18 | 0.66 | 2.27 | 27.00s | 28.24s |
| 50k | 3/5 | 2.28 | 0.57 | 2.25 | 25.57s | 26.89s |
| 60k | 4/5 | 2.21 | 0.53 | 2.54 | 26.60s | 27.95s |
| 70k | 2/5 | 2.09 | 0.69 | 2.41 | 28.08s | 29.08s |
| 80k | 4/5 | 2.10 | 0.70 | 2.63 | 28.09s | 29.24s |
| 90k | 3/5 | 2.03 | 0.60 | 2.57 | 28.64s | 30.13s |

**Findings:**

- **A single-stage cold start reached a deployable 5/5 (40k).** v3 40k beats the
  two-stage v2 40k deployment (§6.10) on speed (2.18 vs 2.093) and mean lap
  (28.24s vs 29.88s), at the cost of running wider (mean_cte 0.66 vs 0.577,
  max_cte 2.27 vs 2.112). The aggressive optimizer config let one cold start do
  what previously needed v1-cold + v2-resume. This is the run's main positive
  signal.
- **The §6.11 defaults (batch=128, scale=1.0) remain unvalidated** — this run
  cannot separate their effect from the extra-aggressive batch and the scale
  change.
- **Late-stage drift persisted.** scale=0.5 + batch=256 did not dampen the
  10-20k peak-collapse-recovery cycle; 70-90k were uniformly worse than 40-60k,
  and the extra 30k steps past 60k bought nothing. The peak shifted only
  marginally later than v1's 30k (§6.10), to the 40-50k zone.

**Deployment decision:** mountain primary stays **v2 40k** (§6.10). v3 40k is kept
as a new-defaults validation artifact and a faster single-stage cold-start
alternative — see README Model Inventory.

### 6.14 Domain randomization: random light + tree shadows on loop (2026-05-28)

**Headline: DINOv2 keeps its generalization under randomized appearance.** With
domain randomization the frozen DINOv2-S policy adapts to *both* randomized
lighting *and* randomly-placed trees and their cast shadows — completing full
laps on layouts it never trained on. §6.12 already showed DINOv2 was robust to
random lighting; this extends it to random trees/shadows, which are a harder
distractor (shadows fall on the road and mimic lane edges).

**Baseline without DR fails.** The plain loop DINOv2 model (`rl_loop_dinov2_v8`),
trained on a single fixed scene, crashes at step 242 (0/1, max_cte 2.10) the
moment random light + trees are enabled. So robustness is not free from the
encoder alone — it has to be trained in.

**Scene-reload domain randomization (the mechanism).** The sim only regenerates
trees/lighting/shadows when the scene is *reloaded*; a normal episode reset
(`reset_car`) keeps the same layout. New code in `train_vae_sac.py`:

- `reload_scene()` — `exit_scene` → settle → re-send `load_scene` until the sim
  re-handshakes (`handler.loaded`). `load_scene` is ignored mid-scene, so the
  exit-to-menu step is required. (Probe tool: `tools/test_scene_reload.py`.)
- **Adaptive step-based reload** (`--scene-reload-alpha`, `--scene-reload-kmin`):
  reload after the car has spent `K = clamp(alpha × recent_ep_len, [kmin,
  max_ep-1])` steps on the current scene, decided only at episode boundaries
  (never force-truncates). `_steps_on_scene` accumulates across crashes and
  resets only on reload, so easy (long-episode) and hard (short-episode) layouts
  contribute roughly equal *steps* to the replay buffer. This fixed a real bias:
  episode-count-based reload let easy scenes (long truncate episodes) dominate
  the buffer. Counter-intuitively, a *smaller* fixed step budget worsens balance
  (an easy scene still spends a full episode before the boundary check), so K
  near `max_ep` balances best; adaptive K gives early-training variety and
  late-training balance. Episode-based `--scene-reload-every N` is retained for
  **eval** (use `N=1` so each eval episode is a fresh random layout).

**Result — `rl_loop_dinov2_randomtree_v2`** (DINOv2-S, hidden=256, batch=256,
`gradient-steps-scale=0.5`, adaptive reload `alpha=3 kmin=200`, raffin reward
`speed=0.15 cte=0.25`, `reward_crash=-20`, `max_ep=2000`). Paired eval — every
checkpoint on the **same** 6 freshly-reloaded random layouts (the fair way to
compare; see `rl/eval_paired_randomized.py`):

| Checkpoint | Trunc | Mean steps | Mean CTE | Mean speed |
| --- | :---: | ---: | ---: | ---: |
| 30k | 3/6 | 1418 | 0.460 | 1.771 |
| 40k | 2/6 | 1106 | 0.457 | 1.795 |
| **50k** | **3/6** | 1262 | **0.405** | 1.842 |
| 60k | 2/6 | 1168 | 0.429 | 1.843 |

Independent per-checkpoint eval (5 ep, fresh layout each) agreed: 30k 4/5, 40k
3/5, 50k 4/5, 60k 3/5. So **~50% truncate on unseen random layouts** — and the
"failures" are long drives (4-6 laps) that eventually drift to cte≈2.0, not early
crashes. Best checkpoint **50k**.

**Three attempts to push past ~50% — all failed (documented so they aren't
re-tried):**

1. **Tighter cte penalty** (0.25→0.30) + lower min-throttle: no clear trunc gain.
   Note the cte penalty is `weight × |cte| × speed`, so lowering speed weakens it
   — raising the weight is partly cancelled by the slower driving it induces.
2. **Steering-change penalty** to damp the persistent left-right weave
   (`abs_steer_delta_mean ≈ 0.30` throughout — i.e. ~73% of the per-step steering
   cap `max_steering_diff=0.2`→±0.4, *not* "15% of full range" as an earlier draft
   wrongly stated). Both linear (`w=0.5`) and squared (`w=2.0`) penalties left
   steer_delta **dead flat at ~0.30** over tens of thousands of steps — zero
   effect. Removed.
3. **Bigger frozen encoder** (DINOv2 ViT-B/14, z=768, hidden=384): steer_delta
   identical (~0.30) and trunc no better, at ~2× the per-step cost (fps ~9 vs
   ~15). Deleted.
4. **Frozen ResNet18 under the same randomized light/tree-shadow setup** (crop_top=0,
   hidden=256, batch=256, `gradient_steps_scale=0.5`, `alive_reward=1.65`,
   adaptive reload `alpha=3 kmin=200`) was stopped at 20k. It produced
   `sac_resnet18_10000_steps.zip` and `sac_resnet18_20000_steps.zip` during the run,
   but the branch was deleted immediately after review: at ~21.4k env steps, the last
   100 episodes averaged ~121 steps, the best recent episode was 179 steps, and
   `lap_count_mean` stayed at 0. Visually, it could pass the first bend on some
   layouts and fail again after a scene reload, indicating strong sensitivity to
   road-shadow layout. Conclusion: **ResNet18's random road-shadow generalization is
   far worse than DINOv2-S**; do not spend more randomtree experiments on frozen
   ResNet unless the goal is a negative baseline.

**Conclusions (where the limits actually are):**

- **The weave is not an encoder-*capacity* problem and not a reward-tunable habit;
  its root cause is still undetermined.** Evidence: `steer_delta` stays ~0.30
  identically across ViT-S and ViT-B, and is unmoved by linear (`w=0.5`) or
  squared (`w=2.0`) steer-change penalties. That rules out encoder *size/
  resolution* and rules out "a free bad habit reward can fix". It does **not**
  distinguish (a) a SAC control-layer high-gain/limit-cycle behavior from (b) a
  *shared* perception weakness of frozen single-frame DINOv2 (noisy/laggy lateral
  estimate the policy must high-frequency-correct). Both predict "identical across
  ViT sizes" and "penalty can't remove it". An earlier draft calling this "a SAC
  control artifact" overstated the evidence.
- **Replay-buffer probe (v2 50k, 300-step run; training-time, includes SAC
  exploration noise, no cte stored):** executed-steer **sign-flip rate 0.21**
  (~one reversal per 5 steps) — a *mid-frequency large-amplitude* sway, **not**
  step-by-step bang-bang (which would be ~0.5). mean|steer|≈0.47; per-step
  |Δsteer|≈0.29 ≈ 73% of the 0.4 cap. The **raw policy** output wants |Δsteer|≈0.51
  = **128% of the cap** — i.e. it continuously saturates `max_steering_diff`,
  which is doing real smoothing work.
- **Untried lever this surfaced:** since the policy pins the `max_steering_diff`
  clamp, *tightening* it (0.2→~0.1) would physically suppress the weave (a hard
  clamp the policy cannot ignore, unlike reward penalties) — at the risk of
  under-steering tight corners. Not yet tested.
- **Still to do for a clean diagnosis:** record per-step cte+steer from a
  *deterministic* eval (buffer has no cte and is exploration-noisy), and/or test
  frame-stacking (if the weave drops, it's a missing-temporal perception issue).
- **The ~50% trunc ceiling on random layouts is not broken by a bigger *frozen*
  encoder.** The remaining hard layouts are heavy-shadow cases where the frozen
  latent likely entangles road-shadow edges with lane edges. The next real lever
  is a **task-adapted encoder** — fine-tuning DINOv2, or a depth / driving-
  segmentation model that is inherently shadow/lighting-invariant — not more RL
  reward tuning (see §8).

> **Eval-clamp caveat (found 2026-05-29):** the paired-eval numbers above were
> produced by `eval_paired_randomized.py`, which until 2026-05-29 **hardcoded
> `max_steering_diff=0.15`** while training uses 0.2 — i.e. a tighter steering
> clamp than the policy trained under, biasing these results slower/more
> conservative. The bug is fixed (now a CLI arg, default 0.2). Independent
> `eval_loop_vae_sac.py` runs (default 0.2) were unaffected. Re-measured at the
> matched 0.2, v2 sits ~4/6 (§6.15 A), so these 3/6-ish figures slightly
> understate v2.

### 6.15 crop / steering-clamp / cte-tolerance probes (2026-05-29)

Three probes against the §6.14 ~50% random-layout ceiling, plus a quantified look
at the steering weave. All deterministic, paired on identical random layouts
(`eval_paired_randomized.py --models`, matched `max_steering_diff=0.2`), unless
noted. Narrative + exact commands: session log 2026-05-29.

**crop (top-crop the camera before the encoder): no significant benefit.**
A `--encoder-crop-top 40` cold start (drop the top third — sky/tree-canopy) was
trained to 130k (resume from 80k at `max_episode_steps=1200`). Paired vs v2:

| model (same 6 layouts) | Trunc | Mean CTE | Mean speed |
| --- | :---: | ---: | ---: |
| v2_50k (crop0) | 2/6 | 0.399 | 1.68 |
| v2_60k (crop0) | 2/6 | 0.403 | 1.89 |
| crop40_130k (crop40) | 4/6 | 0.472 | 1.96 |

crop40_130k leads — but it trained to 130k vs v2's 60k, so the lead conflates crop
with 70k extra steps. A same-steps control (crop40_60k vs v2_60k, same layouts):
v2 **4/6** (cte 0.388) vs crop40 **3/6** (cte 0.540) — at equal steps crop did not
win and ran wider. **Caveat:** that control is not blood-line-equal (v2_60k =
v1-10k-resume + adaptive; crop40_60k = pure cold start), so the gap may be
cold-start-vs-resume rather than crop. **Verdict: crop shows no significant
benefit; not enough evidence to call it harmful. Default to crop0; don't pursue
crop. The clean path to a stronger randomized model is to resume v2 (crop0) to
more steps, not crop.** (`crop40_130k` is a decent artifact — 4/6 — but kept only
as such, not promoted; it is crop=40.)

**Also corrected here:** "training-table looked like a plateau (80k→130k)" was
wrong — deterministic eval showed 80k=3/8 → 110k/130k=5/8 (at msd=0.2, separate
8-layout run). Another training-table ≠ deterministic-eval case.

**Steering weave: root cause still undetermined; one direct lever quantified.**
Buffer probe (v2 50k, 300-step run): executed-steer sign-flip rate **0.21** (~one
reversal / 5 steps — mid-frequency sway, *not* step-by-step bang-bang); raw policy
|Δsteer| ≈ **128% of the 0.4 clamp** (it continuously saturates `max_steering_diff`).
Tightening the clamp at eval (0.2→0.15) cut the *steering-action* amplitude
(|dsteer| 0.30→0.25) **but cte_std did not improve** (~0.51 both) and trunc dropped
sharply (crop40 130k: 5/8 → 0/8) with slower, wider driving. So a tighter clamp
suppresses the *visual* wheel motion, not the underlying lateral instability, and
costs generalization. The control-limit-cycle vs frozen-DINOv2-perception question
is still open (needs deterministic per-step cte/steer frequency analysis +
frame-stacking test).

**`max_cte_error=2.0` is somewhat strict for this track.** Re-running crop40_130k
at `--max-cte-error 2.5` (8 layouts) gave trunc **6/8** vs ~4/8 at 2.0. The two
rescued episodes had max_cte 2.27 / 2.31 (light line-touching just past 2.0); true
losses of control (max_cte 2.50, 2.84) were still not saved. So ~25% of "failures"
at 2.0 are near-misses, not real crashes — but whether cte 2.3 is truly
out-of-bounds depends on the track's drivable width (not measured), and rescued
episodes ride the boundary (high cte_std), i.e. line-touching survival rather than
better driving. Use a looser tolerance only if the track geometry justifies it,
and never compare trunc rates across different `max_cte_error`.

**Resuming past the plateau + a speed-vs-stability finding.** crop40 resumed
130k→200k (paired on the same 6 layouts, msd=0.2): trunc 130k=2/6, 170k=4/6,
200k=3/6 — peak-valley, no breakthrough; resuming past ~130k buys nothing on
trunc. But per-lap times show the real trade-off: on layouts each *can* finish,
**130k laps ~1-1.5 s faster** (steady ~12.0-12.5 s) than 170k/200k (~13.4-13.6 s).
So resume = "slow down to survive more layouts" (130k fast-but-fragile, 170k
stable-but-slower); neither dominates.

**Methodology: read lap time, not `mean_speed`, for driving speed.** `mean_speed`
is the instantaneous speed averaged over *every* eval step — **including the
low-speed / losing-control steps before a crash**. A checkpoint that crashes more
(130k: 2/6) gets its `mean_speed` dragged down by those steps (130k mean_speed
1.666, the *lowest* of the three) even though its actual lap pace is the
*fastest*. So `mean_speed` conflates "how fast it drives" with "how often it
crashes" and is misleading — use lap time for driving speed. Likewise, with only
~6 layouts the absolute trunc/mean_speed swing a lot batch-to-batch (the same
130k scored 4/6 in one batch, 2/6 in another); only within-batch relative
ordering is trustworthy.

### 6.16 Depth/seg probe for the task-adapted encoder (2026-06-12)

The cheap probe called for in §10 ("verify depth/seg output quality on the
cartoonish sim frames before investing"), now done. Script:
`tools/probe_depth_seg.py` — runs Depth Anything V2 Small plus a
Cityscapes-pretrained SegFormer on sim frames and writes
original|depth|seg|road-mask panels.

**Inputs.** 3 fixed-light frames from `data/vae_raw`, plus 5 random-light +
random-tree frames freshly captured (3 separate scene loads = 3 layouts,
`tools/collect_sim_frames.py`, cte-pid driving) including one worst-case frame
with dense dappled tree shadow across the road. Frames in
`data/probe_randomtree_frames/`, panels in `logs/probe_depth_seg*/` (both
local-only, gitignored).

**Segmentation — model size decides.** SegFormer-**B0** does not transfer to
the sim (road class 2-20% of pixels, inconsistent across frames, road confused
with sidewalk/terrain). SegFormer-**B4** transfers well: road 45-49% on
fixed-light frames, consistent, mask stops cleanly at the grass edge and
**covers the ego-car shadow region as road**. Under random tree shadows: still
clean at light/moderate shadow (road 53-55%), but the worst dappled-shadow
frame degrades (road 31%, holes in the mask under the densest patches) and one
frame bled road onto shadowed grass.

**Depth — shadow-immune but lane-blind.** Smooth ground-plane gradient on every
frame, tree trunks/cones pop out, and shadows are invisible by construction
(geometry, not appearance). But lane lines and the road/grass boundary are also
flat, hence invisible — depth alone cannot lane-keep; auxiliary channel only.

**Verdict.** The sim→real domain gap is manageable at B4 scale, and a
seg-derived road mask is exactly the shadow-invariant signal the frozen
encoders lack (§6.14-6.15 ceiling). Used raw, B4 would likely *raise* the ~50%
random-layout ceiling rather than remove it (worst-case dappled shadow still
breaks the mask), and B4 is too heavy for the ~20 Hz control loop. The
recommended path if this is picked up: **auto-label sim frames with B4
pseudo-labels and distill a small seg net** (data is free in the sim; also
solves inference cost), then feed mask (or mask+image) to SAC. Status:
probe positive, line validated as worth pursuing; no training done.

## 7. Practical Pitfalls

- Do not install the PyPI `gym-donkeycar` package — it targets the old `gym` API.
  Install upstream instead with `pip install git+https://github.com/tawnkramer/gym-donkeycar`,
  or clone that repo and `pip install -e ../gym-donkeycar` for local edits. See the
  README's gym-donkeycar section for details.
- Do not mix VAE data from visually different simulator tracks.
- On `donkey-generated-track-v0`, disable simulator `randomlight` before VAE data
  collection, SAC training, and evaluation. Otherwise lighting/domain shift makes VAE
  checkpoints non-reproducible across sim restarts. This issue does not affect
  `donkey-mountain-track-v0` or `donkey-generated-roads-v0`; DINOv2/ResNet encoders
  are also unaffected (see §6.12).
- Do not judge SAC only by training reward or SB3 rolling means.
- The simulator's Time scale and FPS limit menu options change results and are not
  visible to the Python client. Use Time scale 2 with FPS limit 120 for training and
  evaluation; raising Time scale without raising the FPS limit makes frame timing
  alternate (§12).
- Save replay buffers at checkpoints if resume matters.
- Do not resume SAC with replay buffers generated by a different reward function.
- **Eval action/termination params must match training.** `--max-steering-diff`,
  `--min-throttle`, `--max-throttle`, `--max-cte-error`, `--n-command-history`,
  `--encoder-crop-top` all change how the policy is mapped/terminated at eval; if
  they differ from training the results are not comparable (e.g. the 2026-05-29
  `eval_paired` clamp bug ran every paired eval at msd=0.15 vs training's 0.2, §6.15).
  The stale `MAX_STEERING_DIFF=0.15` module constant in `train_vae_sac.py` was
  fixed to 0.2 on 2026-06-12; all entry points now default to 0.2.
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

- **For new loop SAC runs**, the proven reward recipe is v3_h80 90k:
  `--encoder vae`, `--hidden-size 80`, safe_v2 reward defaults (`speed=0.15,
  cte_pen=0.25`), cold start, eval at every checkpoint. Use the best checkpoint
  from deterministic eval. Note: v3_h80 was trained with `--batch-size 64
  --gradient-steps-cap 2000`; current defaults are 128/1000 (see §6.11). A
  mountain DINOv2 run (§6.13) exercised raised optimizer settings and produced a
  deployable single-stage cold start, but used batch=256 + scale=0.5 — the actual
  128/1.0 defaults are still not cleanly validated. See README for the up-to-date
  training command.
- New `train_loop_vae_sac.py --encoder vae` checkpoints intentionally keep the
  historical `sac_loop_vae_*_steps.zip` prefix. Non-VAE encoders use
  `sac_{encoder}_*_steps.zip`.
- **Do not retune reward weights blindly.** v4 (`s=0.20, c=0.20`) and v5
  (`s=0.20, c=0.25`) were both deliberately tested and both produced worse
  deterministic-eval results than v3 (`s=0.15, c=0.25`). See §6.5 for the full ablation.
- **Compare runs by sustained metrics**, not "fastest single lap" (see §6.9).
- **For new ResNet/MobileNet runs use `--encoder-crop-top 0`**, not the legacy 40 that
  the older v4 ResNet was trained with. See §6.6.
- **On `donkey-generated-track-v0`, always disable simulator `randomlight`** before
  VAE collection / SAC training / eval. This was the root cause of the historical
  safe_v2 reproducibility failure. Mountain track and generated-roads are not affected
  by this; DINOv2/ResNet encoders are lighting-robust regardless of track.
- The old `safe_v2 70k` artifact remains a "best ever under matched lighting" reference
  point but is not a reproducible deployment model.
- **For robustness to randomized appearance (random light + trees/shadows), train with
  scene-reload domain randomization** (`--scene-reload-alpha 3 --scene-reload-kmin 200`),
  and eval with `--scene-reload-every 1` (fresh layout per episode). This gets DINOv2 to
  ~50% truncate on unseen random layouts (§6.14). Frozen ResNet18 was stopped at 20k
  with no laps completed and strong scene-to-scene shadow sensitivity, so DINOv2 is
  the better frozen encoder for this condition. **Do not chase higher robustness with
  reward tuning, frozen ResNet, or a bigger frozen DINO encoder** — these were shown
  not to help (§6.14); the next lever is a task-adapted encoder (fine-tune / depth /
  segmentation).
  **Update §12:** a probe and failure analysis lean toward missing dynamic state
  (speed / heading) instead; try that first, then spatial patch-token features.
- **Compare models by laps per episode on the same 30+ random layouts**, with
  intervals (§12). Completion counts on about 12 layouts cannot rank models.
- **The left-right steering weave is not fixable by reward penalties or a bigger
  frozen encoder** (§6.14) — so don't spend reward-tuning effort on it. Root cause
  undetermined (control high-gain vs frozen-DINOv2 lateral-perception weakness). A
  buffer probe showed the policy saturates the `max_steering_diff` clamp, so the
  one untried direct lever is **tightening that clamp** (0.2→~0.1); see §6.14.

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

data/vae_raw/generated_track_loop_fixedlight_*/
  Fixed-light loop VAE raw image captures. Current active loop VAE source data.

data/vae_loop_cones_fixedlight_v1/
  Prepared manifest/train/val split for the fixed-light loop VAE dataset.

data/vae/cache_loop_cones_fixedlight_v1/
  Cropped uint8 memmap cache used by `rl/train_vae.py` for the active fixed-light
  loop VAE encoder.
```

Deleted data categories:

```text
data/vae/cache_loop_cones_v1/
  Removed random-light loop VAE cache.

data/vae_raw/generated_track_loop_cones*/
  Removed random-light loop VAE raw images.

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

Partially exercised on 2026-05-28 (§6.13) but **not cleanly**: that run used
batch=256 + scale=0.5, not the actual defaults (batch=128, scale=1.0). It did
show late-stage drift persisting and the peak shifting only marginally later. A
clean validation of `--batch-size 128` / `--buffer-size 50_000` at `scale=1.0`,
ideally A/B against the old 64/30k baseline on the same env + seed, is still
pending.

### `--gradient-steps-scale` ablation

New CLI flag (default 1.0). `0.5` was run once on 2026-05-28 (§6.13) but
confounded with batch=256, so its effect can't be isolated. A clean 1.0-vs-0.5
A/B at the same batch on the same env / seed is still pending.

### Task-adapted encoder for randomized-environment robustness

Small historical batches in §6.14 reached ~50% truncate with a frozen encoder;
reward tuning and a bigger frozen encoder did not improve those runs. This is
not a proven encoder-only ceiling: §11 adds repeated-layout and captured-failure
evidence, and state/control limitations remain to be isolated. A candidate lever
is a **task-adapted encoder** — fine-tune DINOv2 (warm-start
from a competent policy, low LR, partial unfreeze; needs storing raw images in
the buffer + encoding in the SB3 features_extractor), or swap in a depth /
driving-segmentation model that is inherently shadow/lighting-invariant.
**Probe done 2026-06-12 (§6.16): positive.** SegFormer-B4 (Cityscapes) reads
the sim road through moderate tree shadow; depth is shadow-immune but
lane-blind. Remaining work if picked up: distill B4 pseudo-labels into a small
seg net (B4 itself fails on the densest dappled shadow and is too slow for the
20 Hz loop), then train SAC on the mask features.

### Dynamic-state input (next experiment, from §12)

Add speed, gyro or previous-frame features to the observation, train at the
standard simulator setting, and compare laps paired against v2 50k (baseline at
that setting: 6.08 laps, 31/40). Second lever: DINOv2 patch tokens pooled to a
grid instead of the class token.

Status 2026-11-12: the code for the second lever plus earlier frames exists
(`--encoder dinov2_vits14_grid --encoder-projection ... --latent-lags 2,4`,
projection from `tools/fit_grid_projection.py`). One run of it
(`models/rl_loop_dinov2_grid_lags_v1`) and one class-token control with the same
recipe and simulator setting (`models/rl_loop_dinov2_cls_ctrl_ts2_v1`) are
trained to 60k steps. **Neither has been evaluated**; there is no result yet.
Evaluated 2026-11-13 (§13): grid + lags 50k is **not** better than v2 50k (5.04
against 6.08 laps, 10/24 against 31/40 completed). It beats its own control, but
that control checkpoint is far below v2 50k with the same recipe, so the pair
says little about the observation. A run with simulator ground truth as the
observation laps 32% faster than v2 50k (§13); next is removing its start-line
weave and then using it as a teacher.

### Reproducible layouts and more throughput

`set_random_seed` is not handled on the client port; the private API port is
untried. Several simulator instances are untested; time scale 3 degrades v2 50k
(§12).

### Other open items

- Mountain v1's full checkpoint set is kept for now because v2 derives from
  v1 30k. If v2 stays stable across future work, v1 can be reduced to just
  the 30k zip + buffer.
- Loop deployment v3_h80 90k and v15 112488 remain the production models.
- ResNet mountain v1 (§6.8) is superseded but kept as historical reference.
- No further reward-weight ablations planned on loop (§6.5 closed the loop).
- No further hidden-size ablations planned on loop (h64 → h80 → h128 path
  fully explored).


## 11. Runtime revalidation and control diagnostics (2026-11-12)

All numbers below were measured locally; no policy was retrained. Simulator
settings, source snapshots, hashes, telemetry and plots are local-only artifacts
under `logs/`. The runs do not share reproducible Unity layout seeds.

### Baseline rechecks and recording infrastructure

- WSL2 NAT to the Windows simulator on TCP 9091 works; RTX 5080 CUDA inference
  works. No network-mode or firewall changes were needed.
- Fixed-light VAE `v3_h80` 90k: 5/5 at 3000 steps, 76 laps, mean speed 2.826,
  mean absolute CTE 0.311, mean lap 9.40s. This repeats one fixed scene, not a
  generalization test. Artifact: `logs/baseline_fixedlight_20261112_093541/`.
- DINOv2 randomtree v2 50k: 8/12 fresh random-light/tree layouts at 2000 steps,
  68 laps, layout-mean absolute CTE 0.397, mean lap 13.84s. Artifact:
  `logs/randomtree_baseline_12layouts_v2_50k/`. This small batch does not establish
  an improvement over historical batches on different random layouts.
- Evaluation now records JSON/CSV, exact encoder-input snapshots and optional
  telemetry/tail clips. Loop resume saves effective settings and checks replay
  configuration/hash; changed rewards require a fresh buffer. Historical
  checkpoints remain usable with an explicit missing-metadata warning. VAE
  resume preserves historical best validation loss. See README for CLI details.

### Repeated-layout diagnostic protocol

Same DINOv2 checkpoint twice per layout (`repeat_a`, `repeat_b`), shuffled order,
6 fresh layouts, 2000-step cap, CTE limit 2.0, steering-difference parameter 0.2.
Random light and trees on; random cones off. No scene reload between the two
repeats. Tail capture retains 120 returned encoder-input frames per episode;
image writing happens after the episode. These are repeats of one policy,
not competing models.

| Layout | repeat_a | repeat_b |
|---|---:|---:|
| 1 | 2000 / limit | 2000 / limit |
| 2 | 2000 / limit | 2000 / limit |
| 3 | 2000 / limit | 911 / CTE |
| 4 | 2000 / limit | 2000 / limit |
| 5 | 1502 / CTE | 964 / CTE |
| 6 | 2000 / limit | 2000 / limit |

9/12 episodes reach the cap, 74 laps total; four layouts succeed twice, one is
mixed, one fails twice. Do not treat this as twelve independent-layout trials or
rank `repeat_a` above `repeat_b`: their checkpoint hashes are identical.
Artifact: `logs/randomtree_control_repeat6_tail120/` (21,377 steps, 1,440 PNG tail
frames, 12 GIFs, plots/contact sheets under `control_analysis/`).

### Observed evidence and limits

- All three captured failures cross the negative CTE boundary, with no collision
  telemetry. The visible road/lane lines persist in the terminal images; CTE
  termination is not equivalent to leaving the whole paved road. Both layout-5
  failures occur at the same corner (terminal positions about 0.17 Unity units
  apart), but at different laps/steps. Layout 3 succeeds once and fails once.
- Tail P95 policy times are about 0.81–0.86ms, encoder 4.7–4.94ms, env step
  50.5–51.1ms across failures. No failure-adjacent client stall is evident. Env
  step includes telemetry waiting and encoding: these times do **not** measure
  network RTT, constant actuation delay or Unity acknowledgement latency.
- The mapped command often approaches +1 during the final departure. It is a
  command submitted to the client, not a measured wheel angle. The asynchronous
  simulator does not acknowledge per-command execution; frame k is returned by
  `env.step(command k)` and used for the next decision.
- Offline replay using saved frames plus the previous 20 mapped commands exactly
  reproduces all 15 sampled next actions (max absolute difference 0). This
  validates camera/history/policy alignment for those samples, not simulator
  command-to-frame causality. Script: `tools/verify_policy_alignment.py`
  (originally run from a copy inside the artifact directory; the tool reproduces
  the same numbers); output `alignment_check.json` in that directory.
- A limited offline sensitivity probe freezes command history and applies RGB
  gains 0.9/1.1 to those 15 frames. Across 30 perturbations, median absolute steer
  change is 0.0034, maximum 0.536, and no sign flips occur when the original
  magnitude exceeds 0.2. This is neither a closed-loop result nor a test of local
  tree-shadow invariance. Script: `tools/verify_policy_alignment.py
  --brightness-gains 0.9 1.1`; output `brightness_probe.json` in the same directory.

**Interpretation:** current evidence does not establish perception as the sole
bottleneck, and does not establish control as the sole cause either. Keep the
segmentation/fine-tuning path as a candidate. Prioritize a probe of whether frozen
features predict lane deviation/heading, plus a controlled state/speed-input
comparison, before committing to a heavier encoder training branch. A road-only
mask should not be assumed to preserve the right-lane target or yellow line.
The feature probe is done in §12.

**Validation:** 17 pytest tests pass, including bounded capture, image-copy and
index alignment, context isolation, exception capture, and the earlier resume
regressions. Live Unity capture and offline plot generation also pass. The
simulator remains in random-light/tree mode after these experiments.

## 12. Laps baseline with intervals, state probe, simulator speed-up (2026-11-12)

No policy was retrained. All runs use DINOv2 randomtree v2 50k
(`models/rl_loop_dinov2_randomtree_v2/sac_dinov2_vits14_50000_steps.zip`, crop 0),
random light and trees on, random cones off, 2000-step cap, CTE limit 2.0, a
fresh layout per episode. Artifacts are local-only under `logs/`.

### Primary metric

From here on the primary metric is **laps per episode** on fresh random
light/tree layouts with a fixed step cap, where leaving the track ends the
episode. It rewards speed and staying on the road together. Models are compared
on the same layouts (`rl/eval_paired_randomized.py`); the summary reports mean
laps with a bootstrap 95% interval, a Wilson interval on the completion rate, and
paired tests (`paired_statistics` in `rl/experiment.py`;
`tools/summarize_paired_eval.py` recomputes them for old or aborted runs). Do not
relax throttle range, CTE limit or step cap to raise the number.

### Baseline at four simulator settings

| Simulator setting | Layouts | Laps (95% CI) | Completed (95% CI) | Steps/s incl. reloads | Step interval median (p5-p95) |
|---|---:|---:|---:|---:|---:|
| Time scale 1, FPS limit 60 | 40 | 5.53 (4.72-6.22) | 27/40 = 68% (52-80%) | 18.8 | 50.0 ms (48.0-51.9) |
| Time scale 2, FPS limit 60 | 24 | 5.04 (4.04-5.96) | 13/24 = 54% (35-72%) | 35.0 | 20.1 ms (15.3-34.7) |
| Time scale 2, FPS limit 120 | 40 | 6.08 (5.42-6.62) | 31/40 = 78% (62-88%) | 35.7 | 25.1 ms (23.8-26.3) |
| Time scale 3, FPS limit 180 | 40 | 4.42 (3.50-5.33) | 22/40 = 55% (40-69%) | 48.2 | 16.7 ms (14.7-18.5) |

Artifacts: `logs/randomtree_baseline_40layouts_v2_50k/`,
`logs/randomtree_timescale2_24layouts_v2_50k/`,
`logs/randomtree_timescale2_fps120_40layouts_v2_50k/`,
`logs/randomtree_timescale3_fps180_40layouts_v2_50k/`. The four runs use
different random layouts (unpaired).

- The earlier 8/12 (§11) has a Wilson interval of 39-86%; twelve layouts cannot
  rank models.
- The same checkpoint scores 68%, 54% and 78% in the first three runs. Single-run absolute
  rates on 24-40 layouts are not comparable across runs; compare models paired on
  the same layouts.
- Time scale 2 with FPS limit 120 versus time scale 1: laps difference +0.55
  (95% CI -0.40 to +1.53, Mann-Whitney p=0.29). Lap time (13.43 s vs 13.47 s,
  first lap excluded), mean speed (1.925 vs 1.923), mean |steer change| (0.305
  both) and CTE spread are unchanged. Maximum |CTE| in completed episodes is
  slightly higher (+0.09, CI -0.01 to +0.20, p=0.057).
- Time scale 2 with FPS limit 60 versus time scale 1: laps -0.48 (CI -1.68 to
  +0.72), maximum |CTE| in completed episodes +0.14 (CI +0.00 to +0.27). Not
  significant, but every safety metric moves the wrong way.
- Time scale 3 with FPS limit 180 is **worse**, although its step intervals are
  steady: laps -1.65 against time scale 2 / FPS 120 (CI -2.73 to -0.58, p=0.009)
  and -1.38 against the first and third rows pooled (CI -2.40 to -0.34,
  p=0.012); lap time +0.18 s against time scale 1 (CI +0.07 to +0.27); 7 of the
  18 failures happen within the first 300 steps. The client's roughly 5 ms of
  encoder and policy time is about 15 ms of simulated time at this scale (5 ms
  at time scale 1), which is the likely cause; that this checkpoint loses more
  than a lap to it is consistent with the thin control margin described below.

### Can the car's position be read from the frozen features?

`tools/probe_state_from_features.py` fits ridge regressions from encoder features
to simulator ground truth and predicts only layouts the regression never saw
(grouped 5-fold; penalty chosen inside each training split). Data: the 120-frame
tails of the 40-layout time-scale-1 baseline and of the §11 repeat run, 6240
frames, 46 layouts, 16 failed episodes. Output: `logs/probe_state_v2_50k/`.

| Features | CTE R2 | CTE MAE | Lateral drift R2 |
|---|---:|---:|---:|
| Frame shrunk to 20x15 RGB (reference) | 0.73 | 0.21 | 0.47 |
| DINOv2 class token (the policy input) | 0.73 | 0.21 | 0.47 |
| Mean of patch tokens | 0.74 | 0.21 | 0.48 |
| Patch tokens pooled to a 4x4 grid | 0.93 | 0.11 | 0.59 |

CTE standard deviation in this data is 0.54. "Lateral drift" is the slope of CTE
against distance travelled over the last 5 steps, a stand-in for heading error
(telemetry has no track direction).

- The class token carries coarse lane position: MAE 0.21 against a limit of 2.0,
  and the correct side of the road in 99% of the 406 frames with |CTE| >= 1.
- It is no better than a 20x15 thumbnail. Keeping the spatial layout of the patch
  tokens halves the error.
- At the same distance from the centre line, class-token error is similar in
  completed and failed episodes (MAE 0.18 vs 0.19 for |CTE| < 0.5, 0.22 vs 0.25
  for 0.5-1, 0.33 vs 0.40 for 1-1.5). No sudden loss of the lane is visible in
  failed episodes. Beyond 1.5 (80 frames, failures only) the class-token probe
  underestimates (MAE 0.84); the patch grid does not (0.33).
- Lateral drift is poorly predicted from any single-frame feature. The label is
  noisy, so its attainable R2 is unknown; the result says only that one frame is
  clearly not enough.

Limits: linear probe; tail frames over-represent the approach to a failure; one
checkpoint; a readable feature does not prove the policy uses it.

### What the policy does before leaving the track

From telemetry of the 40-layout time-scale-1 baseline (13 failures, all CTE
limit, 3 within the first 300 steps). The corrective steering direction is
opposite in sign to CTE, as in the PID driver of `tools/collect_sim_frames.py`.

- In 9 of 13 failures the mean mapped steering over the last 5 steps is more
  than 0.7 of full lock toward the centre; the other 4 are between 0.20 toward
  and 0.27 away. As in §11 this is the submitted command, not a wheel angle.
- The departure is one sudden swing: 12 of 13 go from |CTE| > 1 to the limit in
  4-9 steps (0.2-0.45 s of simulated time); one drifts along the edge for 26
  steps. Earlier swings in the last 80 steps of the same episodes mostly peak
  below 1.0 (largest 1.6).
- 60% of all steps hit the steering-change clamp, in completed episodes too.

**Interpretation:** the evidence leans toward the control side. The policy has a
coarse position signal and reacts, but it has no usable signal for how fast it is
moving across the road, so it corrects late and at full lock with little margin.
This is not proven: 4 of 13 failures do not fit, and the probe limits above
apply. The next experiment is to give the policy dynamic state (speed, gyro or
previous-frame features) and compare laps paired against v2 50k. Spatially
pooled patch tokens are a second, independent lever.

### Simulator speed-up

- At time scale 1, 95% of evaluation wall time is the simulator's fixed 20 Hz
  frame clock: of each 50 ms step the encoder takes about 4 ms and the policy
  under 1 ms (§11 repeat run: stepping 1068 s of 1120 s; reload plus reset
  2.9-5.4 s per episode). Client-side optimisation cannot help.
- The simulator menu's **Time scale** speeds this up proportionally: at 2 the
  step interval halves while simulated time per step stays 0.05 s and lap times
  are unchanged.
- **Raise the FPS limit with it.** With time scale 2 and the default FPS limit
  60, step intervals alternate between about 17 ms and 33 ms (half each); with
  FPS limit 120 they are steady at 25 ms. The likely cause, inferred from the
  alternation and not verified in the simulator source for this build, is that
  frames are exchanged once per rendered frame, so a rendered frame at 60 FPS
  spans twice the simulated time.
- **Standard setting from now on: Time scale 2, FPS limit 120** for both training
  and evaluation (1.9x throughput: a 40-layout evaluation takes 33 min instead of
  58). Train and evaluate at the same setting. v2 50k was trained at time scale
  1; its baseline at the standard setting is the third table row.
- Evaluation `metadata.json` and training run configs now record
  `simulator_saved_settings` (time scale, FPS limit, random light/trees/cones),
  read from the Windows registry where the simulator saves its menu options.
  They are the saved values, not the live process state. The first three runs
  above predate this field.
- Not reproducible layouts yet: the installed build (v25.10.06) knows a
  `set_random_seed` message, but on the client port 9091 it logs "No delegates
  for msg type: set_random_seed" and layouts are not reproduced. The private API
  port (9092, key-protected) was not tried.
- Do not go beyond time scale 2 with policies trained at time scale 1 or 2 (see
  the table). Untested: several simulator instances on different ports
  (`--port`), the protocol's synchronous `step_mode`.

**Validation:** 25 pytest tests pass (paired statistics, latent lags and per-model observation in paired evaluation added).

---

## 13. Ground-truth state policy: is perception the limit? (2026-11-12/13)

**Status: trained and evaluated once (one seed per run, 24 paired layouts).**
Short answer: perception limits speed; it is not shown to limit completion.

### Question and setup

Two observation changes in one day (§10 status) gave similar training curves, so
the limit might be the SAC and control setup rather than what the policy sees.
Test: same recipe, simulator setting and seed as the class-token control, but
the observation is ground truth instead of the image
(`--encoder sim_state --track-centerline models/track_centerline/generated_track.npz`,
`rl/track_state.py`): CTE, heading error, forward and sideways speed, yaw rate,
and six centre-line points 0.3 to 3.0 position units ahead in the car's frame
(18 numbers) plus the usual 20-step command history. The run configurations
differ only in the encoder.

The simulator does not report the track, so `tools/fit_track_centerline.py`
recovers the centre line from telemetry of four §12 runs (223,542 points): 534
points, 26.7 position units per lap, 360 degrees of turning, tightest radius
0.85. Its signed distance times 7.98 reproduces the simulator's CTE with
R2 = 0.967 (MAE 0.06, up to about 0.15 off inside bends, exact on straights).
So the CTE limit of 2.0 is 0.25 position units. Telemetry conventions found on
the way: `vel` is in the car's frame (x sideways, z forward), yaw is `car[2]`
(0 faces +z, grows towards +x), `gyro[1]` is the yaw rate, positive CTE is right
of centre, and the track geometry is identical in every layout.

### Training-time comparison (stochastic actions; not an evaluation)

| Run (60k steps each) | Laps logged | Lap time, steps 50-60k (median / best in run) | Steps 50-60k spent in full 2000-step episodes | Mean throttle / forward speed, steps 40-50k |
|---|---:|---:|---:|---:|
| Ground truth (`models/rl_loop_sim_state_ts2_v1`) | 203 | 9.27 s / 8.98 s | 88% (4 of 5 episodes) | 0.47 / 2.56 |
| Class token control (`models/rl_loop_dinov2_cls_ctrl_ts2_v1`) | 146 | 11.88 s / 10.82 s | 40% (2 of 17) | 0.28 / 1.89 |
| Grid + lags 2,4 (`models/rl_loop_dinov2_grid_lags_v1`) | 132 | 12.89 s / 11.78 s | 39% (2 of 14) | 0.27 / 1.92 |

Source: `monitor.csv` and `train.log` in each model directory.

- The ground-truth run was **behind** both vision runs until about 40k steps
  (mean episode length 214 against 320 and 366 at 20-30k). No episode passed 255
  steps before 34k; at 20-30k, 32 of 47 episodes ended at 225-255 steps, one lap
  (about 10.6 s from standstill, 212 steps) plus about 28 steps. At its speed that is the first
  right-hand bend after the start line taken at full speed, the tightest on the
  track and the largest cluster of v2 50k failures (§12 runs: centre-line index
  71-79 of 534). The position is inferred from timing, not telemetry. Minimum
  throttle is 0.2, so slowing for that bend means lifting early on the straight.
- Once past it, the last six episodes are 2000, 1126, 2000, 2000, 2000, 2000
  steps, at lap times about 22% below the control.
- The vision policies sit near minimum throttle (0.27-0.28); the ground-truth
  policy uses 0.47. Mean |steering change| per step is about 0.28-0.32 in all
  three, so the jerky steering is not caused by vision.
- All 22 v2 50k failures in the two 40-layout §12 runs end at CTE of about -2:
  off the outside of a right-hand bend.

### Paired evaluation of the 50k checkpoints (2026-11-13)

`logs/paired_state_vs_vision_24layouts/`: 24 fresh random-light/tree layouts,
each model once per layout in random order, 2000-step cap, time scale 2 with FPS
limit 120 (step interval median 25.0 ms, p5-p95 23.7-26.2 for all three).

| Model | Laps (95% CI) | Completed (95% CI) | Flying lap time | Mean speed | Mean throttle | Mean abs steering change |
|---|---:|---:|---:|---:|---:|---:|
| Ground truth 50k | 9.42 (8.50-10.00) | 21/24 = 88% (69-96%) | 9.18 s | 2.89 | 0.44 | 0.26 |
| Grid + lags 50k | 5.04 (4.04-5.96) | 10/24 = 42% (24-61%) | 12.97 s | 1.99 | 0.25 | 0.29 |
| Class token control 50k | 1.92 (0.92-3.04) | 4/24 = 17% (7-36%) | 12.29 s | 1.97 | 0.29 | 0.31 |
| v2 50k, §12 run at the same setting (unpaired, 40 layouts) | 6.08 (5.42-6.62) | 31/40 = 78% (62-88%) | 13.47 s | | | |

Flying lap time excludes each episode's first lap (standing start).

- Ground truth against the vision models on the same layouts: +4.38 laps over
  grid + lags (CI +2.96 to +5.67; 13 layouts only it completed, 2 only the
  other, p=0.007) and +7.50 over the control (CI +5.92 to +8.79; 18 against 1).
- Ground truth against v2 50k (unpaired): +3.34 laps (CI +2.28 to +4.28) and
  lap time 32% lower. **Completion is not distinguishable**: 21/24 against
  31/40, Fisher p=0.51. Every completed ground-truth episode is exactly 10 laps;
  the gain is speed.
- How it is faster: it lifts to throttle 0.22-0.26 from about 1 position unit
  before the start line until the apex of the first bend (speed 3.4 down to
  2.4), then uses 0.6-0.7 on exits. Before that bend it moves from the right
  (+0.5 CTE) to the left (-0.8 mean). The vision policies hold throttle near the
  0.2 floor all lap.
- Its three failures are the same event: CTE -2.03 at centre-line index 41-48
  of 534, on that move to the left just before the first bend, at speed 2.8,
  after 2, 4 and 10 laps (3 of about 230 passes). The fitted centre line agrees
  with the simulator's CTE to 0.01 there, so a wrong lateral offset is not the
  cause; the 3-degree heading kink at the start line 2 units earlier is not
  excluded as a contributor.
- Steering stays jerky with perfect perception (0.26 per step against
  0.29-0.31), so that belongs to the control setup.
- Grid + lags 50k against v2 50k (unpaired): laps -1.03 (CI -2.16 to +0.10),
  completion 10/24 against 31/40 (Fisher p=0.007). No support for that
  observation change. 10 of its 14 failures are at CTE +2 (right side); every
  control and v2 50k failure is at -2.
- The control 50k has the v2 recipe yet scores 1.92 laps against 6.08. It was
  trained at time scale 2 (v2 at 1) and its 40-50k training window was its worst
  (no full episode). Run or checkpoint variance and training time scale are not
  separated. A single trained run per variant cannot rank observation changes.
- The ground-truth policy's own compute is 0.08 ms per step against 3.8-3.9 ms
  for the image encoders (about 8 ms of simulated time at time scale 2).

### Why the ground-truth policy still fails 3 of 24 (telemetry, 226 passes)

- It weaves on the start straight every lap, phase-locked to the track: median
  steering -0.59 at 30 centre-line points before the start line, +0.58 at 10
  before, -0.67 at 20 after; heading error swings between about +8 and -15
  degrees on a straight road. Heading-error spread is 9.5-10.9 degrees around
  the start line against 4.1-6.1 on the other straights.
- The last left swing comes just before the first right-hand bend. In the three
  failed passes it is full lock (-1.0) and the heading error reaches -24 to -28
  degrees (median pass -14, 5th percentile -22); full right lock from there does
  not bring the car back. 16 of the 223 successful passes also went below CTE
  -1.6 and 3 below -1.8, so the margin is thin every lap.
- The simulator's own CTE has a small jog at the start line (its zero line
  moves about 0.15 CTE and turns 2.75 degrees over about 1 position unit; the
  fitted line follows it to 0.01). Whether it sets off the larger weave there is
  not tested. The vision policy also weaves slightly more there (9.0-9.7
  degrees against 6.0-8.4 elsewhere).
- This settles the question left open in §6.14: the steering weave is present
  with perfect perception, so it comes from the control side, not from a noisy
  lateral estimate. Reward penalties on steering change did not remove it
  (§6.14); tightening `max_steering_diff` is still untried.
- Consequence: 88% is not good enough for a teacher. Fix the weave on the
  ground-truth policy first; each try costs one 36-minute training run without
  any vision in the loop.

### What it does and does not show

- Shown (one seed): with the same SAC recipe, action limits and reward, perfect
  perception gives a policy that brakes for bends and laps 32% faster than the
  best vision checkpoint. The speed ceiling of the vision policies is on the
  perception side, not in SAC or the reward.
- Not shown: that perfect perception makes driving more reliable. 88% against
  78% is within noise, and the failures come from one aggressive spot.
- Open: hidden size was left at 256 (a smaller network was not tried); the
  ground-truth run stalled for about 20k steps at the first bend before learning
  to lift early; one seed.

### Can image features predict the teacher's action? (offline, 2026-11-13)

`tools/probe_teacher_action.py`, output `logs/probe_teacher_action_state_50k/`.
The ground-truth 50k policy labels 18,020 captured frames (150 layouts, the five
§12 runs with tail capture) with the action it would take there; a 2x256 network
predicts that action from each candidate input plus the command history, scored
on layouts held out of its training.

| Input (+ command history) | Steering R2 | Steering MAE | Throttle R2 | Throttle MAE |
|---|---:|---:|---:|---:|
| Command history only | 0.03 | 0.57 | 0.36 | 0.28 |
| Teacher's own state (ceiling of the procedure) | 0.97 | 0.10 | 0.91 | 0.11 |
| Class token (v2 input) | 0.76 | 0.28 | 0.66 | 0.22 |
| Grid, PCA 128 | 0.86 | 0.22 | 0.68 | 0.21 |
| Grid + lags 2,4 | 0.85 | 0.22 | 0.66 | 0.22 |
| Grid, all 6144 | 0.80 | 0.26 | 0.55 | 0.26 |
| Grid + speed, sideways speed, yaw rate from telemetry | 0.87 | 0.21 | 0.72 | 0.20 |

Actions are in [-1, 1]; the teacher's steering has a standard deviation of 0.79.

- The grid features carry most of the teacher's steering decision (0.86 of a
  possible 0.97) and are clearly better than the class token (0.76). Earlier
  frames add nothing here.
- Linear read-out of the teacher's state from the grid features: sideways
  position of the road 0.6 to 3.0 units ahead R2 0.94-0.97, CTE 0.84, heading
  error 0.67, yaw rate 0.41, forward speed about 0. With lags: CTE 0.94, heading
  0.87, yaw rate 0.82, speed still 0.
- **The throttle part is not a fair test.** These frames come from v2 50k, which
  drives at 1.97 +/- 0.09; the teacher drives at 2.92 +/- 0.32 (2.3 to 3.4).
  Its braking decisions happen at speeds the frames never show, and speed
  cannot be judged readable or not when it barely varies. No frames of the
  teacher's own driving exist yet.
- An offline action error does not predict closed-loop driving. This only says
  imitation is not ruled out by missing information in the image features.

### Removing the ground-truth weave: `max_steering_diff` 0.2 -> 0.1 (2026-11-14)

- **Why `max_steering_diff=0.2` weaves on the start straight:** With `0.2` in the
  `[-1, 1]` action space (`|dsteer| <= 0.4` per 50 ms step), the `v1 50k` policy
  reaches full lock (`-1.0` or `+1.0`) in 3-4 steps on the high-speed start
  straight (`3.2-3.4 m/s`) in response to small `+0.1..+0.3` CTE offsets. Once at
  `-1.0` full left lock right after the start line (`idx 20..35`), unwinding to
  `+1.0` full right lock takes 5 steps (`0.25 s`), during which heading error
  overshoots to `-26.7..-27.5` degrees and carries the car across `CTE = -2.03`
  before the first right-hand bend (`idx 41..48`). By contrast, sustained cornering
  on the tightest bend (`idx 60..90`, radius 0.85) only requires `|dsteer| = 0.01..0.18`.
- **Single-variable test (`models/rl_loop_sim_state_ts2_sdiff01_v1`):** Trained
  `--encoder sim_state` for 60k steps at time scale 2, FPS limit 120 with
  `--max-steering-diff 0.1` (`|dsteer| <= 0.2` per step) and all other flags
  identical to `models/rl_loop_sim_state_ts2_v1`.
  - First 2000-step completion at 14k steps (against 35k in `v1`).
  - From step 20,730 to 60,730, completed **20 consecutive 2000-step training
    episodes (40,000 steps, ~185 laps) with 0 crashes**.
  - Screening `30k, 40k, 50k, 60k` (`logs/screen_sim_state_sdiff01_30k_60k`)
    selected `sac_sim_state_60000_steps.zip` (`mean_abs_cte = 0.163`,
    `max_abs_cte = 0.74`). One 50k episode in that screen hit a single 549 ms
    client-side pause (`elapsed_s` jumped by `0.549 s` while `env_step_seconds`
    was `0.3 ms`), during which the asynchronous simulator drove 2.63 m unsteered;
    no such pause occurred in the 24-layout evaluation below.
- **24-layout evaluation (`logs/sim_state_sdiff01_60k_24layouts`, `--capture-tail-steps 220`):**

| Ground-truth (`sim_state`) policy | Laps (95% CI) | Completed (95% Wilson CI) | Flying lap | Speed | Mean \|CTE\| | Max \|CTE\| | \|dsteer\| | Start-straight heading std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v1 50k` (`max_steering_diff=0.2`) | 9.42 (8.46-10.00) | 21/24 = 88% (69-96%) | 9.18 s | 2.75 | 0.30 | 2.04 | 0.26 | 9.5-10.9 deg |
| `sdiff01 60k` (`max_steering_diff=0.1`) | **9.00 (9.00-9.00)** | **24/24 = 100% (86-100%)** | **10.35 s** | **2.52** | **0.158** | **0.770** | **0.162** | **2.11 deg** |

- **What changed on the road (48,000 steps, 216 laps):**
  - Start-straight heading-error spread dropped from `9.5-10.9` degrees to
    **`2.11` degrees** (`5th..95th` percentile `-4.2..+2.6` degrees), matching the
    other straights (`2.08` degrees).
  - Entering the first bend (`idx 40..49`), CTE is `-0.62` median (`5th`
    percentile `-0.70`, worst `-0.77`), leaving at least **`1.23` CTE units of
    margin** on every pass. Across all 48,000 steps, `|CTE|` 99th percentile is
    `0.635` and maximum is `0.770`.
  - Giving up the wide outside-left swing trims flying lap speed from `9.18 s`
    (`10` laps/episode) to `10.35 s` (`9` laps/episode, `best_lap = 10.10 s`),
    which is still **23% faster than vision `v2 50k` (`13.47 s`)** while raising
    mean episode reward from `3278` to **`3553.2`**.

### Offline teacher-action & state probe with `sdiff01 60k` (2026-11-14)

1. **On the 150 layouts / 18,020 frames from the five §12 vision runs**
   (`logs/probe_teacher_action_sdiff01_60k_v2frames/probe_results.json`):
   - Teacher action std: steer `0.725`, throttle `0.652`.
   - With command history: `state` ceiling $R^2 = 0.956$ steer / $0.927$ throttle;
     `grid` (PCA 128) reaches $R^2 = 0.803$ steer / $0.749$ throttle (vs `cls`
     $0.710 / 0.731$); `grid + motion` (`speed, sideways_speed, yaw_rate`) reaches
     **$R^2 = 0.833$ steer (`MAE = 0.222`) / $0.767$ throttle (`MAE = 0.230`)**,
     and $R^2 = 0.743$ on frames with `|CTE| >= 1.0`.
   - Without command history (`grid+motion (no hist)`): steer $R^2 = 0.637$
     (vs `grid (no hist)` $0.192$), because under `max_steering_diff=0.1` the
     teacher's raw pre-clamp output depends on where the previous steering command
     `steer_{t-1}` was (`66.8%` of steps saturate the `0.1` step clamp). A student
      therefore needs at least the previous steering command (or should predict the
     post-clamp applied steering target).
2. **On the teacher's own 24 layouts / 5,184 frames**
   (`logs/probe_teacher_action_sdiff01_60k_self/probe_results.json`, captured via
   `--capture-tail-steps 220` so each episode retains more than a full 10.35 s lap):
   - Because speed now varies across every lap (`2.59 +/- 0.19 m/s` after standing
     start), linear ridge read-out of the teacher's state from `grid + lags` on
     held-out layouts reads:
     - `cte`: **$R^2 = 0.95$**, `sin heading error`: **$R^2 = 0.87$**,
       `sideways speed`: **$R^2 = 0.86$**, `yaw rate`: **$R^2 = 0.93$**,
       `forward speed`: **$R^2 = 0.59$** (was `0.00` on constant-speed `v2 50k`
       frames), and `road 0.3..3.0 ahead: right`: **$R^2 = 0.96 - 0.98$**.

**Next (Step 3):** Train a vision student policy from the stabilized
`sdiff01 60k` teacher (`models/rl_loop_sim_state_ts2_sdiff01_v1/sac_sim_state_60000_steps.zip`)
via teacher rollout collection + online DAgger in the simulator, using `grid`
(PCA 128) + motion telemetry (`speed, sideways_speed, yaw_rate`) + command history
(predicting either the post-clamp applied action or regularizing command history so
closed-loop driving stays responsive to visual features).

**Validation:** 32 pytest tests pass (track state, projection hashing, alignment verification, cache fingerprint, linear lag 15D splicing, and paired residual schedule contracts).

### Stage 3.1: 15-dim geometry perception benchmark (offline, 2026-11-14)

`tools/probe_geometry_15d.py` (single-frame Round 1) and
`tools/probe_geometry_15d_lags.py` (multi-frame `lags=(2, 4)` Round 2), output
`logs/probe_geometry_15d_v1/` (verified via `feature_cache.meta.json` fingerprint
`b3d6ee0e9814`). Across all 174 `(run, layout)` groups (150 vision episodes from §12
+ 24 `sdiff01 60k` teacher episodes; 23,924 total frames, 23,204 valid lagged
frames), each model predicts the 15 geometric dimensions of `TrackState.encode`
(`[0]=cte/2`, `[1:3]=sin/cos(heading_error)`,
`[6:18]=6 lookaheads (right, forward)/lookahead`) under strict 5-fold `GroupKFold`
by `(run, layout)`. All feature standardizations, PCA projections, ridge tuning, and
MLP early stopping are fit strictly inside each training fold. Predicted geometry is
spliced into `[0:3]` and `[6:18]` alongside ground-truth motion `[3:6]`
(`vel[2]/2, vel[0]*4, gyro[1]/2`) and the 40-dim command history to evaluate
teacher-controller action substitution error (both raw pre-clamp and executed
post-clamp command error under `max_steering_diff=0.1`).

Units: CTE is in simulator CTE units (termination limit `+/-2.0`; `1 position unit`
is `~7.98 CTE units`), heading is in degrees (circular angle difference after
unit-circle normalization), and lookahead error is in simulator position units.

| Candidate (on 23,204 valid lagged frames) | CTE MAE / P95 (CTE) | CTE `w20` Bias P95 (CTE) | Heading MAE / P95 (deg) | Lookahead 1.0 MAE / P95 (pos) | ExecSteer MAE / P95 | ExecThrot MAE / P95 | `\|CTE\|>=1` CTE MAE / P95 (CTE) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cls_ridge (1f)` (`384 -> 15`) | 0.193 / 0.517 | 0.506 | 3.74 / 10.17 | 0.056 / 0.122 | 0.044 / 0.289 | 0.031 / 0.104 | 0.483 / 1.158 |
| `pca128_ridge (1f)` (fold-fit `PCA-128 -> 15`) | 0.159 / 0.425 | 0.434 | 3.12 / 8.36 | 0.040 / 0.089 | 0.037 / 0.258 | 0.021 / 0.079 | 0.397 / 0.966 |
| `raw6144_ridge (1f)` (`6144 -> 15`) | 0.090 / 0.243 | 0.244 | 2.17 / 5.94 | 0.026 / 0.054 | 0.023 / 0.163 | 0.013 / 0.049 | 0.205 / 0.550 |
| `raw6144_b64_mlp (1f)` (`6144 -> 64 -> 15`, 3-ens) | 0.087 / 0.243 | 0.238 | 1.95 / 5.60 | 0.026 / 0.054 | 0.022 / 0.160 | 0.012 / 0.047 | 0.224 / 0.561 |
| `pca128_lags_ridge` (`3x128=384 -> 15`) | 0.092 / 0.246 | 0.313 | 1.86 / 4.86 | 0.038 / 0.083 | 0.033 / 0.232 | 0.018 / 0.070 | 0.235 / 0.715 |
| `proj15_lags_ridge` (`3x15=45 -> 15` linear, 1 model) | 0.055 / 0.151 | **0.168** | **1.21 / 3.24** | **0.025 / 0.052** | 0.021 / 0.148 | 0.011 / 0.044 | 0.126 / 0.342 |
| `proj15_lags_mlp` (`3x15=45 -> 64 -> 15`, 3-ens) | 0.059 / 0.162 | 0.184 | 1.19 / 3.16 | 0.024 / 0.051 | 0.021 / 0.149 | 0.011 / 0.043 | 0.141 / 0.374 |
| `shared64_lags_mlp_1m` (single `6144->64, 192->64->15`) | 0.063 / 0.173 | 0.206 | 1.35 / 3.70 | 0.028 / 0.062 | 0.023 / 0.164 | 0.012 / 0.047 | 0.142 / 0.398 |
| `shared64_lags_mlp_ens3` (3-split inner ensemble) | **0.052 / 0.146** | 0.175 | 1.22 / 3.40 | 0.025 / 0.053 | **0.020 / 0.145** | **0.011 / 0.040** | **0.122 / 0.324** |
| `raw18432_lags_ridge` (`3x6144=18432 -> 15`) | 0.055 / 0.146 | 0.171 | 1.22 / 3.30 | 0.027 / 0.056 | 0.022 / 0.154 | 0.012 / 0.044 | 0.133 / 0.393 |

- **Unsupervised `PCA-128` drops ~44% of geometric precision:** Replacing fold-fit
  `PCA-128` with a supervised linear projection on the full `6144`-dim `4x4` patch
  grid cuts single-frame CTE P95 from `0.425` to `0.243 CTE` (`0.966 -> 0.550 CTE`
  on `|CTE| >= 1.0`) and heading P95 from `8.36 deg` to `5.94 deg`.
- **Single-model `proj15_lags_ridge` beats single-model `shared64_lags_mlp_1m` on
  every metric:** `shared64_lags_mlp_ens3` (`0.052 / 0.146 CTE`) averages 3 inner
  models per fold; a single `shared64_lags_mlp_1m` scores `0.063 / 0.173 CTE` (`w20`
  bias P95 `0.206 CTE`, heading P95 `3.70 deg`, `|CTE|>=1` P95 `0.398 CTE`). By
  contrast, the deterministic linear `proj15_lags_ridge` (`6144 -> 15` per frame,
  then `3x15=45 -> 15` across `lags=(2, 4)`) achieves **`0.055 / 0.151 CTE`**,
  **`w20` bias P95 `0.168 CTE`**, and **`Heading MAE/P95 = 1.21 / 3.24 deg`** as a
  single two-matrix affine transform, matching full `18,432`-dim Ridge without
  materializing an `18,432`-dim vector.
- **Exported deployable head:** `models/dinov2_geometry_15d/proj15_lags_ridge.pt`
  (loaded via `DinoGridLinearLag15Encoder` in `rl/geometry_encoder.py`).

### Stage 3.2: Empirical closed-loop tolerance screen with 1-step delay (2026-11-14)

`tools/eval_teacher_tolerance.py`, output
`logs/teacher_tolerance_screen_v1/tolerance_results.json`. Evaluated the frozen
`sdiff01 60k` teacher across 4 randomized light/tree layouts x 6 conditions in
shuffled order per layout (2000 steps max). Multi-step 15-dim residual blocks (length
80) are sampled from the held-out OOF predictions of Stage 3.1; heading residuals are
added as circular angles on the unit circle.

| Condition | Geom Delay | Residual Source | Comp | Mean Steps | Mean Laps | Mean \|CTE\| (CTE) | Max \|CTE\| (CTE) | Speed | Eff. CTE P95 (CTE) | Eff. Head P95 (deg) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 0 steps | None (exact `TrackState`) | **4/4** | 2000.0 | 9.00 | 0.159 | 0.743 | 2.53 | 0.000 | 0.00 deg |
| `delay1_only` | 1 step | None (1-step geom buffer) | **4/4** | 2000.0 | 9.00 | 0.236 | 1.691 | 2.50 | 0.181 | 4.25 deg |
| `shared64_lags_1x_d1` | 1 step | `shared64_lags_mlp_ens3` (`1.0x`) | **2/4** | 1289.2 | 5.50 | 0.270 | 2.227 | 2.37 | 0.258 | 5.96 deg |
| `raw6144_1f_1x_d1` | 1 step | `raw6144_ridge (1f)` (`1.0x`) | **0/4** | 1539.5 | 6.75 | 0.289 | 2.223 | 2.47 | 0.309 | 7.36 deg |
| `pca128_1f_1x_d1` | 1 step | `pca128_ridge (1f)` (`1.0x`) | **0/4** | 240.0 | 0.50 | 0.333 | 2.150 | 2.07 | 0.497 | 9.87 deg |
| `pca128_1f_1.5x_d1` | 1 step | `pca128_ridge (1f)` (`1.5x`) | **0/4** | 224.5 | 0.25 | 0.381 | 2.253 | 2.02 | 0.616 | 12.55 deg |

- **What this 4-layout screen shows:**
  - A 1-step geometry buffer delay alone (`delay1_only`) introduces `0.181 CTE` P95
    and `4.25 deg` heading P95 phase lag in corners, raising `max_abs_cte` from
    `0.743` to `1.691 CTE` (still 4/4 completed).
  - Stacking 1-step delay with `1.0x` held-out residual blocks stratifies the
    representations (`pca128_1f_1x_d1`: 240 mean steps, 0/4; `raw6144_1f_1x_d1`:
    1539.5 mean steps, 0/4; `shared64_lags_1x_d1`: 1289.2 mean steps, 2/4).
- **Important limitations (why this screen does not establish a "safe zone"):**
  1. Only 4 layouts were tested, and all residual conditions included `1-step` extra
     geometry delay (`_d1`); no `0-extra-delay` (`_d0`) residual control or live
     visual closed-loop was included in this run.
  2. Conditions in `logs/teacher_tolerance_screen_v1` used per-condition RNG seeds
     (`layout_idx * 1000 + cond_idx`) and replayed residual blocks across unrelated
     track positions. `tools/eval_teacher_tolerance.py` has now been updated to use a
     single pre-sampled `(block_id, offset)` schedule per layout across all
     conditions, incremental per-episode/per-step saving (`telemetry.jsonl`,
     `episodes.jsonl`, `metadata.json`), `proj15_lags_1x_d0` (0 extra delay), and
     live closed-loop evaluation of `models/dinov2_geometry_15d/proj15_lags_ridge.pt`
     (`vision_proj15_lags_live`).

### Stage 3.3: Paired `d0` vs `d1` + live vision 4-layout pilot (2026-11-15)

`tools/eval_teacher_tolerance.py --mode vision_and_d0 --layouts 4`, output
`logs/vision_proj15_and_d0_v1/tolerance_results.json` (using shared per-layout residual
block schedules across conditions).

| Condition | Geom Delay | Residual / Perception | Comp | Mean Steps | Mean Laps | Mean \|CTE\| (CTE) | Max \|CTE\| (CTE) | Speed | Ep-Avg Eff. CTE P95 | Ep-Avg Eff. Head P95 | Enc ms P50 / P95 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 0 steps | Exact `TrackState` | **4/4** | 2000.0 | 9.00 | 0.157 | 1.155 | 2.52 | 0.000 | 0.00 deg | 0.10 / 0.13 |
| `proj15_lags_1x_d0` | 0 steps | OOF `proj15_lags_ridge` | **4/4** | 2000.0 | 9.00 | 0.188 | 1.176 | 2.54 | 0.137 | 3.01 deg | 0.10 / 0.14 |
| `proj15_lags_1x_d1` | 1 step | OOF `proj15_lags_ridge` | **1/4** | 1062.0 | 4.25 | 0.251 | 2.135 | 2.40 | 0.255 | 5.56 deg | 0.10 / 0.14 |
| `vision_proj15_lags_live` | 0 steps | Live DINOv2 + `proj15_lags_ridge` | **4/4** | 2000.0 | 9.00 | 0.181 | 1.428 | 2.55 | 0.135 | 2.80 deg | 4.11 / 4.69 |

- **Zero extra delay rescues closed-loop stability (`d1` `1/4` -> `d0` `4/4`):** Removing
  the synthetic 1-step delay cuts effective CTE P95 from `0.255` to `0.137 CTE` and
  eliminates 3/4 crashes.
- **Live vision tracks `proj15_lags_1x_d0` closely on 4 layouts:** `4/4` completed (36
  laps total), speed `2.55 m/s` vs baseline `2.52 m/s`, though `mean|CTE|` is `+15.7%`
  higher (`0.181` vs `0.157 CTE`) and Layout 2 peaked at `-1.428 CTE` at step 372
  (`eff_cte_err = +0.250 CTE` underestimate following 9 consecutive `max_steering_diff=0.1`
  rate-limited left-turn steps at `2.98 m/s`).

### Stage 3.4: Round A — 24-layout paired live vision eval & pre-registered split (2026-11-15)

`tools/eval_teacher_tolerance.py --mode round_a_vision_paired --layouts 24 --train-pool-layouts 12 --save-captures --output-dir logs/round_a_vision_24layouts_v1`

- **Frozen model:** `models/dinov2_geometry_15d/proj15_lags_ridge.pt` +
  `models/rl_loop_sim_state_ts2_sdiff01_v1/sac_sim_state_60000_steps.zip`.
- **Pre-registered split (`split_manifest.json`):** Layouts `1..12` = `dagger_train_pool`,
  Layouts `13..24` = `locked_eval_holdout` (reserved exclusively for final evaluation,
  never used for retraining).
- **Full captures saved:** `captures/layout_*.npz` storing full-episode RGB frames,
  15-dim ground-truth/predicted geometry, 3-dim motion, 40-dim command history,
  student actions, counterfactual `teacher(true_state)` actions, and monotonic control-chain
  timestamps (`decision_obs_rx_mono_s` -> `act_sent_mono_s`, `step_dt_ms`).

| Scope | Condition | Comp | Mean Steps | Mean Laps | Mean \|CTE\| (CTE) | Max \|CTE\| (CTE) | `\|CTE\|>=1.0` Steps | Speed | Pooled CTE MAE / P95 / P99 (CTE) | Pooled Head P95 / P99 (deg) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **All 24 layouts** (`47,297` steps) | `baseline` | **23/24** (`95.8%`) | 1970.7 | 8.875 | 0.1585 | 2.086 | 6 | 2.522 | `0.000 / 0.000 / 0.000` | `0.00 / 0.00` |
| **All 24 layouts** (`46,153` steps) | `vision_proj15_lags_live` | **23/24** (`95.8%`) | 1923.0 | 8.625 | 0.1979 | 2.061 | 53 | 2.512 | `0.054 / 0.140 / 0.194` | `3.08 / 4.44` |
| `dagger_train_pool` (L1..12) | `baseline` / `vision` | `11/12` vs `11/12` | 1941.4 vs 1846.1 | 8.75 vs 8.25 | 0.1584 vs 0.2065 | 2.086 vs 2.061 | 6 vs 45 | 2.523 vs 2.482 | `0.055 / 0.140 / 0.199` | `3.18 / 4.69` |
| `locked_eval_holdout` (L13..24) | `baseline` / `vision` | **`12/12` vs `12/12`** | **2000.0 vs 2000.0** | **9.00 vs 9.00** | 0.1585 vs 0.1893 | 1.119 vs 1.300 | **0 vs 8** | 2.520 vs 2.541 | `0.054 / 0.141 / 0.190` | `3.01 / 4.18` |

- **Paired delta (`vision - baseline`, all 24 layouts):** `mean_laps_difference = -0.25`
  (`[-0.75, 0.00]`, `sign_test_p = 1.0`, `only_a = 0, only_b = 0`);
  `mean_abs_cte_diff = +0.0395 CTE` (`[+0.0306, +0.0502]`, `1.249x` ratio);
  `max_abs_cte_diff = +0.2401 CTE` (`[+0.1422, +0.3560]`). On `locked_eval_holdout`,
  `vision` achieved `12/12` (`mean_laps_diff = 0.0`), `mean|CTE| diff = +0.0308 CTE`
  (`1.194x` ratio), and `mean_speed_diff = +0.0207 m/s`.
- **Timing over 46,153 live vision steps:** `encoder_ms` P50/P95/P99/max =
  `4.17 / 4.67 / 5.54 / 81.89 ms` (only 1 spike > 12.25 ms across 46,153 steps, at L06
  step 1021); `policy_ms` P50/P95 = `0.51 / 0.64 ms`; full `decision_obs_to_act_ms`
  P50/P95/P99/max = `4.70 / 5.20 / 6.13 / 82.50 ms` (well within the `25.15 ms` P50
  simulator `step_dt_ms`).
- **Regime breakdown & root gap motivating Round B (`vision_proj15_lags_live`):**
  - `standing_start_1_250` (`5,903` steps): `mean|CTE| = 0.175` (baseline `0.151`),
    `eff_cte MAE/P95 = 0.049 / 0.127 CTE`, `eff_head P95 = 3.06 deg`.
  - `cruise_abs_cte_lt_0_5` (`43,520` steps): `eff_cte MAE/P95 = 0.053 / 0.138 CTE`,
    `eff_head P95 = 3.04 deg`.
  - `moderate_offset_0_5_to_1_0` (`2,580` steps): `eff_cte MAE/P95 = 0.065 / 0.172 CTE`,
    `eff_head P95 = 3.60 deg`.
  - **`large_offset_ge_1_0` (`53` steps, 0.11% of steps):** perception error jumps **4.5x**
    to `eff_cte MAE/P95/max = 0.243 / 0.622 / 0.873 CTE` (`eff_head P95 = 6.01 deg`),
    with strong center-shrinkage underestimation (`+cte`: `-0.312 CTE` mean error;
    `-cte`: `+0.126 CTE`). Worst non-fatal excursion: Layout 3 step 375 (true `+1.737`,
    predicted `+0.864 CTE`).
- **Layout 7 failure analysis:** Both `baseline` (step 1297, `cte=-2.086`) and
  `vision` (step 153, `cte=+2.061`) failed only on Layout 7 (`dagger_train_pool`).
  In `layout_07_vision` steps 145–153, counterfactual `teacher(true_state)` already
  commanded `-1.000` full left alongside the student after crossing `cte = 0` at
  `+25 deg` heading (`max_steering_diff=0.1` saturated during full reversal), confirming
  that Layout 7 contained tight curves near the controller's physical steering-rate limit
  once `|CTE|` swung above `1.0`.

**Next (Round B):** Collect targeted off-center / recovery supervision on fresh training
layouts (`dagger_train_pool` only; never `locked_eval_holdout`), combine with Round A
train-pool frames to enrich `|CTE| in [0.5, 1.9]`, retrain/compare the geometry perception
head, and verify recovery + lateral tracking gains on `locked_eval_holdout`.

### Round A Layout 7 Root-Cause Diagnosis & Minimal Control-Loop Fixes

#### 1. Root-Cause Evidence (Teacher vs. Vision on Layout 7)
Deep inspection of `logs/round_a_vision_24layouts_v1/telemetry.jsonl` and `gym_donkeycar`
(`donkey_sim.py`, `client.py`) showed that `layout_07_baseline` (`step 1297`, `cte=-2.086`,
Lap 6) and `layout_07_vision_proj15_lags_live` (`step 153`, `cte=+2.061`, Lap 0) failed at
different times and trajectories from **distinct upstream triggers** feeding into a **shared
downstream rate-limit reversal bottleneck**, rather than an unavoidable corner physics limit:

- **`layout_07_baseline` (Step 1297, Lap 6) — Control-Cycle Stall + TCP Burst Replay +
  Persistent `dt` Beating -> Rate-Limited Reversal Trap:**
  - Across Laps 0–4 (`steps 249..268, 456..475, 664..683, 868..887, 1074..1093`) on the
    exact same corner (`pos=(6.15..6.19, 4.30..6.90)`), `baseline` tracked cleanly with
    `step_dt_ms in [23.48, 26.81] ms` (`std = 0.53..0.88 ms`), mild steering (`[-0.34, +0.29]`),
    and `max|CTE| <= 0.311` across all 5 laps — ruling out a physical corner limit.
  - Only `layout_07_baseline` among all 12 `dagger_train_pool` baseline episodes exhibited
    elevated cycle jitter (`bad_dt = 22` steps outside `[18, 32] ms`, `dt_std = 2.70 ms` vs
    `0.73..0.87 ms` in the 11 surviving layouts).
  - At `step 1263`, `step_dt_ms` spiked to `93.42 ms` (`sim_wait_ms = 92.69 ms`), followed at
    `steps 1264..1266` by 3 buffered telemetry packets draining in `5.58, 4.40, 2.87 ms`.
    Because `dp / (speed * 0.05)` stayed `0.99..1.00`, Unity had stepped `0.05 s` sim-time per
    packet while `SDClient` (`time.sleep(0.001)` + per-frame loop) replayed all 3 stale buffered
    frames back-to-back, advancing `last_steer` and `command_history` on stale states. From
    `step 1267..1297`, `step_dt_ms` remained trapped in `11.80..36.51 ms` beating
    (`dt_std = 14.81 ms`), amplifying lateral swing (`+0.227 -> -0.146 -> -0.307 -> +0.079`).
  - At `step 1286` (`cte=+0.069`), `executed steer` peaked at `-0.72` (left). Over `steps
    1287..1290`, as `cte` crossed `-0.011 -> -0.449` and heading flipped `-2.9° -> -17.4°`,
    raw policy demanded `+0.10 -> -0.20 -> +0.49 -> +0.85`, **but `max_steering_diff=0.1`
    (`0.20/step`) held executed steering negative (`-0.52, -0.32, -0.12`) for 3 steps**
    (two with `step_dt_ms = 35.40, 32.11 ms`), taking 9 steps (`1286..1295`) to go
    `-0.72 -> +0.99` and exiting at `-2.086`.
- **`layout_07_vision_proj15_lags_live` (Step 153, Lap 0) — Visual Oscillation + Underestimate
  -> Same Downstream Reversal Lag:**
  - Had an early timing disturbance (`steps 11..18`), then `105` consecutive normal-timing steps
    (`steps 49..153`, `step_dt_ms in [23.44, 26.49] ms`) before crashing.
  - Its excursion arose from Lap-0 curve tracking oscillation (`+0.707` at `s=121` -> `-1.098`
    at `s=144` with `+0.117 CTE` visual underestimate at `s=143`), followed by the same
    downstream `max_steering_diff=0.1` reversal bottleneck (`exe = +0.66 -> +0.46 -> +0.26 ->
    +0.06` across `s=144..146` while raw policy commanded `-0.46, -0.91, -1.00`, overshooting
    zero at `+25.1°` heading to `+2.061`).
- **`gym_donkeycar` race & polling bugs found (`donkey_sim.py` / `client.py`):**
  1. `on_telemetry` assigns `self.time_received = time.time()` at line 508 **before** updating
     `pos / speed / vel / yaw / cte / image_array` (torn-read race).
  2. `observe()` and `SDClient.proc_msg()` spin on `time.sleep(0.001)` polling and replay all
     stale queued `"telemetry"` frames inside a burst instead of coalescing TCP bursts.

#### 2. Minimal Fixes Implemented (Teacher & Vision Head Weights Frozen)
- **Fix A — Event-Driven, Torn-Read-Free, Burst-Coalescing Sync (`rl/donkey_sync.py`):**
  `LowJitterSyncController` (toggleable via `set_enabled(bool)` for paired A/B eval):
  - Updates `image_array` and all state fields atomically under `threading.Condition` before
    incrementing `_telemetry_seq` and notifying `observe()` (zero `1 ms` sleep polling, zero torn reads).
  - Uses non-blocking socket + pipe wakeup fd (`os.pipe()`) for sub-millisecond `control` sends.
  - Within each TCP `recv()` burst, preserves `"hit"` / `"over"` / non-telemetry events from
    superseded packets via `coalesce_telemetry_messages()` while decoding and waking `observe()`
    only on the latest telemetry frame (`coalesced_dropped`, `max_burst_seen`).
- **Fix B — Gated Asymmetric Opposite-Sign Unwind-to-Zero (`rl/train_vae_sac.py`):**
  Added configurable `DonkeyVaeSACEnv` parameters (`reversal_unwind_scale=2.0`,
  `reversal_cte_threshold=0.40`, `reversal_head_deg_threshold=8.0`, `reversal_min_last_steer=0.30`):
  - Uses **only observation geometry** (`last_pred_geom` for vision or telemetry `[0:3]` for
    teacher — zero privileged ground-truth leakage for vision).
  - Activates **only** when `(steer * last_steer < 0)`, `|last_steer| >= 0.30`, and
    `(|cte_obs| >= 0.40 or |head_obs| >= 8.0°)`.
  - Computes `eff_max_diff = max(max_diff, min(scale * max_diff, abs(last_steer)))`:
    speeds return **only toward `0.0`** without skipping past `0.0` in a single step, leaving
    centered driving (`98%+` of steps), same-sign turning, and small-angle dithering (`<0.30`)
    untouched (`100%` identical to baseline). Notice an initial `v1` variant without
    `reversal_min_last_steer=0.30` and with `|last_steer| + max_diff` caused `±0.40`
    zero-crossing ping-pong under stress (`logs/teacher_fix_recovery_stress_v1`), which `v2`
    eliminated (`logs/teacher_fix_recovery_stress_v2`).

#### 3. Paired Closed-Loop Verification (`useSeed=0` Fresh Random Layouts; Round A Holdout Untouched)
**Caveat (`useSeed=0`):** With `useSeed=0`, Unity generates fresh random track geometry per reload;
new evaluations cannot replay the exact Round A Layout 7 geometry, so paired comparisons evaluate all
conditions on the same freshly loaded layouts per run. Unit tests: `pytest -q` -> **38 passed**
(including `tests/test_donkey_sync_and_ctrl.py`).

**A. Paired Natural Evaluation (`--mode teacher_fix_paired --layouts 12`, 48 episodes, `logs/teacher_fix_paired_v2`):**

| Condition | Comp | Mean Steps | Mean Laps | Mean \|CTE\| (CTE) | P95 / Max \|CTE\| | Speed | `step_dt_ms` P50/P95 | `bad_dt` Total / 11-ep `dt_std` (excl. L03 freeze) | `coal` / `unwind` | Excursions Recovered |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` (stock sync, no unwind) | 12/12 | 2000.0 | 9.00 | 0.1599 | 0.464 / 0.858 | 2.526 | 24.75 / 25.89 | 16 / 0.845 ms | 0 / 0 | 1/1 (med 5.0 st) |
| `teacher_sync_only` (Fix A only) | 12/12 | 2000.0 | 9.00 | 0.1586 | 0.460 / 1.437 | 2.522 | 25.00 / 25.69 | 14 / 0.636 ms | 2 / 0 | 6/6 (med 7.5 st) |
| `teacher_fixed` (Fix A + Fix B v2) | **12/12** | **2000.0** | **9.00** | 0.1643 | 0.482 / 1.425 | 2.520 | **25.00 / 25.69** | 86 (17 excl. L03)* / **0.540 ms** | 2 / 220 | **9/9** (med **5.0 st**) |
| `vision_fixed_ctrl` (frozen head + A + B v2) | 11/12 | 1950.0 | 8.75 | 0.1851 | 0.497 / 2.153 | 2.531 | 25.00 / 25.69 | 61 / 1.139 ms | 2 / 277 | 23/24 (med 6.0 st) |

- \*Note: `teacher_fixed` on `L03` experienced an external `2295.8 ms` (`2.3 s`) stall (`step 1709`)
  and 50-step host jitter storm (`bad_dt=69`, `coal=2, burst=3`), recovered **8/8** excursions,
  and still completed all 2000 steps (`max|CTE|=1.425`). Across the other 11 layouts, Fix A + Fix B
  cut `step_dt_ms` episode `std` by **36%** (`0.845 -> 0.540 ms`) and locked P50 cleanly onto `25.00 ms`.
- **GPU contention sensitivity finding (`logs/teacher_fix_paired_v1`):** In an earlier `v1` run
  when the Windows host GPU (RTX 5080) was at `82–94%` load (`234–255 W`), `vision_fixed_ctrl`
  DINOv2 `encoder_ms` P50/P95 degraded 3–5x (`15.39 / 24.51 ms`, up to `33 ms`), pushing
  `obs_to_act > 25 ms` step period and causing `3/12` completion (`teacher_fixed` still went `12/12`,
  `max|CTE| = 1.024` vs `baseline` `1.465` and `teacher_sync_only` `11/12`). Once host GPU load
  returned to idle (`v2`), `vision_fixed_ctrl` ran at `enc P50/P95 = 4.33 / 5.33 ms`, improving
  `mean|CTE|` from `0.1979` (Round A) to `0.1851` (`11/12` PASS).
- **Remaining vision bottleneck:** The single `vision_fixed_ctrl` failure (`L04 step 1400`,
  `cte=-2.153`) still showed large-offset visual underestimation (`err=+0.238` at `cte=-1.170`
  growing to `+0.513` at `-2.153`), confirming Round B perception retraining remains necessary.

**B. Paired Recovery Stress Evaluation (`--mode teacher_fix_recovery_stress --layouts 8`, 24 episodes, `logs/teacher_fix_recovery_stress_v2`):**
Deterministic paired disturbance schedule on every layout (`Kick 1`: steps `320..322` `±0.70` steer;
`95 ms` TCP stall: step `900`; `Kick 2`: steps `1300..1302` `∓0.70` steer). Note `nvidia-smi` was at
`~94%` host GPU load during this run (`vision_fixed_ctrl` `enc P50/P95 = 19.36 / 30.26 ms`):

| Condition | Comp | Mean Steps | Mean Laps | Mean \|CTE\| | Speed | Post-Stall (step 900) Deaths | Excursions Recovered |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 0/8 | 779.2 | 3.12 | 0.1995 | 2.350 | **4 / 5** reaching s=900 | `15/23` (`65.2%`) |
| `teacher_fixed` | **3/8** | **1199.4** (+420.1) | **5.25** (+2.12) | **0.1973** | **2.394** | **0 / 5** reaching s=900 | **`23/28` (`82.1%`)** |
| `vision_fixed_ctrl` *(host GPU 94%)* | 0/8 | 369.2 | 1.00 | 0.3618 | 2.170 | — | `56/64` (`87.5%`) |

Under identical disturbances, `teacher_fixed` survived Kick 1, the `95 ms` stall, and Kick 2
completely on 3 layouts (`L01`, `L03`, `L06`), eliminated all post-stall deaths (`0/5` vs `4/5` in
`baseline`), and raised overall excursion recovery from `65.2%` to `82.1%` without harming steady-state
tracking or speed.

#### 4. Audit Follow-Up: Non-Blocking Send Fix, Real TCP Backlog Smoke Test & Entrypoint Wiring
Addressing post-`v2` review findings before full 24-layout / Round B runs:
1. **Non-blocking socket send correctness (`rl/donkey_sync.py`):** Replaced `sock.sendall()`
   inside `_proc_msg_loop` with `sock.send(memoryview(_inflight_bytes)[_inflight_offset:])`, tracking
   unsent payload bytes and byte offset under `_send_lock` (across both `enabled=True` and
   `enabled=False`). Verified in `tests/test_donkey_sync_and_ctrl.py` (`42 passed` in `pytest -q`)
   under chunked partial sends + `BlockingIOError`, mid-flight second-message queueing, real kernel
   `SO_SNDBUF`/`SO_RCVBUF` backpressure, and `pause_recv` coalescing.
2. **Control stall vs. real TCP receive-backlog stress (`tools/eval_teacher_tolerance.py`):**
   Clarified that the earlier `v2` step-900 stall (`time.sleep(0.095)`) paused only the control thread
   (`coalesced_dropped = 0`), leaving the socket reader active. Added `--stress-type {ctrl_stall,tcp_backlog,both}`,
   per-episode `sync_controller.reset_stats()`, `max_burst_seen` tracking for `enabled=False`, and
   `pause_recv(0.095)`. In a 1-layout live simulator smoke test (`logs/smoke_tcp_backlog_v1`,
   `1050` steps, GPU idle `5%`, `enc P95 = 4.84–5.15 ms`), `pause_recv(0.095)` produced genuine
   TCP bursts (`max_burst_seen = 3..4`): sync-enabled conditions coalesced and dropped `2–3` stale
   frames (`coalesced_dropped = 2..3`) whereas stock `baseline` had `coalesced_dropped = 0` and
   replayed stale buffered frames in `1.28 ms` / `1.77 ms` bursts (`steps 901..902`).
3. **Complete within-run paired ablations & opt-in entrypoint integration:**
   - Added `--mode teacher_fix_paired_full` and upgraded `teacher_fix_recovery_stress` to run all
     5 conditions side-by-side (`baseline`, `teacher_sync_only`, `teacher_fixed`, `vision_baseline`,
     `vision_fixed_ctrl`) on identical tracks.
   - Wired opt-in control flags (`--low-jitter-sync` default `False`, `--reversal-unwind-scale`
     default `1.0`) via shared `add_control_fix_args` / `configure_env_control_fixes` helpers into
     `rl/train_vae_sac.py`, `rl/train_loop_vae_sac.py`, `rl/eval_vae_sac.py`,
     `rl/eval_loop_vae_sac.py`, and `rl/eval_paired_randomized.py`, preserving `environment_config()`
     and `validate_model_config()` compatibility for existing checkpoints.
   - **Deferred to next session (paused by user):** Running `--mode teacher_fix_paired_full --layouts 24`
     and 12-layout 5-condition stress evaluations under idle GPU before moving to Round B.

### Stage 3.5: Round B — First Geometry Recovery Collection Batch (2026-11-16)

**Scope decision:** Proceed with recovery supervision using the frozen `sdiff01 60k`
teacher, low-jitter sync enabled, original `max_steering_diff=0.1`, and
`reversal_unwind_scale=1.0` (acceleration disabled). The unfinished five-condition
evaluation is deferred, not a prerequisite for collecting this batch. Neither the
teacher nor the existing visual head was retrained or overwritten.

`rl/recovery_collection.py` adds bounded steering interventions to the existing
capture pipeline. `tools/eval_teacher_tolerance.py --mode round_b_collect` runs
only `teacher_recovery`, always enables full captures, and assigns every newly
loaded scene to `dagger_train_pool`. Existing Round A holdout data is not loaded.
Collection does not require the old perception probe cache.

**Collection procedure:**
- Preserve the first 250 steps without interventions to cover startup/acceleration.
- Start an intervention only when `|CTE| < 0.3`, `|heading| < 5 deg`, speed is at
  least `1.8`, the 1.0/1.5-position lookahead lateral coordinates are small, and
  the teacher's raw steering magnitude is below `0.35`.
- Override raw steering to `+/-0.35` for at most 4 steps, preserving the teacher's
  throttle and the original action mapping/rate limit. Alternate signed release
  targets of `0.6, 1.0, 1.4 CTE`; these are release thresholds, not guaranteed
  achieved offsets. Release earlier at `|CTE| >= 1.65` or `|heading| >= 15 deg`.
- Return control to the unchanged teacher. A recovery requires 8 consecutive
  observations with `|CTE| < 0.3` and `|heading| < 5 deg`; then wait 80 steps before
  another intervention. Unfinished recoveries time out after 160 steps.

**Calibration:** `logs/round_b_recovery_pilot_v1` used stronger/longer interventions
(`0.65`, up to 8 steps) and both episodes ended at the CTE limit. After reducing
interventions and adding the straight-road gate,
`logs/round_b_recovery_pilot_v2` collected 3,368 frames over four scenes, with
18/20 interventions recovered and 73 frames at `|CTE| >= 1`. Both calibration runs
are excluded from the main batch manifest.

**Main run:**

```bash
python tools/eval_teacher_tolerance.py --mode round_b_collect \
  --layouts 12 --max-episode-steps 2000 --device cpu \
  --output-dir logs/round_b_recovery_collect_v1
python tools/summarize_recovery_collection.py logs/round_b_recovery_collect_v1
```

| Coverage / outcome | Count |
|---|---:|
| Newly loaded scenes / captured frames | 12 / 15,187 |
| Unperturbed startup frames (steps 1..250) | 3,000 |
| `0.5 <= |CTE| < 0.8` | 999 |
| `0.8 <= |CTE| < 1.0` | 235 |
| `1.0 <= |CTE| < 1.5` | 321 |
| `1.5 <= |CTE| < 1.9` | 96 |
| `|CTE| >= 1.9` | 23 |
| Positive / negative frames with `|CTE| >= 1` | 294 / 146 |
| Interventions / recovered / CTE-limit exits / horizon-censored | 104 / 94 / 9 / 1 |

The 440 large-offset frames are `2.90%` of this collection. They are correlated
trajectory samples, not 440 independent recovery trials; signed coverage is still
uneven. These episodes deliberately contain perturbations, so the 9 terminal
episodes are not a natural-driving completion benchmark. Speed P10/P50/P90 is
`2.22 / 2.53 / 2.87`; there was no change to the throttle policy. Twelve initial
RGB-frame hashes are distinct, confirming distinct captured appearances.

**Artifacts and validation:**
- `logs/round_b_recovery_collect_v1/captures/`: approximately `0.89 GB` of full
  RGB images, same-frame 15D geometry labels, motion, histories, actual commands,
  counterfactual teacher actions, phase/event IDs, and timestamps.
- `coverage.json`: full-frame shape, CTE/geometry/motion consistency, command-history
  alignment, unchanged nominal actions/throttle, and original steering-rate limit
  checks passed. Same-frame teacher labels versus the next decision's teacher
  action differ by at most `1.31e-6`.
- `dataset_manifest.json`: capture hashes, scene groups, frozen teacher/head hashes,
  calibration exclusions, and explicit `round_a_locked_holdout_used=false`.
  `source/` preserves the exact five source files fingerprinted at run start.
- The teacher SHA256 remains `8d7ca02cf1db3f9d5fe09b7a9e35db04ca7852df16ca413341a27225e14c3e6d`;
  the existing head remains `80b3b1b343f01b8ec6b90f17fb5bdc34a531f3f7a08cc9f28d9f3e8048b59628`.
- `pytest -q`: **47 passed**. New checks cover intervention release, bounded duration,
  startup/corner gates, failed-event accounting, and frozen sync-only configuration.

**Training contract:** `images[k]`, `true_geom[k]`, and `history_post[k]` describe the
same post-step frame. `teacher_true_raw_action_on_frame[k]` labels that frame;
`student_raw_action[k]` is the command that led to it, using the preceding frame.
The `student_*` names are inherited capture fields: this run is teacher-driven
with interventions, not a completed student-on-policy DAgger iteration. Heading
and lookahead labels remain fitted-centerline estimates, with the limitations
described earlier; CTE units are not metres.

**Next:** Fit the same two-stage linear geometry head using training-pool data,
with layout-separated internal validation and explicit reporting on large-offset
and startup subsets. No new perception head or closed-loop improvement is claimed
by this collection result.

### Stage 3.6: Round B — Same-Architecture Geometry Head Retraining (2026-11-16)

`tools/train_geometry_recovery.py` retains the frozen DINOv2-S 4x4 grid and the
existing `6144 -> 15`, `(t,t-2,t-4) 45 -> 15` affine architecture. It combines the
23,924-frame legacy feature cache, Round A **training-pool captures only**, and
the new recovery collection. Before feature extraction/fitting, layouts 10..12
of each capture run are assigned to internal validation. Both Round A conditions
from the same layout stay in the same split. The original Round A locked holdout
files are never opened; calibration collection runs are excluded.

```bash
python tools/train_geometry_recovery.py
```

Artifacts: `logs/round_b_geometry_train_v1/{training_manifest.json,feature_cache.npz,results.json,predictions.npz}`;
candidate: `models/dinov2_geometry_15d_round_b_v1/proj15_lags_ridge.pt`.
The baseline head is preserved at its original path. The manifest fingerprints
all source captures, legacy cache, and baseline weights. Each full-episode capture
includes its reset frame for correct lag context. Evaluation below uses the same
valid-lag mask for both heads (first four frames of each sequence excluded).

There are **69,008 fitting frames across 192 layout groups** and **15,589 internal
validation frames across six groups** (15,553 valid-lag frames). All fitted weights
and standardization parameters use the fitting set only. Five-fold grouped
regularization selection and Stage-1 OOF predictions remain inside that fitting
set; these are not final-test results. Fitting/export took **13.52 s**, excluding
feature extraction and data preparation.

| Internal validation subset | Frames | Baseline CTE MAE / P95 | Round B CTE MAE / P95 | Baseline / Round B heading P95 (deg) |
|---|---:|---:|---:|---:|
| All | 15,553 | 0.0534 / 0.1353 | 0.0396 / 0.0995 | 2.90 / 2.30 |
| Cruise (`\|CTE\| < 0.5`) | 14,650 | 0.0516 / 0.1313 | 0.0377 / 0.0944 | 2.86 / 2.26 |
| Startup (step <= 250) | 2,223 | 0.0481 / 0.1195 | 0.0412 / 0.1042 | 2.75 / 2.48 |
| Large offset (`\|CTE\| >= 1`) | 117 | 0.1828 / 0.6244 | 0.1588 / 0.5672 | 6.84 / 5.84 |
| Positive large offset | 80 | 0.2069 / 0.7004 | 0.1743 / 0.5824 | 7.11 / 6.86 |
| Negative large offset | 37 | 0.1306 / 0.3350 | 0.1254 / 0.3667 | 4.74 / 4.02 |

Overall CTE P95 improves by **26.4%**, but large-offset P95 improves by only
**9.2%**, and negative-large-offset CTE P95 regresses. The small, correlated
large-offset subset does not establish robust recovery. A fresh 8-layout paired
natural-driving screen was then run, with the teacher, throttle/action
mapping, sync enabled, and reversal acceleration disabled in **both** conditions.
Only the visual head changes. Full captures are marked evaluation holdout.

```bash
python tools/eval_teacher_tolerance.py --mode round_b_vision_paired \
  --candidate-geometry-head models/dinov2_geometry_15d_round_b_v1/proj15_lags_ridge.pt \
  --layouts 8 --max-episode-steps 2000 --device cuda \
  --output-dir logs/round_b_vision_paired_8layouts_v1
```

**Completed paired screen (16 episodes):**

| Metric | Baseline head + sync | Round B head + sync |
|---|---:|---:|
| Completion | **8/8** | **7/8** |
| Mean laps | 9.000 | 8.875 |
| Mean episode `abs(CTE)` | 0.18005 | 0.18003 |
| Mean speed | 2.531 | 2.533 |
| Maximum `abs(CTE)` | 1.648 | 2.060 |
| Frames at `abs(CTE) >= 1` | 37 | 53 |
| Pooled perception CTE P95 | 0.1382 | **0.1003** |
| Pooled perception heading P95 | 2.87 deg | **2.33 deg** |
| Pooled encoder P50 / P95 | 4.25 / 4.96 ms | 4.27 / 5.10 ms |

Online perception improvements agree with offline validation, but the fixed
controller does not show better mean lateral tracking. The paired mean CTE
difference is `-0.000016 CTE` (95% bootstrap interval `[-0.01630, +0.01530]`).
The candidate does **not** pass the pre-agreed screen criterion of no fewer
completions; this small screen also does not statistically establish an intrinsic
model regression. All episodes and timing outliers are retained in the primary
comparison, with no post-hoc exclusion of difficult runs.

**Failure/timing observations (association, not an isolated causal test):**
- Candidate L04 recovered from `CTE=-1.634` at step 1567. Its CTE prediction error
  at that peak was only `+0.019`; a `72.33 ms` encoder stall at step 1515 preceded
  the later oscillation (`72.95 ms` observation interval at step 1516).
- Candidate L06 exited at step 1717 (`CTE=-2.060`). A `131.89 ms` observation
  interval at step 1677 was followed by short/irregular intervals and oscillation.
  At exit, CTE prediction error was `-0.105` and heading prediction error was
  `-6.48 deg`; the failure cannot be reduced to CTE underestimation alone.
- Baseline L08 also had large timing disturbances (maximum observation interval
  `380.30 ms`, maximum encoder `193.39 ms`), reached `CTE=-1.648`, and recovered.
  Timing disturbances therefore affect both conditions; the results neither
  isolate timing as the sole cause nor identify the source of the stalls.

`timing_diagnostics.json` retains per-episode peaks and their steps;
`layout06_candidate_failure.png` plots CTE estimates, applied steering, and timing.
`audit.json` confirms 8 complete pairs, identical sync-only control, preserved
motion inputs, aligned histories/actions (maximum next-action difference
`1.31e-6`), evaluation-only captures, and unchanged weight hashes. The original
Round A locked holdout remains unused. Source snapshots are retained alongside
both the training and evaluation runs. **50 pytest tests pass**, including checks
for held-out file exclusion, layout grouping, batch/live lag equivalence, and
paired-head control configuration.

**Decision:** `decision.json` marks the Round B head **not promoted**. Preserve
`models/dinov2_geometry_15d/proj15_lags_ridge.pt` as the baseline; retain the candidate
(`SHA256 6fd7b4aabf25f86b9b8c932e812411034d64c3f0f2d23edaeb3f65ed4fa02657`)
for diagnosis. Do not expand this candidate to the 24-layout final evaluation or
claim a closed-loop improvement. Teacher weights and reversal settings were not
changed. The remaining issue is distinguishing timing/feedback sensitivity from
perception residual structure before choosing another training change.

### Stage 3.7: Remote Desktop / Host Timing Diagnosis — Active Baseline (2026-11-16)

At the user's request, pause further training and investigate whether remote
desktop activity contributes to timing stalls. No remote service was stopped,
no desktop session was disconnected, and no system/GPU settings were changed.

**Process identification correction:** An initial search for only `Sunlogin|Oray`
found the legacy `SunloginService` stopped and missed the active `AweSun.exe`
processes. Windows GPU-engine counters identified PID 15512 as `AweSun`, using
two VideoEncode engines (each roughly 2..3% in the initial snapshots), and
`AweSunService` was running. AweSun is the branding on the
[official English remote-desktop site](https://sunlogin.oray.com/en/).
PID 15512 was created at `2026-11-16 11:06:26 +08:00` and was still alive during
diagnosis, covering the earlier evaluation's time window. This establishes
process presence, not continuous remote-streaming activity throughout that run.
The attempted Service Control Manager event query returned no matching state
events, so no historical stream-on/off conclusion is drawn from it.

`tools/profile_remote_timing.py` adds a read-only diagnostic:
- Uses the same recorded RGB frames and unchanged baseline geometry head at a
  paced 25 ms period, including preprocessing, DINOv2, temporal head, CPU output,
  and CUDA synchronization. It does not drive or query the simulator.
- Independently samples selected Windows process CPU/memory counters and raw
  per-engine GPU utilization approximately once per second. Records UTC times,
  sampling cost, wake lateness, and skipped benchmark slots.
- Never stops services or changes remote settings. The phase label is supplied
  by the operator; observed remote VideoEncode activity is logged independently.

```bash
python tools/profile_remote_timing.py \
  --output-dir logs/remote_timing_diagnosis_v1/remote_active \
  --phase remote_active --duration 120
```

**Observed active baseline:** 4,801 perception calls; P50/P95/P99/max
`4.22 / 5.18 / 5.88 / 9.07 ms`; zero calls above 25 ms and zero skipped slots.
Wake-lateness P99/max was `0.197 / 0.250 ms`. All 122 Windows samples contained
AweSun VideoEncode activity; the maximum observed utilization of an individual
remote encode engine was 3%. The host sampler reported no errors; its wall-time
cost P50/P95 was `353 / 358 ms` per roughly 1-second sample, asynchronously from
perception. These coarse GPU counters cannot attribute individual millisecond
stalls. Raw data and `summary.json` are in the run directory above.

**Limits / pending comparison:** Remote streaming and fast perception coexist
in this 2-minute recorded-frame benchmark. It neither establishes remote desktop
as the cause of earlier stalls nor excludes a contribution under live rendering.
Unity camera capture, TCP reception, actual driving, and historical host counters
were not measured by this test. The subsequent disconnected/reconnected driving
comparisons and their limits are recorded in Stage 3.8 below.

### Stage 3.8: Bounded Runtime Diagnosis — Remote Session, TCP and WSL Clocks (2026-11-16)

All driving below uses the original vision head and frozen 60k teacher, low-jitter
sync enabled, steering difference limit 0.1, and reversal-unwind scale 1.0. No
teacher retraining or candidate promotion was performed. Artifacts are under
`logs/remote_root_cause_v1/`.

| Run | Result | Timing evidence |
| --- | --- | --- |
| `gc_paired_remote_disconnected` | Stock and deferred GC both 3/3 x 2,000 steps | Encoder max 10.97 / 12.48 ms; observed GC max 2.09 ms, no generation-2 collections |
| `continuous_remote_transition` | 10,000/10,000 steps, 48 laps | Despite its name, streaming stayed off throughout; observation interval max 26.39 ms |
| `long_natural_gc` | Failed at step 3,709, 17 laps | Streaming active from the first host sample; observation interval 234.51 ms at step 3,336, encoder max 9.05 ms, GC max 1.15 ms |
| `sync_stage_remote_connected` | 2/2 x 2,000 steps | Fine receive/decode/publish instrumentation installed; max interval 37.98 ms, earlier 234 ms stall not reproduced |
| `transport_paired_remote_connected` | TCP default 1/2; TCP_NODELAY 2/2 | Two fresh layouts with reversed condition order; same model and controls |

The disconnected runs total 22,000 steps. This sequential comparison makes remote
activity a useful lead, not a controlled demonstration that AweSun caused the
stall. In the connected failure, the 234 ms interval preceded the exit by 373
steps; neither a sole cause of that exit nor a Windows GPU-driver cause is proven.
The independent Linux heartbeat recorded no >3 ms wake delay around that event.
It uses the same monotonic clock, so it cannot alone exclude a VM pause that also
affects the clock. Later heartbeat records contain eight 3.0..4.4 ms delays.

**Observed transport issue:** The SDK does not set TCP_NODELAY. A socket snapshot
showed 184 queued send bytes, including 92 not yet transmitted, one unacknowledged
segment, and approximately 20 ms TCP RTT despite sub-millisecond ICMP ping. This
establishes small-command buffering; it does not establish the cause of the
earlier 234 ms interval. During the paired run, 9/469 sampled default-condition
socket snapshots contained unsent bytes (maximum 92), versus 0/250 sampled
TCP_NODELAY snapshots. Sampling started after the first NODELAY episode, so only
its second episode has queue coverage. Disabling Nagle did not materially change
the sampled approximately 19.5 ms RTT. The default-condition failure in this run
had no observation interval above 40 ms, so its outcome cannot validate a fix for
the earlier large stall. Two pairs are insufficient for a reliability claim.

Fine timings use `time.perf_counter()`. A message-sent timestamp means accepted
by the local socket, not transmitted or processed by Unity. Publish-to-return
frame age can include time before observe was called; actual observe handoff is
`return - max(published, observe_start)`.

**Clock check:** This machine uses the `tsc` clocksource; Python perf_counter and
monotonic both use CLOCK_MONOTONIC, while time.time uses CLOCK_REALTIME.
chronyd runs with `-x`, which disables clock adjustment despite the active NTP
service status. Inspected user/system timers and Windows scheduled tasks showed
no matching forced-time-sync task; this is not a complete audit of all host code.
Ordinary wall-clock steps do not step CLOCK_MONOTONIC
([Linux clock documentation](https://man7.org/linux/man-pages/man3/clock_settime.3.html)).
Frequency correction and virtual-machine timing anomalies are separate concerns.

`clock_probe_v1` sampled Linux realtime, monotonic, raw and boottime 12,031 times
over 122 seconds, plus Windows UTC/QPC 1,094 times over 120 seconds. No Linux
clock increment differed from monotonic by more than 0.025 ms; raw-versus-monotonic
elapsed drift was 0.043 ms. Windows UTC-versus-QPC elapsed difference was 0.722 ms.
This short non-driving sample did not reproduce the reported periodic clock
jump. Windows-to-WSL arrival offsets include PowerShell startup and pipe latency
and must not be interpreted directly as clock drift.

A regression test simulates unchanged timestamps, a one-minute backward step,
and a one-hour forward step while receiving real socket-pair telemetry. Enabled
low-jitter observe continues to receive each fresh frame. This verifies the
application's sequence-based wakeup behavior, not WSL or Unity clock correctness.
GC callback records and the background writer are also covered; all 54 tests pass.

**Hard stopping rule requested by the user:** One final original-model driving
run, at most 4,000 steps and a 170-second process limit, with a concurrent
132-second clock probe. No further conditions, retraining, or automatic
extension. If a stall cannot be attributed with concurrent evidence, stop and
report the root cause as unconfirmed. See `stopping_rule.json`; final artifacts
are `clock_live_stopcheck_v1/` and `clock_live_stopcheck_clocks_v1/`.

**Final stop check and decision:** The single final episode completed 4,000 steps
and 19 laps in 100 seconds. Maximum observation interval was 34.54 ms, with no
interval above 40 ms. Maximum |CTE| was 1.949, so completion does not demonstrate a
large safety margin. Concurrent clock sampling covered the full driving run:
13,023 Linux samples over 132 seconds, with no >1 ms discrepancy between clock
increments; the largest was 0.037 ms. Raw-versus-monotonic elapsed drift was
0.048 ms. Windows UTC/QPC elapsed difference was 0.566 ms across 1,186 samples.
Neither the large stall nor a clock jump reproduced. The investigation stopped
at the agreed limit. The historical 234 ms stall's root cause remains
**unconfirmed**; do not label AweSun, WSL clock correction, GC, or Nagle as its
proven cause. Preserve the original model/control baseline, keep TCP_NODELAY as a
diagnostic condition, and do not launch additional diagnosis or retraining
automatically. Machine-readable closure: `bounded_diagnosis_result.json`.

### Stage 3.9: Opt-in Runtime Protection Implementation (2026-11-16)

Following the bounded diagnosis, the user authorized preventive implementation,
not further root-cause experiments or teacher training:

- Shared training/evaluation flags now include `--tcp-nodelay` and
  `--observation-timeout SECONDS`. Existing defaults remain unchanged. The
  timeout requires low-jitter sync; invalid negative/non-finite values fail.
- Observation waiting uses a monotonic deadline. A missing or already-stale
  observation latches `ObservationTimeoutError`, requests zero throttle and full
  handbrake, and prevents subsequent control commands/late frames from silently
  resuming the run. Explicit reset clears the latch. The emergency send preserves
  the existing partial-packet buffer; it never discards half a JSON message.
- The request is best-effort. Diagnostics distinguish socket acceptance from
  acknowledged execution, and do not claim that Unity actually braked. This is
  a client observe-wait guard, not an independent Windows/Unity watchdog; it
  cannot cover a whole-WSL freeze, a blocked policy/encoder before observe, or
  an unreachable/hung simulator. No Unity server source is present in this repo.
- Recorded evaluations save `runtime_failure.json` and a failed run status,
  retaining previous completed episodes. The custom live evaluator also flushes
  the interrupted episode's returned-step telemetry and records its context.
  Do not silently exclude runtime-aborted runs from reliability reporting.
- `vision_baseline_runtime` runs just the original live vision head and frozen
  teacher, with low-jitter sync and no reversal acceleration. Its layouts default
  to evaluation-only; it needs no residual-probe cache. Use explicit
  `--tcp-nodelay --observation-timeout 0.10` for the documented protected recipe.
  The 100 ms threshold is a starting operational cutoff, not a validated safe
  controller error budget. README documents the command and fixed simulator
  settings; no remote-service or display settings were changed automatically.
- Scene-reload timeouts (production and standalone diagnostic) and frame
  collection elapsed-time/rate calculations now use monotonic clocks.

**Validation:** All 66 tests pass. New tests cover real localhost TCP_NODELAY,
normal telemetry followed by silence and a brake request, a stale unconsumed
frame, the latched failure/reset behavior, disabled-guard compatibility, invalid
timeouts, wall-clock-independent scene-reload deadlines, separate runtime-failure
records, and the single-condition baseline configuration. CLI help exposes the
new flags/mode and `git diff --check` passes. No new live simulator experiment,
Windows-side brake execution test, or model training was run for this change.

### Stage 3.10: Live Runtime-Guard Acceptance (2026-11-16)

`logs/runtime_guard_acceptance_v1/` contains a bounded two-case smoke test using
the original live vision head and frozen teacher, timeScale 2/FPS 120, low-jitter
sync, TCP_NODELAY, and a 100 ms observation timeout. The ignored run wrapper
records the injected treatment; it does not change production defaults or weights.

- **Normal:** 400/400 steps in 10 seconds, one lap, max |CTE| 0.656, no timeout.
- **Injected missing observations:** Pause socket reception for 200 ms after the
  220th control message. The guard fired after 100.145 ms and aborted the run
  after 217 returned steps. `runtime_failure.json` reports the stop request
  flushed to the socket and keeps execution acknowledgement false.
- The wrapper observed subsequent telemetry for 0.7 seconds before normal error
  cleanup: speed fell from 2.344 m/s to approximately zero and telemetry sequence
  advanced by 24. This supports a physical stop in this test, without adding an
  execution-acknowledgement protocol or testing a whole-WSL freeze.

`acceptance_result.json` records the passed checks. Both cases retain the source
versions verified against their recorded hashes. No further runtime root-cause
experiments were launched.

### Stage 3.11: Targeted Negative-Offset Recovery Data (2026-11-16)

Added `RecoveryCollectionConfig.directions` and the collection-only CLI option
`--recovery-direction negative`. Balanced directions remain the default. The
negative mode cycles the same three release targets; steering magnitude 0.35,
four-step kick cap, startup/straight-road gates, heading/CTE release guards,
teacher throttle, action history and steering rate limit remain unchanged.
Seven new tests cover negative direction scheduling and invalid signs; all
73 tests pass. This changes collection actions, not the frozen teacher.

One capped batch, `logs/round_b_negative_recovery_v1/`, collected six fresh
training-only layouts, up to 1,500 steps each, with TCP_NODELAY and the 100 ms
guard enabled. Saved simulator settings remained timeScale 2/FPS 120, random
light/trees enabled, cones disabled. No runtime timeout occurred.

| Coverage | Previous recovery batch | Added negative batch |
| --- | --- | --- |
| Frames | 15,187 | 7,698 |
| Startup frames | 3,000 | 1,500 |
| CTE <= -1 frames | 146 | 198 |
| CTE >= +1 frames | 294 | 0 |
| Interventions / recovered | 104 / 94 | 55 / 49 |
| CTE-ended episodes | 9 | 2 |

All full-frame/geometry/motion/history/action-alignment audits pass; maximum
teacher-action timing-alignment error is 8.35e-7. Failed and horizon-truncated
recoveries remain in the captures. Both batches together provide 344 negative
large-offset frames. `coverage.json`, `dataset_manifest.json`, source snapshots,
and the original split manifest preserve provenance; no prior locked evaluation
capture is used for fitting.

The pre-recorded collection target was 200 new negative frames; actual coverage
is 198, 1% short. The batch stopped at its six-layout cap. The documented decision
is to allow one unchanged-architecture offline fit with these samples, without
collecting more layouts or relaxing the validation-performance gates. Validation
remains grouped by layout (last three training-pool layouts of each capture run);
existing internal validation is development data, not final independent evidence.

**Single offline retraining result:**
`logs/round_b_negative_geometry_train_v1/` combines the legacy feature cache with
Round A training-pool captures and the two recovery batches. It fits the unchanged
two-stage linear lag Ridge head on 72,917 frames and validates on 19,384 frames
from nine held-out development layouts (19,336 valid lagged samples). The
grouped fit itself took 5.38 seconds, excluding image feature extraction/cache
construction. Source snapshots and input fingerprints are retained. The export
is `models/dinov2_geometry_15d_negative_v1/proj15_lags_ridge.pt`; the teacher and
original perception checkpoint hashes were verified unchanged.

| Development subset | Frames | Original CTE P95 | New CTE P95 | Original / new heading P95 |
| --- | --- | --- | --- | --- |
| All | 19,336 | 0.13493 | 0.10040 | 2.983 / 2.354 deg |
| Negative CTE <= -1 | 129 | 0.31297 | 0.31617 | 4.719 / 4.355 deg |
| Positive CTE >= +1 | 80 | 0.70038 | 0.58530 | 7.111 / 7.021 deg |
| Absolute CTE >= 1 | 209 | 0.50341 | 0.47537 | 5.421 / 5.164 deg |
| Startup | 2,964 | 0.11933 | 0.10472 | 2.839 / 2.505 deg |

**Decision: not promoted.** The pre-recorded gate requires at least 5% lower
negative-large-offset CTE P95, with aggregate CTE/heading P95 degradation bounded
to 10%. Aggregate metrics pass, but the target tail ratio is 1.010, exceeding
0.95. The negative-tail difference is small and is not claimed statistically
significant; it simply provides no evidence of the required improvement.
No second fit, closed-loop candidate screening, or 24-layout evaluation was
launched. Preserve the useful training captures and experimental head, keep the
original runtime baseline, and retain `offline_gate.json` / `decision.json` as
the explicit stopping decision. These offline results do not establish improved
closed-loop completion or prove that additional data can never help.

### Stage 3.12: Common-Layout Offline Control Comparison (2026-11-16)

The user authorized Step 1 only: compare the three existing heads through the
same frozen teacher, without driving, fitting, or changing the baseline.
`tools/compare_geometry_control.py` consumes the Stage 3.11 feature cache and
uses the same 19,336 valid-lag frames from nine development layouts / 12
trajectories. Both retrained heads excluded these layouts from fitting. Original
locked evaluation captures remain unopened. Artifacts are in
`logs/geometry_control_compare_v1/`.

**Replay semantics and audit:** At frame k, the 18D state is joined with its
recorded **post-command** 40D history to query action k+1. Every head receives
the same motion/history; only the 15D geometric inputs differ. Steering is
mapped with max steering 1, max difference 0.1 (executed delta limit 0.19998),
throttle 0.2..0.7, and no accelerated unwind. This is a one-step counterfactual
on recorded trajectories, not a recursive rollout. True states/history were
checked against each saved capture; maximum replayed true-teacher action
difference from stored live labels was 1.20e-7. The mapper is tested against
production behavior, including saturation and reversals.

| Head | Mapped steering MAE | Mapped steering P95 | Negative-large-offset steering P95 | Missed teacher unwind / 612 eligible frames | Max absolute 20-step mean steering error |
| --- | --- | --- | --- | --- | --- |
| Original | 0.03756 | 0.20535 | 0.12026 | 66 | 0.11642 |
| Round B | 0.03316 | 0.18299 | 0.09684 | 60 | 0.11198 |
| Negative supplement | 0.03296 | 0.18252 | 0.07771 | 58 | 0.11911 |

Steering values are normalized command units, not degrees. The negative-large
subset contains 129 frames from six layouts/trajectories. Relative to the
original head, the negative-supplement head lowers aggregate steering MAE by
12.2% and negative-large steering P95 by 35.4%, even though its negative CTE
P95 did not improve. Its per-layout macro MAE difference is -0.00439, with an
exploratory paired-layout bootstrap interval [-0.00501, -0.00369]. These are
development-set comparisons, not independent final-test confidence claims.

**Risks retained:** Maximum continuous bias does not improve (0.119 vs 0.116),
with the worst window at startup on negative-recovery layout 6, steps 10..29.
Large-offset steering-change direction mismatches are 10 versus 9 frames;
teacher-relative missed-unwind counts are small and correlated. Here unwind
means the teacher reduces steering magnitude by >0.02 from |previous steering|
>=0.3 while |CTE|>=0.5 or |heading|>=8 deg; it does not certify physical recovery.
The true-state teacher is a reference controller, not an infallible expert.

**Input-repair diagnostic:** Keeping the negative head's other inputs fixed,
substituting true lookahead points lowers aggregate mapped steering MAE from
0.03296 to 0.01158; substituting true heading lowers it to 0.02696. Substituting
only true CTE leaves 0.03292. Thus CTE P95 alone is a poor selector for this
controller. These are non-additive counterfactual interventions with correlated
geometric inputs, not proof of independent physical causes.

**Decision:** Preserve the historical Stage 3.11 CTE-gate result and all baseline
weights. This new control-related evidence justifies nominating the negative
head for one four-layout paired closed-loop screen using the unchanged teacher
and identical runtime guards. No promotion or student-controller training is
justified yet. That screen was **not started** during this offline-only step.
`decision.json`, `report.json`, `per_layout.csv`, `worst_bias_windows.csv`, and
validation-only `predictions.npz` retain the evidence and source snapshots.
All 76 tests pass; new checks cover exact production mapping, sequence-safe bias
windows, and teacher-relative direction/unwind masks. `git diff --check` passes.

### Stage 3.13: Four-Layout Paired Screen of the Negative-Supplement Head (2026-11-16)

The user authorized one fresh paired screen. Before execution,
`logs/negative_head_paired_4layouts_v1/screen_plan.json` fixed four layouts, two
conditions per layout, a 2,000-step episode cap and 540-second process deadline,
with no automatic retries, fitting, expansion or promotion. Both conditions use
the frozen original teacher, low-jitter sync, TCP_NODELAY, 100 ms observation
timeout, steering difference 0.1, throttle 0.2..0.7, and no accelerated unwind.
Saved simulator settings remained timeScale 2/FPS 120 with random light/trees on
and cones off. Only the geometry head changes; `vision_round_b` in the artifacts
denotes the **negative-supplement** checkpoint, not the earlier Round B head.

```bash
python tools/eval_teacher_tolerance.py --mode round_b_vision_paired \
  --candidate-geometry-head models/dinov2_geometry_15d_negative_v1/proj15_lags_ridge.pt \
  --layouts 4 --max-episode-steps 2000 --tcp-nodelay \
  --observation-timeout 0.10 --save-captures \
  --output-dir logs/negative_head_paired_4layouts_v1
```

All eight episodes completed: 16,000 total steps, 72 laps, zero runtime aborts.
Source snapshots match the recorded hashes. Evaluation-only captures pass the
motion/history/action-rate-limit audit, next-action alignment checks, and frozen
teacher/head weight hash checks. Original Round A locked captures remain unused.

| Metric | Original head | Negative-supplement head |
| --- | --- | --- |
| Completion / laps | 4/4 / 36 | 4/4 / 36 |
| Mean absolute CTE | 0.17236 | 0.17165 |
| Maximum absolute CTE | 0.96683 | 1.19103 |
| Frames at absolute CTE >= 1 | 0 | 4 |
| Mean speed (m/s) | 2.5372 | 2.5303 |
| Pooled online perception CTE error P95 | 0.13705 | 0.09732 |
| Pooled online heading error P95 (deg) | 2.9579 | 2.3622 |
| Maximum encoder time (ms) | 10.83 | 82.27 |

The paired mean absolute CTE difference is -0.000710 CTE (95% layout-bootstrap
interval [-0.01303, +0.00762]), only a 0.4% point-estimate improvement. Layout 1
favors the candidate; layouts 2..4 have slightly higher candidate mean CTE. The
candidate's perception improvements persist online, but do not yield a clear
lateral-tracking or completion benefit in this screen.

**Timing caveat, without reopening diagnosis:** Candidate layout 4 logged an
82.27 ms encoder call at step 1347; its CTE peak -1.191 occurred at step 1354.
The 82.90 ms observation interval appears in the following control cycle. All
these samples remain in primary metrics. This association does not identify
the stall source or establish that the head caused the larger CTE peak. The
100 ms observation-wait guard neither covers encoder work nor triggered here;
its documented scope remains unchanged.

**Decision:** Completion screening passes, but no clear closed-loop improvement
is demonstrated. Keep the original head as the working baseline and retain the
negative head as experimental; do not overwrite either checkpoint or infer
reliable completion from four layouts. Do not automatically run further
layouts, train a student controller, or reopen runtime root-cause work.
`audit.json`, `decision.json`, `episodes.jsonl`, full captures, telemetry and
`tolerance_results.json` retain the result. No production code changed for this
screen; the previous 76-test validation remains applicable and
`git diff --check` passes.

### Stage 3.14: Sensor Source Audit and Student Input Requirement (2026-11-16)

**User requirement:** Student control inputs may include the camera, wheel
encoder measurements, IMU measurements and self-generated command history.
Direct simulator linear velocity, position, true CTE, road-relative heading,
map/centre-line lookahead, and other privileged state are not student inputs.
Such state may remain isolated for teacher labels, evaluation and diagnostics.
Existing geometry-head pipelines retain true velocity and therefore do not meet
this requirement. Historical results remain valid privileged-motion references;
they do not establish compliant student performance. Do not continue the proposed
control-student fine-tuning with the old motion input contract.

At the user's request, upstream GitHub files were retrieved through the configured
HTTP proxy and pinned to commit `ec04c400f045222375ec82e649bb051fb17aad41`.
The installed Windows `Assembly-CSharp.dll` was independently inspected as .NET IL;
it was not modified or replaced. Evidence, source copies, assembly hash and the
read-only inspection script are under `logs/sensor_interface_audit_v1/`.

- In the local `Car` implementation, velocity is
  `transform.InverseTransformDirection(rb.velocity)`. `TcpCarHandler` divides
  `GetVelocity()` by 8 and sends its magnitude as `speed`; extended `vel_x/y/z`
  use the same vector. Thus the existing speed is rigid-body truth, not a wheel
  encoder. An earlier conversational claim that these components were necessarily
  world-frame was inaccurate: `Car` converts them to body frame, whereas the
  separate `UnityStandardCarAdapter.GetVelocity()` returns Rigidbody velocity
  without this conversion. Runtime vehicle-adapter identity must be established
  before asserting a universal coordinate convention.
- `Car.GetGyro()` returns the stored `rb.angularVelocity`, with no body-frame
  transformation in that update. `GetAccel()` is a finite difference of the
  local velocity vector; it is not by itself a calibrated accelerometer specific
  force model. `gyro_*` / `accel_*` fields exist, but their semantics must be
  verified before using them as a body-mounted IMU. Simulated IMU signals can
  legitimately originate in engine physics; the measurement model, frame and
  units must represent the permitted sensor, without exposing pose/velocity as
  extra policy features.
- The local binary and upstream source contain `Odometry`, and TCP telemetry
  can transmit `o.Label -> o.GetNumberRotations()` for configured components.
  The upstream implementation estimates cumulative absolute rotations from wheel
  mesh quaternion differences and explicitly warns that it is unreliable at
  high wheel spin rates. It also loses rotation direction. This is distinct
  from `speed`. Presence of this class does not prove that the current vehicle
  has an enabled, usable encoder feed. The Python SDK/custom sync currently do
  not preserve arbitrary odometry-label fields as policy observations.

**Status:** Source audit only; no sensor adapter, compliant student, training,
simulator driving or new performance claim was introduced. Next establish the
active vehicle's raw encoder/IMU output, then implement a whitelisted sensor
observation path without falling back to true velocity. The old teacher input
layout cannot simply be relabelled as encoder/IMU data.

### Stage 3.15: First Camera-Only Control Student — Offline Only (2026-11-16)

The user authorized implementation of a student with **no simulator motion
truth**, a 20-minute first-phase budget, one offline fit, and no simulator run.
No encoder or IMU is used in this iteration. The original perception head and
the true-state teacher remain frozen at their prior hashes.

**Implementation:** `rl/visual_student.py` provides a separate image-only
`VisualGeometryEncoder` and deterministic `VisualStudentActor`. The feature API
accepts images only and never calls `TrackState.encode` or receives telemetry.
Three predicted 15D geometric vectors at lags 0,2,4 plus 40 issued commands form
85 student inputs. Since each geometric prediction already uses lagged DINO
features, effective context extends back eight steps. Causal first-frame padding
is reset per trajectory and matches the batch path. No future, cross-condition,
cross-layout, missing-step or prior-episode frame is borrowed.

The actor is 85->256->256->2 with ReLU hidden layers and tanh outputs. Compatible
teacher geometry/history weights initialize it; the three teacher motion columns
are discarded and new historical-geometry columns start at zero. This creates
an adapted initialization, not a valid replacement for the teacher without
training. The student has its own checkpoint format and enforces the 85D input
contract plus geometry-head hash at load time. No critic or RL updates are used.

`tools/train_visual_student.py` consumes existing cached image features and
issued command histories, constructing student inputs before accessing teacher
state. The label branch separately queries the frozen teacher on true state and
the same post-command history at frame k, targeting command k+1. Student inputs
and teacher labels are stored separately. The first fit uses full reset-aligned
capture runs only; legacy tail-only records are not used. There are 39,020
training frames after excluding 9,973 frames from failed training captures, and
19,384 validation frames across the existing nine development layouts. Failed
validation captures stay in the evaluation. No locked evaluation capture enters
the fit. The 1..8-step startup padding remains included and explicitly marked.

**One fit:** AdamW lr 1e-4, weight decay 1e-4, batch 512, at most 15 epochs, seed
42. Loss is mapped steering MSE + 0.1 raw steering MSE + 0.5 mapped throttle MSE.
Steering difference remains 0.1 (executed delta 0.19998), throttle 0.2..0.7, and
accelerated unwind is disabled. Best development MAE selects epoch 14 of 15.
Fitting/evaluation took 15.39 seconds; the entire offline job took 19.39 seconds,
excluding implementation and tests. Artifacts:
`logs/visual_student_no_motion_v1/`; export:
`models/visual_student_no_motion_v1/student.pt`.

| Same-layout development metric | Adapted initialization | Trained camera-only student |
| --- | --- | --- |
| Mapped steering MAE | 0.15281 | 0.03987 |
| Mapped steering P95 | 0.39996 | 0.20339 |
| Mapped throttle MAE | 0.07457 | 0.01931 |
| Missed teacher unwind / 612 eligible frames | 418 | 71 |
| Wrong steering-change direction / 1,743 eligible frames | 532 | 114 |

The mapped-steering MAE ratio is 0.261, large-offset MAE ratio 0.255, and
missed-unwind-rate change -0.567, passing the pre-recorded offline gates
(<=0.8, <=1.1 and <=+0.02 respectively). Student negative-large steering
MAE/P95 is 0.01144 / 0.06627 on 129 frames; aggregate large-offset MAE/P95 is
0.02757 / 0.18699 on 209 frames. These small correlated subsets compare against
the true-state teacher's actions, not an infallible recovery oracle.

**Interpretation:** The student learned from the teacher without motion truth
in its inputs. The roughly 74% MAE reduction is relative to its deliberately
adapted initialization, not the old trained pure-vision RL policy. It is not a
closed-loop improvement claim. Partial observability, startup bias and
student-generated history distribution shift remain untested live risks.

**Runtime entry prepared, not executed:** `rl/eval_visual_student.py` loads only
this student format and the matching frozen visual head. Scoring/termination
may use true CTE/speed; they are separate from policy inputs. Temporal state is
reset at episode boundaries, and the existing runtime guards remain enabled.
A mocked evaluation runs with image and scoring fields but no pose/velocity/
gyro/acceleration fields, validating this separation without driving the sim.

**Validation and stop:** Image-only feature/actor APIs, adversarial telemetry
independence, nested live/cache padding, sequence breaks, teacher-weight transfer,
checkpoint contract/hash enforcement, differentiable action mapping and the
evaluation entry are tested. A real-data audit checks issued-history equality,
teacher-label timing against stored live teacher labels, disjoint layout splits,
and unchanged teacher/head weights. Saving/reloading reproduces student outputs.
The label-alignment maximum difference is 2.39e-7; all 83 tests pass, including
the mocked evaluation without pose/motion fields. Student maximum absolute
20-step mean steering error is still 0.202 (initialization 0.329), so persistent
control bias remains a live-test risk. `input_contract_audit.json` and
`decision.json` retain these checks; `runtime_source/` separately snapshots the
final runtime/evaluation implementation while `source/` preserves the fit code.
The first-phase outcome is an experimental compliant student eligible for a
separately authorized live screen; no automatic refit, DAgger, simulator run,
promotion or replacement of the old policies occurred.

### Stage 3.16: First Camera-Only Student Live Smoke and Replay Audit (2026-11-16)

The user authorized the next step. One bounded smoke was pre-recorded in
`logs/visual_student_smoke_v1/screen_plan.json`: one episode, at most 2,000 steps,
100-second process deadline, no automatic retries, training or layout expansion.
The initial launch stopped before simulator connection because EvalRecord
requires a nonexistent output directory; the prewritten plan had already created
the parent. The same smoke was then launched into a fresh `run/` subdirectory.
This was a preflight path correction, not a repeated driving attempt.

```bash
python rl/eval_visual_student.py --host 172.25.240.1 --episodes 1 \
  --max-episode-steps 2000 --output-dir logs/visual_student_smoke_v1/run
```

Only the camera-derived 85D feature/history vector enters the separately loaded
student actor. No teacher model, encoder sensor, IMU, true motion, map or road
state is used by the policy. The evaluator uses CTE/speed for scoring and
termination only. Low-jitter sync, TCP_NODELAY and the 100 ms observation-wait
guard are enabled; steering difference 0.1 and throttle 0.2..0.7 are unchanged.
Saved simulator settings confirm timeScale 2/FPS 120, random light/trees on,
cones off. The student and frozen head checkpoint hashes match Stage 3.15.

**Result: 1/1 completed, 2,000 steps, 9 laps**, no exit or runtime timeout.
Mean speed was **2.50565 m/s**, mean absolute CTE **0.20811**, maximum absolute
CTE **0.93776** (CTE units, not metres). This is the first compliant camera-only
student result; it is not the previous privileged-motion 23/24 result.

For auditable input provenance, the evaluator now saves the exact 85D decision
inputs and raw actions to `policy_inputs_episode_01.npz`. It also requests
braking after the scored episode, before serializing images/results. A real-run
offline audit verified:

- All 2,000 input vectors contain the specified 45 visual values and 40 issued
  commands; initial history is zero and every subsequent history equals the
  actual preceding command sequence.
- Production mapped actions agree with recorded executed commands to 2.99e-8;
  batch actor replay agrees with recorded raw actions to 1.02e-6.
- Replaying the last 120 exact camera frames, discarding eight warmup frames and
  the final frame with no following decision, reproduces **111 decisions** using
  only images and issued history. Maximum feature and action differences are
  both **zero**. The scoring telemetry is not needed for this replay.

`audit.json`, `audit_replay.py`, source snapshots, camera tail, actual decision
inputs, telemetry, results and `decision.json` preserve the evidence. The mocked
evaluator test also checks that saved policy inputs have 85 columns and no truth
arrays. All 83 tests pass and `git diff --check` passes.

**Decision:** Basic compliant driving smoke passes; no model promotion or
superiority/high-completion-rate claim follows from one layout. No further live
attempt, fitting or runtime root-cause investigation was launched automatically.

### Stage 3.17: Frozen Camera-Only Student Four-Layout Screen (2026-11-17)

The user authorized the proposed bounded four-layout evaluation. The plan is
saved before driving in `logs/visual_student_screen_4layouts_v1/screen_plan.json`:
four fresh scene loads, at most 2,000 steps each, a 360-second process deadline,
no fitting, retries or automatic expansion. Preflight verified that student/head
hashes and the five relevant runtime source files match the Stage 3.16 smoke.
The saved simulator settings also match: timeScale 2, FPS 120, random light and
trees on, cones off, useSeed 0. These are saved menu settings, not independent
measurements of live engine state. Layout IDs denote scene loads, not reusable
seeds, and this run has no paired comparator.

```bash
python rl/eval_visual_student.py --host "$DONKEY_SIM_HOST" --episodes 4 \
  --max-episode-steps 2000 --output-dir logs/visual_student_screen_4layouts_v1/run
```

The existing student and original geometry head stay frozen. Runtime inputs
remain 45 image-derived geometric features plus 40 self-issued command values;
no teacher is loaded and no velocity truth, pose, map, encoder or IMU enters
the actor. Ground truth remains isolated to scoring, termination and logging.
Low-jitter sync, TCP_NODELAY, the 100 ms observation-wait guard, steering
difference 0.1, throttle 0.2..0.7 and disabled accelerated unwind are unchanged.
Each episode explicitly reloads the scene. No production code was changed.

| Layout | Completed / steps | Laps | Mean absolute CTE | Max absolute CTE | Mean speed (m/s) | Max CTE step |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Yes / 2,000 | 9 | 0.17877 | 0.95405 | 2.51614 | 79 |
| 2 | Yes / 2,000 | 9 | 0.19576 | 0.95127 | 2.51849 | 75 |
| 3 | Yes / 2,000 | 9 | 0.18435 | 0.94743 | 2.50256 | 77 |
| 4 | Yes / 2,000 | 9 | 0.21109 | 0.89125 | 2.51332 | 176 |

**Result:** 4/4 completed, 8,000 steps, 36 laps, pooled mean absolute CTE
**0.19249**, maximum **0.95405**, pooled mean speed **2.51263 m/s**, zero samples
at absolute CTE >= 1 and zero runtime aborts. Recorded evaluation elapsed time
was 223.37 seconds. All four maximum CTE values occur in the first 250 steps;
startup remains a useful target for larger evaluation, without inferring a
specific failure mechanism from these successful runs. CTE is in simulator
units, not metres.

**Input and replay audit:** All 8,000 stored inputs have 85 columns, finite
values and no truth arrays. Command histories reset to zero and exactly match
the preceding issued commands. Production action mapping agrees with recorded
commands to 2.99e-8; actor replay agrees with raw actions to 1.35e-6. Replaying
each final 120-image tail, excluding eight context frames and the final frame
without a subsequent decision, reproduces 111 decisions per episode: **444
image-only decisions with zero feature and action error**. Student/head and
encoder tensor hashes match the smoke; all captured source hashes match current
files. Only initial images and final 120 images per episode are retained;
decision inputs, actions and scoring telemetry cover all steps.

**Timing:** Encoder P95 is 4.81..4.87 ms across episodes, maximum 10.65 ms.
Policy P95 is 0.50..0.57 ms. The first policy call at layout 1, step 1 took
212.03 ms at near-zero speed; it remains in the recorded metrics. The existing
100 ms guard covers observation waiting, not policy/encoder work, so absence of
an abort does not assert that every part of the control cycle stayed below
100 ms. No timing root-cause investigation was launched.

Artifacts include `run/{metadata,results}.json`, `run/telemetry.jsonl`, per-episode
`policy_inputs_episode_*.npz`, image tails/source snapshots, `audit_replay.py`,
`audit.json`, `decision.json` and console logs. The real-run input/action/image
replay audit passes; all seven focused tests in `tests/test_visual_student.py`
pass, and `git diff --check` passes. The full historical suite was not rerun.

**Decision:** The four-layout completion screen passes. Keep the student
experimental; no checkpoint or baseline was replaced, and no further live run
or training was launched. Four successes are a small sample (nominal Wilson
95% completion interval approximately 51%..100%, assuming independent trials),
not proof of high reliability or superiority to the historical vision policy.
A larger frozen evaluation is the proposed next step, with startup excursions
and runtime tails retained in the primary metrics.


### Stage 3.18: Frozen Camera-Only Student 24-Layout Evaluation (2026-11-17)

The user authorized the proposed 24-new-layout evaluation after Stage 3.17.
`logs/visual_student_eval_24layouts_v1/eval_plan.json` was saved before driving:
24 fresh scene loads, at most 2,000 steps each, a 1,800-second process deadline,
no training, retries, parameter changes or automatic expansion. This batch is
reported independently of the earlier one-layout smoke and four-layout screen.
There is no paired comparator, and layout IDs are load identifiers rather than
reproducible seeds. Original locked training/evaluation captures were not used
for fitting or model selection in this task.

Preflight matched both student/head checkpoint hashes and every runtime source
hash to the four-layout screen. Saved simulator settings match: timeScale 2,
FPS 120, random light and trees on, cones off, useSeed 0. These remain saved menu
values rather than independently measured live engine settings.

```bash
python rl/eval_visual_student.py --host "$DONKEY_SIM_HOST" --episodes 24 \
  --max-episode-steps 2000 --output-dir logs/visual_student_eval_24layouts_v1/run
```

Runtime sources remain camera-derived features and self-issued commands only:
45 visual values plus 40 command-history values. No teacher is loaded; no
velocity truth, pose, map, wheel encoder or IMU enters the actor. Ground truth
is scoring, termination and logging only. Low-jitter sync, TCP_NODELAY, the
100 ms observation-wait guard, steering difference 0.1, throttle 0.2..0.7 and
disabled accelerated unwind remain unchanged. Each episode reloads the scene.

| Layout | Completed / steps | Laps | Mean absolute CTE | Max absolute CTE | Mean speed (m/s) | Max CTE step |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Yes / 2,000 | 9 | 0.16881 | 0.96403 | 2.53025 | 76 |
| 2 | Yes / 2,000 | 9 | 0.19414 | 1.08611 | 2.51464 | 1420 |
| 3 | Yes / 2,000 | 9 | 0.21647 | 0.98435 | 2.51680 | 813 |
| 4 | Yes / 2,000 | 9 | 0.18241 | 0.94586 | 2.52134 | 1416 |
| 5 | Yes / 2,000 | 9 | 0.21881 | 1.43441 | 2.52805 | 396 |
| 6 | Yes / 2,000 | 9 | 0.17088 | 1.34689 | 2.50948 | 181 |
| 7 | Yes / 2,000 | 9 | 0.21224 | 1.62820 | 2.52898 | 1474 |
| 8 | Yes / 2,000 | 9 | 0.18689 | 0.87360 | 2.50933 | 75 |
| 9 | Yes / 2,000 | 9 | 0.17235 | 0.92485 | 2.50425 | 75 |
| 10 | Yes / 2,000 | 9 | 0.19022 | 0.93475 | 2.49805 | 78 |
| 11 | Yes / 2,000 | 9 | 0.20139 | 1.03583 | 2.54188 | 76 |
| 12 | Yes / 2,000 | 9 | 0.19110 | 1.05718 | 2.53482 | 178 |
| 13 | Yes / 2,000 | 9 | 0.17978 | 0.87859 | 2.51910 | 1427 |
| 14 | Yes / 2,000 | 9 | 0.18352 | 0.98477 | 2.49538 | 77 |
| 15 | Yes / 2,000 | 9 | 0.18474 | 1.23281 | 2.51796 | 181 |
| 16 | Yes / 2,000 | 9 | 0.20816 | 0.99907 | 2.52478 | 711 |
| 17 | Yes / 2,000 | 9 | 0.17460 | 0.98767 | 2.52265 | 76 |
| 18 | Yes / 2,000 | 9 | 0.18771 | 0.97220 | 2.52564 | 1414 |
| 19 | Yes / 2,000 | 9 | 0.19916 | 1.11288 | 2.50474 | 181 |
| 20 | Yes / 2,000 | 9 | 0.19158 | 0.90340 | 2.52331 | 74 |
| 21 | Yes / 2,000 | 9 | 0.18049 | 1.34892 | 2.48676 | 175 |
| 22 | Yes / 2,000 | 9 | 0.20206 | 0.99877 | 2.52020 | 176 |
| 23 | Yes / 2,000 | 9 | 0.17881 | 0.94400 | 2.50855 | 77 |
| 24 | Yes / 2,000 | 9 | 0.21163 | 1.11720 | 2.51588 | 1000 |

**Result:** **24/24 completed, 48,000 steps, 216 laps** (9 laps per episode),
pooled mean absolute CTE **0.19116**, maximum **1.62820**, pooled mean speed
**2.51678 m/s**, zero exits and zero runtime aborts. Recorded evaluation elapsed
time was **1,328.48 seconds (22.14 minutes)**. CTE uses simulator units, not metres.
The nominal Wilson 95% completion interval is **86.2%..100%**, assuming independent
trials; 24 successes do not establish guaranteed reliability or a superiority
claim relative to a different policy or historical unpaired layouts.

**Startup and recovery diagnostics:** 15/24 episode maxima occur in the first
250 steps. Pooled mean absolute CTE is 0.21264 during those startup steps and
0.18810 afterward. Ten layouts contain 60 total samples at absolute CTE >= 1,
forming 14 contiguous excursions (8 beginning at startup, 6 afterward). Every
excursion is followed by a recorded sample below absolute CTE 0.5 and the
episode completes. This is descriptive evidence of recovery on these runs,
not a guarantee for future disturbances. The largest excursion is layout 7,
step 1474, CTE +1.62820; the next sample below 0.5 is step 1481. Thus remaining
tracking excursions are not confined to startup. Exact image tails cover only
the final 120 steps of each episode, so most early excursion images are not
available; full decision inputs/actions and scoring telemetry remain available.

**Input and replay audit:** All 48,000 finite decision inputs have 85 columns
and no truth arrays. Histories start at zero each episode and exactly match
issued commands. Production mapping agrees with logged commands to 2.99e-8;
actor replay agrees with all logged raw actions to 1.68e-6. Replaying the final
120 exact images per episode, excluding eight context frames and the final
frame without a following decision, reproduces **2,664 image-only decisions
with zero feature and action error**. Checkpoint hashes, encoder tensor hashes,
source snapshots and current runtime source hashes all match prior evidence.
Scoring telemetry is unnecessary for the image-to-action replay.

**Timing:** Pooled encoder P50/P95/P99/max is 4.11/4.80/5.92/12.70 ms; policy
P50/P95/P99 is 0.406/0.538/0.795 ms. The first policy call at layout 1, step 1
is the only policy sample above 50 ms, taking **205.12 ms** at near-zero speed.
All encoder samples are below 50 ms. Environment-step time P99/max is
26.61/33.79 ms. The first-call spike remains in primary records, without a
causal attribution or runtime retuning. The 100 ms guard only covers observation
waiting, not policy/encoder work; no abort does not imply every processing
stage stayed below 100 ms.

**Artifacts and validation:** `eval_plan.json`, `run/{metadata,results}.json`,
`run/telemetry.jsonl`, 24 `policy_inputs_episode_*.npz` files, initial images,
image tails, source snapshots, console logs, `audit_replay.py`, `audit.json`,
`analyze_results.py`, `diagnostics.json`, `per_episode.csv` and `decision.json`
are under `logs/visual_student_eval_24layouts_v1/`. Both post-run audit and
descriptive-diagnostic scripts completed successfully. No production code or
weights changed; the previously passing seven focused student tests were not
repeated. Whitespace checks and artifact integrity are recorded in
`validation.json`.

**Decision:** The frozen camera-only student passes this bounded 24-layout
evaluation and remains a candidate baseline. Preserve the checkpoint; do not
automatically train, expand layouts, alter runtime settings, replace deployment
models or infer superiority from the historical privileged-motion 23/24 result.
The next proposed work is offline inspection of startup and cruise excursion
inputs/actions before choosing a targeted control-training change.


### Stage 3.19: Offline Diagnosis of Camera-Only Student Excursions (2026-11-17)

The user authorized starting the recommended offline inspection first. This
stage examines all 14 contiguous absolute-CTE>=1 excursions in Stage 3.18,
without driving, fitting, runtime edits or checkpoint changes. Artifacts are
under `logs/student_excursion_diagnosis_v1/`. `analyze.py` records input hashes
and a diagnostic plan before computation; `summarize.py` derives group summaries.
The initial script preflight used the wrong historical metadata key and stopped
before analysis; it was corrected from `args` to `artifacts`, retaining both
attempts in the console log. This was not a simulator attempt or model fit.

**Alignment and isolation:** Decision k uses camera frame k-1 and its preceding
issued-command history. Its predicted geometry is compared with telemetry k-1,
not the telemetry returned after command k. Stored lag-2/4 geometry matches its
earlier decision inputs exactly. All 48,000 actor outputs and mapped commands
replay within the existing 2e-5 / 1e-6 tolerances. True CTE is telemetry CTE;
reference heading/lookahead are reconstructed with the original hash-verified
fitted centre line and TrackState implementation. SDK `car` order was checked
as `(roll, pitch, yaw)`. Truth and the frozen 60k teacher are used only in this
separate offline diagnostic; no truth enters the live student input path.

Each event has a window from 35 decisions before through 20 after its returned
CTE peak. The primary comparison uses the 15 decisions immediately before
absolute CTE first reaches 1: 210 distinct decisions, with no duplicated samples
across events. The teacher receives the same recorded preceding command history
and the same production mapping. The visual-teacher diagnostic additionally
has the old privileged true-motion inputs, so its comparison with the student
does not isolate imitation error from partial observability.

| Pre-event subset | Events / decisions | CTE prediction MAE | Heading MAE (deg) | Student vs true-state teacher steering MAE | Visual + true-motion teacher MAE | Student MAE after true-geometry replacement |
| --- | --- | --- | --- | --- | --- | --- |
| All | 14 / 210 | 0.06129 | 1.44968 | 0.09117 | 0.03786 | 0.07281 |
| Startup (<=250) | 8 / 120 | 0.04913 | 1.18483 | 0.11745 | 0.02838 | 0.09619 |
| Cruise | 6 / 90 | 0.07751 | 1.80281 | 0.05614 | 0.05050 | 0.04165 |

Steering values are normalized commands, not angles. Geometry replacement is
performed at all three student geometry lags, preserving actual issued history.
The first five decisions are excluded from aggregate comparisons because their
complete lagged true geometry cannot be reconstructed from saved telemetry.
These are single-decision interventions on fixed trajectories; true geometry
can be out of distribution for a student trained on visual predictions. They
are neither recursive rollouts nor proof of a causal improvement.

**Findings:**
- Ten of 14 excursion peaks occur at centre-line indices 306..312 near the same
  bend, including seven startup events and three later-lap events. The repeated
  steps 175..181 therefore identify a repeated route segment, not evidence of
  a timer or reset-padding bug. The direct eight-step padding context has
  already expired, although earlier trajectory differences can still matter.
- Startup pre-event action disagreement is larger despite smaller measured
  geometric error than the cruise subset. Replacing all geometry reduces the
  all-event teacher-relative steering MAE by about 20%, leaving substantial
  disagreement. Across all eligible decisions, replacing CTE alone barely
  changes MAE (0.05222 -> 0.05220); replacing lookahead lowers it to 0.03920.
  CTE error alone is insufficient to select the next intervention.
- In the 210 pre-event decisions, the teacher reduces steering magnitude by
  >0.02 on 45 eligible frames, where previous |steer|>=0.3 and |CTE|>=0.5 or
  |heading|>=8 degrees. The student fails to reduce magnitude by >0.02 on
  20 of these frames. This teacher-relative definition is not a physical
  recovery certificate. Several startup peaks show student steering 0.23..0.40
  higher than the mapped teacher reference.
- The largest excursion (layout 7, peak step 1474) differs from the repeated
  startup case. At steps 1468..1472 both student and true-state teacher request
  strong negative raw steering and produce the same rate-limited commands on
  the recorded history. Thus neither a purely perceptual explanation nor a
  universal student-only explanation fits all events. The original steering
  limit remains unchanged; replay does not establish that loosening it helps.
- Across event windows, maximum policy/encoder/environment-step times are
  1.19/7.66/29.21 ms. The approximately 205 ms first-call policy spike at step 1
  is outside these windows. No long inference stall accompanies these selected
  windows; timing causality and server-side frame timing remain unproven.

**Limits and evidence:** No excursion-time camera images were retained, since
all events precede each episode's last-120-image capture. Saved predictions,
issued histories, telemetry and actor inputs support this diagnosis; lighting
or image content at the excursions cannot be visually inspected. Fourteen
selected events, ten sharing a bend, are not independent trials for significance
claims. The fitted-map heading/lookahead reference is the historical teacher
observation, not a new independent road-geometry measurement.

`report.json`, `summary.json`, `events.csv`, `aligned_windows.csv`,
`per_episode_metrics.csv`, 14 `event_*.png` plots and `excursion_locations.png`
preserve the diagnosis. `largest_excursion_raw_targets.json` retains the
additional teacher raw-action check for steps 1465..1476. Source snapshots,
`analysis_plan.json` and `audit.json` preserve provenance. All input artifacts,
model files and runtime sources remain unchanged. Three focused production
mapping/reference-metric tests pass; the full suite was not rerun.

**Decision:** Prioritize student control-response/temporal-state diagnosis and
fresh training-pool coverage of this bend over another generic CTE-head refit.
The present evidence does not distinguish insufficient supervision from missing
observable motion context, nor justify new sensors or relaxed steering limits.
Any future student fit should use fresh training-pool trajectories and be
judged on fresh evaluation layouts; this batch remains diagnostic data. The
candidate baseline is preserved. First-call warmup remains a separate proposed
runtime improvement; no warmup change, new training or live run was made here.


### Stage 3.20: Fresh Student-Driven Bend Data and One Control Fine-Tune (2026-11-17)

The user authorized targeted collection and a small student fit after Stage 3.19.
`logs/visual_student_bend_recovery_v1/plan.json` was recorded before driving:
eight new layout loads, 500-step cap, layouts 1..6 training and 7..8 development,
then one fine-tune from the original camera-only student. No candidate live
screen, automatic refit or baseline promotion is part of this step. The existing
straight-line kick collector would not target the observed bend/history issue;
instead, the frozen student drives naturally and a frozen teacher labels its
visited states offline (one DAgger-style supplement, no takeovers or kicks).

**Collection:** `tools/collect_visual_student_recovery.py` retains the baseline
85D camera/self-command inputs, original visual head, steering difference 0.1,
throttle 0.2..0.7, disabled accelerated unwind, low-jitter sync, TCP_NODELAY and
100 ms observation-wait timeout. Each episode reloads the scene. Preflight
matched baseline/head hashes and saved simulator settings: timeScale 2, FPS 120,
random light/trees on, cones off, useSeed 0. Saved menu values do not independently
prove live engine state.

The actor receives only the camera/history observation and is called before
privileged label-state construction. No teacher is loaded during collection.
`student_inputs_*.npz` contain observations/raw actions only; pre-action state
and map indices go into separate `label_state_*.npz` files. Every full camera
sequence is saved, including reset (501 images for 500 decisions). Truth remains
available for scoring, termination and isolated offline labels. Braking precedes
archive serialization and is also requested in cleanup.

```bash
python tools/collect_visual_student_recovery.py --host "$DONKEY_SIM_HOST" \
  --output-dir logs/visual_student_bend_recovery_v1/collection \
  --episodes 8 --train-layouts 6 --max-episode-steps 500
python tools/finetune_visual_student_recovery.py
```

Collection completed **8/8 x 500 steps, 4,000 decisions / 16 laps**, with zero
runtime aborts, in **148.55 seconds**. These are original-student collection
results, not new-candidate driving results. Both training (layout 6, max absolute
CTE 1.022) and development (layout 8, 1.192) contain natural large-offset samples.
All trajectories, including any failures if present, stay in their assigned
whole-layout split. No extra layouts or retries were launched.

**Collection audit:** All 4,000 camera decisions replay with zero feature error.
Command history resets and exactly matches issued commands; actor replay and
production mapping pass. State and map indices reconstructed from preceding
telemetry exactly match separate label records for decisions 2..500; the first
uses reset state. Checkpoint hashes remain unchanged. The mocked collector test
checks pre-action alignment and actor-before-label ordering.

**One fit:** `tools/finetune_visual_student_recovery.py` warm-starts the trained
student, retaining its 85->256->256->2 architecture and frozen visual head.
Original masks/groups are preserved: 39,020 old + 3,000 fresh = **42,020 training
decisions**; **19,384 old development** and **1,000 fresh development** decisions,
with **113 fresh bend decisions**. Previous 24-layout evaluation trajectories,
diagnosis windows and locked evaluation captures are excluded from fitting.
Original failed-training exclusions are retained; fresh failures would remain.

Teacher labels use true state and exact preceding issued history offline.
All old labels replay to 1.20e-7. Student features are read before this separate
label branch. Inputs, teacher state/labels and training selection/weights are
saved in separate files. Base weights are old=1, new=2, new bend (centre-line
indices 270..345)=8, multiplied by 2 for eligible teacher-unwind decisions;
validation weights are zero. Truth-based supervised example weighting does not
become an actor input.

AdamW: lr 2e-5, weight decay 1e-4, nominal batch 512, seed 42, at most 10 epochs /
180 seconds. Loss is weighted mapped steering MSE + 0.1 raw steering MSE + 0.5
mapped throttle MSE. One fit finishes 10 epochs in **2.99 seconds**, including
final evaluation/export. The preregistered equal-weight score of old-development
and fresh-bend MAE ratios selects epoch 10. New export:
`models/visual_student_bend_recovery_v1/student.pt`; its SHA256 is retained in
`fit/results.json` and `decision.json`.

| Development subset | Decisions | Original steering MAE / P95 | Candidate steering MAE / P95 | Missed teacher unwind, original -> candidate |
| --- | --- | --- | --- | --- |
| Old development | 19,384 | 0.03987 / 0.20339 | 0.04042 / 0.20511 | 71/612 -> 37/612 |
| Old large-offset | 209 | 0.02757 / 0.18699 | 0.02839 / 0.15905 | 8/87 -> 5/87 |
| Fresh development | 1,000 | 0.06705 / 0.26661 | 0.05516 / 0.24090 | 23/55 -> 5/55 |
| Fresh development bend | 113 | 0.06368 / 0.23946 | 0.03962 / 0.16228 | 14/26 -> 4/26 |

These are mapped command units and fixed-history teacher agreement, not physical
angles, candidate CTE or recursive rollouts. Fresh-bend MAE drops **37.8%** and
fresh-all MAE **17.7%**; old-all MAE rises **1.4%** and old-large MAE **3.0%**.
Old-development wrong-direction frames increase **114 -> 126** despite fewer
missed unwinds. This tradeoff is retained: control does not improve uniformly.
Two fresh development layouts and correlated bend frames are limited evidence,
and these layouts also participate in epoch selection.

All seven preregistered gates pass: >=50 fresh bend decisions; fresh-bend MAE
ratio <=0.90; fresh-all/old-all ratios <=1.05; old-large ratio <=1.10;
missed-unwind-rate increase <=0.02 on old and fresh development. No thresholds
were changed after the run. Teacher unwind is a reference action definition,
not a guaranteed recovery label.

**Validation/artifacts:** `audit_fit.py` independently reloads the candidate and
reproduces all saved development metrics/actions (maximum action difference
zero), checks exact features against camera-only archives, original masks,
disjoint layout groups, zero validation weights and unchanged teacher/head/
baseline hashes. `collection_audit.json`, `fit_audit.json`, full camera sequences,
input/label/selection NPZs, `fit/{training_manifest,results}.json`, source
snapshots, console logs, `plan.json` and `decision.json` preserve the run under
`logs/visual_student_bend_recovery_v1/`. All **86 tests pass**, including new
collector alignment, training-weight isolation and retention-gate tests.
Whitespace/integrity checks are recorded in `validation.json`.

**Decision:** Preserve the original 24/24-tested student as the baseline and the
new weights as an offline-qualified experimental candidate. No candidate live
run, second fit, promotion, new sensor or control-limit change occurred. Next
propose a bounded fresh-layout closed-loop comparison against the original
student with identical control/runtime settings, explicitly checking the
old-development regression. The candidate does not inherit the baseline's
24/24 completion result.


### Stage 3.21: Shared-Layout Closed-Loop Screen of the Bend Student (2026-11-17)

The user authorized the next closed-loop comparison. Before driving,
`logs/visual_student_bend_paired_4layouts_v1/plan.json` fixed four fresh layout
loads, two actors per layout, 2,000-step cap, 900-second process deadline, no
retries, automatic expansion, fitting or promotion. The original student and
Stage 3.20 candidate use the same original frozen geometry head. Preflight
matched all three checkpoint hashes and saved simulator settings to prior runs.
This evaluation uses fresh scenes, not the eight training/development scenes.

**Paired entry and protocol:** `rl/eval_visual_students_paired.py` reloads the
scene once per layout and sets wrapper scene reload to zero; between actors it
only resets the vehicle. The image encoder's internal lag context, outer visual
lags and command histories reset per episode. First-run order is balanced:
layout 1 candidate first, 2 baseline first, 3 candidate first, 4 baseline first
(order seed 42, not a simulator seed). The endpoint protocol and saved initial
views support scene pairing; simulator seeds are not independently verified.

Both actors receive one stateless 85-zero prediction **before connection**.
This common warmup avoids putting the known first-call spike into only one
actor's live episode. It issues no command and changes no visual/history cache.
Warmup took 437.44 ms for baseline and 0.55 ms for candidate before connection;
both are ready before driving. This is an explicit shared evaluation protocol
change from historical single-student runs; those entry defaults are unchanged.
No inference/wait samples are removed from the scored episode records.

All live control inputs remain the same 85 camera/self-command features. No
teacher is loaded and no encoder/IMU, motion truth or map enters either actor.
Truth is scoring, termination and logging only. Both use steering difference
0.1, throttle 0.2..0.7, disabled accelerated unwind, low-jitter sync, TCP_NODELAY
and 100 ms observation-wait timeout. Saved simulator settings remain timeScale 2,
FPS 120, random light/trees on, cones off, useSeed 0 (saved menu values, not an
independent live-engine measurement).

```bash
python rl/eval_visual_students_paired.py --host "$DONKEY_SIM_HOST" \
  --layouts 4 --max-episode-steps 2000 \
  --output-dir logs/visual_student_bend_paired_4layouts_v1/run
```

| Metric | Original baseline | Bend candidate |
| --- | --- | --- |
| Completion / steps / laps | 4/4 / 8,000 / 36 | 4/4 / 8,000 / 36 |
| Mean absolute CTE | 0.18147 | 0.18945 |
| Maximum absolute CTE across all episodes | 1.22731 | 1.16355 |
| Startup mean absolute CTE (first 250 steps) | 0.21290 | 0.18254 |
| Cruise mean absolute CTE (remaining steps) | 0.17698 | 0.19044 |
| Frames with absolute CTE >=1 | 13 | 10 |
| Mean speed (m/s) | 2.51291 | 2.52299 |

All eight episodes complete with no exit or runtime abort, **16,000 steps / 72
laps total**, evaluation elapsed **450.68 seconds**. CTE values are simulator
units, not metres. Both models complete nine laps per episode.

| Layout | Original / candidate mean absolute CTE | Candidate minus original | Original / candidate max absolute CTE |
| --- | --- | --- | --- |
| 1 | 0.16429 / 0.18985 | +0.02556 | 1.01158 / 0.94147 |
| 2 | 0.20110 / 0.20497 | +0.00386 | 1.00396 / 1.08042 |
| 3 | 0.18332 / 0.17995 | -0.00337 | 1.22731 / 1.08662 |
| 4 | 0.17716 / 0.18303 | +0.00588 | 0.92677 / 1.16355 |

**Interpretation:** Startup mean CTE decreases **14.3%** and cruise mean CTE
increases **7.6%**, with both directions occurring on every layout. Whole-episode
mean CTE increases **4.4%**. The paired mean difference is **+0.007983 CTE**,
with exploratory 95% layout-bootstrap interval **[-0.001059, +0.020138]**. This
small screen does not statistically establish a population regression, and it
does not demonstrate an overall improvement. Candidate global maximum is lower,
but per-layout maxima improve on only two of four layouts; mean per-layout peak
difference is +0.02561. Thirteen vs ten large-offset samples are too few to infer
recovery reliability. Candidate speed is approximately 0.4% higher; this does
not isolate a cause of the tracking tradeoff.

The pre-recorded checks are: no fewer completions (passes), paired mean CTE
improvement (fails), no more absolute-CTE>=1 frames (passes). Thus the candidate
**fails the screen**. This preserves the offline improvements from Stage 3.20
without mistaking them for whole-episode closed-loop improvement.

**Replay/provenance audit:** Every initial and returned image is saved; each
2000-decision episode has 2001 camera frames. All **16,000 decision inputs**
reconstruct from the full camera sequences plus actual issued history with
**zero feature error**, including both levels of startup padding. Raw actor
replay error is at most 1.46e-6; mapped commands agree to 2.99e-8. Command-history
resets, lag-2/4 equality and scoring metrics are checked per episode. Both
checkpoints, geometry-head/encoder tensor hashes, current runtime source hashes
and captured source hashes remain unchanged. Initial image mean pixel differences
within each pair are 0.076..0.186 on the 0..255 scale; views are similar, not
bit-identical. Scene identity is supported by the reload/reset protocol rather
than a newly claimed deterministic seed mechanism.

**Timing:** Baseline/candidate policy P99 is 0.529/0.537 ms, max 1.39/1.24 ms;
encoder P99 is 5.83/5.82 ms, max 9.10/8.81 ms. Environment-step maxima are
33.83/29.21 ms. Shared actor warmup occurs outside the recorded live episodes;
no live first-call outlier or wait-guard abort occurs. The 100 ms guard's scope
remains observation waiting, not all computation.

`audit_results.py`, `audit.json`, `decision.json`, `paired_results.png`,
`plot_results.py`, per-episode full camera/input archives, telemetry, source
snapshots, `run/{metadata,results}.json` and `plan.json` preserve the comparison.
All data are evaluation-only and excluded from fitting. The paired-entry tests
verify balanced ordering, missing/duplicate-pair rejection, one reload per pair,
per-episode history resets and execution with no pose/motion fields. All **88
tests pass**; whitespace and integrity checks are in `validation.json`.

**Decision:** Keep `visual_student_no_motion_v1` as the baseline. Preserve the
bend candidate as experimental and do not expand it to a larger evaluation,
refit automatically or promote it. Any further fine-tune should first strengthen
cruise-retention criteria and examine this startup/cruise tradeoff. The four-layout
comparison provides a concrete reason to retain the original, without claiming
that the candidate is universally inferior.


### Stage 3.22: Route/Lap Alignment and Bidirectional Actor Replay (2026-11-17)

The user authorized the planned bounded offline diagnosis. No new simulator
run or driving-model fit was made. Inputs are all eight complete Stage 3.21
trajectories (four shared layouts, 16,000 scored decisions), not only selected
excursions. Both actors, teacher and fitted map remain frozen. Artifacts:
`logs/visual_student_cross_replay_v1/`.

`analyze.py` writes `plan.json` and fingerprints evaluation input/camera archives,
telemetry, model files and centre line before analysis. Each actor receives
exactly the same saved 85D camera/history vector for a given trajectory row.
No command history is swapped independently of its images. Teacher state is
reconstructed from the preceding telemetry row, matching the existing teacher
contract; first decisions lack saved reset telemetry and are omitted from
reference comparisons (8 exclusions, **15,992 reference decisions**). All
16,000 own-actor action replays and actual command mappings pass. The teacher
is a reference controller, not an optimal physical recovery oracle.

**Road/lap alignment:** Each scored frame is projected onto the original fitted
centre line and assigned to one of 24 equal arc-index bins. Lap labels use the
recorded lap counter, not a 250-step cutoff. Every road bin and normal pass is
included. First recorded lap 0 may be partial in general; here both actors have
coverage in all 24 bins on all four layouts. Equal weights are applied per
observed lap/bin and then per layout; correlated frames are not independent
statistical trials. Physical speed is retained as a covariate, not controlled
or used as an actor input.

Candidate-minus-baseline CTE after this alignment is **-0.02156** for first
recorded laps (negative on all four layouts), **+0.01059** for later laps
(positive on 3/4). The first-250-step result is therefore not merely an
artifact of comparing different road coverage. As a coverage sensitivity
check, matching only complete later laps **1..8**, all 24 bins, produces paired
deltas **+0.03081, +0.00346, +0.00133, +0.01251**, mean **+0.01203**. This latter
estimate excludes incomplete final laps without filtering outcomes. Both
estimates retain the first-lap/later-lap tradeoff; neither establishes a
population effect from four layouts.

**Cross replay:** Both actors are queried on both recorded trajectory sources,
with the actual preceding commands for that source. Mapped steering MAE against
the same true-state teacher is averaged equally over layouts:

| Recorded input source | Phase | Original actor | Candidate actor |
| --- | --- | --- | --- |
| Original trajectories | First recorded lap | 0.08558 | 0.06980 |
| Original trajectories | Later laps | 0.04754 | 0.04840 |
| Candidate trajectories | First recorded lap | 0.07828 | 0.07084 |
| Candidate trajectories | Later laps | 0.05207 | 0.05482 |

Candidate improves first-lap agreement on both sources, but worsens later-lap
agreement on both. On original-source later laps, its error increases in 3/4
layouts; on candidate-source later laps it increases in 4/4. Thus later-lap
regression is visible even on the original actor's recorded inputs. The evidence
does not support attributing it solely to the candidate's command history.
Images, geometry, histories and teacher labels all differ between source
trajectories; the table is fixed-history single-decision replay, not a recursive
rollout or a causal decomposition of the closed-loop CTE difference.

**Recovery counts versus persistent error:** On original-source later laps,
missed teacher unwinds drop **27/163 -> 11/163**, while the mean per-layout
maximum absolute 20-step mean signed steering error increases **0.13911 ->
0.15750**. On candidate-source later laps, the corresponding counts are
**13/157 -> 5/157**, and persistent error **0.17121 -> 0.19547**. Fewer missed
unwinds do not establish better continuous tracking. All persistence windows
are contiguous within an episode/subset and never bridge missing segments.

**Both output functions changed:** Given identical original-source later-lap
inputs, candidate-minus-original mean throttle is **+0.001119**, with mean
absolute change 0.003421. On candidate-source inputs the mean difference is
+0.001067. Mapped steering absolute changes are about 0.01365/0.01350.
Later-lap steering is rate-limited on roughly 57% of decisions. These measurements
show that the intervention was not isolated to steering; they do not establish
that throttle or rate limits caused the observed tracking differences.

**Dynamic-information probe decision:** The recorded trigger required candidate
later-lap MAE to improve on original trajectories but worsen on its own
trajectories (macro means and at least 3/4 layout signs each). This did not occur:
later-lap MAE worsens on both sources. No probe is fitted. The current comparison
already exposes a cruise-retention problem under unchanged original inputs;
input observability can still limit both models, and its adequacy is not proved.

**Evidence and validation:** `report.json`, `cross_replay.csv`,
`all_passes_route_bins.csv`, `paired_route_bins.csv`, `action_changes.csv`,
`diagnostic_arrays.npz`, `route_comparison.png`, source snapshots/hashes,
`audit.json` and `validation.json` preserve the result. An initial route
aggregation omitted the trajectory-source filter and yielded identical values;
the sanity check exposed this, the source filter was corrected, and the initial
report is retained separately. `validate.py` independently recomputes every
source-specific route cell and macro replay metric from CSVs, computes the
complete-lap sensitivity check, verifies recorded scoring means and unchanged
input/model/runtime source hashes. All **five relevant mapping/paired-evaluation
tests pass**; the full previously passing 88-test suite was not repeated because
production code is unchanged. Whitespace checks pass.

**Reason ranking and next minimal experiment:** The evidence prioritizes a
first-lap versus later-lap training/selection tradeoff and continuous-error
retention over an immediate sensor or network change. Trajectory-distribution
changes and throttle/steering coupling remain contributors to investigate, not
isolated causes. Preserve the original baseline and do not automatically fit
again. `next_experiment_proposal.json` describes an unexecuted A/A calibration:
four fresh layouts, two identical-original actor runs per layout, 2,000-step
cap, the same shared-scene/reset/warmup protocol. It measures run-to-run
variability before interpreting another small-effect change; four pairs are
only a descriptive calibration and do not define acceptable degradation. A
subsequent steering-only intervention with the original throttle function could
isolate joint-output changes if warranted. Neither experiment is launched here.


### Stage 3.23: Original Student, Ten Fresh Layouts at 3,000 Steps (2026-11-17)

The user explicitly requested ten new random layouts with a 3,000-step cap,
after deciding to retain the original student and end speculative optimization.
This is the requested longer-horizon check only; no A/A experiment, candidate
trial, fitting or automatic follow-up is included. The initial time estimate was
14..16 minutes including scene loading, saving and checks.

Preflight matched the original student/head hashes and existing runtime sources
to the 24-layout run. Saved simulator settings match: randomLight=1,
generateTrees=1, useSeed=0, cones disabled, timeScale=2, FPS=120. These are saved
menu values rather than independent measurements of live engine state. Layouts
are fresh scene reloads, not reproducible seeds. `eval_plan.json` was saved before
driving, with ten episodes, 3,000-step cap and 1,200-second process deadline.

```bash
python rl/eval_visual_student.py --host "$DONKEY_SIM_HOST" --episodes 10 \
  --max-episode-steps 3000 \
  --output-dir logs/visual_student_eval_10layouts_3000_v1/run
```

The model remains `models/visual_student_no_motion_v1/student.pt` (original
SHA256 b00d7252e7a54041b7f34bba065188e02aaba13e606e5b332c21258e2532e6b2).
The 85D actor input remains camera-derived geometry and its own issued-command
history only. No teacher, encoder sensor, IMU, true motion, pose or map is used
for control. Truth is scoring/termination/logging only. The existing single-
student entry is unchanged: steering difference 0.1, throttle 0.2..0.7,
unwind acceleration disabled, low-jitter sync, TCP_NODELAY, 100 ms observation-
wait guard, scene reload every episode. No new actor prewarm was added.

| Layout | Steps | Laps | Mean absolute CTE | Max absolute CTE | Mean speed (m/s) |
| --- | --- | --- | --- | --- | --- |
| 1 | 3,000 | 14 | 0.18630 | 1.13358 | 2.54918 |
| 2 | 3,000 | 14 | 0.19478 | 1.42320 | 2.53160 |
| 3 | 3,000 | 14 | 0.16517 | 0.82645 | 2.53290 |
| 4 | 3,000 | 14 | 0.18142 | 1.00877 | 2.53515 |
| 5 | 3,000 | 14 | 0.17412 | 0.97588 | 2.53743 |
| 6 | 3,000 | 14 | 0.18565 | 1.11976 | 2.53924 |
| 7 | 3,000 | 14 | 0.19708 | 0.95695 | 2.55303 |
| 8 | 3,000 | 14 | 0.19286 | 0.96245 | 2.54815 |
| 9 | 3,000 | 14 | 0.17650 | 1.04630 | 2.53076 |
| 10 | 3,000 | 14 | 0.19283 | 0.89887 | 2.54520 |

**Result: 10/10 completed**, every episode **3,000 steps / 14 laps**,
**30,000 steps / 140 laps total**, zero exits and zero runtime aborts. Mean
absolute CTE is **0.18467**, maximum **1.42320**, below the **2.0 CTE termination
limit**. There are 27 samples at absolute CTE >=1. Mean speed is **2.54026 m/s**.
CTE uses simulator units, not metres. Completion denotes reaching the step cap
without termination, not an independent assertion about every wheel/road edge.

Recorded evaluation elapsed time is **793.58 seconds (13m14s)**; metadata
start-to-finish timestamps span about **13m26s**, including metadata/setup
overhead. The process exits normally. This longer-cap batch is reported
separately from the historical 24 layouts x 2,000 steps, without pooling their
completion rates or interpreting unpaired CTE differences as improvement.

**Audit:** All 30,000 finite 85D inputs and raw actions pass issued-history,
steering/throttle-limit, scoring and action-replay checks. Maximum raw-action
replay error is 1.48e-6; each episode's final 120 saved images reproduce 111
camera/history decisions after context exclusion: **1,110 decisions with zero
feature and image-to-action error**. Student/head and encoder tensor hashes,
source snapshots and current runtime source hashes match prior evidence.

Artifacts are in `logs/visual_student_eval_10layouts_3000_v1/`: `eval_plan.json`,
`run/{metadata,results}.json`, telemetry, per-episode policy input archives,
initial images, final camera tails, source snapshots, `audit_replay.py`,
`audit.json`, `decision.json` and console logs. No production code or weights
changed; the existing tests were not repeated. Real-run replay/integrity and
whitespace checks are recorded in `validation.json`.

**Decision:** User-requested extended verification passes. Retain the original
student; no additional experiment, training or optimization is started.
