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

## 3. Loop Track VAE Data

The loop-track work moved to `donkey-generated-track-v0`. This visual domain differs
enough from `generated_road` that the VAE was trained separately.

Final loop VAE:

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

The 20k checkpoint is intentionally preserved because `safe_v2` was resumed from it.
Other `speed_v1` checkpoints were removed.

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

## 6. Practical Pitfalls

- Do not install the PyPI `gym-donkeycar` package — it targets the old `gym` API.
  Install upstream instead with `pip install git+https://github.com/tawnkramer/gym-donkeycar`,
  or clone that repo and `pip install -e ../gym-donkeycar` for local edits. See the
  README's gym-donkeycar section for details.
- Do not mix VAE data from visually different simulator tracks.
- Do not judge SAC only by training reward or SB3 rolling means.
- Save replay buffers at checkpoints if resume matters.
- Do not resume SAC with replay buffers generated by a different reward function.
- `max_episode_steps` truncation is not a crash reward. It is a TimeLimit stop.
- On a non-closed generated road, not reaching 3000 steps can still be success if the
  vehicle reached the route end.
- If a speed bonus is too large, late training can regress even after the policy has
  learned to drive.

### Monitoring gotchas (learned during safe_v2)

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

## 6.5 Monitoring Practices

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

## 7. Current Recommendation

Use `safe_v2 70k` as the loop-track deployment checkpoint. Treat `speed_v1 20k` as a
reproducible seed for future reward experiments. Avoid further training past the current
70k checkpoint unless a new reward or evaluation target is introduced.
