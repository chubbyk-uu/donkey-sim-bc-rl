# Generated road and early loop

> Historical record. Results, “current” choices and “next” plans below belong to the dated experiments, not the active work queue. See [current status](../../status.md) and [documentation index](../../README.md). Commands assume the repository root; raw `logs/` and `data/` artifacts are local-only unless explicitly noted.

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
