# Experiment Log

This document records the practical experiment history: what worked, what failed, and
what should not be repeated. The short project entrypoint is kept in
[README.md](../README.md).

## 1. Behavioral Cloning

BC was the first baseline. The useful outcomes were mostly diagnostic:

- It verified the image/action data pipeline.
- It exposed action distribution issues, especially under-represented steering classes.
- It provided supervised baselines in `models/bc_*`.

Important BC artifacts:

```text
models/bc_nvidia_slow_006_flip/
models/bc_official_categorical_9bin_sampler6_300/
models/bc_official_categorical_curve_aug_balanced_v1/
```

BC did not become the final control solution. The simulator routes require robust
recovery from off-center states and compounding errors; SAC with VAE latents became the
main path.

Lessons:

- Steering/action imbalance matters. A model can look good on validation loss while
  still failing to recover on curves.
- Evaluation in the simulator is mandatory. Offline BC metrics did not reliably predict
  route completion.
- BC is still useful as a data sanity-check pipeline.

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

- Do not install old PyPI `gym-donkeycar`. Use a compatible source checkout installed
  editable.
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
