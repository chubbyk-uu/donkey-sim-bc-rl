# DINOv2 v6 Loop Track Experiment Log

Standalone process log for the `rl_loop_resnet_v6_dinov2` (initially named
`rl_loop_dinov2_v6`) experiment. Captures everything needed to compare future
DINOv2 runs against this baseline.

Run abandoned at 42k steps; v6 directory deleted. This log is the only record.

## 1. Run Configuration

### Phase 1 — cold start

```bash
python rl/train_loop_vae_sac.py \
    --env-id donkey-generated-track-v0 \
    --output-dir models/rl_loop_dinov2_v6 \
    --encoder dinov2_vits14 \
    --encoder-crop-top 0 \
    --hidden-size 256 \
    --batch-size 128 \
    --min-throttle 0.2 \
    --max-throttle 0.7 \
    --max-cte-error 2.0 \
    --gradient-steps-min 50 \
    --timesteps 80000 \
    --save-replay-buffer --save-final-replay-buffer \
    --device cuda
```

Default reward weights (`speed=0.15, cte_pen=0.25, alive=1.5, min_alive_speed=0.0`).
Default `--gradient-steps-cap=1000`. Default `--buffer-size=60000`.

Stopped manually at ~13.2k steps (sac_loop_vae_10000_steps.zip checkpoint kept).

### Phase 2 — resume from 10k with cap=3000

Same params as phase 1 but additionally:
- `--gradient-steps-cap 3000` (raised from 1000 — chasing 1:1 SAC update ratio
  against `max_episode_steps=3000`)
- `--resume-model models/rl_loop_dinov2_v6/sac_loop_vae_10000_steps.zip`
- `--resume-replay-buffer …_replay_buffer_10000_steps.pkl`
- `--timesteps 70000` (60k additional → target 80k total)

Phase 2 ran from 10k → 42k (≈40 min wall time), then stopped after late-stage
regression became visible (see §4).

## 2. Per-Episode Data (monitor.csv, full content)

```
r,l,t
396.382441,279,15.481319         # ep 1 (post-resume warmup)
4670.013772,3000,169.322922      # TRUNC
2164.423734,1431,268.524317
4890.585232,3000,432.366319      # TRUNC
4921.097165,3000,610.097043      # TRUNC
2090.60342,1309,703.202919
4960.73149,3000,865.577707       # TRUNC
4834.903087,3000,1042.354858     # TRUNC
4845.712948,3000,1219.454462     # TRUNC
523.875347,374,1265.301483
1598.023972,1070,1323.234811
3423.323581,2196,1443.469702
94.588893,75,1467.200827
4707.518233,3000,1619.086667     # TRUNC
1004.902721,673,1679.941442
2205.803418,1407,1757.284684
1947.050494,1222,1831.816533
1485.362785,967,1891.867298
693.2524,460,1924.52265
462.401055,312,1945.333762
2027.923869,1295,2014.086265
3336.862836,2325,2143.051455
4118.403287,2672,2298.039561
1399.450399,902,2367.683605
366.59572,262,2389.861964
```

`r` = total episode reward, `l` = episode length in env steps,
`t` = cumulative wall time (seconds since `t_start`).

`monitor.csv` only logs episodes since the phase-2 resume at 10k. Phase-1
episodes (cold start 0 → 10k) are lost — monitor.csv is rewritten on resume.

## 3. Training-Time Metrics (from train.log, batched averages)

| total_steps | ep_len_mean | ep_rew_mean | speed_mean | reward/step |
| ---: | ---: | ---: | ---: | ---: |
| 5438  | 68  | 80.6 | 1.92 | 1.185 |
| 6293  | 75  | 91.6 | 2.51 | 1.221 |
| 8967  | 102 | 134  | 2.49 | 1.314 |
| 13191 | 143 | 197  | 2.24 | 1.378 |
| 14710 | 155 | 217  | 2.26 | 1.400 |
| 25019 | 256 | 383  | 2.14 | 1.496 |
| 32463 | 320 | 486  | 2.44 | 1.519 |
| 38407 | 378 | 577  | 2.45 | 1.526 |
| 42463 | 417 | 638  | **2.08** | 1.530 |

`ep_len_mean` and `ep_rew_mean` are SB3 rolling means over the last 100
episodes (so they lag — old short episodes still pull the average down
even after the agent improves).

## 4. Phase Breakdown by Truncate Rate

Grouping post-resume episodes chronologically:

| Phase | Episode range | Truncate rate | Mean length | Mean reward/step |
| --- | --- | ---: | ---: | ---: |
| **Early truncate burst** | eps 2–6 | 3/5 | 2560 | 1.61 |
| Mid mixed | eps 7–11 | 1/5 | 1543 | 1.46 |
| **Late regression** | eps 12–25 | 1/14 | 1192 | 1.53 |

The "late regression" window is the headline failure: 14 consecutive
episodes with only 1 truncate, average length collapsing from 2560 to 1192.
Rolling means (`ep_len_mean`) lagged this collapse heavily because the early
truncates still dominate the 100-episode window. The `monitor.csv` raw data
exposed the regression that the rolling means hid.

## 5. Speed at Truncate Episodes

Eight truncate episodes (l=3000, r ranging 4670 → 4961). Per-step reward
1.557 → 1.640. Working backward via the reward formula:

```
reward/step = 1.5 (alive) + speed * (0.15 - 0.25 * abs(cte))
```

with `speed_mean ≈ 2.27` (batch average covering truncate-heavy window),
implied `abs(cte) ≈ 0.33` at truncate-time.

We do **not** have per-episode speed in monitor.csv, only `r`, `l`, `t`.
Future runs should additionally log per-episode mean speed and lap count to
let us compute per-lap time directly.

## 6. Lap Time Estimates (rough, no direct data)

`max_episode_steps = 3000`. Sim runs at ~20 Hz so 3000 steps ≈ 150 s wall
time. A truncate episode at `speed_mean = 2.27` covers approximately
`3000 * 0.05s * 2.27 ≈ 340` sim units. If a generated_track lap is roughly
~30-35 sim units (based on the `progress 437.1` value from v3_h80 truncate
episodes covering ~3000 steps), a truncate corresponds to ~10 laps.

This gives an **implied per-lap time on the order of 12–15 s**, slower than
`safe_v2 70k` (best lap ~8.6 s) and `v3_h80` (best lap ~8.5 s deployment).

These numbers are guesses, not measurements. The right next step is to log
per-lap markers and compute directly.

## 7. Reward Analysis: Why Speed Stalled

At a truncate episode with reward/step ≈ 1.64:

- `alive_reward` contribution: 1.5 (91.5%)
- `speed_reward_weight * speed` contribution: 0.15 × 2.27 = 0.34 (20.7%)
- `cte_speed_penalty`: -0.25 × 0.33 × 2.27 = -0.19 (-11.6%)

The agent gets **91.5% of its per-step reward for free just by being alive**.
The marginal reward for going faster is `0.15 - 0.25 * abs(cte) = 0.0675`
per unit speed at the observed working point. From speed 2.27 to 3.0:
+0.049 per step (+3% on a base of 1.64). From 2.27 to 4.0: +0.117 (+7%).

The speed coefficient sign flips when `abs(cte) > 0.6` — past that point
going faster hurts. The agent's optimum is therefore "low speed near
centerline" rather than "fast lap times". That is what we observe.

## 8. Hypothesis for Late-Stage Regression

After phase-2 reached its first burst of truncates, the policy had a high
density of "good truncate" trajectories in its replay buffer. With
`cap=3000` and `batch=128`, each truncate triggered 3000 batches × 128 =
384k samples seen, but the buffer only held ~60k unique samples. Each
sample was replayed ~6 times per truncate update cycle.

This is over-fitting on a small set of recent trajectories. SAC entropy
likely collapsed, exploration dropped, and the deterministic policy could
no longer handle slight environmental novelty → crashes. Subsequent crash
episodes then dominated the buffer with shorter trajectories, and the
agent could not recover the truncate behaviour within the remaining steps.

This matches the symptom pattern (long sustained truncates → sudden
collapse to mid-length crashes) and the historical pattern noted in
§6.9 (large hidden + aggressive updates produce late-stage instability).

## 9. Configuration Changes Adopted for Next DINOv2 Run

Based on §7 and §8:

| Parameter | v6 | Next | Rationale |
| --- | ---: | ---: | --- |
| `--buffer-size` | 60000 | **30000** | Raffin canonical; recycles old behavior data faster when reward changes |
| `--batch-size` | 128 | **64** | Raffin canonical; higher gradient variance escapes "alive-dominant" local optimum |
| `--gradient-steps-cap` | 3000 (phase 2) | **1000** | 3000 over-fit on single truncate; 1000 is the original phase-1 setting that worked without instability |
| `--gradient-steps-min` | 50 | 50 | Keep — short crash episodes still need minimum learning |
| `--min-alive-speed` | 0.0 | **3.0** | New reward conditioning: `alive_scale = clip(speed/3.0, 0, 1)`; full 1.5 alive only at speed ≥ 3.0; pushes agent past 2.27 plateau |
| All other reward weights | unchanged | unchanged | `speed=0.15 cte_pen=0.25 alive=1.5` reward shape stays |

Five simultaneous changes is a clean-sweep, not an ablation. Acceptable
because four of the five are reverts to Raffin canonical / phase-1 settings;
only `min_alive_speed=3.0` is a true experimental variable.

## 10. Next Run Command (cold start)

```bash
python rl/train_loop_vae_sac.py \
    --env-id donkey-generated-track-v0 \
    --output-dir models/rl_loop_dinov2_v7 \
    --encoder dinov2_vits14 \
    --encoder-crop-top 0 \
    --hidden-size 256 \
    --batch-size 64 \
    --buffer-size 30000 \
    --gradient-steps-cap 1000 \
    --gradient-steps-min 50 \
    --min-throttle 0.2 \
    --max-throttle 0.7 \
    --max-cte-error 2.0 \
    --min-alive-speed 3.0 \
    --timesteps 50000 \
    --save-replay-buffer --save-final-replay-buffer \
    --device cuda
```

## 11. Baselines For Comparison (existing deployment models)

| Model | Encoder | Hidden | Deterministic eval |
| --- | --- | ---: | --- |
| `fixedlight_v3_h80` 90k (deployment best) | VAE 512 | 80 | 3/3 trunc, mean_speed **2.80** |
| `fixedlight_v2_h64` 80k | VAE 512 | 64 | 3/3 trunc, mean_speed 2.689 |
| `resnet_v4_notrees` 50k | ResNet18 512 | 256 | 3/3 trunc, mean_speed 2.131 |
| DINOv2 v6 (this log) | DINOv2-S 384 | 256 | not eval'd; training speed_mean peak 2.45 |

A v7 (DINOv2 + reward fix) run should be evaluated against the
**deterministic eval mean_speed** of these, not the training-time mean.
Compare v7 to ResNet v4 first (same general "frozen ImageNet-like encoder
no track-specific VAE" niche). If v7 beats v4 → DINOv2 self-sup is helping.
If v7 also beats VAE deployment → big surprise, worth ablating components.
