# Session Log: 2026-05-27 — Mountain DINOv2 + training defaults

Two focus areas today:

1. Get DINOv2 working on `donkey-mountain-track-v0` (yesterday's task #36).
2. Investigate sim parallel training (yesterday's task #37) — research only.

End result: mountain DINOv2 deployment shipped (v2 40k, 5/5 truncate), four
failed cold-start variants documented, training defaults raised toward SAC
paper values, parallel sim research concluded as low-priority.

## 1. Mountain DINOv2 v1 — cold start with raffin reward (succeeded)

Configuration: `dinov2_vits14` (z=384), hidden=256, batch=64, cte_target=3.5,
max_cte_error=2.5, raffin reward (`speed=0.15, cte=0.25`) with **stricter
crash penalty** (`reward_crash=-20, crash_speed_weight=10`).

Cold start; stopped at 60k env steps. Trained at ~17 fps, ~75 min wall-clock.
Eval table is in `experiment-log.md` §6.10 (won't repeat here).

The peak was at the **30k checkpoint: 5/5 truncate, mean_speed 1.998,
mean_cte 0.682, max_cte 2.163, mean_lap 31.09s, 15 laps**. First DINOv2
result on mountain at all — historical context: ResNet mountain v1 only
managed 1/3.

40k+ showed the v1/v15-style 10-20k oscillation pattern: 40k=3/5, 50k=0/5
(valley), 60k=4/5 (partial recovery).

## 2. Mountain DINOv2 v2 — resume from v1 30k with stricter reward (current deployment)

Idea: v1 30k drives mountain but with high cte variance. Tighten reward to
trim it — this is the same recipe pattern that worked for v15 on loop.

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
    --timesteps 40000 --seed 42 --device cuda \
    --save-replay-buffer --save-final-replay-buffer
```

**v2 40k: 5/5 truncate, mean_speed 2.093, mean_cte 0.577, max_cte 2.112,
best_lap 28.41s, mean_lap 29.88s, 15 laps.** Beats v1 30k on every metric.

50k stayed 5/5 but mean_cte drifted up to 0.801 — kept as backup, buffer
removed. 60k collapsed (0/5) — deleted.

## 3. Mountain DINOv2 v3 — cold-start with v2-style reward (failed × 4)

Hypothesis: maybe v3 reward shape (`speed=0.12, cte=0.30`) can learn
mountain from cold start, skipping the v1→v2 chain. **It can't.**

Four cold-start attempts:

| Attempt | Seed | max_cte | cte_pen | Result |
|---|---|---|---|---|
| 1 | 42 | 2.5 | 0.30 | Stuck at first corner ~140 step |
| 2 | 7 | 2.5 | 0.30 | Stuck at first corner ~130 step |
| 3 | 7 | **3.0** | 0.30 | Stuck at first corner ~140 step |
| 4 | 7 | 2.5 | **0.25** | Progressed — reached ep_len 230, max_ep 355 (half lap), then stuck at second corner |

Each attempt: ent_coef collapsed to ~0.02 within 5-8k steps, then policy
committed to a "drive ~130-355 step then crash at corner X" basin and never
explored out. Last 25-30 episodes always landed in a narrow length window
(verified from monitor.csv).

Diagnosis:
- The stricter cte penalty actively prevents the exploration needed to
  learn corner-taking under racing-line geometry.
- v1's raffin cte=0.25 was permissive enough to let the agent over-shoot
  the lane in corners; v3's cte=0.30 punishes that and the agent retreats
  to "stay tight, crash early".
- Once committed to a basin, SAC's auto-entropy doesn't recover (no
  exploration → no novel data → no gradient out).

Branch deleted; the **learning is the artifact**: stricter reward needs
prior driving competence. Use v1 cold + v2 resume, not v3 cold.

## 4. Mountain DINOv2 v4 — resume from v1 *20k* with v2 reward (4/5, not deployable)

To test: would resuming from a less-committed earlier v1 checkpoint (20k
instead of 30k) respond better to v2 reward?

| Checkpoint | Trunc | Speed | Mean CTE | Max CTE | Best lap | Mean lap |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| v4 30k | 3/5 | 1.96 | 0.76 | 2.57 | 29.38s | 31.43s |
| v4 40k | 3/5 | 1.91 | 0.77 | 2.62 | 28.98s | 30.04s |
| v4 50k | 3/5 | 2.03 | 1.05 | 2.56 | 27.11s | 28.36s |
| v4 60k | 4/5 | 1.99 | 0.73 | 2.50 | 29.58s | 31.25s |

50k had the **fastest single lap of the day (27.11s)** but mean_cte 1.05
and max_cte 2.56 — racy but unstable. 60k was the most stable v4 checkpoint
but still only 4/5, and v2 40k beats it on every metric.

**Key learning from v4**: training-time trunc rate dropping (we watched it
fall from 42% → 15% over 30k env steps) is misleading. Deterministic eval
at 60k was actually the **best** of the v4 batch. The training-time
"degradation" we observed was exploration noise, not policy collapse.

Branch deleted; learning is recorded.

## 5. Training defaults raised (untested)

After v4's late-stage drift analysis we discussed several SAC stability
knobs. Two were applied as new defaults:

| Default | Old | New | Source / Rationale |
| --- | ---: | ---: | --- |
| `--batch-size` | 64 | **128** | SAC paper uses 256; reduce critic gradient variance |
| `--buffer-size` | 30_000 | **50_000** | Slower buffer turnover → less distribution shift late training |
| `--gradient-steps-scale` | (implicit 1.0) | **1.0** explicit, new CLI flag | New parameter; default unchanged. Allows `<1.0` for future ablation on long-ep overupdate. |
| `--gradient-steps-cap` (`train_vae_sac.py`) | 600 | **1000** | Unified with `train_loop_vae_sac.py`. |

Code change in `CappedDynamicGradientStepsCallback`: formula changed from
`clamp(ep_len, [floor, cap])` to `clamp(ep_len × scale, [floor, cap])`.
With default `scale=1.0`, prior behavior preserved.

**Not yet validated in deterministic eval.** Next cold-start training is
what will tell us if the higher batch/buffer actually helps stability.

## 6. Sim parallel training research (task #37 closed)

Concluded with a "not now" verdict. Key findings:

- **Software side, multi-instance is possible**: gym-donkeycar accepts custom
  port per env; Unity sim binary accepts `--port`. Could launch N sims on
  ports 9091/9092/.../9094 and connect N gym envs to each.
- **Hard SB3 constraint**: `off_policy_algorithm.py:552` asserts
  `train_freq.unit == STEP` when n_envs > 1. Our current
  `train_freq=(1, "episode")` is incompatible. Would require switching to
  step-based training.
- **Donkey-sim quirk that breaks naive multi-env**: when Python pauses for
  gradient updates, the Unity sim does **not** pause — the car keeps moving
  with the last action sent. In single-env episode-based mode this is hidden
  because the natural reset boundary happens at episode end; in multi-env
  step-based mode, N cars all drift simultaneously during the train phase.
- **Multi-car single-sim is not what we want either**: head-to-head racing
  mode (N cars in one Unity scene) is for competition, not parallel RL —
  shared scene, single render pipeline, no real speedup.

Realistic upside: **2-3× wall-clock speedup**, not 4×, with non-trivial
engineering (`build_env` → `SubprocVecEnv`, rewrite `CappedDynamicGradientStepsCallback`,
adapt `DonkeyInfoCallback` for batched infos, coordinate N-way pause-or-reset
behavior at train trigger). Deferred.

## 7. Repository housekeeping

Disk freed today:
- v3 (4 attempts × ~500MB) → 0
- v4 entire branch → 526MB freed
- v15 cleanup: valley + intermediate buffers → 1.2GB freed total
- net: ~3GB

Remaining mountain artifacts:
- `models/rl_dinov2_mountain_v1/` (631MB, all 6 checkpoints + buffers, kept
  for reproducibility of v2 resume)
- `models/rl_dinov2_mountain_v2/` (113MB, 40k deployment + 50k backup zip)
- `models/rl_loop_resnet_mountain_v1/` (unchanged, historical)

## 8. Open follow-ups

- Validate new batch=128 / buffer=50k defaults in the next cold-start run
  (any task) — see if late-stage stability actually improves.
- Mountain DINOv2 v2 50k has 5/5 but rising cte; if mountain deployment
  ever flakes, switch to v2 50k as a check before retraining.
- `--gradient-steps-scale 0.5` ablation deferred — try when there's a
  good A/B comparison setup.
