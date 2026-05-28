# Session Log: 2026-05-28 — New-defaults validation on mountain DINOv2

Goal for the day: take the SAC defaults raised on 2026-05-27 (§6.11) for a test
drive. Instead of re-running the loop h80 recipe, we used mountain DINOv2 (faster
to train) as the validation vehicle. End result: a single-stage cold start reached
a deployable 5/5 checkpoint at 40k — faster than the two-stage v2 deployment — but
the run used a *more aggressive* config than the actual new defaults, so it does
not cleanly validate them.

## 1. Config under test

Cold start on `donkey-mountain-track-v0`, DINOv2-S, permissive v1-recipe reward
(`speed=0.15, cte=0.25`, `reward_crash=-20, crash_speed_weight=10`). The knobs
being exercised:

| Knob | v1 cold-start (old) | This run | §6.11 default |
| --- | ---: | ---: | ---: |
| `--batch-size` | 64 | **256** | 128 |
| `--buffer-size` | 30_000 | **50_000** | 50_000 |
| `--gradient-steps-scale` | 1.0 | **0.5** | 1.0 |
| `--gradient-steps-cap` | 1000 | 1000 | 1000 |

Note the confound: batch=256 is double the new default, and scale=0.5 is the
*ablation* value, not the default. So this run mixes the §6.11 default validation
with the §10 `gradient-steps-scale` ablation, and at a non-default batch. Neither
question gets a clean answer.

```bash
python rl/train_loop_vae_sac.py \
  --env-id donkey-mountain-track-v0 \
  --encoder dinov2_vits14 --encoder-crop-top 0 \
  --hidden-size 256 \
  --batch-size 256 --buffer-size 50000 \
  --gradient-steps-cap 1000 --gradient-steps-min 200 \
  --gradient-steps-scale 0.5 \
  --cte-target 3.5 --max-cte-error 2.5 \
  --min-throttle 0.2 --max-throttle 0.7 \
  --alive-reward 1.5 --speed-reward-weight 0.15 \
  --reward-crash -20 --crash-speed-weight 10 \
  --cte-speed-penalty-weight 0.25 \
  --output-dir models/rl_dinov2_mountain_v3 \
  --timesteps 60000 --seed 42 --device cuda --save-replay-buffer
```

## 2. Training trajectory (cold start, 0→60k)

Healthy throughout. `ent_coef` decayed smoothly (0.087 → 0.030 by 50k) without the
collapse-to-0.02 basin that killed the v3-reward cold starts on 2026-05-27.
`critic_loss` stayed ~6. `ep_len_mean` climbed past 480 by 50k; truncate episodes
(3000-step) started appearing densely after 40k. `gradient_steps_used` tracked
`ep_len × 0.5` (200 floor early, rising to 700-800 on long episodes), confirming
`scale=0.5` was active.

FPS note: early FPS (~10) is lower than v1's reported ~17 because FPS = total
steps / total wall-clock includes the update phase. Early episodes are short
(~40 steps) so the 200-step update phase dominates; FPS rises to ~17-18 as
episodes lengthen. Not a slowdown in env stepping.

## 3. The `--timesteps` resume mistake

After eval (below), we wanted "20k more, to 80k total" and resumed with
`--timesteps 80000`. **Wrong.** On resume `reset_num_timesteps=False`
(`train_loop_vae_sac.py:206`), and SB3 then does `total_timesteps += num_timesteps`
— so the real target became 60k + 80k = **140k**, i.e. 80k *additional* steps. The
correct value for "20k more" is `--timesteps 20000`. This also retroactively
explains why v2 ("resume from 30k + `--timesteps 40000`") actually ran to 70k.

We let it continue and stopped at the 90k checkpoint. A background watcher polled
for the 90k replay buffer then `pkill`ed the trainer — but the `pkill -f` pattern
also matched the watcher's own command line (it contained the pattern string), so
the watcher killed itself right after killing the trainer (both exit 144). Outcome
was still correct: trainer stopped, 90k checkpoint + 170MB replay buffer saved.
**Gotcha for next time: don't put the pkill target pattern literally in the
watcher script, or exclude self with `pgrep -v $$`.**

## 4. Deterministic eval (5 ep × 2000 max steps, all cold-start + resume checkpoints)

| ckpt | Trunc | Speed | Mean CTE | Max CTE | Best lap | Mean lap |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| 30k | 0/5 | 2.04 | 0.59 | 2.76 | 25.90s | 27.80s |
| **40k** | **5/5** | 2.18 | 0.66 | 2.27 | 27.00s | 28.24s |
| 50k | 3/5 | 2.28 | 0.57 | 2.25 | 25.57s | 26.89s |
| 60k | 4/5 | 2.21 | 0.53 | 2.54 | 26.60s | 27.95s |
| 70k | 2/5 | 2.09 | 0.69 | 2.41 | 28.08s | 29.08s |
| 80k | 4/5 | 2.10 | 0.70 | 2.63 | 28.09s | 29.24s |
| 90k | 3/5 | 2.03 | 0.60 | 2.57 | 28.64s | 30.13s |

Only 40k is 5/5. 50k is fastest (best_lap 25.57s) but 3/5. Everything past 60k
drifted: slower laps (29-30s mean), trunc rate 2-4/5, no improvement from the
extra 30k steps.

## 5. Conclusions

- **Single-stage cold start now reaches deployable 5/5.** v3 40k (5/5, speed 2.18,
  mean_lap 28.24s) beats the current two-stage v2 40k deployment (5/5, speed
  2.093, mean_lap 29.88s) on speed and lap time — at the cost of running wider
  (mean_cte 0.66 vs 0.577). This is the run's main positive signal: the aggressive
  optimizer config let one cold start do what previously needed v1-cold + v2-resume.
- **The §6.11 defaults (batch=128, scale=1.0) are still not cleanly validated.**
  This run used batch=256 + scale=0.5. We cannot attribute the result to the
  defaults vs the extra-aggressive batch vs the scale change.
- **Late-stage drift persists.** scale=0.5 + batch=256 did not dampen the
  peak-valley cycle; 70-90k were worse than 40-60k. The peak shifted only
  marginally later than v1's 30k (to 40-50k).
- **Deployment decision: keep v2 40k as mountain primary**; record v3 40k as a
  new-defaults validation artifact and a faster single-stage cold-start
  alternative.

## 6. Reward-design idea discussed (not pursued)

Idea: make `alive_reward` grow with survival step count (longer alive → higher
alive rate). Rejected for two reasons:
1. **Step count is not in the observation** (camera latent only). On a loop track
   the same scene recurs each lap, so identical observations would get different
   rewards depending on step — non-Markovian noise the critic cannot fit, which
   would push `critic_loss` back up right after batch=256 was chosen to reduce it.
2. **Return becomes super-linear in episode length** (O(T²)), blowing up Q-value
   range across episode lengths.
The well-formed version of "reward sustained progress" already exists:
`lap_completion_bonus` (`train_vae_sac.py:56-58`) — discrete, tied to an
observable milestone, bounded. v15 used it successfully on loop. The real lever
for escaping early-crash basins is exploration (ent_coef floor / noise /
curriculum), not reward-slope steepening.
