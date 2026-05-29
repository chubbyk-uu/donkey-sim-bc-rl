# Session Log: 2026-05-28 — Mountain new-defaults + loop domain randomization

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

## 7. Loop domain randomization: surviving random light + trees

Motivation: with the sim's `randomlight` + random trees enabled, the plain loop
DINOv2 model (`rl_loop_dinov2_v8`) crashed at step 242 (0/1). Goal: train DINOv2
to stay robust to random trees/shadows too (it was already lighting-robust,
§6.12), via domain randomization.

### 7.1 Building scene-reload domain randomization

The sim only regenerates trees/lighting when the **scene is reloaded**; an
episode `reset_car` keeps the same layout. Built `reload_scene()` in
`train_vae_sac.py` and a probe `tools/test_scene_reload.py`. Findings while
getting it to work:
- Re-sending `load_scene` mid-scene is **ignored**; must `exit_scene` to the menu
  first, then `load_scene`.
- The sim sends no reliable "exited" signal, so: `exit_scene` → fixed settle →
  re-send `load_scene` and poll `handler.loaded` until it re-handshakes. A
  too-eager retry (zero settle) re-enters before the exit completes and the scene
  never changes; ~1s settle fixed it. Confirmed visually that trees + shadows +
  light all change on reload.

### 7.2 Reload cadence — episode-based → adaptive step-based

- **v1** (episode-based `--scene-reload-every`): cold start, reload every N
  episodes. Worked early (short episodes → frequent reloads), but once episodes
  lengthened, N episodes = many thousands of steps on one layout → the replay
  buffer filled with easy-scene data (easy layouts truncate at 2000; hard ones
  crash in ~100), biasing learning toward easy layouts.
- Fix: **adaptive step-based reload** (`--scene-reload-alpha`,
  `--scene-reload-kmin`): reload after `K = clamp(alpha × recent_ep_len, [kmin,
  max_ep-1])` steps on a scene, checked only at episode boundaries.
  `_steps_on_scene` accumulates across crashes, resets on reload → each layout
  contributes ~equal steps. Design insight: with the no-mid-episode-truncate
  rule, a *small* fixed K worsens balance (an easy scene still spends a full
  episode before the check), so K≈max_ep balances best; tying K to ep_len gives
  early variety + late balance.

### 7.3 Runs

- **v2** — resumed from v1 10k with adaptive reload; the keeper. Paired eval (6
  same layouts): 30k 3/6, 50k 3/6 (best, lowest cte), 40k/60k 2/6. Independent
  5-ep eval: 30k/50k 4/5. ⇒ ~50% truncate on unseen random layouts; "failures"
  are 4-6 lap drives that drift to cte≈2.0.
- **v3** — cold start, cte 0.25→0.30 + min-throttle 0.18: no clear gain, still
  weaves.
- **v4 / v5** — steering-change penalty to kill the weave: v4 (linear w=0.5,
  resume) and v5 (squared w=2.0, cold start). Both left `abs_steer_delta_mean`
  **dead flat at ~0.30** — zero effect. Removed the reward term.
- **ViT-B/14** (z=768, hidden=384): same weave (~0.30), trunc no better, ~2×
  slower (fps 9 vs 15). Deleted.

### 7.4 Findings + cleanup

- The left-right **weave is identical across encoders (ViT-S/ViT-B) and unmoved by
  linear/squared steer penalties** — but its root cause is **undetermined**. (This
  log first called it "a SAC control artifact"; a 2026-05-29 buffer probe showed
  that overstated the evidence — see experiment-log §6.14 for the corrected
  analysis and the untried `max_steering_diff` lever.)
- The ~50% random-layout ceiling is **not broken by a bigger frozen encoder** →
  next lever is a task-adapted encoder (fine-tune / depth / segmentation). Full
  analysis in experiment-log §6.14.
- New tooling kept: `rl/eval_paired_randomized.py` (compare checkpoints on the
  same random layouts — removes the layout-luck confound of independent eval),
  `tools/test_scene_reload.py`.
- Cleanup: kept only `rl_loop_dinov2_randomtree_v2`; deleted v1/v3/v4/v5 and the
  ViT-B run (freed ~5 GB).
- Gotcha noted: an early monitor-loop helper used `grep -oE "[0-9]+$"` on SB3
  table lines that end in `|`, so it never matched and looped forever. Don't
  build polling watchers that way.
