# Loop follow-up, 2026-05-27

> Archived session notes; see [current status](../../status.md).

## Day 2 follow-up (2026-05-27)

Three questions left over from yesterday were resolved this morning:

1. Was v15 112488's "3/3 truncate" a real result, or 3-episode luck?
2. Does a further `+20k` resume to 132k buy anything?
3. Does the 20k-cycle "crash-recover" pattern continue past 112k?

### 1. v15 112488 re-evaluated with 5 episodes — 5/5 truncate confirmed

Same eval command, episodes raised to 5. Every episode hit the
3000-step truncate with 15 laps each; mean_lap variance was
0.43-0.49 across episodes.

| Episode | Steps | Speed | Best lap | Mean lap | Laps |
|---|---:|---:|---:|---:|---:|
| 1 | 3000 | 2.797 | 9.18 | 9.45 | 15 |
| 2 | 3000 | 2.791 | 9.17 | 9.49 | 15 |
| 3 | 3000 | 2.809 | 9.11 | 9.43 | 15 |
| 4 | 3000 | 2.797 | 9.15 | 9.49 | 15 |
| 5 | 3000 | 2.793 | 9.14 | 9.49 | 15 |

**Summary (5ep):** 5/5 truncate, speed 2.797, best_lap 9.11s,
mean_lap 9.47s, 75 laps total, mean_cte 0.356, max_cte 1.72.

The yesterday-reported 3/3 was not a small-sample artefact. 112488 is
locked as the v15 deployment checkpoint.

### 2. +20k resume to 132488 — net regression on stability

Resumed from `sac_loop_vae_112488_steps.zip` + matching replay buffer
with identical reward / hidden / grad config:

```bash
python rl/train_loop_vae_sac.py \
    --env-id donkey-generated-track-v0 \
    --output-dir models/rl_loop_vae_v15 \
    --encoder vae \
    --vae-model models/vae_loop_cones_fixedlight_v1/best.pt \
    --hidden-size 80 \
    --gradient-steps-cap 2000 \
    --gradient-steps-min 200 \
    --min-throttle 0.2 \
    --max-throttle 0.7 \
    --max-cte-error 2.0 \
    --lap-completion-bonus 50 \
    --crash-speed-weight 15 \
    --timesteps 20000 \
    --resume-model models/rl_loop_vae_v15/sac_loop_vae_112488_steps.zip \
    --resume-replay-buffer models/rl_loop_vae_v15/sac_loop_vae_replay_buffer_112488_steps.pkl \
    --save-replay-buffer --save-final-replay-buffer \
    --device cuda
```

Training ran 21,469 env steps (19 min wall-clock, fps 17) producing
new checkpoints at 122488 and 132488. No late-stage `ent_coef`
collapse during this slice (held at 0.04-0.05). best_lap during
training improved 9.15 → 8.88s; mean_lap 9.47 → 9.35s.

Deterministic eval, 5 episodes each:

| Checkpoint | Trunc | Speed | Best lap | Mean lap | Laps total |
|---|---:|---:|---:|---:|---:|
| 112488 (5ep) | **5/5** | 2.797 | 9.11 | 9.47 | 75 |
| **122488** (5ep) | **0/5** | 2.609 | 9.01 | 9.79 | 9 |
| **132488** (5ep) | **2/5** | **2.843** | **8.83** | **9.26** | 53 |

- 122488 = valley, same shape as 82k / 102k from yesterday.
- 132488 has the fastest single-lap and mean-lap numbers in the whole
  v15 run, but ep3 crashed at step 518 (the rest were 3000/1785/3000/1962).
  Faster but not 5/5; under the "100% truncate or it's not deployable"
  rule, 132488 is out.

### 3. The 20k crash-recover cycle continues

Combined v15 picture:

```
72k=good  82k=collapse  92k=good  102k=collapse  112k=good (5/5)  122k=collapse  132k=fast-but-fragile
```

Every 20k env steps, the deterministic eval moves between "stable
truncate" and "fragile / valley". The cycle is reproducible. Root
cause not investigated, but the practical implication is firm: **stop
training v15 at 112488**. Continuing the resume produces oscillation
without sustained gain — peak best_lap drops 0.27s at the "recovery"
local maximum, but the cycle leaves an unstable checkpoint between
every good one.

### 4. Decision

- **Deployment checkpoint**:
  `models/rl_loop_vae_v15/sac_loop_vae_112488_steps.zip`
  (5/5 trunc, mean_lap 9.47s, speed 2.797).
- **Retain on disk**: `sac_loop_vae_132488_steps.zip` and matching
  replay buffer. Faster single-lap policy; useful as a "could go
  faster but unstable" reference. Not deployed.
- **Delete candidates** (not done today): 122488 valley checkpoint
  and intermediate replay buffers at 72336/82336/92336/102488. Would
  free ~700 MB.

### 5. v3_h80 still in play

For reference, the existing v3_h80 90k deployment (mean_lap ~9.3s,
mean_cte 0.315) is slightly faster on mean_lap and tighter on cte
than v15 112488. v15 is **not** a replacement; it is a second valid
deployment recipe using a different reward formulation (lap bonus +
heavier crash, no min_alive_speed). Both should appear in
`experiment-log.md` §8 as co-primary.
