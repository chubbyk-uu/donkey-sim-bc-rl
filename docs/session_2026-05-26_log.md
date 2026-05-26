# Session Log: 2026-05-26 — DINOv2 series + v15 VAE breakthrough

Two-thread session: (1) sweep DINOv2 encoder variants and ablate
hidden / reward to find DINOv2's ceiling; (2) port the v8 reward
additions back to VAE — leading to v15, a new deployment-quality
checkpoint that matches v3_h80.

Everything below is reconciled against `monitor.csv`, eval stdout, and
`git diff`. No values taken from session memory.

## 1. Code changes (uncommitted, `git diff --stat`)

```
rl/eval_loop_vae_sac.py  |  48 ++++++++++++++------
rl/train_loop_vae_sac.py |  24 +++++++---
rl/train_vae_sac.py      | 114 +++++++++++++++++++++++++++--------------------
3 files changed, 117 insertions(+), 69 deletions(-)
```

### Encoder
- Added `dinov2_vits14` to `FrozenPretrainedCnnEncoder.__init__` via
  `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True, trust_repo=True)`.
  Feature dim = 384, downloaded ~85 MB to `~/.cache/torch/hub/`.
- Added `dinov2_vitb14` similarly. Feature dim = 768, downloaded ~330 MB.
- Both added to `--encoder` `choices` in `train_loop_vae_sac.py` and
  `eval_loop_vae_sac.py`.

### Reward shape (new fields in `RaffinRewardConfig`)
- `alive_scale_floor: float = 0.0` — when `min_alive_speed > 0`, the
  alive ramp is now `floor + (1-floor) × clip(speed/min_alive_speed, 0, 1)`
  instead of the hard ramp `clip(speed/min_alive_speed, 0, 1)`. A floor of
  0.5 means stopping still keeps half the alive reward, so the policy is
  not punished as harshly for slowing through corners.
- `lap_completion_bonus: float = 0.0` — sparse +N reward fired in
  `step()` whenever `info["lap_count"]` increments. Detected via
  `cur_lap_count > self._prev_lap_count`. Reset in `reset()`.

### Reward shape (removed as dead / confusing)
- Removed `progress_reward_weight`, `_calculate_progress`,
  `_extract_pos`, `self.last_pos`, `info["delta_pos_distance"]`, and the
  `import math`. The original `progress` was just
  `sqrt(dx² + dz²)` between two position samples — mathematically
  equivalent to `speed × dt`, so it was a second speed reward with no
  new information.
- Removed `throttle_reward_weight` and the `THROTTLE_REWARD_WEIGHT`
  constant. Same redundancy with `speed_reward_weight × speed` for our
  setup (always-positive throttle, flat terrain).

### Logging bug fix in `DonkeyInfoCallback`
- All `_mean` metrics switched from `self.logger.record(...)` to
  `self.logger.record_mean(...)`. With `record()` the displayed value
  was just the last `_on_step` before dump — caused the false
  "speed_mean swinging 1.74 → 3.09 between batches" reading we
  initially misinterpreted as policy oscillation.
- Same fix in `CappedDynamicGradientStepsCallback` for
  `train/gradient_steps_used` and `train/rollout_ep_len`.
- `donkey/last_lap_time_best` stays on `record()` because the callback
  maintains a running min (`self._best_lap_time`) — overwrite is the
  correct semantics.
- Removed `donkey/abs_cte_max` (numerically equivalent to
  `abs_cte_mean` in single-env mode).

### New logging fields
- `donkey/lap_count_mean`, `donkey/last_lap_time_mean`,
  `donkey/last_lap_time_best`, `donkey/forward_vel_mean`,
  `donkey/lateral_vel_mean` (= `abs(info["vel"][0])`).

### `eval_loop_vae_sac.py` extensions
- Per-episode tracking of `lap_count` (via `info["lap_count"]` watching)
  and `last_lap_time` (via `info["last_lap_time"]` watching).
- Per-episode print now ends with `laps=N best_lap=Xs mean_lap=Ys`.
- Summary ends with `total_laps`, `best_lap`, `mean_lap`.

### CLI default changes (`rl/train_loop_vae_sac.py`)
- `--buffer-size`: 60_000 → 30_000 (Raffin canonical).
- `--gradient-steps-min`: 500 → 50 (allow short crash episodes to
  do fewer-than-500 updates).
- `--learning-starts`: 1000 → 500.

## 2. Experiments (chronological, all on `donkey-generated-track-v0`)

Many runs were abandoned and the model directories were deleted to save
disk. The values below are the **post-stop** training/eval numbers that
were captured at the time. Encoder, hidden, reward shape, and timesteps
columns were taken directly from the launch command logged earlier in
the session.

| Exp | Encoder | hidden | Distinctive reward / arch | Steps run | Result | Status |
| --- | --- | ---: | --- | ---: | --- | --- |
| v7 | dinov2_vits14 | 256 | `min_alive=3.0`, `floor=0.0` | 17k | 0 truncate, ep_len ~159, "lap-2 first-corner crash" pattern | deleted |
| **v8** | dinov2_vits14 | 256 | `min_alive=3.0`, `floor=0.6` | **50k** (cold) | **30k = 3/3 trunc, speed 2.414, best_lap 10.39s, mean_lap 10.83s, 39 laps** | **kept (DINOv2 backup champion)** |
| v9 | dinov2_vits14 | 128 | v3_h80 settings (no `min_alive_speed`), `cap=2000`, `ls=1000` | stopped early | inconclusive (mix of hidden + reward change) | deleted |
| v10 | dinov2_vits14 | 128 | v8 reward (single-var hidden ablation) | 40k | 20k 1/3 trunc speed 2.28 best_lap 10.81; 30k 0/3; 40k 0/3 — worse than v8 across the board | deleted |
| v11 | dinov2_vitb14 | 256 | raffin default reward | 40k cold → resume 60k | 50k 3/3 trunc but mean_lap 13.30s (slow); 60k mean_cte drifted to 0.809 | deleted |
| v12 | dinov2_vitb14 | 256 | v8 reward (single-var encoder ablation) | 40k | 30k 3/3 trunc speed 2.094 mean_lap 12.41s — worse than v8 vits14 | deleted |
| v13 | dinov2_vits14 | 256 | v8 reward + `lap_bonus=100` + `crash_speed_weight=15` | 40k cold → resume +30k = 70k | 40k 1/3 trunc; later checkpoints all 0/3, single best_lap dropped to 9.99s but no sustained truncate | deleted |
| v14 | dinov2_vits14 | 256 | v13 reward but `lap_bonus=50`, `grad_min=100`, `grad_cap=2000` | stopped 39k | best_lap 9.21s but 1 trunc / 156 eps — same instability | deleted |
| **v15** | **vae** (fixedlight_v1) | **80** | **v3_h80 base + `lap_bonus=50` + `crash_speed_weight=15` + `grad_min=200`** | **60k cold → +30k → +20k = 112k** | **112k eval = 3/3 trunc, speed 2.795, best_lap 9.15s, mean_lap 9.47s, 45 laps** | **kept (NEW deployment candidate)** |

Notes:
- v6 (DINOv2-S no-reward-fix baseline) was abandoned/deleted in the
  prior session and is documented separately in `docs/dinov2_v6_log.md`.
- The `VisualAdapterExtractor` exploration (failed) was reverted via
  `git checkout` earlier in the day; no trace remains.

## 3. v15 detail (the actual deployment-quality result)

### Configuration

Final launch command (cold start phase, before two resumes):
```bash
python rl/train_loop_vae_sac.py \
    --env-id donkey-generated-track-v0 \
    --output-dir models/rl_loop_vae_v15 \
    --encoder vae \
    --vae-model models/vae_loop_cones_fixedlight_v1/best.pt \
    --hidden-size 80 \
    --learning-starts 1000 \
    --gradient-steps-cap 2000 \
    --gradient-steps-min 200 \
    --min-throttle 0.2 \
    --max-throttle 0.7 \
    --max-cte-error 2.0 \
    --lap-completion-bonus 50 \
    --crash-speed-weight 15 \
    --timesteps 60000 \
    --save-replay-buffer --save-final-replay-buffer \
    --device cuda
```

Then two resumes: `--timesteps 30000` and `--timesteps 20000`,
each with `--resume-model final_model.zip --resume-replay-buffer
final_replay_buffer.pkl`. Final total = ~112488 actual env steps.

History note: an earlier v15 attempt (PID 12736) was launched with
`--gradient-steps-min 50` and `--timesteps 40000`, stopped almost
immediately, the directory removed, and relaunched with the
`--gradient-steps-min 200 --timesteps 60000` shown above. The
above command is the actual one that produced the 60k cold-start
checkpoint set; the first attempt's data was discarded entirely.

What's different from v3_h80's recipe:
- `--lap-completion-bonus 50` (v3_h80 had 0)
- `--crash-speed-weight 15` (v3_h80 used the raffin canonical 5)
- `--gradient-steps-min 200` (v3_h80 used 50)
- Everything else (VAE checkpoint, hidden=80, batch=64 default,
  cap=2000, max_cte=2.0, min/max throttle 0.2/0.7) matches v3_h80.

### Deterministic eval (3 episodes × 3000 max steps)

Verified directly from eval stdout for v15:

| Checkpoint | Trunc | speed | mean_cte | best_lap | mean_lap | total_laps |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 60000 | 0/3 | 2.493 | 0.361 | 9.59s | 10.02s | 15 |
| 72336 | 0/3 | 2.626 | 0.413 | 9.14s | 9.58s | 10 |
| 82336 | 0/3 | 1.127 | 0.268 | — | — | 0 (SAC entropy collapse) |
| **92336** | **2/3** | **2.804** | **0.334** | 9.16s | 9.47s | 35 |
| 102488 | 0/3 | 1.759 | 0.381 | 10.28s | — | 1 (collapse again) |
| **112488** | **3/3** | **2.795** | **0.353** | **9.15s** | **9.47s** | **45** |

Two separate "82k / 102k completely collapsed → 92k / 112k recovered"
events. The lesson is concrete: late-stage SAC volatility is real on
this task, individual checkpoints can be unusable, and we have to eval
multiple candidates rather than trust a single "latest" model.

### Why v15 worked where v13 / v14 failed

Both v13 (lap_bonus=100, dinov2_vits14) and v14 (lap_bonus=50,
dinov2_vits14) used the same lap_completion + heavier crash design.
On DINOv2 they pushed speed up — v13 best_lap touched 9.99s, v14
touched 9.21s — but could not sustain truncate (best truncate rate
was v13 40k @ 1/3). On VAE the same reward gave 3/3 truncate plus
sub-9.5s mean_lap.

Working hypothesis: the lap bonus is an event-rate reward; SAC turns it
into an effective speed pressure of ~0.19 per unit speed (= bonus/lap_dist
× dt = 100/26 × 0.05). On VAE features, which encode lane geometry
directly, the policy can convert this pressure into "drive faster while
staying centered" without losing reliability. On DINOv2 generic features
the same pressure produces "drive faster, hope corners work" — which is
why DINOv2 + this reward → fast single laps but no sustained truncates.

## 4. Final standings (deployment + backup hierarchy)

| Rank | Model | Encoder | Trunc (3 eps) | speed | mean_lap | Role |
| ---: | --- | --- | :---: | ---: | ---: | --- |
| 1 | rl_loop_vae_sac_fixedlight_v3_h80 / 90000 | VAE | 3/3 | 2.80 | ~9.3s | main deployment |
| 1 | **rl_loop_vae_v15 / 112488** | VAE | 3/3 | 2.795 | 9.47s | **new co-deployment candidate** |
| 2 | rl_loop_dinov2_v8 / 30000 | DINOv2-S | 3/3 | 2.414 | 10.83s | DINOv2 backup |
| 3 | rl_loop_vae_sac_resnet_v4_notrees / 50000 | ResNet18 | 3/3 | 2.131 | (~12.5s rough estimate) | legacy backup |

v15 essentially ties v3_h80 on the eval metrics that have direct
numbers (truncate, speed, mean_cte). `mean_lap` is 9.47s vs v3_h80's
~9.3s — about 1.8% slower — but the gap is within the noise of a
3-episode sample.

## 5. Key findings

1. **Encoder ranking on `donkey-generated-track-v0` (frozen, no fine-tune):**
   VAE (track-specific) > DINOv2-S (self-sup, 384) > DINOv2-B
   (self-sup, 768) > ResNet18 (ImageNet supervised, 512).
   DINOv2-B was actually WORSE than DINOv2-S — v12 best 12.41s mean_lap
   vs v8 best 10.83s under identical reward — see v12 row in section 2.

2. **Reward × encoder interaction is real.** The same
   `lap_bonus + heavier crash` reward made v15 (VAE) achieve 3/3
   truncate and matched v3_h80; on DINOv2 it produced single fast
   laps without sustained truncates. Reward designs do not transfer
   across encoders blindly.

3. **Hidden size is not the bottleneck for DINOv2-S.**
   The clean v10 ablation (DINOv2-S, hidden=128, otherwise same as
   v8 hidden=256) reached comparable training ep_len but had worse
   eval reliability (0/3 trunc at 30k vs v8's 3/3). hidden=256 wins
   not by capacity, just by stability.

4. **`record()` vs `record_mean()` matters.** The original
   `donkey/speed_mean` reading was the last `_on_step` value before
   the SB3 dump — a single-step snapshot — not an average. This
   misled us into thinking the policy was wildly oscillating in v13.
   Fix is now in (see code changes section).

5. **The `progress` reward in our codebase was dead code.** The
   "progress" it computed was `√(dx² + dz²)`, i.e. 2D Euclidean
   distance between two pos samples, which is `speed × dt`
   numerically. sim does not expose along-track distance or
   along-track velocity (`forward_vel` is body-axis, not track-tangent).

6. **Late-stage SAC volatility on this task is sharp.** v15 collapsed
   at both 82k and 102k (~60 step episodes), then recovered fully at
   92k and 112k. Always eval multiple checkpoints; do not trust the
   final model alone.

## 6. Disk inventory after cleanup

Kept:
- `models/rl_loop_vae_v15/` — new champion: 10k/20k/30k/40k/50k/60k/72336/82336/92336/102488/112488 + replay buffers + tensorboard.
- `models/rl_loop_dinov2_v8/` — DINOv2 backup: 10k/20k/30k + replays.

Deleted in this session (in order): v6 (yesterday actually), v7, v9, v10, v11, v12, v13, v14, plus various adapter / mountain v2-v5 experiments. Approx 6 GB total freed.

## 7. Open follow-ups

- **Re-eval v15 112488 with more episodes (5–10)** to confirm
  3/3 is robust, not lucky on a 3-episode sample.
- **Promote v15 112488 to co-deployment with v3_h80** in `docs/experiment-log.md` once the 5–10 episode eval is in.
- **commit + push** the uncommitted code changes (encoder additions,
  reward additions, logging fixes, dead code removal).
- v15 reward + DINOv2-S **was not tested**. Earlier DINOv2 runs used
  `min_alive_speed=3.0 + floor=0.6`, not the v15 setup. A clean
  comparison would be: DINOv2-S + v15 reward (no `min_alive_speed`)
  to confirm "v15 reward only works on VAE" rather than "v15 reward
  works only without the alive ramp".
- Per-episode CSV log (discussed during session, not implemented) —
  would expose individual-episode metrics that rolling means hide.

## 8. Configuration reference

### v8 launch (DINOv2 backup, for re-creation)

The actual command run for the v8 retry (PID 5132, the one that completed
50k and produced the 30k checkpoint) was:

```bash
python rl/train_loop_vae_sac.py \
    --env-id donkey-generated-track-v0 \
    --output-dir models/rl_loop_dinov2_v8 \
    --encoder dinov2_vits14 \
    --encoder-crop-top 0 \
    --hidden-size 256 \
    --min-throttle 0.2 \
    --max-throttle 0.7 \
    --max-cte-error 2.0 \
    --min-alive-speed 3.0 \
    --alive-scale-floor 0.6 \
    --timesteps 50000 \
    --save-replay-buffer --save-final-replay-buffer \
    --device cuda
```

No explicit `--learning-starts`. Relied on the (then) CLI default of
`500` — the default had been changed from `1000` to `500` immediately
before this retry. Other defaults in effect at run time:
`batch_size=64`, `buffer_size=30000`, `gradient_steps_cap=1000`,
`gradient_steps_min=50`.

To reproduce identically regardless of future default changes, add the
explicit flag: `--learning-starts 500` and the four above.

History note: an earlier v8 attempt (PID 4989) used the same command
but was stopped almost immediately to change the `learning_starts`
default. That attempt produced no usable artifacts.

### v15 launch (new co-deployment, cold start; then 2× resume)

See section 3.

### Eval command (works for either encoder; pass `--vae-model` only if `--encoder vae`)

```bash
python rl/eval_loop_vae_sac.py \
    --model <checkpoint.zip> \
    --encoder <vae | dinov2_vits14> \
    [--vae-model models/vae_loop_cones_fixedlight_v1/best.pt] \
    --encoder-crop-top 0 \
    --episodes 3 \
    --max-episode-steps 3000 \
    --min-throttle 0.2 \
    --max-throttle 0.7 \
    --max-cte-error 2.0 \
    [--min-alive-speed 3.0 --alive-scale-floor 0.6   # v8 / DINOv2 reward] \
    [--lap-completion-bonus 50 --crash-speed-weight 15   # v15 reward]
```

Reward flags affect reported per-step reward and the termination
threshold (via max-cte-error). They do not change the policy.
