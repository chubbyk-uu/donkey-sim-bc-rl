# Fixed-light loop

> Historical record. Results, “current” choices and “next” plans below belong to the dated experiments, not the active work queue. See [current status](../../status.md) and [documentation index](../../README.md). Commands assume the repository root; raw `logs/` and `data/` artifacts are local-only unless explicitly noted.

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
