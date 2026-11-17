# Mountain track

> Historical record. Results, “current” choices and “next” plans below belong to the dated experiments, not the active work queue. See [current status](../../status.md) and [documentation index](../../README.md). Commands assume the repository root; raw `logs/` and `data/` artifacts are local-only unless explicitly noted.

### 6.7 Porting to other tracks: mountain-track cte and uphill quirks

When prepping a ResNet18 transfer test on `donkey-mountain-track-v0`, two sim-side
differences from `donkey-generated-track-v0` showed up that would silently break
any "reuse the same config" attempt:

1. **`cte = 0` is the YELLOW CENTERLINE, not the right-lane center.** Verified with
   `tools/collect_sim_frames.py --action-mode cte-pid --cte-target 0`: the PID drove
   the car onto the middle yellow line. Default spawn has cte ≈ 3.54 (consistent
   across resets). Sim's own `max_cte` is 8.0 for this env. The trained loop
   policies expect cte ≈ 0 to mean "well on the right lane", so a naive port
   spawns the agent at cte = 3.54, which already exceeds our default
   `--max-cte-error 2.0` wrapper, so the episode terminates at step 1.

   Required adjustments for mountain-track:
   - `--max-cte-error 5.0` minimum, 6.0 safer (spawn-margin), 8.0 to match sim.
   - Any reward shaping that uses `abs(cte)` is now telling the agent to drive on
     the yellow line. For driving-in-right-lane semantics, shape around
     `abs(cte - 3.5)` (or relax cte penalty entirely for this track).

2. **Initial uphill stalls at low throttle.** The default `--throttle 0.12` in
   `tools/collect_sim_frames.py` works on flat generated-track but stalls on the
   mountain uphill — the PID probe never makes progress on its first try.
   Practical defaults for mountain-track: probe with `--throttle 0.25+`, SAC training
   should keep `--max-throttle 0.7` (lowering to 0.5 to be "safe" would also
   fail the climb).

These are env-specific calibrations. Always probe a new track with the cte-pid
collector at `--cte-target 0` first, watch where the car ends up visually, and size
`--max-cte-error` / throttles accordingly.

### 6.8 Cross-track transfer attempt: ResNet on mountain-track (v1; superseded)

**Status (2026-05-27)**: superseded by the DINOv2 pipeline in §6.10. The "v2 planned"
ResNet curriculum below was not executed; mountain DINOv2 v2 became the deployment.

First test of "does the ResNet encoder pipeline transfer to a different track at all?".
Mountain-track was chosen because it's geometrically the most different from
generated-track (uphill/downhill, real-road textures) while still being a closed loop.

Pre-flight findings (see §6.7) led to:
- `--cte-target 3.5` (right-lane center, not 0)
- `--max-cte-error 2.5` initially (later relaxed to 3.5)
- `--encoder-crop-top 0` (new no-crop default for ResNet)
- Same hidden=256, batch=256 as v4 loop ResNet
- `--max-throttle 0.7` (keep default for uphill)

**Phase 1 — cold from scratch, 80k, cte=2.5**

| Checkpoint | Trunc | Steps Mean | Speed | Progress | Mean CTE | Max CTE | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 80k | 0/3 | 449 | 2.810 | 68 | 0.527 | 2.60 | 664 |

Agent learned to drive (mean_speed 2.81 — actually faster than the loop deployment's
2.80!) but every episode terminated when max_cte hit the 2.5 wrapper limit. Inspection
showed max_cte 2.51, 2.52, 2.60 — all clipped by the wrapper, not the policy giving up.

**Phase 2 — resume 80k → 110k with cte=3.5**

| Checkpoint | Trunc | Steps Mean | Speed | Progress | Mean CTE | Max CTE | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **90k** | **1/3** | **1446** | 3.230 | 233 | 0.539 | 3.59 | 2205 |
| 100k | 0/3 | 348 | 3.091 | 56 | 0.643 | 3.63 | 475 |
| 110k | 0/3 | 1302 | 3.366 | 220 | 0.531 | 3.72 | 2001 |

Loosening the cte wrapper from 2.5 to 3.5 immediately bought a 2741-step rollout in
the first dump after resume, and a clean 3000-truncate by the second dump. But
deterministic eval shows the policy is still **fast but fragile**: max_cte still
pushes 3.5+, only 1/3 truncate at the best checkpoint (90k), 100k briefly regressed.

Speed numbers (3.0+) are higher than the loop deployment, almost certainly because
mountain-track has downhill sections that accelerate the car beyond what the loop
gives.

**Mountain throttle distribution at 110k (longest 3000-truncate episode)**:

| Throttle bin | Share |
| --- | ---: |
| [0.20, 0.25) min | 32.4% |
| middle 0.25–0.65 | 46.4% |
| [0.65, 0.70) max | 21.2% |

Less bang-bang than the loop policy (v2_h64 was 40.8% min + 12.1% max). Mountain
agent uses the middle range more, reflecting more nuanced throttle modulation for
uphill/downhill rhythm.

**Open questions / next steps** (planned for next session):

- The current v1 plateau at 1/3 truncate suggests it's not just "needs more training" —
  the agent's fast-but-fragile profile points to the curriculum being too aggressive
  from the start. Plan a `mountain_v2` run with curriculum mode:
  - `--max-throttle 0.5` to cap downhill speed during early training
  - `--max-cte-error 3.0` from the start (no late switching)
  - Same encoder / hidden / batch
  - Cold start, 80k initial
- If v2 still plateaus at 1/3 truncate, the limit is likely ResNet's domain-general
  features on a track ResNet has never specifically seen.

Kept artifacts for reference / future resume:

```text
models/rl_loop_resnet_mountain_v1/sac_resnet18_90000_steps.zip          (v1 best)
models/rl_loop_resnet_mountain_v1/sac_resnet18_replay_buffer_90000_steps.pkl
models/rl_loop_resnet_mountain_v1/sac_resnet18_80000_steps.zip          (pre-resume)
```

All other v1 checkpoints + buffers deleted (2.85 GB freed).

### 6.10 Mountain Track DINOv2 Pipeline

First successful deployment on `donkey-mountain-track-v0`. Built 2026-05-27 after
the ResNet attempt (§6.8) plateaued.

Encoder: **frozen DINOv2-S** (`dinov2_vits14` via torch.hub, z=384). No per-track
VAE collection required — DINOv2 is pretrained.

#### v1 — cold start (raffin reward + max_cte 2.5 + cte_target 3.5)

```bash
python rl/train_loop_vae_sac.py \
    --env-id donkey-mountain-track-v0 \
    --output-dir models/rl_dinov2_mountain_v1 \
    --encoder dinov2_vits14 --encoder-crop-top 0 \
    --hidden-size 256 \
    --batch-size 64 --buffer-size 30000 \
    --learning-starts 500 \
    --gradient-steps-cap 1000 --gradient-steps-min 200 \
    --cte-target 3.5 --max-cte-error 2.5 \
    --min-throttle 0.2 --max-throttle 0.7 \
    --alive-reward 1.5 --speed-reward-weight 0.15 \
    --reward-crash -20 --crash-speed-weight 10 \
    --cte-speed-penalty-weight 0.25 \
    --timesteps 80000 --seed 42 --device cuda \
    --save-replay-buffer --save-final-replay-buffer
```

Stopped at 60k. Deterministic eval (5 ep × 2000 max steps):

| Checkpoint | Trunc | Speed | Mean CTE | Max CTE | Best lap | Mean lap |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| **30k** | **5/5** | 1.998 | 0.682 | 2.163 | 29.44s | 31.09s |
| 40k | 3/5 | 2.124 | 0.778 | 2.619 | 28.16s | 29.14s |
| 50k | 0/5 | 1.747 | 0.874 | 2.748 | — | — |
| 60k | 4/5 | 1.917 | 0.673 | 2.516 | 30.08s | 31.19s |

30k 5/5 is genuine — first DINOv2 mountain win. 40k+ shows the 10-20k oscillation
pattern (50k full valley, 60k partial recovery).

#### v2 — resume from v1 30k with stricter reward (current deployment)

Hypothesis: v1 30k learned to drive but with high cte variance. Tighten reward to
trim it. Resume with smaller LR, slightly stricter cte, heavier crash:

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
    --timesteps 40000 \
    --seed 42 --device cuda \
    --save-replay-buffer --save-final-replay-buffer
```

Deterministic eval:

| Checkpoint | Trunc | Speed | Mean CTE | Max CTE | Best lap | Mean lap |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| **40k** | **5/5** | 2.093 | 0.577 | 2.112 | 28.41s | 29.88s |
| 50k | 5/5 | 2.088 | 0.801 | 2.279 | 28.29s | 29.96s |
| 60k | 0/5 | 2.520 | 0.746 | 2.742 | — | — |

**v2 40k beats v1 30k on every metric** (speed +5%, mean_cte -15%, max_cte -2%,
mean_lap -4%) and stays at 5/5 truncate. **Current mountain deployment.**
50k is 5/5 but mean_cte rising; 60k collapsed (same 10-20k cycle).

#### v3 — cold start with v2-style stricter reward (failed)

Four cold-start attempts trying to learn v2's reward shape from scratch
(seed=42 / 7, with and without max_cte=3.0, cte_pen 0.25 / 0.30). **All four
got stuck at the first corner around 130-160 ep_len** with ent_coef collapsing
to ~0.02 within 5k steps. The stricter cte penalty actively prevented the
exploration needed to learn corner-taking. Branch deleted.

Key learning: **stricter cte penalty is incompatible with cold start on
mountain**. Need to learn corners first under permissive reward, then refine.

#### v4 — resume from v1 *20k* with v2 reward (4/5, not deployable)

Tested whether earlier-stage v1 checkpoint (less committed than 30k) would
respond better to v2 reward. Reached 4/5 truncate at the 60k step but
deterministic eval shows v2 40k still beats it on every metric. Branch deleted.

| Checkpoint | Trunc | Speed | Mean CTE | Max CTE | Best lap | Mean lap |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| 60k | 4/5 | 1.988 | 0.728 | 2.503 | 29.58s | 31.25s |

Confirmed that v2's resume-from-30k path is the right recipe; v4 is left as
documented evidence not as an artifact.

#### Lessons from the mountain DINOv2 series

- **DINOv2-S is a viable encoder on mountain track** — first time we got
  ≥3/5 truncate there without per-track encoder training.
- **Stricter cte reward requires a "knows how to drive" baseline.** v3 cold
  start with stricter reward fails identically across seeds; v2 resume from
  permissive-reward baseline succeeds.
- **The 10-20k peak-collapse-recovery cycle persists on mountain**, identical
  to the loop pattern. Eval every checkpoint, deploy the peak.
- **Training-time trunc rate dropping ≠ deterministic eval degrading.** v4 60k
  had its training-time trunc rate fall from 42% → 15% but deterministic eval
  at 60k was actually the best of the v4 batch (4/5). The training-time drop
  reflects exploration noise; deterministic policy is more stable.

### 6.13 New-defaults validation: mountain DINOv2 cold start (2026-05-28)

First run exercising the raised defaults (§6.11). Used mountain DINOv2 (faster to
train than loop h80) as the test vehicle: a single cold start with the permissive
v1-recipe reward (`speed=0.15, cte=0.25`, `reward_crash=-20, crash_speed=10`).

Config tested vs the actual §6.11 defaults — **note the confound**:

| Knob | This run | §6.11 default |
| --- | ---: | ---: |
| `--batch-size` | **256** | 128 |
| `--buffer-size` | 50_000 | 50_000 |
| `--gradient-steps-scale` | **0.5** | 1.0 |
| `--gradient-steps-cap` | 1000 | 1000 |

Batch is double the default and scale=0.5 is the *ablation* value (§10), not the
default. So this run conflates the §6.11 validation with the `gradient-steps-scale`
ablation at a non-default batch — neither question is answered cleanly.

Deterministic eval (5 ep × 2000 max steps; 30-60k from cold start, 70-90k from a
resume that overshot to 90k — see session log 2026-05-28 §3):

| Checkpoint | Trunc | Speed | Mean CTE | Max CTE | Best lap | Mean lap |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| 30k | 0/5 | 2.04 | 0.59 | 2.76 | 25.90s | 27.80s |
| **40k** | **5/5** | 2.18 | 0.66 | 2.27 | 27.00s | 28.24s |
| 50k | 3/5 | 2.28 | 0.57 | 2.25 | 25.57s | 26.89s |
| 60k | 4/5 | 2.21 | 0.53 | 2.54 | 26.60s | 27.95s |
| 70k | 2/5 | 2.09 | 0.69 | 2.41 | 28.08s | 29.08s |
| 80k | 4/5 | 2.10 | 0.70 | 2.63 | 28.09s | 29.24s |
| 90k | 3/5 | 2.03 | 0.60 | 2.57 | 28.64s | 30.13s |

**Findings:**

- **A single-stage cold start reached a deployable 5/5 (40k).** v3 40k beats the
  two-stage v2 40k deployment (§6.10) on speed (2.18 vs 2.093) and mean lap
  (28.24s vs 29.88s), at the cost of running wider (mean_cte 0.66 vs 0.577,
  max_cte 2.27 vs 2.112). The aggressive optimizer config let one cold start do
  what previously needed v1-cold + v2-resume. This is the run's main positive
  signal.
- **The §6.11 defaults (batch=128, scale=1.0) remain unvalidated** — this run
  cannot separate their effect from the extra-aggressive batch and the scale
  change.
- **Late-stage drift persisted.** scale=0.5 + batch=256 did not dampen the
  10-20k peak-collapse-recovery cycle; 70-90k were uniformly worse than 40-60k,
  and the extra 30k steps past 60k bought nothing. The peak shifted only
  marginally later than v1's 30k (§6.10), to the 40-50k zone.

**Deployment decision:** mountain primary stays **v2 40k** (§6.10). v3 40k is kept
as a new-defaults validation artifact and a faster single-stage cold-start
alternative — see README Model Inventory.
