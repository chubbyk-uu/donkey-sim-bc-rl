# Ground-truth teacher

> Historical record. Results, “current” choices and “next” plans below belong to the dated experiments, not the active work queue. See [current status](../../status.md) and [documentation index](../../README.md). Commands assume the repository root; raw `logs/` and `data/` artifacts are local-only unless explicitly noted.

## 13. Ground-truth state policy: is perception the limit? (2026-11-12/13)

**Status: trained and evaluated once (one seed per run, 24 paired layouts).**
Short answer: perception limits speed; it is not shown to limit completion.

### Question and setup

Two observation changes in one day (§10 status) gave similar training curves, so
the limit might be the SAC and control setup rather than what the policy sees.
Test: same recipe, simulator setting and seed as the class-token control, but
the observation is ground truth instead of the image
(`--encoder sim_state --track-centerline models/track_centerline/generated_track.npz`,
`rl/track_state.py`): CTE, heading error, forward and sideways speed, yaw rate,
and six centre-line points 0.3 to 3.0 position units ahead in the car's frame
(18 numbers) plus the usual 20-step command history. The run configurations
differ only in the encoder.

The simulator does not report the track, so `tools/fit_track_centerline.py`
recovers the centre line from telemetry of four §12 runs (223,542 points): 534
points, 26.7 position units per lap, 360 degrees of turning, tightest radius
0.85. Its signed distance times 7.98 reproduces the simulator's CTE with
R2 = 0.967 (MAE 0.06, up to about 0.15 off inside bends, exact on straights).
So the CTE limit of 2.0 is 0.25 position units. Telemetry conventions found on
the way: `vel` is in the car's frame (x sideways, z forward), yaw is `car[2]`
(0 faces +z, grows towards +x), `gyro[1]` is the yaw rate, positive CTE is right
of centre, and the track geometry is identical in every layout.

### Training-time comparison (stochastic actions; not an evaluation)

| Run (60k steps each) | Laps logged | Lap time, steps 50-60k (median / best in run) | Steps 50-60k spent in full 2000-step episodes | Mean throttle / forward speed, steps 40-50k |
|---|---:|---:|---:|---:|
| Ground truth (`models/rl_loop_sim_state_ts2_v1`) | 203 | 9.27 s / 8.98 s | 88% (4 of 5 episodes) | 0.47 / 2.56 |
| Class token control (`models/rl_loop_dinov2_cls_ctrl_ts2_v1`) | 146 | 11.88 s / 10.82 s | 40% (2 of 17) | 0.28 / 1.89 |
| Grid + lags 2,4 (`models/rl_loop_dinov2_grid_lags_v1`) | 132 | 12.89 s / 11.78 s | 39% (2 of 14) | 0.27 / 1.92 |

Source: `monitor.csv` and `train.log` in each model directory.

- The ground-truth run was **behind** both vision runs until about 40k steps
  (mean episode length 214 against 320 and 366 at 20-30k). No episode passed 255
  steps before 34k; at 20-30k, 32 of 47 episodes ended at 225-255 steps, one lap
  (about 10.6 s from standstill, 212 steps) plus about 28 steps. At its speed that is the first
  right-hand bend after the start line taken at full speed, the tightest on the
  track and the largest cluster of v2 50k failures (§12 runs: centre-line index
  71-79 of 534). The position is inferred from timing, not telemetry. Minimum
  throttle is 0.2, so slowing for that bend means lifting early on the straight.
- Once past it, the last six episodes are 2000, 1126, 2000, 2000, 2000, 2000
  steps, at lap times about 22% below the control.
- The vision policies sit near minimum throttle (0.27-0.28); the ground-truth
  policy uses 0.47. Mean |steering change| per step is about 0.28-0.32 in all
  three, so the jerky steering is not caused by vision.
- All 22 v2 50k failures in the two 40-layout §12 runs end at CTE of about -2:
  off the outside of a right-hand bend.

### Paired evaluation of the 50k checkpoints (2026-11-13)

`logs/paired_state_vs_vision_24layouts/`: 24 fresh random-light/tree layouts,
each model once per layout in random order, 2000-step cap, time scale 2 with FPS
limit 120 (step interval median 25.0 ms, p5-p95 23.7-26.2 for all three).

| Model | Laps (95% CI) | Completed (95% CI) | Flying lap time | Mean speed | Mean throttle | Mean abs steering change |
|---|---:|---:|---:|---:|---:|---:|
| Ground truth 50k | 9.42 (8.50-10.00) | 21/24 = 88% (69-96%) | 9.18 s | 2.89 | 0.44 | 0.26 |
| Grid + lags 50k | 5.04 (4.04-5.96) | 10/24 = 42% (24-61%) | 12.97 s | 1.99 | 0.25 | 0.29 |
| Class token control 50k | 1.92 (0.92-3.04) | 4/24 = 17% (7-36%) | 12.29 s | 1.97 | 0.29 | 0.31 |
| v2 50k, §12 run at the same setting (unpaired, 40 layouts) | 6.08 (5.42-6.62) | 31/40 = 78% (62-88%) | 13.47 s | | | |

Flying lap time excludes each episode's first lap (standing start).

- Ground truth against the vision models on the same layouts: +4.38 laps over
  grid + lags (CI +2.96 to +5.67; 13 layouts only it completed, 2 only the
  other, p=0.007) and +7.50 over the control (CI +5.92 to +8.79; 18 against 1).
- Ground truth against v2 50k (unpaired): +3.34 laps (CI +2.28 to +4.28) and
  lap time 32% lower. **Completion is not distinguishable**: 21/24 against
  31/40, Fisher p=0.51. Every completed ground-truth episode is exactly 10 laps;
  the gain is speed.
- How it is faster: it lifts to throttle 0.22-0.26 from about 1 position unit
  before the start line until the apex of the first bend (speed 3.4 down to
  2.4), then uses 0.6-0.7 on exits. Before that bend it moves from the right
  (+0.5 CTE) to the left (-0.8 mean). The vision policies hold throttle near the
  0.2 floor all lap.
- Its three failures are the same event: CTE -2.03 at centre-line index 41-48
  of 534, on that move to the left just before the first bend, at speed 2.8,
  after 2, 4 and 10 laps (3 of about 230 passes). The fitted centre line agrees
  with the simulator's CTE to 0.01 there, so a wrong lateral offset is not the
  cause; the 3-degree heading kink at the start line 2 units earlier is not
  excluded as a contributor.
- Steering stays jerky with perfect perception (0.26 per step against
  0.29-0.31), so that belongs to the control setup.
- Grid + lags 50k against v2 50k (unpaired): laps -1.03 (CI -2.16 to +0.10),
  completion 10/24 against 31/40 (Fisher p=0.007). No support for that
  observation change. 10 of its 14 failures are at CTE +2 (right side); every
  control and v2 50k failure is at -2.
- The control 50k has the v2 recipe yet scores 1.92 laps against 6.08. It was
  trained at time scale 2 (v2 at 1) and its 40-50k training window was its worst
  (no full episode). Run or checkpoint variance and training time scale are not
  separated. A single trained run per variant cannot rank observation changes.
- The ground-truth policy's own compute is 0.08 ms per step against 3.8-3.9 ms
  for the image encoders (about 8 ms of simulated time at time scale 2).

### Why the ground-truth policy still fails 3 of 24 (telemetry, 226 passes)

- It weaves on the start straight every lap, phase-locked to the track: median
  steering -0.59 at 30 centre-line points before the start line, +0.58 at 10
  before, -0.67 at 20 after; heading error swings between about +8 and -15
  degrees on a straight road. Heading-error spread is 9.5-10.9 degrees around
  the start line against 4.1-6.1 on the other straights.
- The last left swing comes just before the first right-hand bend. In the three
  failed passes it is full lock (-1.0) and the heading error reaches -24 to -28
  degrees (median pass -14, 5th percentile -22); full right lock from there does
  not bring the car back. 16 of the 223 successful passes also went below CTE
  -1.6 and 3 below -1.8, so the margin is thin every lap.
- The simulator's own CTE has a small jog at the start line (its zero line
  moves about 0.15 CTE and turns 2.75 degrees over about 1 position unit; the
  fitted line follows it to 0.01). Whether it sets off the larger weave there is
  not tested. The vision policy also weaves slightly more there (9.0-9.7
  degrees against 6.0-8.4 elsewhere).
- This settles the question left open in §6.14: the steering weave is present
  with perfect perception, so it comes from the control side, not from a noisy
  lateral estimate. Reward penalties on steering change did not remove it
  (§6.14); tightening `max_steering_diff` is still untried.
- Consequence: 88% is not good enough for a teacher. Fix the weave on the
  ground-truth policy first; each try costs one 36-minute training run without
  any vision in the loop.

### What it does and does not show

- Shown (one seed): with the same SAC recipe, action limits and reward, perfect
  perception gives a policy that brakes for bends and laps 32% faster than the
  best vision checkpoint. The speed ceiling of the vision policies is on the
  perception side, not in SAC or the reward.
- Not shown: that perfect perception makes driving more reliable. 88% against
  78% is within noise, and the failures come from one aggressive spot.
- Open: hidden size was left at 256 (a smaller network was not tried); the
  ground-truth run stalled for about 20k steps at the first bend before learning
  to lift early; one seed.

### Can image features predict the teacher's action? (offline, 2026-11-13)

`tools/probe_teacher_action.py`, output `logs/probe_teacher_action_state_50k/`.
The ground-truth 50k policy labels 18,020 captured frames (150 layouts, the five
§12 runs with tail capture) with the action it would take there; a 2x256 network
predicts that action from each candidate input plus the command history, scored
on layouts held out of its training.

| Input (+ command history) | Steering R2 | Steering MAE | Throttle R2 | Throttle MAE |
|---|---:|---:|---:|---:|
| Command history only | 0.03 | 0.57 | 0.36 | 0.28 |
| Teacher's own state (ceiling of the procedure) | 0.97 | 0.10 | 0.91 | 0.11 |
| Class token (v2 input) | 0.76 | 0.28 | 0.66 | 0.22 |
| Grid, PCA 128 | 0.86 | 0.22 | 0.68 | 0.21 |
| Grid + lags 2,4 | 0.85 | 0.22 | 0.66 | 0.22 |
| Grid, all 6144 | 0.80 | 0.26 | 0.55 | 0.26 |
| Grid + speed, sideways speed, yaw rate from telemetry | 0.87 | 0.21 | 0.72 | 0.20 |

Actions are in [-1, 1]; the teacher's steering has a standard deviation of 0.79.

- The grid features carry most of the teacher's steering decision (0.86 of a
  possible 0.97) and are clearly better than the class token (0.76). Earlier
  frames add nothing here.
- Linear read-out of the teacher's state from the grid features: sideways
  position of the road 0.6 to 3.0 units ahead R2 0.94-0.97, CTE 0.84, heading
  error 0.67, yaw rate 0.41, forward speed about 0. With lags: CTE 0.94, heading
  0.87, yaw rate 0.82, speed still 0.
- **The throttle part is not a fair test.** These frames come from v2 50k, which
  drives at 1.97 +/- 0.09; the teacher drives at 2.92 +/- 0.32 (2.3 to 3.4).
  Its braking decisions happen at speeds the frames never show, and speed
  cannot be judged readable or not when it barely varies. No frames of the
  teacher's own driving exist yet.
- An offline action error does not predict closed-loop driving. This only says
  imitation is not ruled out by missing information in the image features.

### Removing the ground-truth weave: `max_steering_diff` 0.2 -> 0.1 (2026-11-14)

- **Why `max_steering_diff=0.2` weaves on the start straight:** With `0.2` in the
  `[-1, 1]` action space (`|dsteer| <= 0.4` per 50 ms step), the `v1 50k` policy
  reaches full lock (`-1.0` or `+1.0`) in 3-4 steps on the high-speed start
  straight (`3.2-3.4 m/s`) in response to small `+0.1..+0.3` CTE offsets. Once at
  `-1.0` full left lock right after the start line (`idx 20..35`), unwinding to
  `+1.0` full right lock takes 5 steps (`0.25 s`), during which heading error
  overshoots to `-26.7..-27.5` degrees and carries the car across `CTE = -2.03`
  before the first right-hand bend (`idx 41..48`). By contrast, sustained cornering
  on the tightest bend (`idx 60..90`, radius 0.85) only requires `|dsteer| = 0.01..0.18`.
- **Single-variable test (`models/rl_loop_sim_state_ts2_sdiff01_v1`):** Trained
  `--encoder sim_state` for 60k steps at time scale 2, FPS limit 120 with
  `--max-steering-diff 0.1` (`|dsteer| <= 0.2` per step) and all other flags
  identical to `models/rl_loop_sim_state_ts2_v1`.
  - First 2000-step completion at 14k steps (against 35k in `v1`).
  - From step 20,730 to 60,730, completed **20 consecutive 2000-step training
    episodes (40,000 steps, ~185 laps) with 0 crashes**.
  - Screening `30k, 40k, 50k, 60k` (`logs/screen_sim_state_sdiff01_30k_60k`)
    selected `sac_sim_state_60000_steps.zip` (`mean_abs_cte = 0.163`,
    `max_abs_cte = 0.74`). One 50k episode in that screen hit a single 549 ms
    client-side pause (`elapsed_s` jumped by `0.549 s` while `env_step_seconds`
    was `0.3 ms`), during which the asynchronous simulator drove 2.63 m unsteered;
    no such pause occurred in the 24-layout evaluation below.
- **24-layout evaluation (`logs/sim_state_sdiff01_60k_24layouts`, `--capture-tail-steps 220`):**

| Ground-truth (`sim_state`) policy | Laps (95% CI) | Completed (95% Wilson CI) | Flying lap | Speed | Mean \|CTE\| | Max \|CTE\| | \|dsteer\| | Start-straight heading std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v1 50k` (`max_steering_diff=0.2`) | 9.42 (8.46-10.00) | 21/24 = 88% (69-96%) | 9.18 s | 2.75 | 0.30 | 2.04 | 0.26 | 9.5-10.9 deg |
| `sdiff01 60k` (`max_steering_diff=0.1`) | **9.00 (9.00-9.00)** | **24/24 = 100% (86-100%)** | **10.35 s** | **2.52** | **0.158** | **0.770** | **0.162** | **2.11 deg** |

- **What changed on the road (48,000 steps, 216 laps):**
  - Start-straight heading-error spread dropped from `9.5-10.9` degrees to
    **`2.11` degrees** (`5th..95th` percentile `-4.2..+2.6` degrees), matching the
    other straights (`2.08` degrees).
  - Entering the first bend (`idx 40..49`), CTE is `-0.62` median (`5th`
    percentile `-0.70`, worst `-0.77`), leaving at least **`1.23` CTE units of
    margin** on every pass. Across all 48,000 steps, `|CTE|` 99th percentile is
    `0.635` and maximum is `0.770`.
  - Giving up the wide outside-left swing trims flying lap speed from `9.18 s`
    (`10` laps/episode) to `10.35 s` (`9` laps/episode, `best_lap = 10.10 s`),
    which is still **23% faster than vision `v2 50k` (`13.47 s`)** while raising
    mean episode reward from `3278` to **`3553.2`**.

### Offline teacher-action & state probe with `sdiff01 60k` (2026-11-14)

1. **On the 150 layouts / 18,020 frames from the five §12 vision runs**
   (`logs/probe_teacher_action_sdiff01_60k_v2frames/probe_results.json`):
   - Teacher action std: steer `0.725`, throttle `0.652`.
   - With command history: `state` ceiling $R^2 = 0.956$ steer / $0.927$ throttle;
     `grid` (PCA 128) reaches $R^2 = 0.803$ steer / $0.749$ throttle (vs `cls`
     $0.710 / 0.731$); `grid + motion` (`speed, sideways_speed, yaw_rate`) reaches
     **$R^2 = 0.833$ steer (`MAE = 0.222`) / $0.767$ throttle (`MAE = 0.230`)**,
     and $R^2 = 0.743$ on frames with `|CTE| >= 1.0`.
   - Without command history (`grid+motion (no hist)`): steer $R^2 = 0.637$
     (vs `grid (no hist)` $0.192$), because under `max_steering_diff=0.1` the
     teacher's raw pre-clamp output depends on where the previous steering command
     `steer_{t-1}` was (`66.8%` of steps saturate the `0.1` step clamp). A student
      therefore needs at least the previous steering command (or should predict the
     post-clamp applied steering target).
2. **On the teacher's own 24 layouts / 5,184 frames**
   (`logs/probe_teacher_action_sdiff01_60k_self/probe_results.json`, captured via
   `--capture-tail-steps 220` so each episode retains more than a full 10.35 s lap):
   - Because speed now varies across every lap (`2.59 +/- 0.19 m/s` after standing
     start), linear ridge read-out of the teacher's state from `grid + lags` on
     held-out layouts reads:
     - `cte`: **$R^2 = 0.95$**, `sin heading error`: **$R^2 = 0.87$**,
       `sideways speed`: **$R^2 = 0.86$**, `yaw rate`: **$R^2 = 0.93$**,
       `forward speed`: **$R^2 = 0.59$** (was `0.00` on constant-speed `v2 50k`
       frames), and `road 0.3..3.0 ahead: right`: **$R^2 = 0.96 - 0.98$**.

**Next (Step 3):** Train a vision student policy from the stabilized
`sdiff01 60k` teacher (`models/rl_loop_sim_state_ts2_sdiff01_v1/sac_sim_state_60000_steps.zip`)
via teacher rollout collection + online DAgger in the simulator, using `grid`
(PCA 128) + motion telemetry (`speed, sideways_speed, yaw_rate`) + command history
(predicting either the post-clamp applied action or regularizing command history so
closed-loop driving stays responsive to visual features).

**Validation:** 32 pytest tests pass (track state, projection hashing, alignment verification, cache fingerprint, linear lag 15D splicing, and paired residual schedule contracts).
