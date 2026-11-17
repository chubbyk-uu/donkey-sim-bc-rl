# Geometry perception and control

> Historical record. Results, “current” choices and “next” plans below belong to the dated experiments, not the active work queue. See [current status](../../status.md) and [documentation index](../../README.md). Commands assume the repository root; raw `logs/` and `data/` artifacts are local-only unless explicitly noted.

### Stage 3.1: 15-dim geometry perception benchmark (offline, 2026-11-14)

`tools/probe_geometry_15d.py` (single-frame Round 1) and
`tools/probe_geometry_15d_lags.py` (multi-frame `lags=(2, 4)` Round 2), output
`logs/probe_geometry_15d_v1/` (verified via `feature_cache.meta.json` fingerprint
`b3d6ee0e9814`). Across all 174 `(run, layout)` groups (150 vision episodes from §12
+ 24 `sdiff01 60k` teacher episodes; 23,924 total frames, 23,204 valid lagged
frames), each model predicts the 15 geometric dimensions of `TrackState.encode`
(`[0]=cte/2`, `[1:3]=sin/cos(heading_error)`,
`[6:18]=6 lookaheads (right, forward)/lookahead`) under strict 5-fold `GroupKFold`
by `(run, layout)`. All feature standardizations, PCA projections, ridge tuning, and
MLP early stopping are fit strictly inside each training fold. Predicted geometry is
spliced into `[0:3]` and `[6:18]` alongside ground-truth motion `[3:6]`
(`vel[2]/2, vel[0]*4, gyro[1]/2`) and the 40-dim command history to evaluate
teacher-controller action substitution error (both raw pre-clamp and executed
post-clamp command error under `max_steering_diff=0.1`).

Units: CTE is in simulator CTE units (termination limit `+/-2.0`; `1 position unit`
is `~7.98 CTE units`), heading is in degrees (circular angle difference after
unit-circle normalization), and lookahead error is in simulator position units.

| Candidate (on 23,204 valid lagged frames) | CTE MAE / P95 (CTE) | CTE `w20` Bias P95 (CTE) | Heading MAE / P95 (deg) | Lookahead 1.0 MAE / P95 (pos) | ExecSteer MAE / P95 | ExecThrot MAE / P95 | `\|CTE\|>=1` CTE MAE / P95 (CTE) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cls_ridge (1f)` (`384 -> 15`) | 0.193 / 0.517 | 0.506 | 3.74 / 10.17 | 0.056 / 0.122 | 0.044 / 0.289 | 0.031 / 0.104 | 0.483 / 1.158 |
| `pca128_ridge (1f)` (fold-fit `PCA-128 -> 15`) | 0.159 / 0.425 | 0.434 | 3.12 / 8.36 | 0.040 / 0.089 | 0.037 / 0.258 | 0.021 / 0.079 | 0.397 / 0.966 |
| `raw6144_ridge (1f)` (`6144 -> 15`) | 0.090 / 0.243 | 0.244 | 2.17 / 5.94 | 0.026 / 0.054 | 0.023 / 0.163 | 0.013 / 0.049 | 0.205 / 0.550 |
| `raw6144_b64_mlp (1f)` (`6144 -> 64 -> 15`, 3-ens) | 0.087 / 0.243 | 0.238 | 1.95 / 5.60 | 0.026 / 0.054 | 0.022 / 0.160 | 0.012 / 0.047 | 0.224 / 0.561 |
| `pca128_lags_ridge` (`3x128=384 -> 15`) | 0.092 / 0.246 | 0.313 | 1.86 / 4.86 | 0.038 / 0.083 | 0.033 / 0.232 | 0.018 / 0.070 | 0.235 / 0.715 |
| `proj15_lags_ridge` (`3x15=45 -> 15` linear, 1 model) | 0.055 / 0.151 | **0.168** | **1.21 / 3.24** | **0.025 / 0.052** | 0.021 / 0.148 | 0.011 / 0.044 | 0.126 / 0.342 |
| `proj15_lags_mlp` (`3x15=45 -> 64 -> 15`, 3-ens) | 0.059 / 0.162 | 0.184 | 1.19 / 3.16 | 0.024 / 0.051 | 0.021 / 0.149 | 0.011 / 0.043 | 0.141 / 0.374 |
| `shared64_lags_mlp_1m` (single `6144->64, 192->64->15`) | 0.063 / 0.173 | 0.206 | 1.35 / 3.70 | 0.028 / 0.062 | 0.023 / 0.164 | 0.012 / 0.047 | 0.142 / 0.398 |
| `shared64_lags_mlp_ens3` (3-split inner ensemble) | **0.052 / 0.146** | 0.175 | 1.22 / 3.40 | 0.025 / 0.053 | **0.020 / 0.145** | **0.011 / 0.040** | **0.122 / 0.324** |
| `raw18432_lags_ridge` (`3x6144=18432 -> 15`) | 0.055 / 0.146 | 0.171 | 1.22 / 3.30 | 0.027 / 0.056 | 0.022 / 0.154 | 0.012 / 0.044 | 0.133 / 0.393 |

- **Unsupervised `PCA-128` drops ~44% of geometric precision:** Replacing fold-fit
  `PCA-128` with a supervised linear projection on the full `6144`-dim `4x4` patch
  grid cuts single-frame CTE P95 from `0.425` to `0.243 CTE` (`0.966 -> 0.550 CTE`
  on `|CTE| >= 1.0`) and heading P95 from `8.36 deg` to `5.94 deg`.
- **Single-model `proj15_lags_ridge` beats single-model `shared64_lags_mlp_1m` on
  every metric:** `shared64_lags_mlp_ens3` (`0.052 / 0.146 CTE`) averages 3 inner
  models per fold; a single `shared64_lags_mlp_1m` scores `0.063 / 0.173 CTE` (`w20`
  bias P95 `0.206 CTE`, heading P95 `3.70 deg`, `|CTE|>=1` P95 `0.398 CTE`). By
  contrast, the deterministic linear `proj15_lags_ridge` (`6144 -> 15` per frame,
  then `3x15=45 -> 15` across `lags=(2, 4)`) achieves **`0.055 / 0.151 CTE`**,
  **`w20` bias P95 `0.168 CTE`**, and **`Heading MAE/P95 = 1.21 / 3.24 deg`** as a
  single two-matrix affine transform, matching full `18,432`-dim Ridge without
  materializing an `18,432`-dim vector.
- **Exported deployable head:** `models/dinov2_geometry_15d/proj15_lags_ridge.pt`
  (loaded via `DinoGridLinearLag15Encoder` in `rl/geometry_encoder.py`).

### Stage 3.2: Empirical closed-loop tolerance screen with 1-step delay (2026-11-14)

`tools/eval_teacher_tolerance.py`, output
`logs/teacher_tolerance_screen_v1/tolerance_results.json`. Evaluated the frozen
`sdiff01 60k` teacher across 4 randomized light/tree layouts x 6 conditions in
shuffled order per layout (2000 steps max). Multi-step 15-dim residual blocks (length
80) are sampled from the held-out OOF predictions of Stage 3.1; heading residuals are
added as circular angles on the unit circle.

| Condition | Geom Delay | Residual Source | Comp | Mean Steps | Mean Laps | Mean \|CTE\| (CTE) | Max \|CTE\| (CTE) | Speed | Eff. CTE P95 (CTE) | Eff. Head P95 (deg) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 0 steps | None (exact `TrackState`) | **4/4** | 2000.0 | 9.00 | 0.159 | 0.743 | 2.53 | 0.000 | 0.00 deg |
| `delay1_only` | 1 step | None (1-step geom buffer) | **4/4** | 2000.0 | 9.00 | 0.236 | 1.691 | 2.50 | 0.181 | 4.25 deg |
| `shared64_lags_1x_d1` | 1 step | `shared64_lags_mlp_ens3` (`1.0x`) | **2/4** | 1289.2 | 5.50 | 0.270 | 2.227 | 2.37 | 0.258 | 5.96 deg |
| `raw6144_1f_1x_d1` | 1 step | `raw6144_ridge (1f)` (`1.0x`) | **0/4** | 1539.5 | 6.75 | 0.289 | 2.223 | 2.47 | 0.309 | 7.36 deg |
| `pca128_1f_1x_d1` | 1 step | `pca128_ridge (1f)` (`1.0x`) | **0/4** | 240.0 | 0.50 | 0.333 | 2.150 | 2.07 | 0.497 | 9.87 deg |
| `pca128_1f_1.5x_d1` | 1 step | `pca128_ridge (1f)` (`1.5x`) | **0/4** | 224.5 | 0.25 | 0.381 | 2.253 | 2.02 | 0.616 | 12.55 deg |

- **What this 4-layout screen shows:**
  - A 1-step geometry buffer delay alone (`delay1_only`) introduces `0.181 CTE` P95
    and `4.25 deg` heading P95 phase lag in corners, raising `max_abs_cte` from
    `0.743` to `1.691 CTE` (still 4/4 completed).
  - Stacking 1-step delay with `1.0x` held-out residual blocks stratifies the
    representations (`pca128_1f_1x_d1`: 240 mean steps, 0/4; `raw6144_1f_1x_d1`:
    1539.5 mean steps, 0/4; `shared64_lags_1x_d1`: 1289.2 mean steps, 2/4).
- **Important limitations (why this screen does not establish a "safe zone"):**
  1. Only 4 layouts were tested, and all residual conditions included `1-step` extra
     geometry delay (`_d1`); no `0-extra-delay` (`_d0`) residual control or live
     visual closed-loop was included in this run.
  2. Conditions in `logs/teacher_tolerance_screen_v1` used per-condition RNG seeds
     (`layout_idx * 1000 + cond_idx`) and replayed residual blocks across unrelated
     track positions. `tools/eval_teacher_tolerance.py` has now been updated to use a
     single pre-sampled `(block_id, offset)` schedule per layout across all
     conditions, incremental per-episode/per-step saving (`telemetry.jsonl`,
     `episodes.jsonl`, `metadata.json`), `proj15_lags_1x_d0` (0 extra delay), and
     live closed-loop evaluation of `models/dinov2_geometry_15d/proj15_lags_ridge.pt`
     (`vision_proj15_lags_live`).

### Stage 3.3: Paired `d0` vs `d1` + live vision 4-layout pilot (2026-11-15)

`tools/eval_teacher_tolerance.py --mode vision_and_d0 --layouts 4`, output
`logs/vision_proj15_and_d0_v1/tolerance_results.json` (using shared per-layout residual
block schedules across conditions).

| Condition | Geom Delay | Residual / Perception | Comp | Mean Steps | Mean Laps | Mean \|CTE\| (CTE) | Max \|CTE\| (CTE) | Speed | Ep-Avg Eff. CTE P95 | Ep-Avg Eff. Head P95 | Enc ms P50 / P95 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 0 steps | Exact `TrackState` | **4/4** | 2000.0 | 9.00 | 0.157 | 1.155 | 2.52 | 0.000 | 0.00 deg | 0.10 / 0.13 |
| `proj15_lags_1x_d0` | 0 steps | OOF `proj15_lags_ridge` | **4/4** | 2000.0 | 9.00 | 0.188 | 1.176 | 2.54 | 0.137 | 3.01 deg | 0.10 / 0.14 |
| `proj15_lags_1x_d1` | 1 step | OOF `proj15_lags_ridge` | **1/4** | 1062.0 | 4.25 | 0.251 | 2.135 | 2.40 | 0.255 | 5.56 deg | 0.10 / 0.14 |
| `vision_proj15_lags_live` | 0 steps | Live DINOv2 + `proj15_lags_ridge` | **4/4** | 2000.0 | 9.00 | 0.181 | 1.428 | 2.55 | 0.135 | 2.80 deg | 4.11 / 4.69 |

- **Zero extra delay rescues closed-loop stability (`d1` `1/4` -> `d0` `4/4`):** Removing
  the synthetic 1-step delay cuts effective CTE P95 from `0.255` to `0.137 CTE` and
  eliminates 3/4 crashes.
- **Live vision tracks `proj15_lags_1x_d0` closely on 4 layouts:** `4/4` completed (36
  laps total), speed `2.55 m/s` vs baseline `2.52 m/s`, though `mean|CTE|` is `+15.7%`
  higher (`0.181` vs `0.157 CTE`) and Layout 2 peaked at `-1.428 CTE` at step 372
  (`eff_cte_err = +0.250 CTE` underestimate following 9 consecutive `max_steering_diff=0.1`
  rate-limited left-turn steps at `2.98 m/s`).

### Stage 3.4: Round A — 24-layout paired live vision eval & pre-registered split (2026-11-15)

`tools/eval_teacher_tolerance.py --mode round_a_vision_paired --layouts 24 --train-pool-layouts 12 --save-captures --output-dir logs/round_a_vision_24layouts_v1`

- **Frozen model:** `models/dinov2_geometry_15d/proj15_lags_ridge.pt` +
  `models/rl_loop_sim_state_ts2_sdiff01_v1/sac_sim_state_60000_steps.zip`.
- **Pre-registered split (`split_manifest.json`):** Layouts `1..12` = `dagger_train_pool`,
  Layouts `13..24` = `locked_eval_holdout` (reserved exclusively for final evaluation,
  never used for retraining).
- **Full captures saved:** `captures/layout_*.npz` storing full-episode RGB frames,
  15-dim ground-truth/predicted geometry, 3-dim motion, 40-dim command history,
  student actions, counterfactual `teacher(true_state)` actions, and monotonic control-chain
  timestamps (`decision_obs_rx_mono_s` -> `act_sent_mono_s`, `step_dt_ms`).

| Scope | Condition | Comp | Mean Steps | Mean Laps | Mean \|CTE\| (CTE) | Max \|CTE\| (CTE) | `\|CTE\|>=1.0` Steps | Speed | Pooled CTE MAE / P95 / P99 (CTE) | Pooled Head P95 / P99 (deg) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **All 24 layouts** (`47,297` steps) | `baseline` | **23/24** (`95.8%`) | 1970.7 | 8.875 | 0.1585 | 2.086 | 6 | 2.522 | `0.000 / 0.000 / 0.000` | `0.00 / 0.00` |
| **All 24 layouts** (`46,153` steps) | `vision_proj15_lags_live` | **23/24** (`95.8%`) | 1923.0 | 8.625 | 0.1979 | 2.061 | 53 | 2.512 | `0.054 / 0.140 / 0.194` | `3.08 / 4.44` |
| `dagger_train_pool` (L1..12) | `baseline` / `vision` | `11/12` vs `11/12` | 1941.4 vs 1846.1 | 8.75 vs 8.25 | 0.1584 vs 0.2065 | 2.086 vs 2.061 | 6 vs 45 | 2.523 vs 2.482 | `0.055 / 0.140 / 0.199` | `3.18 / 4.69` |
| `locked_eval_holdout` (L13..24) | `baseline` / `vision` | **`12/12` vs `12/12`** | **2000.0 vs 2000.0** | **9.00 vs 9.00** | 0.1585 vs 0.1893 | 1.119 vs 1.300 | **0 vs 8** | 2.520 vs 2.541 | `0.054 / 0.141 / 0.190` | `3.01 / 4.18` |

- **Paired delta (`vision - baseline`, all 24 layouts):** `mean_laps_difference = -0.25`
  (`[-0.75, 0.00]`, `sign_test_p = 1.0`, `only_a = 0, only_b = 0`);
  `mean_abs_cte_diff = +0.0395 CTE` (`[+0.0306, +0.0502]`, `1.249x` ratio);
  `max_abs_cte_diff = +0.2401 CTE` (`[+0.1422, +0.3560]`). On `locked_eval_holdout`,
  `vision` achieved `12/12` (`mean_laps_diff = 0.0`), `mean|CTE| diff = +0.0308 CTE`
  (`1.194x` ratio), and `mean_speed_diff = +0.0207 m/s`.
- **Timing over 46,153 live vision steps:** `encoder_ms` P50/P95/P99/max =
  `4.17 / 4.67 / 5.54 / 81.89 ms` (only 1 spike > 12.25 ms across 46,153 steps, at L06
  step 1021); `policy_ms` P50/P95 = `0.51 / 0.64 ms`; full `decision_obs_to_act_ms`
  P50/P95/P99/max = `4.70 / 5.20 / 6.13 / 82.50 ms` (well within the `25.15 ms` P50
  simulator `step_dt_ms`).
- **Regime breakdown & root gap motivating Round B (`vision_proj15_lags_live`):**
  - `standing_start_1_250` (`5,903` steps): `mean|CTE| = 0.175` (baseline `0.151`),
    `eff_cte MAE/P95 = 0.049 / 0.127 CTE`, `eff_head P95 = 3.06 deg`.
  - `cruise_abs_cte_lt_0_5` (`43,520` steps): `eff_cte MAE/P95 = 0.053 / 0.138 CTE`,
    `eff_head P95 = 3.04 deg`.
  - `moderate_offset_0_5_to_1_0` (`2,580` steps): `eff_cte MAE/P95 = 0.065 / 0.172 CTE`,
    `eff_head P95 = 3.60 deg`.
  - **`large_offset_ge_1_0` (`53` steps, 0.11% of steps):** perception error jumps **4.5x**
    to `eff_cte MAE/P95/max = 0.243 / 0.622 / 0.873 CTE` (`eff_head P95 = 6.01 deg`),
    with strong center-shrinkage underestimation (`+cte`: `-0.312 CTE` mean error;
    `-cte`: `+0.126 CTE`). Worst non-fatal excursion: Layout 3 step 375 (true `+1.737`,
    predicted `+0.864 CTE`).
- **Layout 7 failure analysis:** Both `baseline` (step 1297, `cte=-2.086`) and
  `vision` (step 153, `cte=+2.061`) failed only on Layout 7 (`dagger_train_pool`).
  In `layout_07_vision` steps 145–153, counterfactual `teacher(true_state)` already
  commanded `-1.000` full left alongside the student after crossing `cte = 0` at
  `+25 deg` heading (`max_steering_diff=0.1` saturated during full reversal), confirming
  that Layout 7 contained tight curves near the controller's physical steering-rate limit
  once `|CTE|` swung above `1.0`.

**Next (Round B):** Collect targeted off-center / recovery supervision on fresh training
layouts (`dagger_train_pool` only; never `locked_eval_holdout`), combine with Round A
train-pool frames to enrich `|CTE| in [0.5, 1.9]`, retrain/compare the geometry perception
head, and verify recovery + lateral tracking gains on `locked_eval_holdout`.

### Round A Layout 7 Root-Cause Diagnosis & Minimal Control-Loop Fixes

#### 1. Root-Cause Evidence (Teacher vs. Vision on Layout 7)
Deep inspection of `logs/round_a_vision_24layouts_v1/telemetry.jsonl` and `gym_donkeycar`
(`donkey_sim.py`, `client.py`) showed that `layout_07_baseline` (`step 1297`, `cte=-2.086`,
Lap 6) and `layout_07_vision_proj15_lags_live` (`step 153`, `cte=+2.061`, Lap 0) failed at
different times and trajectories from **distinct upstream triggers** feeding into a **shared
downstream rate-limit reversal bottleneck**, rather than an unavoidable corner physics limit:

- **`layout_07_baseline` (Step 1297, Lap 6) — Control-Cycle Stall + TCP Burst Replay +
  Persistent `dt` Beating -> Rate-Limited Reversal Trap:**
  - Across Laps 0–4 (`steps 249..268, 456..475, 664..683, 868..887, 1074..1093`) on the
    exact same corner (`pos=(6.15..6.19, 4.30..6.90)`), `baseline` tracked cleanly with
    `step_dt_ms in [23.48, 26.81] ms` (`std = 0.53..0.88 ms`), mild steering (`[-0.34, +0.29]`),
    and `max|CTE| <= 0.311` across all 5 laps — ruling out a physical corner limit.
  - Only `layout_07_baseline` among all 12 `dagger_train_pool` baseline episodes exhibited
    elevated cycle jitter (`bad_dt = 22` steps outside `[18, 32] ms`, `dt_std = 2.70 ms` vs
    `0.73..0.87 ms` in the 11 surviving layouts).
  - At `step 1263`, `step_dt_ms` spiked to `93.42 ms` (`sim_wait_ms = 92.69 ms`), followed at
    `steps 1264..1266` by 3 buffered telemetry packets draining in `5.58, 4.40, 2.87 ms`.
    Because `dp / (speed * 0.05)` stayed `0.99..1.00`, Unity had stepped `0.05 s` sim-time per
    packet while `SDClient` (`time.sleep(0.001)` + per-frame loop) replayed all 3 stale buffered
    frames back-to-back, advancing `last_steer` and `command_history` on stale states. From
    `step 1267..1297`, `step_dt_ms` remained trapped in `11.80..36.51 ms` beating
    (`dt_std = 14.81 ms`), amplifying lateral swing (`+0.227 -> -0.146 -> -0.307 -> +0.079`).
  - At `step 1286` (`cte=+0.069`), `executed steer` peaked at `-0.72` (left). Over `steps
    1287..1290`, as `cte` crossed `-0.011 -> -0.449` and heading flipped `-2.9° -> -17.4°`,
    raw policy demanded `+0.10 -> -0.20 -> +0.49 -> +0.85`, **but `max_steering_diff=0.1`
    (`0.20/step`) held executed steering negative (`-0.52, -0.32, -0.12`) for 3 steps**
    (two with `step_dt_ms = 35.40, 32.11 ms`), taking 9 steps (`1286..1295`) to go
    `-0.72 -> +0.99` and exiting at `-2.086`.
- **`layout_07_vision_proj15_lags_live` (Step 153, Lap 0) — Visual Oscillation + Underestimate
  -> Same Downstream Reversal Lag:**
  - Had an early timing disturbance (`steps 11..18`), then `105` consecutive normal-timing steps
    (`steps 49..153`, `step_dt_ms in [23.44, 26.49] ms`) before crashing.
  - Its excursion arose from Lap-0 curve tracking oscillation (`+0.707` at `s=121` -> `-1.098`
    at `s=144` with `+0.117 CTE` visual underestimate at `s=143`), followed by the same
    downstream `max_steering_diff=0.1` reversal bottleneck (`exe = +0.66 -> +0.46 -> +0.26 ->
    +0.06` across `s=144..146` while raw policy commanded `-0.46, -0.91, -1.00`, overshooting
    zero at `+25.1°` heading to `+2.061`).
- **`gym_donkeycar` race & polling bugs found (`donkey_sim.py` / `client.py`):**
  1. `on_telemetry` assigns `self.time_received = time.time()` at line 508 **before** updating
     `pos / speed / vel / yaw / cte / image_array` (torn-read race).
  2. `observe()` and `SDClient.proc_msg()` spin on `time.sleep(0.001)` polling and replay all
     stale queued `"telemetry"` frames inside a burst instead of coalescing TCP bursts.

#### 2. Minimal Fixes Implemented (Teacher & Vision Head Weights Frozen)
- **Fix A — Event-Driven, Torn-Read-Free, Burst-Coalescing Sync (`rl/donkey_sync.py`):**
  `LowJitterSyncController` (toggleable via `set_enabled(bool)` for paired A/B eval):
  - Updates `image_array` and all state fields atomically under `threading.Condition` before
    incrementing `_telemetry_seq` and notifying `observe()` (zero `1 ms` sleep polling, zero torn reads).
  - Uses non-blocking socket + pipe wakeup fd (`os.pipe()`) for sub-millisecond `control` sends.
  - Within each TCP `recv()` burst, preserves `"hit"` / `"over"` / non-telemetry events from
    superseded packets via `coalesce_telemetry_messages()` while decoding and waking `observe()`
    only on the latest telemetry frame (`coalesced_dropped`, `max_burst_seen`).
- **Fix B — Gated Asymmetric Opposite-Sign Unwind-to-Zero (`rl/train_vae_sac.py`):**
  Added configurable `DonkeyVaeSACEnv` parameters (`reversal_unwind_scale=2.0`,
  `reversal_cte_threshold=0.40`, `reversal_head_deg_threshold=8.0`, `reversal_min_last_steer=0.30`):
  - Uses **only observation geometry** (`last_pred_geom` for vision or telemetry `[0:3]` for
    teacher — zero privileged ground-truth leakage for vision).
  - Activates **only** when `(steer * last_steer < 0)`, `|last_steer| >= 0.30`, and
    `(|cte_obs| >= 0.40 or |head_obs| >= 8.0°)`.
  - Computes `eff_max_diff = max(max_diff, min(scale * max_diff, abs(last_steer)))`:
    speeds return **only toward `0.0`** without skipping past `0.0` in a single step, leaving
    centered driving (`98%+` of steps), same-sign turning, and small-angle dithering (`<0.30`)
    untouched (`100%` identical to baseline). Notice an initial `v1` variant without
    `reversal_min_last_steer=0.30` and with `|last_steer| + max_diff` caused `±0.40`
    zero-crossing ping-pong under stress (`logs/teacher_fix_recovery_stress_v1`), which `v2`
    eliminated (`logs/teacher_fix_recovery_stress_v2`).

#### 3. Paired Closed-Loop Verification (`useSeed=0` Fresh Random Layouts; Round A Holdout Untouched)
**Caveat (`useSeed=0`):** With `useSeed=0`, Unity generates fresh random track geometry per reload;
new evaluations cannot replay the exact Round A Layout 7 geometry, so paired comparisons evaluate all
conditions on the same freshly loaded layouts per run. Unit tests: `pytest -q` -> **38 passed**
(including `tests/test_donkey_sync_and_ctrl.py`).

**A. Paired Natural Evaluation (`--mode teacher_fix_paired --layouts 12`, 48 episodes, `logs/teacher_fix_paired_v2`):**

| Condition | Comp | Mean Steps | Mean Laps | Mean \|CTE\| (CTE) | P95 / Max \|CTE\| | Speed | `step_dt_ms` P50/P95 | `bad_dt` Total / 11-ep `dt_std` (excl. L03 freeze) | `coal` / `unwind` | Excursions Recovered |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` (stock sync, no unwind) | 12/12 | 2000.0 | 9.00 | 0.1599 | 0.464 / 0.858 | 2.526 | 24.75 / 25.89 | 16 / 0.845 ms | 0 / 0 | 1/1 (med 5.0 st) |
| `teacher_sync_only` (Fix A only) | 12/12 | 2000.0 | 9.00 | 0.1586 | 0.460 / 1.437 | 2.522 | 25.00 / 25.69 | 14 / 0.636 ms | 2 / 0 | 6/6 (med 7.5 st) |
| `teacher_fixed` (Fix A + Fix B v2) | **12/12** | **2000.0** | **9.00** | 0.1643 | 0.482 / 1.425 | 2.520 | **25.00 / 25.69** | 86 (17 excl. L03)* / **0.540 ms** | 2 / 220 | **9/9** (med **5.0 st**) |
| `vision_fixed_ctrl` (frozen head + A + B v2) | 11/12 | 1950.0 | 8.75 | 0.1851 | 0.497 / 2.153 | 2.531 | 25.00 / 25.69 | 61 / 1.139 ms | 2 / 277 | 23/24 (med 6.0 st) |

- \*Note: `teacher_fixed` on `L03` experienced an external `2295.8 ms` (`2.3 s`) stall (`step 1709`)
  and 50-step host jitter storm (`bad_dt=69`, `coal=2, burst=3`), recovered **8/8** excursions,
  and still completed all 2000 steps (`max|CTE|=1.425`). Across the other 11 layouts, Fix A + Fix B
  cut `step_dt_ms` episode `std` by **36%** (`0.845 -> 0.540 ms`) and locked P50 cleanly onto `25.00 ms`.
- **GPU contention sensitivity finding (`logs/teacher_fix_paired_v1`):** In an earlier `v1` run
  when the Windows host GPU (RTX 5080) was at `82–94%` load (`234–255 W`), `vision_fixed_ctrl`
  DINOv2 `encoder_ms` P50/P95 degraded 3–5x (`15.39 / 24.51 ms`, up to `33 ms`), pushing
  `obs_to_act > 25 ms` step period and causing `3/12` completion (`teacher_fixed` still went `12/12`,
  `max|CTE| = 1.024` vs `baseline` `1.465` and `teacher_sync_only` `11/12`). Once host GPU load
  returned to idle (`v2`), `vision_fixed_ctrl` ran at `enc P50/P95 = 4.33 / 5.33 ms`, improving
  `mean|CTE|` from `0.1979` (Round A) to `0.1851` (`11/12` PASS).
- **Remaining vision bottleneck:** The single `vision_fixed_ctrl` failure (`L04 step 1400`,
  `cte=-2.153`) still showed large-offset visual underestimation (`err=+0.238` at `cte=-1.170`
  growing to `+0.513` at `-2.153`), confirming Round B perception retraining remains necessary.

**B. Paired Recovery Stress Evaluation (`--mode teacher_fix_recovery_stress --layouts 8`, 24 episodes, `logs/teacher_fix_recovery_stress_v2`):**
Deterministic paired disturbance schedule on every layout (`Kick 1`: steps `320..322` `±0.70` steer;
`95 ms` TCP stall: step `900`; `Kick 2`: steps `1300..1302` `∓0.70` steer). Note `nvidia-smi` was at
`~94%` host GPU load during this run (`vision_fixed_ctrl` `enc P50/P95 = 19.36 / 30.26 ms`):

| Condition | Comp | Mean Steps | Mean Laps | Mean \|CTE\| | Speed | Post-Stall (step 900) Deaths | Excursions Recovered |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 0/8 | 779.2 | 3.12 | 0.1995 | 2.350 | **4 / 5** reaching s=900 | `15/23` (`65.2%`) |
| `teacher_fixed` | **3/8** | **1199.4** (+420.1) | **5.25** (+2.12) | **0.1973** | **2.394** | **0 / 5** reaching s=900 | **`23/28` (`82.1%`)** |
| `vision_fixed_ctrl` *(host GPU 94%)* | 0/8 | 369.2 | 1.00 | 0.3618 | 2.170 | — | `56/64` (`87.5%`) |

Under identical disturbances, `teacher_fixed` survived Kick 1, the `95 ms` stall, and Kick 2
completely on 3 layouts (`L01`, `L03`, `L06`), eliminated all post-stall deaths (`0/5` vs `4/5` in
`baseline`), and raised overall excursion recovery from `65.2%` to `82.1%` without harming steady-state
tracking or speed.

#### 4. Audit Follow-Up: Non-Blocking Send Fix, Real TCP Backlog Smoke Test & Entrypoint Wiring
Addressing post-`v2` review findings before full 24-layout / Round B runs:
1. **Non-blocking socket send correctness (`rl/donkey_sync.py`):** Replaced `sock.sendall()`
   inside `_proc_msg_loop` with `sock.send(memoryview(_inflight_bytes)[_inflight_offset:])`, tracking
   unsent payload bytes and byte offset under `_send_lock` (across both `enabled=True` and
   `enabled=False`). Verified in `tests/test_donkey_sync_and_ctrl.py` (`42 passed` in `pytest -q`)
   under chunked partial sends + `BlockingIOError`, mid-flight second-message queueing, real kernel
   `SO_SNDBUF`/`SO_RCVBUF` backpressure, and `pause_recv` coalescing.
2. **Control stall vs. real TCP receive-backlog stress (`tools/eval_teacher_tolerance.py`):**
   Clarified that the earlier `v2` step-900 stall (`time.sleep(0.095)`) paused only the control thread
   (`coalesced_dropped = 0`), leaving the socket reader active. Added `--stress-type {ctrl_stall,tcp_backlog,both}`,
   per-episode `sync_controller.reset_stats()`, `max_burst_seen` tracking for `enabled=False`, and
   `pause_recv(0.095)`. In a 1-layout live simulator smoke test (`logs/smoke_tcp_backlog_v1`,
   `1050` steps, GPU idle `5%`, `enc P95 = 4.84–5.15 ms`), `pause_recv(0.095)` produced genuine
   TCP bursts (`max_burst_seen = 3..4`): sync-enabled conditions coalesced and dropped `2–3` stale
   frames (`coalesced_dropped = 2..3`) whereas stock `baseline` had `coalesced_dropped = 0` and
   replayed stale buffered frames in `1.28 ms` / `1.77 ms` bursts (`steps 901..902`).
3. **Complete within-run paired ablations & opt-in entrypoint integration:**
   - Added `--mode teacher_fix_paired_full` and upgraded `teacher_fix_recovery_stress` to run all
     5 conditions side-by-side (`baseline`, `teacher_sync_only`, `teacher_fixed`, `vision_baseline`,
     `vision_fixed_ctrl`) on identical tracks.
   - Wired opt-in control flags (`--low-jitter-sync` default `False`, `--reversal-unwind-scale`
     default `1.0`) via shared `add_control_fix_args` / `configure_env_control_fixes` helpers into
     `rl/train_vae_sac.py`, `rl/train_loop_vae_sac.py`, `rl/eval_vae_sac.py`,
     `rl/eval_loop_vae_sac.py`, and `rl/eval_paired_randomized.py`, preserving `environment_config()`
     and `validate_model_config()` compatibility for existing checkpoints.
   - **Deferred to next session (paused by user):** Running `--mode teacher_fix_paired_full --layouts 24`
     and 12-layout 5-condition stress evaluations under idle GPU before moving to Round B.
