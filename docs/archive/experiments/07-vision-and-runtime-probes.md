# Vision and runtime probes

> Historical record. Results, “current” choices and “next” plans below belong to the dated experiments, not the active work queue. See [current status](../../status.md) and [documentation index](../../README.md). Commands assume the repository root; raw `logs/` and `data/` artifacts are local-only unless explicitly noted.

## 11. Runtime revalidation and control diagnostics (2026-11-12)

All numbers below were measured locally; no policy was retrained. Simulator
settings, source snapshots, hashes, telemetry and plots are local-only artifacts
under `logs/`. The runs do not share reproducible Unity layout seeds.

### Baseline rechecks and recording infrastructure

- WSL2 NAT to the Windows simulator on TCP 9091 works; RTX 5080 CUDA inference
  works. No network-mode or firewall changes were needed.
- Fixed-light VAE `v3_h80` 90k: 5/5 at 3000 steps, 76 laps, mean speed 2.826,
  mean absolute CTE 0.311, mean lap 9.40s. This repeats one fixed scene, not a
  generalization test. Artifact: `logs/baseline_fixedlight_20261112_093541/`.
- DINOv2 randomtree v2 50k: 8/12 fresh random-light/tree layouts at 2000 steps,
  68 laps, layout-mean absolute CTE 0.397, mean lap 13.84s. Artifact:
  `logs/randomtree_baseline_12layouts_v2_50k/`. This small batch does not establish
  an improvement over historical batches on different random layouts.
- Evaluation now records JSON/CSV, exact encoder-input snapshots and optional
  telemetry/tail clips. Loop resume saves effective settings and checks replay
  configuration/hash; changed rewards require a fresh buffer. Historical
  checkpoints remain usable with an explicit missing-metadata warning. VAE
  resume preserves historical best validation loss. See README for CLI details.

### Repeated-layout diagnostic protocol

Same DINOv2 checkpoint twice per layout (`repeat_a`, `repeat_b`), shuffled order,
6 fresh layouts, 2000-step cap, CTE limit 2.0, steering-difference parameter 0.2.
Random light and trees on; random cones off. No scene reload between the two
repeats. Tail capture retains 120 returned encoder-input frames per episode;
image writing happens after the episode. These are repeats of one policy,
not competing models.

| Layout | repeat_a | repeat_b |
|---|---:|---:|
| 1 | 2000 / limit | 2000 / limit |
| 2 | 2000 / limit | 2000 / limit |
| 3 | 2000 / limit | 911 / CTE |
| 4 | 2000 / limit | 2000 / limit |
| 5 | 1502 / CTE | 964 / CTE |
| 6 | 2000 / limit | 2000 / limit |

9/12 episodes reach the cap, 74 laps total; four layouts succeed twice, one is
mixed, one fails twice. Do not treat this as twelve independent-layout trials or
rank `repeat_a` above `repeat_b`: their checkpoint hashes are identical.
Artifact: `logs/randomtree_control_repeat6_tail120/` (21,377 steps, 1,440 PNG tail
frames, 12 GIFs, plots/contact sheets under `control_analysis/`).

### Observed evidence and limits

- All three captured failures cross the negative CTE boundary, with no collision
  telemetry. The visible road/lane lines persist in the terminal images; CTE
  termination is not equivalent to leaving the whole paved road. Both layout-5
  failures occur at the same corner (terminal positions about 0.17 Unity units
  apart), but at different laps/steps. Layout 3 succeeds once and fails once.
- Tail P95 policy times are about 0.81–0.86ms, encoder 4.7–4.94ms, env step
  50.5–51.1ms across failures. No failure-adjacent client stall is evident. Env
  step includes telemetry waiting and encoding: these times do **not** measure
  network RTT, constant actuation delay or Unity acknowledgement latency.
- The mapped command often approaches +1 during the final departure. It is a
  command submitted to the client, not a measured wheel angle. The asynchronous
  simulator does not acknowledge per-command execution; frame k is returned by
  `env.step(command k)` and used for the next decision.
- Offline replay using saved frames plus the previous 20 mapped commands exactly
  reproduces all 15 sampled next actions (max absolute difference 0). This
  validates camera/history/policy alignment for those samples, not simulator
  command-to-frame causality. Script: `tools/verify_policy_alignment.py`
  (originally run from a copy inside the artifact directory; the tool reproduces
  the same numbers); output `alignment_check.json` in that directory.
- A limited offline sensitivity probe freezes command history and applies RGB
  gains 0.9/1.1 to those 15 frames. Across 30 perturbations, median absolute steer
  change is 0.0034, maximum 0.536, and no sign flips occur when the original
  magnitude exceeds 0.2. This is neither a closed-loop result nor a test of local
  tree-shadow invariance. Script: `tools/verify_policy_alignment.py
  --brightness-gains 0.9 1.1`; output `brightness_probe.json` in the same directory.

**Interpretation:** current evidence does not establish perception as the sole
bottleneck, and does not establish control as the sole cause either. Keep the
segmentation/fine-tuning path as a candidate. Prioritize a probe of whether frozen
features predict lane deviation/heading, plus a controlled state/speed-input
comparison, before committing to a heavier encoder training branch. A road-only
mask should not be assumed to preserve the right-lane target or yellow line.
The feature probe is done in §12.

**Validation:** 17 pytest tests pass, including bounded capture, image-copy and
index alignment, context isolation, exception capture, and the earlier resume
regressions. Live Unity capture and offline plot generation also pass. The
simulator remains in random-light/tree mode after these experiments.

## 12. Laps baseline with intervals, state probe, simulator speed-up (2026-11-12)

No policy was retrained. All runs use DINOv2 randomtree v2 50k
(`models/rl_loop_dinov2_randomtree_v2/sac_dinov2_vits14_50000_steps.zip`, crop 0),
random light and trees on, random cones off, 2000-step cap, CTE limit 2.0, a
fresh layout per episode. Artifacts are local-only under `logs/`.

### Primary metric

From here on the primary metric is **laps per episode** on fresh random
light/tree layouts with a fixed step cap, where leaving the track ends the
episode. It rewards speed and staying on the road together. Models are compared
on the same layouts (`rl/eval_paired_randomized.py`); the summary reports mean
laps with a bootstrap 95% interval, a Wilson interval on the completion rate, and
paired tests (`paired_statistics` in `rl/experiment.py`;
`tools/summarize_paired_eval.py` recomputes them for old or aborted runs). Do not
relax throttle range, CTE limit or step cap to raise the number.

### Baseline at four simulator settings

| Simulator setting | Layouts | Laps (95% CI) | Completed (95% CI) | Steps/s incl. reloads | Step interval median (p5-p95) |
|---|---:|---:|---:|---:|---:|
| Time scale 1, FPS limit 60 | 40 | 5.53 (4.72-6.22) | 27/40 = 68% (52-80%) | 18.8 | 50.0 ms (48.0-51.9) |
| Time scale 2, FPS limit 60 | 24 | 5.04 (4.04-5.96) | 13/24 = 54% (35-72%) | 35.0 | 20.1 ms (15.3-34.7) |
| Time scale 2, FPS limit 120 | 40 | 6.08 (5.42-6.62) | 31/40 = 78% (62-88%) | 35.7 | 25.1 ms (23.8-26.3) |
| Time scale 3, FPS limit 180 | 40 | 4.42 (3.50-5.33) | 22/40 = 55% (40-69%) | 48.2 | 16.7 ms (14.7-18.5) |

Artifacts: `logs/randomtree_baseline_40layouts_v2_50k/`,
`logs/randomtree_timescale2_24layouts_v2_50k/`,
`logs/randomtree_timescale2_fps120_40layouts_v2_50k/`,
`logs/randomtree_timescale3_fps180_40layouts_v2_50k/`. The four runs use
different random layouts (unpaired).

- The earlier 8/12 (§11) has a Wilson interval of 39-86%; twelve layouts cannot
  rank models.
- The same checkpoint scores 68%, 54% and 78% in the first three runs. Single-run absolute
  rates on 24-40 layouts are not comparable across runs; compare models paired on
  the same layouts.
- Time scale 2 with FPS limit 120 versus time scale 1: laps difference +0.55
  (95% CI -0.40 to +1.53, Mann-Whitney p=0.29). Lap time (13.43 s vs 13.47 s,
  first lap excluded), mean speed (1.925 vs 1.923), mean |steer change| (0.305
  both) and CTE spread are unchanged. Maximum |CTE| in completed episodes is
  slightly higher (+0.09, CI -0.01 to +0.20, p=0.057).
- Time scale 2 with FPS limit 60 versus time scale 1: laps -0.48 (CI -1.68 to
  +0.72), maximum |CTE| in completed episodes +0.14 (CI +0.00 to +0.27). Not
  significant, but every safety metric moves the wrong way.
- Time scale 3 with FPS limit 180 is **worse**, although its step intervals are
  steady: laps -1.65 against time scale 2 / FPS 120 (CI -2.73 to -0.58, p=0.009)
  and -1.38 against the first and third rows pooled (CI -2.40 to -0.34,
  p=0.012); lap time +0.18 s against time scale 1 (CI +0.07 to +0.27); 7 of the
  18 failures happen within the first 300 steps. The client's roughly 5 ms of
  encoder and policy time is about 15 ms of simulated time at this scale (5 ms
  at time scale 1), which is the likely cause; that this checkpoint loses more
  than a lap to it is consistent with the thin control margin described below.

### Can the car's position be read from the frozen features?

`tools/probe_state_from_features.py` fits ridge regressions from encoder features
to simulator ground truth and predicts only layouts the regression never saw
(grouped 5-fold; penalty chosen inside each training split). Data: the 120-frame
tails of the 40-layout time-scale-1 baseline and of the §11 repeat run, 6240
frames, 46 layouts, 16 failed episodes. Output: `logs/probe_state_v2_50k/`.

| Features | CTE R2 | CTE MAE | Lateral drift R2 |
|---|---:|---:|---:|
| Frame shrunk to 20x15 RGB (reference) | 0.73 | 0.21 | 0.47 |
| DINOv2 class token (the policy input) | 0.73 | 0.21 | 0.47 |
| Mean of patch tokens | 0.74 | 0.21 | 0.48 |
| Patch tokens pooled to a 4x4 grid | 0.93 | 0.11 | 0.59 |

CTE standard deviation in this data is 0.54. "Lateral drift" is the slope of CTE
against distance travelled over the last 5 steps, a stand-in for heading error
(telemetry has no track direction).

- The class token carries coarse lane position: MAE 0.21 against a limit of 2.0,
  and the correct side of the road in 99% of the 406 frames with |CTE| >= 1.
- It is no better than a 20x15 thumbnail. Keeping the spatial layout of the patch
  tokens halves the error.
- At the same distance from the centre line, class-token error is similar in
  completed and failed episodes (MAE 0.18 vs 0.19 for |CTE| < 0.5, 0.22 vs 0.25
  for 0.5-1, 0.33 vs 0.40 for 1-1.5). No sudden loss of the lane is visible in
  failed episodes. Beyond 1.5 (80 frames, failures only) the class-token probe
  underestimates (MAE 0.84); the patch grid does not (0.33).
- Lateral drift is poorly predicted from any single-frame feature. The label is
  noisy, so its attainable R2 is unknown; the result says only that one frame is
  clearly not enough.

Limits: linear probe; tail frames over-represent the approach to a failure; one
checkpoint; a readable feature does not prove the policy uses it.

### What the policy does before leaving the track

From telemetry of the 40-layout time-scale-1 baseline (13 failures, all CTE
limit, 3 within the first 300 steps). The corrective steering direction is
opposite in sign to CTE, as in the PID driver of `tools/collect_sim_frames.py`.

- In 9 of 13 failures the mean mapped steering over the last 5 steps is more
  than 0.7 of full lock toward the centre; the other 4 are between 0.20 toward
  and 0.27 away. As in §11 this is the submitted command, not a wheel angle.
- The departure is one sudden swing: 12 of 13 go from |CTE| > 1 to the limit in
  4-9 steps (0.2-0.45 s of simulated time); one drifts along the edge for 26
  steps. Earlier swings in the last 80 steps of the same episodes mostly peak
  below 1.0 (largest 1.6).
- 60% of all steps hit the steering-change clamp, in completed episodes too.

**Interpretation:** the evidence leans toward the control side. The policy has a
coarse position signal and reacts, but it has no usable signal for how fast it is
moving across the road, so it corrects late and at full lock with little margin.
This is not proven: 4 of 13 failures do not fit, and the probe limits above
apply. The next experiment is to give the policy dynamic state (speed, gyro or
previous-frame features) and compare laps paired against v2 50k. Spatially
pooled patch tokens are a second, independent lever.

### Simulator speed-up

- At time scale 1, 95% of evaluation wall time is the simulator's fixed 20 Hz
  frame clock: of each 50 ms step the encoder takes about 4 ms and the policy
  under 1 ms (§11 repeat run: stepping 1068 s of 1120 s; reload plus reset
  2.9-5.4 s per episode). Client-side optimisation cannot help.
- The simulator menu's **Time scale** speeds this up proportionally: at 2 the
  step interval halves while simulated time per step stays 0.05 s and lap times
  are unchanged.
- **Raise the FPS limit with it.** With time scale 2 and the default FPS limit
  60, step intervals alternate between about 17 ms and 33 ms (half each); with
  FPS limit 120 they are steady at 25 ms. The likely cause, inferred from the
  alternation and not verified in the simulator source for this build, is that
  frames are exchanged once per rendered frame, so a rendered frame at 60 FPS
  spans twice the simulated time.
- **Standard setting from now on: Time scale 2, FPS limit 120** for both training
  and evaluation (1.9x throughput: a 40-layout evaluation takes 33 min instead of
  58). Train and evaluate at the same setting. v2 50k was trained at time scale
  1; its baseline at the standard setting is the third table row.
- Evaluation `metadata.json` and training run configs now record
  `simulator_saved_settings` (time scale, FPS limit, random light/trees/cones),
  read from the Windows registry where the simulator saves its menu options.
  They are the saved values, not the live process state. The first three runs
  above predate this field.
- Not reproducible layouts yet: the installed build (v25.10.06) knows a
  `set_random_seed` message, but on the client port 9091 it logs "No delegates
  for msg type: set_random_seed" and layouts are not reproduced. The private API
  port (9092, key-protected) was not tried.
- Do not go beyond time scale 2 with policies trained at time scale 1 or 2 (see
  the table). Untested: several simulator instances on different ports
  (`--port`), the protocol's synchronous `step_mode`.

**Validation:** 25 pytest tests pass (paired statistics, latent lags and per-model observation in paired evaluation added).

---
