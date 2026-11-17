# Camera-only student and initial validation

> Historical record. Results, “current” choices and “next” plans below belong to the dated experiments, not the active work queue. See [current status](../../status.md) and [documentation index](../../README.md). Commands assume the repository root; raw `logs/` and `data/` artifacts are local-only unless explicitly noted.

### Stage 3.15: First Camera-Only Control Student — Offline Only (2026-11-16)

The user authorized implementation of a student with **no simulator motion
truth**, a 20-minute first-phase budget, one offline fit, and no simulator run.
No encoder or IMU is used in this iteration. The original perception head and
the true-state teacher remain frozen at their prior hashes.

**Implementation:** `rl/visual_student.py` provides a separate image-only
`VisualGeometryEncoder` and deterministic `VisualStudentActor`. The feature API
accepts images only and never calls `TrackState.encode` or receives telemetry.
Three predicted 15D geometric vectors at lags 0,2,4 plus 40 issued commands form
85 student inputs. Since each geometric prediction already uses lagged DINO
features, effective context extends back eight steps. Causal first-frame padding
is reset per trajectory and matches the batch path. No future, cross-condition,
cross-layout, missing-step or prior-episode frame is borrowed.

The actor is 85->256->256->2 with ReLU hidden layers and tanh outputs. Compatible
teacher geometry/history weights initialize it; the three teacher motion columns
are discarded and new historical-geometry columns start at zero. This creates
an adapted initialization, not a valid replacement for the teacher without
training. The student has its own checkpoint format and enforces the 85D input
contract plus geometry-head hash at load time. No critic or RL updates are used.

`tools/train_visual_student.py` consumes existing cached image features and
issued command histories, constructing student inputs before accessing teacher
state. The label branch separately queries the frozen teacher on true state and
the same post-command history at frame k, targeting command k+1. Student inputs
and teacher labels are stored separately. The first fit uses full reset-aligned
capture runs only; legacy tail-only records are not used. There are 39,020
training frames after excluding 9,973 frames from failed training captures, and
19,384 validation frames across the existing nine development layouts. Failed
validation captures stay in the evaluation. No locked evaluation capture enters
the fit. The 1..8-step startup padding remains included and explicitly marked.

**One fit:** AdamW lr 1e-4, weight decay 1e-4, batch 512, at most 15 epochs, seed
42. Loss is mapped steering MSE + 0.1 raw steering MSE + 0.5 mapped throttle MSE.
Steering difference remains 0.1 (executed delta 0.19998), throttle 0.2..0.7, and
accelerated unwind is disabled. Best development MAE selects epoch 14 of 15.
Fitting/evaluation took 15.39 seconds; the entire offline job took 19.39 seconds,
excluding implementation and tests. Artifacts:
`logs/visual_student_no_motion_v1/`; export:
`models/visual_student_no_motion_v1/student.pt`.

| Same-layout development metric | Adapted initialization | Trained camera-only student |
| --- | --- | --- |
| Mapped steering MAE | 0.15281 | 0.03987 |
| Mapped steering P95 | 0.39996 | 0.20339 |
| Mapped throttle MAE | 0.07457 | 0.01931 |
| Missed teacher unwind / 612 eligible frames | 418 | 71 |
| Wrong steering-change direction / 1,743 eligible frames | 532 | 114 |

The mapped-steering MAE ratio is 0.261, large-offset MAE ratio 0.255, and
missed-unwind-rate change -0.567, passing the pre-recorded offline gates
(<=0.8, <=1.1 and <=+0.02 respectively). Student negative-large steering
MAE/P95 is 0.01144 / 0.06627 on 129 frames; aggregate large-offset MAE/P95 is
0.02757 / 0.18699 on 209 frames. These small correlated subsets compare against
the true-state teacher's actions, not an infallible recovery oracle.

**Interpretation:** The student learned from the teacher without motion truth
in its inputs. The roughly 74% MAE reduction is relative to its deliberately
adapted initialization, not the old trained pure-vision RL policy. It is not a
closed-loop improvement claim. Partial observability, startup bias and
student-generated history distribution shift remain untested live risks.

**Runtime entry prepared, not executed:** `rl/eval_visual_student.py` loads only
this student format and the matching frozen visual head. Scoring/termination
may use true CTE/speed; they are separate from policy inputs. Temporal state is
reset at episode boundaries, and the existing runtime guards remain enabled.
A mocked evaluation runs with image and scoring fields but no pose/velocity/
gyro/acceleration fields, validating this separation without driving the sim.

**Validation and stop:** Image-only feature/actor APIs, adversarial telemetry
independence, nested live/cache padding, sequence breaks, teacher-weight transfer,
checkpoint contract/hash enforcement, differentiable action mapping and the
evaluation entry are tested. A real-data audit checks issued-history equality,
teacher-label timing against stored live teacher labels, disjoint layout splits,
and unchanged teacher/head weights. Saving/reloading reproduces student outputs.
The label-alignment maximum difference is 2.39e-7; all 83 tests pass, including
the mocked evaluation without pose/motion fields. Student maximum absolute
20-step mean steering error is still 0.202 (initialization 0.329), so persistent
control bias remains a live-test risk. `input_contract_audit.json` and
`decision.json` retain these checks; `runtime_source/` separately snapshots the
final runtime/evaluation implementation while `source/` preserves the fit code.
The first-phase outcome is an experimental compliant student eligible for a
separately authorized live screen; no automatic refit, DAgger, simulator run,
promotion or replacement of the old policies occurred.

### Stage 3.16: First Camera-Only Student Live Smoke and Replay Audit (2026-11-16)

The user authorized the next step. One bounded smoke was pre-recorded in
`logs/visual_student_smoke_v1/screen_plan.json`: one episode, at most 2,000 steps,
100-second process deadline, no automatic retries, training or layout expansion.
The initial launch stopped before simulator connection because EvalRecord
requires a nonexistent output directory; the prewritten plan had already created
the parent. The same smoke was then launched into a fresh `run/` subdirectory.
This was a preflight path correction, not a repeated driving attempt.

```bash
python rl/eval_visual_student.py --host ${DONKEY_SIM_HOST} --episodes 1 \
  --max-episode-steps 2000 --output-dir logs/visual_student_smoke_v1/run
```

Only the camera-derived 85D feature/history vector enters the separately loaded
student actor. No teacher model, encoder sensor, IMU, true motion, map or road
state is used by the policy. The evaluator uses CTE/speed for scoring and
termination only. Low-jitter sync, TCP_NODELAY and the 100 ms observation-wait
guard are enabled; steering difference 0.1 and throttle 0.2..0.7 are unchanged.
Saved simulator settings confirm timeScale 2/FPS 120, random light/trees on,
cones off. The student and frozen head checkpoint hashes match Stage 3.15.

**Result: 1/1 completed, 2,000 steps, 9 laps**, no exit or runtime timeout.
Mean speed was **2.50565 m/s**, mean absolute CTE **0.20811**, maximum absolute
CTE **0.93776** (CTE units, not metres). This is the first compliant camera-only
student result; it is not the previous privileged-motion 23/24 result.

For auditable input provenance, the evaluator now saves the exact 85D decision
inputs and raw actions to `policy_inputs_episode_01.npz`. It also requests
braking after the scored episode, before serializing images/results. A real-run
offline audit verified:

- All 2,000 input vectors contain the specified 45 visual values and 40 issued
  commands; initial history is zero and every subsequent history equals the
  actual preceding command sequence.
- Production mapped actions agree with recorded executed commands to 2.99e-8;
  batch actor replay agrees with recorded raw actions to 1.02e-6.
- Replaying the last 120 exact camera frames, discarding eight warmup frames and
  the final frame with no following decision, reproduces **111 decisions** using
  only images and issued history. Maximum feature and action differences are
  both **zero**. The scoring telemetry is not needed for this replay.

`audit.json`, `audit_replay.py`, source snapshots, camera tail, actual decision
inputs, telemetry, results and `decision.json` preserve the evidence. The mocked
evaluator test also checks that saved policy inputs have 85 columns and no truth
arrays. All 83 tests pass and `git diff --check` passes.

**Decision:** Basic compliant driving smoke passes; no model promotion or
superiority/high-completion-rate claim follows from one layout. No further live
attempt, fitting or runtime root-cause investigation was launched automatically.

### Stage 3.17: Frozen Camera-Only Student Four-Layout Screen (2026-11-17)

The user authorized the proposed bounded four-layout evaluation. The plan is
saved before driving in `logs/visual_student_screen_4layouts_v1/screen_plan.json`:
four fresh scene loads, at most 2,000 steps each, a 360-second process deadline,
no fitting, retries or automatic expansion. Preflight verified that student/head
hashes and the five relevant runtime source files match the Stage 3.16 smoke.
The saved simulator settings also match: timeScale 2, FPS 120, random light and
trees on, cones off, useSeed 0. These are saved menu settings, not independent
measurements of live engine state. Layout IDs denote scene loads, not reusable
seeds, and this run has no paired comparator.

```bash
python rl/eval_visual_student.py --host "$DONKEY_SIM_HOST" --episodes 4 \
  --max-episode-steps 2000 --output-dir logs/visual_student_screen_4layouts_v1/run
```

The existing student and original geometry head stay frozen. Runtime inputs
remain 45 image-derived geometric features plus 40 self-issued command values;
no teacher is loaded and no velocity truth, pose, map, encoder or IMU enters
the actor. Ground truth remains isolated to scoring, termination and logging.
Low-jitter sync, TCP_NODELAY, the 100 ms observation-wait guard, steering
difference 0.1, throttle 0.2..0.7 and disabled accelerated unwind are unchanged.
Each episode explicitly reloads the scene. No production code was changed.

| Layout | Completed / steps | Laps | Mean absolute CTE | Max absolute CTE | Mean speed (m/s) | Max CTE step |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Yes / 2,000 | 9 | 0.17877 | 0.95405 | 2.51614 | 79 |
| 2 | Yes / 2,000 | 9 | 0.19576 | 0.95127 | 2.51849 | 75 |
| 3 | Yes / 2,000 | 9 | 0.18435 | 0.94743 | 2.50256 | 77 |
| 4 | Yes / 2,000 | 9 | 0.21109 | 0.89125 | 2.51332 | 176 |

**Result:** 4/4 completed, 8,000 steps, 36 laps, pooled mean absolute CTE
**0.19249**, maximum **0.95405**, pooled mean speed **2.51263 m/s**, zero samples
at absolute CTE >= 1 and zero runtime aborts. Recorded evaluation elapsed time
was 223.37 seconds. All four maximum CTE values occur in the first 250 steps;
startup remains a useful target for larger evaluation, without inferring a
specific failure mechanism from these successful runs. CTE is in simulator
units, not metres.

**Input and replay audit:** All 8,000 stored inputs have 85 columns, finite
values and no truth arrays. Command histories reset to zero and exactly match
the preceding issued commands. Production action mapping agrees with recorded
commands to 2.99e-8; actor replay agrees with raw actions to 1.35e-6. Replaying
each final 120-image tail, excluding eight context frames and the final frame
without a subsequent decision, reproduces 111 decisions per episode: **444
image-only decisions with zero feature and action error**. Student/head and
encoder tensor hashes match the smoke; all captured source hashes match current
files. Only initial images and final 120 images per episode are retained;
decision inputs, actions and scoring telemetry cover all steps.

**Timing:** Encoder P95 is 4.81..4.87 ms across episodes, maximum 10.65 ms.
Policy P95 is 0.50..0.57 ms. The first policy call at layout 1, step 1 took
212.03 ms at near-zero speed; it remains in the recorded metrics. The existing
100 ms guard covers observation waiting, not policy/encoder work, so absence of
an abort does not assert that every part of the control cycle stayed below
100 ms. No timing root-cause investigation was launched.

Artifacts include `run/{metadata,results}.json`, `run/telemetry.jsonl`, per-episode
`policy_inputs_episode_*.npz`, image tails/source snapshots, `audit_replay.py`,
`audit.json`, `decision.json` and console logs. The real-run input/action/image
replay audit passes; all seven focused tests in `tests/test_visual_student.py`
pass, and `git diff --check` passes. The full historical suite was not rerun.

**Decision:** The four-layout completion screen passes. Keep the student
experimental; no checkpoint or baseline was replaced, and no further live run
or training was launched. Four successes are a small sample (nominal Wilson
95% completion interval approximately 51%..100%, assuming independent trials),
not proof of high reliability or superiority to the historical vision policy.
A larger frozen evaluation is the proposed next step, with startup excursions
and runtime tails retained in the primary metrics.


### Stage 3.18: Frozen Camera-Only Student 24-Layout Evaluation (2026-11-17)

The user authorized the proposed 24-new-layout evaluation after Stage 3.17.
`logs/visual_student_eval_24layouts_v1/eval_plan.json` was saved before driving:
24 fresh scene loads, at most 2,000 steps each, a 1,800-second process deadline,
no training, retries, parameter changes or automatic expansion. This batch is
reported independently of the earlier one-layout smoke and four-layout screen.
There is no paired comparator, and layout IDs are load identifiers rather than
reproducible seeds. Original locked training/evaluation captures were not used
for fitting or model selection in this task.

Preflight matched both student/head checkpoint hashes and every runtime source
hash to the four-layout screen. Saved simulator settings match: timeScale 2,
FPS 120, random light and trees on, cones off, useSeed 0. These remain saved menu
values rather than independently measured live engine settings.

```bash
python rl/eval_visual_student.py --host "$DONKEY_SIM_HOST" --episodes 24 \
  --max-episode-steps 2000 --output-dir logs/visual_student_eval_24layouts_v1/run
```

Runtime sources remain camera-derived features and self-issued commands only:
45 visual values plus 40 command-history values. No teacher is loaded; no
velocity truth, pose, map, wheel encoder or IMU enters the actor. Ground truth
is scoring, termination and logging only. Low-jitter sync, TCP_NODELAY, the
100 ms observation-wait guard, steering difference 0.1, throttle 0.2..0.7 and
disabled accelerated unwind remain unchanged. Each episode reloads the scene.

| Layout | Completed / steps | Laps | Mean absolute CTE | Max absolute CTE | Mean speed (m/s) | Max CTE step |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Yes / 2,000 | 9 | 0.16881 | 0.96403 | 2.53025 | 76 |
| 2 | Yes / 2,000 | 9 | 0.19414 | 1.08611 | 2.51464 | 1420 |
| 3 | Yes / 2,000 | 9 | 0.21647 | 0.98435 | 2.51680 | 813 |
| 4 | Yes / 2,000 | 9 | 0.18241 | 0.94586 | 2.52134 | 1416 |
| 5 | Yes / 2,000 | 9 | 0.21881 | 1.43441 | 2.52805 | 396 |
| 6 | Yes / 2,000 | 9 | 0.17088 | 1.34689 | 2.50948 | 181 |
| 7 | Yes / 2,000 | 9 | 0.21224 | 1.62820 | 2.52898 | 1474 |
| 8 | Yes / 2,000 | 9 | 0.18689 | 0.87360 | 2.50933 | 75 |
| 9 | Yes / 2,000 | 9 | 0.17235 | 0.92485 | 2.50425 | 75 |
| 10 | Yes / 2,000 | 9 | 0.19022 | 0.93475 | 2.49805 | 78 |
| 11 | Yes / 2,000 | 9 | 0.20139 | 1.03583 | 2.54188 | 76 |
| 12 | Yes / 2,000 | 9 | 0.19110 | 1.05718 | 2.53482 | 178 |
| 13 | Yes / 2,000 | 9 | 0.17978 | 0.87859 | 2.51910 | 1427 |
| 14 | Yes / 2,000 | 9 | 0.18352 | 0.98477 | 2.49538 | 77 |
| 15 | Yes / 2,000 | 9 | 0.18474 | 1.23281 | 2.51796 | 181 |
| 16 | Yes / 2,000 | 9 | 0.20816 | 0.99907 | 2.52478 | 711 |
| 17 | Yes / 2,000 | 9 | 0.17460 | 0.98767 | 2.52265 | 76 |
| 18 | Yes / 2,000 | 9 | 0.18771 | 0.97220 | 2.52564 | 1414 |
| 19 | Yes / 2,000 | 9 | 0.19916 | 1.11288 | 2.50474 | 181 |
| 20 | Yes / 2,000 | 9 | 0.19158 | 0.90340 | 2.52331 | 74 |
| 21 | Yes / 2,000 | 9 | 0.18049 | 1.34892 | 2.48676 | 175 |
| 22 | Yes / 2,000 | 9 | 0.20206 | 0.99877 | 2.52020 | 176 |
| 23 | Yes / 2,000 | 9 | 0.17881 | 0.94400 | 2.50855 | 77 |
| 24 | Yes / 2,000 | 9 | 0.21163 | 1.11720 | 2.51588 | 1000 |

**Result:** **24/24 completed, 48,000 steps, 216 laps** (9 laps per episode),
pooled mean absolute CTE **0.19116**, maximum **1.62820**, pooled mean speed
**2.51678 m/s**, zero exits and zero runtime aborts. Recorded evaluation elapsed
time was **1,328.48 seconds (22.14 minutes)**. CTE uses simulator units, not metres.
The nominal Wilson 95% completion interval is **86.2%..100%**, assuming independent
trials; 24 successes do not establish guaranteed reliability or a superiority
claim relative to a different policy or historical unpaired layouts.

**Startup and recovery diagnostics:** 15/24 episode maxima occur in the first
250 steps. Pooled mean absolute CTE is 0.21264 during those startup steps and
0.18810 afterward. Ten layouts contain 60 total samples at absolute CTE >= 1,
forming 14 contiguous excursions (8 beginning at startup, 6 afterward). Every
excursion is followed by a recorded sample below absolute CTE 0.5 and the
episode completes. This is descriptive evidence of recovery on these runs,
not a guarantee for future disturbances. The largest excursion is layout 7,
step 1474, CTE +1.62820; the next sample below 0.5 is step 1481. Thus remaining
tracking excursions are not confined to startup. Exact image tails cover only
the final 120 steps of each episode, so most early excursion images are not
available; full decision inputs/actions and scoring telemetry remain available.

**Input and replay audit:** All 48,000 finite decision inputs have 85 columns
and no truth arrays. Histories start at zero each episode and exactly match
issued commands. Production mapping agrees with logged commands to 2.99e-8;
actor replay agrees with all logged raw actions to 1.68e-6. Replaying the final
120 exact images per episode, excluding eight context frames and the final
frame without a following decision, reproduces **2,664 image-only decisions
with zero feature and action error**. Checkpoint hashes, encoder tensor hashes,
source snapshots and current runtime source hashes all match prior evidence.
Scoring telemetry is unnecessary for the image-to-action replay.

**Timing:** Pooled encoder P50/P95/P99/max is 4.11/4.80/5.92/12.70 ms; policy
P50/P95/P99 is 0.406/0.538/0.795 ms. The first policy call at layout 1, step 1
is the only policy sample above 50 ms, taking **205.12 ms** at near-zero speed.
All encoder samples are below 50 ms. Environment-step time P99/max is
26.61/33.79 ms. The first-call spike remains in primary records, without a
causal attribution or runtime retuning. The 100 ms guard only covers observation
waiting, not policy/encoder work; no abort does not imply every processing
stage stayed below 100 ms.

**Artifacts and validation:** `eval_plan.json`, `run/{metadata,results}.json`,
`run/telemetry.jsonl`, 24 `policy_inputs_episode_*.npz` files, initial images,
image tails, source snapshots, console logs, `audit_replay.py`, `audit.json`,
`analyze_results.py`, `diagnostics.json`, `per_episode.csv` and `decision.json`
are under `logs/visual_student_eval_24layouts_v1/`. Both post-run audit and
descriptive-diagnostic scripts completed successfully. No production code or
weights changed; the previously passing seven focused student tests were not
repeated. Whitespace checks and artifact integrity are recorded in
`validation.json`.

**Decision:** The frozen camera-only student passes this bounded 24-layout
evaluation and remains a candidate baseline. Preserve the checkpoint; do not
automatically train, expand layouts, alter runtime settings, replace deployment
models or infer superiority from the historical privileged-motion 23/24 result.
The next proposed work is offline inspection of startup and cruise excursion
inputs/actions before choosing a targeted control-training change.
