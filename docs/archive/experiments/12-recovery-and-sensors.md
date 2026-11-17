# Recovery heads and sensor-source audit

> Historical record. Results, “current” choices and “next” plans below belong to the dated experiments, not the active work queue. See [current status](../../status.md) and [documentation index](../../README.md). Commands assume the repository root; raw `logs/` and `data/` artifacts are local-only unless explicitly noted.

### Stage 3.11: Targeted Negative-Offset Recovery Data (2026-11-16)

Added `RecoveryCollectionConfig.directions` and the collection-only CLI option
`--recovery-direction negative`. Balanced directions remain the default. The
negative mode cycles the same three release targets; steering magnitude 0.35,
four-step kick cap, startup/straight-road gates, heading/CTE release guards,
teacher throttle, action history and steering rate limit remain unchanged.
Seven new tests cover negative direction scheduling and invalid signs; all
73 tests pass. This changes collection actions, not the frozen teacher.

One capped batch, `logs/round_b_negative_recovery_v1/`, collected six fresh
training-only layouts, up to 1,500 steps each, with TCP_NODELAY and the 100 ms
guard enabled. Saved simulator settings remained timeScale 2/FPS 120, random
light/trees enabled, cones disabled. No runtime timeout occurred.

| Coverage | Previous recovery batch | Added negative batch |
| --- | --- | --- |
| Frames | 15,187 | 7,698 |
| Startup frames | 3,000 | 1,500 |
| CTE <= -1 frames | 146 | 198 |
| CTE >= +1 frames | 294 | 0 |
| Interventions / recovered | 104 / 94 | 55 / 49 |
| CTE-ended episodes | 9 | 2 |

All full-frame/geometry/motion/history/action-alignment audits pass; maximum
teacher-action timing-alignment error is 8.35e-7. Failed and horizon-truncated
recoveries remain in the captures. Both batches together provide 344 negative
large-offset frames. `coverage.json`, `dataset_manifest.json`, source snapshots,
and the original split manifest preserve provenance; no prior locked evaluation
capture is used for fitting.

The pre-recorded collection target was 200 new negative frames; actual coverage
is 198, 1% short. The batch stopped at its six-layout cap. The documented decision
is to allow one unchanged-architecture offline fit with these samples, without
collecting more layouts or relaxing the validation-performance gates. Validation
remains grouped by layout (last three training-pool layouts of each capture run);
existing internal validation is development data, not final independent evidence.

**Single offline retraining result:**
`logs/round_b_negative_geometry_train_v1/` combines the legacy feature cache with
Round A training-pool captures and the two recovery batches. It fits the unchanged
two-stage linear lag Ridge head on 72,917 frames and validates on 19,384 frames
from nine held-out development layouts (19,336 valid lagged samples). The
grouped fit itself took 5.38 seconds, excluding image feature extraction/cache
construction. Source snapshots and input fingerprints are retained. The export
is `models/dinov2_geometry_15d_negative_v1/proj15_lags_ridge.pt`; the teacher and
original perception checkpoint hashes were verified unchanged.

| Development subset | Frames | Original CTE P95 | New CTE P95 | Original / new heading P95 |
| --- | --- | --- | --- | --- |
| All | 19,336 | 0.13493 | 0.10040 | 2.983 / 2.354 deg |
| Negative CTE <= -1 | 129 | 0.31297 | 0.31617 | 4.719 / 4.355 deg |
| Positive CTE >= +1 | 80 | 0.70038 | 0.58530 | 7.111 / 7.021 deg |
| Absolute CTE >= 1 | 209 | 0.50341 | 0.47537 | 5.421 / 5.164 deg |
| Startup | 2,964 | 0.11933 | 0.10472 | 2.839 / 2.505 deg |

**Decision: not promoted.** The pre-recorded gate requires at least 5% lower
negative-large-offset CTE P95, with aggregate CTE/heading P95 degradation bounded
to 10%. Aggregate metrics pass, but the target tail ratio is 1.010, exceeding
0.95. The negative-tail difference is small and is not claimed statistically
significant; it simply provides no evidence of the required improvement.
No second fit, closed-loop candidate screening, or 24-layout evaluation was
launched. Preserve the useful training captures and experimental head, keep the
original runtime baseline, and retain `offline_gate.json` / `decision.json` as
the explicit stopping decision. These offline results do not establish improved
closed-loop completion or prove that additional data can never help.

### Stage 3.12: Common-Layout Offline Control Comparison (2026-11-16)

The user authorized Step 1 only: compare the three existing heads through the
same frozen teacher, without driving, fitting, or changing the baseline.
`tools/compare_geometry_control.py` consumes the Stage 3.11 feature cache and
uses the same 19,336 valid-lag frames from nine development layouts / 12
trajectories. Both retrained heads excluded these layouts from fitting. Original
locked evaluation captures remain unopened. Artifacts are in
`logs/geometry_control_compare_v1/`.

**Replay semantics and audit:** At frame k, the 18D state is joined with its
recorded **post-command** 40D history to query action k+1. Every head receives
the same motion/history; only the 15D geometric inputs differ. Steering is
mapped with max steering 1, max difference 0.1 (executed delta limit 0.19998),
throttle 0.2..0.7, and no accelerated unwind. This is a one-step counterfactual
on recorded trajectories, not a recursive rollout. True states/history were
checked against each saved capture; maximum replayed true-teacher action
difference from stored live labels was 1.20e-7. The mapper is tested against
production behavior, including saturation and reversals.

| Head | Mapped steering MAE | Mapped steering P95 | Negative-large-offset steering P95 | Missed teacher unwind / 612 eligible frames | Max absolute 20-step mean steering error |
| --- | --- | --- | --- | --- | --- |
| Original | 0.03756 | 0.20535 | 0.12026 | 66 | 0.11642 |
| Round B | 0.03316 | 0.18299 | 0.09684 | 60 | 0.11198 |
| Negative supplement | 0.03296 | 0.18252 | 0.07771 | 58 | 0.11911 |

Steering values are normalized command units, not degrees. The negative-large
subset contains 129 frames from six layouts/trajectories. Relative to the
original head, the negative-supplement head lowers aggregate steering MAE by
12.2% and negative-large steering P95 by 35.4%, even though its negative CTE
P95 did not improve. Its per-layout macro MAE difference is -0.00439, with an
exploratory paired-layout bootstrap interval [-0.00501, -0.00369]. These are
development-set comparisons, not independent final-test confidence claims.

**Risks retained:** Maximum continuous bias does not improve (0.119 vs 0.116),
with the worst window at startup on negative-recovery layout 6, steps 10..29.
Large-offset steering-change direction mismatches are 10 versus 9 frames;
teacher-relative missed-unwind counts are small and correlated. Here unwind
means the teacher reduces steering magnitude by >0.02 from |previous steering|
>=0.3 while |CTE|>=0.5 or |heading|>=8 deg; it does not certify physical recovery.
The true-state teacher is a reference controller, not an infallible expert.

**Input-repair diagnostic:** Keeping the negative head's other inputs fixed,
substituting true lookahead points lowers aggregate mapped steering MAE from
0.03296 to 0.01158; substituting true heading lowers it to 0.02696. Substituting
only true CTE leaves 0.03292. Thus CTE P95 alone is a poor selector for this
controller. These are non-additive counterfactual interventions with correlated
geometric inputs, not proof of independent physical causes.

**Decision:** Preserve the historical Stage 3.11 CTE-gate result and all baseline
weights. This new control-related evidence justifies nominating the negative
head for one four-layout paired closed-loop screen using the unchanged teacher
and identical runtime guards. No promotion or student-controller training is
justified yet. That screen was **not started** during this offline-only step.
`decision.json`, `report.json`, `per_layout.csv`, `worst_bias_windows.csv`, and
validation-only `predictions.npz` retain the evidence and source snapshots.
All 76 tests pass; new checks cover exact production mapping, sequence-safe bias
windows, and teacher-relative direction/unwind masks. `git diff --check` passes.

### Stage 3.13: Four-Layout Paired Screen of the Negative-Supplement Head (2026-11-16)

The user authorized one fresh paired screen. Before execution,
`logs/negative_head_paired_4layouts_v1/screen_plan.json` fixed four layouts, two
conditions per layout, a 2,000-step episode cap and 540-second process deadline,
with no automatic retries, fitting, expansion or promotion. Both conditions use
the frozen original teacher, low-jitter sync, TCP_NODELAY, 100 ms observation
timeout, steering difference 0.1, throttle 0.2..0.7, and no accelerated unwind.
Saved simulator settings remained timeScale 2/FPS 120 with random light/trees on
and cones off. Only the geometry head changes; `vision_round_b` in the artifacts
denotes the **negative-supplement** checkpoint, not the earlier Round B head.

```bash
python tools/eval_teacher_tolerance.py --mode round_b_vision_paired \
  --candidate-geometry-head models/dinov2_geometry_15d_negative_v1/proj15_lags_ridge.pt \
  --layouts 4 --max-episode-steps 2000 --tcp-nodelay \
  --observation-timeout 0.10 --save-captures \
  --output-dir logs/negative_head_paired_4layouts_v1
```

All eight episodes completed: 16,000 total steps, 72 laps, zero runtime aborts.
Source snapshots match the recorded hashes. Evaluation-only captures pass the
motion/history/action-rate-limit audit, next-action alignment checks, and frozen
teacher/head weight hash checks. Original Round A locked captures remain unused.

| Metric | Original head | Negative-supplement head |
| --- | --- | --- |
| Completion / laps | 4/4 / 36 | 4/4 / 36 |
| Mean absolute CTE | 0.17236 | 0.17165 |
| Maximum absolute CTE | 0.96683 | 1.19103 |
| Frames at absolute CTE >= 1 | 0 | 4 |
| Mean speed (m/s) | 2.5372 | 2.5303 |
| Pooled online perception CTE error P95 | 0.13705 | 0.09732 |
| Pooled online heading error P95 (deg) | 2.9579 | 2.3622 |
| Maximum encoder time (ms) | 10.83 | 82.27 |

The paired mean absolute CTE difference is -0.000710 CTE (95% layout-bootstrap
interval [-0.01303, +0.00762]), only a 0.4% point-estimate improvement. Layout 1
favors the candidate; layouts 2..4 have slightly higher candidate mean CTE. The
candidate's perception improvements persist online, but do not yield a clear
lateral-tracking or completion benefit in this screen.

**Timing caveat, without reopening diagnosis:** Candidate layout 4 logged an
82.27 ms encoder call at step 1347; its CTE peak -1.191 occurred at step 1354.
The 82.90 ms observation interval appears in the following control cycle. All
these samples remain in primary metrics. This association does not identify
the stall source or establish that the head caused the larger CTE peak. The
100 ms observation-wait guard neither covers encoder work nor triggered here;
its documented scope remains unchanged.

**Decision:** Completion screening passes, but no clear closed-loop improvement
is demonstrated. Keep the original head as the working baseline and retain the
negative head as experimental; do not overwrite either checkpoint or infer
reliable completion from four layouts. Do not automatically run further
layouts, train a student controller, or reopen runtime root-cause work.
`audit.json`, `decision.json`, `episodes.jsonl`, full captures, telemetry and
`tolerance_results.json` retain the result. No production code changed for this
screen; the previous 76-test validation remains applicable and
`git diff --check` passes.

### Stage 3.14: Sensor Source Audit and Student Input Requirement (2026-11-16)

**User requirement:** Student control inputs may include the camera, wheel
encoder measurements, IMU measurements and self-generated command history.
Direct simulator linear velocity, position, true CTE, road-relative heading,
map/centre-line lookahead, and other privileged state are not student inputs.
Such state may remain isolated for teacher labels, evaluation and diagnostics.
Existing geometry-head pipelines retain true velocity and therefore do not meet
this requirement. Historical results remain valid privileged-motion references;
they do not establish compliant student performance. Do not continue the proposed
control-student fine-tuning with the old motion input contract.

At the user's request, upstream GitHub files were retrieved through the configured
HTTP proxy and pinned to commit `ec04c400f045222375ec82e649bb051fb17aad41`.
The installed Windows `Assembly-CSharp.dll` was independently inspected as .NET IL;
it was not modified or replaced. Evidence, source copies, assembly hash and the
read-only inspection script are under `logs/sensor_interface_audit_v1/`.

- In the local `Car` implementation, velocity is
  `transform.InverseTransformDirection(rb.velocity)`. `TcpCarHandler` divides
  `GetVelocity()` by 8 and sends its magnitude as `speed`; extended `vel_x/y/z`
  use the same vector. Thus the existing speed is rigid-body truth, not a wheel
  encoder. An earlier conversational claim that these components were necessarily
  world-frame was inaccurate: `Car` converts them to body frame, whereas the
  separate `UnityStandardCarAdapter.GetVelocity()` returns Rigidbody velocity
  without this conversion. Runtime vehicle-adapter identity must be established
  before asserting a universal coordinate convention.
- `Car.GetGyro()` returns the stored `rb.angularVelocity`, with no body-frame
  transformation in that update. `GetAccel()` is a finite difference of the
  local velocity vector; it is not by itself a calibrated accelerometer specific
  force model. `gyro_*` / `accel_*` fields exist, but their semantics must be
  verified before using them as a body-mounted IMU. Simulated IMU signals can
  legitimately originate in engine physics; the measurement model, frame and
  units must represent the permitted sensor, without exposing pose/velocity as
  extra policy features.
- The local binary and upstream source contain `Odometry`, and TCP telemetry
  can transmit `o.Label -> o.GetNumberRotations()` for configured components.
  The upstream implementation estimates cumulative absolute rotations from wheel
  mesh quaternion differences and explicitly warns that it is unreliable at
  high wheel spin rates. It also loses rotation direction. This is distinct
  from `speed`. Presence of this class does not prove that the current vehicle
  has an enabled, usable encoder feed. The Python SDK/custom sync currently do
  not preserve arbitrary odometry-label fields as policy observations.

**Status:** Source audit only; no sensor adapter, compliant student, training,
simulator driving or new performance claim was introduced. Next establish the
active vehicle's raw encoder/IMU output, then implement a whitelisted sensor
observation path without falling back to true velocity. The old teacher input
layout cannot simply be relabelled as encoder/IMU data.
