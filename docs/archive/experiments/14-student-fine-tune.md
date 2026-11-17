# Student excursion diagnosis and fine-tune

> Historical record. Results, “current” choices and “next” plans below belong to the dated experiments, not the active work queue. See [current status](../../status.md) and [documentation index](../../README.md). Commands assume the repository root; raw `logs/` and `data/` artifacts are local-only unless explicitly noted.

### Stage 3.19: Offline Diagnosis of Camera-Only Student Excursions (2026-11-17)

The user authorized starting the recommended offline inspection first. This
stage examines all 14 contiguous absolute-CTE>=1 excursions in Stage 3.18,
without driving, fitting, runtime edits or checkpoint changes. Artifacts are
under `logs/student_excursion_diagnosis_v1/`. `analyze.py` records input hashes
and a diagnostic plan before computation; `summarize.py` derives group summaries.
The initial script preflight used the wrong historical metadata key and stopped
before analysis; it was corrected from `args` to `artifacts`, retaining both
attempts in the console log. This was not a simulator attempt or model fit.

**Alignment and isolation:** Decision k uses camera frame k-1 and its preceding
issued-command history. Its predicted geometry is compared with telemetry k-1,
not the telemetry returned after command k. Stored lag-2/4 geometry matches its
earlier decision inputs exactly. All 48,000 actor outputs and mapped commands
replay within the existing 2e-5 / 1e-6 tolerances. True CTE is telemetry CTE;
reference heading/lookahead are reconstructed with the original hash-verified
fitted centre line and TrackState implementation. SDK `car` order was checked
as `(roll, pitch, yaw)`. Truth and the frozen 60k teacher are used only in this
separate offline diagnostic; no truth enters the live student input path.

Each event has a window from 35 decisions before through 20 after its returned
CTE peak. The primary comparison uses the 15 decisions immediately before
absolute CTE first reaches 1: 210 distinct decisions, with no duplicated samples
across events. The teacher receives the same recorded preceding command history
and the same production mapping. The visual-teacher diagnostic additionally
has the old privileged true-motion inputs, so its comparison with the student
does not isolate imitation error from partial observability.

| Pre-event subset | Events / decisions | CTE prediction MAE | Heading MAE (deg) | Student vs true-state teacher steering MAE | Visual + true-motion teacher MAE | Student MAE after true-geometry replacement |
| --- | --- | --- | --- | --- | --- | --- |
| All | 14 / 210 | 0.06129 | 1.44968 | 0.09117 | 0.03786 | 0.07281 |
| Startup (<=250) | 8 / 120 | 0.04913 | 1.18483 | 0.11745 | 0.02838 | 0.09619 |
| Cruise | 6 / 90 | 0.07751 | 1.80281 | 0.05614 | 0.05050 | 0.04165 |

Steering values are normalized commands, not angles. Geometry replacement is
performed at all three student geometry lags, preserving actual issued history.
The first five decisions are excluded from aggregate comparisons because their
complete lagged true geometry cannot be reconstructed from saved telemetry.
These are single-decision interventions on fixed trajectories; true geometry
can be out of distribution for a student trained on visual predictions. They
are neither recursive rollouts nor proof of a causal improvement.

**Findings:**
- Ten of 14 excursion peaks occur at centre-line indices 306..312 near the same
  bend, including seven startup events and three later-lap events. The repeated
  steps 175..181 therefore identify a repeated route segment, not evidence of
  a timer or reset-padding bug. The direct eight-step padding context has
  already expired, although earlier trajectory differences can still matter.
- Startup pre-event action disagreement is larger despite smaller measured
  geometric error than the cruise subset. Replacing all geometry reduces the
  all-event teacher-relative steering MAE by about 20%, leaving substantial
  disagreement. Across all eligible decisions, replacing CTE alone barely
  changes MAE (0.05222 -> 0.05220); replacing lookahead lowers it to 0.03920.
  CTE error alone is insufficient to select the next intervention.
- In the 210 pre-event decisions, the teacher reduces steering magnitude by
  >0.02 on 45 eligible frames, where previous |steer|>=0.3 and |CTE|>=0.5 or
  |heading|>=8 degrees. The student fails to reduce magnitude by >0.02 on
  20 of these frames. This teacher-relative definition is not a physical
  recovery certificate. Several startup peaks show student steering 0.23..0.40
  higher than the mapped teacher reference.
- The largest excursion (layout 7, peak step 1474) differs from the repeated
  startup case. At steps 1468..1472 both student and true-state teacher request
  strong negative raw steering and produce the same rate-limited commands on
  the recorded history. Thus neither a purely perceptual explanation nor a
  universal student-only explanation fits all events. The original steering
  limit remains unchanged; replay does not establish that loosening it helps.
- Across event windows, maximum policy/encoder/environment-step times are
  1.19/7.66/29.21 ms. The approximately 205 ms first-call policy spike at step 1
  is outside these windows. No long inference stall accompanies these selected
  windows; timing causality and server-side frame timing remain unproven.

**Limits and evidence:** No excursion-time camera images were retained, since
all events precede each episode's last-120-image capture. Saved predictions,
issued histories, telemetry and actor inputs support this diagnosis; lighting
or image content at the excursions cannot be visually inspected. Fourteen
selected events, ten sharing a bend, are not independent trials for significance
claims. The fitted-map heading/lookahead reference is the historical teacher
observation, not a new independent road-geometry measurement.

`report.json`, `summary.json`, `events.csv`, `aligned_windows.csv`,
`per_episode_metrics.csv`, 14 `event_*.png` plots and `excursion_locations.png`
preserve the diagnosis. `largest_excursion_raw_targets.json` retains the
additional teacher raw-action check for steps 1465..1476. Source snapshots,
`analysis_plan.json` and `audit.json` preserve provenance. All input artifacts,
model files and runtime sources remain unchanged. Three focused production
mapping/reference-metric tests pass; the full suite was not rerun.

**Decision:** Prioritize student control-response/temporal-state diagnosis and
fresh training-pool coverage of this bend over another generic CTE-head refit.
The present evidence does not distinguish insufficient supervision from missing
observable motion context, nor justify new sensors or relaxed steering limits.
Any future student fit should use fresh training-pool trajectories and be
judged on fresh evaluation layouts; this batch remains diagnostic data. The
candidate baseline is preserved. First-call warmup remains a separate proposed
runtime improvement; no warmup change, new training or live run was made here.


### Stage 3.20: Fresh Student-Driven Bend Data and One Control Fine-Tune (2026-11-17)

The user authorized targeted collection and a small student fit after Stage 3.19.
`logs/visual_student_bend_recovery_v1/plan.json` was recorded before driving:
eight new layout loads, 500-step cap, layouts 1..6 training and 7..8 development,
then one fine-tune from the original camera-only student. No candidate live
screen, automatic refit or baseline promotion is part of this step. The existing
straight-line kick collector would not target the observed bend/history issue;
instead, the frozen student drives naturally and a frozen teacher labels its
visited states offline (one DAgger-style supplement, no takeovers or kicks).

**Collection:** `tools/collect_visual_student_recovery.py` retains the baseline
85D camera/self-command inputs, original visual head, steering difference 0.1,
throttle 0.2..0.7, disabled accelerated unwind, low-jitter sync, TCP_NODELAY and
100 ms observation-wait timeout. Each episode reloads the scene. Preflight
matched baseline/head hashes and saved simulator settings: timeScale 2, FPS 120,
random light/trees on, cones off, useSeed 0. Saved menu values do not independently
prove live engine state.

The actor receives only the camera/history observation and is called before
privileged label-state construction. No teacher is loaded during collection.
`student_inputs_*.npz` contain observations/raw actions only; pre-action state
and map indices go into separate `label_state_*.npz` files. Every full camera
sequence is saved, including reset (501 images for 500 decisions). Truth remains
available for scoring, termination and isolated offline labels. Braking precedes
archive serialization and is also requested in cleanup.

```bash
python tools/collect_visual_student_recovery.py --host "$DONKEY_SIM_HOST" \
  --output-dir logs/visual_student_bend_recovery_v1/collection \
  --episodes 8 --train-layouts 6 --max-episode-steps 500
python tools/finetune_visual_student_recovery.py
```

Collection completed **8/8 x 500 steps, 4,000 decisions / 16 laps**, with zero
runtime aborts, in **148.55 seconds**. These are original-student collection
results, not new-candidate driving results. Both training (layout 6, max absolute
CTE 1.022) and development (layout 8, 1.192) contain natural large-offset samples.
All trajectories, including any failures if present, stay in their assigned
whole-layout split. No extra layouts or retries were launched.

**Collection audit:** All 4,000 camera decisions replay with zero feature error.
Command history resets and exactly matches issued commands; actor replay and
production mapping pass. State and map indices reconstructed from preceding
telemetry exactly match separate label records for decisions 2..500; the first
uses reset state. Checkpoint hashes remain unchanged. The mocked collector test
checks pre-action alignment and actor-before-label ordering.

**One fit:** `tools/finetune_visual_student_recovery.py` warm-starts the trained
student, retaining its 85->256->256->2 architecture and frozen visual head.
Original masks/groups are preserved: 39,020 old + 3,000 fresh = **42,020 training
decisions**; **19,384 old development** and **1,000 fresh development** decisions,
with **113 fresh bend decisions**. Previous 24-layout evaluation trajectories,
diagnosis windows and locked evaluation captures are excluded from fitting.
Original failed-training exclusions are retained; fresh failures would remain.

Teacher labels use true state and exact preceding issued history offline.
All old labels replay to 1.20e-7. Student features are read before this separate
label branch. Inputs, teacher state/labels and training selection/weights are
saved in separate files. Base weights are old=1, new=2, new bend (centre-line
indices 270..345)=8, multiplied by 2 for eligible teacher-unwind decisions;
validation weights are zero. Truth-based supervised example weighting does not
become an actor input.

AdamW: lr 2e-5, weight decay 1e-4, nominal batch 512, seed 42, at most 10 epochs /
180 seconds. Loss is weighted mapped steering MSE + 0.1 raw steering MSE + 0.5
mapped throttle MSE. One fit finishes 10 epochs in **2.99 seconds**, including
final evaluation/export. The preregistered equal-weight score of old-development
and fresh-bend MAE ratios selects epoch 10. New export:
`models/visual_student_bend_recovery_v1/student.pt`; its SHA256 is retained in
`fit/results.json` and `decision.json`.

| Development subset | Decisions | Original steering MAE / P95 | Candidate steering MAE / P95 | Missed teacher unwind, original -> candidate |
| --- | --- | --- | --- | --- |
| Old development | 19,384 | 0.03987 / 0.20339 | 0.04042 / 0.20511 | 71/612 -> 37/612 |
| Old large-offset | 209 | 0.02757 / 0.18699 | 0.02839 / 0.15905 | 8/87 -> 5/87 |
| Fresh development | 1,000 | 0.06705 / 0.26661 | 0.05516 / 0.24090 | 23/55 -> 5/55 |
| Fresh development bend | 113 | 0.06368 / 0.23946 | 0.03962 / 0.16228 | 14/26 -> 4/26 |

These are mapped command units and fixed-history teacher agreement, not physical
angles, candidate CTE or recursive rollouts. Fresh-bend MAE drops **37.8%** and
fresh-all MAE **17.7%**; old-all MAE rises **1.4%** and old-large MAE **3.0%**.
Old-development wrong-direction frames increase **114 -> 126** despite fewer
missed unwinds. This tradeoff is retained: control does not improve uniformly.
Two fresh development layouts and correlated bend frames are limited evidence,
and these layouts also participate in epoch selection.

All seven preregistered gates pass: >=50 fresh bend decisions; fresh-bend MAE
ratio <=0.90; fresh-all/old-all ratios <=1.05; old-large ratio <=1.10;
missed-unwind-rate increase <=0.02 on old and fresh development. No thresholds
were changed after the run. Teacher unwind is a reference action definition,
not a guaranteed recovery label.

**Validation/artifacts:** `audit_fit.py` independently reloads the candidate and
reproduces all saved development metrics/actions (maximum action difference
zero), checks exact features against camera-only archives, original masks,
disjoint layout groups, zero validation weights and unchanged teacher/head/
baseline hashes. `collection_audit.json`, `fit_audit.json`, full camera sequences,
input/label/selection NPZs, `fit/{training_manifest,results}.json`, source
snapshots, console logs, `plan.json` and `decision.json` preserve the run under
`logs/visual_student_bend_recovery_v1/`. All **86 tests pass**, including new
collector alignment, training-weight isolation and retention-gate tests.
Whitespace/integrity checks are recorded in `validation.json`.

**Decision:** Preserve the original 24/24-tested student as the baseline and the
new weights as an offline-qualified experimental candidate. No candidate live
run, second fit, promotion, new sensor or control-limit change occurred. Next
propose a bounded fresh-layout closed-loop comparison against the original
student with identical control/runtime settings, explicitly checking the
old-development regression. The candidate does not inherit the baseline's
24/24 completion result.
