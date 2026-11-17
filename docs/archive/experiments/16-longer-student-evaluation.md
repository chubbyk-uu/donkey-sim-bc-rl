# Longer original-student evaluation

> Historical record. Results, “current” choices and “next” plans below belong to the dated experiments, not the active work queue. See [current status](../../status.md) and [documentation index](../../README.md). Commands assume the repository root; raw `logs/` and `data/` artifacts are local-only unless explicitly noted.

### Stage 3.23: Original Student, Ten Fresh Layouts at 3,000 Steps (2026-11-17)

The user explicitly requested ten new random layouts with a 3,000-step cap,
after deciding to retain the original student and end speculative optimization.
This is the requested longer-horizon check only; no A/A experiment, candidate
trial, fitting or automatic follow-up is included. The initial time estimate was
14..16 minutes including scene loading, saving and checks.

Preflight matched the original student/head hashes and existing runtime sources
to the 24-layout run. Saved simulator settings match: randomLight=1,
generateTrees=1, useSeed=0, cones disabled, timeScale=2, FPS=120. These are saved
menu values rather than independent measurements of live engine state. Layouts
are fresh scene reloads, not reproducible seeds. `eval_plan.json` was saved before
driving, with ten episodes, 3,000-step cap and 1,200-second process deadline.

```bash
python rl/eval_visual_student.py --host "$DONKEY_SIM_HOST" --episodes 10 \
  --max-episode-steps 3000 \
  --output-dir logs/visual_student_eval_10layouts_3000_v1/run
```

The model remains `models/visual_student_no_motion_v1/student.pt` (original
SHA256 b00d7252e7a54041b7f34bba065188e02aaba13e606e5b332c21258e2532e6b2).
The 85D actor input remains camera-derived geometry and its own issued-command
history only. No teacher, encoder sensor, IMU, true motion, pose or map is used
for control. Truth is scoring/termination/logging only. The existing single-
student entry is unchanged: steering difference 0.1, throttle 0.2..0.7,
unwind acceleration disabled, low-jitter sync, TCP_NODELAY, 100 ms observation-
wait guard, scene reload every episode. No new actor prewarm was added.

| Layout | Steps | Laps | Mean absolute CTE | Max absolute CTE | Mean speed (m/s) |
| --- | --- | --- | --- | --- | --- |
| 1 | 3,000 | 14 | 0.18630 | 1.13358 | 2.54918 |
| 2 | 3,000 | 14 | 0.19478 | 1.42320 | 2.53160 |
| 3 | 3,000 | 14 | 0.16517 | 0.82645 | 2.53290 |
| 4 | 3,000 | 14 | 0.18142 | 1.00877 | 2.53515 |
| 5 | 3,000 | 14 | 0.17412 | 0.97588 | 2.53743 |
| 6 | 3,000 | 14 | 0.18565 | 1.11976 | 2.53924 |
| 7 | 3,000 | 14 | 0.19708 | 0.95695 | 2.55303 |
| 8 | 3,000 | 14 | 0.19286 | 0.96245 | 2.54815 |
| 9 | 3,000 | 14 | 0.17650 | 1.04630 | 2.53076 |
| 10 | 3,000 | 14 | 0.19283 | 0.89887 | 2.54520 |

**Result: 10/10 completed**, every episode **3,000 steps / 14 laps**,
**30,000 steps / 140 laps total**, zero exits and zero runtime aborts. Mean
absolute CTE is **0.18467**, maximum **1.42320**, below the **2.0 CTE termination
limit**. There are 27 samples at absolute CTE >=1. Mean speed is **2.54026 m/s**.
CTE uses simulator units, not metres. Completion denotes reaching the step cap
without termination, not an independent assertion about every wheel/road edge.

Recorded evaluation elapsed time is **793.58 seconds (13m14s)**; metadata
start-to-finish timestamps span about **13m26s**, including metadata/setup
overhead. The process exits normally. This longer-cap batch is reported
separately from the historical 24 layouts x 2,000 steps, without pooling their
completion rates or interpreting unpaired CTE differences as improvement.

**Audit:** All 30,000 finite 85D inputs and raw actions pass issued-history,
steering/throttle-limit, scoring and action-replay checks. Maximum raw-action
replay error is 1.48e-6; each episode's final 120 saved images reproduce 111
camera/history decisions after context exclusion: **1,110 decisions with zero
feature and image-to-action error**. Student/head and encoder tensor hashes,
source snapshots and current runtime source hashes match prior evidence.

Artifacts are in `logs/visual_student_eval_10layouts_3000_v1/`: `eval_plan.json`,
`run/{metadata,results}.json`, telemetry, per-episode policy input archives,
initial images, final camera tails, source snapshots, `audit_replay.py`,
`audit.json`, `decision.json` and console logs. No production code or weights
changed; the existing tests were not repeated. Real-run replay/integrity and
whitespace checks are recorded in `validation.json`.

**Decision:** User-requested extended verification passes. Retain the original
student; no additional experiment, training or optimization is started.
