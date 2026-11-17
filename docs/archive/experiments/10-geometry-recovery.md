# Geometry recovery collection and retraining

> Historical record. Results, “current” choices and “next” plans below belong to the dated experiments, not the active work queue. See [current status](../../status.md) and [documentation index](../../README.md). Commands assume the repository root; raw `logs/` and `data/` artifacts are local-only unless explicitly noted.

### Stage 3.5: Round B — First Geometry Recovery Collection Batch (2026-11-16)

**Scope decision:** Proceed with recovery supervision using the frozen `sdiff01 60k`
teacher, low-jitter sync enabled, original `max_steering_diff=0.1`, and
`reversal_unwind_scale=1.0` (acceleration disabled). The unfinished five-condition
evaluation is deferred, not a prerequisite for collecting this batch. Neither the
teacher nor the existing visual head was retrained or overwritten.

`rl/recovery_collection.py` adds bounded steering interventions to the existing
capture pipeline. `tools/eval_teacher_tolerance.py --mode round_b_collect` runs
only `teacher_recovery`, always enables full captures, and assigns every newly
loaded scene to `dagger_train_pool`. Existing Round A holdout data is not loaded.
Collection does not require the old perception probe cache.

**Collection procedure:**
- Preserve the first 250 steps without interventions to cover startup/acceleration.
- Start an intervention only when `|CTE| < 0.3`, `|heading| < 5 deg`, speed is at
  least `1.8`, the 1.0/1.5-position lookahead lateral coordinates are small, and
  the teacher's raw steering magnitude is below `0.35`.
- Override raw steering to `+/-0.35` for at most 4 steps, preserving the teacher's
  throttle and the original action mapping/rate limit. Alternate signed release
  targets of `0.6, 1.0, 1.4 CTE`; these are release thresholds, not guaranteed
  achieved offsets. Release earlier at `|CTE| >= 1.65` or `|heading| >= 15 deg`.
- Return control to the unchanged teacher. A recovery requires 8 consecutive
  observations with `|CTE| < 0.3` and `|heading| < 5 deg`; then wait 80 steps before
  another intervention. Unfinished recoveries time out after 160 steps.

**Calibration:** `logs/round_b_recovery_pilot_v1` used stronger/longer interventions
(`0.65`, up to 8 steps) and both episodes ended at the CTE limit. After reducing
interventions and adding the straight-road gate,
`logs/round_b_recovery_pilot_v2` collected 3,368 frames over four scenes, with
18/20 interventions recovered and 73 frames at `|CTE| >= 1`. Both calibration runs
are excluded from the main batch manifest.

**Main run:**

```bash
python tools/eval_teacher_tolerance.py --mode round_b_collect \
  --layouts 12 --max-episode-steps 2000 --device cpu \
  --output-dir logs/round_b_recovery_collect_v1
python tools/summarize_recovery_collection.py logs/round_b_recovery_collect_v1
```

| Coverage / outcome | Count |
|---|---:|
| Newly loaded scenes / captured frames | 12 / 15,187 |
| Unperturbed startup frames (steps 1..250) | 3,000 |
| `0.5 <= |CTE| < 0.8` | 999 |
| `0.8 <= |CTE| < 1.0` | 235 |
| `1.0 <= |CTE| < 1.5` | 321 |
| `1.5 <= |CTE| < 1.9` | 96 |
| `|CTE| >= 1.9` | 23 |
| Positive / negative frames with `|CTE| >= 1` | 294 / 146 |
| Interventions / recovered / CTE-limit exits / horizon-censored | 104 / 94 / 9 / 1 |

The 440 large-offset frames are `2.90%` of this collection. They are correlated
trajectory samples, not 440 independent recovery trials; signed coverage is still
uneven. These episodes deliberately contain perturbations, so the 9 terminal
episodes are not a natural-driving completion benchmark. Speed P10/P50/P90 is
`2.22 / 2.53 / 2.87`; there was no change to the throttle policy. Twelve initial
RGB-frame hashes are distinct, confirming distinct captured appearances.

**Artifacts and validation:**
- `logs/round_b_recovery_collect_v1/captures/`: approximately `0.89 GB` of full
  RGB images, same-frame 15D geometry labels, motion, histories, actual commands,
  counterfactual teacher actions, phase/event IDs, and timestamps.
- `coverage.json`: full-frame shape, CTE/geometry/motion consistency, command-history
  alignment, unchanged nominal actions/throttle, and original steering-rate limit
  checks passed. Same-frame teacher labels versus the next decision's teacher
  action differ by at most `1.31e-6`.
- `dataset_manifest.json`: capture hashes, scene groups, frozen teacher/head hashes,
  calibration exclusions, and explicit `round_a_locked_holdout_used=false`.
  `source/` preserves the exact five source files fingerprinted at run start.
- The teacher SHA256 remains `8d7ca02cf1db3f9d5fe09b7a9e35db04ca7852df16ca413341a27225e14c3e6d`;
  the existing head remains `80b3b1b343f01b8ec6b90f17fb5bdc34a531f3f7a08cc9f28d9f3e8048b59628`.
- `pytest -q`: **47 passed**. New checks cover intervention release, bounded duration,
  startup/corner gates, failed-event accounting, and frozen sync-only configuration.

**Training contract:** `images[k]`, `true_geom[k]`, and `history_post[k]` describe the
same post-step frame. `teacher_true_raw_action_on_frame[k]` labels that frame;
`student_raw_action[k]` is the command that led to it, using the preceding frame.
The `student_*` names are inherited capture fields: this run is teacher-driven
with interventions, not a completed student-on-policy DAgger iteration. Heading
and lookahead labels remain fitted-centerline estimates, with the limitations
described earlier; CTE units are not metres.

**Next:** Fit the same two-stage linear geometry head using training-pool data,
with layout-separated internal validation and explicit reporting on large-offset
and startup subsets. No new perception head or closed-loop improvement is claimed
by this collection result.

### Stage 3.6: Round B — Same-Architecture Geometry Head Retraining (2026-11-16)

`tools/train_geometry_recovery.py` retains the frozen DINOv2-S 4x4 grid and the
existing `6144 -> 15`, `(t,t-2,t-4) 45 -> 15` affine architecture. It combines the
23,924-frame legacy feature cache, Round A **training-pool captures only**, and
the new recovery collection. Before feature extraction/fitting, layouts 10..12
of each capture run are assigned to internal validation. Both Round A conditions
from the same layout stay in the same split. The original Round A locked holdout
files are never opened; calibration collection runs are excluded.

```bash
python tools/train_geometry_recovery.py
```

Artifacts: `logs/round_b_geometry_train_v1/{training_manifest.json,feature_cache.npz,results.json,predictions.npz}`;
candidate: `models/dinov2_geometry_15d_round_b_v1/proj15_lags_ridge.pt`.
The baseline head is preserved at its original path. The manifest fingerprints
all source captures, legacy cache, and baseline weights. Each full-episode capture
includes its reset frame for correct lag context. Evaluation below uses the same
valid-lag mask for both heads (first four frames of each sequence excluded).

There are **69,008 fitting frames across 192 layout groups** and **15,589 internal
validation frames across six groups** (15,553 valid-lag frames). All fitted weights
and standardization parameters use the fitting set only. Five-fold grouped
regularization selection and Stage-1 OOF predictions remain inside that fitting
set; these are not final-test results. Fitting/export took **13.52 s**, excluding
feature extraction and data preparation.

| Internal validation subset | Frames | Baseline CTE MAE / P95 | Round B CTE MAE / P95 | Baseline / Round B heading P95 (deg) |
|---|---:|---:|---:|---:|
| All | 15,553 | 0.0534 / 0.1353 | 0.0396 / 0.0995 | 2.90 / 2.30 |
| Cruise (`\|CTE\| < 0.5`) | 14,650 | 0.0516 / 0.1313 | 0.0377 / 0.0944 | 2.86 / 2.26 |
| Startup (step <= 250) | 2,223 | 0.0481 / 0.1195 | 0.0412 / 0.1042 | 2.75 / 2.48 |
| Large offset (`\|CTE\| >= 1`) | 117 | 0.1828 / 0.6244 | 0.1588 / 0.5672 | 6.84 / 5.84 |
| Positive large offset | 80 | 0.2069 / 0.7004 | 0.1743 / 0.5824 | 7.11 / 6.86 |
| Negative large offset | 37 | 0.1306 / 0.3350 | 0.1254 / 0.3667 | 4.74 / 4.02 |

Overall CTE P95 improves by **26.4%**, but large-offset P95 improves by only
**9.2%**, and negative-large-offset CTE P95 regresses. The small, correlated
large-offset subset does not establish robust recovery. A fresh 8-layout paired
natural-driving screen was then run, with the teacher, throttle/action
mapping, sync enabled, and reversal acceleration disabled in **both** conditions.
Only the visual head changes. Full captures are marked evaluation holdout.

```bash
python tools/eval_teacher_tolerance.py --mode round_b_vision_paired \
  --candidate-geometry-head models/dinov2_geometry_15d_round_b_v1/proj15_lags_ridge.pt \
  --layouts 8 --max-episode-steps 2000 --device cuda \
  --output-dir logs/round_b_vision_paired_8layouts_v1
```

**Completed paired screen (16 episodes):**

| Metric | Baseline head + sync | Round B head + sync |
|---|---:|---:|
| Completion | **8/8** | **7/8** |
| Mean laps | 9.000 | 8.875 |
| Mean episode `abs(CTE)` | 0.18005 | 0.18003 |
| Mean speed | 2.531 | 2.533 |
| Maximum `abs(CTE)` | 1.648 | 2.060 |
| Frames at `abs(CTE) >= 1` | 37 | 53 |
| Pooled perception CTE P95 | 0.1382 | **0.1003** |
| Pooled perception heading P95 | 2.87 deg | **2.33 deg** |
| Pooled encoder P50 / P95 | 4.25 / 4.96 ms | 4.27 / 5.10 ms |

Online perception improvements agree with offline validation, but the fixed
controller does not show better mean lateral tracking. The paired mean CTE
difference is `-0.000016 CTE` (95% bootstrap interval `[-0.01630, +0.01530]`).
The candidate does **not** pass the pre-agreed screen criterion of no fewer
completions; this small screen also does not statistically establish an intrinsic
model regression. All episodes and timing outliers are retained in the primary
comparison, with no post-hoc exclusion of difficult runs.

**Failure/timing observations (association, not an isolated causal test):**
- Candidate L04 recovered from `CTE=-1.634` at step 1567. Its CTE prediction error
  at that peak was only `+0.019`; a `72.33 ms` encoder stall at step 1515 preceded
  the later oscillation (`72.95 ms` observation interval at step 1516).
- Candidate L06 exited at step 1717 (`CTE=-2.060`). A `131.89 ms` observation
  interval at step 1677 was followed by short/irregular intervals and oscillation.
  At exit, CTE prediction error was `-0.105` and heading prediction error was
  `-6.48 deg`; the failure cannot be reduced to CTE underestimation alone.
- Baseline L08 also had large timing disturbances (maximum observation interval
  `380.30 ms`, maximum encoder `193.39 ms`), reached `CTE=-1.648`, and recovered.
  Timing disturbances therefore affect both conditions; the results neither
  isolate timing as the sole cause nor identify the source of the stalls.

`timing_diagnostics.json` retains per-episode peaks and their steps;
`layout06_candidate_failure.png` plots CTE estimates, applied steering, and timing.
`audit.json` confirms 8 complete pairs, identical sync-only control, preserved
motion inputs, aligned histories/actions (maximum next-action difference
`1.31e-6`), evaluation-only captures, and unchanged weight hashes. The original
Round A locked holdout remains unused. Source snapshots are retained alongside
both the training and evaluation runs. **50 pytest tests pass**, including checks
for held-out file exclusion, layout grouping, batch/live lag equivalence, and
paired-head control configuration.

**Decision:** `decision.json` marks the Round B head **not promoted**. Preserve
`models/dinov2_geometry_15d/proj15_lags_ridge.pt` as the baseline; retain the candidate
(`SHA256 6fd7b4aabf25f86b9b8c932e812411034d64c3f0f2d23edaeb3f65ed4fa02657`)
for diagnosis. Do not expand this candidate to the 24-layout final evaluation or
claim a closed-loop improvement. Teacher weights and reversal settings were not
changed. The remaining issue is distinguishing timing/feedback sensitivity from
perception residual structure before choosing another training change.
