# Paired student evaluation and cross replay

> Historical record. Results, “current” choices and “next” plans below belong to the dated experiments, not the active work queue. See [current status](../../status.md) and [documentation index](../../README.md). Commands assume the repository root; raw `logs/` and `data/` artifacts are local-only unless explicitly noted.

### Stage 3.21: Shared-Layout Closed-Loop Screen of the Bend Student (2026-11-17)

The user authorized the next closed-loop comparison. Before driving,
`logs/visual_student_bend_paired_4layouts_v1/plan.json` fixed four fresh layout
loads, two actors per layout, 2,000-step cap, 900-second process deadline, no
retries, automatic expansion, fitting or promotion. The original student and
Stage 3.20 candidate use the same original frozen geometry head. Preflight
matched all three checkpoint hashes and saved simulator settings to prior runs.
This evaluation uses fresh scenes, not the eight training/development scenes.

**Paired entry and protocol:** `rl/eval_visual_students_paired.py` reloads the
scene once per layout and sets wrapper scene reload to zero; between actors it
only resets the vehicle. The image encoder's internal lag context, outer visual
lags and command histories reset per episode. First-run order is balanced:
layout 1 candidate first, 2 baseline first, 3 candidate first, 4 baseline first
(order seed 42, not a simulator seed). The endpoint protocol and saved initial
views support scene pairing; simulator seeds are not independently verified.

Both actors receive one stateless 85-zero prediction **before connection**.
This common warmup avoids putting the known first-call spike into only one
actor's live episode. It issues no command and changes no visual/history cache.
Warmup took 437.44 ms for baseline and 0.55 ms for candidate before connection;
both are ready before driving. This is an explicit shared evaluation protocol
change from historical single-student runs; those entry defaults are unchanged.
No inference/wait samples are removed from the scored episode records.

All live control inputs remain the same 85 camera/self-command features. No
teacher is loaded and no encoder/IMU, motion truth or map enters either actor.
Truth is scoring, termination and logging only. Both use steering difference
0.1, throttle 0.2..0.7, disabled accelerated unwind, low-jitter sync, TCP_NODELAY
and 100 ms observation-wait timeout. Saved simulator settings remain timeScale 2,
FPS 120, random light/trees on, cones off, useSeed 0 (saved menu values, not an
independent live-engine measurement).

```bash
python rl/eval_visual_students_paired.py --host "$DONKEY_SIM_HOST" \
  --layouts 4 --max-episode-steps 2000 \
  --output-dir logs/visual_student_bend_paired_4layouts_v1/run
```

| Metric | Original baseline | Bend candidate |
| --- | --- | --- |
| Completion / steps / laps | 4/4 / 8,000 / 36 | 4/4 / 8,000 / 36 |
| Mean absolute CTE | 0.18147 | 0.18945 |
| Maximum absolute CTE across all episodes | 1.22731 | 1.16355 |
| Startup mean absolute CTE (first 250 steps) | 0.21290 | 0.18254 |
| Cruise mean absolute CTE (remaining steps) | 0.17698 | 0.19044 |
| Frames with absolute CTE >=1 | 13 | 10 |
| Mean speed (m/s) | 2.51291 | 2.52299 |

All eight episodes complete with no exit or runtime abort, **16,000 steps / 72
laps total**, evaluation elapsed **450.68 seconds**. CTE values are simulator
units, not metres. Both models complete nine laps per episode.

| Layout | Original / candidate mean absolute CTE | Candidate minus original | Original / candidate max absolute CTE |
| --- | --- | --- | --- |
| 1 | 0.16429 / 0.18985 | +0.02556 | 1.01158 / 0.94147 |
| 2 | 0.20110 / 0.20497 | +0.00386 | 1.00396 / 1.08042 |
| 3 | 0.18332 / 0.17995 | -0.00337 | 1.22731 / 1.08662 |
| 4 | 0.17716 / 0.18303 | +0.00588 | 0.92677 / 1.16355 |

**Interpretation:** Startup mean CTE decreases **14.3%** and cruise mean CTE
increases **7.6%**, with both directions occurring on every layout. Whole-episode
mean CTE increases **4.4%**. The paired mean difference is **+0.007983 CTE**,
with exploratory 95% layout-bootstrap interval **[-0.001059, +0.020138]**. This
small screen does not statistically establish a population regression, and it
does not demonstrate an overall improvement. Candidate global maximum is lower,
but per-layout maxima improve on only two of four layouts; mean per-layout peak
difference is +0.02561. Thirteen vs ten large-offset samples are too few to infer
recovery reliability. Candidate speed is approximately 0.4% higher; this does
not isolate a cause of the tracking tradeoff.

The pre-recorded checks are: no fewer completions (passes), paired mean CTE
improvement (fails), no more absolute-CTE>=1 frames (passes). Thus the candidate
**fails the screen**. This preserves the offline improvements from Stage 3.20
without mistaking them for whole-episode closed-loop improvement.

**Replay/provenance audit:** Every initial and returned image is saved; each
2000-decision episode has 2001 camera frames. All **16,000 decision inputs**
reconstruct from the full camera sequences plus actual issued history with
**zero feature error**, including both levels of startup padding. Raw actor
replay error is at most 1.46e-6; mapped commands agree to 2.99e-8. Command-history
resets, lag-2/4 equality and scoring metrics are checked per episode. Both
checkpoints, geometry-head/encoder tensor hashes, current runtime source hashes
and captured source hashes remain unchanged. Initial image mean pixel differences
within each pair are 0.076..0.186 on the 0..255 scale; views are similar, not
bit-identical. Scene identity is supported by the reload/reset protocol rather
than a newly claimed deterministic seed mechanism.

**Timing:** Baseline/candidate policy P99 is 0.529/0.537 ms, max 1.39/1.24 ms;
encoder P99 is 5.83/5.82 ms, max 9.10/8.81 ms. Environment-step maxima are
33.83/29.21 ms. Shared actor warmup occurs outside the recorded live episodes;
no live first-call outlier or wait-guard abort occurs. The 100 ms guard's scope
remains observation waiting, not all computation.

`audit_results.py`, `audit.json`, `decision.json`, `paired_results.png`,
`plot_results.py`, per-episode full camera/input archives, telemetry, source
snapshots, `run/{metadata,results}.json` and `plan.json` preserve the comparison.
All data are evaluation-only and excluded from fitting. The paired-entry tests
verify balanced ordering, missing/duplicate-pair rejection, one reload per pair,
per-episode history resets and execution with no pose/motion fields. All **88
tests pass**; whitespace and integrity checks are in `validation.json`.

**Decision:** Keep `visual_student_no_motion_v1` as the baseline. Preserve the
bend candidate as experimental and do not expand it to a larger evaluation,
refit automatically or promote it. Any further fine-tune should first strengthen
cruise-retention criteria and examine this startup/cruise tradeoff. The four-layout
comparison provides a concrete reason to retain the original, without claiming
that the candidate is universally inferior.


### Stage 3.22: Route/Lap Alignment and Bidirectional Actor Replay (2026-11-17)

The user authorized the planned bounded offline diagnosis. No new simulator
run or driving-model fit was made. Inputs are all eight complete Stage 3.21
trajectories (four shared layouts, 16,000 scored decisions), not only selected
excursions. Both actors, teacher and fitted map remain frozen. Artifacts:
`logs/visual_student_cross_replay_v1/`.

`analyze.py` writes `plan.json` and fingerprints evaluation input/camera archives,
telemetry, model files and centre line before analysis. Each actor receives
exactly the same saved 85D camera/history vector for a given trajectory row.
No command history is swapped independently of its images. Teacher state is
reconstructed from the preceding telemetry row, matching the existing teacher
contract; first decisions lack saved reset telemetry and are omitted from
reference comparisons (8 exclusions, **15,992 reference decisions**). All
16,000 own-actor action replays and actual command mappings pass. The teacher
is a reference controller, not an optimal physical recovery oracle.

**Road/lap alignment:** Each scored frame is projected onto the original fitted
centre line and assigned to one of 24 equal arc-index bins. Lap labels use the
recorded lap counter, not a 250-step cutoff. Every road bin and normal pass is
included. First recorded lap 0 may be partial in general; here both actors have
coverage in all 24 bins on all four layouts. Equal weights are applied per
observed lap/bin and then per layout; correlated frames are not independent
statistical trials. Physical speed is retained as a covariate, not controlled
or used as an actor input.

Candidate-minus-baseline CTE after this alignment is **-0.02156** for first
recorded laps (negative on all four layouts), **+0.01059** for later laps
(positive on 3/4). The first-250-step result is therefore not merely an
artifact of comparing different road coverage. As a coverage sensitivity
check, matching only complete later laps **1..8**, all 24 bins, produces paired
deltas **+0.03081, +0.00346, +0.00133, +0.01251**, mean **+0.01203**. This latter
estimate excludes incomplete final laps without filtering outcomes. Both
estimates retain the first-lap/later-lap tradeoff; neither establishes a
population effect from four layouts.

**Cross replay:** Both actors are queried on both recorded trajectory sources,
with the actual preceding commands for that source. Mapped steering MAE against
the same true-state teacher is averaged equally over layouts:

| Recorded input source | Phase | Original actor | Candidate actor |
| --- | --- | --- | --- |
| Original trajectories | First recorded lap | 0.08558 | 0.06980 |
| Original trajectories | Later laps | 0.04754 | 0.04840 |
| Candidate trajectories | First recorded lap | 0.07828 | 0.07084 |
| Candidate trajectories | Later laps | 0.05207 | 0.05482 |

Candidate improves first-lap agreement on both sources, but worsens later-lap
agreement on both. On original-source later laps, its error increases in 3/4
layouts; on candidate-source later laps it increases in 4/4. Thus later-lap
regression is visible even on the original actor's recorded inputs. The evidence
does not support attributing it solely to the candidate's command history.
Images, geometry, histories and teacher labels all differ between source
trajectories; the table is fixed-history single-decision replay, not a recursive
rollout or a causal decomposition of the closed-loop CTE difference.

**Recovery counts versus persistent error:** On original-source later laps,
missed teacher unwinds drop **27/163 -> 11/163**, while the mean per-layout
maximum absolute 20-step mean signed steering error increases **0.13911 ->
0.15750**. On candidate-source later laps, the corresponding counts are
**13/157 -> 5/157**, and persistent error **0.17121 -> 0.19547**. Fewer missed
unwinds do not establish better continuous tracking. All persistence windows
are contiguous within an episode/subset and never bridge missing segments.

**Both output functions changed:** Given identical original-source later-lap
inputs, candidate-minus-original mean throttle is **+0.001119**, with mean
absolute change 0.003421. On candidate-source inputs the mean difference is
+0.001067. Mapped steering absolute changes are about 0.01365/0.01350.
Later-lap steering is rate-limited on roughly 57% of decisions. These measurements
show that the intervention was not isolated to steering; they do not establish
that throttle or rate limits caused the observed tracking differences.

**Dynamic-information probe decision:** The recorded trigger required candidate
later-lap MAE to improve on original trajectories but worsen on its own
trajectories (macro means and at least 3/4 layout signs each). This did not occur:
later-lap MAE worsens on both sources. No probe is fitted. The current comparison
already exposes a cruise-retention problem under unchanged original inputs;
input observability can still limit both models, and its adequacy is not proved.

**Evidence and validation:** `report.json`, `cross_replay.csv`,
`all_passes_route_bins.csv`, `paired_route_bins.csv`, `action_changes.csv`,
`diagnostic_arrays.npz`, `route_comparison.png`, source snapshots/hashes,
`audit.json` and `validation.json` preserve the result. An initial route
aggregation omitted the trajectory-source filter and yielded identical values;
the sanity check exposed this, the source filter was corrected, and the initial
report is retained separately. `validate.py` independently recomputes every
source-specific route cell and macro replay metric from CSVs, computes the
complete-lap sensitivity check, verifies recorded scoring means and unchanged
input/model/runtime source hashes. All **five relevant mapping/paired-evaluation
tests pass**; the full previously passing 88-test suite was not repeated because
production code is unchanged. Whitespace checks pass.

**Reason ranking and next minimal experiment:** The evidence prioritizes a
first-lap versus later-lap training/selection tradeoff and continuous-error
retention over an immediate sensor or network change. Trajectory-distribution
changes and throttle/steering coupling remain contributors to investigate, not
isolated causes. Preserve the original baseline and do not automatically fit
again. `next_experiment_proposal.json` describes an unexecuted A/A calibration:
four fresh layouts, two identical-original actor runs per layout, 2,000-step
cap, the same shared-scene/reset/warmup protocol. It measures run-to-run
variability before interpreting another small-effect change; four pairs are
only a descriptive calibration and do not define acceptable degradation. A
subsequent steering-only intervention with the original throttle function could
isolate joint-output changes if warranted. Neither experiment is launched here.
