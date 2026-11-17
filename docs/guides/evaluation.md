# Evaluation and interpretation

Run commands from the repository root. Current model choice: [status](../status.md).

## Student entries

`rl/eval_visual_student.py` loads only the student `.pt` format and matching
geometry head. See the [quick start](../../README.md#quick-start). It reloads
the scene each episode and saves exact 85D decision inputs plus raw actions.
Scoring truth does not enter the actor. Default tail capture is 120 images.

Shared-layout comparison of two students:

```bash
python rl/eval_visual_students_paired.py --host "$DONKEY_SIM_HOST" \
  --layouts 4 --max-episode-steps 2000 \
  --output-dir "logs/student_paired_$(date +%Y%m%d_%H%M%S)"
```

This runs the original and experimental candidate on each fresh scene, resetting
only the car between actors, balancing first-run order and clearing image/command
history each episode. Both stateless actors are prewarmed before connection.
All camera frames and decision inputs are saved as evaluation-only data. This
entry is available tooling, not a scheduled repeat or an endorsed candidate upgrade.

## Parameters and result meaning

| Setting | Camera-only student | Legacy visual SAC |
| --- | --- | --- |
| Steering difference | fixed 0.1 (executed delta about 0.2) | match checkpoint; typical historical value 0.2 |
| Throttle | 0.2..0.7 | match checkpoint |
| Low-jitter / TCP_NODELAY / timeout | enabled / enabled / 0.10 s | opt-in; timeout disabled by default |
| Encoder/motion truth | not actor inputs | depends on selected observation; `sim_state` is privileged |

A time-limit completion is reaching the step cap without termination. An exit
is a scoring/termination event, not a wheel-edge detector. CTE units are not
metres. A runtime timeout is a failed evaluation and must be reported separately,
not silently removed or counted as a model-only CTE failure. See [runtime scope](runtime.md).

Randomization varies lighting/trees/shadows on fixed road geometry. IDs denote
loads, not reproducible seeds. Compare equal step caps/settings and use shared
layouts for model comparisons. Treat layouts as the comparison units, not
individual correlated frames or laps. Offline action MAE, missed teacher unwinds
and one-step replay do not prove closed-loop improvement.

Portable evidence is in [results](../results/README.md); full input/telemetry/
image audits are preserved locally under each recorded run's `logs/` folder.
Use separate fresh output directories. Historical raw data are not shipped.

## Records and legacy SAC tooling

`python tools/compare_geometry_control.py --output-dir logs/geometry_control_compare_v1`
compares the original and two recovery-trained geometry heads offline using the
existing negative-recovery feature cache and the same held-out development
layouts. It requires no simulator or training. Results include mapped steering
errors, teacher-relative unwind/direction mismatches, continuous bias windows,
and counterfactual geometry repairs. Histories stay fixed to the recorded
post-command histories; these are action-agreement diagnostics, not recursive
closed-loop completion estimates. Use a new output directory for each analysis.

The legacy SAC loop evaluation scripts write `metadata.json`, `results.json`, `episodes.csv`,
and initial camera frames to a new directory under `logs/`. Metadata records actual
encoder tensor hashes, and `source/` preserves the evaluation source. Use `--output-dir`
to choose a **new** directory and `--save-telemetry` to also save stepwise JSONL
(actions, CTE, speed, lap counts, termination reason, and wall-clock timing).
Use `--capture-tail-steps 120` to retain the final 120 returned camera frames
(about 6 simulator seconds at nominal 20 Hz; wall time depends on timeScale)
of **both** failed and time-limited episodes. This
also enables telemetry. Images stay in a bounded RAM buffer during driving;
episode completion writes PNGs, a GIF and a step-indexed manifest. Captured
images are copies of the actual encoder input, rather than a later live frame.
Telemetry includes pose/velocity plus client policy, encoder and env-step timing.
The mapped steering command has no execution acknowledgement from Unity;
`env_step_seconds` includes telemetry waiting and encoding, and is not network RTT.

For a legacy SAC run (not a student `.pt` checkpoint), use:

```bash
python tools/analyze_control_failures.py logs/your_run
```

This produces per-episode time plots, frame contact sheets (when captured), and
JSON/CSV metrics. `tools/verify_policy_alignment.py logs/your_run` replays
the captured tail frames of failed episodes offline and checks that the policy
reproduces the recorded actions; `--brightness-gains 0.9 1.1` adds an offline
RGB-gain sensitivity probe. To test repeatability on the same layout, pass the same policy
twice with different labels via `--models`; these labels represent repeats, not
different trained models.

Non-finite oscillation periods are JSON `null`. Failures retain completed episodes
and mark metadata as failed. Layout numbers identify scene loads within a run;
they are not reproducible simulator seeds. Paired evaluation reloads even the
first layout, aborts on reload timeout, and shuffles model order with `--order-seed`.

The paired summary ends with the uncertainty over layouts. The primary metric is
laps per episode (fixed step cap, leaving the track ends the episode), reported
with a bootstrap 95% interval, followed by a Wilson interval on each completion
rate, a bootstrap interval on mean episode length, and for each pair of models the
per-layout laps difference and an exact sign test on the layouts where only one
completed. The same
numbers are stored under `statistics` in `results.json`;
`python tools/summarize_paired_eval.py logs/your_run` recomputes them for
older or aborted runs. Small layout counts give broad intervals; sample sizes should match the
new task rather than an automatic requirement to keep running experiments.

```bash
python rl/eval_loop_vae_sac.py --episodes 5 --max-episode-steps 3000 \
  --output-dir "logs/loop_fixedlight_check_$(date +%Y%m%d_%H%M%S)" --save-telemetry

python rl/eval_paired_randomized.py --encoder dinov2_vits14 \
  --model-dir models/rl_loop_dinov2_randomtree_v2 --steps 50000 \
  --layouts 12 --max-episode-steps 2000 --save-telemetry
```

The loop evaluator defaults to the primary VAE 90k policy. Match simulator
lighting/tree settings to the checkpoint; these settings are external to Python.
The simulator's Time scale and FPS limit also change results: the standard is
Time scale 2 with FPS limit 120 for both training and evaluation (about 1.9x
faster; see experiment-log §12). Evaluation metadata and training run configs
record the simulator's saved settings as `simulator_saved_settings`.
New loop training checkpoints embed encoder/crop/action configuration. Evaluation
and resume reject incompatible configurations; old checkpoints warn that metadata
is unavailable. Pretrained encoder names identify the model family; upstream
pretrained weights/source revisions are not yet pinned by this contract.
