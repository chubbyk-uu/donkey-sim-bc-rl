# Donkey Simulator BC/RL

Camera-based driving in Donkey Simulator: behavioral cloning, reinforcement
learning, and a camera-only student distilled from a state teacher. Python runs
in WSL2 against the Windows simulator.

[中文说明](README-CN.md) · [Documentation](docs/README.md) · [Current status](docs/status.md)

## Tracks and representative models

The project covers three simulator environments. Models were trained for their
respective tracks; the latest student has only been validated on the generated loop.

| Track | Environment ID | Work covered |
| --- | --- | --- |
| Generated open road | `donkey-generated-roads-v0` | Human-demonstration BC; early VAE + SAC; intersections/generalization probes |
| Generated closed loop | `donkey-generated-track-v0` | Fixed-light VAE + SAC; DINOv2/ResNet; random-light/tree-shadow training; teacher and student |
| Mountain loop | `donkey-mountain-track-v0` | Separately trained DINOv2 + SAC for hills and right-lane driving; earlier ResNet attempts |

Representative retained models and historical results:

| Scenario / role | Model and checkpoint | Result / usage |
| --- | --- | --- |
| Random-light loop with tree shadows | `visual_student_no_motion_v1` + `dinov2_geometry_15d` | Current camera-only baseline; 24/24 at 2,000 steps and 10/10 at 3,000 |
| Fixed-light loop | `rl_loop_vae_sac_fixedlight_v3_h80` 90k + `vae_loop_cones_fixedlight_v1` | Primary VAE policy; later recheck 5/5 at 3,000, speed 2.826 m/s, mean absolute CTE 0.311 |
| Fixed-light loop, alternate checkpoint | `rl_loop_vae_v15` **112488** + the same VAE | Retained long-evaluation policy; 5/5 at 3,000; later 132488 is reference only |
| Mountain | `rl_dinov2_mountain_v2` **40k** | Primary Mountain policy; 5/5 at 2,000, speed 2.093 m/s, mean lane-relative absolute CTE 0.577 |
| Random-light/tree-shadow loop, visual SAC | `rl_loop_dinov2_randomtree_v2` 50k | Earlier RL baseline; 31/40 at 2,000 in the standard-setting recheck |
| Loop with changing light | `rl_loop_dinov2_v8` 30k | Frozen DINOv2 + SAC backup; plain-light variation tests passed, tree-shadow robustness was insufficient |
| Fixed-light loop, ResNet backup | `rl_loop_vae_sac_resnet_v4_notrees` 50k | Retained visual SAC branch; requires `--encoder-crop-top 40` |
| Open-road BC, regression | `bc_nvidia_slow_006_flip` | Nvidia-style CNN; 3/3 reached 2,000 with steering scale 1.8 |
| Open-road BC, categorical | `bc_official_categorical_curve_aug_balanced_v1` | 11-bin steering classifier with class balancing; uses steering scale 1.4 and smoothing 0.1 |
| Privileged state teacher | `rl_loop_sim_state_ts2_sdiff01_v1` 60k | True-state SAC reference for distillation and diagnosis |

These are separate historical batches with different tracks, appearance,
termination rules and step caps. They are not a paired leaderboard. Open-road BC
uses a different CTE boundary from the loop. Mountain measures error relative to
`cte_target=3.5`, rather than raw CTE zero. See the [model catalog](docs/guides/models.md)
for exact weight paths, backup checkpoints and command requirements.

The early open-road **Raffin VAE + SAC** branch reached near the route end in
training, but did not reliably handle intersections on another generated road.
Its policy and VAE weights are no longer present in Git/LFS or the working copy;
the [historical record](docs/archive/experiments/02-generated-road-and-early-loop.md)
is retained. The BC, loop and Mountain artifacts listed above remain versioned.

Results: [BC](docs/archive/experiments/01-behavioral-cloning.md),
[fixed-light loop](docs/archive/experiments/03-fixed-light-loop.md),
[v15 recheck](docs/archive/sessions/session_2026-05-27_loop_followup.md),
[Mountain](docs/archive/experiments/04-mountain.md),
[encoder comparisons](docs/archive/experiments/05-encoders-and-randomization.md),
[November baseline rechecks](docs/archive/experiments/07-vision-and-runtime-probes.md).

## How the approaches relate

BC learns commands from human demonstrations. Visual SAC learns control through
simulator rewards using a track-specific VAE or a frozen DINOv2/ResNet encoder.
The state teacher uses privileged simulator state, then supplies offline action
labels and compatible actor weights for the camera-only student. The student's
actor is initialized from the teacher, not the old visual-SAC actor. See
[architecture and lineage](docs/guides/student.md) for the input and training details.

## Current focus: camera-only student

**The random-light loop student phase is complete.** Use
`visual_student_no_motion_v1` for the generated
loop with random lighting and tree shadows. Its driving inputs are camera-derived
geometry and its own previously issued commands. No true velocity, pose, CTE,
map, wheel encoder or IMU enters the student.

| Separate evaluation batch | Completed | Steps/episode | Total laps | Mean / max absolute CTE |
| --- | --- | --- | --- | --- |
| 24 new random-appearance layouts | 24/24 | 2,000 | 216 | 0.1912 / 1.6282 |
| 10 additional new layouts | 10/10 | 3,000 | 140 | 0.1847 / 1.4232 |

Both batches had zero exits and runtime aborts. CTE termination is
`abs(CTE) > 2.0`; CTE is in simulator units, not metres. The longer batch averaged
2.540 m/s and took about 13.5 minutes. Random layouts vary light/trees/shadows;
road geometry is fixed. Results do not establish unseen-track transfer.

The bend-finetuned candidate is **not promoted**: first-segment gains did not
improve whole-episode paired CTE. Further training, A/A calibration and sensor
work are not scheduled. See [decisions](docs/status.md) and [portable results](docs/results/README.md).

## Quick start

Run from the repository root. Existing environment: `rlpy312` (Python 3.12).
New machines should first follow [setup](docs/guides/setup.md), including
PyTorch, the pinned upstream SDK and Git LFS.

```bash
conda activate rlpy312
git lfs pull
export DONKEY_SIM_HOST="$(ip route | awk '/default/ {print $3}')"
export DONKEY_SIM_PORT=9091
```

### Camera-only student: random-light loop

Windows simulator settings: generated loop, **timeScale 2, FPS 120, random light
on, trees on, cones off, useSeed off**. Metadata records saved menu values;
it cannot independently confirm running engine state.

```bash
python rl/eval_visual_student.py \
  --student models/visual_student_no_motion_v1/student.pt \
  --geometry-head models/dinov2_geometry_15d/proj15_lags_ridge.pt \
  --host "$DONKEY_SIM_HOST" --port "$DONKEY_SIM_PORT" \
  --episodes 3 --max-episode-steps 2000 \
  --output-dir "logs/student_smoke_$(date +%Y%m%d_%H%M%S)"
```

Use a new output directory each time. The student entry defaults to low-jitter
sync, TCP_NODELAY and a 100 ms observation-wait timeout. Scoring/termination/logs
may use truth; the actor does not. The first run may download/cache DINOv2.
For the longer check use `--episodes 10 --max-episode-steps 3000`.
Both actor and geometry-head `.pt` files are required; SAC `.zip` files are not
interchangeable with the student.

### VAE + SAC: fixed-light loop

Disable random lighting and trees; match cone appearance to the fixed-light
checkpoint. Both the VAE encoder and SAC policy are required. Run the retained
primary policy:

```bash
python rl/eval_loop_vae_sac.py \
  --env-id donkey-generated-track-v0 \
  --encoder vae --vae-model models/vae_loop_cones_fixedlight_v1/best.pt \
  --model models/rl_loop_vae_sac_fixedlight_v3_h80/sac_loop_vae_90000_steps.zip \
  --episodes 3 --max-episode-steps 2000 \
  --output-dir "logs/fixedlight_smoke_$(date +%Y%m%d_%H%M%S)"
```

For the v15 long-evaluation policy, replace `--model` with
`models/rl_loop_vae_v15/sac_loop_vae_112488_steps.zip` and use
`--episodes 5 --max-episode-steps 3000`, with the same VAE and appearance settings.

### DINOv2 + SAC: Mountain

Open the Mountain scene. Its right-lane reference is CTE 3.5, and climbing needs
the retained throttle range 0.2–0.7. This policy uses DINOv2, without a VAE:

```bash
python rl/eval_loop_vae_sac.py \
  --env-id donkey-mountain-track-v0 --encoder dinov2_vits14 \
  --model models/rl_dinov2_mountain_v2/sac_dinov2_vits14_40000_steps.zip \
  --cte-target 3.5 --max-cte-error 2.5 --min-throttle 0.2 --max-throttle 0.7 \
  --episodes 5 --max-episode-steps 2000 \
  --output-dir "logs/mountain_smoke_$(date +%Y%m%d_%H%M%S)"
```

BC evaluation uses its own entrypoints: [regression and categorical commands](docs/archive/experiments/01-behavioral-cloning.md).
Other visual-SAC branches use the [model-specific commands and settings](docs/guides/models.md#legacy-evaluation-commands).

## Documentation

| Need | Guide |
| --- | --- |
| Environment, simulator connection, weights | [Setup](docs/guides/setup.md) |
| Old visual RL, teacher and student relationship | [Student architecture](docs/guides/student.md) |
| Fixed-light VAE, Mountain, backups, diagnostics | [Model catalog and commands](docs/guides/models.md) |
| Shared-layout comparison and result interpretation | [Evaluation](docs/guides/evaluation.md) |
| Sync, transport and timeout scope | [Runtime protection](docs/guides/runtime.md) |
| Historical training and safe resume | [Training recipes](docs/guides/training.md) |
| Past experiments and retired branches | [Experiment index](docs/experiment-log.md), [archive](docs/archive/README.md) |

## Repository and validation

`rl/` contains actors, encoders, environment wrappers and evaluators; `bc/` has
older demonstration baselines; `tools/` has collection/training/diagnostics;
`tests/` has simulator-free checks; `models/` has selected Git/LFS artifacts;
`docs/` has current guides, compact evidence and archives.

```bash
python -m pytest tests -q
```

The baseline passed 88 tests. Raw images, replay buffers, feature caches and full
`logs/` are local-only. Offline fits require those datasets restored; inference
can use the checked-in checkpoints. DINOv2's backbone/source are downloaded
separately; metadata records actual tensor hashes.
