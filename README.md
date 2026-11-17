# Donkey Simulator BC/RL

Camera-based driving in Donkey Simulator: behavioral cloning, reinforcement
learning, and a camera-only student distilled from a state teacher. Python runs
in WSL2 against the Windows simulator.

[中文说明](README-CN.md) · [Documentation](docs/README.md) · [Current status](docs/status.md)

## Current baseline

**This phase is complete.** Use `visual_student_no_motion_v1` for the generated
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
