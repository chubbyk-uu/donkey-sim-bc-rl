# Current status and decisions

Last reviewed: 2026-11-17.

## Accepted phase result

The camera-only phase is complete in the tested generated-loop random-light /
tree-shadow setting. Keep `models/visual_student_no_motion_v1/student.pt` with
`models/dinov2_geometry_15d/proj15_lags_ridge.pt`.

| Independent batch | Completed | Step cap | Laps | Mean / maximum absolute CTE |
| --- | --- | --- | --- | --- |
| Original student, 24 layouts | 24/24 | 2,000 | 216 | 0.19116 / 1.62820 |
| Same student, 10 further layouts | 10/10 | 3,000 | 140 | 0.18467 / 1.42320 |

Both batches have zero CTE exits and runtime aborts; do not pool their different
step caps. Longer-batch mean speed is 2.54026 m/s. The fixed road's appearance
is randomized; layout IDs are scene loads, not reproducible seeds. Saved settings:
timeScale 2, FPS 120, randomLight=1, trees=1, cones=0, useSeed=0. These menu values
are not independent live-engine measurements.

The actor sees images and its own issued commands only. True telemetry is
scoring/termination/logging, with `abs(CTE) > 2.0` termination in simulator units.
No wheel encoder or IMU is integrated. All decision-input/action audits pass;
2,664 image-tail decisions from the first batch and 1,110 from the longer batch
reproduce with zero feature/action error after context exclusion.

See [portable results](results/README.md) and [student architecture](guides/student.md).

## Decisions retained

- Original student: accepted baseline, weights fixed.
- True-state teacher: offline-label/diagnostic reference, not camera-only driving.
- Geometry-head + teacher with true motion: privileged reference, including the
  historical 23/24 result. It is not the current compliant student.
- Recovery-trained geometry heads: not promoted; better perception error did not
  establish better driving.
- Bend-finetuned student: not promoted. Four shared-layout pairs completed for
  both actors, but candidate whole-episode mean CTE rose 4.4%.
- Route/lap cross replay retains first-lap benefit and later-lap degradation;
  fewer missed teacher unwinds did not preserve continuous steering error.
  Motion probes were deferred; current input sufficiency is not proved.

Evidence: [paired diagnosis](archive/experiments/15-paired-diagnosis.md) and
[longer evaluation](archive/experiments/16-longer-student-evaluation.md).

## Active scope and optional ideas

There is **no scheduled training, simulator experiment or upgrade**. The user
accepted the phase and requested documentation cleanup. Earlier “next” and
“TODO” text is historical unless confirmed here.

A/A calibration, isolated steering/throttle tests, motion probes, encoder/IMU
integration, new tracks and higher speed are optional future tasks requiring
new objectives. There is no requirement to produce another checkpoint.
These sampled results do not establish universal reliability, superiority over
the teacher or transfer to a new track/real car.

## Reproduction boundaries

The [model catalog](guides/models.md) distinguishes accepted, diagnostic and
experimental Git/LFS artifacts. Raw logs, images, caches and replay buffers are
local-only. Portable results retain original result hashes for verification.
Single-student evaluation does not prewarm the actor; paired student evaluation
prewarms both stateless actors before connection. Record this protocol difference.
