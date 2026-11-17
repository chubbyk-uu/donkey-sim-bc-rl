# Student architecture and lineage

The accepted model is `visual_student_no_motion_v1`. The bend-finetuned student
is experimental and has not replaced it.

## Old visual RL, teacher and student

| Model | Driving observation | Training | Relationship |
| --- | --- | --- | --- |
| Historical visual SAC | 384 global image features + 40 self-command values | SAC reinforcement learning | Earlier baseline/data source; its actor weights do not initialize the student |
| State teacher | 18 privileged state/reference values + 40 commands | SAC, stable 60k checkpoint | Supplies offline actions and compatible actor weights |
| Camera-only student | 45 predicted geometry/history values + 40 commands | Supervised cloning/distillation | Accepted baseline |

Teacher state includes true simulator CTE/motion and heading/lookahead derived
from simulator pose plus the fitted track centre line. Historical visual SAC
also used true CTE/speed to calculate rewards; label/reward supervision is
distinct from privileged driving observations.

## Inference path

```text
images -> frozen DINOv2-S 4x4 grid (6144D) -> trained geometry head (15D)
current / lag-2 / lag-4 predicted geometry (45D) + own issued commands (40D)
       -> deterministic 85 -> 256 -> 256 -> 2 actor -> action mapping
```

The 15 geometry values are CTE/2, sine/cosine of relative heading, and six
lookahead points with two coordinates each. They are image predictions at
inference, not map queries or true CTE. The head already uses temporal context,
giving eight effective image steps. Both image-history levels and the 20-command
history reset causally per episode. No explicit speed estimate or encoder/IMU
feed is integrated.

`VisualGeometryEncoder.encode` accepts images; `VisualStudentActor.predict`
accepts only 85 features. True state remains isolated to scoring, termination
and logs. The live student does not load a teacher. Issued command history is
not an actual steering/wheel-speed sensor reading.

## Training and initialization

The geometry head learns image-to-geometry labels. The original student uses
39,020 training and 19,384 development decisions from separate layout groups.
A frozen teacher is queried offline on the frame's true state and identical
preceding issued history to label the next action. Student inputs and teacher
labels are separate archives.

Compatible teacher geometry/history input columns, hidden layers and output
weights initialize the student. Three true-motion input columns are discarded;
new historical-geometry columns start at zero. The adapted network is then
trained independently. It is not a teacher simply run with telemetry removed.
DINOv2 and the geometry head stay frozen. There are no SAC critic or RL updates
in this student fit.

This provides task-specific geometry, explicit per-frame action targets and
learned control initialization. These are concrete design differences, not an
isolated causal attribution of the driving gain. Control limits, spatial
features and runtime synchronization also changed across historical branches.
Time context may help infer dynamics; precise recovery of true motion is unproved.

See [actor/feature implementation](../../rl/visual_student.py) and
[original training code](../../tools/train_visual_student.py).

## Evidence and rejected follow-ups

The original student completed 24/24 at 2,000 steps and 10/10 at 3,000 steps.
See [status](../status.md) and [results](../results/README.md). The privileged
teacher also completed a historical 24-layout run. Those unpaired batches do
not establish student superiority. The road remains fixed while appearance
changes; unseen-track transfer has not been tested.

The bend fine-tune improved development teacher-action agreement but raised
whole-episode paired CTE. Cross replay shows worse later-lap action agreement
on both actors' recorded trajectories despite fewer missed teacher unwinds.
The candidate remains experimental; no further optimization is scheduled.

## Data requirements and entry points

- `tools/train_visual_student.py`: original fit; requires historical caches,
  capture provenance and teacher labels/model. Its default export already exists.
- `tools/collect_visual_student_recovery.py`: student-driven fresh collection;
  complete camera sequences and isolated pre-action state labels.
- `tools/finetune_visual_student_recovery.py`: one fit from a preregistered plan,
  maintaining whole-layout splits and the accepted baseline.

Raw logs/caches are not in Git. Future authorized fits need restored inputs
and new output/export paths. These entries are not self-contained new-clone
retraining commands. Inference only needs the actor, head and DINOv2 backbone.

Training evidence: [original student](../archive/experiments/13-camera-only-student.md)
and [fine-tune](../archive/experiments/14-student-fine-tune.md).
