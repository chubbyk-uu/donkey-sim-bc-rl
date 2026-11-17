# Runtime protection

**Student input requirement (2026-11-16):** Only camera observations, wheel
encoder/IMU measurements, and the controller's own command history are permitted
as student control inputs. The geometry-head + frozen-teacher runs below still
consume simulator velocity truth and are **privileged-motion reference runs**,
not compliant student benchmarks. Historical 23/24 completion must not be reported
as camera-only or encoder/IMU-only performance. Any future encoder/IMU use requires verification of the actual sensor path.
The accepted camera-only student uses no such sensor feed. See the sensor audit in the
[experiment log](../archive/experiments/12-recovery-and-sensors.md#stage-314-sensor-source-audit-and-student-input-requirement-2026-11-16).

Normal RL training/evaluation entry points support these opt-in flags:

```bash
--low-jitter-sync --tcp-nodelay --observation-timeout 0.10
```

Low-jitter sync uses sequence-based frame wakeups and coalesces queued telemetry.
TCP_NODELAY disables Nagle buffering on the client socket. Observation timeout is
in **wall seconds measured with a monotonic clock**, requires low-jitter sync,
and is disabled by default (`0`). The initial `0.10` threshold is an operational
stop threshold, not a measured controller safety boundary. The disabled default
applies to legacy RL entries; dedicated student evaluation/collection entries
explicitly enable low-jitter sync, TCP_NODELAY and a 0.10-second timeout.
These flags do not
change model weights, throttle limits, or steering limits. Existing entry-point
defaults remain unchanged; do not compare results across different runtime flags
without recording the difference.

For a privileged-motion diagnostic reference using the original live vision head
and frozen 60k teacher, the single-condition entry below enables sync. This
command does not satisfy the student input requirement above:

```bash
python tools/eval_teacher_tolerance.py --mode vision_baseline_runtime \
  --layouts 1 --max-episode-steps 2000 --tcp-nodelay \
  --observation-timeout 0.10 --save-captures \
  --output-dir "logs/privileged_guard_check_$(date +%Y%m%d_%H%M%S)"
```

Use a new output directory for each run. Keep Windows Donkey Sim at
`timeScale=2`, FPS limit `120`, random light and trees enabled, cones disabled.
Keep display/window state and background load consistent; for unattended
evaluations, disconnect remote streaming without locking or minimizing the
simulator. This is a precaution, not a finding that remote desktop caused stalls.

Missing observations or an unconsumed stale frame trigger a best-effort zero
throttle/full handbrake command and raise `ObservationTimeoutError`. The run
aborts; late frames cannot resume control until an explicit reset. Recorded
evaluations save `runtime_failure.json` and mark the run failed, retaining
completed episodes separately. Such runs must not be reported as complete
model-only evaluations or silently removed from reliability counts.

The guard operates in the WSL client while waiting for observations. Socket
acceptance does not acknowledge braking in Unity. It cannot protect against
the whole WSL VM freezing, a blocked policy/encoder before observe is called,
or an unreachable/hung simulator. A command-age watchdog inside the Windows
simulator is needed for those cases; this repository contains no Unity server
source, so no server-side watchdog has been installed. Scene-reload deadlines
and collection-rate measurements use monotonic clocks to tolerate wall-clock
corrections.

For training-data collection that specifically supplements negative offsets,
`tools/eval_teacher_tolerance.py --mode round_b_collect` also accepts
`--recovery-direction negative`. This retains the existing bounded steering
intervention, unchanged teacher throttle/control limits, and full recovery
labels; the default remains balanced directions. Collection layouts are
training data and must stay separate from final evaluation layouts.

The accepted student phase is complete; the diagnostic and collection commands above describe available tooling, not scheduled work.
