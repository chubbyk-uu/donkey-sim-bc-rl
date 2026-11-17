# Runtime timing and guard acceptance

> Historical record. Results, “current” choices and “next” plans below belong to the dated experiments, not the active work queue. See [current status](../../status.md) and [documentation index](../../README.md). Commands assume the repository root; raw `logs/` and `data/` artifacts are local-only unless explicitly noted.

### Stage 3.7: Remote Desktop / Host Timing Diagnosis — Active Baseline (2026-11-16)

At the user's request, pause further training and investigate whether remote
desktop activity contributes to timing stalls. No remote service was stopped,
no desktop session was disconnected, and no system/GPU settings were changed.

**Process identification correction:** An initial search for only `Sunlogin|Oray`
found the legacy `SunloginService` stopped and missed the active `AweSun.exe`
processes. Windows GPU-engine counters identified PID 15512 as `AweSun`, using
two VideoEncode engines (each roughly 2..3% in the initial snapshots), and
`AweSunService` was running. AweSun is the branding on the
[official English remote-desktop site](https://sunlogin.oray.com/en/).
PID 15512 was created at `2026-11-16 11:06:26 +08:00` and was still alive during
diagnosis, covering the earlier evaluation's time window. This establishes
process presence, not continuous remote-streaming activity throughout that run.
The attempted Service Control Manager event query returned no matching state
events, so no historical stream-on/off conclusion is drawn from it.

`tools/profile_remote_timing.py` adds a read-only diagnostic:
- Uses the same recorded RGB frames and unchanged baseline geometry head at a
  paced 25 ms period, including preprocessing, DINOv2, temporal head, CPU output,
  and CUDA synchronization. It does not drive or query the simulator.
- Independently samples selected Windows process CPU/memory counters and raw
  per-engine GPU utilization approximately once per second. Records UTC times,
  sampling cost, wake lateness, and skipped benchmark slots.
- Never stops services or changes remote settings. The phase label is supplied
  by the operator; observed remote VideoEncode activity is logged independently.

```bash
python tools/profile_remote_timing.py \
  --output-dir logs/remote_timing_diagnosis_v1/remote_active \
  --phase remote_active --duration 120
```

**Observed active baseline:** 4,801 perception calls; P50/P95/P99/max
`4.22 / 5.18 / 5.88 / 9.07 ms`; zero calls above 25 ms and zero skipped slots.
Wake-lateness P99/max was `0.197 / 0.250 ms`. All 122 Windows samples contained
AweSun VideoEncode activity; the maximum observed utilization of an individual
remote encode engine was 3%. The host sampler reported no errors; its wall-time
cost P50/P95 was `353 / 358 ms` per roughly 1-second sample, asynchronously from
perception. These coarse GPU counters cannot attribute individual millisecond
stalls. Raw data and `summary.json` are in the run directory above.

**Limits / pending comparison:** Remote streaming and fast perception coexist
in this 2-minute recorded-frame benchmark. It neither establishes remote desktop
as the cause of earlier stalls nor excludes a contribution under live rendering.
Unity camera capture, TCP reception, actual driving, and historical host counters
were not measured by this test. The subsequent disconnected/reconnected driving
comparisons and their limits are recorded in Stage 3.8 below.

### Stage 3.8: Bounded Runtime Diagnosis — Remote Session, TCP and WSL Clocks (2026-11-16)

All driving below uses the original vision head and frozen 60k teacher, low-jitter
sync enabled, steering difference limit 0.1, and reversal-unwind scale 1.0. No
teacher retraining or candidate promotion was performed. Artifacts are under
`logs/remote_root_cause_v1/`.

| Run | Result | Timing evidence |
| --- | --- | --- |
| `gc_paired_remote_disconnected` | Stock and deferred GC both 3/3 x 2,000 steps | Encoder max 10.97 / 12.48 ms; observed GC max 2.09 ms, no generation-2 collections |
| `continuous_remote_transition` | 10,000/10,000 steps, 48 laps | Despite its name, streaming stayed off throughout; observation interval max 26.39 ms |
| `long_natural_gc` | Failed at step 3,709, 17 laps | Streaming active from the first host sample; observation interval 234.51 ms at step 3,336, encoder max 9.05 ms, GC max 1.15 ms |
| `sync_stage_remote_connected` | 2/2 x 2,000 steps | Fine receive/decode/publish instrumentation installed; max interval 37.98 ms, earlier 234 ms stall not reproduced |
| `transport_paired_remote_connected` | TCP default 1/2; TCP_NODELAY 2/2 | Two fresh layouts with reversed condition order; same model and controls |

The disconnected runs total 22,000 steps. This sequential comparison makes remote
activity a useful lead, not a controlled demonstration that AweSun caused the
stall. In the connected failure, the 234 ms interval preceded the exit by 373
steps; neither a sole cause of that exit nor a Windows GPU-driver cause is proven.
The independent Linux heartbeat recorded no >3 ms wake delay around that event.
It uses the same monotonic clock, so it cannot alone exclude a VM pause that also
affects the clock. Later heartbeat records contain eight 3.0..4.4 ms delays.

**Observed transport issue:** The SDK does not set TCP_NODELAY. A socket snapshot
showed 184 queued send bytes, including 92 not yet transmitted, one unacknowledged
segment, and approximately 20 ms TCP RTT despite sub-millisecond ICMP ping. This
establishes small-command buffering; it does not establish the cause of the
earlier 234 ms interval. During the paired run, 9/469 sampled default-condition
socket snapshots contained unsent bytes (maximum 92), versus 0/250 sampled
TCP_NODELAY snapshots. Sampling started after the first NODELAY episode, so only
its second episode has queue coverage. Disabling Nagle did not materially change
the sampled approximately 19.5 ms RTT. The default-condition failure in this run
had no observation interval above 40 ms, so its outcome cannot validate a fix for
the earlier large stall. Two pairs are insufficient for a reliability claim.

Fine timings use `time.perf_counter()`. A message-sent timestamp means accepted
by the local socket, not transmitted or processed by Unity. Publish-to-return
frame age can include time before observe was called; actual observe handoff is
`return - max(published, observe_start)`.

**Clock check:** This machine uses the `tsc` clocksource; Python perf_counter and
monotonic both use CLOCK_MONOTONIC, while time.time uses CLOCK_REALTIME.
chronyd runs with `-x`, which disables clock adjustment despite the active NTP
service status. Inspected user/system timers and Windows scheduled tasks showed
no matching forced-time-sync task; this is not a complete audit of all host code.
Ordinary wall-clock steps do not step CLOCK_MONOTONIC
([Linux clock documentation](https://man7.org/linux/man-pages/man3/clock_settime.3.html)).
Frequency correction and virtual-machine timing anomalies are separate concerns.

`clock_probe_v1` sampled Linux realtime, monotonic, raw and boottime 12,031 times
over 122 seconds, plus Windows UTC/QPC 1,094 times over 120 seconds. No Linux
clock increment differed from monotonic by more than 0.025 ms; raw-versus-monotonic
elapsed drift was 0.043 ms. Windows UTC-versus-QPC elapsed difference was 0.722 ms.
This short non-driving sample did not reproduce the reported periodic clock
jump. Windows-to-WSL arrival offsets include PowerShell startup and pipe latency
and must not be interpreted directly as clock drift.

A regression test simulates unchanged timestamps, a one-minute backward step,
and a one-hour forward step while receiving real socket-pair telemetry. Enabled
low-jitter observe continues to receive each fresh frame. This verifies the
application's sequence-based wakeup behavior, not WSL or Unity clock correctness.
GC callback records and the background writer are also covered; all 54 tests pass.

**Hard stopping rule requested by the user:** One final original-model driving
run, at most 4,000 steps and a 170-second process limit, with a concurrent
132-second clock probe. No further conditions, retraining, or automatic
extension. If a stall cannot be attributed with concurrent evidence, stop and
report the root cause as unconfirmed. See `stopping_rule.json`; final artifacts
are `clock_live_stopcheck_v1/` and `clock_live_stopcheck_clocks_v1/`.

**Final stop check and decision:** The single final episode completed 4,000 steps
and 19 laps in 100 seconds. Maximum observation interval was 34.54 ms, with no
interval above 40 ms. Maximum |CTE| was 1.949, so completion does not demonstrate a
large safety margin. Concurrent clock sampling covered the full driving run:
13,023 Linux samples over 132 seconds, with no >1 ms discrepancy between clock
increments; the largest was 0.037 ms. Raw-versus-monotonic elapsed drift was
0.048 ms. Windows UTC/QPC elapsed difference was 0.566 ms across 1,186 samples.
Neither the large stall nor a clock jump reproduced. The investigation stopped
at the agreed limit. The historical 234 ms stall's root cause remains
**unconfirmed**; do not label AweSun, WSL clock correction, GC, or Nagle as its
proven cause. Preserve the original model/control baseline, keep TCP_NODELAY as a
diagnostic condition, and do not launch additional diagnosis or retraining
automatically. Machine-readable closure: `bounded_diagnosis_result.json`.

### Stage 3.9: Opt-in Runtime Protection Implementation (2026-11-16)

Following the bounded diagnosis, the user authorized preventive implementation,
not further root-cause experiments or teacher training:

- Shared training/evaluation flags now include `--tcp-nodelay` and
  `--observation-timeout SECONDS`. Existing defaults remain unchanged. The
  timeout requires low-jitter sync; invalid negative/non-finite values fail.
- Observation waiting uses a monotonic deadline. A missing or already-stale
  observation latches `ObservationTimeoutError`, requests zero throttle and full
  handbrake, and prevents subsequent control commands/late frames from silently
  resuming the run. Explicit reset clears the latch. The emergency send preserves
  the existing partial-packet buffer; it never discards half a JSON message.
- The request is best-effort. Diagnostics distinguish socket acceptance from
  acknowledged execution, and do not claim that Unity actually braked. This is
  a client observe-wait guard, not an independent Windows/Unity watchdog; it
  cannot cover a whole-WSL freeze, a blocked policy/encoder before observe, or
  an unreachable/hung simulator. No Unity server source is present in this repo.
- Recorded evaluations save `runtime_failure.json` and a failed run status,
  retaining previous completed episodes. The custom live evaluator also flushes
  the interrupted episode's returned-step telemetry and records its context.
  Do not silently exclude runtime-aborted runs from reliability reporting.
- `vision_baseline_runtime` runs just the original live vision head and frozen
  teacher, with low-jitter sync and no reversal acceleration. Its layouts default
  to evaluation-only; it needs no residual-probe cache. Use explicit
  `--tcp-nodelay --observation-timeout 0.10` for the documented protected recipe.
  The 100 ms threshold is a starting operational cutoff, not a validated safe
  controller error budget. README documents the command and fixed simulator
  settings; no remote-service or display settings were changed automatically.
- Scene-reload timeouts (production and standalone diagnostic) and frame
  collection elapsed-time/rate calculations now use monotonic clocks.

**Validation:** All 66 tests pass. New tests cover real localhost TCP_NODELAY,
normal telemetry followed by silence and a brake request, a stale unconsumed
frame, the latched failure/reset behavior, disabled-guard compatibility, invalid
timeouts, wall-clock-independent scene-reload deadlines, separate runtime-failure
records, and the single-condition baseline configuration. CLI help exposes the
new flags/mode and `git diff --check` passes. No new live simulator experiment,
Windows-side brake execution test, or model training was run for this change.

### Stage 3.10: Live Runtime-Guard Acceptance (2026-11-16)

`logs/runtime_guard_acceptance_v1/` contains a bounded two-case smoke test using
the original live vision head and frozen teacher, timeScale 2/FPS 120, low-jitter
sync, TCP_NODELAY, and a 100 ms observation timeout. The ignored run wrapper
records the injected treatment; it does not change production defaults or weights.

- **Normal:** 400/400 steps in 10 seconds, one lap, max |CTE| 0.656, no timeout.
- **Injected missing observations:** Pause socket reception for 200 ms after the
  220th control message. The guard fired after 100.145 ms and aborted the run
  after 217 returned steps. `runtime_failure.json` reports the stop request
  flushed to the socket and keeps execution acknowledgement false.
- The wrapper observed subsequent telemetry for 0.7 seconds before normal error
  cleanup: speed fell from 2.344 m/s to approximately zero and telemetry sequence
  advanced by 24. This supports a physical stop in this test, without adding an
  execution-acknowledgement protocol or testing a whole-WSL freeze.

`acceptance_result.json` records the passed checks. Both cases retain the source
versions verified against their recorded hashes. No further runtime root-cause
experiments were launched.
