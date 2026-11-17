# Experiment record index

The detailed chronological log is split into topic files under [archive](archive/README.md). Current decisions are in [status](status.md); usage is in [the documentation index](README.md). Historical next-step suggestions are not active tasks.

| Record | Scope |
| --- | --- |
| [Behavioral cloning](archive/experiments/01-behavioral-cloning.md) | 1. Behavioral Cloning |
| [Generated road and early loop](archive/experiments/02-generated-road-and-early-loop.md) | 2. Single Generated Road: VAE + SAC |
| [Fixed-light loop](archive/experiments/03-fixed-light-loop.md) | 6. Loop Track RL: Fixed-Light VAE Pipeline |
| [Mountain track](archive/experiments/04-mountain.md) | 6.7 Porting to other tracks: mountain-track cte and uphill quirks |
| [Encoders and appearance randomization](archive/experiments/05-encoders-and-randomization.md) | 6.11 Training defaults updated 2026-05-27 |
| [Historical operations and retired backlog](archive/experiments/06-historical-operations.md) | Retired operational advice/backlog |
| [Vision and runtime probes](archive/experiments/07-vision-and-runtime-probes.md) | 11. Runtime revalidation and control diagnostics (2026-11-12) |
| [Ground-truth teacher](archive/experiments/08-state-teacher.md) | 13. Ground-truth state policy: is perception the limit? (2026-11-12/13) |
| [Geometry perception and control](archive/experiments/09-geometry-and-control.md) | Stage 3.1: 15-dim geometry perception benchmark (offline, 2026-11-14) |
| [Geometry recovery collection and retraining](archive/experiments/10-geometry-recovery.md) | Stage 3.5: Round B — First Geometry Recovery Collection Batch (2026-11-16) |
| [Runtime timing and guard acceptance](archive/experiments/11-runtime-timing.md) | Stage 3.7: Remote Desktop / Host Timing Diagnosis — Active Baseline (2026-11-16) |
| [Recovery heads and sensor-source audit](archive/experiments/12-recovery-and-sensors.md) | Stage 3.11: Targeted Negative-Offset Recovery Data (2026-11-16) |
| [Camera-only student and initial validation](archive/experiments/13-camera-only-student.md) | Stage 3.15: First Camera-Only Control Student — Offline Only (2026-11-16) |
| [Student excursion diagnosis and fine-tune](archive/experiments/14-student-fine-tune.md) | Stage 3.19: Offline Diagnosis of Camera-Only Student Excursions (2026-11-17) |
| [Paired student evaluation and cross replay](archive/experiments/15-paired-diagnosis.md) | Stage 3.21: Shared-Layout Closed-Loop Screen of the Bend Student (2026-11-17) |
| [Longer original-student evaluation](archive/experiments/16-longer-student-evaluation.md) | Stage 3.23: Original Student, Ten Fresh Layouts at 3,000 Steps (2026-11-17) |

## Legacy section links

These anchors retain links to the former monolithic log. Detailed subsections now live in the topic files.

<a id="1-behavioral-cloning"></a>
- [1. Behavioral Cloning](archive/experiments/01-behavioral-cloning.md#1-behavioral-cloning)
<a id="2-single-generated-road-vae--sac"></a>
- [2. Single Generated Road: VAE + SAC](archive/experiments/02-generated-road-and-early-loop.md#2-single-generated-road-vae--sac)
<a id="3-loop-track-vae-data-random-light-v1-removed--superseded-by-6"></a>
- [3. Loop Track VAE Data (random-light v1, removed — superseded by §6)](archive/experiments/02-generated-road-and-early-loop.md#3-loop-track-vae-data-random-light-v1-removed--superseded-by-6)
<a id="4-loop-track-rl-failed-reward-branches"></a>
- [4. Loop Track RL: Failed Reward Branches](archive/experiments/02-generated-road-and-early-loop.md#4-loop-track-rl-failed-reward-branches)
<a id="5-loop-track-rl-safe_v2"></a>
- [5. Loop Track RL: safe_v2](archive/experiments/02-generated-road-and-early-loop.md#5-loop-track-rl-safe_v2)
<a id="55-loop-track-rl-frozen-resnet-encoder-v4"></a>
- [5.5 Loop Track RL: Frozen ResNet Encoder v4](archive/experiments/02-generated-road-and-early-loop.md#55-loop-track-rl-frozen-resnet-encoder-v4)
<a id="6-loop-track-rl-fixed-light-vae-pipeline"></a>
- [6. Loop Track RL: Fixed-Light VAE Pipeline](archive/experiments/03-fixed-light-loop.md#6-loop-track-rl-fixed-light-vae-pipeline)
<a id="7-practical-pitfalls"></a>
- [7. Practical Pitfalls](archive/experiments/06-historical-operations.md#7-practical-pitfalls)
<a id="8-design-recommendations"></a>
- [8. Design Recommendations](archive/experiments/06-historical-operations.md#8-design-recommendations)
<a id="9-current-data-and-model-inventory"></a>
- [9. Current Data And Model Inventory](archive/experiments/06-historical-operations.md#9-current-data-and-model-inventory)
<a id="10-open-todos"></a>
- [10. Open TODOs](archive/experiments/06-historical-operations.md#10-historical-backlog)
<a id="11-runtime-revalidation-and-control-diagnostics-2026-11-12"></a>
- [11. Runtime revalidation and control diagnostics (2026-11-12)](archive/experiments/07-vision-and-runtime-probes.md#11-runtime-revalidation-and-control-diagnostics-2026-11-12)
<a id="12-laps-baseline-with-intervals-state-probe-simulator-speed-up-2026-11-12"></a>
- [12. Laps baseline with intervals, state probe, simulator speed-up (2026-11-12)](archive/experiments/07-vision-and-runtime-probes.md#12-laps-baseline-with-intervals-state-probe-simulator-speed-up-2026-11-12)
<a id="13-ground-truth-state-policy-is-perception-the-limit-2026-11-1213"></a>
- [13. Ground-truth state policy: is perception the limit? (2026-11-12/13)](archive/experiments/08-state-teacher.md#13-ground-truth-state-policy-is-perception-the-limit-2026-11-1213)
<a id="stage-31-15-dim-geometry-perception-benchmark-offline-2026-11-14"></a>
- [Stage 3.1: 15-dim geometry perception benchmark (offline, 2026-11-14)](archive/experiments/09-geometry-and-control.md#stage-31-15-dim-geometry-perception-benchmark-offline-2026-11-14)
<a id="stage-32-empirical-closed-loop-tolerance-screen-with-1-step-delay-2026-11-14"></a>
- [Stage 3.2: Empirical closed-loop tolerance screen with 1-step delay (2026-11-14)](archive/experiments/09-geometry-and-control.md#stage-32-empirical-closed-loop-tolerance-screen-with-1-step-delay-2026-11-14)
<a id="stage-33-paired-d0-vs-d1--live-vision-4-layout-pilot-2026-11-15"></a>
- [Stage 3.3: Paired `d0` vs `d1` + live vision 4-layout pilot (2026-11-15)](archive/experiments/09-geometry-and-control.md#stage-33-paired-d0-vs-d1--live-vision-4-layout-pilot-2026-11-15)
<a id="stage-34-round-a--24-layout-paired-live-vision-eval--pre-registered-split-2026-11-15"></a>
- [Stage 3.4: Round A — 24-layout paired live vision eval & pre-registered split (2026-11-15)](archive/experiments/09-geometry-and-control.md#stage-34-round-a--24-layout-paired-live-vision-eval--pre-registered-split-2026-11-15)
<a id="stage-35-round-b--first-geometry-recovery-collection-batch-2026-11-16"></a>
- [Stage 3.5: Round B — First Geometry Recovery Collection Batch (2026-11-16)](archive/experiments/10-geometry-recovery.md#stage-35-round-b--first-geometry-recovery-collection-batch-2026-11-16)
<a id="stage-36-round-b--same-architecture-geometry-head-retraining-2026-11-16"></a>
- [Stage 3.6: Round B — Same-Architecture Geometry Head Retraining (2026-11-16)](archive/experiments/10-geometry-recovery.md#stage-36-round-b--same-architecture-geometry-head-retraining-2026-11-16)
<a id="stage-37-remote-desktop--host-timing-diagnosis--active-baseline-2026-11-16"></a>
- [Stage 3.7: Remote Desktop / Host Timing Diagnosis — Active Baseline (2026-11-16)](archive/experiments/11-runtime-timing.md#stage-37-remote-desktop--host-timing-diagnosis--active-baseline-2026-11-16)
<a id="stage-38-bounded-runtime-diagnosis--remote-session-tcp-and-wsl-clocks-2026-11-16"></a>
- [Stage 3.8: Bounded Runtime Diagnosis — Remote Session, TCP and WSL Clocks (2026-11-16)](archive/experiments/11-runtime-timing.md#stage-38-bounded-runtime-diagnosis--remote-session-tcp-and-wsl-clocks-2026-11-16)
<a id="stage-39-opt-in-runtime-protection-implementation-2026-11-16"></a>
- [Stage 3.9: Opt-in Runtime Protection Implementation (2026-11-16)](archive/experiments/11-runtime-timing.md#stage-39-opt-in-runtime-protection-implementation-2026-11-16)
<a id="stage-310-live-runtime-guard-acceptance-2026-11-16"></a>
- [Stage 3.10: Live Runtime-Guard Acceptance (2026-11-16)](archive/experiments/11-runtime-timing.md#stage-310-live-runtime-guard-acceptance-2026-11-16)
<a id="stage-311-targeted-negative-offset-recovery-data-2026-11-16"></a>
- [Stage 3.11: Targeted Negative-Offset Recovery Data (2026-11-16)](archive/experiments/12-recovery-and-sensors.md#stage-311-targeted-negative-offset-recovery-data-2026-11-16)
<a id="stage-312-common-layout-offline-control-comparison-2026-11-16"></a>
- [Stage 3.12: Common-Layout Offline Control Comparison (2026-11-16)](archive/experiments/12-recovery-and-sensors.md#stage-312-common-layout-offline-control-comparison-2026-11-16)
<a id="stage-313-four-layout-paired-screen-of-the-negative-supplement-head-2026-11-16"></a>
- [Stage 3.13: Four-Layout Paired Screen of the Negative-Supplement Head (2026-11-16)](archive/experiments/12-recovery-and-sensors.md#stage-313-four-layout-paired-screen-of-the-negative-supplement-head-2026-11-16)
<a id="stage-314-sensor-source-audit-and-student-input-requirement-2026-11-16"></a>
- [Stage 3.14: Sensor Source Audit and Student Input Requirement (2026-11-16)](archive/experiments/12-recovery-and-sensors.md#stage-314-sensor-source-audit-and-student-input-requirement-2026-11-16)
<a id="stage-315-first-camera-only-control-student--offline-only-2026-11-16"></a>
- [Stage 3.15: First Camera-Only Control Student — Offline Only (2026-11-16)](archive/experiments/13-camera-only-student.md#stage-315-first-camera-only-control-student--offline-only-2026-11-16)
<a id="stage-316-first-camera-only-student-live-smoke-and-replay-audit-2026-11-16"></a>
- [Stage 3.16: First Camera-Only Student Live Smoke and Replay Audit (2026-11-16)](archive/experiments/13-camera-only-student.md#stage-316-first-camera-only-student-live-smoke-and-replay-audit-2026-11-16)
<a id="stage-317-frozen-camera-only-student-four-layout-screen-2026-11-17"></a>
- [Stage 3.17: Frozen Camera-Only Student Four-Layout Screen (2026-11-17)](archive/experiments/13-camera-only-student.md#stage-317-frozen-camera-only-student-four-layout-screen-2026-11-17)
<a id="stage-318-frozen-camera-only-student-24-layout-evaluation-2026-11-17"></a>
- [Stage 3.18: Frozen Camera-Only Student 24-Layout Evaluation (2026-11-17)](archive/experiments/13-camera-only-student.md#stage-318-frozen-camera-only-student-24-layout-evaluation-2026-11-17)
<a id="stage-319-offline-diagnosis-of-camera-only-student-excursions-2026-11-17"></a>
- [Stage 3.19: Offline Diagnosis of Camera-Only Student Excursions (2026-11-17)](archive/experiments/14-student-fine-tune.md#stage-319-offline-diagnosis-of-camera-only-student-excursions-2026-11-17)
<a id="stage-320-fresh-student-driven-bend-data-and-one-control-fine-tune-2026-11-17"></a>
- [Stage 3.20: Fresh Student-Driven Bend Data and One Control Fine-Tune (2026-11-17)](archive/experiments/14-student-fine-tune.md#stage-320-fresh-student-driven-bend-data-and-one-control-fine-tune-2026-11-17)
<a id="stage-321-shared-layout-closed-loop-screen-of-the-bend-student-2026-11-17"></a>
- [Stage 3.21: Shared-Layout Closed-Loop Screen of the Bend Student (2026-11-17)](archive/experiments/15-paired-diagnosis.md#stage-321-shared-layout-closed-loop-screen-of-the-bend-student-2026-11-17)
<a id="stage-322-routelap-alignment-and-bidirectional-actor-replay-2026-11-17"></a>
- [Stage 3.22: Route/Lap Alignment and Bidirectional Actor Replay (2026-11-17)](archive/experiments/15-paired-diagnosis.md#stage-322-routelap-alignment-and-bidirectional-actor-replay-2026-11-17)
<a id="stage-323-original-student-ten-fresh-layouts-at-3000-steps-2026-11-17"></a>
- [Stage 3.23: Original Student, Ten Fresh Layouts at 3,000 Steps (2026-11-17)](archive/experiments/16-longer-student-evaluation.md#stage-323-original-student-ten-fresh-layouts-at-3000-steps-2026-11-17)
