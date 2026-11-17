# Donkey Simulator BC/RL

Behavioral cloning and reinforcement learning experiments for Donkey Simulator,
running from WSL2 against a Windows Donkey Simulator instance.

```
camera image → (crop) → encoder → SAC policy → steer / throttle
```

Detailed experiment history and design decisions: [docs/experiment-log.md](docs/experiment-log.md)
Chinese README: [README-CN.md](README-CN.md)

**Latest (2026-11-17):** The frozen camera-only student completed **24/24 fresh
random layouts, 48,000 steps / 216 laps**, using images and its own command
history only. Mean absolute CTE was 0.1912, maximum 1.6282, and mean speed
2.5168 m/s. Input/action/image-replay audits pass. Keep it as a candidate
baseline: this batch does not establish guaranteed reliability or superiority
to another policy. See experiment log Stage 3.18 and
`logs/visual_student_eval_24layouts_v1/`.

**Longer episodes (2026-11-17):** The unchanged original student also completed
**10/10 new random-light/tree-shadow layouts at 3,000 steps each**, 30,000 steps /
140 laps, with no exit or runtime abort. Mean absolute CTE is 0.18467, maximum
1.42320 (termination limit 2.0), and mean speed 2.54026 m/s. Report this batch
separately from the 2,000-step evaluation. See Stage 3.23 and
`logs/visual_student_eval_10layouts_3000_v1/`.

**Experimental candidate (2026-11-17):** One bend-focused fine-tune from this baseline
uses 3,000 new training decisions and 1,000 decisions from two separate new
development layouts. Bend steering MAE drops 37.8%, while original-development
MAE rises 1.4%; all preregistered offline gates pass. The new
`models/visual_student_bend_recovery_v1/student.pt` then completed 4/4 fresh shared
layouts alongside the original, but mean CTE rose 4.4%: startup improved 14.3%,
cruise worsened 7.6%. It fails the mean-CTE screen and is **not promoted**.
The 24/24 result above still belongs to `visual_student_no_motion_v1`.
See Stages 3.20–3.21 and `logs/visual_student_bend_paired_4layouts_v1/`.

Offline route-aligned cross replay (Stage 3.22) confirms a first-lap/later-lap
tradeoff: candidate later-lap teacher-action MAE increases on both actors'
recorded trajectories. Complete laps 1..8 matched over 24 road bins have
candidate CTE +0.01203 on average, positive on all four layouts. No additional
driving or training followed. Motion probes are deferred; the next proposal
is baseline A/A repeatability calibration. See `logs/visual_student_cross_replay_v1/`.

**Baseline snapshot (2026-11-13): random-layout baselines, feature probes, and ground-truth state ceiling completed.** The project has working BC
baselines, a single generated-road VAE+SAC baseline, fixed-light loop deployments,
a Mountain DINOv2 deployment, and a domain-randomization study for
random light + tree shadows. The strongest fixed-light loop policies still use a
track-specific VAE; DINOv2 is the best frozen encoder for lighting/domain changes.
At the standard simulator setting (Time scale 2, FPS limit 120), `rl_loop_dinov2_randomtree_v2` 50k averages 6.08 laps per episode (31/40 = 78% completed).
Offline feature probes show that 4x4 pooled DINOv2 patch tokens halve CTE error compared to the class token, and a diagnostic policy trained on simulator ground-truth state (`--encoder sim_state`) laps 32% faster (9.18s vs 13.47s, 9.42 laps, 21/24 completed) while still exhibiting control-side steering weave on the start straight.
See [experiment log §11–§13](docs/experiment-log.md#11-runtime-revalidation-and-control-diagnostics-2026-11-12).

---

## Quick Start

### 1. Python Environment

The existing local environment for this project is **`rlpy312` (Python 3.12)**.
Shell history confirms it was used to run `rl/eval_loop_vae_sac.py`. The old
`rl311` / Python 3.11 setup example is superseded by the environment below.
Keep `rlpy312` while this project is in use.

Activate the existing environment before training or evaluation:

```bash
conda activate rlpy312
```

On a new machine, create the environment and install dependencies:

```bash
conda create -n rlpy312 python=3.12
conda activate rlpy312
# first install PyTorch for the local CUDA version (see below), then:
pip install -r requirements.txt
```

`requirements.txt` pins the versions of the working environment, including the
upstream gym-donkeycar commit from step 2.

Install PyTorch with the command matching the local CUDA version from the official
PyTorch selector.

### 2. gym-donkeycar

Do **not** install the PyPI package; it targets the old Gym API and is incompatible:

```bash
# Wrong:
pip install gym-donkeycar

# Correct — install from upstream:
pip install git+https://github.com/tawnkramer/gym-donkeycar
```

Or clone locally for editable installs:

```bash
git clone https://github.com/tawnkramer/gym-donkeycar ../gym-donkeycar
pip install -e ../gym-donkeycar
```

### 3. Simulator Connection

The Windows Donkey Simulator talks to the WSL2 Python client over TCP 9091. The
Windows host IP changes per boot; derive it from the default route:

```bash
export DONKEY_SIM_HOST="$(ip route | awk '/default/ {print $3}')"
export DONKEY_SIM_PORT=9091
```

Persist these in `~/.bashrc`. Verify with `nc -vz "$DONKEY_SIM_HOST" 9091` once
the simulator is running.

### 4. Model Files

Model files are stored in Git LFS. After cloning:

```bash
git lfs pull
```

### 5. Smoke-Test A Deployment

With the simulator open on `generated_track`, run the primary fixed-light loop
policy:

```bash
python rl/eval_loop_vae_sac.py \
  --encoder vae \
  --vae-model models/vae_loop_cones_fixedlight_v1/best.pt \
  --model models/rl_loop_vae_sac_fixedlight_v3_h80/sac_loop_vae_90000_steps.zip \
  --episodes 3 --max-episode-steps 2000
```

---

## Current Deployments

### Recommended Models

| Scenario | Model | Encoder | Eval | Speed | CTE | Notes |
|---|---|---|:---:|---:|---:|---|
| Fixed-light loop (`donkey-generated-track-v0`) | `rl_loop_vae_sac_fixedlight_v3_h80` 90k | VAE | 3/3 trunc @ 2000 | 2.800 | 0.315 | Primary loop deployment |
| Fixed-light loop, longer cap | `rl_loop_vae_v15` 112k | VAE | 5/5 trunc @ 3000 | 2.797 | 0.356 | Co-primary / long-eval deployment |
| Mountain (`donkey-mountain-track-v0`) | `rl_dinov2_mountain_v2` 40k | DINOv2-S | 5/5 trunc @ 2000 | 2.093 | 0.577 | Primary mountain deployment |

### Research And Backup Models

| Scenario | Model | Encoder | Eval | Notes |
|---|---|---|:---:|---|
| Loop, lighting-robust frozen encoder | `rl_loop_dinov2_v8` 30k | DINOv2-S | 3/3 trunc @ 2000 | No VAE collection needed; slower than VAE but robust to random lighting |
| Loop, ResNet fallback | `rl_loop_vae_sac_resnet_v4_notrees` 50k | ResNet18 | 3/3 trunc @ 3000 | Historical backup; trained with `--encoder-crop-top 40` |
| Loop, random light + trees/shadows | `rl_loop_dinov2_randomtree_v2` 50k | DINOv2-S | ~50% random-layout trunc | Best frozen-encoder domain-randomized branch |
| Single generated road | `rl_vae_sac_raffin_v1/final_model.zip` | VAE | 1390+ steps near route end | Original Raffin-style non-closed-road baseline |
| Behavioral cloning | `bc_nvidia_slow_006_flip`, `bc_official_categorical_curve_aug_balanced_v1` | CNN | Historical | Useful baselines, both weaker than RL |

### Scenario Summary

| If you care about... | Use this first | Why |
|---|---|---|
| Fastest fixed-light loop driving | VAE loop deployment | Track-specific VAE is fastest and most centered when lighting is fixed |
| No VAE collection / lighting changes | DINOv2 loop backup | Frozen DINOv2 handles random lighting far better than VAE |
| Random light + random tree shadows | DINOv2 domain-randomized model | Best tested frozen encoder, but still capped around partial success |
| Mountain track | DINOv2 mountain v2 40k | Stable 5/5 truncate with right-lane CTE target |
| Original generated road | Raffin VAE+SAC final model | Good historical baseline; non-closed route ends before 3000 steps |
| BC comparison | Regression CNN first | More stable than categorical, though both need steering scale and lose to RL |

### Eval Commands

Primary fixed-light loop VAE:

```bash
python rl/eval_loop_vae_sac.py \
  --encoder vae \
  --vae-model models/vae_loop_cones_fixedlight_v1/best.pt \
  --model models/rl_loop_vae_sac_fixedlight_v3_h80/sac_loop_vae_90000_steps.zip \
  --episodes 3 --max-episode-steps 2000
```

DINOv2 loop alternative (no VAE collection needed, lighting-robust):

```bash
python rl/eval_loop_vae_sac.py \
  --encoder dinov2_vits14 \
  --model models/rl_loop_dinov2_v8/sac_dinov2_vits14_30000_steps.zip \
  --episodes 3 --max-episode-steps 2000
```

ResNet18 loop alternative:

```bash
python rl/eval_loop_vae_sac.py \
  --encoder resnet18 --encoder-crop-top 40 \
  --model models/rl_loop_vae_sac_resnet_v4_notrees/sac_resnet18_50000_steps.zip \
  --episodes 3 --max-episode-steps 2000
```

> The v4 ResNet checkpoint was trained with `--encoder-crop-top 40` (legacy); pass
> the same flag at eval. New ResNet runs should use `--encoder-crop-top 0`.

Mountain primary:

```bash
python rl/eval_loop_vae_sac.py \
  --env-id donkey-mountain-track-v0 \
  --encoder dinov2_vits14 \
  --cte-target 3.5 --max-cte-error 2.5 \
  --max-throttle 0.7 \
  --model models/rl_dinov2_mountain_v2/sac_dinov2_vits14_40000_steps.zip \
  --episodes 5 --max-episode-steps 2000
```

Mountain sim conventions: spawn cte ≈ 3.54 (right-lane center), yellow centerline =
cte=0. Use `--cte-target 3.5` so reward and termination both measure `abs(cte - 3.5)`.
Use `--max-throttle 0.7` for uphill segments.

Single generated-road baseline:

The original Raffin-style VAE+SAC baseline. Drives a fixed generated road (not a
closed loop — a good run reaches the end before 3000 steps). Kept as a
starting-point reference; not the current main target.

**These two model files are not in the repository or Git LFS** (they were never
committed and are absent from the current working copy), so this command needs
them restored or retrained first.

```bash
python rl/eval_vae_sac.py \
  --model models/rl_vae_sac_raffin_v1/final_model.zip \
  --vae-model models/vae_raffin_v1/best.pt \
  --episodes 3
```

Behavioral cloning baselines:

Two BC approaches were trained on slow-driving demonstration data:

**Regression CNN** (`bc_nvidia_slow_006_flip`):

```bash
python bc/eval_bc.py \
  --model models/bc_nvidia_slow_006_flip/best.pt \
  --steering-scale 1.8 \
  --episodes 3
```

**Official-style categorical** (`bc_official_categorical_curve_aug_balanced_v1`):

```bash
python bc/eval_bc_official_categorical.py \
  --model models/bc_official_categorical_curve_aug_balanced_v1/best.pt \
  --steering-scale 1.4 \
  --episodes 3
```

Both models output steering values that are too small for closed-loop driving —
the `--steering-scale` flag compensates at eval time (1.8× regression, 1.4×
categorical). Regression was more stable in closed-loop driving; categorical had
sharper corner response but was less consistent. Both are clearly outperformed by
the RL policies.

Training data lives in `data/slow_data_raw/` (6 generated-road tubs) and
`data/curated_cornering_v*_clean/` (augmented cornering subsets). See
[experiment log §1](docs/experiment-log.md#1-behavioral-cloning) for the full
BC experiment history.

---

## Training

### Loop Track — VAE + SAC (recommended)

Requires fixed lighting in the simulator (`randomlight` disabled) for data
collection, SAC training, and evaluation. The current fixed-light loop VAE was
trained from an 80k-frame right-lane dataset with lateral PID offsets:

```text
cte  0.0: 30k
cte +0.5:  8k    cte -0.5:  8k
cte +1.0:  8k    cte -1.0:  8k
cte +1.5:  7k    cte -1.5:  7k
cte +2.0:  2k    cte -2.0:  2k
```

Collect each bucket separately with `tools/collect_sim_frames.py`, then prepare
one manifest and cache from all bucket directories:

```bash
# 1. Collect images (example bucket; repeat for each cte target/count above)
python tools/collect_sim_frames.py \
  --env-id donkey-generated-track-v0 \
  --action-mode cte-pid \
  --cte-target 0.0 \
  --frames 30000 \
  --output-dir data/vae_raw/generated_track_loop_fixedlight_cte_0_30k

# 2. Build train/val manifest from all fixed-light buckets
python tools/prepare_vae_dataset.py \
  --source data/vae_raw/generated_track_loop_fixedlight_cte_0_30k \
  --source data/vae_raw/generated_track_loop_fixedlight_cte_p05_8k \
  --source data/vae_raw/generated_track_loop_fixedlight_cte_m05_8k \
  --source data/vae_raw/generated_track_loop_fixedlight_cte_p10_8k \
  --source data/vae_raw/generated_track_loop_fixedlight_cte_m10_8k \
  --source data/vae_raw/generated_track_loop_fixedlight_cte_p15_7k \
  --source data/vae_raw/generated_track_loop_fixedlight_cte_m15_7k \
  --source data/vae_raw/generated_track_loop_fixedlight_cte_p20_2k \
  --source data/vae_raw/generated_track_loop_fixedlight_cte_m20_2k \
  --output-dir data/vae_loop_cones_fixedlight_v1 \
  --dedupe

# 3. Convert the manifest to the cropped uint8 memmap cache used by train_vae.py
python tools/build_vae_cache.py \
  --manifest data/vae_loop_cones_fixedlight_v1/manifest.jsonl \
  --output-dir data/vae/cache_loop_cones_fixedlight_v1

# 4. Train VAE encoder
python rl/train_vae.py \
  --cache-dir data/vae/cache_loop_cones_fixedlight_v1 \
  --output-dir models/vae_loop_cones_fixedlight_v1 \
  --epochs 20

# 5. Train SAC (verified v3_h80 recipe: safe_v2 reward, hidden=80)
python rl/train_loop_vae_sac.py \
  --encoder vae \
  --vae-model models/vae_loop_cones_fixedlight_v1/best.pt \
  --output-dir models/rl_loop_new \
  --hidden-size 80 \
  --batch-size 64 --buffer-size 60000 \
  --gradient-steps-min 50 \
  --gradient-steps-cap 2000 \
  --alive-reward 1.5 \
  --speed-reward-weight 0.15 \
  --cte-speed-penalty-weight 0.25 \
  --reward-crash -10 --crash-speed-weight 5 \
  --min-throttle 0.2 --max-throttle 0.7 \
  --max-cte-error 2.0 --max-episode-steps 3000 \
  --timesteps 100000 --save-replay-buffer --device cuda
```

Evaluate at each 10k checkpoint; deploy the best deterministic-eval result (not the
final step). The 10-20k peak-valley-recovery cycle is normal — see
[experiment log §6.9](docs/experiment-log.md#69-lessons-from-the-fixed-light-loop-work).
Current CLI defaults are `--batch-size 128 --gradient-steps-cap 1000`; those newer
defaults are useful for fresh experiments but have not cleanly reproduced the deployed
v3_h80 result yet.

### Loop Track — DINOv2 / ResNet (no VAE collection)

```bash
python rl/train_loop_vae_sac.py \
  --encoder dinov2_vits14 \
  --output-dir models/rl_loop_dinov2_new \
  --hidden-size 256 \
  --batch-size 128 \
  --gradient-steps-min 50 \
  --gradient-steps-cap 1000 \
  --timesteps 60000 --save-replay-buffer --device cuda
```

Replace `--encoder dinov2_vits14` with `resnet18` for the ResNet variant.
DINOv2 is lighting-robust (robust to `randomlight`); VAE is not.

For robustness to **random light + trees/shadows**, enable both in the sim and add
scene-reload domain randomization: `--scene-reload-alpha 3 --scene-reload-kmin 200`
(reloads the scene periodically so the policy sees many random tree/shadow/light
layouts). Eval with `--scene-reload-every 1` (fresh layout per episode). DINOv2
reaches ~50% truncate on unseen random layouts this way — see
[experiment log §6.14](docs/experiment-log.md#614-domain-randomization-random-light--tree-shadows-on-loop-2026-05-28).
Frozen ResNet18 was tested under the same random light + tree-shadow setup and was
stopped at 20k: it never completed a lap and recent episodes stayed under 200 steps.
For shadowed-road robustness, DINOv2 is clearly stronger than ResNet18.

### Mountain Track — DINOv2

Mountain requires a two-stage approach: cold start with permissive reward, then
resume with stricter reward. Cold-starting with the strict reward fails (prevents
corner exploration). See
[experiment log §6.10](docs/experiment-log.md#610-mountain-track-dinov2-pipeline).

```bash
# Stage 1: cold start (permissive reward)
python rl/train_loop_vae_sac.py \
  --env-id donkey-mountain-track-v0 \
  --encoder dinov2_vits14 --hidden-size 256 \
  --batch-size 128 --buffer-size 50000 \
  --cte-target 3.5 --max-cte-error 2.5 \
  --min-throttle 0.2 --max-throttle 0.7 \
  --alive-reward 1.5 --speed-reward-weight 0.15 \
  --reward-crash -20 --crash-speed-weight 10 \
  --cte-speed-penalty-weight 0.25 \
  --gradient-steps-cap 1000 --gradient-steps-min 200 \
  --output-dir models/rl_dinov2_mountain_new_v1 \
  --timesteps 60000 --save-replay-buffer --device cuda

# Stage 2: resume from best cold-start checkpoint with stricter cte penalty
python rl/train_loop_vae_sac.py \
  --env-id donkey-mountain-track-v0 \
  --encoder dinov2_vits14 --hidden-size 256 \
  --batch-size 128 --buffer-size 50000 \
  --cte-target 3.5 --max-cte-error 2.5 \
  --min-throttle 0.2 --max-throttle 0.7 \
  --alive-reward 1.5 --speed-reward-weight 0.12 \
  --reward-crash -20 --crash-speed-weight 10 \
  --cte-speed-penalty-weight 0.30 \
  --gradient-steps-cap 1000 --gradient-steps-min 200 \
  --learning-rate 2e-4 --override-learning-rate \
  --resume-model models/rl_dinov2_mountain_new_v1/<best_checkpoint>.zip \
  --output-dir models/rl_dinov2_mountain_new_v2 \
  --timesteps 40000 --save-replay-buffer --device cuda
```

---

## Camera-only student

`models/visual_student_no_motion_v1/student.pt` is a separately trained
deterministic student actor. Its runtime inputs are **camera-derived features
and its own issued commands only**; no velocity, gyro, acceleration, pose, CTE,
map or simulator state is passed to the actor. No encoder/IMU is currently used.

The frozen original DINOv2 geometry head predicts 15 geometric features from
images. Current and lag-2/lag-4 predictions form 45 values, joined with 40 issued
command-history values: **85 inputs**. The geometry head already has temporal
context, so complete context extends back eight steps. Reset padding is causal
and identical in batch preparation and live inference. The existing 58D
privileged teacher is used for offline labels/weight initialization only.

The first fit uses 39,020 frames from successful full training-pool trajectories,
with 19,384 validation frames from nine separate development layouts. Failed
validation trajectories are retained. Its offline action-error gate passes,
and its first camera-only closed-loop smoke completed **1/1 fresh layout,
2,000 steps / 9 laps**, mean speed 2.506 m/s, mean absolute CTE 0.208 and maximum
0.938. This is not a reliability estimate. The improvement over its adapted
initialization is not evidence of superiority to the historical pure-vision RL
policy. See `logs/visual_student_no_motion_v1/{results,input_contract_audit}.json`.

The subsequent frozen four-layout screen on 2026-11-17 completed **4/4 layouts,
8,000 steps / 36 laps**, with mean absolute CTE 0.1925, maximum 0.9540 and mean
speed 2.5126 m/s. There were no `abs(CTE) >= 1` samples or runtime aborts.
All 8,000 decision inputs/actions pass replay and issued-history checks; 444
tail decisions reproduce from images/history with zero feature/action error.
All four maximum CTE values occur in the first 250 steps, so startup remains
an evaluation focus. Four successes do not establish a reliable completion
rate or an improvement over another policy. Weights remain unchanged and
experimental. See `logs/visual_student_screen_4layouts_v1/{audit,decision}.json`.

A separate frozen 24-layout evaluation subsequently completed **24/24 layouts,
48,000 steps / 216 laps**, mean absolute CTE 0.1912, maximum 1.6282, and mean
speed 2.5168 m/s, with zero runtime aborts. Ten layouts reached `abs(CTE) >= 1`
(60 frames total), then recovered and completed. Fifteen of 24 episode maxima
occurred in the first 250 steps; the largest excursion occurred during cruise.
All inputs/actions pass replay checks and 2,664 image-only tail decisions
reproduce with zero feature/action error. The nominal Wilson 95% completion
interval is 86.2%..100%, assuming independent trials. Keep this batch separate
from the earlier smoke/screen; it is not a paired superiority test. Weights are
unchanged, with no deployment replacement. Full artifacts:
`logs/visual_student_eval_24layouts_v1/`; experiment log Stage 3.18.

Offline inspection of all 14 large-offset excursions prioritizes student
control-response diagnosis: pre-event steering MAE against the frozen teacher
is 0.0912, retaining 0.0728 even after a one-step truth-geometry replacement.
Ten excursion peaks share one bend. These fixed-trajectory interventions do
not establish a physical cause or justify changing steering limits. That
diagnostic stage made no model or runtime change; see Stage 3.19 and
`logs/student_excursion_diagnosis_v1/`.

The subsequent single bend-recovery fit keeps the same 85D input contract,
frozen visual head and action limits. Fresh student-driven collection uses eight
new layouts (six training, two development), 500 steps each, saving all camera
frames. Privileged state is isolated for offline labels and sample weighting.
The fit uses the original 39,020 training decisions plus 3,000 fresh decisions;
the historical 24-layout evaluation is excluded. On the new development bend's
113 decisions, mapped steering MAE falls 0.06368 -> 0.03962 and missed teacher
unwind falls 14/26 -> 4/26. Original-development MAE rises 0.03987 -> 0.04042;
wrong-direction decisions rise 114 -> 126 despite fewer missed unwinds.
All seven offline gates pass; this is fixed-history action agreement, not
candidate driving performance. Preserve the original model as the live-tested
baseline. See Stage 3.20 and `logs/visual_student_bend_recovery_v1/`.

The subsequent four-layout paired screen runs each student on the same scene,
resetting only the car between actors, with balanced order and identical
stateless actor warmup before connection. Both complete 4/4, 8,000 steps / 36
laps. Candidate mean absolute CTE is 0.18945 vs original 0.18147 (+4.4%);
startup is 0.18254 vs 0.21290 (-14.3%), while cruise is 0.19044 vs 0.17698
(+7.6%). Every layout shows this startup/cruise tradeoff. The mean-CTE paired
difference interval includes zero, so this small screen does not establish a
population regression; it does fail the preregistered improvement condition.
Keep the original baseline. All 16,000 full-image/input decisions pass replay;
see Stage 3.21 and `logs/visual_student_bend_paired_4layouts_v1/`.

Training entry point (default output paths already contain the first run):

```bash
python tools/train_visual_student.py
```

It performs one bounded fit, preserves existing exports, and accepts new
`--output-dir` / `--export` paths for explicitly requested future runs. Student
inputs and privileged teacher labels are saved in separate NPZ files.

The dedicated evaluation entry point validates the input contract and frozen
head hash and resets both levels of temporal history. When a live test is
authorized and the simulator host is configured, use:

```bash
python rl/eval_visual_student.py --host "$DONKEY_SIM_HOST" --episodes 4 \
  --output-dir logs/visual_student_closed_loop_v1
```

This enables low-jitter sync, TCP_NODELAY and a 100 ms observation-wait timeout by
default. Ground truth is accessible to scoring/termination/logging only, not the
student feature or actor APIs. The actor is a distinct `.pt` format; it cannot
be replaced with a legacy SAC teacher `.zip`. No live test was launched as part
of the first offline implementation. The subsequently authorized single-layout
smoke and image-only replay audit are in `logs/visual_student_smoke_v1/`.

## Runtime timing protection

**Student input requirement (2026-11-16):** Only camera observations, wheel
encoder/IMU measurements, and the controller's own command history are permitted
as student control inputs. The geometry-head + frozen-teacher runs below still
consume simulator velocity truth and are **privileged-motion reference runs**,
not compliant student benchmarks. Historical 23/24 completion must not be reported
as camera-only or encoder/IMU-only performance. Establish and verify the actual
sensor path before further student training. See the sensor audit in the
[experiment log](docs/experiment-log.md#stage-314-sensor-source-audit-and-student-input-requirement-2026-11-16).

Normal RL training/evaluation entry points support these opt-in flags:

```bash
--low-jitter-sync --tcp-nodelay --observation-timeout 0.10
```

Low-jitter sync uses sequence-based frame wakeups and coalesces queued telemetry.
TCP_NODELAY disables Nagle buffering on the client socket. Observation timeout is
in **wall seconds measured with a monotonic clock**, requires low-jitter sync,
and is disabled by default (`0`). The initial `0.10` threshold is an operational
stop threshold, not a measured controller safety boundary. These flags do not
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
  --output-dir logs/vision_baseline_guarded_v1
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

## Evaluation records and safe resume

`python tools/compare_geometry_control.py --output-dir logs/geometry_control_compare_v1`
compares the original and two recovery-trained geometry heads offline using the
existing negative-recovery feature cache and the same held-out development
layouts. It requires no simulator or training. Results include mapped steering
errors, teacher-relative unwind/direction mismatches, continuous bias windows,
and counterfactual geometry repairs. Histories stay fixed to the recorded
post-command histories; these are action-agreement diagnostics, not recursive
closed-loop completion estimates. Use a new output directory for each analysis.

Both loop evaluation scripts write `metadata.json`, `results.json`, `episodes.csv`,
and initial camera frames to a new directory under `logs/`. Metadata records actual
encoder tensor hashes, and `source/` preserves the evaluation source. Use `--output-dir`
to choose a **new** directory and `--save-telemetry` to also save stepwise JSONL
(actions, CTE, speed, lap counts, termination reason, and wall-clock timing).
Use `--capture-tail-steps 120` to retain the final 120 returned camera frames
(about 6 seconds at 20 Hz) of **both** failed and time-limited episodes. This
also enables telemetry. Images stay in a bounded RAM buffer during driving;
episode completion writes PNGs, a GIF and a step-indexed manifest. Captured
images are copies of the actual encoder input, rather than a later live frame.
Telemetry includes pose/velocity plus client policy, encoder and env-step timing.
The mapped steering command has no execution acknowledgement from Unity;
`env_step_seconds` includes telemetry waiting and encoding, and is not network RTT.

Analyze a completed run with:

```bash
python tools/analyze_control_failures.py logs/<run_directory>
```

This produces per-episode time plots, frame contact sheets (when captured), and
JSON/CSV metrics. `tools/verify_policy_alignment.py logs/<run_directory>` replays
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
`python tools/summarize_paired_eval.py logs/<run_directory>` recomputes them for
older or aborted runs. With 12 layouts, 8/12 completions spans roughly 39%-86%, so
use 30+ layouts before comparing models.

```bash
python rl/eval_loop_vae_sac.py --episodes 5 --max-episode-steps 3000 \
  --output-dir logs/loop_fixedlight_check --save-telemetry

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

Loop training saves requested and effective parameters in `run_config.json`
(a resume into the same directory writes `run_config_from_<steps>_steps.json`
and appends to `monitor.csv` instead of replacing them).
`--timesteps` means **additional steps** when resuming. SAC inherits algorithm
parameters from its checkpoint; explicitly requesting a different inherited batch
size, buffer size, architecture, etc. now fails instead of silently ignoring it.
Use `--override-learning-rate` to change LR; the override persists in saved models.
Dynamic gradient updates apply only to episode-based training.

On resume, SAC updates are held until the replay buffer holds
`--resume-warmup-steps` transitions (default 2000; 0 disables). The loaded policy
drives during this warm-up. Without it, a resume with no buffer would run
`--gradient-steps-min` updates on the first short episode alone. The default is
a starting value and has not been tuned in the simulator.

With `--save-replay-buffer`, the policy checkpoint is written on schedule and the
buffer at the next episode end, so the control loop does not stall mid-episode.
The buffer file keeps the checkpoint's step number and also contains the rest of
that episode.

Replay buffers saved by the loop trainer have a `.pkl.json` sidecar with their
configuration and file hash. A reward/termination/encoder/action mismatch rejects
reuse: resume **without the buffer** when changing rewards, as in the Mountain
stage-2 example above. Historical runs reused old rewards, so this safer recipe
requires fresh validation rather than being an exact reproduction of those runs.
For historical buffers without metadata, `--allow-legacy-replay-buffer` explicitly
opts into unverified compatibility; it does not bypass a known metadata mismatch.
These contract checks apply to `train_loop_vae_sac.py`; the original generated-road
`train_vae_sac.py` entrypoint remains a legacy trainer.

Run simulator-free regression checks in `rlpy312`:

```bash
python -m pytest -q
```

---

## Repository Map

```text
rl/
  train_vae.py                        VAE encoder training
  train_vae_sac.py                    shared env / reward / encoder code
  train_loop_vae_sac.py               SAC training entrypoint (loop + mountain)
  eval_loop_vae_sac.py                evaluation entrypoint (reports weave metrics:
                                      |dsteer|, cte_std, steer/cte oscillation period)
  eval_paired_randomized.py           compare models on the SAME random layouts (removes
                                      layout-luck confound). --models label:zip:crop lets you
                                      mix dirs/crops in one paired run; also reports weave
  experiment.py                       argument/checkpoint/replay contracts and the
                                      evaluation record (JSON/CSV, telemetry, tail capture)
  track_state.py                      --encoder sim_state: simulator ground truth (position on the
                                      road, speed, road ahead) as the observation; a diagnostic
  eval_vae_sac.py                     legacy single generated-road evaluation
  vae.py, vae_dataset.py              ConvVAE model and memmap dataset

bc/
  train_bc.py                         BC regression CNN training
  eval_bc.py                          BC regression CNN evaluation
  train_bc_official_categorical.py    BC categorical training
  eval_bc_official_categorical.py     BC categorical evaluation

tools/
  collect_sim_frames.py               VAE frame collection from simulator
  prepare_vae_dataset.py
  build_vae_cache.py
  inspect_loop_replay_throttle.py     per-episode throttle/CTE analysis from replay buffer
  test_scene_reload.py                probe: reload the sim scene to confirm trees/light regenerate
  analyze_control_failures.py         plots/contact sheets/metrics from a recorded evaluation
  verify_policy_alignment.py          offline replay of captured frames through encoder + policy
  summarize_paired_eval.py            confidence intervals / paired tests for a paired-eval run
  probe_state_from_features.py        can cte / lateral drift be regressed from encoder features
                                      (CLS vs patch tokens) on unseen layouts; uses captured frames
  fit_grid_projection.py              PCA projection for --encoder dinov2_vits14_grid
  fit_track_centerline.py             recover the track centre line from telemetry (for sim_state)
  probe_teacher_action.py             offline probe: can image features + motion predict the sim_state teacher's action
  probe_depth_seg.py                  depth + Cityscapes segmentation probe on sim frames
  extract_cornering_segments.py       BC: cut cornering segments out of tubs
  diag_official_categorical.py        BC: categorical model diagnostics

tests/                                simulator-free regression tests (python -m pytest -q)

docs/
  experiment-log.md                   full experiment history and design decisions
  session_*.md                        per-session working notes
```

---

## Model Inventory

All models below are tracked in Git LFS. Replay buffers (`.pkl`) are excluded from the
repo — regenerate by resuming training with `--save-replay-buffer`.

### Loop Track

| Directory | Checkpoint | Encoder | Eval | Notes |
|---|---|---|:---:|---|
| `rl_loop_vae_sac_fixedlight_v3_h80` | `sac_loop_vae_90000_steps.zip` | VAE | 3/3 | **Primary** |
| `rl_loop_vae_v15` | `sac_loop_vae_112488_steps.zip` | VAE | 5/5 | Co-primary (lap bonus reward) |
| `rl_loop_vae_v15` | `sac_loop_vae_132488_steps.zip` | VAE | 2/5 | Reference only (faster laps, not deployable) |
| `rl_loop_vae_sac_fixedlight_v2_h64` | `sac_loop_vae_100000_steps.zip` | VAE | 3/3 | Secondary (hidden=64, more centered) |
| `rl_loop_dinov2_v8` | `sac_dinov2_vits14_30000_steps.zip` | DINOv2-S | 3/3 | Lighting-robust alternative |
| `rl_loop_dinov2_randomtree_v2` | `sac_dinov2_vits14_50000_steps.zip` | DINOv2-S | ~50% rand | Domain-randomized: robust to random light + trees/shadows ([§6.14](docs/experiment-log.md#614-domain-randomization-random-light--tree-shadows-on-loop-2026-05-28); the old "3/6" was measured under the §6.15 eval-clamp bug — ~4/6 at matched params) |
| `rl_loop_dinov2_randomtree_crop40_v1` | `sac_dinov2_vits14_130000_steps.zip`, `170000_steps.zip` | DINOv2-S | artifact | Crop40 randomtree probe; not promoted because crop benefit was not isolated ([§6.15](docs/experiment-log.md#615-crop--steering-clamp--cte-tolerance-probes-2026-05-29)) |
| `rl_loop_vae_sac_resnet_v4_notrees` | `sac_resnet18_50000_steps.zip` | ResNet18 | 3/3 | Backup, needs `--encoder-crop-top 40` |
| `rl_loop_sim_state_ts2_sdiff01_v1` | `sac_sim_state_60000_steps.zip` | `sim_state` | **24/24** rand | **Ground-truth teacher** (`--max-steering-diff 0.1`, `--track-centerline models/track_centerline/generated_track.npz`): 100% completion across 24 random-tree/light layouts (216 laps, `mean_abs_cte=0.158`, `max_abs_cte=0.770`; [§13](docs/experiment-log.md#13-separating-perception-from-control-with-a-ground-truth-track-state-2026-11-12)) |

VAE encoder shared by all VAE-based loop models: `vae_loop_cones_fixedlight_v1/best.pt`
New `--encoder vae` training checkpoints are written with the historical
`sac_loop_vae_*_steps.zip` prefix for compatibility with these artifacts.

### Mountain Track

| Directory | Checkpoint | Encoder | Eval | Notes |
|---|---|---|:---:|---|
| `rl_dinov2_mountain_v2` | `sac_dinov2_vits14_40000_steps.zip` | DINOv2-S | 5/5 | **Primary** |
| `rl_dinov2_mountain_v2` | `sac_dinov2_vits14_50000_steps.zip` | DINOv2-S | 5/5 | Backup (higher CTE) |
| `rl_dinov2_mountain_v3` | `sac_dinov2_vits14_40000_steps.zip` | DINOv2-S | 5/5 | Alternative; faster single-stage cold start (new-defaults validation, [§6.13](docs/experiment-log.md#613-new-defaults-validation-mountain-dinov2-cold-start-2026-05-28)) |
| `rl_dinov2_mountain_v1` | `sac_dinov2_vits14_*.zip` (6 ckpts) | DINOv2-S | — | Cold-start branch; kept for v2 reproducibility |
| `rl_loop_resnet_mountain_v1` | `sac_resnet18_90000_steps.zip` | ResNet18 | 1/3 | Historical; superseded by DINOv2 pipeline |

### Other

| Directory | Notes |
|---|---|
| `vae_loop_cones_fixedlight_v1` | Fixed-light loop VAE encoder (required for VAE-based loop models) |
| `vae_raffin_v1` | Generated-road VAE encoder (single-road baseline only). **Not in the repository/LFS.** |
| `rl_vae_sac_raffin_v1` | Generated-road SAC policy (single-road baseline). **Not in the repository/LFS.** |
| `bc_nvidia_slow_006_flip` | Best BC regression model; needs ~1.8× steering scale at eval |
| `bc_official_categorical_curve_aug_balanced_v1` | Best BC categorical model; needs ~1.4× steering scale |
