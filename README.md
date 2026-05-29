# Donkey Simulator BC/RL

Behavioral cloning and reinforcement learning experiments for Donkey Simulator,
running from WSL2 against a Windows Donkey Simulator instance.

```
camera image → (crop) → encoder → SAC policy → steer / throttle
```

Detailed experiment history and design decisions: [docs/experiment-log.md](docs/experiment-log.md)

**Status (2026-05-29): stable milestone reached.** The project now has working BC
baselines, a single generated-road VAE+SAC baseline, fixed-light loop deployments,
a Mountain DINOv2 deployment, and a documented domain-randomization study for
random light + tree shadows. The strongest fixed-light loop policies still use a
track-specific VAE; DINOv2 is the best frozen encoder for lighting/domain changes.
The remaining open problem is robust perception under heavy random tree shadows:
frozen encoders plateau around partial success, so the next meaningful lever is a
task-adapted encoder (fine-tuned DINOv2, depth, or segmentation), not more reward
tuning.

---

## Quick Start

### 1. Python Environment

```bash
conda create -n rl311 python=3.11
conda activate rl311
pip install "stable-baselines3[extra]" gymnasium numpy pillow opencv-python tqdm tensorboard
```

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
  --batch-size 64 \
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
  --resume-replay-buffer models/rl_dinov2_mountain_new_v1/<best_checkpoint>_replay_buffer.pkl \
  --output-dir models/rl_dinov2_mountain_new_v2 \
  --timesteps 40000 --save-replay-buffer --device cuda
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
| `vae_raffin_v1` | Generated-road VAE encoder (single-road baseline only) |
| `rl_vae_sac_raffin_v1` | Generated-road SAC policy (single-road baseline) |
| `bc_nvidia_slow_006_flip` | Best BC regression model; needs ~1.8× steering scale at eval |
| `bc_official_categorical_curve_aug_balanced_v1` | Best BC categorical model; needs ~1.4× steering scale |
