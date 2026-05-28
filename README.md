# Donkey Simulator BC/RL

Behavioral cloning and reinforcement learning experiments for Donkey Simulator,
running from WSL2 against a Windows Donkey Simulator instance.

```
camera image → (crop) → encoder → SAC policy → steer / throttle
```

Detailed experiment history and design decisions: [docs/experiment-log.md](docs/experiment-log.md)

---

## Current Deployments

### Loop Track (`donkey-generated-track-v0`)

| Model | Encoder | Eval | Speed | CTE | Max steps |
|---|---|:---:|---:|---:|---:|
| `rl_loop_vae_sac_fixedlight_v3_h80` 90k | VAE (fixed-light) | 3/3 trunc | 2.800 | 0.315 | 2000 |
| `rl_loop_vae_v15` 112k | VAE (fixed-light) | 5/5 trunc | 2.797 | 0.356 | 3000 |
| `rl_loop_vae_sac_resnet_v4_notrees` 50k | ResNet18 (frozen) | 3/3 trunc | 2.131 | 0.424 | 3000 |
| `rl_loop_dinov2_v8` 30k | DINOv2-S (frozen) | 3/3 trunc | 2.414 | 0.565 | 2000 |

Evaluate the primary VAE deployment:

```bash
python rl/eval_loop_vae_sac.py \
  --encoder vae \
  --vae-model models/vae_loop_cones_fixedlight_v1/best.pt \
  --model models/rl_loop_vae_sac_fixedlight_v3_h80/sac_loop_vae_90000_steps.zip \
  --episodes 3 --max-episode-steps 2000
```

Evaluate the DINOv2 alternative (no VAE collection needed, lighting-robust):

```bash
python rl/eval_loop_vae_sac.py \
  --encoder dinov2_vits14 \
  --model models/rl_loop_dinov2_v8/sac_dinov2_vits14_30000_steps.zip \
  --episodes 3 --max-episode-steps 2000
```

Evaluate the ResNet18 alternative (no VAE collection needed):

```bash
python rl/eval_loop_vae_sac.py \
  --encoder resnet18 --encoder-crop-top 40 \
  --model models/rl_loop_vae_sac_resnet_v4_notrees/sac_resnet18_50000_steps.zip \
  --episodes 3 --max-episode-steps 2000
```

> The v4 ResNet checkpoint was trained with `--encoder-crop-top 40` (legacy); pass
> the same flag at eval. New ResNet runs should use `--encoder-crop-top 0`.

### Mountain Track (`donkey-mountain-track-v0`)

| Model | Encoder | Eval | Speed | Mean CTE |
|---|---|:---:|---:|---:|
| `rl_dinov2_mountain_v2` 40k | DINOv2-S (frozen) | 5/5 trunc | 2.093 | 0.577 |

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

---

## Other Baselines

### Single Generated Road (`donkey-generated-roads-v0`)

The original Raffin-style VAE+SAC baseline. Drives a fixed generated road (not a
closed loop — a good run reaches the end before 3000 steps). Kept as a
starting-point reference; not the current main target.

```bash
python rl/eval_vae_sac.py \
  --model models/rl_vae_sac_raffin_v1/final_model.zip \
  --vae-model models/vae_raffin_v1/best.pt \
  --episodes 3
```

### Behavioral Cloning

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

## Environment Setup

### Connecting to the simulator

The Windows Donkey Simulator talks to the WSL2 Python client over TCP 9091. The
Windows host IP changes per boot — derive it from the default route:

```bash
export DONKEY_SIM_HOST="$(ip route | awk '/default/ {print $3}')"
export DONKEY_SIM_PORT=9091
```

Persist in `~/.bashrc`. Verify with `nc -vz "$DONKEY_SIM_HOST" 9091` once the
simulator is running.

### Python

```bash
conda create -n rl311 python=3.11
conda activate rl311
pip install "stable-baselines3[extra]" gymnasium numpy pillow opencv-python tqdm tensorboard
```

Install PyTorch with the command matching the local CUDA version from the official
PyTorch selector.

### gym-donkeycar

Do **not** install the PyPI package (old Gym API, incompatible):

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

### Models (Git LFS)

Model files are stored in Git LFS. After cloning:

```bash
git lfs pull
```

---

## Training

### Loop Track — VAE + SAC (recommended)

Requires fixed lighting in the simulator (`randomlight` disabled).

```bash
# 1. Collect images
python tools/collect_sim_frames.py ...
python tools/prepare_vae_dataset.py ...
python tools/build_vae_cache.py ...

# 2. Train VAE encoder
python rl/train_vae.py \
  --cache-dir data/vae/cache_loop_cones_fixedlight_v1 \
  --output-dir models/vae_loop_cones_fixedlight_v1 \
  --epochs 20

# 3. Train SAC (proven recipe: safe_v2 reward, hidden=80)
python rl/train_loop_vae_sac.py \
  --encoder vae \
  --vae-model models/vae_loop_cones_fixedlight_v1/best.pt \
  --output-dir models/rl_loop_new \
  --hidden-size 80 \
  --batch-size 128 \
  --gradient-steps-min 50 \
  --gradient-steps-cap 1000 \
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
  eval_loop_vae_sac.py                evaluation entrypoint

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
| `rl_loop_vae_sac_resnet_v4_notrees` | `sac_resnet18_50000_steps.zip` | ResNet18 | 3/3 | Backup, needs `--encoder-crop-top 40` |

VAE encoder shared by all VAE-based loop models: `vae_loop_cones_fixedlight_v1/best.pt`

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
