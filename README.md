# Donkey Simulator BC/RL

This repository contains behavioral cloning and reinforcement learning experiments for
Donkey Simulator. Training and evaluation run from WSL2 against a Windows Donkey
Simulator instance.

The current deployment recommendation on `donkey-generated-track-v0` loop track is the
fixed-light VAE+SAC `v3_h80 90k` checkpoint:

```text
model:        models/rl_loop_vae_sac_fixedlight_v3_h80/sac_loop_vae_90000_steps.zip
vae:          models/vae_loop_cones_fixedlight_v1/best.pt
eval:         3/3 episodes reached max_episode_steps=2000
mean speed:   2.800
progress:     280
mean |cte|:   0.315
max |cte|:    1.80
```

The earlier `v2_h64 100k` is kept as a secondary (slightly slower 2.689 speed but
even more centered 0.301 cte). The two reward-weight ablations after v3_h80
(v4 with s=0.20/c=0.20 and v5 with s=0.20/c=0.25) both lost to v3_h80 in
deterministic eval and were deleted; the original safe_v2 reward weights
`s=0.15, c=0.25` remain the best-performing reward shape — see
[experiment log §6.5](docs/experiment-log.md#65-reward-weight-ablations-v4-s20c20-and-v5-s20c25).

The whole fixed-light branch was deliberately rebuilt from scratch with `randomlight`
disabled in the simulator, so the VAE training images, SAC training rollouts, and eval
all share the same fixed lighting. It is reproducible across simulator restarts, which
the older `rl_loop_vae_sac_safe_v2` SAC run (trained on a random-light VAE) was not.

A historical "best at peak" result existed under uncontrolled lighting:

```text
model:        models/rl_loop_vae_sac_safe_v2/sac_loop_vae_70000_steps.zip  (REMOVED)
eval:         5/5 truncate at max=3000, mean speed 2.914, mean |cte| 0.333
caveat:       trained with randomlight ON; not reproducible across sim restarts
note:         the famous "8.56s lap" was a training-time fastest single lap (with
              exploration noise), not a deterministic-eval number
```

That artifact and its random-light VAE encoder were removed. It is preserved in the
experiment log only as a "best under matched lighting, no longer deployable" data point.

See [docs/experiment-log.md](docs/experiment-log.md) for the full BC, single-road RL,
and loop-track RL experiment history.

## Goal

The project goal is to learn reliable autonomous driving policies in Donkey Simulator,
starting from image observations:

```text
camera image -> (optional top crop) -> encoder latent/features -> SAC policy -> steer/throttle
```

The VAE encoder crops the top 40 px because the VAE was trained on cropped 80x160
frames. The frozen ResNet encoder defaults to no crop (`--encoder-crop-top 0`) — see
the "Main Commands" section.

Two RL environments are tracked separately:

- `donkey-generated-roads-v0`: a generated single road. The original Raffin-style
  VAE+SAC baseline can drive most of the fixed route, but the route is not a closed
  loop and intersection/generalization behavior is weak.
- `donkey-generated-track-v0`: a closed loop track. This is the current main branch.
  Current deployment uses a fixed-light VAE + SAC pipeline (`fixedlight_v3_h80` 90k).
  A frozen-ResNet18 alternative is kept as a backup that needs no VAE data collection.

BC models are kept as historical baselines and diagnostics. Among RL branches, the
fixed-light VAE pipeline is the recommended deployment; ResNet18 is the
no-data-collection alternative; the old random-light `rl_loop_vae_sac_safe_v2` SAC
run was the former leader but is non-reproducible (its VAE used uncontrolled lighting)
and was removed.

## Environment Setup

The expected layout is:

```text
Windows Donkey Simulator  <->  WSL2 Python client
                           TCP 9091
```

Set the simulator host in WSL. The Windows host IP visible from WSL2 changes per
machine and even per boot, so derive it from the default route rather than hard-coding:

```bash
export DONKEY_SIM_HOST="$(ip route | awk '/default/ {print $3}')"
export DONKEY_SIM_PORT=9091
```

Persist those in `~/.bashrc` if needed. Sanity check with `echo $DONKEY_SIM_HOST` and
`nc -vz "$DONKEY_SIM_HOST" "$DONKEY_SIM_PORT"` once the simulator is running.

### Python

The current working environment uses Python 3.11:

```bash
conda create -n rl311 python=3.11
conda activate rl311
```

Install the ML/runtime dependencies:

```bash
pip install "stable-baselines3[extra]" gymnasium numpy pillow opencv-python tqdm tensorboard
```

Install PyTorch using the command appropriate for the local CUDA version from the
official PyTorch selector. The current experiments used CUDA on an RTX 4080 Laptop GPU.

### gym-donkeycar

Do **not** install the PyPI package:

```bash
pip install gym-donkeycar   # WRONG — old Gym API, incompatible
```

The PyPI release lags behind upstream and still targets the old `gym` API. This project
uses Gymnasium-style training code, so install from the upstream `tawnkramer/gym-donkeycar`
repository as that project's own README recommends:

```bash
pip uninstall -y gym-donkeycar
pip install git+https://github.com/tawnkramer/gym-donkeycar
```

If you want to edit the dependency locally instead, clone and install editable:

```bash
git clone https://github.com/tawnkramer/gym-donkeycar ../gym-donkeycar
pip install -e ../gym-donkeycar
```

Sanity check:

```bash
python -c "import gym_donkeycar; print(gym_donkeycar.__file__)"
```

It should point to the editable clone or the `git+https` install location, not an old
site-packages PyPI install.

## Main Commands

Evaluate the current best loop-track policy (`fixedlight_v3_h80` 90k):

```bash
python rl/eval_loop_vae_sac.py \
  --encoder vae \
  --vae-model models/vae_loop_cones_fixedlight_v1/best.pt \
  --model models/rl_loop_vae_sac_fixedlight_v3_h80/sac_loop_vae_90000_steps.zip \
  --episodes 3 \
  --max-episode-steps 2000
```

Evaluate the ResNet18 alternative (no VAE collection needed):

```bash
python rl/eval_loop_vae_sac.py \
  --encoder resnet18 \
  --encoder-crop-top 40 \
  --model models/rl_loop_vae_sac_resnet_v4_notrees/sac_resnet18_50000_steps.zip \
  --episodes 3 \
  --max-episode-steps 2000
```

Note: that v4 ResNet checkpoint was trained with the legacy MARGIN_TOP=40 crop, so
`--encoder-crop-top 40` is required at eval. New ResNet runs should use the default
`--encoder-crop-top 0` (no crop) — keeping the full 120x160 sim image before resizing
to 224x224 reduces aspect ratio distortion and gives ResNet's ImageNet filters a
more natural input.

Train a new fixed-light loop VAE branch from scratch:

```bash
# 1. Collect fixed-light images (disable randomlight in the sim first):
python tools/collect_sim_frames.py ...
python tools/prepare_vae_dataset.py ...
python tools/build_vae_cache.py ...

# 2. Train the VAE encoder:
python rl/train_vae.py \
  --cache-dir data/vae/cache_loop_cones_fixedlight_v1 \
  --output-dir models/vae_loop_cones_fixedlight_v1 \
  --epochs 20

# 3. Train SAC on the new VAE (proven safe_v2 reward, hidden=80):
python rl/train_loop_vae_sac.py \
  --vae-model models/vae_loop_cones_fixedlight_v1/best.pt \
  --output-dir models/rl_loop_vae_sac_fixedlight_v3_h80 \
  --hidden-size 80 \
  --batch-size 64 \
  --gradient-steps-min 50 \
  --gradient-steps-cap 2000 \
  --timesteps 100000 \
  --save-replay-buffer \
  --save-final-replay-buffer \
  --device cuda
```

Evaluate the original single-road VAE+SAC baseline:

```bash
python rl/eval_vae_sac.py \
  --model models/rl_vae_sac_raffin_v1/final_model.zip \
  --vae-model models/vae_raffin_v1/best.pt
```

## Current Project State

### Loop Track

**Primary deployment** (fixed-light VAE + SAC, cold-started, hidden=80):

```text
models/rl_loop_vae_sac_fixedlight_v3_h80/sac_loop_vae_90000_steps.zip
models/vae_loop_cones_fixedlight_v1/best.pt                              (encoder)
```

Eval: 3/3 truncate at max=2000, mean_speed 2.800, mean |cte| 0.315, max |cte| 1.80.

**Secondary** (hidden=64 variant, even more centered but slower):

```text
models/rl_loop_vae_sac_fixedlight_v2_h64/sac_loop_vae_100000_steps.zip
models/rl_loop_vae_sac_fixedlight_v2_h64/sac_loop_vae_90000_steps.zip   (backup)
```

Eval: 3/3 truncate, mean_speed 2.689, mean |cte| 0.301, max |cte| 1.65.

**Backup** (frozen ResNet18, no VAE collection needed):

```text
models/rl_loop_vae_sac_resnet_v4_notrees/sac_resnet18_50000_steps.zip
```

Eval: 3/3 truncate at max=3000, mean_speed 2.131, mean |cte| 0.424. ~32% slower than
the fixed-light VAE policy but doesn't depend on VAE data collection.

**Co-deployment (alternative reward recipe, fixed-light VAE)**:

```text
models/rl_loop_vae_v15/sac_loop_vae_112488_steps.zip
```

Eval: 5/5 truncate at max=3000, mean_speed 2.797, mean |cte| 0.356, best_lap 9.11s,
mean_lap 9.47s. Uses `--lap-completion-bonus 50 --crash-speed-weight 15` on top of the
v3_h80 base. Slightly slower / wider than v3_h80 but reaches the same 100% truncate bar.
See [experiment log §8](docs/experiment-log.md#8-current-recommendation).

### Mountain Track

**Primary deployment** (DINOv2-S encoder + SAC, resumed from v1 30k with stricter
cte / heavier crash reward):

```text
models/rl_dinov2_mountain_v2/sac_dinov2_vits14_40000_steps.zip
```

Eval: 5/5 truncate at max=2000, mean_speed 2.093, mean |cte| 0.577, max |cte| 2.112,
best_lap 28.41s, mean_lap 29.88s, 15 laps. Encoder is frozen DINOv2-S (`dinov2_vits14`,
z=384) — no per-track VAE collection required.

Mountain sim conventions: spawn cte ≈ 3.54 (yellow centerline = cte=0, right-lane
center = cte=3.5); uphill needs `--max-throttle 0.7`. Use `--cte-target 3.5` so
termination and reward measure `abs(cte - 3.5)`. See
[experiment log §6.10](docs/experiment-log.md#610-mountain-track-dinov2-pipeline)
for the full v1 cold + v2 resume + v3/v4 cold-start failure story.

**Reward formula (`safe_v2`, used by both VAE branches and the ResNet branch)**:

```text
reward = 1.5 + 0.15 * speed - 0.25 * abs(cte - cte_target) * speed
crash  = -10 - 5 * speed
```

The full CLI flags that produced `fixedlight_v3_h80` 90k (which also list throttle /
steering / cte caps):

```text
--encoder vae
--vae-model models/vae_loop_cones_fixedlight_v1/best.pt
--hidden-size 80
--batch-size 64
--gradient-steps-min 50
--gradient-steps-cap 2000
--alive-reward 1.5
--speed-reward-weight 0.15
--cte-speed-penalty-weight 0.25
--reward-crash -10
--crash-speed-weight 5
--min-throttle 0.2
--max-throttle 0.7
--max-steering-diff 0.2
--max-cte-error 2.0
--max-episode-steps 3000
--timesteps 100000
```

Two reward-weight ablations after v3_h80 were tested and both lost to v3 in
deterministic eval:
- **v4 (s=0.20, c=0.20)**: weakening cte penalty made the actor settle into a
  "ride brake / wider line" policy that was slower per lap. Deleted.
- **v5 (s=0.20, c=0.25)**: stronger speed weight only — produced individual fast
  laps but the policy became fragile (crashed every 1-3 laps in eval). Deleted.

The `safe_v2` weights `(speed=0.15, cte_pen=0.25)` remain the best balance under
the current encoder. See
[experiment log §6.5](docs/experiment-log.md#65-reward-weight-ablations-v4-s20c20-and-v5-s20c25).

For new ResNet18 / MobileNet runs, leave `--encoder-crop-top 0` (the default). The
historical v4 ResNet checkpoint used `--encoder-crop-top 40` (legacy MARGIN_TOP
carried over from the VAE pipeline); it works but stretches the cropped 80x160
vertically by 2.8x when resizing to 224x224, distorting natural shapes for
ImageNet-pretrained features. The full 120x160 → 224x224 path only stretches 1.87x
vertically, which is closer to ImageNet's natural-image aspect.

Schedule for `fixedlight_v3_h80`: single 100k cold-start run, no resume needed. Best
deterministic eval was at the 90k checkpoint; 100k showed sharp regression (0/3
truncate at 933 mean steps). See
[experiment log §6.4](docs/experiment-log.md#64-fixedlight_v3_h80--slightly-faster-than-v2_h64)
for the full eval table.

**Historical (removed)**:

```text
models/rl_loop_vae_sac_safe_v2/sac_loop_vae_70000_steps.zip
```

Best ever observed (5/5 truncate at max=3000, mean speed 2.914, mean |cte| 0.333) but
used random-light VAE so the result was not reproducible across simulator restarts.
Removed; documented in the experiment log as a non-current artifact. The often-cited
"8.56s fastest lap" was training-time exploration noise, not a deterministic-eval
number.

### Single Generated Road

The Raffin-style VAE+SAC baseline is functional on the fixed generated road:

```text
VAE:   models/vae_raffin_v1/best.pt
SAC:   models/rl_vae_sac_raffin_v1/final_model.zip
```

The route is not closed, so a good run reaches the end before 3000 steps. This branch is
useful as a baseline, but not the current main deployment target.

### Behavioral Cloning

BC models remain in `models/bc_*` as historical baselines. They are useful for dataset
inspection and supervised diagnostics, but the current best control result is RL. The
two BC routes were regression CNN and official-style categorical. Regression was
generally more stable in closed-loop driving; both routes needed evaluation-time
steering amplification because raw steering outputs were too small (`1.8x` for the
regression baseline, about `1.4x` for the categorical baseline).

## Repository Map

```text
bc/train_bc.py                          BC regression CNN training
bc/eval_bc.py                           BC regression CNN evaluation
bc/train_bc_official_categorical.py     BC official-style categorical training
bc/eval_bc_official_categorical.py      BC official-style categorical evaluation
rl/train_vae.py                         VAE training
rl/train_vae_sac.py                     shared VAE+SAC environment/reward code
rl/train_loop_vae_sac.py                loop-track SAC training entrypoint
rl/eval_loop_vae_sac.py                 loop-track evaluation entrypoint
tools/collect_sim_frames.py             VAE frame collection from simulator
tools/prepare_vae_dataset.py
tools/build_vae_cache.py
tools/inspect_loop_replay_throttle.py   per-episode throttle/cte analysis from a saved SAC replay buffer
logs/                                   saved eval/train logs
docs/experiment-log.md                  detailed experiment history
```

## Current Data And Model Inventory

`data/` currently keeps only BC data and compressed raw backups. The old random-light
loop VAE images and manifests were removed so the next loop VAE run starts clean.

```text
data/slow_data_raw/
  Six slow generated-road tubs used by the BC regression/categorical baselines.

data/curated_cornering_v1_clean/
  First curated cornering subset used to improve BC recovery behavior.

data/curated_cornering_v2_clean/
  Second curated cornering subset used by the official-style categorical BC branch.

data/Cornering data.zip
data/cornorraw.zip
  Compressed backups of raw cornering data. The extracted copies were removed.
```

`models/` currently contains:

```text
models/bc_nvidia_slow_006_flip/
  Best BC regression baseline. Closed-loop steering needs about 1.8x scale.

models/bc_official_categorical_curve_aug_balanced_v1/
  Best retained BC categorical baseline. Closed-loop steering needs about 1.4x scale.

models/vae_raffin_v1/
  Generated-road VAE encoder for the original single-road RL baseline.

models/rl_vae_sac_raffin_v1/
  Generated-road SAC policy using models/vae_raffin_v1.

models/vae_loop_cones_fixedlight_v1/
  Fixed-light loop VAE encoder (80k frames, randomlight disabled).

models/rl_loop_vae_sac_fixedlight_v3_h80/
  PRIMARY loop-track deployment. Best checkpoint sac_loop_vae_90000_steps.zip
  (3/3 truncate, mean_speed 2.800, mean |cte| 0.315). hidden=80, batch=64,
  fixed-light VAE encoder, safe_v2 reward.

models/rl_loop_vae_sac_fixedlight_v2_h64/
  Secondary loop-track checkpoint (older, slower, more centered). Best
  sac_loop_vae_100000_steps.zip; backup sac_loop_vae_90000_steps.zip.

models/rl_loop_vae_sac_resnet_v4_notrees/
  Backup loop-track SAC branch using frozen ResNet18 features. Best checkpoint is
  sac_resnet18_50000_steps.zip. Trained with legacy --encoder-crop-top 40 (must
  pass that flag at eval time). New ResNet runs should use --encoder-crop-top 0.

models/rl_loop_vae_v15/
  Co-primary loop-track deployment with alternative reward (lap_bonus=50,
  crash_speed_weight=15). Best checkpoint sac_loop_vae_112488_steps.zip
  (5/5 truncate, mean_speed 2.797, mean_lap 9.47s). Also retains
  sac_loop_vae_132488_steps.zip as a faster-but-not-deployable reference.

models/rl_dinov2_mountain_v1/
  Mountain-track DINOv2 cold-start branch. Best checkpoint
  sac_dinov2_vits14_30000_steps.zip (5/5 truncate, mean_speed 1.998, mean_lap 31.09s).
  Superseded by mountain_v2 40k.

models/rl_dinov2_mountain_v2/
  Mountain-track deployment. Resumed from mountain_v1 30k with stricter cte
  penalty (0.25→0.30), heavier crash (-10→-20, crash_speed 5→10), and LR
  override 2e-4. Best checkpoint sac_dinov2_vits14_40000_steps.zip
  (5/5 truncate, mean_speed 2.093, mean_lap 29.88s).
```
