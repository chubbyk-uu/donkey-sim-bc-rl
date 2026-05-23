# Donkey Simulator BC/RL

This repository contains behavioral cloning and reinforcement learning experiments for
Donkey Simulator. Training and evaluation run from WSL2 against a Windows Donkey
Simulator instance.

The best historical result was on `donkey-generated-track-v0` loop track:

```text
model:        models/rl_loop_vae_sac_safe_v2/sac_loop_vae_70000_steps.zip
vae:          models/vae_loop_cones_v1/best.pt
eval:         5/5 episodes reached max_episode_steps=3000
mean speed:    2.914
mean progress: 437.1
mean |cte|:    0.333
max |cte|:     1.715
```

Important caveat: that VAE result was trained/evaluated while simulator random lighting
was not controlled. On `generated_track`, `randomlight` changes the scene color tone
between launches, so the old VAE checkpoint did not reliably reproduce the original
eval when lighting changed. The old random-light loop VAE data, encoder, and dependent
SAC models have therefore been removed. The next VAE loop run should recollect data
with random lighting disabled.

See [docs/experiment-log.md](docs/experiment-log.md) for the full BC, single-road RL,
and loop-track RL experiment history.

## Goal

The project goal is to learn reliable autonomous driving policies in Donkey Simulator,
starting from image observations:

```text
camera image -> crop top 40 px -> encoder latent/features -> SAC policy -> steer/throttle
```

Two RL environments are tracked separately:

- `donkey-generated-roads-v0`: a generated single road. The original Raffin-style
  VAE+SAC baseline can drive most of the fixed route, but the route is not a closed
  loop and intersection/generalization behavior is weak.
- `donkey-generated-track-v0`: a closed loop track. This is the current main branch.
  The old loop-specific VAE branch is being reset because its data used uncontrolled
  random lighting. The currently retained loop checkpoint uses a frozen ResNet18
  encoder; the next VAE branch should be trained from newly collected fixed-light data.

BC models are kept as historical baselines and diagnostics. The historical strongest
loop result was VAE+SAC under matched lighting, while the retained frozen ResNet encoder
branch is slower but avoids VAE image collection and appears less sensitive to visual
setup changes.

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

Evaluate the retained loop-track ResNet policy:

```bash
python rl/eval_loop_vae_sac.py \
  --encoder resnet18 \
  --hidden-size 256 \
  --model models/rl_loop_vae_sac_resnet_v4_notrees/sac_loop_vae_50000_steps.zip \
  --episodes 5
```

Train a new fixed-light loop VAE branch after recollecting images:

```bash
python tools/collect_sim_frames.py ...
python tools/prepare_vae_dataset.py ...
python tools/build_vae_cache.py ...
python rl/train_vae.py \
  --cache-dir data/vae/cache_loop_cones_fixedlight_v1 \
  --output-dir models/vae_loop_cones_fixedlight_v1 \
  --epochs 20
```

Then train SAC on the new VAE:

```bash
python rl/train_loop_vae_sac.py \
  --vae-model models/vae_loop_cones_fixedlight_v1/best.pt \
  --output-dir models/rl_loop_vae_sac_fixedlight_v1 \
  --timesteps 70000 \
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

Historical best checkpoint, now removed from `models/`:

```text
models/rl_loop_vae_sac_safe_v2/sac_loop_vae_70000_steps.zip
```

It was removed together with its random-light VAE encoder:

```text
models/vae_loop_cones_v1/best.pt
models/rl_loop_vae_sac_safe_v2/
models/rl_loop_vae_sac_speed_v1/
```

The historical result is documented in
[docs/experiment-log.md](docs/experiment-log.md), but it should be treated as a
non-current artifact because lighting was not controlled.

`safe_v2` reward defaults:

```text
reward = 1.5 + 0.15 * speed - 0.25 * abs(cte) * speed
crash  = -10 - 5 * speed
min/max throttle:      0.2 / 0.7
max steering diff:     0.2
max cte error:         2.0
max episode steps:     3000
```

The matching CLI flags (already the defaults in `rl/train_loop_vae_sac.py`):

```text
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
--gradient-steps-cap 1000
--gradient-steps-min 500
```

Newer ResNet encoder branch:

```text
encoder:       frozen ImageNet ResNet18
SAC hidden:    256 (VAE safe_v2 used 64)
best so far:   models/rl_loop_vae_sac_resnet_v4_notrees/sac_loop_vae_50000_steps.zip
eval:          3/3 truncate, mean speed 2.131, progress 319.5, mean |cte| 0.424
tradeoff:      no VAE image collection, but about 27% slower than VAE safe_v2 70k
```

This is the only retained loop-track checkpoint family at the moment.

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

models/rl_loop_vae_sac_resnet_v4_notrees/
  Retained loop-track SAC branch using frozen ResNet18 features. Best checkpoint so far:
  sac_loop_vae_50000_steps.zip.
```
