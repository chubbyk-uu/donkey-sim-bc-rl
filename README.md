# Donkey Sim Learning Project

This project uses Donkey Simulator on Windows and Python training code in WSL2.
The current connection path is:

```text
WSL Python -> 127.0.0.1:9091 -> Windows Donkey Simulator
```

The simulator connection, image observation, telemetry, and control commands have
been verified with `smoke_test.py`.

## Goals

The project will follow proven Donkey Car workflows before adding custom
experiments.

1. Reproduce the official Donkey Car behavior cloning workflow.
2. Build reusable Gymnasium/RL environment and evaluation tools.
3. Implement the stronger RL path used by existing Donkey RL projects:
   collect images, train an AE/VAE representation model, then train a continuous
   control policy with SAC or TQC.

## Current Environment

Verified packages:

```text
Python 3.11.15
gym-donkeycar 1.3.1
gymnasium 1.2.3
stable-baselines3 2.8.0
torch 2.11.0+cu128
CUDA 12.8 build
GPU: NVIDIA GeForce RTX 4080 Laptop GPU
```

The current Donkey Gymnasium environment uses:

```text
observation: camera image, shape (120, 160, 3)
action: [steering, throttle]
info: cte, speed, pos, hit, lap_count, last_lap_time, etc.
```

## Smoke Test

Start Donkey Simulator on Windows, then run:

```bash
python smoke_test.py
```

The default target is:

```text
host: 127.0.0.1
port: 9091
env: donkey-warren-track-v0
```

Use a different track with:

```bash
python smoke_test.py --env-id donkey-generated-track-v0
```

Use a different host or port with:

```bash
python smoke_test.py --host 127.0.0.1 --port 9091
```

The environment id determines the map. For example:

```text
donkey-warren-track-v0 -> warren
```

The simulator can load the selected scene automatically as long as the simulator
process is running and accepting Gym TCP connections.

## Development Plan

### Phase 1: Official Behavior Cloning Baseline

This is the Donkey Car official deep learning workflow.

The standard flow is:

```text
human driving -> tub data -> train CNN -> autopilot inference
```

Donkey records camera images, steering, and throttle while a human drives. The
trained model then predicts steering and throttle from the camera image.

Initial tasks:

1. Create or configure a Donkey car app for simulator use.
2. Drive in the simulator and record tub data.
3. Clean bad records if needed.
4. Train a Donkey behavior cloning model.
5. Evaluate the model in the simulator.

Reference commands from the official simulator workflow:

```bash
python manage.py drive
donkey train --tub ./data --model models/mypilot.h5
python manage.py drive --model models/mypilot.h5
```

### Phase 2: RL Interface And Evaluation Tools

This phase keeps RL code separate from Donkey's official behavior cloning flow.
The purpose is to make experiments repeatable.

Planned files:

```text
ppo_common.py or rl_common.py   shared environment creation and wrappers
eval_policy.py                 load a policy and report reward/cte/speed/laps
record_rollout.py              save observations, actions, rewards, and info
runs/                          TensorBoard logs
models/                        saved policies and checkpoints
data/                          collected images, tubs, rollouts, or AE datasets
```

This phase should not introduce a custom neural network unless it is required
for an experiment. It should first provide reliable environment creation,
logging, saving, and evaluation.

### Phase 3: AE/VAE + Continuous Control RL

Existing Donkey RL projects often avoid training directly from raw images. A
common pattern is:

```text
collect images -> train AE/VAE -> encode observations -> train SAC/TQC policy
```

The AE/VAE compresses camera images into a smaller latent state. The RL policy
then learns vehicle control from that latent state, which is usually faster and
more stable than training directly from raw pixels.

Candidate RL algorithms:

```text
SAC: Soft Actor-Critic, strong off-policy continuous control baseline
TQC: Truncated Quantile Critics, SAC-style algorithm from sb3-contrib
```

Implementation order:

1. Collect an image dataset from simulator driving.
2. Train an AE or VAE and inspect reconstruction quality.
3. Add a Gymnasium wrapper that replaces raw images with latent vectors.
4. Train SAC as the first continuous control baseline.
5. Add TQC after SAC is working.
6. Compare against behavior cloning and optional PPO baselines.

## References

- Donkey Car official autopilot workflow:
  https://docs.donkeycar.com/guide/train_autopilot/
- Donkey Car simulator workflow:
  https://docs.donkeycar.com/guide/deep_learning/simulator/
- Gymnasium Donkey environment:
  https://github.com/tawnkramer/gym-donkeycar
- Learning to Drive in a Day:
  https://arxiv.org/abs/1807.00412
- Donkey RL with AE/VAE and SAC/PPO:
  https://github.com/ian0/donkeycar-rl
- DDPG + VAE Donkey implementation:
  https://github.com/r7vme/learning-to-drive-in-a-day
- SB3/RL Zoo Donkey TQC model:
  https://huggingface.co/araffin/tqc-donkey-avc-sparkfun-v0
