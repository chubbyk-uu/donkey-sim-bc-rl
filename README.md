# Donkey Sim Learning Project

This project uses Donkey Simulator on Windows and Python training code in WSL2.
The current connection path is:

```text
WSL Python -> 127.0.0.1:9091 -> Windows Donkey Simulator
```

The simulator connection, image observation, telemetry, and control commands have
been verified with `smoke_test.py`.

## Goals

The project starts with behavior cloning because it is the standard Donkey Car
baseline and gives a useful reference before reinforcement learning.

1. Build a reliable behavior cloning dataset from Windows Donkey Simulator.
2. Train and evaluate PyTorch driving models from WSL2.
3. Compare single-frame CNN, temporal CNN, and CNN+RNN/GRU behavior cloning.
4. Prepare the RL path after imitation learning is measurable:
   collect images, train an AE/VAE representation, then train SAC or TQC.

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

Do not use `donkeycar[pc]` in this environment. It pulled in an older
TensorFlow/DonkeyCar stack and downgraded packages such as `numpy`, which
conflicted with the current Gymnasium-based `gym-donkeycar` workflow.

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

## Data Collection

The usable manual recording path found for the current Windows simulator is:

```text
Windows Donkey Simulator -> generated_road -> No Rec / w Rec recording
```

Important findings:

- Manual driving through the normal scene menu can drive the car, but does not
  necessarily record data.
- In the current simulator build, `generated_road` is the scene that exposes the
  recording switch and writes `record_*.json` plus camera images.
- A Gymnasium client from WSL creates its own simulator-controlled car. It is
  not a passive recorder for the Windows gamepad car.
- Generated roads are random across loads; even with seed settings they should
  be treated as different routes.

Recommended recording plan:

```text
D:\WSL\log_001
D:\WSL\log_002
D:\WSL\log_003
D:\WSL\log_004
D:\WSL\log_005
```

Record at least 5 different generated roads. Keep speed slow and stable. Include
some mild recovery driving where the car is slightly off-center and then steered
back toward the road center. A useful first target is 30k-50k frames total.

Copy a Windows log directory into WSL-local storage before training:

```bash
cp -r /mnt/d/WSL/log_001 data/generated_road_001
```

Training directly from `/mnt/d/...` works, but many small image files are much
slower there than on the WSL filesystem.

## Dataset Inspection

Inspect a copied dataset before training:

```bash
python inspect_dataset.py data/generated_road_001 --sample-images 3
```

For the current cleaned dataset:

```text
data/generated_road_bc_002
records: 11941
images: 11941
image shape: (160, 120), RGB
angle range: about -0.43 to +0.43
throttle mean: about 0.10
```

The latest dataset did not need head trimming. Training used `--drop-tail 5`
because the final frames contained stop/end artifacts.

## Behavior Cloning Baselines

### Single-Frame CNN

The current first baseline is a PyTorch port of the Donkey/PilotNet-style CNN:

```text
current RGB image -> CNN -> steer, throttle
```

Train it with CUDA:

```bash
python train_bc.py \
  --data-dir data/generated_road_bc_002 \
  --drop-head 0 \
  --drop-tail 5 \
  --output-dir models/bc_nvidia_generated_road_002 \
  --batch-size 128 \
  --num-workers 2 \
  --epochs 100 \
  --patience 6
```

Observed result:

```text
best val_loss: 0.000579
best epoch: 86
model: models/bc_nvidia_generated_road_002/best.pt
```

This model can drive some generated roads, but still fails on harder random
routes. A 5-route random evaluation saw survival steps from about 641 to 1000.

### Temporal CNN

We tested frame stacking:

```text
history=10
frame_stride=2
input frames: t-18, t-16, ..., t
RGB channels: 30
```

From-scratch training with 30 input channels regressed toward predicting the
mean action. The mean-action baseline MSE was about `0.00725`, and the
from-scratch 10-frame model did not beat it.

A better variant initialized the 30-channel model from the trained single-frame
checkpoint:

- copy the single-frame convolution weights into the current-frame channels;
- initialize older-frame channels to zero;
- copy the rest of the model weights.

That immediately reached validation loss around `0.00052`, slightly better than
the single-frame baseline, but this is not yet enough evidence to prefer it.

### CNN+RNN/GRU Plan

The next recommended model is a recurrent behavior cloning model:

```text
sequence of RGB frames
-> shared CNN encoder per frame
-> GRU or LSTM
-> dense head
-> steer, throttle
```

This is based on the DonkeyCar RNN model family. DonkeyCar documents an RNN
model using a sequence of images, TimeDistributed convolution layers, LSTM
layers, dense layers, and driving outputs. The official command family also
includes `--type=rnn`.

Start with a small GRU, not a large LSTM:

```text
sequence_len: 8
frame_stride: 1 or 2
CNN encoder: current PilotNet-style conv stack
RNN: GRU hidden_size=128, num_layers=1
head: Linear(128 -> 50 -> 2)
optimizer: Adam, lr=1e-4 or 3e-4
batch_size: 64 or 96
```

Use more data before relying on the GRU result. A recurrent model can overfit
one route just as easily as a CNN if the demonstrations are narrow.

## Evaluation

Run a trained behavior cloning model in the simulator:

```bash
python eval_bc.py \
  --model models/bc_nvidia_generated_road_002/best.pt \
  --env-id donkey-generated-roads-v0 \
  --host 127.0.0.1 \
  --port 9091 \
  --steps 1000 \
  --sleep 0.02
```

Use multiple random generated roads for evaluation, not a single lucky route.
Track at least:

```text
survival_steps: how long before failure/reset
mean_abs_cte: average absolute cross-track error
max_abs_cte: worst absolute cross-track error
reward: simulator reward
```

`CTE` means cross-track error: distance from the road centerline. `cte=0` is
near the center. Larger `abs(cte)` means the car is closer to leaving the road.
In this simulator, failures have often appeared near `abs(cte) ~= 8`.

## RL Interface And Evaluation Tools

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

## AE/VAE + Continuous Control RL

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
- Donkey Car command documentation, including `--type=rnn`:
  https://docs.donkeycar.com/utility/donkey/
- Donkey Car Keras RNN model description:
  https://donkeycar.cn/parts/keras/
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
