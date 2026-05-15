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
3. Use a single-frame CNN behavior cloning baseline as the current working
   policy.
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

It is acceptable to record a racing-line style dataset that cuts inside corners
instead of strictly following the yellow centerline. The behavior cloning target
is the demonstrated driving style, not the simulator centerline. Keep the style
consistent and avoid very late saves, crashes, and ambiguous recovery actions.

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

The current working dataset is:

```text
data/slow_data_raw/slow_data/road1
data/slow_data_raw/slow_data/road2
data/slow_data_raw/slow_data/road3
data/slow_data_raw/slow_data/road4
data/slow_data_raw/slow_data/road5
data/slow_data_raw/slow_data/road6
records: 72999 total
images: 72999 total
image shape: (160, 120), RGB
overall throttle mean: about 0.095
overall throttle p95: about 0.17
```

The current 6-road slow dataset did not use head or tail trimming for the active
baseline. Older single-route and fast datasets are kept only as historical
experiments.

## Behavior Cloning Baselines

### Single-Frame CNN

The current first baseline is a PyTorch port of the Donkey/PilotNet-style CNN:

```text
current RGB image -> CNN -> steer, throttle
```

Train the current working baseline with CUDA:

```bash
python train_bc.py \
  --data-dir \
    data/slow_data_raw/slow_data/road1 \
    data/slow_data_raw/slow_data/road2 \
    data/slow_data_raw/slow_data/road3 \
    data/slow_data_raw/slow_data/road4 \
    data/slow_data_raw/slow_data/road5 \
    data/slow_data_raw/slow_data/road6 \
  --drop-head 0 \
  --drop-tail 0 \
  --min-throttle 0.0 \
  --max-abs-angle 0.8 \
  --output-dir models/bc_nvidia_slow_006_random_split \
  --batch-size 256 \
  --num-workers 4 \
  --epochs 140 \
  --patience 12 \
  --learning-rate 0.001 \
  --val-split 0.2
```

Observed result:

```text
best val_loss: 0.001056
best epoch: 118
model: models/bc_nvidia_slow_006_random_split/best.pt
```

The raw model output is under-steered in closed-loop driving. A fixed actuator
calibration is therefore used at evaluation time:

```text
throttle_max: 0.35
steering_scale: 2.4
steering_limit: 0.8
```

This is a control calibration, not route-specific logic. It does not use CTE,
position, road geometry, or future route information.

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

### CNN+RNN/GRU Status

The GRU path is currently paused. Keep the scripts for reference, but do not use
them as the active baseline:

```text
train_bc_gru.py
eval_bc_gru.py
models/bc_gru_slow_006_random_split_seq8
```

The tested GRU model was:

```text
sequence of RGB frames
-> shared CNN encoder per frame
-> GRU
-> dense head
-> steer, throttle
```

Offline random-split validation loss was low, but simulator testing showed
unstable closed-loop behavior. In particular, feeding the true recent frame
history caused large steering spikes on non-sharp turns. Feeding the same GRU
with the current frame repeated across the sequence removed those spikes and
improved survival, which indicates that the learned temporal dynamics were the
problem.

DonkeyCar does document an RNN model family using sequence images,
TimeDistributed convolution layers, LSTM layers, and dense layers. If temporal
models are revisited, reproduce that official LSTM-style architecture or a
bounded/categorical variant instead of continuing the current GRU directly.

## Evaluation

Run a trained behavior cloning model in the simulator:

```bash
python eval_bc.py \
  --model models/bc_nvidia_slow_006_random_split/best.pt \
  --env-id donkey-generated-roads-v0 \
  --host 127.0.0.1 \
  --port 9091 \
  --episodes 1 \
  --max-episode-steps 3000 \
  --recreate-env-each-episode \
  --exit-scene-between-episodes \
  --scene-reload-delay 3.0 \
  --sleep 0.02 \
  --throttle-max 0.35 \
  --steering-scale 2.4 \
  --steering-limit 0.8 \
  --device cuda
```

Use multiple random generated roads for evaluation, but label generated-road
crossings and visual flicker separately. The current training data intentionally
avoids crossing roads, so crossing/flicker failures are out-of-distribution
failures and should not be mixed into normal-route success rate.

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

Current evaluation notes:

```text
2.0 / 0.65 / 0.35:
  conservative, stable, tends to run close to the edge.

2.4 / 0.8 / 0.35:
  current preferred setting; stronger cornering and better centerline behavior
  on normal roads, but crossing/flicker scenes remain OOD.

1.8 / 0.8 / 0.4:
  faster and visually good, but lower safety margin.
```

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
