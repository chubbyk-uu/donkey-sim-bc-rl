# Training recipes

These are historical reproduction recipes, not scheduled experiments. Restore
the required local data and use fresh output paths. For the accepted student's
different training path, see [student architecture](student.md).

## Fixed-light loop: VAE + SAC

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
final step). Historical runs showed a 10-20k peak-valley-recovery cycle — see
[experiment log §6.9](../archive/experiments/03-fixed-light-loop.md#69-lessons-from-the-fixed-light-loop-work).
Current CLI defaults are `--batch-size 128 --gradient-steps-cap 1000`; those newer
defaults have not cleanly reproduced the deployed
v3_h80 result yet.

## Historical visual SAC: DINOv2 / ResNet

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
layouts). Eval with `--scene-reload-every 1` (fresh layout per episode). Small May
DINOv2 SAC batches reached roughly 50% completion; the later standard-setting
v2 recheck was 31/40. These are historical SAC results, not the current student's
24/24 and 10/10 results — see
[experiment log §6.14](../archive/experiments/05-encoders-and-randomization.md#614-domain-randomization-random-light--tree-shadows-on-loop-2026-05-28).
Frozen ResNet18 was tested under the same random light + tree-shadow setup and was
stopped at 20k: it never completed a lap and recent episodes stayed under 200 steps.
For shadowed-road robustness, DINOv2 is clearly stronger than ResNet18.

## Mountain: DINOv2

The deployed Mountain v2 uses two stages: permissive cold start followed by
stricter reward. The historical strict-reward cold-start branch failed. See
[experiment log §6.10](../archive/experiments/04-mountain.md#610-mountain-track-dinov2-pipeline).

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

# Stage 2: example using the retained historical v1 30k seed
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
  --resume-model models/rl_dinov2_mountain_v1/sac_dinov2_vits14_30000_steps.zip \
  --output-dir models/rl_dinov2_mountain_new_v2 \
  --timesteps 40000 --save-replay-buffer --device cuda
```

---

For a fully new two-stage run, replace the retained stage-1 seed path above
with the checkpoint selected by deterministic evaluation of the new stage-1 run.

## Resume and replay-buffer contracts

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

These are historical/reproduction recipes, not scheduled work. Current student training is described in [the student guide](student.md). Raw datasets and replay buffers are not shipped in Git.
