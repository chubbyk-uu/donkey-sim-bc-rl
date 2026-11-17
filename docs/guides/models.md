# Model catalog and legacy commands

Current status: [accepted baseline and decisions](../status.md). Run commands from the repository root.

## Choose by scenario

| Scenario | First choice | Evidence / settings |
| --- | --- | --- |
| Random light and tree shadows, generated loop | `visual_student_no_motion_v1` + original geometry head | 24/24 at 2k; 10/10 at 3k; camera/self-command only |
| Fixed-light loop | `rl_loop_vae_sac_fixedlight_v3_h80` 90k + fixed-light VAE | Historical 3/3 at 2k, speed 2.800; match checkpoint appearance |
| Fixed-light longer cap | `rl_loop_vae_v15` 112488 + same VAE | Historical 5/5 at 3k; later 132488 is not the accepted checkpoint |
| Mountain | `rl_dinov2_mountain_v2` 40k | Historical 5/5 at 2k; CTE target 3.5, throttle max 0.7 |
| True-state reference | `rl_loop_sim_state_ts2_sdiff01_v1` 60k + centre line | Privileged teacher, not camera-only deployment |

The older DINOv2 randomtree model reached 31/40 in a later 2k recheck; the older ~50% figure came from small May batches. Historical batches/protocols do not provide a paired ranking against the current student.

## Tracked model catalog

All paths below are versioned project artifacts. `.pt`/`.zip` weights use Git LFS; the small centre-line `.npz` uses ordinary Git. Raw replay buffers are excluded. The pretrained DINOv2 backbone is downloaded separately.

| Artifact | Role |
| --- | --- |
| `models/bc_nvidia_slow_006_flip/best.pt` | Historical generated-road regression BC; eval steering scale 1.8 |
| `models/bc_official_categorical_curve_aug_balanced_v1/best.pt` | Historical categorical BC; eval steering scale 1.4 |
| `models/dinov2_geometry_15d/proj15_lags_ridge.pt` | Required frozen geometry head for accepted student |
| `models/dinov2_geometry_15d_negative_v1/proj15_lags_ridge.pt` | Experimental negative-recovery head; not promoted |
| `models/dinov2_geometry_15d_round_b_v1/proj15_lags_ridge.pt` | Experimental recovery head; not promoted |
| `models/dinov2_grid_projection/pca128.pt` | Historical grid/PCA projection |
| `models/rl_dinov2_mountain_v1/sac_dinov2_vits14_10000_steps.zip` | Historical cold-start seed series; v2 derives from 30k |
| `models/rl_dinov2_mountain_v1/sac_dinov2_vits14_20000_steps.zip` | Historical cold-start seed series; v2 derives from 30k |
| `models/rl_dinov2_mountain_v1/sac_dinov2_vits14_30000_steps.zip` | Historical cold-start seed series; v2 derives from 30k |
| `models/rl_dinov2_mountain_v1/sac_dinov2_vits14_40000_steps.zip` | Historical cold-start seed series; v2 derives from 30k |
| `models/rl_dinov2_mountain_v1/sac_dinov2_vits14_50000_steps.zip` | Historical cold-start seed series; v2 derives from 30k |
| `models/rl_dinov2_mountain_v1/sac_dinov2_vits14_60000_steps.zip` | Historical cold-start seed series; v2 derives from 30k |
| `models/rl_dinov2_mountain_v2/sac_dinov2_vits14_40000_steps.zip` | 40k: Mountain deployment; 50k: backup |
| `models/rl_dinov2_mountain_v2/sac_dinov2_vits14_50000_steps.zip` | 40k: Mountain deployment; 50k: backup |
| `models/rl_dinov2_mountain_v3/sac_dinov2_vits14_40000_steps.zip` | Historical single-stage alternative |
| `models/rl_loop_dinov2_randomtree_crop40_v1/sac_dinov2_vits14_130000_steps.zip` | Historical crop40 probe; not promoted |
| `models/rl_loop_dinov2_randomtree_crop40_v1/sac_dinov2_vits14_170000_steps.zip` | Historical crop40 probe; not promoted |
| `models/rl_loop_dinov2_randomtree_v2/sac_dinov2_vits14_50000_steps.zip` | Historical shadowed-loop visual-SAC baseline, 31/40 at 2k in November |
| `models/rl_loop_dinov2_v8/sac_dinov2_vits14_30000_steps.zip` | Lighting-robust historical visual-SAC backup |
| `models/rl_loop_resnet_mountain_v1/sac_resnet18_80000_steps.zip` | Historical Mountain ResNet; superseded |
| `models/rl_loop_resnet_mountain_v1/sac_resnet18_90000_steps.zip` | Historical Mountain ResNet; superseded |
| `models/rl_loop_sim_state_ts2_sdiff01_v1/sac_sim_state_60000_steps.zip` | Privileged state teacher, stable 60k |
| `models/rl_loop_vae_sac_fixedlight_v2_h64/sac_loop_vae_100000_steps.zip` | Historical fixed-light backup |
| `models/rl_loop_vae_sac_fixedlight_v2_h64/sac_loop_vae_90000_steps.zip` | Historical fixed-light backup |
| `models/rl_loop_vae_sac_fixedlight_v3_h80/sac_loop_vae_90000_steps.zip` | Fixed-light loop deployment, 90k |
| `models/rl_loop_vae_sac_resnet_v4_notrees/sac_resnet18_50000_steps.zip` | Fixed-light legacy backup; crop-top 40 required |
| `models/rl_loop_vae_v15/sac_loop_vae_112488_steps.zip` | 112488: fixed-light co-deployment; 132488: reference, not promoted |
| `models/rl_loop_vae_v15/sac_loop_vae_132488_steps.zip` | 112488: fixed-light co-deployment; 132488: reference, not promoted |
| `models/track_centerline/generated_track.npz` | Fitted map for teacher/diagnosis; not a student input |
| `models/vae_loop_cones_fixedlight_v1/best.pt` | Required encoder for fixed-light loop VAE policies |
| `models/visual_student_bend_recovery_v1/student.pt` | Experimental bend student; not promoted |
| `models/visual_student_no_motion_v1/student.pt` | Accepted camera-only baseline |

The original Raffin generated-road files `models/rl_vae_sac_raffin_v1/final_model.zip` and `models/vae_raffin_v1/best.pt` are absent from Git/LFS and the working copy. Their historical record is retained, but they are not runnable deployments from a clone.

## Legacy evaluation commands

The [quick start](../../README.md#quick-start) runs the current student. The generic loop evaluator still defaults to the historical VAE model; specify encoder/model explicitly when using other branches. Fixed-light VAE requires random lighting disabled and tree/cone appearance matching its checkpoint.

```bash
python rl/eval_loop_vae_sac.py \
  --encoder vae --vae-model models/vae_loop_cones_fixedlight_v1/best.pt \
  --model models/rl_loop_vae_sac_fixedlight_v3_h80/sac_loop_vae_90000_steps.zip \
  --episodes 3 --max-episode-steps 2000

python rl/eval_loop_vae_sac.py \
  --env-id donkey-mountain-track-v0 --encoder dinov2_vits14 \
  --cte-target 3.5 --max-cte-error 2.5 --max-throttle 0.7 \
  --model models/rl_dinov2_mountain_v2/sac_dinov2_vits14_40000_steps.zip \
  --episodes 5 --max-episode-steps 2000

python rl/eval_loop_vae_sac.py \
  --encoder dinov2_vits14 \
  --model models/rl_loop_dinov2_randomtree_v2/sac_dinov2_vits14_50000_steps.zip \
  --episodes 3 --max-episode-steps 2000
```

Mountain spawn CTE is about 3.54; `--cte-target 3.5` measures right-lane error relative to 3.5. Legacy ResNet v4 requires `--encoder-crop-top 40`; new ResNet training normally uses 0. Match steering/throttle/history settings to each checkpoint. A historical `sim_state` teacher command must use steering difference 0.1 and the centre-line observation, and is explicitly privileged.

Detailed recipes and data dependencies are in [training](training.md).
