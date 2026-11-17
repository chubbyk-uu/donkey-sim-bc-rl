# Encoders and appearance randomization

> Historical record. Results, “current” choices and “next” plans below belong to the dated experiments, not the active work queue. See [current status](../../status.md) and [documentation index](../../README.md). Commands assume the repository root; raw `logs/` and `data/` artifacts are local-only unless explicitly noted.

### 6.11 Training defaults updated 2026-05-27

After the v4 instability analysis, two SAC defaults were raised toward the
SAC paper's values for late-training stability. A `gradient_steps_scale`
parameter was also added for future ablation.

| Default | Old | New | Rationale |
| --- | ---: | ---: | --- |
| `--batch-size` | 64 | **128** | Reduce critic gradient variance. SAC paper uses 256; 128 is a midpoint. |
| `--buffer-size` | 30_000 | **50_000** | Slower buffer turnover → less distribution shift during late training. |
| `--gradient-steps-scale` | (implicit 1.0) | **1.0** (explicit) | New CLI flag. Default keeps existing behavior (1 update per env step). Set <1 (e.g. 0.5) to dampen long-episode overupdate; experimental, not yet validated. |
| `--gradient-steps-cap` (`train_vae_sac.py`) | 600 | **1000** | Unified with `train_loop_vae_sac.py`. |

`CappedDynamicGradientStepsCallback` formula changed from
`clamp(ep_len, [floor, cap])` to `clamp(ep_len * scale, [floor, cap])`. With
default `scale=1.0` this is identical to prior behavior.

Not yet validated in a deterministic-eval run — applies starting from the
next cold-start training.

### 6.12 Encoder generalization: lighting robustness comparison (2026-05-27)

Single-episode eval on `donkey-generated-track-v0`, 2000-step cap. "Random
light" = simulator `randomlight` enabled (lighting direction randomised each
episode); "Fixed light" = simulator default stable lighting.

**Models tested**

| Encoder | Model | Checkpoint |
| --- | --- | --- |
| VAE (task-specific, fixedlight) | rl_loop_vae_fixedlight_v3_h80 | 90k |
| DINOv2 ViT-S (frozen ImageNet-pretrained) | rl_loop_dinov2_v8 | 30k |
| ResNet18 (frozen ImageNet-pretrained) | rl_loop_vae_sac_resnet_v4_notrees | 50k |

**Results**

| Encoder | Light | Trunc | Speed | Mean CTE | Lap time |
| --- | --- | :---: | ---: | ---: | ---: |
| VAE | Fixed | 1/1 | 2.742 | 0.296 | 9.65 s/lap (5 laps) |
| VAE | Random | 0/1 | — | — | crash at step 269 |
| DINOv2 | Fixed | 1/1 | 2.356 | 0.565 | 11.03 s/lap (4 laps) |
| DINOv2 | Random (×3) | 3/3 | 2.38 avg | — | 10.97 s/lap avg |
| ResNet18 | Random (×3) | 3/3 | 2.09 avg | — | 12.88 s/lap avg |

(ResNet was only tested under random light; VAE was not tested for a second
fixed-light run. Fixed-light VAE lap count is 5 laps / 1000 steps.)

**Conclusions**

- **VAE is task- and lighting-specific.** Trained on fixed-light data, it
  crashes immediately when lighting shifts. This is expected: the VAE
  latent space encodes lighting as part of scene appearance; distribution
  shift to OOD lighting breaks the encoder's implicit assumptions.
- **Frozen pretrained encoders (DINOv2, ResNet18) are lighting-robust.** Both
  survive all three random-light runs without issue. ImageNet pretraining
  covers a wide distribution of lighting conditions; a few random-light
  episodes are in-distribution for these encoders.
- **DINOv2 > ResNet18 in lap time** (10.97 s vs 12.88 s avg) under random
  light, consistent with DINOv2's stronger perceptual features from
  self-supervised pretraining.
- **Practical guidance**: use VAE when lighting is controlled end-to-end
  (fastest laps, lowest CTE); use DINOv2 when lighting may vary (robust,
  only ~1.3 s/lap slower). ResNet18 is a reliable fallback but lags DINOv2.

### 6.14 Domain randomization: random light + tree shadows on loop (2026-05-28)

**Headline: DINOv2 keeps its generalization under randomized appearance.** With
domain randomization the frozen DINOv2-S policy adapts to *both* randomized
lighting *and* randomly-placed trees and their cast shadows — completing full
laps on layouts it never trained on. §6.12 already showed DINOv2 was robust to
random lighting; this extends it to random trees/shadows, which are a harder
distractor (shadows fall on the road and mimic lane edges).

**Baseline without DR fails.** The plain loop DINOv2 model (`rl_loop_dinov2_v8`),
trained on a single fixed scene, crashes at step 242 (0/1, max_cte 2.10) the
moment random light + trees are enabled. So robustness is not free from the
encoder alone — it has to be trained in.

**Scene-reload domain randomization (the mechanism).** The sim only regenerates
trees/lighting/shadows when the scene is *reloaded*; a normal episode reset
(`reset_car`) keeps the same layout. New code in `train_vae_sac.py`:

- `reload_scene()` — `exit_scene` → settle → re-send `load_scene` until the sim
  re-handshakes (`handler.loaded`). `load_scene` is ignored mid-scene, so the
  exit-to-menu step is required. (Probe tool: `tools/test_scene_reload.py`.)
- **Adaptive step-based reload** (`--scene-reload-alpha`, `--scene-reload-kmin`):
  reload after the car has spent `K = clamp(alpha × recent_ep_len, [kmin,
  max_ep-1])` steps on the current scene, decided only at episode boundaries
  (never force-truncates). `_steps_on_scene` accumulates across crashes and
  resets only on reload, so easy (long-episode) and hard (short-episode) layouts
  contribute roughly equal *steps* to the replay buffer. This fixed a real bias:
  episode-count-based reload let easy scenes (long truncate episodes) dominate
  the buffer. Counter-intuitively, a *smaller* fixed step budget worsens balance
  (an easy scene still spends a full episode before the boundary check), so K
  near `max_ep` balances best; adaptive K gives early-training variety and
  late-training balance. Episode-based `--scene-reload-every N` is retained for
  **eval** (use `N=1` so each eval episode is a fresh random layout).

**Result — `rl_loop_dinov2_randomtree_v2`** (DINOv2-S, hidden=256, batch=256,
`gradient-steps-scale=0.5`, adaptive reload `alpha=3 kmin=200`, raffin reward
`speed=0.15 cte=0.25`, `reward_crash=-20`, `max_ep=2000`). Paired eval — every
checkpoint on the **same** 6 freshly-reloaded random layouts (the fair way to
compare; see `rl/eval_paired_randomized.py`):

| Checkpoint | Trunc | Mean steps | Mean CTE | Mean speed |
| --- | :---: | ---: | ---: | ---: |
| 30k | 3/6 | 1418 | 0.460 | 1.771 |
| 40k | 2/6 | 1106 | 0.457 | 1.795 |
| **50k** | **3/6** | 1262 | **0.405** | 1.842 |
| 60k | 2/6 | 1168 | 0.429 | 1.843 |

Independent per-checkpoint eval (5 ep, fresh layout each) agreed: 30k 4/5, 40k
3/5, 50k 4/5, 60k 3/5. So **~50% truncate on unseen random layouts** — and the
"failures" are long drives (4-6 laps) that eventually drift to cte≈2.0, not early
crashes. Best checkpoint **50k**.

**Three attempts to push past ~50% — all failed (documented so they aren't
re-tried):**

1. **Tighter cte penalty** (0.25→0.30) + lower min-throttle: no clear trunc gain.
   Note the cte penalty is `weight × |cte| × speed`, so lowering speed weakens it
   — raising the weight is partly cancelled by the slower driving it induces.
2. **Steering-change penalty** to damp the persistent left-right weave
   (`abs_steer_delta_mean ≈ 0.30` throughout — i.e. ~73% of the per-step steering
   cap `max_steering_diff=0.2`→±0.4, *not* "15% of full range" as an earlier draft
   wrongly stated). Both linear (`w=0.5`) and squared (`w=2.0`) penalties left
   steer_delta **dead flat at ~0.30** over tens of thousands of steps — zero
   effect. Removed.
3. **Bigger frozen encoder** (DINOv2 ViT-B/14, z=768, hidden=384): steer_delta
   identical (~0.30) and trunc no better, at ~2× the per-step cost (fps ~9 vs
   ~15). Deleted.
4. **Frozen ResNet18 under the same randomized light/tree-shadow setup** (crop_top=0,
   hidden=256, batch=256, `gradient_steps_scale=0.5`, `alive_reward=1.65`,
   adaptive reload `alpha=3 kmin=200`) was stopped at 20k. It produced
   `sac_resnet18_10000_steps.zip` and `sac_resnet18_20000_steps.zip` during the run,
   but the branch was deleted immediately after review: at ~21.4k env steps, the last
   100 episodes averaged ~121 steps, the best recent episode was 179 steps, and
   `lap_count_mean` stayed at 0. Visually, it could pass the first bend on some
   layouts and fail again after a scene reload, indicating strong sensitivity to
   road-shadow layout. Conclusion: **ResNet18's random road-shadow generalization is
   far worse than DINOv2-S**; do not spend more randomtree experiments on frozen
   ResNet unless the goal is a negative baseline.

**Conclusions (where the limits actually are):**

- **The weave is not an encoder-*capacity* problem and not a reward-tunable habit;
  its root cause is still undetermined.** Evidence: `steer_delta` stays ~0.30
  identically across ViT-S and ViT-B, and is unmoved by linear (`w=0.5`) or
  squared (`w=2.0`) steer-change penalties. That rules out encoder *size/
  resolution* and rules out "a free bad habit reward can fix". It does **not**
  distinguish (a) a SAC control-layer high-gain/limit-cycle behavior from (b) a
  *shared* perception weakness of frozen single-frame DINOv2 (noisy/laggy lateral
  estimate the policy must high-frequency-correct). Both predict "identical across
  ViT sizes" and "penalty can't remove it". An earlier draft calling this "a SAC
  control artifact" overstated the evidence.
- **Replay-buffer probe (v2 50k, 300-step run; training-time, includes SAC
  exploration noise, no cte stored):** executed-steer **sign-flip rate 0.21**
  (~one reversal per 5 steps) — a *mid-frequency large-amplitude* sway, **not**
  step-by-step bang-bang (which would be ~0.5). mean|steer|≈0.47; per-step
  |Δsteer|≈0.29 ≈ 73% of the 0.4 cap. The **raw policy** output wants |Δsteer|≈0.51
  = **128% of the cap** — i.e. it continuously saturates `max_steering_diff`,
  which is doing real smoothing work.
- **Untried lever this surfaced:** since the policy pins the `max_steering_diff`
  clamp, *tightening* it (0.2→~0.1) would physically suppress the weave (a hard
  clamp the policy cannot ignore, unlike reward penalties) — at the risk of
  under-steering tight corners. Not yet tested.
- **Still to do for a clean diagnosis:** record per-step cte+steer from a
  *deterministic* eval (buffer has no cte and is exploration-noisy), and/or test
  frame-stacking (if the weave drops, it's a missing-temporal perception issue).
- **The ~50% trunc ceiling on random layouts is not broken by a bigger *frozen*
  encoder.** The remaining hard layouts are heavy-shadow cases where the frozen
  latent likely entangles road-shadow edges with lane edges. The next real lever
  is a **task-adapted encoder** — fine-tuning DINOv2, or a depth / driving-
  segmentation model that is inherently shadow/lighting-invariant — not more RL
  reward tuning (see §8).

> **Eval-clamp caveat (found 2026-05-29):** the paired-eval numbers above were
> produced by `eval_paired_randomized.py`, which until 2026-05-29 **hardcoded
> `max_steering_diff=0.15`** while training uses 0.2 — i.e. a tighter steering
> clamp than the policy trained under, biasing these results slower/more
> conservative. The bug is fixed (now a CLI arg, default 0.2). Independent
> `eval_loop_vae_sac.py` runs (default 0.2) were unaffected. Re-measured at the
> matched 0.2, v2 sits ~4/6 (§6.15 A), so these 3/6-ish figures slightly
> understate v2.

### 6.15 crop / steering-clamp / cte-tolerance probes (2026-05-29)

Three probes against the §6.14 ~50% random-layout ceiling, plus a quantified look
at the steering weave. All deterministic, paired on identical random layouts
(`eval_paired_randomized.py --models`, matched `max_steering_diff=0.2`), unless
noted. Narrative + exact commands: session log 2026-05-29.

**crop (top-crop the camera before the encoder): no significant benefit.**
A `--encoder-crop-top 40` cold start (drop the top third — sky/tree-canopy) was
trained to 130k (resume from 80k at `max_episode_steps=1200`). Paired vs v2:

| model (same 6 layouts) | Trunc | Mean CTE | Mean speed |
| --- | :---: | ---: | ---: |
| v2_50k (crop0) | 2/6 | 0.399 | 1.68 |
| v2_60k (crop0) | 2/6 | 0.403 | 1.89 |
| crop40_130k (crop40) | 4/6 | 0.472 | 1.96 |

crop40_130k leads — but it trained to 130k vs v2's 60k, so the lead conflates crop
with 70k extra steps. A same-steps control (crop40_60k vs v2_60k, same layouts):
v2 **4/6** (cte 0.388) vs crop40 **3/6** (cte 0.540) — at equal steps crop did not
win and ran wider. **Caveat:** that control is not blood-line-equal (v2_60k =
v1-10k-resume + adaptive; crop40_60k = pure cold start), so the gap may be
cold-start-vs-resume rather than crop. **Verdict: crop shows no significant
benefit; not enough evidence to call it harmful. Default to crop0; don't pursue
crop. The clean path to a stronger randomized model is to resume v2 (crop0) to
more steps, not crop.** (`crop40_130k` is a decent artifact — 4/6 — but kept only
as such, not promoted; it is crop=40.)

**Also corrected here:** "training-table looked like a plateau (80k→130k)" was
wrong — deterministic eval showed 80k=3/8 → 110k/130k=5/8 (at msd=0.2, separate
8-layout run). Another training-table ≠ deterministic-eval case.

**Steering weave: root cause still undetermined; one direct lever quantified.**
Buffer probe (v2 50k, 300-step run): executed-steer sign-flip rate **0.21** (~one
reversal / 5 steps — mid-frequency sway, *not* step-by-step bang-bang); raw policy
|Δsteer| ≈ **128% of the 0.4 clamp** (it continuously saturates `max_steering_diff`).
Tightening the clamp at eval (0.2→0.15) cut the *steering-action* amplitude
(|dsteer| 0.30→0.25) **but cte_std did not improve** (~0.51 both) and trunc dropped
sharply (crop40 130k: 5/8 → 0/8) with slower, wider driving. So a tighter clamp
suppresses the *visual* wheel motion, not the underlying lateral instability, and
costs generalization. The control-limit-cycle vs frozen-DINOv2-perception question
is still open (needs deterministic per-step cte/steer frequency analysis +
frame-stacking test).

**`max_cte_error=2.0` is somewhat strict for this track.** Re-running crop40_130k
at `--max-cte-error 2.5` (8 layouts) gave trunc **6/8** vs ~4/8 at 2.0. The two
rescued episodes had max_cte 2.27 / 2.31 (light line-touching just past 2.0); true
losses of control (max_cte 2.50, 2.84) were still not saved. So ~25% of "failures"
at 2.0 are near-misses, not real crashes — but whether cte 2.3 is truly
out-of-bounds depends on the track's drivable width (not measured), and rescued
episodes ride the boundary (high cte_std), i.e. line-touching survival rather than
better driving. Use a looser tolerance only if the track geometry justifies it,
and never compare trunc rates across different `max_cte_error`.

**Resuming past the plateau + a speed-vs-stability finding.** crop40 resumed
130k→200k (paired on the same 6 layouts, msd=0.2): trunc 130k=2/6, 170k=4/6,
200k=3/6 — peak-valley, no breakthrough; resuming past ~130k buys nothing on
trunc. But per-lap times show the real trade-off: on layouts each *can* finish,
**130k laps ~1-1.5 s faster** (steady ~12.0-12.5 s) than 170k/200k (~13.4-13.6 s).
So resume = "slow down to survive more layouts" (130k fast-but-fragile, 170k
stable-but-slower); neither dominates.

**Methodology: read lap time, not `mean_speed`, for driving speed.** `mean_speed`
is the instantaneous speed averaged over *every* eval step — **including the
low-speed / losing-control steps before a crash**. A checkpoint that crashes more
(130k: 2/6) gets its `mean_speed` dragged down by those steps (130k mean_speed
1.666, the *lowest* of the three) even though its actual lap pace is the
*fastest*. So `mean_speed` conflates "how fast it drives" with "how often it
crashes" and is misleading — use lap time for driving speed. Likewise, with only
~6 layouts the absolute trunc/mean_speed swing a lot batch-to-batch (the same
130k scored 4/6 in one batch, 2/6 in another); only within-batch relative
ordering is trustworthy.

### 6.16 Depth/seg probe for the task-adapted encoder (2026-06-12)

The cheap probe called for in §10 ("verify depth/seg output quality on the
cartoonish sim frames before investing"), now done. Script:
`tools/probe_depth_seg.py` — runs Depth Anything V2 Small plus a
Cityscapes-pretrained SegFormer on sim frames and writes
original|depth|seg|road-mask panels.

**Inputs.** 3 fixed-light frames from `data/vae_raw`, plus 5 random-light +
random-tree frames freshly captured (3 separate scene loads = 3 layouts,
`tools/collect_sim_frames.py`, cte-pid driving) including one worst-case frame
with dense dappled tree shadow across the road. Frames in
`data/probe_randomtree_frames/`, panels in `logs/probe_depth_seg*/` (both
local-only, gitignored).

**Segmentation — model size decides.** SegFormer-**B0** does not transfer to
the sim (road class 2-20% of pixels, inconsistent across frames, road confused
with sidewalk/terrain). SegFormer-**B4** transfers well: road 45-49% on
fixed-light frames, consistent, mask stops cleanly at the grass edge and
**covers the ego-car shadow region as road**. Under random tree shadows: still
clean at light/moderate shadow (road 53-55%), but the worst dappled-shadow
frame degrades (road 31%, holes in the mask under the densest patches) and one
frame bled road onto shadowed grass.

**Depth — shadow-immune but lane-blind.** Smooth ground-plane gradient on every
frame, tree trunks/cones pop out, and shadows are invisible by construction
(geometry, not appearance). But lane lines and the road/grass boundary are also
flat, hence invisible — depth alone cannot lane-keep; auxiliary channel only.

**Verdict.** The sim→real domain gap is manageable at B4 scale, and a
seg-derived road mask is exactly the shadow-invariant signal the frozen
encoders lack (§6.14-6.15 ceiling). Used raw, B4 would likely *raise* the ~50%
random-layout ceiling rather than remove it (worst-case dappled shadow still
breaks the mask), and B4 is too heavy for the ~20 Hz control loop. The
recommended path if this is picked up: **auto-label sim frames with B4
pseudo-labels and distill a small seg net** (data is free in the sim; also
solves inference cost), then feed mask (or mask+image) to SAC. Status:
probe positive, line validated as worth pursuing; no training done.
