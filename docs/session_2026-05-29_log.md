# Session Log: 2026-05-29 — crop / weave / max_cte probes on the domain-randomized loop

Follow-up to the 2026-05-28 domain-randomization work (experiment-log §6.14). Goal:
probe whether (a) a top crop, (b) tighter steering, or (c) a looser cte tolerance
move the ~50% random-layout truncate ceiling — and quantify the persistent
steering weave. Also stood up a second sim for parallel eval. Several tooling
bugs found and fixed along the way.

## 1. crop40 training (cold start 80k → resume to 130k)

`--encoder-crop-top 40` cold start (otherwise the §6.14 randomtree recipe: DINOv2-S,
hidden=256, batch=256, scale=0.5, adaptive reload alpha=3 kmin=200, raffin reward,
`reward_crash=-20`). 80k, then resumed (loading the 80k buffer) to 130k with
`--max-episode-steps 1200` (down from 2000) and the same adaptive reload (kmax
auto-tracks max_ep-1).

Training-table read of the resume looked like a plateau (ep_len flat ~690-790,
cte ~0.45, steer_delta ~0.31). **That read was wrong** — deterministic eval later
showed 80k→130k actually improved (see §3). Logged here as another instance of
"training-table ≠ deterministic eval."

## 2. Steering-weave investigation + a reference-frame correction

The left-right weave persisted (`abs_steer_delta_mean ≈ 0.30`) and was unmoved by
yesterday's steer-change penalties. Two corrections this session:

- **Reference-frame error fixed.** I'd called 0.30 "~15% of full range" — wrong.
  The right denominator is the per-step clamp `max_steering_diff=0.2`→±0.4, so 0.30
  is ~73% of the cap. The earlier "smooth should be 0.05-0.1" baseline was made up.
- **Buffer probe** (v2 50k buffer, 300-step run): executed-steer **sign-flip rate
  0.21** (~one reversal per 5 steps — mid-frequency sway, *not* step-by-step
  bang-bang); raw policy |Δsteer| ≈ 128% of the clamp (it continuously saturates
  `max_steering_diff`). cte is **not stored** in the SB3 buffer, so weave could
  only be measured on steer there.
- Conclusion corrected: earlier notes calling the weave "a SAC control artifact"
  overstated the evidence. ViT-S/ViT-B identical + penalty-immune only rules out
  encoder *capacity* and "reward-fixable habit"; control-limit-cycle vs shared
  frozen-DINOv2 lateral-perception weakness are **not** distinguished. Undetermined.

## 3. Parallel second sim + paired evals

Launched a second sim instance on port **9081** (first stays on 9091). GPU has
plenty of headroom (~14.7 GB free), so eval on 9081 runs fully parallel with
training/eval on 9091. This removes the "wait for the sim to free up" bottleneck.

**3-way paired** (same 6 random layouts, matched params msd=0.2/mt=0.2):

| model | trunc | mean_cte | mean_spd | laps |
| --- | :---: | ---: | ---: | ---: |
| v2_50k (crop0) | 2/6 | 0.399 | 1.68 | 21 |
| v2_60k (crop0) | 2/6 | 0.403 | 1.89 | 29 |
| crop40_130k (crop40) | 4/6 | 0.472 | 1.96 | 31 |

crop40_130k led — but it trained to 130k vs v2's 60k, so the lead conflates crop
with 70k extra steps.

**A — same-steps control** (crop40_60k vs v2_60k, same 6 layouts, msd=0.2):

| model | trunc | mean_cte | cte_std |
| --- | :---: | ---: | ---: |
| v2_60k (crop0) | 4/6 | 0.388 | 0.453 |
| crop40_60k (crop40) | 3/6 | 0.540 | 0.555 |

At equal steps crop40 did not win (3/6 vs 4/6) and ran wider (cte 0.54 vs 0.39).
**Caveat the user raised and I accepted:** this is *not* a blood-line-equal test —
v2_60k is `v1-10k-resume + adaptive`, crop40_60k is a pure cold start. So the
gap may be cold-start-vs-resume, not crop. Honest verdict: **crop shows no
significant benefit; not enough to call it harmful. Default to crop0 going
forward; don't pursue crop further.**

**B — max_cte_error scan** (crop40_130k, msd=0.2, 8 random layouts at cte tol 2.5):
trunc **6/8**. Per-episode max_cte: 1.57/1.58/1.70/1.97 (stable, <2.0), **2.27/2.31
(rescued by 2.5 — would OUT at 2.0)**, 2.50/2.84 (true loss of control, not saved).
⇒ vs ~4/8 at 2.0. So `max_cte_error=2.0` is somewhat strict for this track: ~25%
of "failures" are just light line-touching at cte 2.27-2.31, not real crashes.
Whether 2.3 is truly out-of-bounds depends on track width (not in the data).
Rescued episodes ride the boundary (high cte_std), so it's "line-touching survival,"
not better driving.

## 4. Bugs found + fixed (tooling)

- **`eval_paired_randomized.py` hardcoded `max_steering_diff = MAX_STEERING_DIFF`
  (0.15)** while training uses the CLI default 0.2. So every prior paired eval
  (the §6.14 v2 numbers, the ViT-B numbers) was run at a *tighter* clamp than
  training — biasing them slower/more conservative. Fixed: `--max-steering-diff`
  is now a CLI arg (default 0.2). The independent `eval_loop_vae_sac.py` (default
  0.2) was never affected, which explains part of the old "paired 3/6 vs
  independent 4/5" gap.
- **`--model-dir` was `required=True`**, so `--models` runs crashed at argparse.
  Fixed to optional (required only with `--steps`).
- Several env params in `eval_paired` were hardcoded (`max_steering`,
  `n_command_history`, `cte_speed_penalty_weight`); all now CLI-tunable with
  training-matched defaults.

## 5. Tooling added this session

- `eval_paired_randomized.py` **`--models` interface**: compare arbitrary
  checkpoints across directories and crops in one paired run
  (`label:zip_path:crop_top,...`). Works because DINOv2 z=384 for any crop, so obs
  dim/action space match and the env's encoder can be swapped per model on the
  same layout. Enabled the 3-way crop0-vs-crop40 comparison.
- **Weave metrics** added to both eval scripts: per-episode `|dsteer|`, `cte_std`,
  and `steer_period`/`cte_period` via local-extrema (diff sign-flip) counting,
  which is DC-immune (a steady off-center bias cancels, so only real oscillation
  is counted — handles the "car drives off to one side" case the user flagged).

## 6. Where things stand

- Random-layout ceiling still ~50-75% (depending on cte tolerance); crop didn't
  move it; tighter steering didn't (hurts trunc).
- Best randomized artifact: `crop40_130k` (3-way 4/6) — but it's crop=40 (not the
  recommended direction) and its edge is steps, not crop. The clean way to a
  stronger randomized model is to **resume v2 (crop0) to more steps**, not crop.
- Weave root cause still undetermined; next real lever remains a task-adapted
  encoder (§6.14 / §10), not reward/crop/clamp tuning.
