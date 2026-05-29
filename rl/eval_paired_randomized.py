from __future__ import annotations
"""Paired randomized eval: compare several checkpoints on the SAME set of random
layouts.

Independent per-checkpoint randomized eval (eval_loop_vae_sac.py with
--scene-reload-every 1) draws a different random layout set for each checkpoint,
so with few episodes the luck of the draw (easy vs heavy-shadow layouts) can
dominate the score and make checkpoints incomparable. This script removes that
confound: for each random layout it runs EVERY checkpoint on that same layout
(scene reloaded once per layout; only reset_car between checkpoints, so the trees
/ shadows / lighting stay fixed within a layout). Differences then reflect policy
quality, not which layouts each happened to draw.

Example:
    python rl/eval_paired_randomized.py \
      --env-id donkey-generated-track-v0 --encoder dinov2_vits14 \
      --model-dir models/rl_loop_dinov2_randomtree_v2 \
      --steps 30000,40000,50000,60000 \
      --layouts 12 --max-episode-steps 2000
"""

import argparse
import os
import sys
from pathlib import Path

import gymnasium as gym
import gym_donkeycar  # noqa: F401
import numpy as np
import torch
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from rl.train_vae_sac import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    DonkeyVaeSACEnv,
    MAX_STEERING,
    MAX_STEERING_DIFF,
    N_COMMAND_HISTORY,
    RaffinRewardConfig,
    make_encoder,
    reload_scene,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-id", default="donkey-generated-track-v0")
    p.add_argument("--host", default=os.environ.get("DONKEY_SIM_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("DONKEY_SIM_PORT", "9091")))
    p.add_argument("--encoder",
                   choices=["vae", "resnet18", "mobilenet_v3_small", "dinov2_vits14", "dinov2_vitb14"],
                   default="dinov2_vits14")
    p.add_argument("--encoder-crop-top", type=int, default=0)
    p.add_argument("--vae-model", type=Path, default=Path("models/vae_loop_cones_fixedlight_v1/best.pt"))
    p.add_argument("--model-dir", type=Path, default=None,
                   help="Directory holding sac_{encoder}_{step}_steps.zip checkpoints. "
                        "Required with --steps; ignored when --models is given.")
    p.add_argument("--steps", default=None,
                   help="Comma-separated checkpoint steps (all from --model-dir at "
                        "--encoder-crop-top), e.g. 30000,40000,50000,60000")
    p.add_argument("--models", default=None,
                   help="Explicit per-model specs for cross-dir / cross-crop comparison: "
                        "comma-separated 'label:zip_path:crop_top'. Overrides --steps. "
                        "All must share the same encoder family (same z_size; DINOv2 is "
                        "384 for any crop). Example: "
                        "'v2_50k:models/rl_loop_dinov2_randomtree_v2/sac_dinov2_vits14_50000_steps.zip:0,"
                        "crop40_130k:models/rl_loop_dinov2_randomtree_crop40_v1/sac_dinov2_vits14_130000_steps.zip:40'")
    p.add_argument("--layouts", type=int, default=12,
                   help="Number of distinct random layouts; every checkpoint is run on each.")
    p.add_argument("--max-episode-steps", type=int, default=2000)
    p.add_argument("--min-throttle", type=float, default=0.2)
    p.add_argument("--max-steering-diff", type=float, default=0.2,
                   help="Per-step steering-change clamp. MUST match training "
                        "(train_loop_vae_sac.py default is 0.2, NOT the 0.15 "
                        "MAX_STEERING_DIFF constant).")
    p.add_argument("--max-throttle", type=float, default=0.7)
    p.add_argument("--max-steering", type=float, default=MAX_STEERING,
                   help="Steering output range. MUST match training (default 1.0).")
    p.add_argument("--n-command-history", type=int, default=N_COMMAND_HISTORY,
                   help="Command-history length in the obs. MUST match training "
                        "(default 20) or the model will fail to load.")
    p.add_argument("--max-cte-error", type=float, default=2.0,
                   help="Termination threshold on |cte-target|. MUST match training.")
    p.add_argument("--cte-target", type=float, default=0.0)
    p.add_argument("--cte-speed-penalty-weight", type=float, default=0.25,
                   help="Only affects the (unreported) reward value, not trunc/cte/"
                        "steer metrics; exposed for completeness.")
    return p.parse_args()


def _osc_period(x) -> float:
    """Oscillation period (steps) via local-extrema counting on the first
    difference. DC-immune: a steady off-center bias or held turn cancels in the
    difference, so only the real back-and-forth wiggle is counted. inf if none."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 3:
        return float("inf")
    s = np.sign(np.diff(x))
    nz = s[s != 0]
    if len(nz) < 2:
        return float("inf")
    flips = int(np.sum(nz[1:] != nz[:-1]))
    return (2.0 * (len(x) - 1) / flips) if flips else float("inf")


def run_episode(env, model, max_steps: int) -> dict:
    obs, _ = env.reset()  # reset_car only (scene reload is disabled on the wrapper)
    total_r = steps = 0
    max_cte = cte_sum = speed_sum = 0.0
    terminated = truncated = False
    ep_lap_times: list[float] = []
    prev_lap = 0
    last_seen_lap = 0.0
    ep_steers: list[float] = []
    ep_steer_deltas: list[float] = []
    ep_ctes_signed: list[float] = []
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_r += float(reward)
        steps += 1
        abs_cte = float(info.get("abs_cte", 0.0))
        max_cte = max(max_cte, abs_cte)
        cte_sum += abs_cte
        speed_sum += float(info.get("speed", 0.0))
        ep_steers.append(float(info.get("rl_steer", 0.0)))
        ep_steer_deltas.append(abs(float(info.get("rl_steer_delta", 0.0))))
        ep_ctes_signed.append(float(info.get("cte", 0.0)))
        cur_lap = int(info.get("lap_count", 0))
        cur_last = float(info.get("last_lap_time", 0.0))
        if cur_lap > prev_lap and cur_last > 0 and cur_last != last_seen_lap:
            ep_lap_times.append(cur_last)
            last_seen_lap = cur_last
        prev_lap = cur_lap
    return {
        "steps": steps,
        "trunc": bool(truncated and not terminated),
        "mean_cte": cte_sum / max(1, steps),
        "max_cte": max_cte,
        "mean_speed": speed_sum / max(1, steps),
        "laps": prev_lap,
        "best_lap": min(ep_lap_times) if ep_lap_times else 0.0,
        "steer_delta": float(np.mean(ep_steer_deltas)) if ep_steer_deltas else 0.0,
        "cte_std": float(np.std(ep_ctes_signed)) if ep_ctes_signed else 0.0,
        "steer_period": _osc_period(ep_steers),
        "cte_period": _osc_period(ep_ctes_signed),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # build (label, zip_path, crop_top) specs from --models or --model-dir/--steps
    specs: list[tuple[str, Path, int]] = []
    if args.models:
        for item in args.models.split(","):
            label, path, crop = item.rsplit(":", 2)
            specs.append((label.strip(), Path(path.strip()), int(crop)))
    elif args.steps:
        if args.model_dir is None:
            raise SystemExit("--steps requires --model-dir")
        for s in args.steps.split(","):
            if s.strip():
                step = int(s.strip())
                specs.append((f"{step // 1000}k",
                              args.model_dir / f"sac_{args.encoder}_{step}_steps.zip",
                              args.encoder_crop_top))
    else:
        raise SystemExit("provide --models or --steps")
    labels = [lb for lb, _, _ in specs]
    crop_of = {lb: cr for lb, _, cr in specs}

    # one frozen encoder per distinct crop (DINOv2 z_size is identical across crops,
    # so obs dim / action space are the same and we can swap the env's encoder per model)
    encoders = {}
    for _, _, cr in specs:
        if cr not in encoders:
            encoders[cr] = make_encoder(args.encoder, device=device,
                                        vae_checkpoint=args.vae_model, crop_top=cr)

    conf = {"host": args.host, "port": args.port, "cam_resolution": (CAMERA_HEIGHT, CAMERA_WIDTH, 3)}
    base_env = gym.make(args.env_id, conf=conf)
    dk_env = DonkeyVaeSACEnv(
        base_env,
        vae=encoders[specs[0][2]],
        min_throttle=args.min_throttle,
        max_throttle=args.max_throttle,
        max_steering=args.max_steering,
        max_steering_diff=args.max_steering_diff,
        n_command_history=args.n_command_history,
        reward_config=RaffinRewardConfig(
            max_cte_error=args.max_cte_error,
            cte_target=args.cte_target,
            cte_speed_penalty_weight=args.cte_speed_penalty_weight,
        ),
        scene_reload_every=0,   # reloads are driven manually here (once per layout)
        scene_reload_alpha=0.0,
    )
    env = TimeLimit(dk_env, max_episode_steps=args.max_episode_steps)
    controller = env.unwrapped.viewer
    scene = controller.handler.SceneToLoad

    models = {}
    for label, path, cr in specs:
        if not path.exists():
            raise FileNotFoundError(path)
        models[label] = SAC.load(str(path), device=device)
        print(f"loaded [{label}] crop={cr} {path}")

    results = {lb: [] for lb in labels}
    trunc_matrix = []  # per layout: dict label->bool

    for layout in range(1, args.layouts + 1):
        if layout == 1:
            # reuse the scene gym.make already loaded; skips a redundant reload
            print(f"\n===== layout {layout}/{args.layouts}: using initial scene =====", flush=True)
        else:
            print(f"\n===== layout {layout}/{args.layouts}: reloading scene =====", flush=True)
            if not reload_scene(controller, scene):
                print("  WARNING: reload timed out; reusing current scene for this layout", flush=True)
        row = {}
        for label in labels:
            dk_env.vae = encoders[crop_of[label]]   # swap encoder crop to this model's
            res = run_episode(env, models[label], args.max_episode_steps)
            results[label].append(res)
            row[label] = res["trunc"]
            tag = "TRUNC" if res["trunc"] else "OUT  "
            print(f"  [{label:>12}] {tag} steps={res['steps']:5d} "
                  f"mean_cte={res['mean_cte']:.3f} max_cte={res['max_cte']:.2f} "
                  f"laps={res['laps']} best_lap={res['best_lap']:.2f}s", flush=True)
        trunc_matrix.append(row)

    # ---- paired summary ----
    print("\n\n========== PAIRED SUMMARY (same layouts for all models) ==========")
    print(f"{'model':>12} | {'trunc':>7} | {'mean_steps':>10} | {'mean_cte':>8} | "
          f"{'max_cte':>7} | {'mean_spd':>8} | {'laps':>5}")
    print("-" * 78)
    for label in labels:
        rs = results[label]
        n = len(rs)
        tr = sum(1 for r in rs if r["trunc"])
        mean_steps = np.mean([r["steps"] for r in rs])
        mean_cte = np.mean([r["mean_cte"] for r in rs])
        max_cte = max(r["max_cte"] for r in rs)
        mean_spd = np.mean([r["mean_speed"] for r in rs])
        laps = sum(r["laps"] for r in rs)
        print(f"{label:>12} | {tr:>3}/{n:<3} | {mean_steps:>10.0f} | {mean_cte:>8.3f} | "
              f"{max_cte:>7.2f} | {mean_spd:>8.3f} | {laps:>5}")

    # ---- weave summary (steering oscillation; lower |dsteer|/cte_std = calmer) ----
    print(f"\n{'model':>12} | {'|dsteer|':>8} | {'cte_std':>8} | {'steer_period':>12} | {'cte_period':>10}")
    print("-" * 62)
    for label in labels:
        rs = results[label]
        sd = np.mean([r["steer_delta"] for r in rs])
        cstd = np.mean([r["cte_std"] for r in rs])
        fsp = [r["steer_period"] for r in rs if np.isfinite(r["steer_period"])]
        fcp = [r["cte_period"] for r in rs if np.isfinite(r["cte_period"])]
        sp = np.mean(fsp) if fsp else float("inf")
        cp = np.mean(fcp) if fcp else float("inf")
        print(f"{label:>12} | {sd:>8.3f} | {cstd:>8.3f} | {sp:>9.1f}st | {cp:>7.1f}st")

    # ---- per-layout truncation matrix (shows the comparison is paired) ----
    print("\n--- per-layout truncation (T=trunc, .=out) ---")
    header = "layout | " + " ".join(f"{lb:>12}" for lb in labels)
    print(header)
    for i, row in enumerate(trunc_matrix, 1):
        cells = " ".join(f"{'T' if row[lb] else '.':>12}" for lb in labels)
        print(f"{i:>6} | {cells}")

    env.close()


if __name__ == "__main__":
    main()
