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
    p.add_argument("--model-dir", type=Path, required=True,
                   help="Directory holding sac_{encoder}_{step}_steps.zip checkpoints.")
    p.add_argument("--steps", required=True,
                   help="Comma-separated checkpoint step numbers, e.g. 30000,40000,50000,60000")
    p.add_argument("--layouts", type=int, default=12,
                   help="Number of distinct random layouts; every checkpoint is run on each.")
    p.add_argument("--max-episode-steps", type=int, default=2000)
    p.add_argument("--min-throttle", type=float, default=0.2)
    p.add_argument("--max-throttle", type=float, default=0.7)
    p.add_argument("--max-cte-error", type=float, default=2.0)
    p.add_argument("--cte-target", type=float, default=0.0)
    return p.parse_args()


def run_episode(env, model, max_steps: int) -> dict:
    obs, _ = env.reset()  # reset_car only (scene reload is disabled on the wrapper)
    total_r = steps = 0
    max_cte = cte_sum = speed_sum = 0.0
    terminated = truncated = False
    ep_lap_times: list[float] = []
    prev_lap = 0
    last_seen_lap = 0.0
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_r += float(reward)
        steps += 1
        abs_cte = float(info.get("abs_cte", 0.0))
        max_cte = max(max_cte, abs_cte)
        cte_sum += abs_cte
        speed_sum += float(info.get("speed", 0.0))
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
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps_list = [int(s) for s in args.steps.split(",") if s.strip()]

    vae = make_encoder(args.encoder, device=device, vae_checkpoint=args.vae_model,
                       crop_top=args.encoder_crop_top)
    conf = {"host": args.host, "port": args.port, "cam_resolution": (CAMERA_HEIGHT, CAMERA_WIDTH, 3)}
    base_env = gym.make(args.env_id, conf=conf)
    env = DonkeyVaeSACEnv(
        base_env,
        vae=vae,
        min_throttle=args.min_throttle,
        max_throttle=args.max_throttle,
        max_steering=MAX_STEERING,
        max_steering_diff=MAX_STEERING_DIFF,
        n_command_history=N_COMMAND_HISTORY,
        reward_config=RaffinRewardConfig(
            max_cte_error=args.max_cte_error,
            cte_target=args.cte_target,
            cte_speed_penalty_weight=0.25,
        ),
        scene_reload_every=0,   # reloads are driven manually here (once per layout)
        scene_reload_alpha=0.0,
    )
    env = TimeLimit(env, max_episode_steps=args.max_episode_steps)
    controller = env.unwrapped.viewer
    scene = controller.handler.SceneToLoad

    models = {}
    for step in steps_list:
        path = args.model_dir / f"sac_{args.encoder}_{step}_steps.zip"
        if not path.exists():
            raise FileNotFoundError(path)
        models[step] = SAC.load(str(path), device=device)
        print(f"loaded {path}")

    # results[step] = list over layouts of per-episode dicts
    results = {step: [] for step in steps_list}
    trunc_matrix = []  # per layout: dict step->bool

    for layout in range(1, args.layouts + 1):
        if layout == 1:
            # reuse the scene gym.make already loaded; skips a redundant reload
            print(f"\n===== layout {layout}/{args.layouts}: using initial scene =====", flush=True)
        else:
            print(f"\n===== layout {layout}/{args.layouts}: reloading scene =====", flush=True)
            if not reload_scene(controller, scene):
                print("  WARNING: reload timed out; reusing current scene for this layout", flush=True)
        row = {}
        for step in steps_list:
            res = run_episode(env, models[step], args.max_episode_steps)
            results[step].append(res)
            row[step] = res["trunc"]
            tag = "TRUNC" if res["trunc"] else "OUT  "
            print(f"  [{step//1000:>3}k] {tag} steps={res['steps']:5d} "
                  f"mean_cte={res['mean_cte']:.3f} max_cte={res['max_cte']:.2f} "
                  f"laps={res['laps']} best_lap={res['best_lap']:.2f}s", flush=True)
        trunc_matrix.append(row)

    # ---- paired summary ----
    print("\n\n========== PAIRED SUMMARY (same layouts for all checkpoints) ==========")
    print(f"{'ckpt':>6} | {'trunc':>7} | {'mean_steps':>10} | {'mean_cte':>8} | "
          f"{'max_cte':>7} | {'mean_spd':>8} | {'laps':>5}")
    print("-" * 70)
    for step in steps_list:
        rs = results[step]
        n = len(rs)
        tr = sum(1 for r in rs if r["trunc"])
        mean_steps = np.mean([r["steps"] for r in rs])
        mean_cte = np.mean([r["mean_cte"] for r in rs])
        max_cte = max(r["max_cte"] for r in rs)
        mean_spd = np.mean([r["mean_speed"] for r in rs])
        laps = sum(r["laps"] for r in rs)
        print(f"{step//1000:>5}k | {tr:>3}/{n:<3} | {mean_steps:>10.0f} | {mean_cte:>8.3f} | "
              f"{max_cte:>7.2f} | {mean_spd:>8.3f} | {laps:>5}")

    # ---- per-layout truncation matrix (shows the comparison is paired) ----
    print("\n--- per-layout truncation (T=trunc, .=out) ---")
    header = "layout | " + " ".join(f"{s//1000:>3}k" for s in steps_list)
    print(header)
    for i, row in enumerate(trunc_matrix, 1):
        cells = " ".join(f"{'T' if row[s] else '.':>4}" for s in steps_list)
        print(f"{i:>6} | {cells}")

    env.close()


if __name__ == "__main__":
    main()
