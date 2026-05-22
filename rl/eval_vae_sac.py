from __future__ import annotations

import argparse
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
    FrozenVaeEncoder,
    MAX_CTE_ERROR,
    MAX_STEERING,
    MAX_STEERING_DIFF,
    MAX_THROTTLE,
    MIN_THROTTLE,
    N_COMMAND_HISTORY,
    RaffinRewardConfig,
    Z_SIZE,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("models/rl_vae_sac_raffin_v1/final_model.zip"))
    p.add_argument("--vae-model", type=Path, default=Path("models/vae_raffin_v1/best.pt"))
    p.add_argument("--env-id", default="donkey-generated-roads-v0")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9091)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-episode-steps", type=int, default=3000)
    p.add_argument("--min-throttle", type=float, default=MIN_THROTTLE)
    p.add_argument("--max-throttle", type=float, default=MAX_THROTTLE)
    p.add_argument("--deterministic", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = FrozenVaeEncoder(args.vae_model, device=device, z_size=Z_SIZE)
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
        reward_config=RaffinRewardConfig(),
    )
    env = TimeLimit(env, max_episode_steps=args.max_episode_steps)

    model = SAC.load(str(args.model), env=env, device=device)
    print(f"Loaded {args.model}")

    results = []
    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        total_r, steps, max_cte = 0.0, 0, 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += float(reward)
            steps += 1
            max_cte = max(max_cte, float(info.get("abs_cte", 0.0)))
        outcome = "TRUNC(3000)" if (truncated and not terminated) else "CRASH"
        print(f"ep {ep:2d}: steps={steps:4d} rew={total_r:7.1f} max_cte={max_cte:.2f} {outcome}")
        results.append({"steps": steps, "reward": total_r, "outcome": outcome})

    steps_arr = [r["steps"] for r in results]
    rew_arr = [r["reward"] for r in results]
    truncs = sum(1 for r in results if r["outcome"].startswith("TRUNC"))
    long_runs = sum(1 for r in results if r["steps"] >= 1000)
    print(
        "\n--- summary ---\n"
        f"episodes:     {len(results)}\n"
        f"steps  mean:  {np.mean(steps_arr):.0f}   median: {int(np.median(steps_arr))}   max: {max(steps_arr)}   min: {min(steps_arr)}\n"
        f"reward mean:  {np.mean(rew_arr):.1f}\n"
        f"ep>=1000:     {long_runs}/{len(results)}\n"
        f"truncated@3k: {truncs}/{len(results)}"
    )
    env.close()


if __name__ == "__main__":
    main()
