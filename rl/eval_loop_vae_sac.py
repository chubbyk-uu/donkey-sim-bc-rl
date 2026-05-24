from __future__ import annotations

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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VAE+SAC on the generated_track loop.")
    parser.add_argument("--model", type=Path, default=Path("models/rl_loop_vae_sac_v1/final_model.zip"))
    parser.add_argument("--encoder", choices=["vae", "resnet18", "mobilenet_v3_small"], default="vae",
                        help="Image encoder. Must match what the SAC model was trained with.")
    parser.add_argument("--encoder-crop-top", type=int, default=0,
                        help="Top-row pixels to crop before encoding (ResNet/MobileNet only). "
                             "Must match what the model was trained with. Use 40 for the older "
                             "v4 ResNet checkpoint; use 0 for new-style ResNet runs.")
    parser.add_argument("--vae-model", type=Path, default=Path("models/vae_loop_cones_fixedlight_v1/best.pt"),
                        help="Only used when --encoder=vae.")
    parser.add_argument("--env-id", default="donkey-generated-track-v0")
    parser.add_argument("--host", default=os.environ.get("DONKEY_SIM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DONKEY_SIM_PORT", "9091")))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=3000)
    parser.add_argument("--min-throttle", type=float, default=0.2)
    parser.add_argument("--max-throttle", type=float, default=0.7)
    parser.add_argument("--max-steering", type=float, default=MAX_STEERING)
    parser.add_argument("--max-steering-diff", type=float, default=0.2)
    parser.add_argument("--max-cte-error", type=float, default=2.0)
    parser.add_argument("--throttle-reward-weight", type=float, default=0.0)
    parser.add_argument("--reward-crash", type=float, default=-10.0)
    parser.add_argument("--crash-speed-weight", type=float, default=5.0)
    parser.add_argument("--alive-reward", type=float, default=1.5)
    parser.add_argument("--speed-reward-weight", type=float, default=0.15)
    parser.add_argument("--min-alive-speed", type=float, default=0.0)
    parser.add_argument("--progress-reward-weight", type=float, default=0.0)
    parser.add_argument("--cte-speed-penalty-weight", type=float, default=0.25)
    parser.add_argument("--deterministic", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = make_encoder(args.encoder, device=device, vae_checkpoint=args.vae_model,
                       crop_top=args.encoder_crop_top)
    print(f"Encoder: {args.encoder}  z_size={vae.z_size}  crop_top={args.encoder_crop_top}")
    conf = {"host": args.host, "port": args.port, "cam_resolution": (CAMERA_HEIGHT, CAMERA_WIDTH, 3)}
    base_env = gym.make(args.env_id, conf=conf)
    env = DonkeyVaeSACEnv(
        base_env,
        vae=vae,
        min_throttle=args.min_throttle,
        max_throttle=args.max_throttle,
        max_steering=args.max_steering,
        max_steering_diff=args.max_steering_diff,
        n_command_history=N_COMMAND_HISTORY,
        reward_config=RaffinRewardConfig(
            max_cte_error=args.max_cte_error,
            throttle_reward_weight=args.throttle_reward_weight,
            reward_crash=args.reward_crash,
            crash_speed_weight=args.crash_speed_weight,
            alive_reward=args.alive_reward,
            speed_reward_weight=args.speed_reward_weight,
            min_alive_speed=args.min_alive_speed,
            progress_reward_weight=args.progress_reward_weight,
            cte_speed_penalty_weight=args.cte_speed_penalty_weight,
        ),
    )
    env = TimeLimit(env, max_episode_steps=args.max_episode_steps)

    model = SAC.load(str(args.model), env=env, device=device)
    print(f"Loaded {args.model}")

    results = []
    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        total_r, steps, max_cte, cte_sum, speed_sum, progress_sum = 0.0, 0, 0.0, 0.0, 0.0, 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += float(reward)
            steps += 1
            abs_cte = float(info.get("abs_cte", 0.0))
            max_cte = max(max_cte, abs_cte)
            cte_sum += abs_cte
            speed_sum += float(info.get("speed", 0.0))
            progress_sum += float(info.get("delta_pos_distance", 0.0))
        outcome = f"TRUNC({args.max_episode_steps})" if (truncated and not terminated) else "OUT"
        mean_cte = cte_sum / max(1, steps)
        mean_speed = speed_sum / max(1, steps)
        print(
            f"ep {ep:2d}: steps={steps:5d} rew={total_r:8.1f} "
            f"mean_speed={mean_speed:.3f} progress={progress_sum:.1f} "
            f"mean_cte={mean_cte:.3f} max_cte={max_cte:.2f} {outcome}"
        )
        results.append(
            {
                "steps": steps,
                "reward": total_r,
                "mean_speed": mean_speed,
                "mean_cte": mean_cte,
                "max_cte": max_cte,
                "progress": progress_sum,
                "outcome": outcome,
            }
        )

    steps_arr = np.asarray([r["steps"] for r in results], dtype=np.float32)
    rew_arr = np.asarray([r["reward"] for r in results], dtype=np.float32)
    speed_arr = np.asarray([r["mean_speed"] for r in results], dtype=np.float32)
    cte_arr = np.asarray([r["mean_cte"] for r in results], dtype=np.float32)
    max_cte_arr = np.asarray([r["max_cte"] for r in results], dtype=np.float32)
    progress_arr = np.asarray([r["progress"] for r in results], dtype=np.float32)
    truncs = sum(1 for r in results if r["outcome"].startswith("TRUNC"))
    print(
        "\n--- summary ---\n"
        f"episodes:       {len(results)}\n"
        f"steps mean:     {steps_arr.mean():.0f}   median: {int(np.median(steps_arr))}   max: {int(steps_arr.max())}   min: {int(steps_arr.min())}\n"
        f"reward mean:    {rew_arr.mean():.1f}\n"
        f"speed mean:     {speed_arr.mean():.3f}\n"
        f"progress mean:  {progress_arr.mean():.1f}\n"
        f"mean_abs_cte:   {cte_arr.mean():.3f}\n"
        f"max_abs_cte:    {max_cte_arr.max():.3f}\n"
        f"truncated:      {truncs}/{len(results)}"
    )
    env.close()


if __name__ == "__main__":
    main()
