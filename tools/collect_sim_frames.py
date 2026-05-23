from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import gymnasium as gym
import gym_donkeycar  # noqa: F401 - registers donkey gym environments
import numpy as np
from PIL import Image


CAMERA_HEIGHT = 120
CAMERA_WIDTH = 160


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect camera frames from any gym-donkeycar scene.")
    parser.add_argument("--env-id", default="donkey-generated-track-v0")
    parser.add_argument("--host", default=os.environ.get("DONKEY_SIM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DONKEY_SIM_PORT", "9091")))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=30000)
    parser.add_argument("--max-episode-steps", type=int, default=3000)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--action-mode",
        choices=["random-smooth", "zero", "sweep", "cte-pid"],
        default="cte-pid",
    )
    parser.add_argument("--throttle", type=float, default=0.12)
    parser.add_argument("--steer-noise", type=float, default=0.12)
    parser.add_argument("--steer-limit", type=float, default=1.0)
    parser.add_argument("--pid-kp", type=float, default=7.0)
    parser.add_argument("--pid-ki", type=float, default=0.0)
    parser.add_argument("--pid-kd", type=float, default=20.0)
    parser.add_argument(
        "--cte-target",
        type=float,
        default=0.0,
        help="Target CTE for cte-pid mode. Non-zero values intentionally bias the car laterally.",
    )
    parser.add_argument("--step-delay", type=float, default=0.0)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    return parser.parse_args()


def obs_to_image(obs: np.ndarray) -> Image.Image:
    array = np.asarray(obs)
    if array.ndim == 3 and array.shape[:2] == (CAMERA_HEIGHT, CAMERA_WIDTH):
        image = array
    elif array.ndim == 3 and array.shape[1:] == (CAMERA_HEIGHT, CAMERA_WIDTH):
        image = np.transpose(array, (1, 2, 0))
    else:
        raise ValueError(f"unexpected observation shape {array.shape}")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(image, mode="RGB")


def choose_action(
    mode: str,
    step: int,
    steer: float,
    rng: random.Random,
    throttle: float,
    steer_noise: float,
    steer_limit: float,
    *,
    cte: float = 0.0,
    last_cte: float = 0.0,
    cte_integral: float = 0.0,
    cte_target: float = 0.0,
    pid_kp: float = 7.0,
    pid_ki: float = 0.0,
    pid_kd: float = 20.0,
) -> tuple[np.ndarray, float]:
    if mode == "zero":
        steer = 0.0
    elif mode == "sweep":
        steer = steer_limit * np.sin(step / 45.0)
    elif mode == "cte-pid":
        cte_error = cte - cte_target
        last_cte_error = last_cte - cte_target
        cte_delta = cte_error - last_cte_error
        steer = -(pid_kp * cte_error + pid_ki * cte_integral + pid_kd * cte_delta)
        steer = float(np.clip(steer, -steer_limit, steer_limit))
    else:
        steer = 0.92 * steer + rng.gauss(0.0, steer_noise)
        steer = float(np.clip(steer, -steer_limit, steer_limit))

    return np.array([steer, throttle], dtype=np.float32), steer


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    conf = {
        "host": args.host,
        "port": args.port,
        "cam_resolution": (CAMERA_HEIGHT, CAMERA_WIDTH, 3),
    }
    env = gym.make(args.env_id, conf=conf)

    frame_count = 0
    episode = 0
    steer = 0.0
    last_cte = 0.0
    cte_integral = 0.0
    obs, info = env.reset(seed=args.seed)
    start_time = time.time()

    with manifest_path.open("w", encoding="utf-8") as manifest:
        while frame_count < args.frames:
            episode += 1
            episode_step = 0
            terminated = truncated = False
            while frame_count < args.frames and not (terminated or truncated):
                cte = float(info.get("cte", 0.0))
                cte_integral += cte - args.cte_target
                action, steer = choose_action(
                    args.action_mode,
                    episode_step,
                    steer,
                    rng,
                    args.throttle,
                    args.steer_noise,
                    args.steer_limit,
                    cte=cte,
                    last_cte=last_cte,
                    cte_integral=cte_integral,
                    cte_target=args.cte_target,
                    pid_kp=args.pid_kp,
                    pid_ki=args.pid_ki,
                    pid_kd=args.pid_kd,
                )
                obs, reward, terminated, truncated, info = env.step(action)
                last_cte = cte
                episode_step += 1

                if episode_step % args.save_every == 0:
                    frame_count += 1
                    image_name = f"{args.env_id}_ep{episode:04d}_step{episode_step:05d}_frame{frame_count:07d}.jpg"
                    image_path = args.output_dir / image_name
                    obs_to_image(obs).save(image_path, quality=args.jpeg_quality)
                    row = {
                        "path": image_path.as_posix(),
                        "env_id": args.env_id,
                        "episode": episode,
                        "episode_step": episode_step,
                        "frame": frame_count,
                        "steer": float(action[0]),
                        "throttle": float(action[1]),
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "cte": float(info.get("cte", 0.0)),
                        "speed": float(info.get("speed", 0.0)),
                        "hit": info.get("hit"),
                    }
                    manifest.write(json.dumps(row, sort_keys=True) + "\n")

                if episode_step >= args.max_episode_steps:
                    truncated = True
                if args.step_delay > 0:
                    time.sleep(args.step_delay)

            if frame_count % 500 == 0 or frame_count >= args.frames:
                elapsed = max(time.time() - start_time, 1e-6)
                print(
                    f"frames={frame_count}/{args.frames} episode={episode} "
                    f"last_ep_steps={episode_step} rate={frame_count / elapsed:.1f} fps"
                )
            if frame_count < args.frames:
                obs, info = env.reset()
                steer = 0.0
                last_cte = 0.0
                cte_integral = 0.0

    env.close()
    print(f"wrote {frame_count} frames to {args.output_dir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
