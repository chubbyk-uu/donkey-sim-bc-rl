import argparse
from collections import deque
import hashlib
import random
import time
from pathlib import Path

import gymnasium as gym
import gym_donkeycar  # noqa: F401 - registers donkey environments
import numpy as np
import torch

from eval_bc import clamp_action
from train_bc_official_categorical import OfficialCategoricalModel


def obs_to_chw(obs: np.ndarray) -> np.ndarray:
    image = np.asarray(obs, dtype=np.uint8)
    return np.transpose(image, (2, 0, 1))


def preprocess(frame_buffer: deque[np.ndarray], device: torch.device) -> torch.Tensor:
    frame = frame_buffer[-1]
    return torch.from_numpy(frame).unsqueeze(0).to(device)


def init_frame_buffer(obs: np.ndarray) -> deque[np.ndarray]:
    frame_buffer = deque(maxlen=1)
    frame_buffer.append(obs_to_chw(obs))
    return frame_buffer


def set_eval_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def obs_checksum(obs: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(obs).tobytes()).hexdigest()[:12]


def decode_action(
    model,
    image_tensor: torch.Tensor,
    steering_centers: torch.Tensor,
    throttle_centers: torch.Tensor,
) -> np.ndarray:
    outputs = model(image_tensor)
    steer_idx = outputs["steer_logits"].argmax(dim=-1)
    throttle_idx = outputs["throttle_logits"].argmax(dim=-1)
    steer = steering_centers[steer_idx]
    throttle = throttle_centers[throttle_idx]
    return torch.stack([steer, throttle], dim=-1).detach().cpu().numpy()[0]


def run_episode(env, model, device, steering_centers, throttle_centers, args, episode_index: int, global_step_start: int):
    episode_seed = None if args.seed is None else args.seed if args.same_seed_each_episode else args.seed + episode_index
    if episode_seed is None:
        obs, info = env.reset()
    else:
        try:
            obs, info = env.reset(seed=episode_seed)
        except TypeError:
            obs, info = env.reset()
    frame_buffer = init_frame_buffer(obs)
    total_reward = 0.0
    max_abs_cte = 0.0
    cte_sum = 0.0
    cte_count = 0
    last_action = np.asarray([0.0, 0.0], dtype=np.float32)

    print(
        f"episode_start episode={episode_index:02d} seed={episode_seed} "
        f"obs_checksum={obs_checksum(obs)}"
    )

    for local_step in range(args.max_episode_steps):
        with torch.no_grad():
            pred = decode_action(model, preprocess(frame_buffer, device), steering_centers, throttle_centers)
        action = clamp_action(
            pred,
            args.throttle_scale,
            args.throttle_min,
            args.throttle_max,
            args.steering_limit,
            args.steering_scale,
        )
        if args.steer_smoothing > 0.0:
            action[0] = args.steer_smoothing * last_action[0] + (1.0 - args.steer_smoothing) * action[0]
        if args.throttle_smoothing > 0.0:
            action[1] = args.throttle_smoothing * last_action[1] + (1.0 - args.throttle_smoothing) * action[1]
        last_action = action.copy()

        obs, reward, terminated, truncated, info = env.step(action)
        frame_buffer.append(obs_to_chw(obs))
        total_reward += float(reward)

        cte = info.get("cte")
        if cte is not None:
            abs_cte = abs(float(cte))
            max_abs_cte = max(max_abs_cte, abs_cte)
            cte_sum += abs_cte
            cte_count += 1

        global_step = global_step_start + local_step
        if local_step % 25 == 0:
            print(
                f"episode={episode_index:02d} step={local_step:04d} global={global_step:04d} "
                f"action=[{action[0]:+.3f}, {action[1]:.3f}] raw=[{pred[0]:+.3f}, {pred[1]:.3f}] "
                f"reward={reward:.4f} total={total_reward:.2f} speed={info.get('speed')} "
                f"cte={info.get('cte')} hit={info.get('hit')}"
            )

        if terminated or truncated:
            return {
                "episode": episode_index,
                "steps": local_step + 1,
                "reward": total_reward,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "mean_abs_cte": cte_sum / max(1, cte_count),
                "max_abs_cte": max_abs_cte,
            }

        if args.sleep > 0:
            time.sleep(args.sleep)

    return {
        "episode": episode_index,
        "steps": args.max_episode_steps,
        "reward": total_reward,
        "terminated": False,
        "truncated": False,
        "mean_abs_cte": cte_sum / max(1, cte_count),
        "max_abs_cte": max_abs_cte,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--env-id", default="donkey-generated-roads-v0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--exit-scene-between-episodes", action="store_true")
    parser.add_argument("--scene-reload-delay", type=float, default=3.0)
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument("--throttle-scale", type=float, default=1.0)
    parser.add_argument("--throttle-min", type=float, default=0.0)
    parser.add_argument("--throttle-max", type=float, default=1.0)
    parser.add_argument("--steering-limit", type=float, default=1.0)
    parser.add_argument("--steering-scale", type=float, default=1.0)
    parser.add_argument("--steer-smoothing", type=float, default=0.0)
    parser.add_argument("--throttle-smoothing", type=float, default=0.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--same-seed-each-episode",
        action="store_true",
        help="Use exactly --seed for every episode instead of seed + episode_index.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        set_eval_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint = torch.load(args.model, map_location=device)
    if checkpoint.get("head_type") != "official_categorical_v1":
        raise ValueError(f"unsupported checkpoint head_type: {checkpoint.get('head_type')}")

    steering_centers_np = np.asarray(checkpoint["steering_centers"], dtype=np.float32)
    throttle_centers_np = np.asarray(checkpoint["throttle_centers"], dtype=np.float32)
    steering_centers = torch.tensor(steering_centers_np, dtype=torch.float32, device=device)
    throttle_centers = torch.tensor(throttle_centers_np, dtype=torch.float32, device=device)
    model = OfficialCategoricalModel(len(steering_centers_np), len(throttle_centers_np)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"device: {device}")
    print(f"model: {args.model}")
    print(f"steering_centers: {steering_centers_np.tolist()}")
    print(f"throttle_centers: {throttle_centers_np.tolist()}")
    print(
        f"decode_mode: argmax steering_scale: {args.steering_scale} steering_limit: {args.steering_limit} "
        f"steer_smoothing: {args.steer_smoothing} throttle_max: {args.throttle_max}"
    )
    print(f"env: {args.env_id} {args.host}:{args.port} seed={args.seed}")

    summaries = []
    global_step = 0
    for episode_index in range(args.episodes):
        episode_seed = None if args.seed is None else args.seed if args.same_seed_each_episode else args.seed + episode_index
        conf = {
            "host": args.host,
            "port": args.port,
            "cam_resolution": (120, 160, 3),
            "throttle_min": args.throttle_min,
            "throttle_max": args.throttle_max,
        }
        if episode_seed is not None:
            conf.update({"useSeed": True, "seed": episode_seed})
        env = gym.make(args.env_id, conf=conf)
        try:
            summary = run_episode(env, model, device, steering_centers, throttle_centers, args, episode_index, global_step)
            summaries.append(summary)
            global_step += summary["steps"]
            print(
                f"episode_summary episode={episode_index:02d} steps={summary['steps']} "
                f"reward={summary['reward']:.2f} mean_abs_cte={summary['mean_abs_cte']:.3f} "
                f"max_abs_cte={summary['max_abs_cte']:.3f} terminated={summary['terminated']} "
                f"truncated={summary['truncated']}"
            )
        finally:
            if args.exit_scene_between_episodes and hasattr(env.unwrapped, "viewer"):
                env.unwrapped.viewer.exit_scene()
                if args.scene_reload_delay > 0:
                    time.sleep(args.scene_reload_delay)
            env.close()

    if summaries:
        steps = np.asarray([summary["steps"] for summary in summaries], dtype=np.float32)
        rewards = np.asarray([summary["reward"] for summary in summaries], dtype=np.float32)
        mean_abs_cte = np.asarray([summary["mean_abs_cte"] for summary in summaries], dtype=np.float32)
        max_abs_cte = np.asarray([summary["max_abs_cte"] for summary in summaries], dtype=np.float32)
        print(
            "eval_summary "
            f"episodes={len(summaries)} steps_mean={steps.mean():.1f} steps_min={steps.min():.0f} "
            f"steps_max={steps.max():.0f} reward_mean={rewards.mean():.2f} "
            f"mean_abs_cte={mean_abs_cte.mean():.3f} max_abs_cte={max_abs_cte.max():.3f}"
        )


if __name__ == "__main__":
    main()
