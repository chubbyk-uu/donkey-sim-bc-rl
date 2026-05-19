import argparse
from collections import deque
import time
from pathlib import Path

import gymnasium as gym
import gym_donkeycar  # noqa: F401 - registers donkey environments
import numpy as np
import torch

from train_bc import NvidiaDonkeyModel


def obs_to_chw(obs: np.ndarray) -> np.ndarray:
    image = np.asarray(obs, dtype=np.float32)
    return np.transpose(image, (2, 0, 1))


def preprocess(frame_buffer: deque[np.ndarray], history: int, frame_stride: int, device: torch.device) -> torch.Tensor:
    frames = list(frame_buffer)[::frame_stride]
    if len(frames) != history:
        raise RuntimeError(f"expected {history} frames, got {len(frames)}")
    image_stack = np.concatenate(frames, axis=0)
    tensor = torch.from_numpy(image_stack).unsqueeze(0).to(device)
    return tensor


def clamp_action(
    action: np.ndarray,
    throttle_scale: float,
    throttle_min: float,
    throttle_max: float,
    steering_limit: float = 1.0,
    steering_scale: float = 1.0,
) -> np.ndarray:
    steer = float(action[0]) * steering_scale
    steer = float(np.clip(steer, -steering_limit, steering_limit))
    throttle = float(action[1]) * throttle_scale
    throttle = float(np.clip(throttle, throttle_min, throttle_max))
    return np.asarray([steer, throttle], dtype=np.float32)


def init_frame_buffer(obs: np.ndarray, history: int, frame_stride: int) -> deque[np.ndarray]:
    history_span = (history - 1) * frame_stride
    frame_buffer = deque(maxlen=history_span + 1)
    first_frame = obs_to_chw(obs)
    for _ in range(history_span + 1):
        frame_buffer.append(first_frame)
    return frame_buffer


def run_episode(env, model, device, history, frame_stride, args, episode_index: int, global_step_start: int):
    obs, info = env.reset()
    frame_buffer = init_frame_buffer(obs, history, frame_stride)
    total_reward = 0.0
    max_abs_cte = 0.0
    cte_sum = 0.0
    cte_count = 0
    last_action = np.asarray([0.0, 0.0], dtype=np.float32)

    for local_step in range(args.max_episode_steps):
        with torch.no_grad():
            pred = model(preprocess(frame_buffer, history, frame_stride, device)).detach().cpu().numpy()[0]
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
    parser.add_argument("--model", type=Path, default=Path("models/bc_nvidia_generated_road_001/best.pt"))
    parser.add_argument("--env-id", default="donkey-generated-roads-v0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--recreate-env-each-episode", action="store_true")
    parser.add_argument("--exit-scene-between-episodes", action="store_true")
    parser.add_argument("--scene-reload-delay", type=float, default=2.0)
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument("--throttle-scale", type=float, default=1.0)
    parser.add_argument("--throttle-min", type=float, default=0.0)
    parser.add_argument("--throttle-max", type=float, default=1.0)
    parser.add_argument("--steering-limit", type=float, default=1.0)
    parser.add_argument("--steering-scale", type=float, default=1.0)
    parser.add_argument("--steer-smoothing", type=float, default=0.0)
    parser.add_argument("--throttle-smoothing", type=float, default=0.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint = torch.load(args.model, map_location=device)
    model_config = checkpoint.get("config", {})
    history = int(model_config.get("history", 1))
    frame_stride = int(model_config.get("frame_stride", 1))
    model = NvidiaDonkeyModel(input_channels=history * 3).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    conf = {
        "host": args.host,
        "port": args.port,
        "cam_resolution": (120, 160, 3),
        "throttle_min": args.throttle_min,
        "throttle_max": args.throttle_max,
    }

    print(f"device: {device}")
    print(f"model: {args.model}")
    print(f"history: {history} frame_stride: {frame_stride}")
    print(
        f"steering_limit: {args.steering_limit} steer_smoothing: {args.steer_smoothing} "
        f"steering_scale: {args.steering_scale} throttle_smoothing: {args.throttle_smoothing}"
    )
    print(f"env: {args.env_id} {args.host}:{args.port}")

    if args.episodes > 0:
        summaries = []
        global_step = 0
        shared_env = None
        try:
            if not args.recreate_env_each_episode:
                shared_env = gym.make(args.env_id, conf=conf)
            for episode_index in range(args.episodes):
                env = gym.make(args.env_id, conf=conf) if args.recreate_env_each_episode else shared_env
                try:
                    summary = run_episode(env, model, device, history, frame_stride, args, episode_index, global_step)
                    summaries.append(summary)
                    global_step += summary["steps"]
                    print(
                        f"episode_summary episode={episode_index:02d} steps={summary['steps']} "
                        f"reward={summary['reward']:.2f} mean_abs_cte={summary['mean_abs_cte']:.3f} "
                        f"max_abs_cte={summary['max_abs_cte']:.3f} terminated={summary['terminated']} "
                        f"truncated={summary['truncated']}"
                    )
                finally:
                    if args.recreate_env_each_episode:
                        if args.exit_scene_between_episodes and hasattr(env.unwrapped, "viewer"):
                            env.unwrapped.viewer.exit_scene()
                            if args.scene_reload_delay > 0:
                                time.sleep(args.scene_reload_delay)
                        env.close()
        finally:
            if shared_env is not None:
                shared_env.close()

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
        return

    env = gym.make(args.env_id, conf=conf)

    total_reward = 0.0
    try:
        obs, info = env.reset()
        history_span = (history - 1) * frame_stride
        frame_buffer = deque(maxlen=history_span + 1)
        first_frame = obs_to_chw(obs)
        for _ in range(history_span + 1):
            frame_buffer.append(first_frame)

        for step in range(args.steps):
            with torch.no_grad():
                pred = model(preprocess(frame_buffer, history, frame_stride, device)).detach().cpu().numpy()[0]
            action = clamp_action(
                pred,
                args.throttle_scale,
                args.throttle_min,
                args.throttle_max,
                args.steering_limit,
                args.steering_scale,
            )
            obs, reward, terminated, truncated, info = env.step(action)
            frame_buffer.append(obs_to_chw(obs))
            total_reward += float(reward)

            if step % 25 == 0:
                print(
                    f"step={step:04d} action=[{action[0]:+.3f}, {action[1]:.3f}] "
                    f"raw=[{pred[0]:+.3f}, {pred[1]:.3f}] reward={reward:.4f} "
                    f"total={total_reward:.2f} speed={info.get('speed')} cte={info.get('cte')} "
                    f"hit={info.get('hit')}"
                )

            if terminated or truncated:
                print(f"episode ended at step={step}; terminated={terminated} truncated={truncated}")
                obs, info = env.reset()
                first_frame = obs_to_chw(obs)
                frame_buffer.clear()
                for _ in range(history_span + 1):
                    frame_buffer.append(first_frame)

            if args.sleep > 0:
                time.sleep(args.sleep)
    finally:
        env.close()


if __name__ == "__main__":
    main()
