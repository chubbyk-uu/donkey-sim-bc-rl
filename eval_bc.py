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


def clamp_action(action: np.ndarray, throttle_scale: float, throttle_min: float, throttle_max: float) -> np.ndarray:
    steer = float(np.clip(action[0], -1.0, 1.0))
    throttle = float(action[1]) * throttle_scale
    throttle = float(np.clip(throttle, throttle_min, throttle_max))
    return np.asarray([steer, throttle], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/bc_nvidia_generated_road_001/best.pt"))
    parser.add_argument("--env-id", default="donkey-generated-roads-v0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument("--throttle-scale", type=float, default=1.0)
    parser.add_argument("--throttle-min", type=float, default=0.0)
    parser.add_argument("--throttle-max", type=float, default=1.0)
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
    print(f"env: {args.env_id} {args.host}:{args.port}")
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
            action = clamp_action(pred, args.throttle_scale, args.throttle_min, args.throttle_max)
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
