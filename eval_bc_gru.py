import argparse
from collections import deque
import time
from pathlib import Path

import gymnasium as gym
import gym_donkeycar  # noqa: F401 - registers donkey environments
import numpy as np
import torch
from PIL import Image, ImageDraw

from eval_bc import clamp_action
from train_bc_gru import CnnGruDonkeyModel


def obs_to_chw(obs: np.ndarray) -> np.ndarray:
    image = np.asarray(obs, dtype=np.uint8)
    return np.transpose(image, (2, 0, 1))


def init_frame_buffer(obs: np.ndarray, sequence_length: int, frame_stride: int) -> deque[np.ndarray]:
    sequence_span = (sequence_length - 1) * frame_stride
    frame_buffer = deque(maxlen=sequence_span + 1)
    first_frame = obs_to_chw(obs)
    for _ in range(sequence_span + 1):
        frame_buffer.append(first_frame)
    return frame_buffer


def preprocess(
    frame_buffer: deque[np.ndarray],
    sequence_length: int,
    frame_stride: int,
    device: torch.device,
    repeat_current_frame: bool = False,
) -> torch.Tensor:
    if repeat_current_frame:
        frames = [frame_buffer[-1]] * sequence_length
    else:
        frames = list(frame_buffer)[::frame_stride]
    if len(frames) != sequence_length:
        raise RuntimeError(f"expected {sequence_length} frames, got {len(frames)}")
    sequence = np.stack(frames, axis=0)
    return torch.from_numpy(sequence).unsqueeze(0).to(device)


def clamp_steering(action: np.ndarray, steering_limit: float) -> np.ndarray:
    if steering_limit >= 1.0:
        return action
    clamped = action.copy()
    clamped[0] = float(np.clip(clamped[0], -steering_limit, steering_limit))
    return clamped


def save_spike_frames(
    frame_buffer: deque[np.ndarray],
    output_dir: Path,
    episode_index: int,
    local_step: int,
    pred: np.ndarray,
    action: np.ndarray,
    info: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = list(frame_buffer)
    thumbs = []
    for frame in frames:
        image = np.transpose(frame, (1, 2, 0))
        thumb = Image.fromarray(image).resize((240, 180))
        thumbs.append(thumb)

    width = 4 * 240
    height = 2 * 180
    grid = Image.new("RGB", (width, height), (20, 20, 20))
    for index, thumb in enumerate(thumbs[-8:]):
        grid.paste(thumb, ((index % 4) * 240, (index // 4) * 180))

    draw = ImageDraw.Draw(grid)
    label = (
        f"ep={episode_index} step={local_step} raw=[{pred[0]:+.3f},{pred[1]:.3f}] "
        f"action=[{action[0]:+.3f},{action[1]:.3f}] speed={info.get('speed')} cte={info.get('cte')}"
    )
    draw.rectangle((0, 0, width, 24), fill=(0, 0, 0))
    draw.text((4, 4), label, fill=(255, 255, 255))
    grid.save(output_dir / f"ep{episode_index:02d}_step{local_step:04d}_steer{pred[0]:+.3f}.jpg", quality=92)


def run_episode(env, model, device, sequence_length, frame_stride, args, episode_index: int, global_step_start: int):
    obs, info = env.reset()
    frame_buffer = init_frame_buffer(obs, sequence_length, frame_stride)
    total_reward = 0.0
    max_abs_cte = 0.0
    cte_sum = 0.0
    cte_count = 0

    for local_step in range(args.max_episode_steps):
        with torch.no_grad():
            pred = model(
                preprocess(frame_buffer, sequence_length, frame_stride, device, args.repeat_current_frame)
            ).detach().cpu().numpy()[0]
        action = clamp_action(pred, args.throttle_scale, args.throttle_min, args.throttle_max)
        action = clamp_steering(action, args.steering_limit)
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

        if abs(float(pred[0])) >= args.spike_steering_threshold:
            args.spike_count += 1
            if args.save_spike_frames:
                save_spike_frames(
                    frame_buffer,
                    args.save_spike_frames,
                    episode_index,
                    local_step,
                    pred,
                    action,
                    info,
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
                "spike_count": args.spike_count,
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
        "spike_count": args.spike_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--env-id", default="donkey-generated-roads-v0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--exit-scene-between-episodes", action="store_true")
    parser.add_argument("--scene-reload-delay", type=float, default=3.0)
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument("--throttle-scale", type=float, default=1.0)
    parser.add_argument("--throttle-min", type=float, default=0.0)
    parser.add_argument("--throttle-max", type=float, default=1.0)
    parser.add_argument("--steering-limit", type=float, default=1.0)
    parser.add_argument("--repeat-current-frame", action="store_true")
    parser.add_argument("--spike-steering-threshold", type=float, default=0.6)
    parser.add_argument("--save-spike-frames", type=Path)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    args.spike_count = 0

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint = torch.load(args.model, map_location=device)
    config = checkpoint["config"]
    sequence_length = int(config["sequence_length"])
    frame_stride = int(config["frame_stride"])
    model = CnnGruDonkeyModel(
        feature_dim=int(config["feature_dim"]),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
    ).to(device)
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
    print(f"sequence_length: {sequence_length} frame_stride: {frame_stride}")
    print(
        f"repeat_current_frame: {args.repeat_current_frame} steering_limit: {args.steering_limit} "
        f"spike_threshold: {args.spike_steering_threshold}"
    )
    print(f"env: {args.env_id} {args.host}:{args.port}")

    summaries = []
    global_step = 0
    for episode_index in range(args.episodes):
        env = gym.make(args.env_id, conf=conf)
        try:
            summary = run_episode(env, model, device, sequence_length, frame_stride, args, episode_index, global_step)
            summaries.append(summary)
            global_step += summary["steps"]
            print(
                f"episode_summary episode={episode_index:02d} steps={summary['steps']} "
                f"reward={summary['reward']:.2f} mean_abs_cte={summary['mean_abs_cte']:.3f} "
                f"max_abs_cte={summary['max_abs_cte']:.3f} terminated={summary['terminated']} "
                f"truncated={summary['truncated']} spikes_total={args.spike_count}"
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
            f"mean_abs_cte={mean_abs_cte.mean():.3f} max_abs_cte={max_abs_cte.max():.3f} "
            f"spikes_total={args.spike_count}"
        )


if __name__ == "__main__":
    main()
