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


def _osc_period(x) -> tuple[float, float]:
    """Oscillation period via local-extrema counting (sign flips of the first
    difference). Robust to a DC offset / slow drift — i.e. if the car drives
    off-center or holds a steady turn, that bias cancels in the difference, so
    only the actual back-and-forth wiggle is counted. Returns
    (period_in_steps, extrema_per_step); period=inf if no oscillation.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 3:
        return float("inf"), 0.0
    d = np.diff(x)
    s = np.sign(d)
    nz = s[s != 0]                      # drop flat segments
    if len(nz) < 2:
        return float("inf"), 0.0
    flips = int(np.sum(nz[1:] != nz[:-1]))   # local extrema (turning points)
    if flips == 0:
        return float("inf"), 0.0
    extrema_rate = flips / (len(x) - 1)
    period = 2.0 * (len(x) - 1) / flips        # 2 extrema (one peak+one trough) per cycle
    return period, extrema_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VAE+SAC on the generated_track loop.")
    parser.add_argument("--model", type=Path, default=Path("models/rl_loop_vae_sac_v1/final_model.zip"))
    parser.add_argument("--encoder",
                        choices=["vae", "resnet18", "mobilenet_v3_small", "dinov2_vits14", "dinov2_vitb14"],
                        default="vae",
                        help="Image encoder. Must match what the SAC model was trained with.")
    parser.add_argument("--encoder-crop-top", type=int, default=0,
                        help="Top-row pixels to crop before encoding. Applies to ALL frozen "
                             "pretrained encoders (DINOv2, ResNet, MobileNet). The VAE always "
                             "crops by its own fixed MARGIN_TOP and is unaffected by this flag. "
                             "MUST match what the model was trained with. Use 40 for the older "
                             "v4 ResNet checkpoint; use 0 for the standard DINOv2/ResNet runs.")
    parser.add_argument("--vae-model", type=Path, default=Path("models/vae_loop_cones_fixedlight_v1/best.pt"),
                        help="Only used when --encoder=vae.")
    parser.add_argument("--env-id", default="donkey-generated-track-v0")
    parser.add_argument("--host", default=os.environ.get("DONKEY_SIM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DONKEY_SIM_PORT", "9091")))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=3000)
    parser.add_argument("--scene-reload-every", type=int, default=0,
                        help="Reload the sim scene every N episodes so each episode "
                             "gets a fresh per-scene randomization (trees/shadows/"
                             "lighting). Use 1 to randomize every eval episode. "
                             "0 = off (default).")
    parser.add_argument("--min-throttle", type=float, default=0.2)
    parser.add_argument("--max-throttle", type=float, default=0.7)
    parser.add_argument("--max-steering", type=float, default=MAX_STEERING)
    parser.add_argument("--max-steering-diff", type=float, default=0.2)
    parser.add_argument("--max-cte-error", type=float, default=2.0,
                        help="Max allowed |cte - cte_target| before episode terminates.")
    parser.add_argument("--cte-target", type=float, default=0.0,
                        help="The cte value treated as 'lane center'. Use 3.5 for "
                             "mountain-track (right lane spawn) or keep 0 for generated-track.")
    parser.add_argument("--reward-crash", type=float, default=-10.0)
    parser.add_argument("--crash-speed-weight", type=float, default=5.0)
    parser.add_argument("--alive-reward", type=float, default=1.5)
    parser.add_argument("--speed-reward-weight", type=float, default=0.15)
    parser.add_argument("--min-alive-speed", type=float, default=0.0)
    parser.add_argument("--alive-scale-floor", type=float, default=0.0,
                        help="Match training value (only matters if min_alive_speed>0).")
    parser.add_argument("--lap-completion-bonus", type=float, default=0.0,
                        help="Match training value (affects reported reward only).")
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
            reward_crash=args.reward_crash,
            crash_speed_weight=args.crash_speed_weight,
            alive_reward=args.alive_reward,
            speed_reward_weight=args.speed_reward_weight,
            min_alive_speed=args.min_alive_speed,
            alive_scale_floor=args.alive_scale_floor,
            lap_completion_bonus=args.lap_completion_bonus,
            cte_speed_penalty_weight=args.cte_speed_penalty_weight,
            cte_target=args.cte_target,
        ),
        scene_reload_every=args.scene_reload_every,
    )
    env = TimeLimit(env, max_episode_steps=args.max_episode_steps)

    model = SAC.load(str(args.model), env=env, device=device)
    print(f"Loaded {args.model}")

    results = []
    all_lap_times: list[float] = []
    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        total_r, steps, max_cte, cte_sum, speed_sum = 0.0, 0, 0.0, 0.0, 0.0
        terminated = truncated = False
        ep_lap_times: list[float] = []
        prev_lap_count = 0
        last_seen_lap_time = 0.0
        ep_steers: list[float] = []
        ep_steer_deltas: list[float] = []
        ep_ctes_signed: list[float] = []
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=args.deterministic)
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
            cur_lap_count = int(info.get("lap_count", 0))
            cur_last_lap = float(info.get("last_lap_time", 0.0))
            if cur_lap_count > prev_lap_count and cur_last_lap > 0 and cur_last_lap != last_seen_lap_time:
                ep_lap_times.append(cur_last_lap)
                last_seen_lap_time = cur_last_lap
            prev_lap_count = cur_lap_count
        outcome = f"TRUNC({args.max_episode_steps})" if (truncated and not terminated) else "OUT"
        mean_cte = cte_sum / max(1, steps)
        mean_speed = speed_sum / max(1, steps)
        lap_count = prev_lap_count
        best_lap = min(ep_lap_times) if ep_lap_times else 0.0
        mean_lap = float(np.mean(ep_lap_times)) if ep_lap_times else 0.0
        all_lap_times.extend(ep_lap_times)
        mean_steer_delta = float(np.mean(ep_steer_deltas)) if ep_steer_deltas else 0.0
        cte_std = float(np.std(ep_ctes_signed)) if ep_ctes_signed else 0.0
        steer_period, _ = _osc_period(ep_steers)
        cte_period, _ = _osc_period(ep_ctes_signed)
        print(
            f"ep {ep:2d}: steps={steps:5d} rew={total_r:8.1f} "
            f"mean_speed={mean_speed:.3f} "
            f"mean_cte={mean_cte:.3f} max_cte={max_cte:.2f} "
            f"laps={lap_count} best_lap={best_lap:.2f}s mean_lap={mean_lap:.2f}s {outcome}\n"
            f"        weave: |dsteer|={mean_steer_delta:.3f} steer_period={steer_period:.1f}st "
            f"cte_std={cte_std:.3f} cte_period={cte_period:.1f}st"
        )
        results.append(
            {
                "steps": steps,
                "reward": total_r,
                "mean_speed": mean_speed,
                "mean_cte": mean_cte,
                "max_cte": max_cte,
                "outcome": outcome,
                "lap_count": lap_count,
                "best_lap": best_lap,
                "mean_lap": mean_lap,
                "steer_delta": mean_steer_delta,
                "cte_std": cte_std,
                "steer_period": steer_period,
                "cte_period": cte_period,
            }
        )

    steps_arr = np.asarray([r["steps"] for r in results], dtype=np.float32)
    rew_arr = np.asarray([r["reward"] for r in results], dtype=np.float32)
    speed_arr = np.asarray([r["mean_speed"] for r in results], dtype=np.float32)
    cte_arr = np.asarray([r["mean_cte"] for r in results], dtype=np.float32)
    max_cte_arr = np.asarray([r["max_cte"] for r in results], dtype=np.float32)
    truncs = sum(1 for r in results if r["outcome"].startswith("TRUNC"))
    total_laps = sum(r["lap_count"] for r in results)
    overall_best = min(all_lap_times) if all_lap_times else 0.0
    overall_mean_lap = float(np.mean(all_lap_times)) if all_lap_times else 0.0
    sd_mean = float(np.mean([r["steer_delta"] for r in results]))
    ctestd_mean = float(np.mean([r["cte_std"] for r in results]))
    finite_sp = [r["steer_period"] for r in results if np.isfinite(r["steer_period"])]
    finite_cp = [r["cte_period"] for r in results if np.isfinite(r["cte_period"])]
    sp_mean = float(np.mean(finite_sp)) if finite_sp else float("inf")
    cp_mean = float(np.mean(finite_cp)) if finite_cp else float("inf")
    print(
        "\n--- summary ---\n"
        f"episodes:       {len(results)}\n"
        f"steps mean:     {steps_arr.mean():.0f}   median: {int(np.median(steps_arr))}   max: {int(steps_arr.max())}   min: {int(steps_arr.min())}\n"
        f"reward mean:    {rew_arr.mean():.1f}\n"
        f"speed mean:     {speed_arr.mean():.3f}\n"
        f"mean_abs_cte:   {cte_arr.mean():.3f}\n"
        f"max_abs_cte:    {max_cte_arr.max():.3f}\n"
        f"truncated:      {truncs}/{len(results)}\n"
        f"total_laps:     {total_laps}   best_lap: {overall_best:.2f}s   mean_lap: {overall_mean_lap:.2f}s\n"
        f"weave:          |dsteer|={sd_mean:.3f}  cte_std={ctestd_mean:.3f}  "
        f"steer_period={sp_mean:.1f}st  cte_period={cp_mean:.1f}st  "
        f"(lower |dsteer|/cte_std = calmer; larger period = slower sway)"
    )
    env.close()


if __name__ == "__main__":
    main()
