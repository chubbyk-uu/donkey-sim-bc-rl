from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

try:
    from rl.train_vae_sac import (
        CappedDynamicGradientStepsCallback,
        DonkeyInfoCallback,
        FrozenVaeEncoder,
        MAX_STEERING,
        MAX_STEERING_DIFF,
        N_COMMAND_HISTORY,
        Z_SIZE,
        build_env,
    )
except ImportError:
    from train_vae_sac import (
        CappedDynamicGradientStepsCallback,
        DonkeyInfoCallback,
        FrozenVaeEncoder,
        MAX_STEERING,
        MAX_STEERING_DIFF,
        N_COMMAND_HISTORY,
        Z_SIZE,
        build_env,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VAE+SAC on the generated_track loop.")
    parser.add_argument("--env-id", default="donkey-generated-track-v0")
    parser.add_argument("--host", default=os.environ.get("DONKEY_SIM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DONKEY_SIM_PORT", "9091")))
    parser.add_argument("--vae-model", type=Path, default=Path("models/vae_loop_cones_v1/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/rl_loop_vae_sac_v1"))
    parser.add_argument("--resume-model", type=Path, default=None)
    parser.add_argument("--resume-replay-buffer", type=Path, default=None)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--min-throttle", type=float, default=0.2)
    parser.add_argument("--max-throttle", type=float, default=0.7)
    parser.add_argument("--max-steering", type=float, default=MAX_STEERING)
    parser.add_argument("--max-steering-diff", type=float, default=0.2)
    parser.add_argument("--n-command-history", type=int, default=N_COMMAND_HISTORY)
    parser.add_argument("--max-cte-error", type=float, default=2.0)
    parser.add_argument("--throttle-reward-weight", type=float, default=0.0)
    parser.add_argument("--reward-crash", type=float, default=-10.0)
    parser.add_argument("--crash-speed-weight", type=float, default=5.0)
    parser.add_argument("--alive-reward", type=float, default=1.5)
    parser.add_argument("--speed-reward-weight", type=float, default=0.15)
    parser.add_argument("--min-alive-speed", type=float, default=0.0)
    parser.add_argument("--progress-reward-weight", type=float, default=0.0)
    parser.add_argument("--cte-speed-penalty-weight", type=float, default=0.25,
                        help="Per-step penalty: -w * |cte| * speed. Punishes 'fast while off-line' "
                             "to bring corner-crash signal forward in time. v2 default 0.25.")

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=60_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--train-freq-unit", choices=["step", "episode"], default="episode")
    parser.add_argument("--gradient-steps", type=int, default=-1)
    parser.add_argument("--gradient-steps-cap", type=int, default=1000)
    parser.add_argument("--gradient-steps-min", type=int, default=500,
                        help="Floor on dynamic gradient_steps (used with train_freq=episode). "
                             "Guarantees at least this many off-policy updates per training cycle.")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", default="auto_0.1")
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--max-episode-steps", type=int, default=3000)
    parser.add_argument("--save-replay-buffer", action="store_true")
    parser.add_argument("--save-final-replay-buffer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    vae = FrozenVaeEncoder(args.vae_model, device=device, z_size=Z_SIZE)
    env = build_env(args, vae)
    env.action_space.seed(args.seed)

    policy_kwargs = {
        "net_arch": {"pi": [64, 64], "qf": [64, 64]},
    }

    if args.resume_model is not None:
        model = SAC.load(
            args.resume_model,
            env=env,
            device=device,
            tensorboard_log=str(args.output_dir / "tensorboard"),
        )
        if args.resume_replay_buffer is not None:
            model.load_replay_buffer(str(args.resume_replay_buffer))
            print(f"Loaded replay buffer from {args.resume_replay_buffer}")
        print(f"Resumed SAC model from {args.resume_model}")
    else:
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            train_freq=(args.train_freq, args.train_freq_unit),
            gradient_steps=args.gradient_steps,
            ent_coef=args.ent_coef,
            gamma=args.gamma,
            tau=args.tau,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(args.output_dir / "tensorboard"),
            verbose=1,
            seed=args.seed,
            device=device,
        )

    callbacks = [
        CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(args.output_dir),
            name_prefix="sac_loop_vae",
            save_replay_buffer=args.save_replay_buffer,
            save_vecnormalize=False,
        ),
        DonkeyInfoCallback(),
    ]
    if args.gradient_steps_cap is not None and args.gradient_steps_cap > 0:
        callbacks.append(CappedDynamicGradientStepsCallback(
            cap=args.gradient_steps_cap,
            floor=args.gradient_steps_min,
        ))

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=args.resume_model is None,
        )
    finally:
        model.save(str(args.output_dir / "final_model"))
        if args.save_final_replay_buffer:
            model.save_replay_buffer(str(args.output_dir / "final_replay_buffer"))
        env.close()


if __name__ == "__main__":
    main()
