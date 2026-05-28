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
        MAX_STEERING,
        MAX_STEERING_DIFF,
        N_COMMAND_HISTORY,
        build_env,
        make_encoder,
    )
except ImportError:
    from train_vae_sac import (
        CappedDynamicGradientStepsCallback,
        DonkeyInfoCallback,
        MAX_STEERING,
        MAX_STEERING_DIFF,
        N_COMMAND_HISTORY,
        build_env,
        make_encoder,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VAE+SAC on the generated_track loop.")
    parser.add_argument("--env-id", default="donkey-generated-track-v0")
    parser.add_argument("--host", default=os.environ.get("DONKEY_SIM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DONKEY_SIM_PORT", "9091")))
    parser.add_argument("--encoder",
                        choices=["vae", "resnet18", "mobilenet_v3_small", "dinov2_vits14", "dinov2_vitb14"],
                        default="vae",
                        help="Image encoder. 'vae' uses --vae-model checkpoint; "
                             "'resnet18' / 'mobilenet_v3_small' use frozen ImageNet-pretrained weights; "
                             "'dinov2_vits14' / 'dinov2_vitb14' use frozen DINOv2 ViT-S(384)/B(768).")
    parser.add_argument("--encoder-crop-top", type=int, default=0,
                        help="Top-row pixels to crop before encoding (ResNet/MobileNet only; VAE "
                             "uses its own fixed crop). Default 0 (no crop) reduces aspect ratio "
                             "distortion when resizing to 224x224. Set to 40 to reproduce the "
                             "older v4 ResNet behavior.")
    parser.add_argument("--vae-model", type=Path, default=Path("models/vae_loop_cones_fixedlight_v1/best.pt"),
                        help="Only used when --encoder=vae.")
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
    parser.add_argument("--max-cte-error", type=float, default=2.0,
                        help="Max allowed |cte - cte_target| before episode terminates.")
    parser.add_argument("--cte-target", type=float, default=0.0,
                        help="The cte value treated as 'lane center'. On generated-track this "
                             "is 0 (right-lane center). On mountain-track spawn is at cte~3.54, "
                             "so set --cte-target 3.5 to make the agent drive in the right lane "
                             "(termination and cte_speed_penalty both measure |cte - target|).")
    parser.add_argument("--reward-crash", type=float, default=-10.0)
    parser.add_argument("--crash-speed-weight", type=float, default=5.0)
    parser.add_argument("--alive-reward", type=float, default=1.5)
    parser.add_argument("--speed-reward-weight", type=float, default=0.15)
    parser.add_argument("--min-alive-speed", type=float, default=0.0)
    parser.add_argument("--alive-scale-floor", type=float, default=0.0,
                        help="Minimum alive_scale at speed=0 when min_alive_speed>0. "
                             "Default 0 means alive_scale ramps from 0 to 1 linearly (sharp). "
                             "Setting 0.5 makes it ramp from 0.5 to 1 (softer), so the agent "
                             "keeps half its alive reward even when slowing for corners.")
    parser.add_argument("--lap-completion-bonus", type=float, default=0.0,
                        help="Discrete reward added when info['lap_count'] increments. "
                             "0 disables (default). 100 = a completed lap is worth ~67 "
                             "steps of alive reward at alive=1.5.")
    parser.add_argument("--cte-speed-penalty-weight", type=float, default=0.25,
                        help="Per-step penalty: -w * |cte| * speed. Punishes 'fast while off-line' "
                             "to bring corner-crash signal forward in time. v2 default 0.25.")

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--override-learning-rate", action="store_true",
                        help="On --resume-model, override the saved LR with --learning-rate. "
                             "Default behavior on resume is to keep the saved LR.")
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64,
                        help="SAC MLP hidden layer size (pi and qf use [h, h]). VAE-trained "
                             "safe_v2 used 64; ResNet/DINO need 256+ to learn projection.")
    parser.add_argument("--learning-starts", type=int, default=500)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--train-freq-unit", choices=["step", "episode"], default="episode")
    parser.add_argument("--gradient-steps", type=int, default=-1)
    parser.add_argument("--gradient-steps-cap", type=int, default=1000)
    parser.add_argument("--gradient-steps-min", type=int, default=50,
                        help="Floor on dynamic gradient_steps (used with train_freq=episode). "
                             "Guarantees at least this many off-policy updates per training cycle.")
    parser.add_argument("--gradient-steps-scale", type=float, default=1.0,
                        help="Multiplier on episode length before clamping to [min, cap]. "
                             "Default 1.0 = 1 update per env step (raffin-style). "
                             "Set <1 (e.g. 0.5) to dampen overfit on long truncate episodes; "
                             "experimental — not yet validated in deterministic eval.")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", default="auto_0.1")
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--max-episode-steps", type=int, default=3000)
    parser.add_argument("--scene-reload-every", type=int, default=0,
                        help="Episode-based scene reload: reload every N episodes. "
                             "0 = off (default). Superseded by --scene-reload-alpha "
                             "if that is set.")
    parser.add_argument("--scene-reload-alpha", type=float, default=0.0,
                        help="Adaptive step-based scene reload for domain "
                             "randomization. Reload after ~K steps on a scene, "
                             "K = clamp(alpha * recent_ep_len, [kmin, max_episode_steps-1]). "
                             "alpha is roughly 'episodes per scene'. 0 = off (default).")
    parser.add_argument("--scene-reload-kmin", type=int, default=200,
                        help="Lower bound on K for --scene-reload-alpha (min steps "
                             "per scene before a reload).")
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

    vae = make_encoder(args.encoder, device=device, vae_checkpoint=args.vae_model,
                       crop_top=args.encoder_crop_top)
    print(f"Encoder: {args.encoder}  z_size={vae.z_size}  crop_top={args.encoder_crop_top}")
    env = build_env(args, vae)
    env.action_space.seed(args.seed)

    hidden = args.hidden_size
    policy_kwargs = {
        "net_arch": {"pi": [hidden, hidden], "qf": [hidden, hidden]},
    }
    print(f"SAC MLP net_arch: pi/qf = [{hidden}, {hidden}]")

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
        if args.override_learning_rate:
            from stable_baselines3.common.utils import get_schedule_fn
            new_lr = args.learning_rate
            model.lr_schedule = get_schedule_fn(new_lr)
            for opt in [model.actor.optimizer, model.critic.optimizer]:
                for pg in opt.param_groups:
                    pg["lr"] = new_lr
            if model.ent_coef_optimizer is not None:
                for pg in model.ent_coef_optimizer.param_groups:
                    pg["lr"] = new_lr
            print(f"Override learning_rate to {new_lr} on resume")
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
            name_prefix=f"sac_{args.encoder}",
            save_replay_buffer=args.save_replay_buffer,
            save_vecnormalize=False,
        ),
        DonkeyInfoCallback(),
    ]
    if args.gradient_steps_cap is not None and args.gradient_steps_cap > 0:
        callbacks.append(CappedDynamicGradientStepsCallback(
            cap=args.gradient_steps_cap,
            floor=args.gradient_steps_min,
            scale=args.gradient_steps_scale,
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
