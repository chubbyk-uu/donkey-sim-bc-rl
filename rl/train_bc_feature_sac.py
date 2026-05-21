from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import gym_donkeycar  # noqa: F401 - registers donkey gym environments
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed

try:
    from rl.donkey_sac_env import DonkeyRewardConfig, DonkeySACEnv
except ImportError:
    from donkey_sac_env import DonkeyRewardConfig, DonkeySACEnv


class RegressionBCFeatureExtractor(BaseFeaturesExtractor):
    """NVIDIA-style BC encoder without dropout for SAC actor/critic."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 50):
        super().__init__(observation_space, features_dim)

        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 32, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.features(sample).flatten(1).shape[1]

        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.float() / 255.0
        return self.trunk(self.features(x))


class BCImageHistoryExtractor(BaseFeaturesExtractor):
    """BC encoder for image + raw command-history concat for Dict observation."""

    BC_FEATURES_DIM = 50

    def __init__(self, observation_space: gym.spaces.Dict):
        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError("BCImageHistoryExtractor requires a Dict observation space")
        image_space = observation_space.spaces["image"]
        history_space = observation_space.spaces.get("history")
        history_dim = int(history_space.shape[0]) if history_space is not None else 0
        super().__init__(observation_space, self.BC_FEATURES_DIM + history_dim)

        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 32, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            sample = torch.as_tensor(image_space.sample()[None]).float()
            n_flatten = self.features(sample).flatten(1).shape[1]

        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, self.BC_FEATURES_DIM),
            nn.ReLU(inplace=True),
        )
        self._history_dim = history_dim

    def forward(self, observations: dict) -> torch.Tensor:
        image = observations["image"].float() / 255.0
        feat = self.trunk(self.features(image))
        if self._history_dim > 0:
            feat = torch.cat([feat, observations["history"].float()], dim=1)
        return feat


def freeze_features_extractor(model: SAC) -> int:
    extractors = [
        model.policy.actor.features_extractor,
        model.policy.critic.features_extractor,
        model.policy.critic_target.features_extractor,
    ]
    n_frozen = 0
    for extractor in extractors:
        for p in extractor.parameters():
            if p.requires_grad:
                p.requires_grad = False
                n_frozen += 1
    print(f"Froze {n_frozen} extractor tensors (actor + critic + critic_target)")
    return n_frozen


def _copy_if_shape_matches(
    target_state: dict[str, torch.Tensor],
    source_state: dict[str, torch.Tensor],
    copied: list[str],
    source_key: str,
    target_key: str,
) -> None:
    if source_key not in source_state:
        raise KeyError(f"BC checkpoint is missing {source_key}")
    if target_key not in target_state:
        raise KeyError(f"SAC extractor is missing {target_key}")
    if source_state[source_key].shape != target_state[target_key].shape:
        raise ValueError(
            f"Shape mismatch for {source_key} -> {target_key}: "
            f"{tuple(source_state[source_key].shape)} vs {tuple(target_state[target_key].shape)}"
        )
    target_state[target_key].copy_(source_state[source_key])
    copied.append(f"{source_key}->{target_key}")


def load_regression_bc_features(model: SAC, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=model.device)
    source_state = checkpoint.get("model_state_dict", checkpoint)

    conv_map = {
        "features.0": "features.0",
        "features.3": "features.2",
        "features.6": "features.4",
        "features.9": "features.6",
        "features.12": "features.8",
    }
    linear_map = {
        "head.1": "trunk.1",
        "head.4": "trunk.3",
    }

    extractors = [
        model.policy.actor.features_extractor,
        model.policy.critic.features_extractor,
        model.policy.critic_target.features_extractor,
    ]

    for extractor in extractors:
        target_state = extractor.state_dict()
        copied: list[str] = []

        for source_prefix, target_prefix in conv_map.items():
            _copy_if_shape_matches(target_state, source_state, copied, f"{source_prefix}.weight", f"{target_prefix}.weight")
            _copy_if_shape_matches(target_state, source_state, copied, f"{source_prefix}.bias", f"{target_prefix}.bias")

        for source_prefix, target_prefix in linear_map.items():
            _copy_if_shape_matches(target_state, source_state, copied, f"{source_prefix}.weight", f"{target_prefix}.weight")
            _copy_if_shape_matches(target_state, source_state, copied, f"{source_prefix}.bias", f"{target_prefix}.bias")

        extractor.load_state_dict(target_state)
        if len(copied) != 14:
            raise RuntimeError(f"Expected to copy 14 tensors into extractor, copied {len(copied)}")

    print(f"Loaded regression BC feature weights from {checkpoint_path}")


class DonkeyInfoCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        ctes = [float(info["abs_cte"]) for info in infos if "abs_cte" in info]
        speeds = [float(info.get("speed", 0.0)) for info in infos]
        steers = [float(info["rl_steer"]) for info in infos if "rl_steer" in info]
        steer_deltas = [float(info["rl_steer_delta"]) for info in infos if "rl_steer_delta" in info]

        if ctes:
            self.logger.record("donkey/abs_cte_mean", float(np.mean(ctes)))
            self.logger.record("donkey/abs_cte_max", float(np.max(ctes)))
        if speeds:
            self.logger.record("donkey/speed_mean", float(np.mean(speeds)))
        if steers:
            self.logger.record("donkey/abs_steer_mean", float(np.mean(np.abs(steers))))
        if steer_deltas:
            self.logger.record("donkey/abs_steer_delta_mean", float(np.mean(np.abs(steer_deltas))))
        return True


def build_env(args: argparse.Namespace) -> gym.Env:
    conf = {
        "host": args.host,
        "port": args.port,
        "cam_resolution": (120, 160, 3),
    }
    base_env = gym.make(args.env_id, conf=conf)
    reward_config = DonkeyRewardConfig(
        alive_reward=args.alive_reward,
        throttle_reward_weight=args.throttle_reward_weight,
        terminal_cte=args.terminal_cte,
        crash_penalty=args.crash_penalty,
        crash_throttle_weight=args.crash_throttle_weight,
        steer_penalty=args.steer_penalty,
        steer_delta_penalty=args.steer_delta_penalty,
    )
    env = DonkeySACEnv(
        base_env,
        fixed_throttle=args.fixed_throttle,
        max_steering=args.max_steering,
        min_throttle=args.min_throttle,
        max_throttle=args.max_throttle,
        action_smoothing=args.action_smoothing,
        max_steer_delta=args.max_steer_delta,
        n_command_history=args.n_command_history,
        reward_config=reward_config,
    )
    return Monitor(env, filename=str(Path(args.output_dir) / "monitor.csv"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC with regression BC feature initialization.")
    parser.add_argument("--env-id", default="donkey-generated-roads-v0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--output-dir", type=Path, default=Path("models/rl_sac_bc_feature_v1"))
    parser.add_argument("--bc-model", type=Path, default=None, help="Regression BC checkpoint. Omit for pure SAC.")
    parser.add_argument("--resume-model", type=Path, default=None, help="Existing SAC .zip checkpoint to continue.")
    parser.add_argument("--resume-replay-buffer", type=Path, default=None, help="Optional replay buffer .pkl to continue off-policy training.")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--fixed-throttle", type=float, default=None)
    parser.add_argument("--max-steering", type=float, default=0.6)
    parser.add_argument("--min-throttle", type=float, default=0.05)
    parser.add_argument("--max-throttle", type=float, default=0.4)
    parser.add_argument("--action-smoothing", type=float, default=0.5)
    parser.add_argument("--max-steer-delta", type=float, default=None)
    parser.add_argument(
        "--n-command-history",
        type=int,
        default=10,
        help="Past action steps fed to policy/critic (0 disables; uses Dict obs when >0).",
    )
    parser.add_argument(
        "--freeze-extractor",
        action="store_true",
        default=False,
        help="Freeze conv+trunk of features extractor after BC load (or after resume).",
    )

    parser.add_argument("--alive-reward", type=float, default=1.0)
    parser.add_argument("--throttle-reward-weight", type=float, default=0.1)
    parser.add_argument("--terminal-cte", type=float, default=4.0)
    parser.add_argument("--crash-penalty", type=float, default=10.0)
    parser.add_argument("--crash-throttle-weight", type=float, default=5.0)
    parser.add_argument("--steer-penalty", type=float, default=0.0)
    parser.add_argument("--steer-delta-penalty", type=float, default=0.0)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=3000)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--train-freq-unit", choices=["step", "episode"], default="step")
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--ent-coef", default="auto_0.1")
    parser.add_argument("--use-sde", action="store_true", default=True)
    parser.add_argument("--no-use-sde", action="store_false", dest="use_sde")
    parser.add_argument(
        "--use-sde-at-warmup",
        action="store_true",
        default=False,
        help="Enable gSDE during warmup (default off: uniform random actions fill buffer).",
    )
    parser.add_argument("--sde-sample-freq", type=int, default=64)
    parser.add_argument("--log-std-init", type=float, default=-2.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--checkpoint-freq", type=int, default=5_000)
    parser.add_argument("--save-replay-buffer", action="store_true", help="Save replay buffer at checkpoints. Large for image observations.")
    parser.add_argument("--save-final-replay-buffer", action="store_true", help="Save final replay buffer on exit. Large for image observations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)

    env = build_env(args)
    env.action_space.seed(args.seed)

    if args.n_command_history > 0:
        policy_class = "MultiInputPolicy"
        policy_kwargs = {
            "features_extractor_class": BCImageHistoryExtractor,
            "features_extractor_kwargs": {},
            "net_arch": {"pi": [64, 64], "qf": [64, 64]},
            "log_std_init": args.log_std_init,
            "normalize_images": False,
        }
    else:
        policy_class = "CnnPolicy"
        policy_kwargs = {
            "features_extractor_class": RegressionBCFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 50},
            "net_arch": {"pi": [64, 64], "qf": [64, 64]},
            "log_std_init": args.log_std_init,
            "normalize_images": False,
        }

    if args.resume_model is not None:
        model = SAC.load(
            args.resume_model,
            env=env,
            device=args.device,
            tensorboard_log=str(args.output_dir / "tensorboard"),
        )
        if args.resume_replay_buffer is not None:
            model.load_replay_buffer(str(args.resume_replay_buffer))
            print(f"Loaded replay buffer from {args.resume_replay_buffer}")
        print(f"Resumed SAC model from {args.resume_model}")
    else:
        model = SAC(
            policy_class,
            env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            train_freq=(args.train_freq, args.train_freq_unit),
            gradient_steps=args.gradient_steps,
            ent_coef=args.ent_coef,
            use_sde=args.use_sde,
            use_sde_at_warmup=args.use_sde_at_warmup,
            sde_sample_freq=args.sde_sample_freq,
            gamma=args.gamma,
            tau=args.tau,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(args.output_dir / "tensorboard"),
            verbose=1,
            seed=args.seed,
            device=args.device,
        )

    if args.resume_model is None and args.bc_model is not None:
        load_regression_bc_features(model, args.bc_model)
    elif args.resume_model is None:
        print("No BC checkpoint provided; training pure SAC from random visual features.")

    if args.freeze_extractor:
        freeze_features_extractor(model)

    callbacks = [
        CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(args.output_dir),
            name_prefix="sac_bc_feature",
            save_replay_buffer=args.save_replay_buffer,
            save_vecnormalize=False,
        ),
        DonkeyInfoCallback(),
    ]

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
