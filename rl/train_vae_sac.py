from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path

import gymnasium as gym
import gym_donkeycar  # noqa: F401 - registers donkey gym environments
import numpy as np
import torch
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

try:
    from rl.vae import ConvVAE
except ImportError:
    from vae import ConvVAE


CAMERA_HEIGHT = 120
CAMERA_WIDTH = 160
MARGIN_TOP = CAMERA_HEIGHT // 3
IMAGE_HEIGHT = CAMERA_HEIGHT - MARGIN_TOP
IMAGE_WIDTH = CAMERA_WIDTH
Z_SIZE = 512

MIN_THROTTLE = 0.4
MAX_THROTTLE = 0.6
MAX_STEERING = 1.0
MAX_STEERING_DIFF = 0.15
N_COMMAND_HISTORY = 20
MAX_CTE_ERROR = 2.0
THROTTLE_REWARD_WEIGHT = 0.1
REWARD_CRASH = -10.0
CRASH_SPEED_WEIGHT = 5.0
ALIVE_REWARD = 1.0
SPEED_REWARD_WEIGHT = 0.0
MIN_ALIVE_SPEED = 0.0
PROGRESS_REWARD_WEIGHT = 0.0


@dataclass
class RaffinRewardConfig:
    max_cte_error: float = MAX_CTE_ERROR
    throttle_reward_weight: float = THROTTLE_REWARD_WEIGHT
    reward_crash: float = REWARD_CRASH
    crash_speed_weight: float = CRASH_SPEED_WEIGHT
    alive_reward: float = ALIVE_REWARD
    speed_reward_weight: float = SPEED_REWARD_WEIGHT
    min_alive_speed: float = MIN_ALIVE_SPEED
    progress_reward_weight: float = PROGRESS_REWARD_WEIGHT
    cte_speed_penalty_weight: float = 0.0


class FrozenVaeEncoder:
    def __init__(self, checkpoint_path: Path, device: torch.device, z_size: int = Z_SIZE) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model = ConvVAE(z_size=z_size).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.device = device
        self.z_size = z_size

    @torch.no_grad()
    def encode(self, obs: np.ndarray) -> np.ndarray:
        image = np.asarray(obs, dtype=np.uint8)
        image = image[MARGIN_TOP:, :, :]
        if image.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
            raise ValueError(f"unexpected VAE image shape {image.shape}, expected {(IMAGE_HEIGHT, IMAGE_WIDTH, 3)}")
        tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)).copy()).to(self.device)
        tensor = tensor.unsqueeze(0).float() / 255.0
        mu, _ = self.model.encode(tensor)
        return mu.squeeze(0).detach().cpu().numpy().astype(np.float32)


class FrozenPretrainedCnnEncoder:
    """ImageNet-pretrained CNN trunk used as a frozen feature extractor.

    Same encode() contract as FrozenVaeEncoder so DonkeyVaeSACEnv can use either.
    Output dim is exposed via self.z_size.
    """

    def __init__(self, model_name: str, device: torch.device, crop_top: int = 0) -> None:
        import torchvision.models as tvm

        if model_name == "resnet18":
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1
            model = tvm.resnet18(weights=weights)
            model.fc = torch.nn.Identity()
            feature_dim = 512
        elif model_name == "mobilenet_v3_small":
            weights = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            model = tvm.mobilenet_v3_small(weights=weights)
            model.classifier = torch.nn.Identity()
            feature_dim = 576
        else:
            raise ValueError(f"unsupported pretrained encoder: {model_name}")

        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.device = device
        self.z_size = feature_dim
        self.model_name = model_name
        self.crop_top = max(0, int(crop_top))
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode(self, obs: np.ndarray) -> np.ndarray:
        image = np.asarray(obs, dtype=np.uint8)
        if self.crop_top > 0:
            image = image[self.crop_top:, :, :]
        expected_h = CAMERA_HEIGHT - self.crop_top
        if image.shape != (expected_h, CAMERA_WIDTH, 3):
            raise ValueError(
                f"unexpected image shape {image.shape}, expected {(expected_h, CAMERA_WIDTH, 3)}"
            )
        tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)).copy()).to(self.device)
        tensor = tensor.unsqueeze(0).float() / 255.0
        tensor = torch.nn.functional.interpolate(
            tensor, size=(224, 224), mode="bilinear", align_corners=False
        )
        tensor = (tensor - self._mean) / self._std
        feat = self.model(tensor)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)


def make_encoder(encoder_name: str, device: torch.device, vae_checkpoint: Path | None = None,
                 crop_top: int = 0):
    """Factory: build an encoder by name. Returns object with .encode(obs) and .z_size.

    `crop_top` only affects pretrained CNN encoders. The VAE encoder always uses its
    fixed MARGIN_TOP crop (since the VAE was trained on cropped frames).
    """
    if encoder_name == "vae":
        if vae_checkpoint is None:
            raise ValueError("--vae-model is required when --encoder=vae")
        return FrozenVaeEncoder(vae_checkpoint, device=device, z_size=Z_SIZE)
    if encoder_name in {"resnet18", "mobilenet_v3_small"}:
        return FrozenPretrainedCnnEncoder(encoder_name, device=device, crop_top=crop_top)
    raise ValueError(f"unsupported encoder: {encoder_name}")


class DonkeyVaeSACEnv(gym.Wrapper):
    """Raffin-style Donkey env: frozen VAE latent + command history."""

    def __init__(
        self,
        env: gym.Env,
        *,
        vae: FrozenVaeEncoder,
        min_throttle: float = MIN_THROTTLE,
        max_throttle: float = MAX_THROTTLE,
        max_steering: float = MAX_STEERING,
        max_steering_diff: float = MAX_STEERING_DIFF,
        n_command_history: int = N_COMMAND_HISTORY,
        reward_config: RaffinRewardConfig | None = None,
    ) -> None:
        super().__init__(env)
        self.vae = vae
        self.min_throttle = min_throttle
        self.max_throttle = max_throttle
        self.max_steering = max_steering
        self.max_steering_diff = max_steering_diff
        self.n_command_history = n_command_history
        self.reward_config = reward_config or RaffinRewardConfig()
        self.command_history = np.zeros((self.n_command_history, 2), dtype=np.float32)
        self.last_pos: tuple[float, float, float] | None = None

        self.action_space = gym.spaces.Box(
            low=np.array([-self.max_steering, -1.0], dtype=np.float32),
            high=np.array([self.max_steering, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        obs_dim = self.vae.z_size + 2 * self.n_command_history
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.last_throttle = self.min_throttle

    def reset(self, **kwargs):
        self.command_history.fill(0.0)
        self.last_throttle = self.min_throttle
        obs, info = self.env.reset(**kwargs)
        self.last_pos = None
        return self._build_obs(obs), info

    def step(self, action):
        raw_action = np.asarray(action, dtype=np.float32).reshape(-1)
        prev_steer = float(self.command_history[-1, 0]) if self.n_command_history > 0 else 0.0
        executed = self._map_and_clip_action(raw_action)
        obs, original_reward, terminated, truncated, info = self.env.step(executed)
        terminated = self._is_game_over(terminated, info)
        progress = self._calculate_progress(info)
        reward = self._calculate_reward(terminated, executed[1], info, progress)
        self._push_history(executed)

        info = dict(info)
        cte = float(info.get("cte", 0.0))
        info.update(
            {
                "original_reward": original_reward,
                "rl_steer": float(executed[0]),
                "rl_steer_delta": float(executed[0]) - prev_steer,
                "rl_throttle": float(executed[1]),
                "abs_cte": abs(cte),
                "delta_pos_distance": progress,
                "vae_reward": reward,
            }
        )
        return self._build_obs(obs), reward, terminated, truncated, info

    def _map_and_clip_action(self, action: np.ndarray) -> np.ndarray:
        steer = float(np.clip(action[0], -self.max_steering, self.max_steering))
        t = float(np.clip((action[1] + 1.0) / 2.0, 0.0, 1.0))
        throttle = (1.0 - t) * self.min_throttle + self.max_throttle * t

        if self.n_command_history > 0:
            prev_steer = float(self.command_history[-1, 0])
            max_diff = (self.max_steering_diff - 1e-5) * (2.0 * self.max_steering)
            diff = float(np.clip(steer - prev_steer, -max_diff, max_diff))
            steer = prev_steer + diff

        self.last_throttle = float(throttle)
        return np.array([steer, throttle], dtype=np.float32)

    def _push_history(self, action: np.ndarray) -> None:
        if self.n_command_history <= 0:
            return
        self.command_history[:-1] = self.command_history[1:]
        self.command_history[-1] = action

    def _build_obs(self, obs: np.ndarray) -> np.ndarray:
        z = self.vae.encode(obs)
        if self.n_command_history <= 0:
            return z
        return np.concatenate([z, self.command_history.reshape(-1)], dtype=np.float32)

    def _is_game_over(self, terminated: bool, info: dict) -> bool:
        cte = abs(float(info.get("cte", 0.0)))
        return bool(terminated or cte > self.reward_config.max_cte_error)

    def _calculate_reward(self, terminated: bool, throttle: float, info: dict, progress: float) -> float:
        cfg = self.reward_config
        speed = float(info.get("speed", 0.0))
        if terminated:
            norm_throttle = (throttle - self.min_throttle) / max(self.max_throttle - self.min_throttle, 1e-6)
            crash_speed = speed if (cfg.speed_reward_weight > 0.0 or cfg.progress_reward_weight > 0.0) else norm_throttle
            return float(cfg.reward_crash - cfg.crash_speed_weight * crash_speed)
        progress_reward = cfg.progress_reward_weight * progress
        throttle_reward = cfg.throttle_reward_weight * (throttle / max(self.max_throttle, 1e-6))
        speed_reward = cfg.speed_reward_weight * speed
        abs_cte = abs(float(info.get("cte", 0.0)))
        cte_speed_penalty = cfg.cte_speed_penalty_weight * abs_cte * speed
        if cfg.min_alive_speed > 0.0:
            alive_scale = float(np.clip(speed / cfg.min_alive_speed, 0.0, 1.0))
        else:
            alive_scale = 1.0
        return float(
            cfg.alive_reward * alive_scale + throttle_reward + speed_reward + progress_reward - cte_speed_penalty
        )

    def _calculate_progress(self, info: dict) -> float:
        current_pos = self._extract_pos(info)
        previous_pos = self.last_pos
        self.last_pos = current_pos
        if previous_pos is None or current_pos is None:
            return 0.0
        dx = current_pos[0] - previous_pos[0]
        dz = current_pos[2] - previous_pos[2]
        return float(math.hypot(dx, dz))

    @staticmethod
    def _extract_pos(info: dict) -> tuple[float, float, float] | None:
        pos = info.get("pos")
        if pos is None or len(pos) < 3:
            return None
        return float(pos[0]), float(pos[1]), float(pos[2])


class CappedDynamicGradientStepsCallback(BaseCallback):
    """gradient_steps = clamp(episode_length, [floor, cap]).

    Used with train_freq=(1, 'episode'). Each training cycle scales updates with
    collected data, capped to prevent runaway compute on long episodes and
    floored to guarantee a minimum amount of off-policy learning per cycle.
    """

    def __init__(self, cap: int, floor: int = 1) -> None:
        super().__init__()
        self.cap = cap
        self.floor = max(1, int(floor))
        self._rollout_start_ts = 0

    def _on_rollout_start(self) -> None:
        self._rollout_start_ts = int(self.model.num_timesteps)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        ep_len = max(1, int(self.model.num_timesteps) - self._rollout_start_ts)
        new_grads = max(self.floor, min(ep_len, self.cap))
        self.model.gradient_steps = new_grads
        self.logger.record("train/gradient_steps_used", new_grads)
        self.logger.record("train/rollout_ep_len", ep_len)


class DonkeyInfoCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True
        ctes = [float(info["abs_cte"]) for info in infos if "abs_cte" in info]
        speeds = [float(info.get("speed", 0.0)) for info in infos]
        steers = [float(info["rl_steer"]) for info in infos if "rl_steer" in info]
        throttles = [float(info["rl_throttle"]) for info in infos if "rl_throttle" in info]
        progresses = [float(info["delta_pos_distance"]) for info in infos if "delta_pos_distance" in info]
        if ctes:
            self.logger.record("donkey/abs_cte_mean", float(np.mean(ctes)))
            self.logger.record("donkey/abs_cte_max", float(np.max(ctes)))
        if speeds:
            self.logger.record("donkey/speed_mean", float(np.mean(speeds)))
        if steers:
            self.logger.record("donkey/abs_steer_mean", float(np.mean(np.abs(steers))))
        deltas = [abs(float(info["rl_steer_delta"])) for info in infos if "rl_steer_delta" in info]
        if deltas:
            self.logger.record("donkey/abs_steer_delta_mean", float(np.mean(deltas)))
        if throttles:
            self.logger.record("donkey/throttle_mean", float(np.mean(throttles)))
        if progresses:
            self.logger.record("donkey/progress_mean", float(np.mean(progresses)))
        return True


def build_env(args: argparse.Namespace, vae: FrozenVaeEncoder) -> gym.Env:
    conf = {
        "host": args.host,
        "port": args.port,
        "cam_resolution": (CAMERA_HEIGHT, CAMERA_WIDTH, 3),
    }
    base_env = gym.make(args.env_id, conf=conf)
    env = DonkeyVaeSACEnv(
        base_env,
        vae=vae,
        min_throttle=args.min_throttle,
        max_throttle=args.max_throttle,
        max_steering=args.max_steering,
        max_steering_diff=args.max_steering_diff,
        n_command_history=args.n_command_history,
        reward_config=RaffinRewardConfig(
            max_cte_error=args.max_cte_error,
            throttle_reward_weight=args.throttle_reward_weight,
            reward_crash=args.reward_crash,
            crash_speed_weight=args.crash_speed_weight,
            alive_reward=args.alive_reward,
            speed_reward_weight=args.speed_reward_weight,
            min_alive_speed=args.min_alive_speed,
            progress_reward_weight=args.progress_reward_weight,
            cte_speed_penalty_weight=getattr(args, "cte_speed_penalty_weight", 0.0),
        ),
    )
    env = TimeLimit(env, max_episode_steps=args.max_episode_steps)
    return Monitor(env, filename=str(args.output_dir / "monitor.csv"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Raffin-style SAC on frozen VAE latent observations.")
    parser.add_argument("--env-id", default="donkey-generated-roads-v0")
    parser.add_argument("--host", default=os.environ.get("DONKEY_SIM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DONKEY_SIM_PORT", "9091")))
    parser.add_argument("--vae-model", type=Path, default=Path("models/vae_raffin_v1/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/rl_vae_sac_raffin_v1"))
    parser.add_argument("--resume-model", type=Path, default=None)
    parser.add_argument("--resume-replay-buffer", type=Path, default=None)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--min-throttle", type=float, default=MIN_THROTTLE)
    parser.add_argument("--max-throttle", type=float, default=MAX_THROTTLE)
    parser.add_argument("--max-steering", type=float, default=MAX_STEERING)
    parser.add_argument("--max-steering-diff", type=float, default=MAX_STEERING_DIFF)
    parser.add_argument("--n-command-history", type=int, default=N_COMMAND_HISTORY)
    parser.add_argument("--max-cte-error", type=float, default=MAX_CTE_ERROR)
    parser.add_argument("--throttle-reward-weight", type=float, default=THROTTLE_REWARD_WEIGHT)
    parser.add_argument("--reward-crash", type=float, default=REWARD_CRASH)
    parser.add_argument("--crash-speed-weight", type=float, default=CRASH_SPEED_WEIGHT)
    parser.add_argument("--alive-reward", type=float, default=ALIVE_REWARD)
    parser.add_argument("--speed-reward-weight", type=float, default=SPEED_REWARD_WEIGHT)
    parser.add_argument("--min-alive-speed", type=float, default=MIN_ALIVE_SPEED)
    parser.add_argument("--progress-reward-weight", type=float, default=PROGRESS_REWARD_WEIGHT)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=300)
    parser.add_argument(
        "--train-freq",
        type=int,
        default=1,
        help="Default schedule: train after every episode (`--train-freq 1 --train-freq-unit episode`).",
    )
    parser.add_argument("--train-freq-unit", choices=["step", "episode"], default="episode")
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=-1,
        help="-1 (default) = use the just-finished episode's length, so updates scale with data collected. "
             "Use --gradient-steps-cap to upper-bound this dynamically.",
    )
    parser.add_argument(
        "--gradient-steps-cap",
        type=int,
        default=600,
        help="gradient_steps becomes min(episode_length, cap) per training cycle. "
             "Default 600 matches the intended Raffin-style schedule (proportional updates with a 600 ceiling). "
             "Pass 0 or a negative value to disable the cap.",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", default="auto_0.1")
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="TimeLimit cap. Defaults to 3000 in episode mode (matches Raffin) and to --train-freq in step mode.",
    )
    parser.add_argument("--save-replay-buffer", action="store_true")
    parser.add_argument("--save-final-replay-buffer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_episode_steps is None:
        args.max_episode_steps = args.train_freq if args.train_freq_unit == "step" else 3000
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
            device=args.device,
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
            device=args.device,
        )

    callbacks = [
        CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(args.output_dir),
            name_prefix="sac_vae_raffin",
            save_replay_buffer=args.save_replay_buffer,
            save_vecnormalize=False,
        ),
        DonkeyInfoCallback(),
    ]
    if args.gradient_steps_cap is not None and args.gradient_steps_cap > 0:
        callbacks.append(CappedDynamicGradientStepsCallback(cap=args.gradient_steps_cap))

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
