from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, fields
import os
import time
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
    from rl.donkey_sync import add_control_fix_args, configure_env_control_fixes
    from rl.vae import ConvVAE
    from rl.experiment import save_replay_metadata
    from rl.track_state import TrackState
except ImportError:
    from donkey_sync import add_control_fix_args, configure_env_control_fixes
    from vae import ConvVAE
    from experiment import save_replay_metadata
    from track_state import TrackState


CAMERA_HEIGHT = 120
CAMERA_WIDTH = 160
MARGIN_TOP = CAMERA_HEIGHT // 3
IMAGE_HEIGHT = CAMERA_HEIGHT - MARGIN_TOP
IMAGE_WIDTH = CAMERA_WIDTH
Z_SIZE = 512

MIN_THROTTLE = 0.4
MAX_THROTTLE = 0.6
MAX_STEERING = 1.0
MAX_STEERING_DIFF = 0.2
N_COMMAND_HISTORY = 20
MAX_CTE_ERROR = 2.0
REWARD_CRASH = -10.0
CRASH_SPEED_WEIGHT = 5.0
ALIVE_REWARD = 1.0
SPEED_REWARD_WEIGHT = 0.0
MIN_ALIVE_SPEED = 0.0


@dataclass
class RaffinRewardConfig:
    max_cte_error: float = MAX_CTE_ERROR
    reward_crash: float = REWARD_CRASH
    crash_speed_weight: float = CRASH_SPEED_WEIGHT
    alive_reward: float = ALIVE_REWARD
    speed_reward_weight: float = SPEED_REWARD_WEIGHT
    min_alive_speed: float = MIN_ALIVE_SPEED
    alive_scale_floor: float = 0.0  # minimum alive_scale at speed=0 when min_alive_speed>0.
                                     # 0.0 → ramp from 0 (default, sharp).
                                     # 0.5 → ramp from 0.5 (softer; full alive at min_alive_speed,
                                     # half at speed=0). Tolerates corner slowdowns better.
    lap_completion_bonus: float = 0.0  # discrete reward added each time info['lap_count']
                                        # increments. e.g. 100 = "completing a lap is worth ~67 steps
                                        # of alive reward". Always added (even on terminate step).
    cte_speed_penalty_weight: float = 0.0
    cte_target: float = 0.0  # offset reference for "lane center" — on generated-track 0
                              # is right-lane center; on mountain-track right lane is at
                              # cte=3.5, so set cte_target=3.5 there so abs(cte-target)
                              # measures how far off the desired line the car is.


def reward_config_from_args(args) -> RaffinRewardConfig:
    """One mapping from CLI arguments to rewards, shared by training and both evals."""
    return RaffinRewardConfig(**{f.name: getattr(args, f.name, f.default) for f in fields(RaffinRewardConfig)})


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
        elif model_name == "dinov2_vits14":
            model = torch.hub.load("facebookresearch/dinov2:main", "dinov2_vits14",
                                   pretrained=True, trust_repo=True)
            feature_dim = 384
        elif model_name == "dinov2_vitb14":
            model = torch.hub.load("facebookresearch/dinov2:main", "dinov2_vitb14",
                                   pretrained=True, trust_repo=True)
            feature_dim = 768
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


GRID_SIDE = 4


class FrozenDinoGridEncoder(FrozenPretrainedCnnEncoder):
    """DINOv2-S patch tokens pooled to a 4x4 grid, then a fixed linear projection.

    The class token used by "dinov2_vits14" discards where things are in the
    image; the pooled grid keeps it and reads lane position about twice as
    accurately (experiment-log 12). The grid has 16 * 384 values, so a
    projection fitted offline (tools/fit_grid_projection.py) reduces it to a
    few whitened components.
    """

    def __init__(self, device: torch.device, crop_top: int = 0, projection: Path | None = None) -> None:
        super().__init__("dinov2_vits14", device=device, crop_top=crop_top)
        self.model_name = "dinov2_vits14_grid"
        self.z_size = GRID_SIDE * GRID_SIDE * self.z_size
        self.projection = None
        if projection is not None:
            saved = torch.load(projection, map_location=device, weights_only=True)
            self.projection = (saved["mean"].float(), saved["components"].float(), saved["scale"].float())
            self.z_size = int(saved["components"].shape[0])

    @torch.no_grad()
    def grid_features(self, images: np.ndarray) -> torch.Tensor:
        """(N, H, W, 3) uint8 frames -> (N, 16 * 384) pooled patch tokens."""
        tensor = torch.from_numpy(np.array(images[:, self.crop_top:])).to(self.device)  # copy: frames may be read-only
        tensor = tensor.permute(0, 3, 1, 2).contiguous().float() / 255.0
        tensor = torch.nn.functional.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
        patches = self.model.forward_features((tensor - self._mean) / self._std)["x_norm_patchtokens"]
        side = int(patches.shape[1] ** 0.5)  # row-major patch grid
        grid = patches.transpose(1, 2).reshape(len(images), -1, side, side)
        return torch.nn.functional.adaptive_avg_pool2d(grid, (GRID_SIDE, GRID_SIDE)).flatten(1)

    @torch.no_grad()
    def encode(self, obs: np.ndarray) -> np.ndarray:
        image = np.asarray(obs, dtype=np.uint8)
        if image.shape != (CAMERA_HEIGHT, CAMERA_WIDTH, 3):
            raise ValueError(f"unexpected image shape {image.shape}")
        features = self.grid_features(image[None])
        if self.projection is not None:
            mean, components, scale = self.projection
            features = (features - mean) @ components.T / scale
        return features.squeeze(0).cpu().numpy().astype(np.float32)


def make_encoder(encoder_name: str, device: torch.device, vae_checkpoint: Path | None = None,
                 crop_top: int = 0, projection: Path | None = None, track: Path | None = None):
    """Factory: build an encoder by name. Returns object with .encode(obs) and .z_size.

    `crop_top` applies to all frozen pretrained encoders (DINOv2, ResNet, MobileNet).
    The VAE encoder does not read this arg but is NOT uncropped: it always crops by
    its own fixed MARGIN_TOP (the VAE was trained on cropped frames), see
    FrozenVaeEncoder.encode.
    """
    if encoder_name == "vae":
        if vae_checkpoint is None:
            raise ValueError("--vae-model is required when --encoder=vae")
        return FrozenVaeEncoder(vae_checkpoint, device=device, z_size=Z_SIZE)
    if encoder_name in {"resnet18", "mobilenet_v3_small", "dinov2_vits14", "dinov2_vitb14"}:
        return FrozenPretrainedCnnEncoder(encoder_name, device=device, crop_top=crop_top)
    if encoder_name == "dinov2_vits14_grid":
        if projection is None:
            raise ValueError("--encoder-projection is required when --encoder=dinov2_vits14_grid")
        return FrozenDinoGridEncoder(device=device, crop_top=crop_top, projection=projection)
    if encoder_name == "sim_state":
        if track is None:
            raise ValueError("--track-centerline is required when --encoder=sim_state")
        return TrackState.load(track)
    raise ValueError(f"unsupported encoder: {encoder_name}")


def reload_scene(
    controller,
    scene: str,
    timeout: float = 30.0,
    exit_settle: float = 1.0,
    retry_every: float = 1.5,
) -> bool:
    """Reload the sim scene, regenerating per-scene randomization (trees,
    shadows, lighting). Used for domain randomization during training.

    The handshake is asymmetric: load completion has a state signal
    (`handler.loaded`, set by on_car_loaded / on_need_car_config), but the
    exit→menu transition has no reliable incoming message. So after exit_scene
    we wait a fixed `exit_settle` before the first load (sending load_scene too
    soon steps on the exit and the sim never leaves the scene), then re-send
    load_scene every `retry_every` until `loaded` flips True — so a slow exit
    self-corrects. Returns True on confirmed reload, False on timeout.
    """
    handler = controller.handler
    handler.loaded = False
    handler.send_exit_scene()
    time.sleep(exit_settle)  # no exit signal exists — let the exit complete first
    deadline = time.monotonic() + timeout
    while not handler.loaded:
        if time.monotonic() >= deadline:
            return False
        handler.send_load_scene(scene)
        time.sleep(retry_every)
    return True


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
        latent_lags: tuple[int, ...] = (),
        reward_config: RaffinRewardConfig | None = None,
        scene_reload_every: int = 0,
        scene_reload_alpha: float = 0.0,
        scene_reload_kmin: int = 200,
        scene_reload_kmax: int = 2000,
        reversal_unwind_scale: float = 1.0,
        reversal_cte_threshold: float = 0.40,
        reversal_head_deg_threshold: float = 8.0,
        reversal_min_last_steer: float = 0.30,
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
        self.last_steer = 0.0
        self._prev_lap_count = 0
        self.reversal_unwind_scale = float(reversal_unwind_scale)
        self.reversal_cte_threshold = float(reversal_cte_threshold)
        self.reversal_head_deg_threshold = float(reversal_head_deg_threshold)
        self.reversal_min_last_steer = float(reversal_min_last_steer)
        self._last_geom_cte_abs = 0.0
        self._last_geom_head_abs_deg = 0.0
        self._last_reversal_unwind_active = False
        # Scene-reload (domain randomization). Two mutually exclusive modes:
        #   adaptive step-based (scene_reload_alpha > 0): reload after the car has
        #     spent ~K steps on the current scene, K = clamp(alpha * recent_ep_len,
        #     [kmin, kmax]). Reload is decided only at episode boundaries (never
        #     force-truncates), so max_episode_steps stays reachable and SAC's
        #     truncation handling is untouched. _steps_on_scene accumulates across
        #     crashes on the same scene and only resets on an actual reload, so
        #     easy (long-episode) and hard (short-episode) scenes contribute
        #     roughly equal steps to the buffer.
        #   episode-based (scene_reload_every > 0): reload every N episodes (used
        #     by eval with N=1 to randomize every episode).
        self.scene_reload_every = scene_reload_every
        self.scene_reload_alpha = scene_reload_alpha
        self.scene_reload_kmin = scene_reload_kmin
        self.scene_reload_kmax = scene_reload_kmax
        self._reset_count = 0
        self._episode_steps = 0
        self._steps_on_scene = 0
        self._ep_len_ema: float | None = None

        self.action_space = gym.spaces.Box(
            low=np.array([-self.max_steering, -1.0], dtype=np.float32),
            high=np.array([self.max_steering, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.set_observation(vae, latent_lags)
        self.last_throttle = self.min_throttle

    def set_observation(self, encoder, latent_lags: tuple[int, ...] = ()) -> None:
        """Choose the encoder and which earlier latents the observation carries.

        The observation is [latent now, latent `lag` steps ago for each lag,
        command history]. Earlier latents let the policy see how the view is
        changing (how fast it is drifting across the road), which one frame
        cannot show. Call between episodes only.
        """
        self.vae = encoder
        self.latent_lags = tuple(int(lag) for lag in latent_lags)
        if any(lag <= 0 for lag in self.latent_lags):
            raise ValueError("latent_lags must be positive step counts")
        self._latents: deque = deque(maxlen=max(self.latent_lags, default=0) + 1)
        obs_dim = encoder.z_size * (1 + len(self.latent_lags)) + 2 * self.n_command_history
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def set_control_fixes(
        self,
        *,
        reversal_unwind_scale: float | None = None,
        reversal_cte_threshold: float | None = None,
        reversal_head_deg_threshold: float | None = None,
        reversal_min_last_steer: float | None = None,
    ) -> None:
        """Configure gated opposite-sign steering unwind between episodes."""
        if reversal_unwind_scale is not None:
            self.reversal_unwind_scale = float(reversal_unwind_scale)
        if reversal_cte_threshold is not None:
            self.reversal_cte_threshold = float(reversal_cte_threshold)
        if reversal_head_deg_threshold is not None:
            self.reversal_head_deg_threshold = float(reversal_head_deg_threshold)
        if reversal_min_last_steer is not None:
            self.reversal_min_last_steer = float(reversal_min_last_steer)

    def attach_sync_controller(self, sync_controller) -> None:
        prev = getattr(self, "_sync_controller", None)
        if prev is not None and prev is not sync_controller:
            prev.close()
        self._sync_controller = sync_controller

    def close(self) -> None:
        sync = getattr(self, "_sync_controller", None)
        self._sync_controller = None
        try:
            if sync is not None:
                sync.close()
        finally:
            super().close()

    def reset(self, **kwargs):
        self.command_history.fill(0.0)
        self._latents.clear()
        self.last_steer = 0.0
        self.last_throttle = self.min_throttle
        self._prev_lap_count = 0
        self._last_geom_cte_abs = 0.0
        self._last_geom_head_abs_deg = 0.0
        self._last_reversal_unwind_active = False
        self._reset_count += 1
        # update episode-length EMA from the episode that just ended
        if self._episode_steps > 0:
            if self._ep_len_ema is None:
                self._ep_len_ema = float(self._episode_steps)
            else:
                self._ep_len_ema = 0.9 * self._ep_len_ema + 0.1 * self._episode_steps
        self._episode_steps = 0

        do_reload, reason = False, ""
        if self.scene_reload_alpha > 0.0:  # adaptive step-based
            ema = self._ep_len_ema if self._ep_len_ema is not None else float(self.scene_reload_kmin)
            k = int(min(self.scene_reload_kmax,
                        max(self.scene_reload_kmin, self.scene_reload_alpha * ema)))
            if self._steps_on_scene >= k:
                do_reload = True
                reason = f"steps_on_scene={self._steps_on_scene} >= K={k} (ema_eplen={ema:.0f})"
        elif self.scene_reload_every > 0:  # episode-based (eval / legacy)
            if self._reset_count % self.scene_reload_every == 0:
                do_reload = True
                reason = f"every {self.scene_reload_every} episodes"

        if do_reload:
            controller = self.env.unwrapped.viewer
            scene = controller.handler.SceneToLoad
            print(f"[scene-reload] reset #{self._reset_count}: reloading {scene} ({reason})",
                  flush=True)
            if not reload_scene(controller, scene):
                raise RuntimeError(f"Scene reload timed out at reset #{self._reset_count}; layout not counted")
            self._steps_on_scene = 0

        obs, info = self.env.reset(**kwargs)
        self._prev_lap_count = int(info.get("lap_count", 0))
        return self._build_obs(obs, info), info

    def step(self, action):
        self._episode_steps += 1
        self._steps_on_scene += 1
        raw_action = np.asarray(action, dtype=np.float32).reshape(-1)
        prev_steer = self.last_steer
        executed = self._map_and_clip_action(raw_action)
        obs, original_reward, terminated, truncated, info = self.env.step(executed)
        if "cte" not in info or not np.isfinite(float(info["cte"])):
            raise ValueError("Missing or non-finite CTE telemetry")
        base_terminated = terminated
        terminated = self._is_game_over(terminated, info)
        reward = self._calculate_reward(terminated, executed[1], info)
        cur_lap_count = int(info.get("lap_count", 0))
        if cur_lap_count > self._prev_lap_count and self.reward_config.lap_completion_bonus > 0.0:
            reward += self.reward_config.lap_completion_bonus * (cur_lap_count - self._prev_lap_count)
        self._prev_lap_count = cur_lap_count
        self._push_history(executed)

        info = dict(info)
        cte = float(info.get("cte", 0.0))
        info.update(
            {
                "original_reward": original_reward,
                "rl_steer": float(executed[0]),
                "rl_steer_delta": float(executed[0]) - prev_steer,
                "rl_throttle": float(executed[1]),
                "abs_cte": abs(cte - self.reward_config.cte_target),
                "vae_reward": reward,
                "reversal_unwind_active": bool(self._last_reversal_unwind_active),
                "termination_reason": ("simulator" if base_terminated else "cte_limit") if terminated else None,
            }
        )
        encoded_obs = self._build_obs(obs, info)
        info["encoder_seconds"] = self.last_encode_seconds
        return encoded_obs, reward, terminated, truncated, info

    def _map_and_clip_action(self, action: np.ndarray) -> np.ndarray:
        steer = float(np.clip(action[0], -self.max_steering, self.max_steering))
        t = float(np.clip((action[1] + 1.0) / 2.0, 0.0, 1.0))
        throttle = (1.0 - t) * self.min_throttle + self.max_throttle * t

        # Rate limit against the last executed command; it must not depend on
        # how much command history the observation carries.
        max_diff = (self.max_steering_diff - 1e-5) * (2.0 * self.max_steering)
        eff_max_diff = max_diff
        unwind_active = False
        if (
            self.reversal_unwind_scale > 1.0
            and steer * self.last_steer < 0.0
            and abs(self.last_steer) >= self.reversal_min_last_steer
            and (
                self._last_geom_cte_abs >= self.reversal_cte_threshold
                or self._last_geom_head_abs_deg >= self.reversal_head_deg_threshold
            )
        ):
            # Asymmetric unwind-to-zero only: speed return toward 0 when stuck at large
            # opposite steering under excursion, without crossing 0 or causing ping-pong.
            eff_max_diff = max(max_diff, min(self.reversal_unwind_scale * max_diff, abs(self.last_steer)))
            unwind_active = eff_max_diff > max_diff + 1e-7 and abs(steer - self.last_steer) > max_diff + 1e-7
        self._last_reversal_unwind_active = unwind_active
        steer = self.last_steer + float(np.clip(steer - self.last_steer, -eff_max_diff, eff_max_diff))

        self.last_steer = float(np.float32(steer))  # as stored in the command history
        self.last_throttle = float(throttle)
        return np.array([steer, throttle], dtype=np.float32)

    def _push_history(self, action: np.ndarray) -> None:
        if self.n_command_history <= 0:
            return
        self.command_history[:-1] = self.command_history[1:]
        self.command_history[-1] = action

    def _build_obs(self, obs: np.ndarray, info: dict) -> np.ndarray:
        self.last_raw_obs = np.array(obs, copy=True)
        started = time.perf_counter()
        if getattr(self.vae, "uses_telemetry", False):  # ground-truth state instead of the image
            z = self.vae.encode(self.last_raw_obs, info)
        else:
            z = self.vae.encode(self.last_raw_obs)
        self.last_encode_seconds = time.perf_counter() - started
        geom = getattr(self.vae, "last_pred_geom", None)
        if geom is not None and len(geom) >= 3:
            self._last_geom_cte_abs = abs(float(geom[0])) * 2.0
            self._last_geom_head_abs_deg = abs(float(np.degrees(np.arctan2(geom[1], geom[2]))))
        elif getattr(self.vae, "uses_telemetry", False) and len(z) >= 3:
            self._last_geom_cte_abs = abs(float(z[0])) * 2.0
            self._last_geom_head_abs_deg = abs(float(np.degrees(np.arctan2(z[1], z[2]))))
        else:
            self._last_geom_cte_abs = 0.0
            self._last_geom_head_abs_deg = 0.0
        # Newest first; right after a reset every earlier latent is the first frame.
        self._latents.appendleft(z)
        while len(self._latents) < self._latents.maxlen:
            self._latents.append(z)
        parts = [z] + [self._latents[lag] for lag in self.latent_lags]
        if self.n_command_history > 0:
            parts.append(self.command_history.reshape(-1))
        return parts[0] if len(parts) == 1 else np.concatenate(parts, dtype=np.float32)

    def _is_game_over(self, terminated: bool, info: dict) -> bool:
        cte_dev = abs(float(info.get("cte", 0.0)) - self.reward_config.cte_target)
        return bool(terminated or cte_dev > self.reward_config.max_cte_error)

    def _calculate_reward(self, terminated: bool, throttle: float, info: dict) -> float:
        cfg = self.reward_config
        speed = float(info.get("speed", 0.0))
        if terminated:
            norm_throttle = (throttle - self.min_throttle) / max(self.max_throttle - self.min_throttle, 1e-6)
            crash_speed = speed if cfg.speed_reward_weight > 0.0 else norm_throttle
            return float(cfg.reward_crash - cfg.crash_speed_weight * crash_speed)
        speed_reward = cfg.speed_reward_weight * speed
        cte_dev = abs(float(info.get("cte", 0.0)) - cfg.cte_target)
        cte_speed_penalty = cfg.cte_speed_penalty_weight * cte_dev * speed
        if cfg.min_alive_speed > 0.0:
            ramp = float(np.clip(speed / cfg.min_alive_speed, 0.0, 1.0))
            alive_scale = cfg.alive_scale_floor + (1.0 - cfg.alive_scale_floor) * ramp
        else:
            alive_scale = 1.0
        return float(
            cfg.alive_reward * alive_scale + speed_reward - cte_speed_penalty
        )


class ConfigCheckpointCallback(CheckpointCallback):
    """Checkpoint the policy on schedule; write the replay buffer (plus its
    configuration sidecar) at the next episode end.

    Pickling and hashing the buffer takes long enough that doing it mid-episode
    leaves the car driving on a stale command. At an episode end the car has
    just been reset, and no update has run since the policy checkpoint, so the
    saved pair is consistent; the buffer holds the rest of that episode too.
    """

    def __init__(self, *args, save_replay_buffer: bool = False, **kwargs) -> None:
        super().__init__(*args, save_replay_buffer=False, **kwargs)
        self.deferred_replay_buffer = save_replay_buffer
        self._pending_replay_path: Path | None = None

    def _save_pending_replay(self) -> None:
        path, self._pending_replay_path = self._pending_replay_path, None
        if path is not None:
            self.model.save_replay_buffer(str(path))
            save_replay_metadata(path, self.model)

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.deferred_replay_buffer and self.n_calls % self.save_freq == 0:
            self._pending_replay_path = Path(self._checkpoint_path("replay_buffer_", extension="pkl"))
        if self._pending_replay_path is not None and np.any(self.locals.get("dones", [True])):
            self._save_pending_replay()
        return result

    def _on_training_end(self) -> None:
        self._save_pending_replay()


class ReplayWarmupCallback(BaseCallback):
    """Hold SAC updates until the replay buffer holds `min_transitions`.

    For resuming without a replay buffer: the loaded policy drives (it is never
    replaced by random actions, unlike `learning_starts`) and nothing is
    updated until there is enough fresh data to sample from. Must come after
    CappedDynamicGradientStepsCallback in the callback list.

    SB3 skips the update when gradient_steps is 0. The zero is held only
    between a rollout's end and the next rollout's start, so checkpoints
    written during a rollout never contain it.
    """

    def __init__(self, min_transitions: int) -> None:
        super().__init__()
        self.min_transitions = int(min_transitions)
        self._held: int | None = None

    def release(self) -> None:
        if self._held is not None:
            self.model.gradient_steps, self._held = self._held, None

    def _on_rollout_start(self) -> None:
        self.release()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.model.replay_buffer.size() < self.min_transitions:
            self._held, self.model.gradient_steps = self.model.gradient_steps, 0

    def _on_training_end(self) -> None:
        self.release()


class CappedDynamicGradientStepsCallback(BaseCallback):
    """gradient_steps = clamp(episode_length * scale, [floor, cap]).

    Used with train_freq=(1, 'episode'). Each training cycle scales updates with
    collected data, capped to prevent runaway compute on long episodes and
    floored to guarantee a minimum amount of off-policy learning per cycle.
    `scale` < 1 reduces updates relative to env steps (useful to dampen
    late-training overfit on long truncate episodes).
    """

    def __init__(self, cap: int, floor: int = 1, scale: float = 1.0) -> None:
        super().__init__()
        self.cap = cap
        self.floor = max(1, int(floor))
        self.scale = float(scale)
        self._rollout_start_ts = 0

    def _on_rollout_start(self) -> None:
        self._rollout_start_ts = int(self.model.num_timesteps)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.model.train_freq.unit.value != "episode":
            return
        ep_len = max(1, int(self.model.num_timesteps) - self._rollout_start_ts)
        scaled = int(ep_len * self.scale)
        new_grads = max(self.floor, min(scaled, self.cap))
        self.model.gradient_steps = new_grads
        # use record_mean: per-rollout values aggregated across dump window
        self.logger.record_mean("train/gradient_steps_used", new_grads)
        self.logger.record_mean("train/rollout_ep_len", ep_len)


class DonkeyInfoCallback(BaseCallback):
    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._best_lap_time: float = float("inf")

    def _on_step(self) -> bool:
        # NOTE: use record_mean (accumulating average across dump window), not record
        # (overwrite — would only show the value at the last _on_step before dump).
        infos = self.locals.get("infos", [])
        if not infos:
            return True
        ctes = [float(info["abs_cte"]) for info in infos if "abs_cte" in info]
        speeds = [float(info.get("speed", 0.0)) for info in infos]
        steers = [float(info["rl_steer"]) for info in infos if "rl_steer" in info]
        throttles = [float(info["rl_throttle"]) for info in infos if "rl_throttle" in info]
        if ctes:
            self.logger.record_mean("donkey/abs_cte_mean", float(np.mean(ctes)))
        if speeds:
            self.logger.record_mean("donkey/speed_mean", float(np.mean(speeds)))
        if steers:
            self.logger.record_mean("donkey/abs_steer_mean", float(np.mean(np.abs(steers))))
        deltas = [abs(float(info["rl_steer_delta"])) for info in infos if "rl_steer_delta" in info]
        if deltas:
            self.logger.record_mean("donkey/abs_steer_delta_mean", float(np.mean(deltas)))
        if throttles:
            self.logger.record_mean("donkey/throttle_mean", float(np.mean(throttles)))

        lap_counts = [int(info["lap_count"]) for info in infos if "lap_count" in info]
        if lap_counts:
            self.logger.record_mean("donkey/lap_count_mean", float(np.mean(lap_counts)))
        lap_times = [float(info["last_lap_time"]) for info in infos
                     if "last_lap_time" in info and float(info["last_lap_time"]) > 0]
        if lap_times:
            self.logger.record_mean("donkey/last_lap_time_mean", float(np.mean(lap_times)))
            batch_min = float(np.min(lap_times))
            if batch_min < self._best_lap_time:
                self._best_lap_time = batch_min
            # best is a running min across the whole run, OK to record() (overwrite is correct semantics)
            self.logger.record("donkey/last_lap_time_best", self._best_lap_time)

        forward_vels = [float(info["forward_vel"]) for info in infos if "forward_vel" in info]
        if forward_vels:
            self.logger.record_mean("donkey/forward_vel_mean", float(np.mean(forward_vels)))
        lateral_vels = [abs(float(info["vel"][0])) for info in infos
                        if "vel" in info and hasattr(info["vel"], "__len__") and len(info["vel"]) >= 1]
        if lateral_vels:
            self.logger.record_mean("donkey/lateral_vel_mean", float(np.mean(lateral_vels)))
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
        latent_lags=getattr(args, "latent_lags", ()),
        reward_config=reward_config_from_args(args),
        scene_reload_every=getattr(args, "scene_reload_every", 0),
        scene_reload_alpha=getattr(args, "scene_reload_alpha", 0.0),
        scene_reload_kmin=getattr(args, "scene_reload_kmin", 200),
        scene_reload_kmax=max(1, args.max_episode_steps - 1),
    )
    configure_env_control_fixes(env, args)
    env = TimeLimit(env, max_episode_steps=args.max_episode_steps)
    # Resuming into the same directory appends to its episode log.
    return Monitor(env, filename=str(args.output_dir / "monitor.csv"),
                   override_existing=getattr(args, "resume_model", None) is None)


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
    parser.add_argument("--reward-crash", type=float, default=REWARD_CRASH)
    parser.add_argument("--crash-speed-weight", type=float, default=CRASH_SPEED_WEIGHT)
    parser.add_argument("--alive-reward", type=float, default=ALIVE_REWARD)
    parser.add_argument("--speed-reward-weight", type=float, default=SPEED_REWARD_WEIGHT)
    parser.add_argument("--min-alive-speed", type=float, default=MIN_ALIVE_SPEED)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=128)
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
        default=1000,
        help="gradient_steps becomes clamp(episode_length * scale, [min, cap]) per cycle. "
             "Pass 0 or a negative value to disable the cap.",
    )
    parser.add_argument("--gradient-steps-min", type=int, default=50,
                        help="Floor on dynamic gradient_steps.")
    parser.add_argument("--gradient-steps-scale", type=float, default=1.0,
                        help="Multiplier on episode length before clamping. "
                             "Default 1.0 = 1 update per env step (raffin-style). "
                             "Set <1 to dampen overfit on long episodes (experimental).")
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
    add_control_fix_args(parser)
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
