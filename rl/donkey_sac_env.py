from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass
class DonkeyRewardConfig:
    """Raffin / learning-to-drive-in-5-minutes canonical reward.

    alive while inside track + throttle bonus; crash penalty proportional to
    throttle on terminate. steer_penalty / steer_delta_penalty default to 0 and
    are kept as optional shaping for snake-mitigation if needed.
    """

    alive_reward: float = 1.0
    throttle_reward_weight: float = 0.1
    terminal_cte: float = 4.0
    crash_penalty: float = 10.0
    crash_throttle_weight: float = 5.0
    steer_penalty: float = 0.0
    steer_delta_penalty: float = 0.0


class DonkeySACEnv(gym.Wrapper):
    """Donkey Gym wrapper for image-based SAC.

    The policy receives cropped channel-first RGB images and outputs steering.
    Throttle is fixed by default so the first RL stage only learns lateral
    control.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        fixed_throttle: float | None = None,
        max_steering: float = 0.5,
        min_throttle: float = 0.05,
        max_throttle: float = 0.6,
        action_smoothing: float = 0.0,
        max_steer_delta: float | None = None,
        n_command_history: int = 0,
        reward_config: DonkeyRewardConfig | None = None,
    ) -> None:
        super().__init__(env)
        self.fixed_throttle = fixed_throttle
        self.max_steering = max_steering
        self.min_throttle = min_throttle
        self.max_throttle = max_throttle
        self.action_smoothing = float(np.clip(action_smoothing, 0.0, 0.99))
        self.max_steer_delta = max_steer_delta
        self.reward_config = reward_config or DonkeyRewardConfig()
        self.prev_steer = 0.0
        self.n_command_history = max(0, int(n_command_history))

        action_dim = 1 if fixed_throttle is not None else 2
        self._action_dim = action_dim
        self._history = np.zeros((self.n_command_history, action_dim), dtype=np.float32)

        image_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, 110, 160),
            dtype=np.uint8,
        )
        if self.n_command_history > 0:
            history_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.n_command_history * action_dim,),
                dtype=np.float32,
            )
            self.observation_space = gym.spaces.Dict(
                {"image": image_space, "history": history_space}
            )
        else:
            self.observation_space = image_space

        if fixed_throttle is None:
            self.action_space = gym.spaces.Box(
                low=np.array([-self.max_steering, self.min_throttle], dtype=np.float32),
                high=np.array([self.max_steering, self.max_throttle], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = gym.spaces.Box(
                low=-self.max_steering,
                high=self.max_steering,
                shape=(1,),
                dtype=np.float32,
            )

    def _normalize_executed(self, steer: float, throttle: float) -> np.ndarray:
        steer_norm = float(np.clip(steer / max(self.max_steering, 1e-6), -1.0, 1.0))
        if self.fixed_throttle is not None:
            return np.array([steer_norm], dtype=np.float32)
        thr_range = max(self.max_throttle - self.min_throttle, 1e-6)
        thr_norm = float(np.clip(2.0 * (throttle - self.min_throttle) / thr_range - 1.0, -1.0, 1.0))
        return np.array([steer_norm, thr_norm], dtype=np.float32)

    def _push_history(self, normalized: np.ndarray) -> None:
        if self.n_command_history == 0:
            return
        self._history[:-1] = self._history[1:]
        self._history[-1] = normalized

    def _build_obs(self, image: np.ndarray):
        if self.n_command_history > 0:
            return {"image": image, "history": self._history.flatten()}
        return image

    def reset(self, **kwargs):
        self.prev_steer = 0.0
        if self.n_command_history > 0:
            self._history.fill(0.0)
        obs, info = self.env.reset(**kwargs)
        print(f"[reset] initial cte={info.get('cte', '?')}", flush=True)
        return self._build_obs(self._preprocess_obs(obs)), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        target_steer = float(np.clip(action[0], -self.max_steering, self.max_steering))
        steer = (
            self.action_smoothing * self.prev_steer
            + (1.0 - self.action_smoothing) * target_steer
        )
        if self.max_steer_delta is not None:
            delta = np.clip(
                steer - self.prev_steer,
                -self.max_steer_delta,
                self.max_steer_delta,
            )
            steer = self.prev_steer + float(delta)
        steer = float(np.clip(steer, -1.0, 1.0))
        steer_delta = steer - self.prev_steer

        if self.fixed_throttle is None:
            throttle = float(np.clip(action[1], self.min_throttle, self.max_throttle))
        else:
            throttle = float(self.fixed_throttle)

        obs, original_reward, terminated, truncated, info = self.env.step([steer, throttle])
        reward, terminated = self._calculate_reward(terminated, info, steer, throttle)
        self.prev_steer = steer
        self._push_history(self._normalize_executed(steer, throttle))

        info = dict(info)
        info.update(
            {
                "original_reward": original_reward,
                "rl_steer": steer,
                "rl_steer_delta": steer_delta,
                "rl_throttle": throttle,
                "abs_cte": abs(float(info.get("cte", 0.0))),
            }
        )

        return self._build_obs(self._preprocess_obs(obs)), reward, terminated, truncated, info

    def _preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.uint8)
        obs = obs[10:, :, :]
        return np.transpose(obs, (2, 0, 1)).copy()

    def _calculate_reward(self, terminated: bool, info: dict, steer: float, throttle: float) -> tuple[float, bool]:
        cfg = self.reward_config
        cte = abs(float(info.get("cte", 0.0)))

        if cte >= cfg.terminal_cte:
            terminated = True

        if terminated:
            throttle_range = max(self.max_throttle - self.min_throttle, 1e-6)
            norm_throttle = float(np.clip((throttle - self.min_throttle) / throttle_range, 0.0, 1.0))
            return float(-cfg.crash_penalty - cfg.crash_throttle_weight * norm_throttle), terminated

        throttle_reward = cfg.throttle_reward_weight * (throttle / max(self.max_throttle, 1e-6))
        reward = cfg.alive_reward + float(throttle_reward)

        steer_delta = steer - self.prev_steer
        reward -= cfg.steer_penalty * float(steer**2)
        reward -= cfg.steer_delta_penalty * float(steer_delta**2)

        return float(reward), terminated
