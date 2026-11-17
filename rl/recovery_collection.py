"""Short steering interventions for geometry-labelled recovery collection.

This changes data collection actions only. The frozen teacher and its normal
action mapping regain control as soon as the bounded intervention ends.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RecoveryCollectionConfig:
    targets: tuple[float, ...] = (0.6, 1.0, 1.4)
    directions: tuple[int, ...] = (1, -1)
    steering: float = 0.35
    max_kick_steps: int = 4
    warmup_steps: int = 250
    cooldown_steps: int = 80
    recovery_timeout_steps: int = 160
    release_abs_cte: float = 1.65
    release_heading_deg: float = 15.0

    def __post_init__(self):
        if not self.directions or any(sign not in (-1, 1) for sign in self.directions):
            raise ValueError("Recovery directions must contain only +1 or -1")


class RecoveryCollector:
    def __init__(self, layout: int, config: RecoveryCollectionConfig | None = None):
        self.config = config or RecoveryCollectionConfig()
        self.phase = "cruise"
        self.next_goal = (layout - 1) % (len(self.config.directions) * len(self.config.targets))
        self.ready_step = self.config.warmup_steps
        self.stable_steps = 0
        self.events: list[dict] = []
        self.active: dict | None = None

    def config_dict(self) -> dict:
        return asdict(self.config)

    def action(self, teacher_action: np.ndarray, step: int, state: np.ndarray) -> tuple[np.ndarray, dict]:
        """Use pre-action geometry; returned metadata describes the action at `step`."""
        cfg = self.config
        cte = float(state[0]) * 2.0
        heading = abs(float(np.degrees(np.arctan2(state[1], state[2]))))
        speed = float(state[3]) * 2.0
        if self.active is not None:
            self.active["peak_abs_cte"] = max(self.active["peak_abs_cte"], abs(cte))

        if self.phase == "kick":
            elapsed = step - self.active["start_step"]
            reason = (
                "target" if self.active["sign"] * cte >= self.active["target_abs_cte"] else
                "cte_guard" if abs(cte) >= cfg.release_abs_cte else
                "heading_guard" if heading >= cfg.release_heading_deg else
                "duration" if elapsed >= cfg.max_kick_steps else None
            )
            if reason is not None:
                self.phase = "recovery"
                self.active.update(release_step=step, release_reason=reason)

        if self.phase == "recovery":
            self.stable_steps = self.stable_steps + 1 if abs(cte) < 0.3 and heading < 5.0 else 0
            recovered = self.stable_steps >= 8
            timed_out = step - self.active["release_step"] >= cfg.recovery_timeout_steps
            if recovered or timed_out:
                self.active.update(end_step=step, recovered=recovered, end_reason="recovered" if recovered else "timeout")
                self.active = None
                self.phase = "cruise"
                self.ready_step = step + cfg.cooldown_steps
                self.stable_steps = 0

        # Start only from a centred, moving, approximately straight-driving state.
        straight_ahead = abs(float(state[10])) < 0.10 and abs(float(state[12])) < 0.15
        if (
            self.phase == "cruise" and step >= self.ready_step
            and abs(cte) < 0.3 and heading < 5.0 and speed >= 1.8
            and straight_ahead and abs(float(teacher_action[0])) < 0.35
        ):
            goal = self.next_goal
            self.next_goal = (goal + 1) % (len(cfg.directions) * len(cfg.targets))
            self.active = {
                "id": len(self.events) + 1, "start_step": step,
                "sign": cfg.directions[goal % len(cfg.directions)],
                "target_abs_cte": cfg.targets[goal // len(cfg.directions)],
                "peak_abs_cte": abs(cte), "recovered": False,
            }
            self.events.append(self.active)
            self.phase = "kick"

        applied = np.array(teacher_action, dtype=np.float32, copy=True)
        if self.phase == "kick":
            applied[0] = self.active["sign"] * cfg.steering
        note = {
            "recovery_phase": self.phase,
            "intervention_id": self.active["id"] if self.active else 0,
            "target_signed_cte": self.active["sign"] * self.active["target_abs_cte"] if self.active else 0.0,
        }
        return applied, note

    def finish(self, step: int, reason: str, final_cte: float) -> None:
        if self.active is not None:
            self.active["peak_abs_cte"] = max(self.active["peak_abs_cte"], abs(final_cte))
            self.active.update(end_step=step, end_reason=reason, recovered=False)
