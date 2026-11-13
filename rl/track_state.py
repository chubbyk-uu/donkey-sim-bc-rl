"""Observation built from simulator ground truth instead of the camera image.

A diagnostic, not a deployable policy input: it answers how well the same SAC
recipe drives when perception is perfect. The policy gets where the car is
relative to the road, how it is moving, and the road ahead:

    [cte / 2, sin and cos of the heading error, forward speed / 2,
     sideways speed * 4, yaw rate / 2,
     (right, forward) of the centre line at each LOOKAHEAD distance, divided
     by that distance]

All in the car's own frame, scaled to stay within about [-1, 1].
The centre line comes from tools/fit_track_centerline.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

LOOKAHEAD = (0.3, 0.6, 1.0, 1.5, 2.0, 3.0)  # position units; the car covers about 0.1 per step


class TrackState:
    model_name = "sim_state"
    crop_top = 0
    uses_telemetry = True
    z_size = 6 + 2 * len(LOOKAHEAD)

    def __init__(self, centre: np.ndarray, cte_per_unit: float = 1.0) -> None:
        self.centre = np.asarray(centre, dtype=np.float64)
        self.cte_per_unit = float(cte_per_unit)
        ahead = np.roll(self.centre, -1, axis=0) - np.roll(self.centre, 1, axis=0)
        self.tangent = ahead / np.linalg.norm(ahead, axis=1, keepdims=True)
        self.spacing = float(np.linalg.norm(self.centre[1] - self.centre[0]))
        self._steps_ahead = [round(distance / self.spacing) for distance in LOOKAHEAD]

    @classmethod
    def load(cls, path: Path) -> "TrackState":
        saved = np.load(path)
        return cls(saved["centre"], float(saved["cte_per_unit"]))

    def state_dict(self) -> dict[str, torch.Tensor]:
        """What identifies this observation, for the evaluation record."""
        return {"centre": torch.from_numpy(self.centre), "lookahead": torch.tensor(LOOKAHEAD)}

    @staticmethod
    def _axes(yaw_degrees: float) -> tuple[np.ndarray, np.ndarray]:
        """Unit vectors (forward, right) of the car in the (x, z) plane; yaw 0 faces +z and grows towards +x."""
        yaw = np.radians(yaw_degrees)
        forward = np.array([np.sin(yaw), np.cos(yaw)])
        return forward, np.array([forward[1], -forward[0]])

    def locate(self, position, yaw_degrees: float) -> tuple[int, float, float]:
        """(index of the nearest centre-line point, distance to the right of it, heading error in radians).

        Plain nearest point: separate stretches of this track are never closer
        than 1.6 position units, six times the distance at which an episode ends.
        """
        position = np.asarray(position, dtype=np.float64)
        forward, _ = self._axes(yaw_degrees)
        index = int(np.argmin(np.einsum("ij,ij->i", self.centre - position, self.centre - position)))
        tangent = self.tangent[index]
        offset = position - self.centre[index]
        right_of_centre = float(offset[0] * tangent[1] - offset[1] * tangent[0])
        heading_error = float(np.arctan2(forward[0] * tangent[1] - forward[1] * tangent[0], forward @ tangent))
        return index, right_of_centre, heading_error

    def encode(self, image, info: dict) -> np.ndarray:
        position = np.array([info["pos"][0], info["pos"][2]], dtype=np.float64)
        yaw = float(info["car"][2])
        forward, right = self._axes(yaw)
        index, _, heading_error = self.locate(position, yaw)
        features = [float(info["cte"]) / 2.0, np.sin(heading_error), np.cos(heading_error),
                    float(info["vel"][2]) / 2.0, float(info["vel"][0]) * 4.0, float(info["gyro"][1]) / 2.0]
        for distance, steps in zip(LOOKAHEAD, self._steps_ahead):
            target = self.centre[(index + steps) % len(self.centre)] - position
            features += [target @ right / distance, target @ forward / distance]
        return np.asarray(features, dtype=np.float32)
