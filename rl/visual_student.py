"""Camera-only geometry and student actor. No telemetry, map, or TrackState API."""
from __future__ import annotations

from collections import deque
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

OBSERVATION_CONTRACT = {"format": "camera_geometry_history_85_v1", "sources": ["camera", "self_commands"],
                        "geometry_dim": 15, "geometry_lags": [2, 4], "history_steps": 20,
                        "observation_dim": 85, "motion_truth": False}
STUDENT_FORMAT = "visual_student_no_motion_v1"
TEACHER_GEOMETRY = [0, 1, 2] + list(range(6, 18))


def padded_lag_indices(groups, steps, lags=(2, 4)):
    """Pad within each contiguous sequence; expose reset/complete-context masks."""
    groups, steps = np.asarray(groups), np.asarray(steps)
    indices = np.empty((len(steps), 1 + len(lags)), np.int64)
    full = np.zeros(len(steps), bool)
    reset_known = np.zeros(len(steps), bool)
    for group in dict.fromkeys(groups):
        rows = np.flatnonzero(groups == group)
        rows = rows[np.argsort(steps[rows], kind="stable")]
        if np.any(np.diff(steps[rows]) == 0):
            raise ValueError("Duplicate frame step in a sequence")
        for segment in np.split(rows, np.flatnonzero(np.diff(steps[rows]) != 1) + 1):
            position = np.arange(len(segment))
            for column, lag in enumerate((0, *lags)):
                indices[segment, column] = segment[np.maximum(position - lag, 0)]
            full[segment] = position >= max(lags, default=0)
            reset_known[segment] = steps[segment[0]] == 0
    return indices, full, reset_known


def normalize_geometry(values):
    out = np.array(values, np.float32, copy=True)
    out[..., 1:3] /= np.maximum(np.linalg.norm(out[..., 1:3], axis=-1, keepdims=True), 1e-6)
    return out


class GeometryProjection:
    """Frozen image-feature projection with matching batch/live causal padding."""

    def __init__(self, checkpoint: Path, device="cpu"):
        self.path = Path(checkpoint)
        saved = torch.load(self.path, map_location=device, weights_only=True)
        if saved.get("format") != "proj15_lags_ridge_v1" or saved.get("lags") != [2, 4]:
            raise ValueError("Expected the frozen two-stage geometry projection")
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.crop_top = int(saved.get("crop_top", 0))
        self.weights = {k: saved[k].float().to(device) for k in ["w1", "b1", "w2", "b2"]}
        self.device = torch.device(device)
        self.buffer = deque(maxlen=5)

    def reset(self):
        self.buffer.clear()

    @torch.no_grad()
    def step(self, grid):
        w = self.weights
        value = (torch.as_tensor(grid, dtype=torch.float32, device=self.device).reshape(1, 6144) @ w["w1"] + w["b1"])[0]
        self.buffer.appendleft(value)
        while len(self.buffer) < 5:
            self.buffer.append(value)
        stacked = torch.cat([self.buffer[i] for i in [0, 2, 4]])
        return normalize_geometry((stacked @ w["w2"] + w["b2"]).cpu().numpy())

    @torch.no_grad()
    def batch(self, grid, indices):
        w = self.weights
        stage1 = np.empty((len(grid), 15), np.float32)
        for i in range(0, len(grid), 4096):
            x = torch.as_tensor(grid[i:i+4096], dtype=torch.float32, device=self.device)
            stage1[i:i+len(x)] = (x @ w["w1"] + w["b1"]).cpu().numpy()
        stacked = np.concatenate([stage1[indices[:, i]] for i in range(3)], axis=1)
        return normalize_geometry((torch.from_numpy(stacked).to(self.device) @ w["w2"] + w["b2"]).cpu().numpy())


class VisualGeometryEncoder:
    uses_telemetry = False
    z_size = 15
    model_name = "dinov2_geometry_image_only"

    def __init__(self, checkpoint: Path, device="cpu", dino_encoder=None):
        self.head = GeometryProjection(checkpoint, device)
        self.crop_top = self.head.crop_top
        self.head_sha256 = self.head.sha256
        if dino_encoder is None:
            from rl.train_vae_sac import FrozenDinoGridEncoder
            dino_encoder = FrozenDinoGridEncoder(device=torch.device(device), crop_top=self.crop_top, projection=None)
        if dino_encoder.crop_top != self.crop_top:
            raise ValueError("Image crop differs from frozen geometry head")
        self.dino = dino_encoder
        self.last_pred_geom = None

    def reset_episode(self):
        self.head.reset()
        self.last_pred_geom = None

    def state_dict(self):
        state = {"geometry." + key: value for key, value in self.head.weights.items()}
        if hasattr(self.dino, "model"):
            state.update({"dino." + key: value for key, value in self.dino.model.state_dict().items()})
        return state

    @torch.no_grad()
    def encode(self, image):
        grid = self.dino.grid_features(np.asarray(image, np.uint8)[None])
        self.last_pred_geom = self.head.step(grid)
        return self.last_pred_geom.copy()


def student_observation(geometry_at_lags, history):
    geometry, history = np.asarray(geometry_at_lags, np.float32), np.asarray(history, np.float32)
    if geometry.shape[-2:] != (3, 15) or history.shape[-1:] != (40,):
        raise ValueError("Student needs three 15D predicted geometries and 40 self commands")
    out = np.concatenate([geometry.reshape(*geometry.shape[:-2], 45), history], axis=-1)
    if not np.isfinite(out).all():
        raise ValueError("Non-finite student input")
    return out


def prepare_visual_inputs(grid, history, groups, steps, projection):
    """Only cached image features and issued commands enter the student path."""
    indices, full, known_reset = padded_lag_indices(groups, steps)
    geometry = projection.batch(grid, indices)
    observations = student_observation(geometry[indices], history)
    full_context = full[indices].all(axis=1)
    return observations, known_reset | full_context, full_context, geometry


def map_actions(raw, history):
    """Differentiable production mapping; no motion state or reversal acceleration."""
    previous = history[..., -2]
    steer = previous + (raw[..., 0].clamp(-1, 1) - previous).clamp(-0.19998, 0.19998)
    throttle = 0.2 + 0.5 * ((raw[..., 1] + 1) / 2).clamp(0, 1)
    return torch.stack([steer, throttle], dim=-1)


class VisualStudentActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(85, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, 2), nn.Tanh())

    def forward(self, observation):
        if observation.shape[-1] != 85:
            raise ValueError("Camera-only student requires exactly 85 inputs")
        return self.net(observation)

    @torch.no_grad()
    def initialize_from_teacher(self, teacher):
        actor = teacher.actor
        if actor.latent_pi[0].weight.shape != (256, 58):
            raise ValueError("Unexpected teacher architecture")
        self.net[0].weight.zero_()
        self.net[0].weight[:, :15].copy_(actor.latent_pi[0].weight[:, TEACHER_GEOMETRY])
        self.net[0].weight[:, 45:].copy_(actor.latent_pi[0].weight[:, 18:58])
        self.net[0].bias.copy_(actor.latent_pi[0].bias)
        self.net[2].load_state_dict(actor.latent_pi[2].state_dict())
        self.net[4].load_state_dict(actor.mu.state_dict())

    @torch.no_grad()
    def predict(self, observation, deterministic=True):
        if not deterministic:
            raise ValueError("Student is a deterministic behavior-cloning actor")
        values = torch.as_tensor(observation, dtype=torch.float32, device=next(self.parameters()).device)
        if values.ndim not in (1, 2) or values.shape[-1] != 85 or not torch.isfinite(values).all():
            raise ValueError("Invalid camera-only student observation")
        return self(values).cpu().numpy(), None

    def save(self, path, *, geometry_head_sha256, metadata=None):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": STUDENT_FORMAT, "observation_contract": OBSERVATION_CONTRACT,
                    "geometry_head_sha256": geometry_head_sha256, "state_dict": self.state_dict(),
                    "metadata": metadata or {}}, path)

    @classmethod
    def load(cls, path, *, geometry_head_sha256, device="cpu"):
        try:
            saved = torch.load(path, map_location=device, weights_only=True)
        except Exception as exc:
            raise ValueError("Not a supported camera-only student checkpoint") from exc
        if saved.get("format") != STUDENT_FORMAT or saved.get("observation_contract") != OBSERVATION_CONTRACT:
            raise ValueError("Checkpoint is not the 85D camera-only student")
        if saved.get("geometry_head_sha256") != geometry_head_sha256:
            raise ValueError("Student's frozen geometry head hash differs")
        model = cls().to(device)
        model.load_state_dict(saved["state_dict"])
        model.eval()
        return model
