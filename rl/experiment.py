"""Configuration contracts and durable evaluation records (no simulator needed)."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
import time
import warnings
from collections import deque
from datetime import datetime, timezone
from pathlib import Path


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_args(args) -> None:
    for key, value in vars(args).items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"--{key.replace('_', '-')} must be finite")
    for name in ("episodes", "layouts", "max_episode_steps", "batch_size", "buffer_size", "train_freq", "timesteps", "checkpoint_freq"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("capture_tail_steps", "resume_warmup_steps"):
        if getattr(args, name, 0) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")
    if not 0 <= args.min_throttle <= args.max_throttle <= 1:
        raise ValueError("Require 0 <= min_throttle <= max_throttle <= 1")
    if not 0 < args.max_steering_diff <= 1 or args.max_steering <= 0:
        raise ValueError("Require max_steering > 0 and 0 < max_steering_diff <= 1")
    if args.max_cte_error <= 0 or getattr(args, "n_command_history", 20) < 0:
        raise ValueError("Require max_cte_error > 0 and n_command_history >= 0")
    if not 0 <= getattr(args, "encoder_crop_top", 0) < 120:
        raise ValueError("encoder_crop_top must be in [0, 120)")
    if getattr(args, "resume_replay_buffer", None) and not getattr(args, "resume_model", None):
        raise ValueError("--resume-replay-buffer requires --resume-model")
    if getattr(args, "gradient_steps_cap", 0) > 0:
        if not 1 <= args.gradient_steps_min <= args.gradient_steps_cap or args.gradient_steps_scale <= 0:
            raise ValueError("Require 1 <= gradient_steps_min <= cap and gradient_steps_scale > 0")
        if getattr(args, "train_freq_unit", None) == "episode" and args.gradient_steps != -1:
            raise ValueError("Use --gradient-steps-cap 0 for fixed --gradient-steps; dynamic updates would override it")


def environment_config(args) -> dict:
    encoder = getattr(args, "encoder", "vae")
    return {
        "env_id": args.env_id,
        "observation": {
            "encoder": encoder,
            "vae_sha256": sha256(args.vae_model) if encoder == "vae" else None,
            "crop_top": 40 if encoder == "vae" else args.encoder_crop_top,
            "n_command_history": getattr(args, "n_command_history", 20),
        },
        "action": {key: getattr(args, key) for key in ("min_throttle", "max_throttle", "max_steering", "max_steering_diff")},
        "termination": {"max_cte_error": args.max_cte_error, "cte_target": getattr(args, "cte_target", 0.0)},
        "reward": {key: getattr(args, key, default) for key, default in {
            "reward_crash": -10.0, "crash_speed_weight": 5.0, "alive_reward": 1.0,
            "speed_reward_weight": 0.0, "min_alive_speed": 0.0, "alive_scale_floor": 0.0,
            "lap_completion_bonus": 0.0, "cte_speed_penalty_weight": 0.0,
        }.items()},
    }


def validate_model_config(model, current: dict) -> None:
    saved = getattr(model, "donkey_config", None)
    if saved is None:
        warnings.warn("Legacy checkpoint: encoder/action metadata unavailable; compatibility cannot be verified.", stacklevel=2)
        return
    for key in ("env_id", "observation", "action"):
        if saved.get(key) != current[key]:
            raise ValueError(f"Checkpoint {key} mismatch: saved={saved.get(key)!r}, requested={current[key]!r}")


def replay_metadata_path(path: Path) -> Path:
    return Path(str(path) + ".json")


def save_replay_metadata(path: Path, model) -> None:
    write_json(replay_metadata_path(path), {
        "environment": model.donkey_config, "num_timesteps": model.num_timesteps,
        "sha256": sha256(path),
    })


def validate_replay(path: Path, current: dict, allow_legacy: bool = False) -> None:
    sidecar = replay_metadata_path(path)
    if not sidecar.exists():
        if not allow_legacy:
            raise ValueError("Replay buffer metadata missing. Omit the old buffer, or explicitly use --allow-legacy-replay-buffer if its encoder, actions and rewards are known to match.")
        warnings.warn("Using legacy replay buffer with unverifiable rewards/configuration.", stacklevel=2)
        return
    saved = json.loads(sidecar.read_text())
    if saved["environment"] != current:
        raise ValueError("Replay buffer configuration/rewards differ. Resume without this buffer; stored rewards are not recalculated.")
    if saved.get("sha256") != sha256(path):
        raise ValueError("Replay buffer hash differs from its metadata")


def algorithm_config(model) -> dict:
    freq = model.train_freq
    return json_value({
        "batch_size": model.batch_size, "buffer_size": model.replay_buffer.buffer_size,
        "learning_starts": model.learning_starts, "learning_rate": model.lr_schedule(1),
        "train_freq": [freq.frequency, freq.unit.value], "gradient_steps": model.gradient_steps,
        "gamma": model.gamma, "tau": model.tau, "seed": model.seed,
        "policy_kwargs": model.policy_kwargs,
    })


class EvalRecord:
    def __init__(self, args, files: dict[str, Path]):
        self.path = args.output_dir or Path("logs") / ("eval_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        self.path.mkdir(parents=True, exist_ok=False)
        try:
            repo = Path(__file__).resolve().parent
            head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL, text=True).strip()
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, stderr=subprocess.DEVNULL, text=True))
        except (OSError, subprocess.CalledProcessError):
            head, dirty = None, None
        self.metadata = {
            "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
            "args": vars(args), "git_head": head, "git_dirty": dirty,
            "python": platform.python_version(),
            "versions": {key: importlib.metadata.version(key) for key in ("torch", "stable-baselines3", "gymnasium", "gym-donkeycar", "numpy")},
            "files": {key: {"path": str(path), "sha256": sha256(path)} for key, path in files.items()},
            "layout_note": "Layout numbers identify loads within this run; they are not reproducible simulator seeds.",
        }
        source_dir = self.path / "source"
        source_dir.mkdir()
        self.metadata["sources"] = {}
        for source in Path(__file__).parent.glob("*.py"):
            shutil.copy2(source, source_dir / source.name)
            self.metadata["sources"][source.name] = sha256(source_dir / source.name)
        self.results = []
        self.started = time.monotonic()
        self.telemetry = None
        self.capture = deque(maxlen=getattr(args, "capture_tail_steps", 0))
        self.capture_context = None
        self.metadata["telemetry_enabled"] = bool(args.save_telemetry or self.capture.maxlen)
        write_json(self.path / "metadata.json", self.metadata)
        print(f"Evaluation output: {self.path.resolve()}", flush=True)

    def __enter__(self):
        if self.metadata["telemetry_enabled"]:
            self.telemetry = (self.path / "telemetry.jsonl").open("w", buffering=1)
        return self

    def step(self, context, step, action, reward, terminated, truncated, info, *, image=None):
        elapsed = time.monotonic() - self.started
        if self.capture.maxlen:
            if image is None:
                raise ValueError("Capture requires the exact returned camera observation")
            if context != self.capture_context or step == 1:
                self.capture.clear()
                self.capture_context = dict(context)
            self.capture.append((step, elapsed, image.copy()))
        if self.telemetry is not None:
            keys = ("cte", "abs_cte", "speed", "rl_steer", "rl_steer_delta", "rl_throttle", "lap_count", "last_lap_time", "hit", "termination_reason",
                    "pos", "vel", "gyro", "accel", "car", "forward_vel",
                    "policy_seconds", "env_step_seconds", "encoder_seconds")
            row = dict(context, step=step, elapsed_s=elapsed,
                       action=action, reward=reward, terminated=terminated, truncated=truncated,
                       **{key: info.get(key) for key in keys})
            self.telemetry.write(json.dumps(json_value(row), allow_nan=False) + "\n")
        if step % 500 == 0:
            print(f"[progress] {context} step={step} laps={info.get('lap_count', 0)}", flush=True)

    def frame(self, context, image):
        from PIL import Image

        token = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()[:12]
        name = f"initial_{token}.png"
        Image.fromarray(image.astype("uint8")).save(self.path / name)
        return name

    def encoder(self, encoder):
        """Record the actual tensors, including cached pretrained weights."""
        digest = hashlib.sha256()
        for name, tensor in sorted(encoder.model.state_dict().items()):
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(f"{name}:{array.dtype}:{array.shape}".encode())
            digest.update(array.tobytes())
        self.metadata.setdefault("encoders", []).append({
            "family": getattr(encoder, "model_name", "vae"),
            "feature_dim": encoder.z_size, "state_sha256": digest.hexdigest(),
            "crop_top": getattr(encoder, "crop_top", 40),
        })
        write_json(self.path / "metadata.json", self.metadata)

    def episode(self, result):
        result = dict(result)
        if self.capture:
            result["capture"] = self._flush_capture()
        self.results.append(result)
        write_json(self.path / "results.json", {"episodes": self.results})
        with (self.path / "episodes.csv").open("w", newline="") as stream:
            keys = list(dict.fromkeys(key for row in self.results for key in row))
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows({k: json.dumps(json_value(v), allow_nan=False) if isinstance(v, (dict, list))
                              else json_value(v) for k, v in row.items()} for row in self.results)

    def _flush_capture(self):
        from PIL import Image

        token = hashlib.sha256(json.dumps(self.capture_context, sort_keys=True).encode()).hexdigest()[:12]
        folder = self.path / f"capture_{token}"
        folder.mkdir()
        frames, index = [], []
        for step, elapsed, array in self.capture:
            name = f"step_{step:06d}.png"
            frame = Image.fromarray(array.astype("uint8"))
            frame.save(folder / name)
            frames.append(frame)
            index.append({"step": step, "elapsed_s": elapsed, "file": name})
        durations = [max(1, round((b["elapsed_s"] - a["elapsed_s"]) * 1000)) for a, b in zip(index, index[1:])]
        durations.append(durations[-1] if durations else 50)
        frames[0].save(folder / "tail.gif", save_all=True, append_images=frames[1:], duration=durations, loop=0, disposal=2)
        write_json(folder / "index.json", {
            "context": self.capture_context,
            "frame_semantics": "Frame k is returned by env.step(command k) and used for the next decision. The asynchronous simulator does not acknowledge command execution or provide frame timestamps. Times are client wall-clock.",
            "frames": index,
        })
        self.capture.clear()
        return {"directory": folder.name, "animation": f"{folder.name}/tail.gif",
                "first_step": index[0]["step"], "last_step": index[-1]["step"], "frames": len(index)}

    def finish(self, summary):
        write_json(self.path / "results.json", {"summary": summary, "episodes": self.results})

    def __exit__(self, exc_type, exc, tb):
        if self.telemetry is not None:
            self.telemetry.close()
        if exc and self.capture:
            self.metadata["partial_capture"] = self._flush_capture()
        self.metadata.update(status="failed" if exc else "completed", error=repr(exc) if exc else None,
                             finished_at=datetime.now(timezone.utc).isoformat(), elapsed_s=time.monotonic() - self.started)
        write_json(self.path / "metadata.json", self.metadata)
