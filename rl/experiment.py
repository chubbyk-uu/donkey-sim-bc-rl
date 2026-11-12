"""Configuration contracts and durable evaluation records (no simulator needed)."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
import re
import shutil
import struct
import subprocess
import tempfile
import time
import warnings
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


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


def simulator_saved_settings() -> dict | None:
    """Simulator menu options that change results but are invisible to the client.

    Read from the Windows registry where the simulator saves them (time scale,
    FPS limit, random light / trees / cones). These are the saved values, not
    the state of the running process. None when they cannot be read, e.g.
    outside WSL.
    """
    reg = Path("/mnt/c/Windows/System32/reg.exe")
    if not reg.exists():
        return None
    try:
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "simulator.reg"
            windows_path = subprocess.check_output(["wslpath", "-w", str(target)], text=True).strip()
            subprocess.run([str(reg), "export", r"HKCU\Software\DonkeyCar\donkey_sim", windows_path, "/y"],
                           cwd="/mnt/c", check=True, capture_output=True, timeout=20)
            text = target.read_text(encoding="utf-16")
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None
    names = "timeScale|fps|randomLight|generateTrees|generateRandomCones|useSeed"
    settings = {}
    for name, kind, raw in re.findall(rf'"({names})_h\d+"=(dword|hex\(4\)):([0-9a-f,]+)', text):
        # Unity stores ints as dwords and floats as 8-byte doubles.
        settings[name] = int(raw, 16) if kind == "dword" else struct.unpack("<d", bytes.fromhex(raw.replace(",", "")))[0]
    return settings or None


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a rate estimated from n independent trials."""
    if n == 0:
        return 0.0, 1.0
    rate = successes / n
    half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n))
    centre, scale = rate + z * z / (2 * n), 1 + z * z / n
    return max(0.0, (centre - half) / scale), min(1.0, (centre + half) / scale)


def bootstrap_mean_interval(values, resamples: int = 10000) -> tuple[float, float]:
    """95% percentile bootstrap interval for a mean; fixed seed so reruns agree."""
    values = np.asarray(values, dtype=float)
    draws = np.random.default_rng(0).integers(len(values), size=(resamples, len(values)))
    low, high = np.percentile(values[draws].mean(axis=1), [2.5, 97.5])
    return float(low), float(high)


def paired_statistics(episodes: list[dict]) -> dict:
    """How far a paired evaluation can be trusted.

    The primary metric is laps per episode (step cap fixed, leaving the track
    ends the episode), which rewards both speed and staying on the road. Per
    model: mean laps, completion rate and mean episode length with 95% intervals
    over layouts. Per pair of models: the mean per-layout difference in laps and
    in episode length, and an exact sign test on the layouts where exactly one
    of them completed.
    """
    runs: dict[str, dict] = {}
    for episode in episodes:
        runs.setdefault(episode["model"], {})[episode["layout"]] = episode
    models = {}
    for label, by_layout in runs.items():
        steps = [e["steps"] for e in by_layout.values()]
        laps = [e["laps"] for e in by_layout.values()]
        done = sum(bool(e["trunc"]) for e in by_layout.values())
        models[label] = {
            "layouts": len(steps),
            "mean_laps": float(np.mean(laps)), "mean_laps_ci95": bootstrap_mean_interval(laps),
            "completions": done,
            "completion_rate": done / len(steps),
            "completion_rate_ci95": wilson_interval(done, len(steps)),
            "mean_steps": float(np.mean(steps)), "mean_steps_ci95": bootstrap_mean_interval(steps),
        }
    pairs = []
    for a, b in itertools.combinations(runs, 2):
        shared = sorted(runs[a].keys() & runs[b].keys())
        if not shared:
            continue
        only_a = sum(bool(runs[a][k]["trunc"]) and not runs[b][k]["trunc"] for k in shared)
        only_b = sum(bool(runs[b][k]["trunc"]) and not runs[a][k]["trunc"] for k in shared)
        differing = only_a + only_b
        tail = sum(math.comb(differing, i) for i in range(min(only_a, only_b) + 1))
        difference = [runs[a][k]["steps"] - runs[b][k]["steps"] for k in shared]
        laps = [runs[a][k]["laps"] - runs[b][k]["laps"] for k in shared]
        pairs.append({
            "a": a, "b": b, "layouts": len(shared),
            "mean_laps_difference": float(np.mean(laps)),
            "mean_laps_difference_ci95": bootstrap_mean_interval(laps),
            "only_a_completed": only_a, "only_b_completed": only_b,
            "sign_test_p": min(1.0, 2 * tail / 2 ** differing),
            "mean_steps_difference": float(np.mean(difference)),
            "mean_steps_difference_ci95": bootstrap_mean_interval(difference),
        })
    return {"models": models, "pairs": pairs}


def format_paired_statistics(statistics: dict) -> str:
    lines = [f"{'model':>12} | {'laps (95% CI)':>19} | {'completed':>9} | {'rate (95% CI)':>19} | mean steps (95% CI)"]
    for label, row in statistics["models"].items():
        low, high = row["completion_rate_ci95"]
        steps_low, steps_high = row["mean_steps_ci95"]
        laps_low, laps_high = row["mean_laps_ci95"]
        lines.append(f"{label:>12} | {row['mean_laps']:>5.2f} ({laps_low:>4.2f} - {laps_high:>4.2f}) | "
                     f"{row['completions']:>4}/{row['layouts']:<4} | "
                     f"{row['completion_rate']:>4.0%} ({low:>4.0%} - {high:>4.0%}) | "
                     f"{row['mean_steps']:.0f} ({steps_low:.0f} - {steps_high:.0f})")
    for pair in statistics["pairs"]:
        low, high = pair["mean_steps_difference_ci95"]
        laps_low, laps_high = pair["mean_laps_difference_ci95"]
        lines.append(f"{pair['a']} vs {pair['b']} on {pair['layouts']} shared layouts: "
                     f"laps difference {pair['mean_laps_difference']:+.2f} ({laps_low:+.2f} to {laps_high:+.2f}); "
                     f"only {pair['a']} completed {pair['only_a_completed']}, "
                     f"only {pair['b']} completed {pair['only_b_completed']}, "
                     f"sign test p={pair['sign_test_p']:.3f}; "
                     f"steps difference {pair['mean_steps_difference']:+.0f} ({low:+.0f} to {high:+.0f})")
    return "\n".join(lines)


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
    if any(lag <= 0 for lag in getattr(args, "latent_lags", ())):
        raise ValueError("--latent-lags must be positive step counts")
    if getattr(args, "encoder", "vae").endswith("_grid") and getattr(args, "encoder_projection", None) is None:
        raise ValueError("--encoder-projection is required for a grid encoder")
    if getattr(args, "resume_replay_buffer", None) and not getattr(args, "resume_model", None):
        raise ValueError("--resume-replay-buffer requires --resume-model")
    if getattr(args, "gradient_steps_cap", 0) > 0:
        if not 1 <= args.gradient_steps_min <= args.gradient_steps_cap or args.gradient_steps_scale <= 0:
            raise ValueError("Require 1 <= gradient_steps_min <= cap and gradient_steps_scale > 0")
        if getattr(args, "train_freq_unit", None) == "episode" and args.gradient_steps != -1:
            raise ValueError("Use --gradient-steps-cap 0 for fixed --gradient-steps; dynamic updates would override it")


def parse_lags(text: str) -> tuple[int, ...]:
    """'2,4' -> (2, 4); '' -> ()."""
    return tuple(int(item) for item in text.split(",") if item.strip())


def environment_config(args) -> dict:
    encoder = getattr(args, "encoder", "vae")
    observation = {
        "encoder": encoder,
        "vae_sha256": sha256(args.vae_model) if encoder == "vae" else None,
        "crop_top": 40 if encoder == "vae" else args.encoder_crop_top,
        "n_command_history": getattr(args, "n_command_history", 20),
    }
    # Added only when used, so configurations saved before these existed still compare equal.
    if encoder.endswith("_grid"):
        observation["projection_sha256"] = sha256(args.encoder_projection)
    if getattr(args, "latent_lags", ()):
        observation["latent_lags"] = list(args.latent_lags)
    return {
        "env_id": args.env_id,
        "observation": observation,
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
            "simulator_saved_settings": simulator_saved_settings(),
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

    def finish(self, summary, **extra):
        write_json(self.path / "results.json", {"summary": summary, **extra, "episodes": self.results})

    def __exit__(self, exc_type, exc, tb):
        if self.telemetry is not None:
            self.telemetry.close()
        if exc and self.capture:
            self.metadata["partial_capture"] = self._flush_capture()
        self.metadata.update(status="failed" if exc else "completed", error=repr(exc) if exc else None,
                             finished_at=datetime.now(timezone.utc).isoformat(), elapsed_s=time.monotonic() - self.started)
        write_json(self.path / "metadata.json", self.metadata)
