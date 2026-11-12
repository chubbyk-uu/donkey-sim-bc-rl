"""Replay captured tail frames through the encoder and policy, offline.

For every failed episode of a completed capture run (an evaluation made with
--capture-tail-steps and telemetry), rebuild the observation from the saved
frame plus the preceding mapped commands and check that the policy reproduces
the recorded next action. This validates camera/history/policy alignment for
the sampled steps; it says nothing about simulator command-to-frame causality.

With --brightness-gains it also re-encodes each frame with scaled RGB (command
history frozen) and reports how far the steering output moves. That is an
offline sensitivity probe, not a closed-loop result and not a test of local
tree-shadow invariance.

Usage (from anywhere):
    python tools/verify_policy_alignment.py logs/<run_directory> --brightness-gains 0.9 1.1
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from stable_baselines3 import SAC

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rl.train_vae_sac import make_encoder  # noqa: E402

TOLERANCE = 1e-5


def recorded_path(value) -> Path:
    """Runs record paths as given on the command line, usually repo-relative."""
    path = Path(value)
    return path if path.is_absolute() or path.exists() else REPO / path


def key(row):
    return row.get("layout"), row.get("model"), row.get("episode")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--brightness-gains", type=float, nargs="*", default=[],
                        help="RGB gains for the sensitivity probe, e.g. 0.9 1.1. Empty skips it.")
    parser.add_argument("--output-dir", type=Path, help="Where to write the JSON results; defaults to the run directory.")
    args = parser.parse_args()
    run = args.run
    output = args.output_dir or run
    output.mkdir(parents=True, exist_ok=True)

    meta = json.loads((run / "metadata.json").read_text())
    if meta["status"] != "completed":
        raise ValueError("Verify a completed run")
    run_args = meta["args"]
    episodes = json.loads((run / "results.json").read_text())["episodes"]
    traces = defaultdict(list)
    for line in (run / "telemetry.jsonl").read_text().splitlines():
        row = json.loads(line)
        traces[key(row)].append(row)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Paired runs record one policy file per label and a crop per --models
    # entry; single-model runs record one file named "policy".
    crops = {}
    for item in (run_args.get("models") or "").split(","):
        if item.strip():
            label, _, crop = item.rsplit(":", 2)
            crops[label.strip()] = int(crop)
    encoders, models, checks, probes = {}, {}, [], []
    history_length = run_args["n_command_history"]

    for episode in episodes:
        failed = not episode.get("trunc", episode.get("outcome", "").startswith("TRUNC"))
        if not failed or not episode.get("capture"):
            continue
        label = episode.get("model", "policy")
        crop = crops.get(label, run_args["encoder_crop_top"])
        if crop not in encoders:
            vae = run_args.get("vae_model")
            encoders[crop] = make_encoder(run_args["encoder"], device, crop_top=crop,
                                          vae_checkpoint=recorded_path(vae) if vae else None)
        if label not in models:
            models[label] = SAC.load(recorded_path(meta["files"][label]["path"]), device=device)
        trace = traces[key(episode)]
        folder = run / episode["capture"]["directory"]
        frames = json.loads((folder / "index.json").read_text())["frames"]
        for index in sorted({0, len(frames) // 2, len(frames) - 10, len(frames) - 5, len(frames) - 2}):
            if not 0 <= index < len(frames):
                continue
            step = frames[index]["step"]  # frame k is the input to action k+1 = trace[k]
            if step >= len(trace):
                continue
            with Image.open(folder / frames[index]["file"]) as image:
                original = np.asarray(image.convert("RGB"))
            history = np.zeros((history_length, 2), np.float32)
            previous = trace[max(0, step - history_length):step] if history_length else []
            if previous:
                history[-len(previous):] = [[r["rl_steer"], r["rl_throttle"]] for r in previous]

            def act(frame, history=history, crop=crop, label=label):
                obs = np.concatenate([encoders[crop].encode(frame), history.reshape(-1)], dtype=np.float32)
                return models[label].predict(obs, deterministic=True)[0]

            predicted = act(original)
            error = float(np.max(np.abs(predicted - np.array(trace[step]["action"], np.float32))))
            context = {"layout": episode.get("layout"), "model": episode.get("model"),
                       "episode": episode.get("episode"), "frame_step": step}
            checks.append(dict(context, action_step=step + 1, max_abs_error=error))
            print(checks[-1], flush=True)
            for gain in args.brightness_gains:
                changed = act(np.clip(original.astype(np.float32) * gain, 0, 255).astype(np.uint8))
                probes.append(dict(context, gain=gain, original_steer=float(predicted[0]),
                                   changed_steer=float(changed[0]),
                                   steer_delta=abs(float(changed[0] - predicted[0])),
                                   sign_flip=bool(abs(predicted[0]) > .2 and predicted[0] * changed[0] < 0)))

    if not checks:
        raise ValueError("No failed episode with a capture to verify")
    result = {"checks": checks, "max_abs_error": max(r["max_abs_error"] for r in checks),
              "tolerance": TOLERANCE, "passed": all(r["max_abs_error"] <= TOLERANCE for r in checks)}
    (output / "alignment_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print({k: v for k, v in result.items() if k != "checks"}, flush=True)
    if probes:
        probe = {"purpose": "Offline visual sensitivity probe, not a closed-loop performance test",
                 "gains": args.brightness_gains, "histories_frozen": True, "samples": probes,
                 "median_abs_steer_delta": float(np.median([r["steer_delta"] for r in probes])),
                 "max_abs_steer_delta": max(r["steer_delta"] for r in probes),
                 "sign_flips": sum(r["sign_flip"] for r in probes)}
        (output / "brightness_probe.json").write_text(json.dumps(probe, indent=2) + "\n")
        print({k: v for k, v in probe.items() if k != "samples"}, flush=True)
    if not result["passed"]:
        raise SystemExit("Alignment check failed")


if __name__ == "__main__":
    main()
