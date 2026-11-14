"""Could a vision policy imitate the ground-truth policy? Offline feasibility check.

For every captured frame of earlier evaluation runs, the ground-truth policy
(`--encoder sim_state`, the "teacher") is asked what it would do in that state,
using the telemetry and the command history of the episode the frame comes from.
A small network is then trained to predict that action from what a vision policy
would see, and scored on layouts it never saw.

Inputs compared (each with the 20-step command history, like the policies):
    commands only   reference: how much the history alone gives away
    state           the teacher's own 18 numbers: ceiling of this procedure
    cls             DINOv2 class token (the v2 policy input)
    grid            patch tokens pooled 4x4, PCA to 128
    grid + lags     the same plus the frames 2 and 4 steps earlier
    grid full       patch tokens pooled 4x4, all 6144 numbers
    ... + motion    with forward speed, sideways speed and yaw rate from telemetry
                    (what a wheel encoder and a gyro would give a real car)

It also reports how well each of the teacher's 18 state numbers can be read from
the image features (ridge regression), to show which information is missing.

The frames are the last steps of episodes driven by a vision policy, so they
over-represent the approach to a failure and its slower driving.

Usage:
    python tools/probe_teacher_action.py logs/<run> [logs/<run> ...] \
        --teacher models/rl_loop_sim_state_ts2_v1/sac_sim_state_50000_steps.zip \
        --track-centerline models/track_centerline/generated_track.npz \
        --encoder-projection models/dinov2_grid_projection/pca128.pt \
        --output-dir logs/probe_teacher_action_<name>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from stable_baselines3 import SAC

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from probe_state_from_features import extract_features, held_out_predictions, load_frames  # noqa: E402
from rl.track_state import LOOKAHEAD, TrackState  # noqa: E402

LAGS = (2, 4)
HISTORY = 20
STATE_NAMES = (["cte", "sin heading error", "cos heading error", "forward speed", "sideways speed", "yaw rate"]
               + [f"road {d:g} ahead: {axis}" for d in LOOKAHEAD for axis in ("right", "forward")])


def teacher_inputs(runs: list[Path], records: list[dict], track: TrackState) -> tuple[np.ndarray, np.ndarray]:
    """(state, command history) the teacher would observe at each frame, as the environment builds them."""
    traces = defaultdict(dict)
    for run in runs:
        with (run / "telemetry.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                traces[run.name, row.get("layout"), row.get("model")][row["step"]] = row
    states, histories = [], []
    for record in records:
        trace = traces[record["run"], record["layout"], record["model"]]
        states.append(track.encode(None, trace[record["step"]]))
        history = np.zeros((HISTORY, 2), np.float32)  # zeros before the episode started, oldest first
        for slot, step in enumerate(range(record["step"] - HISTORY + 1, record["step"] + 1)):
            if step in trace:
                history[slot] = (trace[step]["rl_steer"], trace[step]["rl_throttle"])
        histories.append(history.reshape(-1))
    return np.stack(states), np.stack(histories)


def lagged(features: np.ndarray, records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Features with those of the frames LAGS steps earlier appended; mask of frames that have them."""
    where = {(r["run"], r["layout"], r["model"], r["step"]): i for i, r in enumerate(records)}
    earlier = [[where.get((r["run"], r["layout"], r["model"], r["step"] - lag)) for lag in LAGS] for r in records]
    valid = np.array([all(index is not None for index in row) for row in earlier])
    out = np.zeros((len(records), features.shape[1] * (1 + len(LAGS))), np.float32)
    for i in np.flatnonzero(valid):
        out[i] = np.concatenate([features[i]] + [features[j] for j in earlier[i]])
    return out, valid


def network_predictions(x: np.ndarray, y: np.ndarray, groups: np.ndarray, device: torch.device) -> np.ndarray:
    """Predictions of a 2x256 network for every sample, from a network that never saw its layout."""
    prediction = np.zeros_like(y)
    for fold, (train, test) in enumerate(GroupKFold(5).split(x, y, groups)):
        torch.manual_seed(fold)
        mean, std = x[train].mean(0), x[train].std(0) + 1e-6
        inputs = torch.tensor((x - mean) / std, dtype=torch.float32, device=device)
        targets = torch.tensor(y, dtype=torch.float32, device=device)
        net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, 256),
                                  torch.nn.ReLU(), torch.nn.Linear(256, y.shape[1])).to(device)
        optimiser = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        index = torch.tensor(train, device=device)
        for _ in range(60):
            for batch in index[torch.randperm(len(index), device=device)].split(256):
                optimiser.zero_grad()
                torch.nn.functional.mse_loss(net(inputs[batch]), targets[batch]).backward()
                optimiser.step()
        with torch.no_grad():
            prediction[test] = net(inputs[torch.tensor(test, device=device)]).cpu().numpy()
    return prediction


def r2(truth: np.ndarray, predicted: np.ndarray) -> float:
    return float(1 - np.sum((predicted - truth) ** 2) / np.sum((truth - truth.mean(0)) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--track-centerline", type=Path, required=True)
    parser.add_argument("--encoder-projection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    records = [record for run in args.runs for record in load_frames(run)]
    state, history = teacher_inputs(args.runs, records, TrackState.load(args.track_centerline))
    teacher = SAC.load(str(args.teacher), device="cpu")
    action, _ = teacher.predict(np.concatenate([state, history], axis=1), deterministic=True)  # steer, throttle in [-1, 1]

    features = extract_features(records, device, crop_top=0)
    saved = torch.load(args.encoder_projection, map_location="cpu", weights_only=True)
    grid = ((features["patch_grid"] - saved["mean"].numpy()) @ saved["components"].numpy().T / saved["scale"].numpy()).astype(np.float32)
    grid_lags, valid = lagged(grid, records)
    inputs = {"commands only": history[:, :0], "state": state, "cls": features["cls"], "grid": grid,
              "grid + lags": grid_lags, "grid full": features["patch_grid"],
              "cls + motion": np.concatenate([features["cls"], state[:, 3:6]], axis=1),
              "grid + motion": np.concatenate([grid, state[:, 3:6]], axis=1),
              "grid+lags+motion": np.concatenate([grid_lags, state[:, 3:6]], axis=1)}

    # Same frames for every input: those whose earlier frames were captured too.
    groups = np.array([f"{r['run']}/{r['layout']}" for r in records])[valid]
    action, state, history = action[valid], state[valid], history[valid]
    far = np.abs(state[:, 0] * 2.0) >= 1.0
    result = {"runs": [str(run) for run in args.runs], "teacher": str(args.teacher), "frames": int(valid.sum()),
              "layouts": len(set(groups)), "teacher_action_std": action.std(0).tolist(), "action": {}, "state_from_features": {}}
    print(f"{valid.sum()} frames, {len(set(groups))} layouts; teacher action std: steer {action[:, 0].std():.3f}, throttle {action[:, 1].std():.3f}")
    print(f"{'input':<17} {'steer R2':>9} {'steer MAE':>10} {'throttle R2':>12} {'throttle MAE':>13} {'steer R2 when |cte|>=1':>23}")
    for name, x in inputs.items():
        predicted = network_predictions(np.concatenate([x[valid], history], axis=1), action, groups, device)
        row = {"steer_r2": r2(action[:, 0], predicted[:, 0]), "steer_mae": float(np.abs(predicted[:, 0] - action[:, 0]).mean()),
               "throttle_r2": r2(action[:, 1], predicted[:, 1]), "throttle_mae": float(np.abs(predicted[:, 1] - action[:, 1]).mean()),
               "steer_r2_abs_cte_ge_1": r2(action[far, 0], predicted[far, 0]), "frames_abs_cte_ge_1": int(far.sum())}
        result["action"][name] = row
        print(f"{name:<17} {row['steer_r2']:>9.3f} {row['steer_mae']:>10.3f} {row['throttle_r2']:>12.3f} "
              f"{row['throttle_mae']:>13.3f} {row['steer_r2_abs_cte_ge_1']:>23.3f}", flush=True)

    print("\nteacher state read from image features (ridge, unseen layouts), R2:")
    print(f"{'state number':<28} {'cls':>7} {'grid':>7} {'grid + lags':>12}")
    readable = {name: np.stack([held_out_predictions(inputs[name][valid], state[:, i], groups) for i in range(state.shape[1])], axis=1)
                for name in ("cls", "grid", "grid + lags")}
    for i, label in enumerate(STATE_NAMES):
        scores = {name: r2(state[:, i], predicted[:, i]) for name, predicted in readable.items()}
        result["state_from_features"][label] = scores
        print(f"{label:<28} {scores['cls']:>7.2f} {scores['grid']:>7.2f} {scores['grid + lags']:>12.2f}")

    (args.output_dir / "probe_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output_dir / 'probe_results.json'}")


if __name__ == "__main__":
    main()
