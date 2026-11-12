"""Can the car's position on the road be read from the frozen encoder features?

Pairs every captured frame of one or more evaluation runs (made with
--capture-tail-steps) with the simulator's ground truth from telemetry, then
fits ridge regressions from several feature sets to it. Every prediction is for
a layout the regression never saw, so the score measures what transfers to new
trees and lighting.

Feature sets (DINOv2, same preprocessing as the policy):
    cls         the class token: exactly what the SAC policy receives
    patch_mean  mean of the patch tokens
    patch_grid  patch tokens average-pooled to a 4x4 grid (keeps where things are)
    pixels      the frame shrunk to 20x15 RGB: a no-encoder reference

Targets:
    cte      signed cross-track error at the frame, in simulator units
    lateral  slope of cte against distance travelled over the last 5 steps:
             how fast the car is drifting across the road, a stand-in for the
             heading error (telemetry has no track direction). cte and position
             use different length units, so only the relative size is meaningful.

Frames are the last N of each episode, so they over-represent the approach to a
failure; that is the regime of interest, not a uniform sample of driving.

Usage:
    python tools/probe_state_from_features.py logs/<run> [logs/<run> ...] \
        --output-dir logs/probe_state_<name>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rl.train_vae_sac import make_encoder  # noqa: E402

ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
LAST_FRAMES = 20  # "just before leaving the track"
DRIFT_WINDOW = 5
CTE_BINS = (0.0, 0.5, 1.0, 1.5, np.inf)


def load_frames(run: Path) -> list[dict]:
    """One record per captured frame: image path, ground truth and episode outcome."""
    episodes = json.loads((run / "results.json").read_text())["episodes"]
    traces = defaultdict(dict)
    with (run / "telemetry.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            traces[row.get("layout"), row.get("model"), row.get("episode")][row["step"]] = row
    records = []
    for episode in episodes:
        if not episode.get("capture"):
            continue
        trace = traces[episode.get("layout"), episode.get("model"), episode.get("episode")]
        folder = run / episode["capture"]["directory"]
        failed = not episode.get("trunc", episode.get("outcome", "").startswith("TRUNC"))
        for frame in json.loads((folder / "index.json").read_text())["frames"]:
            step = frame["step"]  # frame k and telemetry row k both come from env.step k
            row = trace[step]
            lateral = float("nan")
            if step - DRIFT_WINDOW in trace:
                window = [trace[s] for s in range(step - DRIFT_WINDOW, step + 1)]
                path = np.array([w["pos"] for w in window], float)[:, [0, 2]]
                distance = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
                if distance[-1] > 0.02 * DRIFT_WINDOW:
                    lateral = float(np.polyfit(distance, [w["cte"] for w in window], 1)[0])
            records.append({
                "run": run.name, "layout": episode.get("layout"), "model": episode.get("model"),
                "step": step, "failed": failed, "steps_to_end": episode["steps"] - step,
                "image": folder / frame["file"], "cte": float(row["cte"]), "lateral": lateral,
            })
    return records


@torch.no_grad()
def extract_features(records: list[dict], device: torch.device, crop_top: int, batch: int = 64) -> dict[str, np.ndarray]:
    encoder = make_encoder("dinov2_vits14", device=device, crop_top=crop_top)
    out = defaultdict(list)
    for start in range(0, len(records), batch):
        images = np.stack([np.asarray(Image.open(r["image"]).convert("RGB")) for r in records[start:start + batch]])
        if start == 0:
            policy_input = encoder.encode(images[0])
        tensor = torch.from_numpy(images[:, crop_top:]).to(device).permute(0, 3, 1, 2).float() / 255.0
        out["pixels"].append(torch.nn.functional.adaptive_avg_pool2d(tensor, (15, 20)).flatten(1).cpu().numpy())
        tensor = torch.nn.functional.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
        tokens = encoder.model.forward_features((tensor - encoder._mean) / encoder._std)
        patches = tokens["x_norm_patchtokens"]  # (batch, 16*16, dim), row-major
        side = int(patches.shape[1] ** 0.5)
        grid = patches.transpose(1, 2).reshape(len(images), -1, side, side)
        out["cls"].append(tokens["x_norm_clstoken"].cpu().numpy())
        out["patch_mean"].append(patches.mean(1).cpu().numpy())
        out["patch_grid"].append(torch.nn.functional.adaptive_avg_pool2d(grid, (4, 4)).flatten(1).cpu().numpy())
        print(f"encoded {min(start + batch, len(records))}/{len(records)} frames", flush=True)
    features = {name: np.concatenate(parts).astype(np.float32) for name, parts in out.items()}
    if not np.allclose(features["cls"][0], policy_input, atol=1e-3):
        raise RuntimeError("Probe features differ from what the policy encoder produces")
    return features


def held_out_predictions(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Ridge predictions for each sample from a model that never saw its layout.

    The penalty is chosen inside each training split, again holding out layouts.
    """
    prediction = np.full(len(y), np.nan)
    for train, test in GroupKFold(min(5, len(set(groups)))).split(x, y, groups):
        scaler = StandardScaler().fit(x[train])
        x_train, x_test = scaler.transform(x[train]), scaler.transform(x[test])
        errors = np.zeros(len(ALPHAS))
        for fit, validate in GroupKFold(min(3, len(set(groups[train])))).split(x_train, y[train], groups[train]):
            for i, alpha in enumerate(ALPHAS):
                model = Ridge(alpha=alpha).fit(x_train[fit], y[train][fit])
                errors[i] += np.sum((model.predict(x_train[validate]) - y[train][validate]) ** 2)
        prediction[test] = Ridge(alpha=ALPHAS[int(np.argmin(errors))]).fit(x_train, y[train]).predict(x_test)
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop-top", type=int, default=0, help="Encoder crop of the policy being diagnosed.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = [record for run in args.runs for record in load_frames(run)]
    groups = np.array([f"{r['run']}/{r['layout']}" for r in records])
    if len(set(groups)) < 3:
        raise SystemExit("Need captured frames from at least 3 layouts")
    features = extract_features(records, torch.device(args.device), args.crop_top)
    failed = np.array([r["failed"] for r in records])
    near_end = failed & (np.array([r["steps_to_end"] for r in records]) < LAST_FRAMES)
    segments = {"completed episodes": ~failed, "failed, earlier in the tail": failed & ~near_end,
                f"failed, last {LAST_FRAMES} frames": near_end}

    result = {"runs": [str(run) for run in args.runs], "frames": len(records), "layouts": len(set(groups)),
              "failed_episodes": len({(r["run"], r["layout"], r["model"]) for r in records if r["failed"]}),
              "targets": {}}
    predictions = {}
    for target in ("cte", "lateral"):
        y = np.array([r[target] for r in records])
        valid = np.isfinite(y)
        report = {"samples": int(valid.sum()), "std": float(y[valid].std()), "features": {}}
        for name, x in features.items():
            predicted = np.full(len(y), np.nan)
            predicted[valid] = held_out_predictions(x[valid], y[valid], groups[valid])
            predictions[target, name] = predicted
            error = predicted - y
            row = {"r2": float(1 - np.nansum(error ** 2) / np.sum((y[valid] - y[valid].mean()) ** 2)),
                   "mae": float(np.nanmean(np.abs(error))),
                   "mae_by_segment": {label: float(np.nanmean(np.abs(error[mask]))) for label, mask in segments.items()
                                      if (mask & valid).any()}}
            if target == "cte":
                # Same distance from the centre line, completed vs failed episodes:
                # separates "hard to read here" from "hard to read far from centre".
                row["mae_by_abs_cte"] = {
                    f"{low:g}-{high:g}": {label: {"mae": float(np.nanmean(np.abs(error[mask]))), "frames": int(mask.sum())}
                                         for label, mask in (("completed", valid & ~failed & (np.abs(y) >= low) & (np.abs(y) < high)),
                                                             ("failed", valid & failed & (np.abs(y) >= low) & (np.abs(y) < high)))
                                         if mask.any()}
                    for low, high in zip(CTE_BINS, CTE_BINS[1:])}
                far = valid & (np.abs(y) >= 1.0)
                row["side_correct_when_abs_cte_ge_1"] = float(np.mean(np.sign(predicted[far]) == np.sign(y[far])))
                row["samples_abs_cte_ge_1"] = int(far.sum())
            report["features"][name] = row
            print(f"{target:>8} from {name:<10} R2={row['r2']:+.3f} MAE={row['mae']:.3f}  "
                  + "  ".join(f"[{label}: {value:.3f}]" for label, value in row["mae_by_segment"].items()), flush=True)
        result["targets"][target] = report

    (args.output_dir / "probe_results.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output_dir / "predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        columns = sorted(predictions)
        writer.writerow(["run", "layout", "model", "step", "failed", "steps_to_end", "cte", "lateral"]
                        + [f"{target}_from_{name}" for target, name in columns])
        for i, r in enumerate(records):
            writer.writerow([r["run"], r["layout"], r["model"], r["step"], int(r["failed"]), r["steps_to_end"],
                             r["cte"], r["lateral"]] + [predictions[column][i] for column in columns])
    print(f"wrote {args.output_dir / 'probe_results.json'} and predictions.csv")


if __name__ == "__main__":
    main()
