"""Retrain the unchanged two-stage linear geometry head on train-pool captures.

Round A locked holdout files are never opened. The last three training-pool
layouts of each capture run form internal validation; the legacy feature cache
is training data only. All head fitting sees training layouts only.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from probe_geometry_15d_lags import build_lag_indices, fit_and_export_proj15_lags_ridge  # noqa: E402
from rl.geometry_encoder import (  # noqa: E402
    GEOM_IDX, SPLIT_DAGGER_TRAIN_POOL, assert_not_locked_holdout,
    normalize_sincos_np, sha256_file, verify_cache_fingerprint,
)
from rl.train_vae_sac import FrozenDinoGridEncoder  # noqa: E402


def capture_plan(runs: list[Path], validation_layouts: int = 3) -> list[dict]:
    plan = []
    for run in runs:
        manifest = json.loads((run / "split_manifest.json").read_text())
        files = manifest["files"][SPLIT_DAGGER_TRAIN_POOL]
        entries = []
        for name in files:
            path = run / name
            assert_not_locked_holdout(path)
            with np.load(path, allow_pickle=False) as cap:
                if str(cap["split"]) != SPLIT_DAGGER_TRAIN_POOL:
                    raise ValueError(f"Capture is not in training pool: {path}")
                layout = int(cap["layout"])
                condition = str(cap["condition"])
                frames = len(cap["step"]) + 1  # include reset frame for exact lag padding
            entries.append({"path": str(path), "sha256": sha256_file(path),
                            "run": run.name, "layout": str(layout), "model": condition,
                            "frames": frames, "group": f"{run.name}/{layout}"})
        layouts = sorted({int(e["layout"]) for e in entries})
        if validation_layouts < 1 or len(layouts) <= validation_layouts:
            raise ValueError("Need both training and validation layouts in each capture run")
        held = set(layouts[-validation_layouts:])
        for entry in entries:
            entry["internal_validation"] = int(entry["layout"]) in held
        plan.extend(entries)
    return plan


@torch.no_grad()
def predict_head(head: Path, grid: np.ndarray, lag_indices: np.ndarray, device: torch.device) -> np.ndarray:
    saved = torch.load(head, map_location=device, weights_only=True)
    if saved["format"] != "proj15_lags_ridge_v1" or list(saved["lags"]) != [2, 4]:
        raise ValueError("Expected the existing linear lag-head architecture")
    stage1 = np.empty((len(grid), 15), dtype=np.float32)
    for start in range(0, len(grid), 4096):
        x = torch.from_numpy(grid[start:start + 4096]).to(device=device, dtype=torch.float32)
        stage1[start:start + len(x)] = (x @ saved["w1"] + saved["b1"]).cpu().numpy()
    stacked = np.concatenate([stage1[lag_indices[:, k]] for k in range(3)], axis=1)
    pred = (torch.from_numpy(stacked).to(device) @ saved["w2"] + saved["b2"]).cpu().numpy()
    return normalize_sincos_np(pred)


def metrics(pred: np.ndarray, state: np.ndarray, mask: np.ndarray) -> dict:
    if not np.any(mask):
        return {"frames": 0}
    p, truth = pred[mask], state[mask][:, GEOM_IDX]
    cte = (p[:, 0] - truth[:, 0]) * 2
    heading = np.degrees(np.arctan2(
        p[:, 1] * truth[:, 2] - p[:, 2] * truth[:, 1],
        p[:, 1] * truth[:, 1] + p[:, 2] * truth[:, 2],
    ))
    la1 = np.linalg.norm(p[:, 7:9] - truth[:, 7:9], axis=1)
    return {"frames": int(mask.sum()), "cte_mae": float(np.mean(abs(cte))),
            "cte_p95": float(np.percentile(abs(cte), 95)), "cte_p99": float(np.percentile(abs(cte), 99)),
            "cte_signed_bias": float(np.mean(cte)),
            "heading_mae_deg": float(np.mean(abs(heading))), "heading_p95_deg": float(np.percentile(abs(heading), 95)),
            "lookahead_1_pos_p95": float(np.percentile(la1, 95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-runs", type=Path, nargs="+", default=[Path("logs/round_a_vision_24layouts_v1"), Path("logs/round_b_recovery_collect_v1")])
    parser.add_argument("--legacy-cache", type=Path, default=Path("logs/probe_geometry_15d_v1/feature_cache.npz"))
    parser.add_argument("--baseline-head", type=Path, default=Path("models/dinov2_geometry_15d/proj15_lags_ridge.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/round_b_geometry_train_v1"))
    parser.add_argument("--export-head", type=Path, default=Path("models/dinov2_geometry_15d_round_b_v1/proj15_lags_ridge.pt"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-layouts", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.export_head.resolve() == args.baseline_head.resolve() or args.export_head.exists():
        parser.error("Export must be a new path, preserving the frozen baseline")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_num_threads(4)
    legacy_meta = verify_cache_fingerprint(args.legacy_cache)
    plan = capture_plan(args.capture_runs, args.validation_layouts)
    spec = {"captures": plan, "legacy_cache": str(args.legacy_cache),
            "legacy_sha256": sha256_file(args.legacy_cache), "encoder": "dinov2_vits14_grid",
            "crop_top": 0, "grid_side": 4, "lags": [2, 4],
            "baseline_head_sha256": sha256_file(args.baseline_head)}
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    spec["fingerprint_sha256"] = fingerprint
    spec_path = args.output_dir / "training_manifest.json"
    cache_path = args.output_dir / "feature_cache.npz"
    if spec_path.exists() and json.loads(spec_path.read_text())["fingerprint_sha256"] != fingerprint:
        raise RuntimeError("Training inputs changed; use a new output directory")
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")

    if not cache_path.exists():
        old = np.load(args.legacy_cache, allow_pickle=False)
        arrays = {k: [old[k]] for k in ("patch_grid", "state", "history", "run", "layout", "model", "step")}
        arrays["internal_validation"] = [np.zeros(len(old["state"]), dtype=bool)]
        encoder = FrozenDinoGridEncoder(device=device, crop_top=0, projection=None)
        for i, entry in enumerate(plan):
            with np.load(entry["path"], allow_pickle=False) as cap:
                images = np.concatenate([cap["init_image_0"][None], cap["images"]])
                feats = []
                for start in range(0, len(images), args.batch_size):
                    feats.append(encoder.grid_features(images[start:start + args.batch_size]).cpu().numpy())
                n = len(images)
                arrays["patch_grid"].append(np.concatenate(feats))
                arrays["state"].append(np.concatenate([cap["init_true_state_0"][None], cap["true_state"]]))
                arrays["history"].append(np.concatenate([np.zeros((1, 40), dtype=np.float32), cap["history_post"]]))
                for key in ("run", "layout", "model"):
                    arrays[key].append(np.full(n, entry[key]))
                arrays["step"].append(np.arange(n, dtype=np.int32))
                arrays["internal_validation"].append(np.full(n, entry["internal_validation"], dtype=bool))
            print(f"features {i + 1}/{len(plan)} {entry['group']} {entry['model']} {n} frames", flush=True)
        packed = {k: np.concatenate(v) for k, v in arrays.items()}
        np.savez(cache_path, **packed)
        del arrays, packed, encoder, old
        gc.collect()
        torch.cuda.empty_cache()

    cache = np.load(cache_path, allow_pickle=False)
    grid, state = cache["patch_grid"], cache["state"]
    run, layout, model, step = (cache[k] for k in ("run", "layout", "model", "step"))
    val = cache["internal_validation"]
    groups = np.array([f"{r}/{l}" for r, l in zip(run, layout)])
    assert not set(groups[val]) & set(groups[~val])
    train_lags, train_valid = build_lag_indices(run[~val], layout[~val], model[~val], step[~val])
    print(f"FIT {np.sum(~val)} frames/{len(set(groups[~val]))} layouts; VALIDATE {val.sum()} frames/{len(set(groups[val]))} layouts", flush=True)
    started = time.perf_counter()
    fit_and_export_proj15_lags_ridge(
        grid[~val], state[~val][:, GEOM_IDX], train_lags, train_valid, groups[~val],
        device, args.export_head,
        {"fingerprint_sha256": fingerprint, "crop_top": 0, "track_centerline_sha256": legacy_meta["track_centerline_sha256"]},
    )
    lag_indices, valid = build_lag_indices(run, layout, model, step)
    abs_cte = np.abs(state[:, 0] * 2)
    subsets = {"validation_all": val & valid,
               "validation_cruise": val & valid & (abs_cte < 0.5),
               "validation_large_offset": val & valid & (abs_cte >= 1),
               "validation_positive_large": val & valid & (state[:, 0] >= 0.5),
               "validation_negative_large": val & valid & (state[:, 0] <= -0.5),
               "validation_startup": val & valid & (step <= 250),
               "validation_round_a": val & valid & (run == "round_a_vision_24layouts_v1"),
               "validation_recovery": val & valid & (run == "round_b_recovery_collect_v1")}
    report = {"manifest_fingerprint": fingerprint, "fit_seconds": time.perf_counter() - started,
              "training_frames": int(np.sum(~val)), "validation_frames": int(np.sum(val)),
              "validation_groups": sorted(set(groups[val])), "round_a_locked_holdout_used": False,
              "models": {}}
    predictions = {}
    for label, path in (("baseline", args.baseline_head), ("candidate", args.export_head)):
        pred = predict_head(path, grid, lag_indices, device)
        predictions[label] = pred
        report["models"][label] = {"head": str(path), "sha256": sha256_file(path),
                                  "subsets": {name: metrics(pred, state, mask) for name, mask in subsets.items()}}
        print(label, json.dumps(report["models"][label]["subsets"], indent=2), flush=True)
    np.savez(args.output_dir / "predictions.npz", **predictions, valid=valid, internal_validation=val)
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
