"""One offline, fixed-history comparison of geometry heads through the frozen teacher."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from stable_baselines3 import SAC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl.geometry_encoder import GEOM_IDX, sha256_file  # noqa: E402
from tools.probe_geometry_15d_lags import build_lag_indices  # noqa: E402
from tools.train_geometry_recovery import metrics as geometry_metrics, predict_head  # noqa: E402


def executed_actions(raw, history):
    """Match the frozen baseline mapper; history is POST-command at the input frame."""
    previous = history[:, -2]
    limit = (0.1 - 1e-5) * 2.0
    steer = previous + np.clip(np.clip(raw[:, 0], -1, 1) - previous, -limit, limit)
    throttle = 0.2 + 0.5 * np.clip((raw[:, 1] + 1) / 2, 0, 1)
    return np.column_stack([steer, throttle]).astype(np.float32)


def sequence_segments(run, layout, condition, step):
    """Do not bridge conditions, layouts, missing steps, or excluded lag padding."""
    keys = np.array([f"{r}/{l}/{c}" for r, l, c in zip(run, layout, condition)])
    segments = []
    for key in dict.fromkeys(keys):
        indices = np.flatnonzero(keys == key)
        indices = indices[np.argsort(step[indices], kind="stable")]
        for part in np.split(indices, np.flatnonzero(np.diff(step[indices]) != 1) + 1):
            if len(part):
                segments.append(part)
    return segments


def action_metrics(raw, mapped, true_raw, true_mapped, history, mask, state):
    n = int(mask.sum())
    if not n:
        return {"frames": 0}
    error = mapped[:, 0] - true_mapped[:, 0]
    previous = history[:, -2]
    target_delta = true_mapped[:, 0] - previous
    delta = mapped[:, 0] - previous
    heading = np.degrees(np.arctan2(state[:, 1], state[:, 2]))
    offset = (np.abs(state[:, 0] * 2) >= 0.5) | (np.abs(heading) >= 8)
    unwind = mask & offset & (np.abs(previous) >= 0.3) & (np.abs(true_mapped[:, 0]) < np.abs(previous) - 0.02)
    missed = unwind & (np.abs(mapped[:, 0]) >= np.abs(previous) - 0.02)
    direction_eligible = mask & offset & (np.abs(target_delta) > 0.02)
    wrong = direction_eligible & (delta * target_delta < 0) & (np.abs(delta) > 0.02)
    return {
        "frames": n, "steer_mae": float(np.mean(np.abs(error[mask]))),
        "steer_p95": float(np.percentile(np.abs(error[mask]), 95)),
        "steer_signed_bias": float(np.mean(error[mask])),
        "raw_steer_mae": float(np.mean(np.abs(raw[mask, 0] - true_raw[mask, 0]))),
        "raw_steer_p95": float(np.percentile(np.abs(raw[mask, 0] - true_raw[mask, 0]), 95)),
        "throttle_mae": float(np.mean(np.abs(mapped[mask, 1] - true_mapped[mask, 1]))),
        "teacher_unwind_frames": int(unwind.sum()), "missed_unwind_frames": int(missed.sum()),
        "missed_unwind_rate": float(missed.sum() / unwind.sum()) if unwind.any() else None,
        "teacher_direction_frames": int(direction_eligible.sum()), "wrong_direction_frames": int(wrong.sum()),
        "wrong_direction_rate": float(wrong.sum() / direction_eligible.sum()) if direction_eligible.any() else None,
    }


def persistent_bias(error, segments, window=20):
    windows = []
    correlations = []
    for indices in segments:
        x = error[indices]
        if len(x) >= window:
            means = np.convolve(x, np.ones(window) / window, mode="valid")
            for start in np.argsort(np.abs(means))[-3:]:
                windows.append((float(means[start]), indices[start:start + window]))
        if len(x) > 2 and x[:-1].std() > 1e-8 and x[1:].std() > 1e-8:
            correlations.append(float(np.corrcoef(x[:-1], x[1:])[0, 1]))
    windows.sort(key=lambda item: abs(item[0]), reverse=True)
    return {"window_steps": window, "max_abs_mean_error": abs(windows[0][0]) if windows else None,
            "sequence_mean_lag1_autocorrelation": float(np.mean(correlations)) if correlations else None}, windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, default=Path("logs/round_b_negative_geometry_train_v1"))
    parser.add_argument("--teacher", type=Path, default=Path("models/rl_loop_sim_state_ts2_sdiff01_v1/sac_sim_state_60000_steps.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/geometry_control_compare_v1"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(4)
    heads = {
        "baseline": Path("models/dinov2_geometry_15d/proj15_lags_ridge.pt"),
        "round_b": Path("models/dinov2_geometry_15d_round_b_v1/proj15_lags_ridge.pt"),
        "negative": Path("models/dinov2_geometry_15d_negative_v1/proj15_lags_ridge.pt"),
    }
    manifest = json.loads((args.training_dir / "training_manifest.json").read_text())
    canonical = {k: v for k, v in manifest.items() if k != "fingerprint_sha256"}
    import hashlib
    assert hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest() == manifest["fingerprint_sha256"]
    with np.load(args.training_dir / "feature_cache.npz", allow_pickle=False) as cache:
        state, history = cache["state"], cache["history"]
        run, layout, condition, step = (cache[k] for k in ("run", "layout", "model", "step"))
        validation = cache["internal_validation"]
        lag_indices, valid = build_lag_indices(run, layout, condition, step)
        keep = validation & valid
        grid = cache["patch_grid"]
        predictions = {label: predict_head(path, grid, lag_indices, torch.device(args.device))[keep]
                       for label, path in heads.items()}
        del grid
    groups_full = np.array([f"{r}/{l}" for r, l in zip(run, layout)])
    assert not set(groups_full[validation]) & set(groups_full[~validation])
    # Both retrained heads excluded every common validation layout from fitting.
    for training in [Path("logs/round_b_geometry_train_v1"), args.training_dir]:
        spec = json.loads((training / "training_manifest.json").read_text())
        fitted = {row["group"] for row in spec["captures"] if not row["internal_validation"]}
        assert not fitted & set(groups_full[keep])
    state, history = state[keep], history[keep]
    run, layout, condition, step, groups = (x[keep] for x in (run, layout, condition, step, groups_full))
    segments = sequence_segments(run, layout, condition, step)
    teacher = SAC.load(str(args.teacher), device="cpu")

    def query(s):
        return np.concatenate([teacher.predict(np.concatenate([s[i:i+2048], history[i:i+2048]], axis=1),
                                               deterministic=True)[0] for i in range(0, len(s), 2048)])

    true_raw = query(state)
    true_mapped = executed_actions(true_raw, history)
    # Independently audit frame/action alignment against saved live teacher labels.
    alignment = 0.0
    for entry in manifest["captures"]:
        if not entry["internal_validation"]:
            continue
        mask = (run == entry["run"]) & (layout == entry["layout"]) & (condition == entry["model"])
        with np.load(entry["path"], allow_pickle=False) as cap:
            indices = step[mask] - 1
            np.testing.assert_array_equal(state[mask], cap["true_state"][indices])
            np.testing.assert_array_equal(history[mask], cap["history_post"][indices])
            alignment = max(alignment, float(np.max(np.abs(true_raw[mask] - cap["teacher_true_raw_action_on_frame"][indices]))))
    assert alignment < 1e-4, alignment
    subsets = {
        "all": np.ones(len(state), bool), "startup": step <= 250,
        "cruise": np.abs(state[:, 0] * 2) < 0.5,
        "large_offset": np.abs(state[:, 0] * 2) >= 1,
        "negative_large": state[:, 0] * 2 <= -1, "positive_large": state[:, 0] * 2 >= 1,
    }
    report = {
        "teacher": {"path": str(args.teacher), "sha256": sha256_file(args.teacher)},
        "training_manifest_sha256": sha256_file(args.training_dir / "training_manifest.json"),
        "validation_frames": len(state), "validation_groups": sorted(set(groups)),
        "saved_teacher_action_max_difference": alignment,
        "semantics": {"history": "Post-command history at frame k, used to predict command k+1; identical for all heads.",
                      "mapping": "steering limit 1, difference .1 (executed delta .19998); throttle .2..7; unwind disabled",
                      "unwind": "Teacher reduces steering magnitude by >.02 from |previous steer|>=.3, with |CTE|>=.5 or |heading|>=8deg.",
                      "direction": "Steering-change direction relative to the same true-state teacher; deadband .02.",
                      "reference": "Frozen true-state teacher, not guaranteed optimal recovery labels.",
                      "limits": "Fixed recorded trajectories/histories; no recursive rollout, future stability, timing benefit or closed-loop completion claim."},
        "subsets": {k: {"frames": int(mask.sum()), "layout_groups": len(set(groups[mask])),
                         "sequences": len(set(zip(run[mask], layout[mask], condition[mask])))} for k, mask in subsets.items()},
        "models": {},
    }
    per_layout, window_rows, saved = [], [], {"state": state, "history": history, "run": run,
        "layout": layout, "condition": condition, "step": step, "true_raw": true_raw, "true_mapped": true_mapped}
    for label, geometry in predictions.items():
        s = state.copy()
        s[:, GEOM_IDX] = geometry
        raw = query(s)
        mapped = executed_actions(raw, history)
        error = mapped[:, 0] - true_mapped[:, 0]
        persistence, windows = persistent_bias(error, segments)
        info = {"head": str(heads[label]), "sha256": sha256_file(heads[label]),
                "action": {name: action_metrics(raw, mapped, true_raw, true_mapped, history, mask, state)
                           for name, mask in subsets.items()},
                "geometry": {name: geometry_metrics(geometry, state, mask) for name, mask in subsets.items()},
                "persistent_bias": persistence, "counterfactual_geometry_repairs": {}}
        for name, indices in {"cte": [0], "heading": [1, 2], "lookahead": list(range(6, 18))}.items():
            repaired = s.copy()
            repaired[:, indices] = state[:, indices]
            repaired_raw = query(repaired)
            repaired_mapped = executed_actions(repaired_raw, history)
            info["counterfactual_geometry_repairs"][name] = {
                subset: action_metrics(repaired_raw, repaired_mapped, true_raw, true_mapped, history, mask, state)
                for subset, mask in subsets.items()}
        for group in sorted(set(groups)):
            per_layout.append(dict(model=label, group=group,
                **action_metrics(raw, mapped, true_raw, true_mapped, history, groups == group, state)))
        for bias, indices in windows[:5]:
            first, last = indices[0], indices[-1]
            window_rows.append({"model": label, "group": groups[first], "condition": condition[first],
                                "start_step": int(step[first]), "end_step": int(step[last]), "signed_bias": bias,
                                "max_abs_cte": float(np.max(np.abs(state[indices, 0] * 2)))})
        report["models"][label] = info
        saved[label + "_geometry"], saved[label + "_raw"], saved[label + "_mapped"] = geometry, raw, mapped
        print(label, json.dumps(info["action"]["all"]), flush=True)
    baseline_layout_mae = np.array([r["steer_mae"] for r in per_layout if r["model"] == "baseline"])
    rng = np.random.default_rng(42)
    bootstrap_indices = rng.integers(0, len(baseline_layout_mae), size=(5000, len(baseline_layout_mae)))
    report["paired_macro_layout_mae"] = {}
    for label in heads:
        values = np.array([r["steer_mae"] for r in per_layout if r["model"] == label])
        delta = values - baseline_layout_mae
        report["paired_macro_layout_mae"][label] = {"mean_mae": float(values.mean()), "difference_vs_baseline": float(delta.mean()),
            "bootstrap_95_interval": np.percentile(delta[bootstrap_indices].mean(axis=1), [2.5, 97.5]).tolist(),
            "note": "Exploratory paired layout bootstrap; nine development layouts, not independent final-test evidence."}
    for filename, rows in [("per_layout.csv", per_layout), ("worst_bias_windows.csv", window_rows)]:
        with (args.output_dir / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    np.savez(args.output_dir / "predictions.npz", **saved)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    source_dir = args.output_dir / "source"
    source_dir.mkdir()
    for source in [Path(__file__), ROOT / "tools/train_geometry_recovery.py", ROOT / "rl/train_vae_sac.py"]:
        shutil.copy2(source, source_dir / source.name)
    print("Saved", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
