"""Stage 3.1: Strictly fold-isolated 15-dim geometry perception benchmark.

Evaluates how accurately candidate visual representations predict the 15 geometric
inputs of `TrackState.encode` (rl/track_state.py:70-80) on held-out layouts, and
how much teacher action discrepancy those geometric prediction residuals induce when
spliced into the frozen ground-truth teacher (`sac_sim_state_60000_steps.zip`) with
identical motion state (`[3:6]`) and 40-dim command history.

First-round candidates (5-fold GroupKFold by `(run, layout)`, inner 3-fold CV):
  1. `cls_ridge`          DINOv2 CLS token (384) + Ridge
  2. `pca128_ridge`       Fold-fit whitened PCA-128 on 4x4 patch grid + Ridge
  3. `raw6144_ridge`      4x4 patch grid (6144) + Ridge (per-target alpha via SVD)
  4. `raw6144_b64_lin`    Supervised linear bottleneck 6144 -> 64 -> 15
  5. `raw6144_b64_mlp`    Supervised MLP bottleneck 6144 -> 64 (LayerNorm+GELU) -> 15
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from stable_baselines3 import SAC

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from probe_state_from_features import extract_features, load_frames  # noqa: E402
from probe_teacher_action import teacher_inputs  # noqa: E402
from rl.geometry_encoder import (  # noqa: E402
    GEOM_IDX,
    MOTION_IDX,
    compute_cache_fingerprint,
    verify_cache_fingerprint,
)
from rl.track_state import LOOKAHEAD, TrackState  # noqa: E402

ALPHAS = np.logspace(-1, 6, 16)


def load_or_build_cache(
    runs: list[Path],
    centerline_path: Path,
    cache_path: Path,
    device: torch.device,
    crop_top: int = 0,
    rebuild_cache: bool = False,
) -> dict[str, np.ndarray]:
    meta_path = cache_path.with_suffix(".meta.json")
    expected_meta = compute_cache_fingerprint(
        runs=runs,
        centerline_path=centerline_path,
        crop_top=crop_top,
        encoder_name="dinov2_vits14",
    )
    if cache_path.exists() and not rebuild_cache:
        verify_cache_fingerprint(
            cache_path=cache_path,
            runs=runs,
            centerline_path=centerline_path,
            crop_top=crop_top,
            encoder_name="dinov2_vits14",
        )
        print(
            f"Loading cached features from {cache_path} "
            f"(verified fingerprint {expected_meta['fingerprint_sha256'][:12]}) ...",
            flush=True,
        )
        data = np.load(cache_path, allow_pickle=False)
        return {k: data[k] for k in data.files}

    t0 = time.perf_counter()
    records = [record for run in runs for record in load_frames(run)]
    track = TrackState.load(centerline_path)
    state, history = teacher_inputs(runs, records, track)
    features = extract_features(records, device, crop_top=crop_top)

    payload = {
        "cls": features["cls"].astype(np.float32),
        "patch_grid": features["patch_grid"].astype(np.float32),
        "state": state.astype(np.float32),
        "history": history.astype(np.float32),
        "run": np.array([r["run"] for r in records]),
        "layout": np.array([str(r["layout"]) for r in records]),
        "model": np.array([str(r["model"] or "") for r in records]),
        "step": np.array([int(r["step"]) for r in records], dtype=np.int32),
        "failed": np.array([bool(r["failed"]) for r in records], dtype=bool),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **payload)
    expected_meta["n_frames"] = len(records)
    meta_path.write_text(json.dumps(expected_meta, indent=2) + "\n")
    print(f"Cached {len(records)} frames to {cache_path} in {time.perf_counter() - t0:.1f}s", flush=True)
    return payload


def svd_ridge_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    alphas: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Return predictions of shape (len(alphas), len(x_eval), y_dim) using Gram eigh."""
    xt = torch.from_numpy(x_train).to(device=device, dtype=torch.float32)
    yt = torch.from_numpy(y_train).to(device=device, dtype=torch.float32)
    xe = torch.from_numpy(x_eval).to(device=device, dtype=torch.float32)

    y_mean = yt.mean(0, keepdim=True)
    yt_c = yt - y_mean
    gram = xt.T @ xt  # (D, D)
    evals, evecs = torch.linalg.eigh(gram)  # ascending
    evals = evals.clamp_min(0.0)
    xty = xt.T @ yt_c  # (D, y_dim)
    proj_y = evecs.T @ xty  # (D, y_dim)
    proj_x = xe @ evecs     # (n_eval, D)

    alpha_t = torch.tensor(alphas, device=device, dtype=torch.float32).unsqueeze(1)  # (A, 1)
    inv = 1.0 / (evals.unsqueeze(0) + alpha_t)  # (A, D)
    weights = inv.unsqueeze(2) * proj_y.unsqueeze(0)  # (A, D, y_dim)
    preds = torch.einsum("nd,ady->any", proj_x, weights) + y_mean.unsqueeze(0)
    return preds


def cv_ridge_predictions(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    device: torch.device,
    pca_components: int | None = None,
) -> np.ndarray:
    """5-fold GroupKFold with inner 3-fold GroupKFold selecting per-target alpha."""
    out = np.zeros_like(y, dtype=np.float32)
    for train, test in GroupKFold(5).split(x, y, groups):
        if pca_components is not None:
            xt_raw = torch.from_numpy(x[train]).to(device=device, dtype=torch.float32)
            mean = xt_raw.mean(0)
            centred = xt_raw - mean
            cov = (centred.T @ centred) / (len(train) - 1)
            evals, evecs = torch.linalg.eigh(cov)
            evals = evals.flip(0)[:pca_components].clamp_min(1e-6)
            comps = evecs.flip(1)[:, :pca_components].T  # (k, D)
            scale = evals.sqrt()
            x_tr = ((xt_raw - mean) @ comps.T / scale).cpu().numpy()
            xe_raw = torch.from_numpy(x[test]).to(device=device, dtype=torch.float32)
            x_te = ((xe_raw - mean) @ comps.T / scale).cpu().numpy()
        else:
            mean = x[train].mean(0)
            std = x[train].std(0) + 1e-6
            x_tr = ((x[train] - mean) / std).astype(np.float32)
            x_te = ((x[test] - mean) / std).astype(np.float32)

        val_sse = np.zeros((len(ALPHAS), y.shape[1]), dtype=np.float64)
        for fit, val in GroupKFold(3).split(x_tr, y[train], groups[train]):
            f_mean = x_tr[fit].mean(0)
            f_std = x_tr[fit].std(0) + 1e-6
            preds = svd_ridge_fit_predict(
                (x_tr[fit] - f_mean) / f_std,
                y[train][fit],
                (x_tr[val] - f_mean) / f_std,
                ALPHAS,
                device,
            ).cpu().numpy()
            val_sse += ((preds - y[train][val][None, :, :]) ** 2).sum(axis=1)

        best_alpha_idx = np.argmin(val_sse, axis=0)  # (y_dim,)
        all_preds = svd_ridge_fit_predict(x_tr, y[train], x_te, ALPHAS, device).cpu().numpy()
        for j in range(y.shape[1]):
            out[test, j] = all_preds[best_alpha_idx[j], :, j]
    return out


class BottleneckHead(nn.Module):
    def __init__(self, in_dim: int, bottleneck: int, out_dim: int, nonlinear: bool) -> None:
        super().__init__()
        if nonlinear:
            self.net = nn.Sequential(
                nn.Linear(in_dim, bottleneck),
                nn.LayerNorm(bottleneck),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(bottleneck, out_dim),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, bottleneck, bias=False),
                nn.Linear(bottleneck, out_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def cv_bottleneck_predictions(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    device: torch.device,
    bottleneck: int = 64,
    nonlinear: bool = False,
    max_epochs: int = 50,
) -> np.ndarray:
    """5-fold GroupKFold with inner 3-fold validation for early stopping & ensemble."""
    out = np.zeros_like(y, dtype=np.float32)
    for fold, (train, test) in enumerate(GroupKFold(5).split(x, y, groups)):
        x_mean, x_std = x[train].mean(0), x[train].std(0) + 1e-6
        y_mean, y_std = y[train].mean(0), y[train].std(0) + 1e-6

        x_tr = torch.from_numpy((x[train] - x_mean) / x_std).to(device=device, dtype=torch.float32)
        y_tr = torch.from_numpy((y[train] - y_mean) / y_std).to(device=device, dtype=torch.float32)
        x_te = torch.from_numpy((x[test] - x_mean) / x_std).to(device=device, dtype=torch.float32)

        fold_preds = np.zeros((len(test), y.shape[1]), dtype=np.float32)
        inner_splits = list(GroupKFold(3).split(x[train], y[train], groups[train]))
        for inner_id, (fit, val) in enumerate(inner_splits):
            torch.manual_seed(fold * 10 + inner_id)
            model = BottleneckHead(x.shape[1], bottleneck, y.shape[1], nonlinear=nonlinear).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
            fit_t = torch.tensor(fit, device=device)
            val_t = torch.tensor(val, device=device)

            best_loss = float("inf")
            best_state = None
            for _ in range(max_epochs):
                model.train()
                for batch in fit_t[torch.randperm(len(fit_t), device=device)].split(256):
                    opt.zero_grad()
                    loss = nn.functional.mse_loss(model(x_tr[batch]), y_tr[batch])
                    loss.backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    vloss = float(nn.functional.mse_loss(model(x_tr[val_t]), y_tr[val_t]).item())
                if vloss < best_loss:
                    best_loss = vloss
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                pred_norm = model(x_te).cpu().numpy()
            fold_preds += (pred_norm * y_std + y_mean) / len(inner_splits)
        out[test] = fold_preds
    return out


def map_and_clip_batch(
    raw_action: np.ndarray,
    prev_steer: np.ndarray,
    max_steering: float = 1.0,
    max_steering_diff: float = 0.1,
    min_throttle: float = 0.2,
    max_throttle: float = 0.7,
) -> np.ndarray:
    """Vectorized exact replica of DonkeyControlEnv._map_and_clip_action."""
    steer = np.clip(raw_action[:, 0], -max_steering, max_steering)
    t = np.clip((raw_action[:, 1] + 1.0) / 2.0, 0.0, 1.0)
    throttle = (1.0 - t) * min_throttle + max_throttle * t
    max_diff = (max_steering_diff - 1e-5) * (2.0 * max_steering)
    steer = prev_steer + np.clip(steer - prev_steer, -max_diff, max_diff)
    return np.stack([steer.astype(np.float32), throttle.astype(np.float32)], axis=1)


def windowed_bias_stats(
    signed_error: np.ndarray,
    run: np.ndarray,
    layout: np.ndarray,
    model: np.ndarray,
    step: np.ndarray,
    mask: np.ndarray,
    windows: tuple[int, ...] = (10, 20),
) -> dict[str, dict[str, float]]:
    """Compute per-episode max |rolling_mean(error, W)| over strictly contiguous steps."""
    episodes = defaultdict(list)
    for idx in np.flatnonzero(mask):
        episodes[(run[idx], layout[idx], model[idx])].append((int(step[idx]), float(signed_error[idx])))

    out = {}
    for w in windows:
        ep_max_abs = []
        for items in episodes.values():
            items.sort(key=lambda pair: pair[0])
            steps_arr = np.array([p[0] for p in items], dtype=np.int32)
            errs_arr = np.array([p[1] for p in items], dtype=np.float64)
            if len(items) < w:
                continue
            csum = np.concatenate([[0.0], np.cumsum(errs_arr)])
            win_means = (csum[w:] - csum[:-w]) / w
            contiguous = (steps_arr[w - 1:] - steps_arr[:len(steps_arr) - w + 1]) == (w - 1)
            if contiguous.any():
                ep_max_abs.append(float(np.max(np.abs(win_means[contiguous]))))
        if ep_max_abs:
            arr = np.array(ep_max_abs)
            out[f"w{w}"] = {
                "episodes": len(arr),
                "median_max_abs": float(np.median(arr)),
                "p95_max_abs": float(np.percentile(arr, 95)),
                "max_abs": float(np.max(arr)),
            }
        else:
            out[f"w{w}"] = {"episodes": 0, "median_max_abs": None, "p95_max_abs": None, "max_abs": None}
    return out


def evaluate_predictions(
    pred_geom: np.ndarray,
    true_state: np.ndarray,
    history: np.ndarray,
    teacher: SAC,
    true_raw_act: np.ndarray,
    true_exec_cmd: np.ndarray,
    run: np.ndarray,
    layout: np.ndarray,
    model: np.ndarray,
    step: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Compute physical geometry errors, windowed bias, and teacher action errors on `mask`."""
    idx = np.flatnonzero(mask)
    pg = pred_geom[idx]
    ts = true_state[idx]
    hist = history[idx]

    # 1. CTE physical error (units: CTE, where track limit is 2.0)
    cte_true = ts[:, 0] * 2.0
    cte_pred = pg[:, 0] * 2.0
    cte_err = cte_pred - cte_true

    # 2. Heading circular error (units: degrees) after unit-circle projection
    sin_t, cos_t = ts[:, 1], ts[:, 2]
    norm = np.maximum(np.hypot(pg[:, 1], pg[:, 2]), 1e-6)
    sin_p, cos_p = pg[:, 1] / norm, pg[:, 2] / norm
    heading_err_rad = np.arctan2(sin_p * cos_t - cos_p * sin_t, sin_p * sin_t + cos_p * cos_t)
    heading_err_deg = np.degrees(heading_err_rad)

    # 3. Lookahead points restored to position units (multiply by distance)
    lookahead_report = {}
    for k, dist in enumerate(LOOKAHEAD):
        r_true = ts[:, 6 + 2 * k] * dist
        f_true = ts[:, 6 + 2 * k + 1] * dist
        r_pred = pg[:, 3 + 2 * k] * dist
        f_pred = pg[:, 3 + 2 * k + 1] * dist
        r_err = np.abs(r_pred - r_true)
        f_err = np.abs(f_pred - f_true)
        euclid = np.hypot(r_pred - r_true, f_pred - f_true)
        lookahead_report[f"{dist:g}"] = {
            "right_mae_pos": float(r_err.mean()),
            "right_p95_pos": float(np.percentile(r_err, 95)),
            "forward_mae_pos": float(f_err.mean()),
            "forward_p95_pos": float(np.percentile(f_err, 95)),
            "euclid_mae_pos": float(euclid.mean()),
            "euclid_p95_pos": float(np.percentile(euclid, 95)),
        }

    # 4. Substitute into teacher with unit-normalized sin/cos and identical motion + history
    state_sub = ts.copy()
    state_sub[:, 0] = pg[:, 0]
    state_sub[:, 1] = sin_p
    state_sub[:, 2] = cos_p
    state_sub[:, 6:18] = pg[:, 3:15]
    pred_raw_act, _ = teacher.predict(np.concatenate([state_sub, hist], axis=1), deterministic=True)
    prev_steer = hist[:, -2]
    pred_exec_cmd = map_and_clip_batch(pred_raw_act, prev_steer)

    # Also test unnormalized sin/cos substitution for comparison
    state_sub_raw = ts.copy()
    state_sub_raw[:, 0:3] = pg[:, 0:3]
    state_sub_raw[:, 6:18] = pg[:, 3:15]
    pred_raw_act_unnorm, _ = teacher.predict(np.concatenate([state_sub_raw, hist], axis=1), deterministic=True)

    raw_steer_err = np.abs(pred_raw_act[:, 0] - true_raw_act[idx, 0])
    raw_throt_err = np.abs(pred_raw_act[:, 1] - true_raw_act[idx, 1])
    exec_steer_err = np.abs(pred_exec_cmd[:, 0] - true_exec_cmd[idx, 0])
    exec_throt_err = np.abs(pred_exec_cmd[:, 1] - true_exec_cmd[idx, 1])

    # Full-array signed errors for windowed bias on `mask`
    full_cte_err = (pred_geom[:, 0] - true_state[:, 0]) * 2.0
    full_norm = np.maximum(np.hypot(pred_geom[:, 1], pred_geom[:, 2]), 1e-6)
    full_sin_p, full_cos_p = pred_geom[:, 1] / full_norm, pred_geom[:, 2] / full_norm
    full_head_deg = np.degrees(
        np.arctan2(
            full_sin_p * true_state[:, 2] - full_cos_p * true_state[:, 1],
            full_sin_p * true_state[:, 1] + full_cos_p * true_state[:, 2],
        )
    )

    return {
        "frames": int(len(idx)),
        "cte": {
            "mae": float(np.abs(cte_err).mean()),
            "p95": float(np.percentile(np.abs(cte_err), 95)),
            "r2": float(1.0 - np.sum(cte_err ** 2) / np.sum((cte_true - cte_true.mean()) ** 2)),
            "window_bias": windowed_bias_stats(full_cte_err, run, layout, model, step, mask),
        },
        "heading_deg": {
            "mae": float(np.abs(heading_err_deg).mean()),
            "p95": float(np.percentile(np.abs(heading_err_deg), 95)),
            "window_bias": windowed_bias_stats(full_head_deg, run, layout, model, step, mask),
        },
        "lookahead_pos": lookahead_report,
        "teacher_action": {
            "raw_steer_mae": float(raw_steer_err.mean()),
            "raw_steer_p95": float(np.percentile(raw_steer_err, 95)),
            "raw_steer_mae_unnorm_sincos": float(np.abs(pred_raw_act_unnorm[:, 0] - true_raw_act[idx, 0]).mean()),
            "raw_throttle_mae": float(raw_throt_err.mean()),
            "raw_throttle_p95": float(np.percentile(raw_throt_err, 95)),
            "exec_steer_mae": float(exec_steer_err.mean()),
            "exec_steer_p95": float(np.percentile(exec_steer_err, 95)),
            "exec_throttle_mae": float(exec_throt_err.mean()),
            "exec_throttle_p95": float(np.percentile(exec_throt_err, 95)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--teacher", type=Path, default=Path("models/rl_loop_sim_state_ts2_sdiff01_v1/sac_sim_state_60000_steps.zip"))
    parser.add_argument("--track-centerline", type=Path, default=Path("models/track_centerline/generated_track.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/probe_geometry_15d_v1"))
    parser.add_argument("--crop-top", type=int, default=0)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    cache_path = args.output_dir / "feature_cache.npz"
    cache = load_or_build_cache(
        args.runs,
        args.track_centerline,
        cache_path,
        device,
        crop_top=args.crop_top,
        rebuild_cache=args.rebuild_cache,
    )
    cls_feat = cache["cls"]
    grid_feat = cache["patch_grid"]
    state = cache["state"]
    history = cache["history"]
    run = cache["run"]
    layout = cache["layout"]
    model = cache["model"]
    step = cache["step"]

    y_geom = state[:, GEOM_IDX]
    groups = np.array([f"{r}/{l}" for r, l in zip(run, layout)])

    teacher = SAC.load(str(args.teacher), device="cpu")
    true_raw_act, _ = teacher.predict(np.concatenate([state, history], axis=1), deterministic=True)
    true_exec_cmd = map_and_clip_batch(true_raw_act, history[:, -2])

    is_teacher_run = run == "sim_state_sdiff01_60k_24layouts"
    abs_cte = np.abs(state[:, 0] * 2.0)
    subsets = {
        "all": np.ones(len(state), dtype=bool),
        "teacher_24layouts": is_teacher_run,
        "v2_abs_cte_lt_1": (~is_teacher_run) & (abs_cte < 1.0),
        "v2_abs_cte_ge_1": (~is_teacher_run) & (abs_cte >= 1.0),
    }

    candidates = [
        ("cls_ridge", lambda: cv_ridge_predictions(cls_feat, y_geom, groups, device)),
        ("pca128_ridge", lambda: cv_ridge_predictions(grid_feat, y_geom, groups, device, pca_components=128)),
        ("raw6144_ridge", lambda: cv_ridge_predictions(grid_feat, y_geom, groups, device)),
        ("raw6144_b64_lin", lambda: cv_bottleneck_predictions(grid_feat, y_geom, groups, device, bottleneck=64, nonlinear=False)),
        ("raw6144_b64_mlp", lambda: cv_bottleneck_predictions(grid_feat, y_geom, groups, device, bottleneck=64, nonlinear=True)),
    ]

    results = {
        "runs": [str(r) for r in args.runs],
        "teacher": str(args.teacher),
        "units": {
            "cte": "simulator CTE units (track termination limit is +/-2.0; ~7.98 CTE units per position unit)",
            "heading": "degrees (circular angle difference after unit-circle normalization)",
            "lookahead": "simulator position units (1 position unit is ~7.98 CTE units)",
        },
        "frames": int(len(state)),
        "layouts": int(len(set(groups))),
        "candidates": {},
    }
    saved_preds = {"true_geom": y_geom}

    header = (
        f"{'candidate':<17} {'subset':<18} {'fit_s':>6} "
        f"{'CTE(u) MAE/P95':>14} {'CTE w20 P95':>11} "
        f"{'Head MAE/P95':>14} {'LA1.0(pos) M/P95':>16} "
        f"{'RawSteer MAE/P95':>16} {'ExecSteer MAE/P95':>17} {'ExecThrot MAE/P95':>17}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for name, fn in candidates:
        t0 = time.perf_counter()
        pred_geom = fn()
        elapsed = time.perf_counter() - t0
        saved_preds[f"pred_{name}"] = pred_geom
        results["candidates"][name] = {"fit_seconds": elapsed, "subsets": {}}

        for sub_name, mask in subsets.items():
            sub_res = evaluate_predictions(
                pred_geom, state, history, teacher, true_raw_act, true_exec_cmd,
                run, layout, model, step, mask,
            )
            results["candidates"][name]["subsets"][sub_name] = sub_res
            c = sub_res["cte"]
            h = sub_res["heading_deg"]
            la1 = sub_res["lookahead_pos"]["1"]
            ta = sub_res["teacher_action"]
            w20 = c["window_bias"]["w20"]["p95_max_abs"]
            w20_str = f"{w20:.3f}" if w20 is not None else "n/a"
            print(
                f"{name:<17} {sub_name:<18} {elapsed:>5.1f}s "
                f"{c['mae']:>5.3f}/{c['p95']:<5.3f}   {w20_str:>11} "
                f"{h['mae']:>5.2f}°/{h['p95']:<5.2f}° "
                f"{la1['euclid_mae_pos']:>6.3f}/{la1['euclid_p95_pos']:<6.3f}   "
                f"{ta['raw_steer_mae']:>6.3f}/{ta['raw_steer_p95']:<6.3f}    "
                f"{ta['exec_steer_mae']:>7.3f}/{ta['exec_steer_p95']:<7.3f}   "
                f"{ta['exec_throttle_mae']:>7.3f}/{ta['exec_throttle_p95']:<7.3f}",
                flush=True,
            )

    np.savez(args.output_dir / "oof_geometry_predictions.npz", **saved_preds)
    (args.output_dir / "probe_geometry_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nSaved results to {args.output_dir / 'probe_geometry_results.json'}", flush=True)


if __name__ == "__main__":
    main()
