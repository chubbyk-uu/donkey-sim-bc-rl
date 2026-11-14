"""Stage 3.1 Round 2: Lagged (t, t-2, t-4) 15-dim geometry perception benchmark.

Reuses `feature_cache.npz` and `oof_geometry_predictions.npz` from Round 1, restricts
all comparisons to the exact same `valid` frame mask (frames whose t-2 and t-4
frames were captured in the same episode), and compares:
  - Single-frame winners from Round 1 (re-scored on the exact `valid` mask):
      `pca128_ridge (1f)`, `raw6144_ridge (1f)`, `raw6144_b64_mlp (1f)`
  - Multi-frame lagged candidates (`LAGS = (2, 4)`), strictly fold-isolated:
      1. `pca128_lags_ridge`         Fold-fit PCA-128 per frame, stack (t, t-2, t-4) [384D] + Ridge
      2. `proj15_lags_ridge`         Fold-fit single-frame 6144->15 Ridge, stack (t, t-2, t-4) [45D] + Ridge
      3. `proj15_lags_mlp`           Fold-fit single-frame 6144->15 Ridge, stack (t, t-2, t-4) [45D] + MLP
      4. `shared64_lags_mlp`         Shared per-frame 6144->64 encoder over (t, t-2, t-4) [192D] -> 64 -> 15
      5. `raw18432_lags_ridge`       Full 3x6144 = 18432D Ridge (via dual Gram Cholesky solve)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from stable_baselines3 import SAC

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from probe_geometry_15d import (  # noqa: E402
    ALPHAS,
    GEOM_IDX,
    evaluate_predictions,
    map_and_clip_batch,
    svd_ridge_fit_predict,
)

LAGS = (2, 4)
FAST_ALPHAS = np.array([1e1, 1e2, 1e3, 1e4, 3e4, 1e5, 3e5, 1e6], dtype=np.float32)


def build_lag_indices(
    run: np.ndarray,
    layout: np.ndarray,
    model: np.ndarray,
    step: np.ndarray,
    lags: tuple[int, ...] = LAGS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (idx_matrix of shape (N, 1+len(lags)), valid_mask of shape (N,))."""
    where = {(run[i], layout[i], model[i], int(step[i])): i for i in range(len(step))}
    idx_mat = np.zeros((len(step), 1 + len(lags)), dtype=np.int32)
    valid = np.ones(len(step), dtype=bool)
    for i in range(len(step)):
        idx_mat[i, 0] = i
        key_prefix = (run[i], layout[i], model[i])
        s = int(step[i])
        for k, lag in enumerate(lags, start=1):
            j = where.get((*key_prefix, s - lag))
            if j is None:
                valid[i] = False
                break
            idx_mat[i, k] = j
    return idx_mat, valid


def primal_cholesky_ridge(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_eval: torch.Tensor,
    alphas: np.ndarray,
) -> np.ndarray:
    """Solve primal Ridge (X^T X + alpha I) W = X^T y_c via fast GPU Cholesky."""
    y_mean = y_tr.mean(0, keepdim=True)
    yt_c = y_tr - y_mean
    gram = x_tr.T @ x_tr
    xty = x_tr.T @ yt_c
    d = gram.shape[0]
    eye = torch.eye(d, device=gram.device, dtype=gram.dtype)
    preds = []
    for alpha in alphas:
        chol, _ = torch.linalg.cholesky_ex(gram + float(alpha) * eye)
        w = torch.cholesky_solve(xty, chol)
        preds.append((x_eval @ w + y_mean).cpu().numpy())
    return np.stack(preds, axis=0)


def dual_cholesky_ridge(
    k_tr: torch.Tensor,
    y_tr: torch.Tensor,
    k_eval_tr: torch.Tensor,
    alphas: np.ndarray,
) -> np.ndarray:
    """Solve dual Ridge (K_tr + alpha I) c = y_c via fast GPU Cholesky."""
    y_mean = y_tr.mean(0, keepdim=True)
    yt_c = y_tr - y_mean
    n = k_tr.shape[0]
    eye = torch.eye(n, device=k_tr.device, dtype=k_tr.dtype)
    preds = []
    for alpha in alphas:
        chol, _ = torch.linalg.cholesky_ex(k_tr + float(alpha) * eye)
        coeff = torch.cholesky_solve(yt_c, chol)
        preds.append((k_eval_tr @ coeff + y_mean).cpu().numpy())
    return np.stack(preds, axis=0)


def cv_pca128_lags_ridge(
    grid: np.ndarray,
    y_all: np.ndarray,
    idx_mat: np.ndarray,
    valid: np.ndarray,
    groups_all: np.ndarray,
    device: torch.device,
    pca_components: int = 128,
) -> np.ndarray:
    """Fold-fit PCA-128 on single frames, stack (t, t-2, t-4), then fit Ridge."""
    valid_idx = np.flatnonzero(valid)
    y_val = y_all[valid_idx]
    groups_val = groups_all[valid_idx]
    out = np.zeros_like(y_all, dtype=np.float32)

    for train_v, test_v in GroupKFold(5).split(valid_idx, y_val, groups_val):
        train_groups = set(groups_val[train_v])
        frame_train_mask = np.array([g in train_groups for g in groups_all], dtype=bool)

        xt_raw = torch.from_numpy(grid[frame_train_mask]).to(device=device, dtype=torch.float32)
        mean = xt_raw.mean(0)
        centred = xt_raw - mean
        cov = (centred.T @ centred) / (len(xt_raw) - 1)
        evals, evecs = torch.linalg.eigh(cov)
        evals = evals.flip(0)[:pca_components].clamp_min(1e-6)
        comps = evecs.flip(1)[:, :pca_components].T
        scale = evals.sqrt()

        all_proj = ((torch.from_numpy(grid).to(device=device, dtype=torch.float32) - mean) @ comps.T / scale).cpu().numpy()
        stacked = np.concatenate([all_proj[idx_mat[valid_idx, k]] for k in range(idx_mat.shape[1])], axis=1)

        x_tr, x_te = stacked[train_v], stacked[test_v]
        s_mean, s_std = x_tr.mean(0), x_tr.std(0) + 1e-6
        x_tr = (x_tr - s_mean) / s_std
        x_te = (x_te - s_mean) / s_std

        val_sse = np.zeros((len(ALPHAS), y_val.shape[1]), dtype=np.float64)
        for fit, val in GroupKFold(3).split(x_tr, y_val[train_v], groups_val[train_v]):
            f_mean, f_std = x_tr[fit].mean(0), x_tr[fit].std(0) + 1e-6
            preds = svd_ridge_fit_predict(
                (x_tr[fit] - f_mean) / f_std,
                y_val[train_v][fit],
                (x_tr[val] - f_mean) / f_std,
                ALPHAS,
                device,
            ).cpu().numpy()
            val_sse += ((preds - y_val[train_v][val][None, :, :]) ** 2).sum(axis=1)

        best_alpha_idx = np.argmin(val_sse, axis=0)
        all_preds = svd_ridge_fit_predict(x_tr, y_val[train_v], x_te, ALPHAS, device).cpu().numpy()
        for j in range(y_val.shape[1]):
            out[valid_idx[test_v], j] = all_preds[best_alpha_idx[j], :, j]
    return out


def cv_proj15_lags(
    grid: np.ndarray,
    y_all: np.ndarray,
    idx_mat: np.ndarray,
    valid: np.ndarray,
    groups_all: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-stage fold-isolated model using fast GPU Cholesky for Stage 1."""
    valid_idx = np.flatnonzero(valid)
    y_val = y_all[valid_idx]
    groups_val = groups_all[valid_idx]
    out_ridge = np.zeros_like(y_all, dtype=np.float32)
    out_mlp = np.zeros_like(y_all, dtype=np.float32)

    for fold, (train_v, test_v) in enumerate(GroupKFold(5).split(valid_idx, y_val, groups_val)):
        train_groups = set(groups_val[train_v])
        frame_tr_idx = np.flatnonzero(np.array([g in train_groups for g in groups_all], dtype=bool))
        frame_te_idx = np.flatnonzero(np.array([g not in train_groups for g in groups_all], dtype=bool))

        x_tr_raw = grid[frame_tr_idx]
        y_tr_raw = y_all[frame_tr_idx]
        g_tr_raw = groups_all[frame_tr_idx]

        mean_tr, std_tr = x_tr_raw.mean(0), x_tr_raw.std(0) + 1e-6
        x_tr_norm = torch.from_numpy((x_tr_raw - mean_tr) / std_tr).to(device=device, dtype=torch.float32)
        x_te_norm = torch.from_numpy((grid[frame_te_idx] - mean_tr) / std_tr).to(device=device, dtype=torch.float32)
        y_tr_t = torch.from_numpy(y_tr_raw).to(device=device, dtype=torch.float32)

        oof_tr_15 = np.zeros_like(y_tr_raw, dtype=np.float32)
        val_sse = np.zeros((len(FAST_ALPHAS), y_all.shape[1]), dtype=np.float64)
        inner_splits = list(GroupKFold(3).split(x_tr_raw, y_tr_raw, g_tr_raw))
        inner_preds_cache = []
        for fit, val in inner_splits:
            fit_t = torch.tensor(fit, device=device)
            val_t = torch.tensor(val, device=device)
            preds = primal_cholesky_ridge(x_tr_norm[fit_t], y_tr_t[fit_t], x_tr_norm[val_t], FAST_ALPHAS)
            val_sse += ((preds - y_tr_raw[val][None, :, :]) ** 2).sum(axis=1)
            inner_preds_cache.append((val, preds))

        best_alpha_idx = np.argmin(val_sse, axis=0)
        for val, preds in inner_preds_cache:
            for j in range(y_all.shape[1]):
                oof_tr_15[val, j] = preds[best_alpha_idx[j], :, j]

        te_all_preds = primal_cholesky_ridge(x_tr_norm, y_tr_t, x_te_norm, FAST_ALPHAS)
        frame_pred_15 = np.zeros_like(y_all, dtype=np.float32)
        frame_pred_15[frame_tr_idx] = oof_tr_15
        for j in range(y_all.shape[1]):
            frame_pred_15[frame_te_idx, j] = te_all_preds[best_alpha_idx[j], :, j]

        stacked_45 = np.concatenate([frame_pred_15[idx_mat[valid_idx, k]] for k in range(idx_mat.shape[1])], axis=1)
        s_tr, s_te = stacked_45[train_v], stacked_45[test_v]
        s_mean, s_std = s_tr.mean(0), s_tr.std(0) + 1e-6
        s_tr_n = (s_tr - s_mean) / s_std
        s_te_n = (s_te - s_mean) / s_std

        # Stage 2A: Ridge on 45D
        val_sse2 = np.zeros((len(ALPHAS), y_val.shape[1]), dtype=np.float64)
        for fit, val in GroupKFold(3).split(s_tr_n, y_val[train_v], groups_val[train_v]):
            preds2 = svd_ridge_fit_predict(s_tr_n[fit], y_val[train_v][fit], s_tr_n[val], ALPHAS, device).cpu().numpy()
            val_sse2 += ((preds2 - y_val[train_v][val][None, :, :]) ** 2).sum(axis=1)
        best_a2 = np.argmin(val_sse2, axis=0)
        preds_te2 = svd_ridge_fit_predict(s_tr_n, y_val[train_v], s_te_n, ALPHAS, device).cpu().numpy()
        for j in range(y_val.shape[1]):
            out_ridge[valid_idx[test_v], j] = preds_te2[best_a2[j], :, j]

        # Stage 2B: MLP on 45D
        y_mean, y_std = y_val[train_v].mean(0), y_val[train_v].std(0) + 1e-6
        y_tr_v_t = torch.from_numpy((y_val[train_v] - y_mean) / y_std).to(device=device, dtype=torch.float32)
        s_tr_t = torch.from_numpy(s_tr_n).to(device=device, dtype=torch.float32)
        s_te_t = torch.from_numpy(s_te_n).to(device=device, dtype=torch.float32)

        fold_mlp_preds = np.zeros((len(test_v), y_val.shape[1]), dtype=np.float32)
        inner_v_splits = list(GroupKFold(3).split(s_tr_n, y_val[train_v], groups_val[train_v]))
        for inner_id, (fit, val) in enumerate(inner_v_splits):
            torch.manual_seed(fold * 10 + inner_id)
            net = nn.Sequential(
                nn.Linear(45, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(64, y_val.shape[1]),
            ).to(device)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
            fit_t = torch.tensor(fit, device=device)
            val_t = torch.tensor(val, device=device)
            best_loss = float("inf")
            best_state = None
            for _ in range(40):
                net.train()
                for batch in fit_t[torch.randperm(len(fit_t), device=device)].split(256):
                    opt.zero_grad()
                    loss = nn.functional.mse_loss(net(s_tr_t[batch]), y_tr_v_t[batch])
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    vloss = float(nn.functional.mse_loss(net(s_tr_t[val_t]), y_tr_v_t[val_t]).item())
                if vloss < best_loss:
                    best_loss = vloss
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            net.load_state_dict(best_state)
            net.eval()
            with torch.no_grad():
                p_norm = net(s_te_t).cpu().numpy()
            fold_mlp_preds += (p_norm * y_std + y_mean) / len(inner_v_splits)
        out_mlp[valid_idx[test_v]] = fold_mlp_preds

    return out_ridge, out_mlp


class SharedFrameLagMLP(nn.Module):
    """Per-frame 6144 -> 64 shared encoder applied to (t, t-2, t-4), then 192 -> 64 -> 15."""

    def __init__(self, in_dim: int = 6144, n_frames: int = 3, latent_dim: int = 64, out_dim: int = 15) -> None:
        super().__init__()
        self.frame_enc = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(latent_dim * n_frames, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, out_dim),
        )

    def forward(self, frame_bank: torch.Tensor, idx_batch: torch.Tensor) -> torch.Tensor:
        b, _ = idx_batch.shape
        flat_frames = frame_bank[idx_batch.reshape(-1)]
        z = self.frame_enc(flat_frames).reshape(b, -1)
        return self.head(z)


def cv_shared64_lags_mlp(
    grid: np.ndarray,
    y_all: np.ndarray,
    idx_mat: np.ndarray,
    valid: np.ndarray,
    groups_all: np.ndarray,
    device: torch.device,
    max_epochs: int = 35,
) -> np.ndarray:
    valid_idx = np.flatnonzero(valid)
    y_val = y_all[valid_idx]
    groups_val = groups_all[valid_idx]
    idx_val = idx_mat[valid_idx]
    out = np.zeros_like(y_all, dtype=np.float32)

    for fold, (train_v, test_v) in enumerate(GroupKFold(5).split(valid_idx, y_val, groups_val)):
        train_groups = set(groups_val[train_v])
        frame_tr_mask = np.array([g in train_groups for g in groups_all], dtype=bool)
        x_mean, x_std = grid[frame_tr_mask].mean(0), grid[frame_tr_mask].std(0) + 1e-6
        y_mean, y_std = y_val[train_v].mean(0), y_val[train_v].std(0) + 1e-6

        frame_bank = torch.from_numpy((grid - x_mean) / x_std).to(device=device, dtype=torch.float32)
        y_tr_t = torch.from_numpy((y_val[train_v] - y_mean) / y_std).to(device=device, dtype=torch.float32)
        idx_tr_t = torch.from_numpy(idx_val[train_v]).to(device=device, dtype=torch.long)
        idx_te_t = torch.from_numpy(idx_val[test_v]).to(device=device, dtype=torch.long)

        fold_preds = np.zeros((len(test_v), y_val.shape[1]), dtype=np.float32)
        inner_splits = list(GroupKFold(3).split(train_v, y_val[train_v], groups_val[train_v]))
        for inner_id, (fit, val) in enumerate(inner_splits):
            torch.manual_seed(fold * 10 + inner_id)
            model = SharedFrameLagMLP(in_dim=grid.shape[1], n_frames=idx_mat.shape[1], latent_dim=64, out_dim=y_val.shape[1]).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
            fit_t = torch.tensor(fit, device=device)
            val_t = torch.tensor(val, device=device)
            best_loss = float("inf")
            best_state = None
            for _ in range(max_epochs):
                model.train()
                for batch in fit_t[torch.randperm(len(fit_t), device=device)].split(512):
                    opt.zero_grad()
                    loss = nn.functional.mse_loss(model(frame_bank, idx_tr_t[batch]), y_tr_t[batch])
                    loss.backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    vloss = float(nn.functional.mse_loss(model(frame_bank, idx_tr_t[val_t]), y_tr_t[val_t]).item())
                if vloss < best_loss:
                    best_loss = vloss
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                p_norm = model(frame_bank, idx_te_t).cpu().numpy()
            fold_preds += (p_norm * y_std + y_mean) / len(inner_splits)
        out[valid_idx[test_v]] = fold_preds
    return out


def cv_raw18432_lags_ridge(
    grid: np.ndarray,
    y_all: np.ndarray,
    idx_mat: np.ndarray,
    valid: np.ndarray,
    groups_all: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Full 3x6144 = 18432D Ridge solved via fast GPU dual Gram Cholesky."""
    valid_idx = np.flatnonzero(valid)
    y_val = y_all[valid_idx]
    groups_val = groups_all[valid_idx]
    idx_val = idx_mat[valid_idx]
    out = np.zeros_like(y_all, dtype=np.float32)
    alphas_18k = FAST_ALPHAS * 3.0

    for train_v, test_v in GroupKFold(5).split(valid_idx, y_val, groups_val):
        train_groups = set(groups_val[train_v])
        frame_tr_mask = np.array([g in train_groups for g in groups_all], dtype=bool)
        f_mean, f_std = grid[frame_tr_mask].mean(0), grid[frame_tr_mask].std(0) + 1e-6
        norm_grid = torch.from_numpy((grid - f_mean) / f_std).to(device=device, dtype=torch.float32)

        n_tr = len(train_v)
        n_te = len(test_v)
        k_tr = torch.zeros((n_tr, n_tr), device=device, dtype=torch.float32)
        k_te_tr = torch.zeros((n_te, n_tr), device=device, dtype=torch.float32)
        tr_rows = idx_val[train_v]
        te_rows = idx_val[test_v]
        for slot in range(idx_val.shape[1]):
            z_tr = norm_grid[tr_rows[:, slot]]
            z_te = norm_grid[te_rows[:, slot]]
            k_tr.addmm_(z_tr, z_tr.T)
            k_te_tr.addmm_(z_te, z_tr.T)

        val_sse = np.zeros((len(alphas_18k), y_val.shape[1]), dtype=np.float64)
        for fit, val in GroupKFold(3).split(train_v, y_val[train_v], groups_val[train_v]):
            fit_t = torch.tensor(fit, device=device)
            val_t = torch.tensor(val, device=device)
            k_fit = k_tr[fit_t][:, fit_t]
            k_val_fit = k_tr[val_t][:, fit_t]
            yt = torch.from_numpy(y_val[train_v][fit]).to(device=device, dtype=torch.float32)
            preds = dual_cholesky_ridge(k_fit, yt, k_val_fit, alphas_18k)
            val_sse += ((preds - y_val[train_v][val][None, :, :]) ** 2).sum(axis=1)

        best_alpha_idx = np.argmin(val_sse, axis=0)
        yt = torch.from_numpy(y_val[train_v]).to(device=device, dtype=torch.float32)
        all_preds = dual_cholesky_ridge(k_tr, yt, k_te_tr, alphas_18k)
        for j in range(y_val.shape[1]):
            out[valid_idx[test_v], j] = all_preds[best_alpha_idx[j], :, j]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe-dir", type=Path, default=Path("logs/probe_geometry_15d_v1"))
    parser.add_argument("--teacher", type=Path, default=Path("models/rl_loop_sim_state_ts2_sdiff01_v1/sac_sim_state_60000_steps.zip"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    cache = np.load(args.probe_dir / "feature_cache.npz", allow_pickle=False)
    oof_r1 = np.load(args.probe_dir / "oof_geometry_predictions.npz", allow_pickle=False)

    grid_feat = cache["patch_grid"]
    state = cache["state"]
    history = cache["history"]
    run = cache["run"]
    layout = cache["layout"]
    model = cache["model"]
    step = cache["step"]

    y_geom = state[:, GEOM_IDX]
    groups = np.array([f"{r}/{l}" for r, l in zip(run, layout)])
    idx_mat, valid = build_lag_indices(run, layout, model, step, LAGS)
    print(f"Valid lagged frames (t, t-2, t-4): {valid.sum()} / {len(valid)} across {len(set(groups[valid]))} layouts", flush=True)

    teacher = SAC.load(str(args.teacher), device="cpu")
    true_raw_act, _ = teacher.predict(np.concatenate([state, history], axis=1), deterministic=True)
    true_exec_cmd = map_and_clip_batch(true_raw_act, history[:, -2])

    is_teacher_run = run == "sim_state_sdiff01_60k_24layouts"
    abs_cte = np.abs(state[:, 0] * 2.0)
    subsets = {
        "all_valid": valid,
        "teacher_24layouts": valid & is_teacher_run,
        "v2_abs_cte_lt_1": valid & (~is_teacher_run) & (abs_cte < 1.0),
        "v2_abs_cte_ge_1": valid & (~is_teacher_run) & (abs_cte >= 1.0),
    }

    header = (
        f"{'candidate':<21} {'subset':<18} {'fit_s':>6} "
        f"{'CTE MAE/P95':>13} {'CTE w20 P95':>11} "
        f"{'Head MAE/P95':>14} {'LA1.0 MAE/P95':>15} "
        f"{'RawSteer MAE/P95':>16} {'ExecSteer MAE/P95':>17} {'ExecThrot MAE/P95':>17}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)

    results = {"valid_frames": int(valid.sum()), "candidates": {}}
    saved_preds = {k: oof_r1[k] for k in oof_r1.files}
    saved_preds["valid_lags_mask"] = valid

    def report_candidate(name: str, elapsed: float, pred_geom: np.ndarray) -> None:
        key = name.replace(" ", "_").replace("(", "").replace(")", "")
        saved_preds[f"pred_{key}"] = pred_geom
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
                f"{name:<21} {sub_name:<18} {elapsed:>5.1f}s "
                f"{c['mae']:>5.3f}/{c['p95']:<5.3f}   {w20_str:>11} "
                f"{h['mae']:>5.2f}°/{h['p95']:<5.2f}° "
                f"{la1['euclid_mae_pos']:>6.3f}/{la1['euclid_p95_pos']:<6.3f}   "
                f"{ta['raw_steer_mae']:>6.3f}/{ta['raw_steer_p95']:<6.3f}    "
                f"{ta['exec_steer_mae']:>7.3f}/{ta['exec_steer_p95']:<7.3f}   "
                f"{ta['exec_throttle_mae']:>7.3f}/{ta['exec_throttle_p95']:<7.3f}",
                flush=True,
            )

    # 1. Single-frame baselines on the exact same `valid` mask
    report_candidate("pca128_ridge (1f)", 0.0, oof_r1["pred_pca128_ridge"])
    report_candidate("raw6144_ridge (1f)", 0.0, oof_r1["pred_raw6144_ridge"])
    report_candidate("raw6144_b64_mlp (1f)", 0.0, oof_r1["pred_raw6144_b64_mlp"])

    # 2. Lagged candidates
    t0 = time.perf_counter()
    pred_pca128_lags = cv_pca128_lags_ridge(grid_feat, y_geom, idx_mat, valid, groups, device)
    report_candidate("pca128_lags_ridge", time.perf_counter() - t0, pred_pca128_lags)

    t0 = time.perf_counter()
    pred_proj15_ridge, pred_proj15_mlp = cv_proj15_lags(grid_feat, y_geom, idx_mat, valid, groups, device)
    t_proj15 = time.perf_counter() - t0
    report_candidate("proj15_lags_ridge", t_proj15, pred_proj15_ridge)
    report_candidate("proj15_lags_mlp", t_proj15, pred_proj15_mlp)

    t0 = time.perf_counter()
    pred_shared64_mlp = cv_shared64_lags_mlp(grid_feat, y_geom, idx_mat, valid, groups, device)
    report_candidate("shared64_lags_mlp", time.perf_counter() - t0, pred_shared64_mlp)

    t0 = time.perf_counter()
    pred_raw18432_ridge = cv_raw18432_lags_ridge(grid_feat, y_geom, idx_mat, valid, groups, device)
    report_candidate("raw18432_lags_ridge", time.perf_counter() - t0, pred_raw18432_ridge)

    np.savez(args.probe_dir / "oof_geometry_predictions_lags.npz", **saved_preds)
    (args.probe_dir / "probe_geometry_lags_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nSaved lagged results to {args.probe_dir / 'probe_geometry_lags_results.json'}", flush=True)


if __name__ == "__main__":
    main()
