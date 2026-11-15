"""Stage 3.2 / 3.3: Closed-loop teacher tolerance & live linear-lag vision evaluation.

Supports paired evaluation on randomized simulator layouts across:
  - `baseline`               Exact TrackState.encode, 0 extra delay
  - `delay1_only`            1-step geometry buffer delay (motion [3:6] & history current)
  - `proj15_lags_1x_d0`      0 extra delay + 1.0x held-out residual blocks from `proj15_lags_ridge`
  - `proj15_lags_1x_d1`      1-step geom delay + 1.0x held-out residual blocks from `proj15_lags_ridge`
  - `vision_proj15_lags_live` Live DINOv2 4x4 grid + `proj15_lags_ridge.pt` (15D geom) + live motion [3:6]
  - Additional residual conditions (`shared64_lags`, `raw6144_1f`, `pca128_1f` at 1.0x/1.5x, d0/d1)

Design safeguards:
  1. All residual conditions on layout L share a single pre-sampled `(block_id, offset)`
     schedule (seeded only by `layout_idx` and `order_seed`), so `1.0x` vs `1.5x`,
     `d0` vs `d1`, and model A vs model B replay the exact same held-out trajectory
     frames at every step.
  2. Units are explicitly reported in simulator CTE units (`cte`, limit +/-2.0; ~7.98
     CTE units per position unit), degrees (`heading`), and position units (`lookahead`).
  3. Step-by-step `telemetry.jsonl`, per-episode `episodes.jsonl`, and `tolerance_results.json`
     are flushed incrementally after every episode, alongside `metadata.json` with SHA256
     hashes of all input artifacts.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import gym_donkeycar  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rl.donkey_sync import LowJitterSyncController  # noqa: E402
from rl.experiment import (  # noqa: E402
    bootstrap_mean_interval,
    paired_statistics,
    simulator_saved_settings,
    wilson_interval,
)
from rl.geometry_encoder import (  # noqa: E402
    GEOM_IDX,
    MOTION_IDX,
    SPLIT_DAGGER_TRAIN_POOL,
    SPLIT_LOCKED_EVAL_HOLDOUT,
    DinoGridLinearLag15Encoder,
    assign_layout_split,
    build_layout_block_schedule,
    extract_excursion_windows,
    load_aligned_residual_blocks,
    sha256_file,
    verify_cache_fingerprint,
)
from rl.track_state import TrackState  # noqa: E402
from rl.train_vae_sac import (  # noqa: E402
    DonkeyVaeSACEnv,
    FrozenDinoGridEncoder,
    RaffinRewardConfig,
    reload_scene,
)


class PerturbedTrackStateEncoder:
    """Wraps TrackState.encode with deterministic aligned 15D residual replay and geometry delay."""

    uses_telemetry = True
    z_size = 18

    def __init__(
        self,
        track: TrackState,
        blocks: list[dict] | None = None,
        pred_key: str | None = None,
        schedule: list[tuple[int, int]] | None = None,
        scale: float = 0.0,
        delay_geom_steps: int = 0,
    ) -> None:
        self.track = track
        self.blocks = blocks
        self.pred_key = pred_key
        self.schedule = schedule or []
        self.scale = float(scale)
        self.delay_geom_steps = int(delay_geom_steps)

        self._step_cursor: int = 0
        self._prev_geom: tuple[float, float, float, np.ndarray] | None = None
        self.last_obs_rx_mono_s: float = 0.0
        self.last_encode_done_mono_s: float = 0.0
        self.last_block_id: int = -1
        self.last_block_offset: int = -1
        self.last_block_regime: str = "none"
        self.last_eff_cte_err: float = 0.0
        self.last_eff_head_deg: float = 0.0
        self.last_eff_la1_err_pos: float = 0.0
        self.last_true_state: np.ndarray = np.zeros(18, dtype=np.float32)
        self.last_pred_state: np.ndarray = np.zeros(18, dtype=np.float32)
        self.last_true_geom: np.ndarray = np.zeros(15, dtype=np.float32)
        self.last_pred_geom: np.ndarray = np.zeros(15, dtype=np.float32)
        self.last_motion: np.ndarray = np.zeros(3, dtype=np.float32)
        self.injected_cte_errs: list[float] = []
        self.injected_head_degs: list[float] = []
        self.injected_la1_errs: list[float] = []

    def reset_episode(self, seed: int | None = None) -> None:
        self._step_cursor = 0
        self._prev_geom = None
        self.last_obs_rx_mono_s = 0.0
        self.last_encode_done_mono_s = 0.0
        self.last_block_id = -1
        self.last_block_offset = -1
        self.last_block_regime = "none"
        self.last_eff_cte_err = 0.0
        self.last_eff_head_deg = 0.0
        self.last_eff_la1_err_pos = 0.0
        self.last_true_state = np.zeros(18, dtype=np.float32)
        self.last_pred_state = np.zeros(18, dtype=np.float32)
        self.last_true_geom = np.zeros(15, dtype=np.float32)
        self.last_pred_geom = np.zeros(15, dtype=np.float32)
        self.last_motion = np.zeros(3, dtype=np.float32)
        self.injected_cte_errs.clear()
        self.injected_head_degs.clear()
        self.injected_la1_errs.clear()

    def _next_residual(self) -> tuple[float, float, np.ndarray]:
        if not self.blocks or not self.pred_key or self.scale == 0.0:
            self.last_block_id = -1
            self.last_block_offset = -1
            self.last_block_regime = "none"
            return 0.0, 0.0, np.zeros(12, dtype=np.float32)

        b_idx, off = self.schedule[self._step_cursor % len(self.schedule)]
        self._step_cursor += 1
        blk = self.blocks[b_idx]
        m_res = blk["models"][self.pred_key]
        self.last_block_id = int(blk["block_id"])
        self.last_block_offset = int(off)
        self.last_block_regime = str(blk["regime"])
        return (
            float(m_res["d_cte_half"][off]) * self.scale,
            float(m_res["d_theta"][off]) * self.scale,
            m_res["d_lookahead"][off] * self.scale,
        )

    def encode(self, image, info: dict) -> np.ndarray:
        self.last_obs_rx_mono_s = time.perf_counter()
        raw = self.track.encode(image, info).astype(np.float32)  # (18,)
        d_cte_half, d_theta, d_la = self._next_residual()

        cte_half = float(raw[0]) + d_cte_half
        theta_true = float(np.arctan2(raw[1], raw[2]))
        theta_pert = theta_true + d_theta
        sin_p, cos_p = float(np.sin(theta_pert)), float(np.cos(theta_pert))
        la_pert = (raw[6:18] + d_la).astype(np.float32)

        cur_geom = (cte_half, sin_p, cos_p, la_pert)
        if self.delay_geom_steps > 0:
            used_geom = cur_geom if self._prev_geom is None else self._prev_geom
            self._prev_geom = cur_geom
        else:
            used_geom = cur_geom

        # Effective error seen by controller (in CTE units, degrees, and pos units at 1.0 ahead)
        self.last_eff_cte_err = (used_geom[0] - float(raw[0])) * 2.0
        eff_theta = float(np.arctan2(used_geom[1], used_geom[2]))
        self.last_eff_head_deg = float(
            np.degrees(np.arctan2(np.sin(eff_theta - theta_true), np.cos(eff_theta - theta_true)))
        )
        self.last_eff_la1_err_pos = float(
            np.hypot(used_geom[3][4] - raw[10], used_geom[3][5] - raw[11]) * 1.0
        )
        self.injected_cte_errs.append(self.last_eff_cte_err)
        self.injected_head_degs.append(self.last_eff_head_deg)
        self.injected_la1_errs.append(self.last_eff_la1_err_pos)

        out = raw.copy()
        out[0] = used_geom[0]
        out[1] = used_geom[1]
        out[2] = used_geom[2]
        out[6:18] = used_geom[3]
        self.last_encode_done_mono_s = time.perf_counter()
        self.last_true_state = raw
        self.last_pred_state = out
        self.last_true_geom = raw[GEOM_IDX].astype(np.float32)
        self.last_pred_geom = out[GEOM_IDX].astype(np.float32)
        self.last_motion = raw[MOTION_IDX].astype(np.float32)
        # Motion [3:6] stays at current step `raw[3:6]`
        return out


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def _summarize_step_pool(step_pool: list[dict]) -> dict:
    if not step_pool:
        return {"steps": 0}
    abs_cte = np.asarray([s["abs_cte"] for s in step_pool], dtype=np.float32)
    cte_signed = np.asarray([s["cte"] for s in step_pool], dtype=np.float32)
    eff_cte = np.asarray([s["eff_cte_err_units"] for s in step_pool], dtype=np.float32)
    abs_eff_cte = np.abs(eff_cte)
    eff_head = np.asarray([s["eff_head_deg"] for s in step_pool], dtype=np.float32)
    abs_eff_head = np.abs(eff_head)
    eff_la1 = np.asarray([s["eff_la1_err_pos"] for s in step_pool], dtype=np.float32)
    enc_ms = np.asarray([s["encoder_ms"] for s in step_pool], dtype=np.float32)
    pol_ms = np.asarray([s.get("policy_ms", 0.0) for s in step_pool], dtype=np.float32)
    obs_act_ms = np.asarray([s.get("decision_obs_to_act_ms", 0.0) for s in step_pool], dtype=np.float32)
    dt_ms = np.asarray([s.get("step_dt_ms", 0.0) for s in step_pool], dtype=np.float32)
    steer_gap = np.asarray([s.get("steer_action_gap", 0.0) for s in step_pool], dtype=np.float32)
    throt_gap = np.asarray([s.get("throttle_action_gap", 0.0) for s in step_pool], dtype=np.float32)
    dsteer = np.abs(np.asarray([s["rl_steer_delta"] for s in step_pool], dtype=np.float32))

    return {
        "steps": int(len(step_pool)),
        "mean_abs_cte_units": float(np.mean(abs_cte)),
        "p95_abs_cte_units": float(np.percentile(abs_cte, 95)),
        "max_abs_cte_units": float(np.max(abs_cte)),
        "frames_abs_cte_ge_0_8": int(np.sum(abs_cte >= 0.8)),
        "frames_abs_cte_ge_1_0": int(np.sum(abs_cte >= 1.0)),
        "rate_limit_sat_fraction": float(np.mean(dsteer >= 0.199)),
        "eff_cte_err_mae_units": float(np.mean(abs_eff_cte)),
        "eff_cte_err_p95_units": float(np.percentile(abs_eff_cte, 95)),
        "eff_cte_err_p99_units": float(np.percentile(abs_eff_cte, 99)),
        "eff_cte_err_max_units": float(np.max(abs_eff_cte)),
        "eff_cte_signed_mean_units": float(np.mean(eff_cte)),
        "eff_head_deg_mae": float(np.mean(abs_eff_head)),
        "eff_head_deg_p95": float(np.percentile(abs_eff_head, 95)),
        "eff_head_deg_p99": float(np.percentile(abs_eff_head, 99)),
        "eff_head_deg_max": float(np.max(abs_eff_head)),
        "eff_la1_err_mae_pos": float(np.mean(eff_la1)),
        "eff_la1_err_p95_pos": float(np.percentile(eff_la1, 95)),
        "eff_la1_err_p99_pos": float(np.percentile(eff_la1, 99)),
        "steer_action_gap_mae": float(np.mean(steer_gap)),
        "steer_action_gap_p95": float(np.percentile(steer_gap, 95)),
        "steer_action_gap_max": float(np.max(steer_gap)),
        "throttle_action_gap_mae": float(np.mean(throt_gap)),
        "throttle_action_gap_p95": float(np.percentile(throt_gap, 95)),
        "encoder_ms_p50": float(np.percentile(enc_ms, 50)),
        "encoder_ms_p95": float(np.percentile(enc_ms, 95)),
        "encoder_ms_p99": float(np.percentile(enc_ms, 99)),
        "encoder_ms_max": float(np.max(enc_ms)),
        "policy_ms_p50": float(np.percentile(pol_ms, 50)),
        "policy_ms_p95": float(np.percentile(pol_ms, 95)),
        "decision_obs_to_act_ms_p50": float(np.percentile(obs_act_ms, 50)),
        "decision_obs_to_act_ms_p95": float(np.percentile(obs_act_ms, 95)),
        "decision_obs_to_act_ms_p99": float(np.percentile(obs_act_ms, 99)),
        "decision_obs_to_act_ms_max": float(np.max(obs_act_ms)),
        "step_dt_ms_p50": float(np.percentile(dt_ms, 50)),
        "step_dt_ms_p95": float(np.percentile(dt_ms, 95)),
        "step_dt_ms_p99": float(np.percentile(dt_ms, 99)),
        "step_dt_ms_max": float(np.max(dt_ms)),
        "step_dt_ms_std": float(np.std(dt_ms)),
        "bad_dt_steps": int(np.sum((dt_ms < 18.0) | (dt_ms > 32.0))),
        "bad_dt_fraction": float(np.mean((dt_ms < 18.0) | (dt_ms > 32.0))),
        "reversal_unwind_steps": int(sum(1 for s in step_pool if s.get("reversal_unwind_active", False))),
        "cte_positive_mean_err_units": float(np.mean(eff_cte[cte_signed >= 0.5])) if np.any(cte_signed >= 0.5) else 0.0,
        "cte_negative_mean_err_units": float(np.mean(eff_cte[cte_signed <= -0.5])) if np.any(cte_signed <= -0.5) else 0.0,
    }


def summarize_episodes(
    condition_names: list[str],
    all_episodes: list[dict],
    step_pools_by_cond: dict[str, list[dict]] | None = None,
) -> dict:
    summary: dict = {}
    for cname in condition_names:
        rows = [r for r in all_episodes if r["condition"] == cname]
        if not rows:
            continue
        done = int(sum(1 for r in rows if r["completed"]))
        n_ep = len(rows)
        steps_list = [r["steps"] for r in rows]
        laps_list = [r["laps"] for r in rows]
        mean_cte_list = [r["mean_abs_cte"] for r in rows]
        max_cte_list = [r["max_abs_cte"] for r in rows]
        spd_list = [r["mean_speed"] for r in rows]

        cond_entry: dict = {
            "completed": done,
            "episodes": n_ep,
            "completion_rate": float(done / n_ep),
            "completion_rate_wilson_ci95": list(wilson_interval(done, n_ep)),
            "mean_steps": float(np.mean(steps_list)),
            "mean_steps_ci95": list(bootstrap_mean_interval(steps_list)),
            "mean_laps": float(np.mean(laps_list)),
            "mean_laps_ci95": list(bootstrap_mean_interval(laps_list)),
            "mean_abs_cte_units": float(np.mean(mean_cte_list)),
            "mean_abs_cte_ci95_units": list(bootstrap_mean_interval(mean_cte_list)),
            "mean_episode_max_abs_cte_units": float(np.mean(max_cte_list)),
            "max_abs_cte_units": float(np.max(max_cte_list)),
            "mean_speed": float(np.mean(spd_list)),
            "mean_speed_ci95": list(bootstrap_mean_interval(spd_list)),
            # Episode-averaged P95s (explicitly labeled to distinguish from pooled step P95)
            "mean_eff_cte_p95_units": float(np.mean([r["eff_cte_err_p95_units"] for r in rows])),
            "mean_eff_head_p95_deg": float(np.mean([r["eff_head_deg_p95"] for r in rows])),
            "mean_eff_la1_p95_pos": float(np.mean([r["eff_la1_err_p95_pos"] for r in rows])),
            "encoder_ms_p50": float(np.mean([r["encoder_ms_p50"] for r in rows])),
            "encoder_ms_p95": float(np.mean([r["encoder_ms_p95"] for r in rows])),
            "max_encoder_ms_p95": float(np.max([r["encoder_ms_p95"] for r in rows])),
            "mean_decision_obs_to_act_ms_p95": float(np.mean([r.get("decision_obs_to_act_ms_p95", 0.0) for r in rows])),
            "mean_step_dt_ms_p95": float(np.mean([r.get("step_dt_ms_p95", 0.0) for r in rows])),
            "bad_dt_steps_total": int(sum(r.get("bad_dt_steps", 0) for r in rows)),
            "coalesced_dropped_total": int(sum(r.get("coalesced_dropped", 0) for r in rows)),
            "max_burst_seen": int(max((r.get("max_burst_seen", 1) for r in rows), default=1)),
            "reversal_unwind_steps_total": int(sum(r.get("reversal_unwind_steps", 0) for r in rows)),
        }
        ep_excursions = [exc for r in rows for exc in r.get("excursions", [])]
        rec_durations = [exc["duration_steps"] for exc in ep_excursions if exc.get("recovered")]
        cond_entry["excursion_count"] = len(ep_excursions)
        cond_entry["excursion_recovered_count"] = int(sum(1 for exc in ep_excursions if exc.get("recovered")))
        cond_entry["excursion_recovery_rate"] = (
            float(cond_entry["excursion_recovered_count"] / len(ep_excursions)) if ep_excursions else None
        )
        cond_entry["excursion_median_recovery_steps"] = (
            float(np.median(rec_durations)) if rec_durations else None
        )

        if step_pools_by_cond and cname in step_pools_by_cond:
            pool = step_pools_by_cond[cname]
            cond_entry["pooled_all_steps"] = _summarize_step_pool(pool)
            cond_entry["regime_breakdown"] = {
                "standing_start_1_250": _summarize_step_pool([s for s in pool if s["step"] <= 250]),
                "cruise_abs_cte_lt_0_5": _summarize_step_pool([s for s in pool if s["abs_cte"] < 0.5]),
                "moderate_offset_0_5_to_1_0": _summarize_step_pool([s for s in pool if 0.5 <= s["abs_cte"] < 1.0]),
                "large_offset_ge_1_0": _summarize_step_pool([s for s in pool if s["abs_cte"] >= 1.0]),
            }

        # Breakdown by pre-registered split (dagger_train_pool vs locked_eval_holdout)
        by_split: dict = {}
        for split_name in (SPLIT_DAGGER_TRAIN_POOL, SPLIT_LOCKED_EVAL_HOLDOUT):
            s_rows = [r for r in rows if r.get("split") == split_name]
            if not s_rows:
                continue
            s_done = int(sum(1 for r in s_rows if r["completed"]))
            by_split[split_name] = {
                "completed": s_done,
                "episodes": len(s_rows),
                "completion_rate": float(s_done / len(s_rows)),
                "completion_rate_wilson_ci95": list(wilson_interval(s_done, len(s_rows))),
                "mean_steps": float(np.mean([r["steps"] for r in s_rows])),
                "mean_laps": float(np.mean([r["laps"] for r in s_rows])),
                "mean_abs_cte_units": float(np.mean([r["mean_abs_cte"] for r in s_rows])),
                "max_abs_cte_units": float(np.max([r["max_abs_cte"] for r in s_rows])),
                "mean_speed": float(np.mean([r["mean_speed"] for r in s_rows])),
            }
            if step_pools_by_cond and cname in step_pools_by_cond:
                s_pool = [s for s in step_pools_by_cond[cname] if s.get("split") == split_name]
                by_split[split_name]["pooled_steps"] = _summarize_step_pool(s_pool)
        if by_split:
            cond_entry["by_split"] = by_split

        summary[cname] = cond_entry
    return summary


def _paired_trajectory_delta(eps: list[dict], ref_name: str, cand_name: str) -> dict | None:
    ref_by_l = {e["layout"]: e for e in eps if e["condition"] == ref_name}
    cand_by_l = {e["layout"]: e for e in eps if e["condition"] == cand_name}
    shared = sorted(ref_by_l.keys() & cand_by_l.keys())
    if not shared:
        return None
    cte_diffs = [cand_by_l[k]["mean_abs_cte"] - ref_by_l[k]["mean_abs_cte"] for k in shared]
    max_cte_diffs = [cand_by_l[k]["max_abs_cte"] - ref_by_l[k]["max_abs_cte"] for k in shared]
    spd_diffs = [cand_by_l[k]["mean_speed"] - ref_by_l[k]["mean_speed"] for k in shared]
    ref_mean_cte = float(np.mean([ref_by_l[k]["mean_abs_cte"] for k in shared]))
    cand_mean_cte = float(np.mean([cand_by_l[k]["mean_abs_cte"] for k in shared]))
    return {
        "candidate": cand_name,
        "reference": ref_name,
        "shared_layouts": len(shared),
        "mean_abs_cte_diff_units": float(np.mean(cte_diffs)),
        "mean_abs_cte_diff_ci95_units": list(bootstrap_mean_interval(cte_diffs)),
        "mean_abs_cte_ratio": float(cand_mean_cte / max(ref_mean_cte, 1e-8)),
        "max_abs_cte_diff_units": float(np.mean(max_cte_diffs)),
        "max_abs_cte_diff_ci95_units": list(bootstrap_mean_interval(max_cte_diffs)),
        "mean_speed_diff": float(np.mean(spd_diffs)),
        "mean_speed_diff_ci95": list(bootstrap_mean_interval(spd_diffs)),
    }


def compute_paired_vs_baseline(all_episodes: list[dict], condition_names: list[str]) -> dict:
    """Compute paired layout-by-layout comparisons (overall and per split)."""
    if "baseline" not in condition_names or len(condition_names) < 2:
        return {}
    out: dict = {}
    scopes = [
        ("all_layouts", all_episodes),
        (SPLIT_DAGGER_TRAIN_POOL, [e for e in all_episodes if e.get("split") == SPLIT_DAGGER_TRAIN_POOL]),
        (SPLIT_LOCKED_EVAL_HOLDOUT, [e for e in all_episodes if e.get("split") == SPLIT_LOCKED_EVAL_HOLDOUT]),
    ]
    for scope_name, eps in scopes:
        if not eps:
            continue
        formatted_for_exp = [
            {
                "model": e["condition"],
                "layout": e["layout"],
                "steps": e["steps"],
                "laps": e["laps"],
                "trunc": bool(e["completed"]),
            }
            for e in eps
        ]
        p_stats = paired_statistics(formatted_for_exp)
        extra_pairs = []
        for cname in condition_names:
            if cname == "baseline":
                continue
            pair = _paired_trajectory_delta(eps, "baseline", cname)
            if pair is not None:
                pair["mean_abs_cte_ratio_vs_baseline"] = pair.pop("mean_abs_cte_ratio")
                extra_pairs.append(pair)
        p_stats["trajectory_deltas_vs_baseline"] = extra_pairs

        control_pairs = []
        for ref_c, cand_c in (
            ("baseline", "teacher_sync_only"),
            ("teacher_sync_only", "teacher_fixed"),
            ("vision_baseline", "vision_fixed_ctrl"),
            ("baseline", "vision_baseline"),
            ("teacher_fixed", "vision_fixed_ctrl"),
        ):
            if ref_c in condition_names and cand_c in condition_names:
                pair = _paired_trajectory_delta(eps, ref_c, cand_c)
                if pair is not None:
                    control_pairs.append(pair)
        if control_pairs:
            p_stats["paired_control_deltas"] = control_pairs
        out[scope_name] = p_stats
    return out


def build_conditions(mode: str) -> tuple[list[tuple[str, str | None, float, int]], list[str]]:
    """Return (condition_specs, required_pred_keys).

    Each condition_spec is `(name, pred_key_or_live, scale, delay_geom_steps)`.
    When `pred_key_or_live == "__LIVE_VISION__"`, `DinoGridLinearLag15Encoder` is used.
    """
    if mode == "round_a_vision_paired":
        specs = [
            ("baseline", None, 0.0, 0),
            ("vision_proj15_lags_live", "__LIVE_VISION__", 0.0, 0),
        ]
    elif mode == "vision_and_d0":
        specs = [
            ("baseline", None, 0.0, 0),
            ("proj15_lags_1x_d0", "pred_proj15_lags_ridge", 1.0, 0),
            ("proj15_lags_1x_d1", "pred_proj15_lags_ridge", 1.0, 1),
            ("vision_proj15_lags_live", "__LIVE_VISION__", 0.0, 0),
        ]
    elif mode == "tolerance_d0_vs_d1":
        specs = [
            ("baseline", None, 0.0, 0),
            ("delay1_only", None, 0.0, 1),
            ("proj15_lags_1x_d0", "pred_proj15_lags_ridge", 1.0, 0),
            ("proj15_lags_1x_d1", "pred_proj15_lags_ridge", 1.0, 1),
            ("pca128_1f_1x_d0", "pred_pca128_ridge_1f", 1.0, 0),
            ("pca128_1f_1x_d1", "pred_pca128_ridge_1f", 1.0, 1),
            ("pca128_1f_1.5x_d1", "pred_pca128_ridge_1f", 1.5, 1),
            ("vision_proj15_lags_live", "__LIVE_VISION__", 0.0, 0),
        ]
    elif mode == "teacher_fix_paired":
        specs = [
            ("baseline", None, 0.0, 0),
            ("teacher_sync_only", None, 0.0, 0),
            ("teacher_fixed", None, 0.0, 0),
            ("vision_fixed_ctrl", "__LIVE_VISION__", 0.0, 0),
        ]
    elif mode == "teacher_fix_paired_full":
        specs = [
            ("baseline", None, 0.0, 0),
            ("teacher_sync_only", None, 0.0, 0),
            ("teacher_fixed", None, 0.0, 0),
            ("vision_baseline", "__LIVE_VISION__", 0.0, 0),
            ("vision_fixed_ctrl", "__LIVE_VISION__", 0.0, 0),
        ]
    elif mode == "teacher_fix_recovery_stress":
        specs = [
            ("baseline", None, 0.0, 0),
            ("teacher_sync_only", None, 0.0, 0),
            ("teacher_fixed", None, 0.0, 0),
            ("vision_baseline", "__LIVE_VISION__", 0.0, 0),
            ("vision_fixed_ctrl", "__LIVE_VISION__", 0.0, 0),
        ]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    pred_keys = sorted({k for _, k, _, _ in specs if k is not None and k != "__LIVE_VISION__"})
    return specs, pred_keys


def condition_control_config(condition_name: str) -> dict:
    """Return control-loop sync and bounded reversal unwind parameters per condition."""
    if condition_name in ("baseline", "vision_baseline"):
        return {"sync_enabled": False, "reversal_unwind_scale": 1.0}
    if condition_name == "teacher_sync_only":
        return {"sync_enabled": True, "reversal_unwind_scale": 1.0}
    if condition_name in ("teacher_fixed", "vision_fixed_ctrl"):
        return {"sync_enabled": True, "reversal_unwind_scale": 2.0}
    return {"sync_enabled": False, "reversal_unwind_scale": 1.0}


def apply_recovery_stress(
    action: np.ndarray,
    next_step: int,
    layout_idx: int,
    stress_type: str = "ctrl_stall",
) -> tuple[np.ndarray, float, str | None]:
    """Deterministic paired lateral kick + stall/TCP-backlog schedule for stress testing.

    Returns `(action_to_step, stall_or_pause_sec, disturbance_tag)`:
    - Steps 320..322: 3-step steering kick (`+/-0.70` alternating by layout parity).
    - Step 900 (`ctrl_stall`): `95 ms` control-thread `sleep` while recv thread runs.
    - Step 900 (`tcp_backlog`): `95 ms` real TCP socket `recv()` pause so packets queue in OS buffer.
    - `both`: `stall_95ms` at step 600 + `tcp_recv_pause_95ms` at step 900 (plus both kicks).
    - Steps 1300..1302: 3-step steering kick in the opposite direction.
    """
    act = np.array(action, dtype=np.float32, copy=True)
    sign_1 = 1.0 if (layout_idx % 2 == 1) else -1.0
    if 320 <= next_step <= 322:
        act[0] = sign_1 * 0.70
        return act, 0.0, "kick_1"
    if stress_type == "ctrl_stall" and next_step == 900:
        return act, 0.095, "stall_95ms"
    if stress_type == "tcp_backlog" and next_step == 900:
        return act, 0.095, "tcp_recv_pause_95ms"
    if stress_type == "both":
        if next_step == 600:
            return act, 0.095, "stall_95ms"
        if next_step == 900:
            return act, 0.095, "tcp_recv_pause_95ms"
    if 1300 <= next_step <= 1302:
        act[0] = -sign_1 * 0.70
        return act, 0.0, "kick_2"
    return act, 0.0, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="172.25.240.1")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument(
        "--mode",
        choices=(
            "round_a_vision_paired",
            "vision_and_d0",
            "tolerance_d0_vs_d1",
            "teacher_fix_paired",
            "teacher_fix_paired_full",
            "teacher_fix_recovery_stress",
        ),
        default="vision_and_d0",
    )
    parser.add_argument(
        "--stress-type",
        choices=("ctrl_stall", "tcp_backlog", "both"),
        default="ctrl_stall",
        help="Disturbance type at stall window when --mode=teacher_fix_recovery_stress.",
    )
    parser.add_argument("--regime", choices=("all", "cruising", "corner", "high_cte"), default="all")
    parser.add_argument("--teacher", type=Path, default=Path("models/rl_loop_sim_state_ts2_sdiff01_v1/sac_sim_state_60000_steps.zip"))
    parser.add_argument("--track-centerline", type=Path, default=Path("models/track_centerline/generated_track.npz"))
    parser.add_argument("--probe-dir", type=Path, default=Path("logs/probe_geometry_15d_v1"))
    parser.add_argument("--geometry-head", type=Path, default=Path("models/dinov2_geometry_15d/proj15_lags_ridge.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/vision_proj15_and_d0_v1"))
    parser.add_argument("--layouts", type=int, default=4)
    parser.add_argument(
        "--train-pool-layouts",
        type=int,
        default=None,
        help="First K layouts are pre-assigned to dagger_train_pool; layouts K+1..N are locked_eval_holdout (default: N // 2).",
    )
    parser.add_argument(
        "--save-captures",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save full-episode camera frames + 15D labels + motion + history + timestamps to captures/*.npz (default: True for round_a_vision_paired).",
    )
    parser.add_argument("--max-episode-steps", type=int, default=2000)
    parser.add_argument("--block-len", type=int, default=80)
    parser.add_argument("--order-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    save_captures = (args.mode == "round_a_vision_paired") if args.save_captures is None else bool(args.save_captures)
    train_pool_cutoff = (args.layouts // 2) if args.train_pool_layouts is None else int(args.train_pool_layouts)
    captures_dir = args.output_dir / "captures"
    if save_captures:
        captures_dir.mkdir(parents=True, exist_ok=True)

    cache_meta = verify_cache_fingerprint(args.probe_dir / "feature_cache.npz")
    track = TrackState.load(args.track_centerline)
    condition_specs, pred_keys = build_conditions(args.mode)
    condition_names = [c[0] for c in condition_specs]

    blocks = (
        load_aligned_residual_blocks(
            args.probe_dir,
            pred_keys=pred_keys,
            min_block_len=args.block_len,
            regime_filter=args.regime,
        )
        if pred_keys
        else []
    )

    # Precompute one shared residual schedule per layout so all conditions on layout L
    # replay the exact same `(block_id, offset)` sequence.
    layout_schedules: dict[int, list[tuple[int, int]]] = {}
    if blocks:
        for layout_idx in range(1, args.layouts + 1):
            layout_schedules[layout_idx] = build_layout_block_schedule(
                blocks=blocks,
                total_steps=args.max_episode_steps + 10,
                block_len=args.block_len,
                seed=args.order_seed * 10000 + layout_idx,
            )

    uses_live_vision = any(k == "__LIVE_VISION__" for _, k, _, _ in condition_specs)
    live_vision_enc: DinoGridLinearLag15Encoder | None = None
    if uses_live_vision:
        dino_enc = FrozenDinoGridEncoder(device=device, crop_top=0, projection=None)
        live_vision_enc = DinoGridLinearLag15Encoder(
            head_path=args.geometry_head,
            track=track,
            device=device,
            dino_encoder=dino_enc,
        )
        # Warm up GPU once before connecting/stepping so first-step latency is steady-state
        dummy_img = np.zeros((120, 160, 3), dtype=np.uint8)
        _ = dino_enc.grid_features(dummy_img[None])
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    split_plan = {
        "rule": (
            f"Layouts 1..{train_pool_cutoff} -> {SPLIT_DAGGER_TRAIN_POOL}; "
            f"Layouts {train_pool_cutoff + 1}..{args.layouts} -> {SPLIT_LOCKED_EVAL_HOLDOUT}"
        ),
        "train_pool_cutoff": train_pool_cutoff,
        "dagger_train_pool_layouts": list(range(1, train_pool_cutoff + 1)),
        "locked_eval_holdout_layouts": list(range(train_pool_cutoff + 1, args.layouts + 1)),
        "policy": (
            "Pre-registered before Round A execution: any data from dagger_train_pool used in "
            "retraining is excluded from final evaluation; locked_eval_holdout is never used for training."
        ),
    }

    metadata = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": get_git_commit(),
        "mode": args.mode,
        "stress_type": args.stress_type if args.mode == "teacher_fix_recovery_stress" else None,
        "regime_filter": args.regime,
        "layouts": args.layouts,
        "max_episode_steps": args.max_episode_steps,
        "block_len": args.block_len,
        "order_seed": args.order_seed,
        "save_captures": save_captures,
        "split_plan": split_plan,
        "simulator_saved_settings": simulator_saved_settings(),
        "units": {
            "cte": "simulator CTE units (limit +/-2.0; ~7.98 CTE units per position unit)",
            "heading": "degrees",
            "lookahead": "simulator position units",
            "encoder_ms": "milliseconds per step (full encoder.encode call)",
            "decision_obs_to_act_ms": "milliseconds from observation arrival (Frame k-1) to command dispatch (Action k)",
            "step_dt_ms": "milliseconds between consecutive observation arrivals (Frame k-1 -> Frame k)",
        },
        "artifacts": {
            "teacher": str(args.teacher),
            "teacher_sha256": sha256_file(args.teacher),
            "track_centerline": str(args.track_centerline),
            "track_centerline_sha256": sha256_file(args.track_centerline),
            "oof_predictions": str(args.probe_dir / "oof_geometry_predictions_lags.npz"),
            "oof_predictions_sha256": sha256_file(args.probe_dir / "oof_geometry_predictions_lags.npz"),
            "cache_fingerprint_sha256": cache_meta.get("fingerprint_sha256"),
            "geometry_head": str(args.geometry_head) if uses_live_vision else None,
            "geometry_head_sha256": live_vision_enc.head_sha256 if live_vision_enc is not None else None,
        },
        "n_aligned_blocks": len(blocks),
        "conditions": condition_names,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    telemetry_path = args.output_dir / "telemetry.jsonl"
    episodes_path = args.output_dir / "episodes.jsonl"
    results_path = args.output_dir / "tolerance_results.json"
    split_manifest_path = args.output_dir / "split_manifest.json"
    telemetry_path.write_text("")
    episodes_path.write_text("")

    teacher = SAC.load(str(args.teacher), device="cpu")
    dummy_enc = PerturbedTrackStateEncoder(track)

    base_env = gym.make(
        "donkey-generated-track-v0",
        conf={
            "host": args.host,
            "port": args.port,
            "body_style": "donkey",
            "body_rgb": (128, 128, 128),
            "car_name": "geom_eval",
            "font_size": 100,
            "cam_resolution": (120, 160, 3),
            "cam_offset": (0, 0, 0),
            "max_cte": 20.0,
        },
    )
    env = DonkeyVaeSACEnv(
        base_env,
        vae=dummy_enc,
        reward_config=RaffinRewardConfig(
            reward_crash=-20.0,
            crash_speed_weight=5.0,
            alive_reward=1.5,
            speed_reward_weight=0.15,
            cte_speed_penalty_weight=0.25,
            max_cte_error=2.0,
            cte_target=0.0,
        ),
        min_throttle=0.2,
        max_throttle=0.7,
        max_steering=1.0,
        max_steering_diff=0.1,
        n_command_history=20,
        scene_reload_every=0,
    )

    rng = np.random.default_rng(args.order_seed)
    all_episodes: list[dict] = []
    step_pools_by_cond: dict[str, list[dict]] = {c: [] for c in condition_names}
    all_excursions: list[dict] = []
    manifest_files: dict[str, list[str]] = {
        SPLIT_DAGGER_TRAIN_POOL: [],
        SPLIT_LOCKED_EVAL_HOLDOUT: [],
    }
    controller = env.env.unwrapped.viewer
    sync_controller = LowJitterSyncController(controller, enabled=False).install()
    scene = controller.handler.SceneToLoad
    ep_counter = 0

    try:
        for layout_idx in range(1, args.layouts + 1):
            split_name = assign_layout_split(
                layout_idx,
                total_layouts=args.layouts,
                train_pool_layouts=train_pool_cutoff,
            )
            print(
                f"\n=== Reloading scene for Layout {layout_idx}/{args.layouts} [split={split_name}] ===",
                flush=True,
            )
            if not reload_scene(controller, scene):
                raise RuntimeError(f"Scene reload timed out on layout {layout_idx}")

            order = list(range(len(condition_specs)))
            rng.shuffle(order)

            for cond_idx in order:
                ep_counter += 1
                cname, pkey, scale, d_steps = condition_specs[cond_idx]
                ctrl_cfg = condition_control_config(cname)
                sync_controller.set_enabled(ctrl_cfg["sync_enabled"])
                env.set_control_fixes(reversal_unwind_scale=ctrl_cfg["reversal_unwind_scale"])
                if pkey == "__LIVE_VISION__":
                    assert live_vision_enc is not None
                    enc = live_vision_enc
                    enc.reset_episode()
                else:
                    enc = PerturbedTrackStateEncoder(
                        track=track,
                        blocks=blocks,
                        pred_key=pkey,
                        schedule=layout_schedules.get(layout_idx, []),
                        scale=scale,
                        delay_geom_steps=d_steps,
                    )
                    enc.reset_episode()
                env.set_observation(enc, latent_lags=())

                obs, info = env.reset()
                sync_controller.reset_stats()
                init_image_0 = np.array(env.last_raw_obs, dtype=np.uint8, copy=True)
                init_true_state_0 = enc.last_true_state.copy()
                init_pred_geom_0 = enc.last_pred_geom.copy()
                prev_obs_rx_s = float(enc.last_obs_rx_mono_s)
                prev_encode_done_s = float(enc.last_encode_done_mono_s)

                steps = 0
                total_reward = 0.0
                ctes_signed: list[float] = []
                abs_ctes: list[float] = []
                speeds: list[float] = []
                steers: list[float] = []
                dsteers_signed: list[float] = []
                dsteers: list[float] = []
                throttles: list[float] = []
                eff_cte_errs: list[float] = []
                eff_head_degs: list[float] = []
                eff_la1_errs: list[float] = []
                enc_ms_list: list[float] = []
                pol_ms_list: list[float] = []
                obs_to_act_ms_list: list[float] = []
                sim_wait_ms_list: list[float] = []
                step_dt_ms_list: list[float] = []

                # In-RAM full-episode buffers (flushed after episode ends; zero in-loop I/O)
                ep_images: list[np.ndarray] = []
                ep_true_state: list[np.ndarray] = []
                ep_pred_state: list[np.ndarray] = []
                ep_true_geom: list[np.ndarray] = []
                ep_pred_geom: list[np.ndarray] = []
                ep_motion: list[np.ndarray] = []
                ep_hist_pre: list[np.ndarray] = []
                ep_hist_post: list[np.ndarray] = []
                ep_raw_act: list[np.ndarray] = []
                ep_exec_cmd: list[np.ndarray] = []
                ep_pos: list[list[float]] = []
                ep_car: list[list[float]] = []
                ep_laps: list[int] = []
                ep_dec_obs_rx_s: list[float] = []
                ep_dec_enc_done_s: list[float] = []
                ep_act_sent_s: list[float] = []
                ep_frame_rx_s: list[float] = []
                ep_frame_enc_done_s: list[float] = []
                ep_step_dicts: list[dict] = []

                t0 = time.perf_counter()
                term_reason = "truncated"

                while steps < args.max_episode_steps:
                    decision_obs_rx_s = prev_obs_rx_s
                    decision_encode_done_s = prev_encode_done_s
                    hist_pre = env.command_history.reshape(-1).copy()

                    t_pol0 = time.perf_counter()
                    action, _ = teacher.predict(obs, deterministic=True)
                    disturb_tag = None
                    if args.mode == "teacher_fix_recovery_stress":
                        action, stress_dur_s, disturb_tag = apply_recovery_stress(
                            action, steps + 1, layout_idx, stress_type=args.stress_type
                        )
                        if disturb_tag == "stall_95ms" and stress_dur_s > 0.0:
                            time.sleep(stress_dur_s)
                        elif disturb_tag == "tcp_recv_pause_95ms" and stress_dur_s > 0.0:
                            sync_controller.pause_recv(stress_dur_s)
                    t_act_sent = time.perf_counter()

                    obs, reward, terminated, truncated, info = env.step(action)
                    steps += 1
                    total_reward += float(reward)

                    frame_rx_s = float(enc.last_obs_rx_mono_s)
                    frame_encode_done_s = float(enc.last_encode_done_mono_s)
                    prev_obs_rx_s = frame_rx_s
                    prev_encode_done_s = frame_encode_done_s

                    cte_val = float(info.get("cte", 0.0))
                    abs_cte = float(info["abs_cte"])
                    spd = float(info.get("speed", 0.0))
                    rl_steer = float(info["rl_steer"])
                    rl_dsteer = float(info["rl_steer_delta"])
                    rl_throt = float(info["rl_throttle"])
                    dst = abs(rl_dsteer)
                    enc_ms = float(info.get("encoder_seconds", 0.0)) * 1000.0
                    pol_ms = (t_act_sent - t_pol0) * 1000.0
                    obs_to_act_ms = (t_act_sent - decision_obs_rx_s) * 1000.0
                    sim_wait_ms = (frame_rx_s - t_act_sent) * 1000.0
                    step_dt_ms = (frame_rx_s - decision_obs_rx_s) * 1000.0

                    eff_cte_u = float(enc.last_eff_cte_err)
                    eff_head_d = float(enc.last_eff_head_deg)
                    eff_la1_p = float(enc.last_eff_la1_err_pos)
                    pos_xyz = [float(x) for x in info.get("pos", (0.0, 0.0, 0.0))]
                    car_pyr = [float(x) for x in info.get("car", (0.0, 0.0, 0.0))]
                    lap_cnt = int(info.get("lap_count", 0))

                    ctes_signed.append(cte_val)
                    abs_ctes.append(abs_cte)
                    speeds.append(spd)
                    steers.append(rl_steer)
                    dsteers_signed.append(rl_dsteer)
                    dsteers.append(dst)
                    throttles.append(rl_throt)
                    eff_cte_errs.append(eff_cte_u)
                    eff_head_degs.append(eff_head_d)
                    eff_la1_errs.append(eff_la1_p)
                    enc_ms_list.append(enc_ms)
                    pol_ms_list.append(pol_ms)
                    obs_to_act_ms_list.append(obs_to_act_ms)
                    sim_wait_ms_list.append(sim_wait_ms)
                    step_dt_ms_list.append(step_dt_ms)

                    ep_true_state.append(enc.last_true_state)
                    ep_pred_state.append(enc.last_pred_state)
                    ep_true_geom.append(enc.last_true_geom)
                    ep_pred_geom.append(enc.last_pred_geom)
                    ep_motion.append(enc.last_motion)
                    ep_hist_pre.append(hist_pre)
                    ep_hist_post.append(env.command_history.reshape(-1).copy())
                    ep_raw_act.append(np.asarray(action, dtype=np.float32).reshape(2))
                    ep_exec_cmd.append(np.array([rl_steer, rl_throt], dtype=np.float32))
                    ep_pos.append(pos_xyz)
                    ep_car.append(car_pyr)
                    ep_laps.append(lap_cnt)
                    ep_dec_obs_rx_s.append(decision_obs_rx_s)
                    ep_dec_enc_done_s.append(decision_encode_done_s)
                    ep_act_sent_s.append(t_act_sent)
                    ep_frame_rx_s.append(frame_rx_s)
                    ep_frame_enc_done_s.append(frame_encode_done_s)
                    if save_captures:
                        ep_images.append(env.last_raw_obs)

                    ep_step_dicts.append({
                        "episode": ep_counter,
                        "layout": layout_idx,
                        "split": split_name,
                        "condition": cname,
                        "step": steps,
                        "cte": cte_val,
                        "abs_cte": abs_cte,
                        "speed": spd,
                        "pos": pos_xyz,
                        "car": car_pyr,
                        "lap_count": lap_cnt,
                        "rl_steer": rl_steer,
                        "rl_steer_delta": rl_dsteer,
                        "rl_throttle": rl_throt,
                        "eff_cte_err_units": eff_cte_u,
                        "eff_head_deg": eff_head_d,
                        "eff_la1_err_pos": eff_la1_p,
                        "block_id": int(getattr(enc, "last_block_id", -1)),
                        "block_offset": int(getattr(enc, "last_block_offset", -1)),
                        "block_regime": str(getattr(enc, "last_block_regime", "live")),
                        "encoder_ms": enc_ms,
                        "policy_ms": pol_ms,
                        "decision_obs_to_act_ms": obs_to_act_ms,
                        "sim_wait_ms": sim_wait_ms,
                        "step_dt_ms": step_dt_ms,
                        "reversal_unwind_active": bool(info.get("reversal_unwind_active", False)),
                        "disturbance": disturb_tag,
                    })

                    if terminated:
                        term_reason = str(info.get("termination_reason") or "terminated")
                        break
                    if truncated:
                        break

                elapsed = time.perf_counter() - t0
                laps = int(info.get("lap_count", 0))
                completed = steps >= args.max_episode_steps and term_reason == "truncated"

                # Post-episode batched counterfactual teacher query (zero in-loop latency!)
                true_state_arr = np.stack(ep_true_state, axis=0)
                pred_state_arr = np.stack(ep_pred_state, axis=0)
                hist_post_arr = np.stack(ep_hist_post, axis=0)
                teacher_true_act, _ = teacher.predict(
                    np.concatenate([true_state_arr, hist_post_arr], axis=1),
                    deterministic=True,
                )
                student_next_act, _ = teacher.predict(
                    np.concatenate([pred_state_arr, hist_post_arr], axis=1),
                    deterministic=True,
                )
                steer_gaps = np.abs(student_next_act[:, 0] - teacher_true_act[:, 0]).astype(np.float32)
                throt_gaps = np.abs(student_next_act[:, 1] - teacher_true_act[:, 1]).astype(np.float32)
                for idx_s, s_dict in enumerate(ep_step_dicts):
                    s_dict["steer_action_gap"] = float(steer_gaps[idx_s])
                    s_dict["throttle_action_gap"] = float(throt_gaps[idx_s])
                step_pools_by_cond[cname].extend(ep_step_dicts)

                excursions = extract_excursion_windows(
                    cte=np.asarray(ctes_signed, dtype=np.float32),
                    eff_cte_err=np.asarray(eff_cte_errs, dtype=np.float32),
                    eff_head_deg=np.asarray(eff_head_degs, dtype=np.float32),
                    rl_steer=np.asarray(steers, dtype=np.float32),
                    rl_steer_delta=np.asarray(dsteers_signed, dtype=np.float32),
                    rl_throttle=np.asarray(throttles, dtype=np.float32),
                )
                for exc in excursions:
                    all_excursions.append({
                        "episode": ep_counter,
                        "layout": layout_idx,
                        "split": split_name,
                        "condition": cname,
                        "episode_completed": completed,
                        **exc,
                    })

                capture_rel_path: str | None = None
                if save_captures:
                    cap_file = captures_dir / f"layout_{layout_idx:02d}_{cname}_{split_name}.npz"
                    np.savez(
                        cap_file,
                        split=np.array(split_name),
                        layout=np.int32(layout_idx),
                        episode=np.int32(ep_counter),
                        condition=np.array(cname),
                        completed=np.bool_(completed),
                        termination_reason=np.array(term_reason),
                        step=np.arange(1, steps + 1, dtype=np.int32),
                        init_image_0=init_image_0,
                        init_true_state_0=init_true_state_0,
                        init_pred_geom_0=init_pred_geom_0,
                        images=np.stack(ep_images, axis=0).astype(np.uint8),
                        true_state=true_state_arr,
                        pred_state=pred_state_arr,
                        true_geom=np.stack(ep_true_geom, axis=0),
                        pred_geom=np.stack(ep_pred_geom, axis=0),
                        motion=np.stack(ep_motion, axis=0),
                        history_pre=np.stack(ep_hist_pre, axis=0),
                        history_post=hist_post_arr,
                        student_raw_action=np.stack(ep_raw_act, axis=0),
                        student_exec_cmd=np.stack(ep_exec_cmd, axis=0),
                        student_next_raw_action_on_frame=student_next_act.astype(np.float32),
                        teacher_true_raw_action_on_frame=teacher_true_act.astype(np.float32),
                        cte=np.asarray(ctes_signed, dtype=np.float32),
                        speed=np.asarray(speeds, dtype=np.float32),
                        pos=np.asarray(ep_pos, dtype=np.float32),
                        car=np.asarray(ep_car, dtype=np.float32),
                        lap_count=np.asarray(ep_laps, dtype=np.int32),
                        eff_cte_err_units=np.asarray(eff_cte_errs, dtype=np.float32),
                        eff_head_deg=np.asarray(eff_head_degs, dtype=np.float32),
                        eff_la1_err_pos=np.asarray(eff_la1_errs, dtype=np.float32),
                        decision_obs_rx_mono_s=np.asarray(ep_dec_obs_rx_s, dtype=np.float64),
                        decision_encode_done_mono_s=np.asarray(ep_dec_enc_done_s, dtype=np.float64),
                        act_sent_mono_s=np.asarray(ep_act_sent_s, dtype=np.float64),
                        frame_rx_mono_s=np.asarray(ep_frame_rx_s, dtype=np.float64),
                        frame_encode_done_mono_s=np.asarray(ep_frame_enc_done_s, dtype=np.float64),
                        encoder_ms=np.asarray(enc_ms_list, dtype=np.float32),
                        policy_ms=np.asarray(pol_ms_list, dtype=np.float32),
                        decision_obs_to_act_ms=np.asarray(obs_to_act_ms_list, dtype=np.float32),
                        sim_wait_ms=np.asarray(sim_wait_ms_list, dtype=np.float32),
                        step_dt_ms=np.asarray(step_dt_ms_list, dtype=np.float32),
                    )
                    capture_rel_path = f"captures/{cap_file.name}"
                    manifest_files[split_name].append(capture_rel_path)
                    split_manifest_path.write_text(
                        json.dumps(
                            {
                                "split_plan": split_plan,
                                "files": manifest_files,
                            },
                            indent=2,
                        )
                        + "\n"
                    )

                eff_cte_p95 = float(np.percentile(np.abs(eff_cte_errs), 95)) if eff_cte_errs else 0.0
                eff_head_p95 = float(np.percentile(np.abs(eff_head_degs), 95)) if eff_head_degs else 0.0
                eff_la1_p95 = float(np.percentile(np.abs(eff_la1_errs), 95)) if eff_la1_errs else 0.0
                enc_p50 = float(np.percentile(enc_ms_list, 50)) if enc_ms_list else 0.0
                enc_p95 = float(np.percentile(enc_ms_list, 95)) if enc_ms_list else 0.0
                obs_act_p95 = float(np.percentile(obs_to_act_ms_list, 95)) if obs_to_act_ms_list else 0.0
                dt_p95 = float(np.percentile(step_dt_ms_list, 95)) if step_dt_ms_list else 0.0
                dt_arr = np.asarray(step_dt_ms_list, dtype=np.float32) if step_dt_ms_list else np.zeros(0, dtype=np.float32)
                bad_dt_steps = int(np.sum((dt_arr < 18.0) | (dt_arr > 32.0))) if len(dt_arr) else 0
                dt_std = float(np.std(dt_arr)) if len(dt_arr) else 0.0
                unwind_steps = int(sum(1 for s in ep_step_dicts if s.get("reversal_unwind_active", False)))
                coalesced_dropped = int(sync_controller.coalesced_dropped)
                max_burst_seen = int(sync_controller.max_burst_seen)

                ep_row = {
                    "episode": ep_counter,
                    "layout": layout_idx,
                    "split": split_name,
                    "condition": cname,
                    "sync_enabled": ctrl_cfg["sync_enabled"],
                    "reversal_unwind_scale": ctrl_cfg["reversal_unwind_scale"],
                    "completed": completed,
                    "steps": steps,
                    "laps": laps,
                    "termination_reason": term_reason,
                    "mean_abs_cte": float(np.mean(abs_ctes)),
                    "max_abs_cte": float(np.max(abs_ctes)),
                    "frames_abs_cte_ge_1_0": int(np.sum(np.asarray(abs_ctes) >= 1.0)),
                    "mean_speed": float(np.mean(speeds)),
                    "mean_abs_dsteer": float(np.mean(dsteers)),
                    "rate_limit_sat_steps": int(np.sum(np.asarray(dsteers) >= 0.199)),
                    "reversal_unwind_steps": unwind_steps,
                    "reward": total_reward,
                    "eff_cte_err_p95_units": eff_cte_p95,
                    "eff_head_deg_p95": eff_head_p95,
                    "eff_la1_err_p95_pos": eff_la1_p95,
                    "steer_action_gap_mae": float(np.mean(steer_gaps)),
                    "steer_action_gap_p95": float(np.percentile(steer_gaps, 95)),
                    "encoder_ms_p50": enc_p50,
                    "encoder_ms_p95": enc_p95,
                    "decision_obs_to_act_ms_p95": obs_act_p95,
                    "step_dt_ms_p95": dt_p95,
                    "step_dt_ms_std": dt_std,
                    "bad_dt_steps": bad_dt_steps,
                    "coalesced_dropped": coalesced_dropped,
                    "max_burst_seen": max_burst_seen,
                    "wall_seconds": elapsed,
                    "capture_npz": capture_rel_path,
                    "excursions": excursions,
                }
                all_episodes.append(ep_row)

                # Incremental flush of step telemetry, episode record, and aggregate results
                with telemetry_path.open("a") as f_tel:
                    f_tel.write("\n".join(json.dumps(r) for r in ep_step_dicts) + "\n")
                with episodes_path.open("a") as f_ep:
                    f_ep.write(json.dumps(ep_row) + "\n")
                cur_summary = summarize_episodes(condition_names, all_episodes, step_pools_by_cond)
                cur_paired = compute_paired_vs_baseline(all_episodes, condition_names)
                results_path.write_text(
                    json.dumps(
                        {
                            "metadata": metadata,
                            "layouts_completed": layout_idx,
                            "summary": cur_summary,
                            "paired_vs_baseline": cur_paired,
                            "excursions": all_excursions,
                            "episodes": all_episodes,
                        },
                        indent=2,
                    )
                    + "\n"
                )

                print(
                    f"  [L{layout_idx:02d} {split_name[:6]} {cname:<23}] {'PASS' if completed else 'FAIL':<4} "
                    f"steps={steps:>4} laps={laps:>2} mean|CTE|={ep_row['mean_abs_cte']:.3f} "
                    f"max|CTE|={ep_row['max_abs_cte']:.3f} (|CTE|>=1:{ep_row['frames_abs_cte_ge_1_0']}) spd={ep_row['mean_speed']:.2f} "
                    f"errP95(CTE={eff_cte_p95:.2f}u, Head={eff_head_p95:.1f}°) "
                    f"enc={enc_p50:.2f}/{enc_p95:.2f}ms obs2actP95={obs_act_p95:.2f}ms ({elapsed:.1f}s)",
                    flush=True,
                )
    finally:
        sync_controller.close()
        env.close()

    summary = summarize_episodes(condition_names, all_episodes, step_pools_by_cond)
    print("\n=== Closed-Loop Evaluation Summary ===", flush=True)
    print(
        f"{'condition':<24} {'comp':>6} {'steps':>7} {'laps':>6} "
        f"{'mean|CTE|(u)':>12} {'max|CTE|(u)':>11} {'speed':>6} "
        f"{'epP95/poolP95_CTE':>18} {'epP95/poolP95_Head':>19} {'enc_P50/P95(ms)':>16}",
        flush=True,
    )
    for cname in condition_names:
        if cname not in summary:
            continue
        s = summary[cname]
        pool_s = s.get("pooled_all_steps", {})
        comp_str = f"{s['completed']}/{s['episodes']}"
        cte_p95_str = f"{s['mean_eff_cte_p95_units']:.3f}/{pool_s.get('eff_cte_err_p95_units', 0.0):.3f}u"
        head_p95_str = f"{s['mean_eff_head_p95_deg']:.2f}°/{pool_s.get('eff_head_deg_p95', 0.0):.2f}°"
        enc_str = f"{s['encoder_ms_p50']:.2f}/{s['encoder_ms_p95']:.2f}"
        print(
            f"{cname:<24} {comp_str:>6} {s['mean_steps']:>7.1f} {s['mean_laps']:>6.2f} "
            f"{s['mean_abs_cte_units']:>12.3f} {s['max_abs_cte_units']:>11.3f} {s['mean_speed']:>6.2f} "
            f"{cte_p95_str:>18} {head_p95_str:>19} {enc_str:>16}",
            flush=True,
        )
    print(f"Saved results to {results_path}", flush=True)


if __name__ == "__main__":
    main()
