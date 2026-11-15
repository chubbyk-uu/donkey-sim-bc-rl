from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rl.geometry_encoder import (
    SPLIT_DAGGER_TRAIN_POOL,
    SPLIT_LOCKED_EVAL_HOLDOUT,
    DinoGridLinearLag15Encoder,
    assert_not_locked_holdout,
    assign_layout_split,
    build_layout_block_schedule,
    compute_cache_fingerprint,
    extract_excursion_windows,
    load_aligned_residual_blocks,
    splice_geometry_into_state,
    verify_cache_fingerprint,
)
from rl.track_state import TrackState
from tools.eval_teacher_tolerance import (
    PerturbedTrackStateEncoder,
    compute_paired_vs_baseline,
    summarize_episodes,
)


def _circle_track(radius: float = 5.0, n: int = 200) -> TrackState:
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    centre = np.column_stack([radius * np.sin(angles), radius * np.cos(angles)])
    return TrackState(centre, cte_per_unit=8.0)


def test_cache_fingerprint_detects_stale_inputs(tmp_path: Path) -> None:
    run = tmp_path / "run_a"
    cap_dir = run / "captures" / "ep1"
    cap_dir.mkdir(parents=True)
    (cap_dir / "index.json").write_text(json.dumps({"frames": [{"step": 1, "file": "f1.png"}]}))
    (run / "results.json").write_text(
        json.dumps({"episodes": [{"episode": 1, "capture": {"directory": "captures/ep1"}}]})
    )
    (run / "telemetry.jsonl").write_text(json.dumps({"step": 1, "cte": 0.0}) + "\n")
    centerline = tmp_path / "track.npz"
    centerline.write_bytes(b"centerline_v1")

    cache_path = tmp_path / "feature_cache.npz"
    cache_path.write_bytes(b"dummy")

    # Missing .meta.json must raise RuntimeError
    with pytest.raises(RuntimeError, match="missing"):
        verify_cache_fingerprint(cache_path, runs=[run], centerline_path=centerline)

    meta = compute_cache_fingerprint(runs=[run], centerline_path=centerline, crop_top=0)
    cache_path.with_suffix(".meta.json").write_text(json.dumps(meta))

    # Matching inputs succeed
    verified = verify_cache_fingerprint(cache_path, runs=[run], centerline_path=centerline, crop_top=0)
    assert verified["fingerprint_sha256"] == meta["fingerprint_sha256"]

    # Changing crop_top, centerline, or telemetry must fail verification
    with pytest.raises(RuntimeError, match="mismatch"):
        verify_cache_fingerprint(cache_path, runs=[run], centerline_path=centerline, crop_top=40)

    centerline.write_bytes(b"centerline_v2")
    with pytest.raises(RuntimeError, match="mismatch"):
        verify_cache_fingerprint(cache_path, runs=[run], centerline_path=centerline, crop_top=0)


def test_linear_lag15_encoder_splices_geometry_and_preserves_motion(tmp_path: Path) -> None:
    track = _circle_track()
    head_path = tmp_path / "proj15_lags_ridge.pt"

    w1 = torch.zeros((6144, 15), dtype=torch.float32)
    w1[0, 0] = 1.0  # pass feature[0] into z1[0]
    b1 = torch.zeros(15, dtype=torch.float32)

    w2 = torch.zeros((45, 15), dtype=torch.float32)
    # Output[0] (cte/2) = 0.5 * z_t[0] + 0.3 * z_{t-2}[0] + 0.2 * z_{t-4}[0]
    w2[0, 0] = 0.5
    w2[15, 0] = 0.3
    w2[30, 0] = 0.2
    b2 = torch.zeros(15, dtype=torch.float32)
    b2[1] = 3.0  # unnormalized sin
    b2[2] = 4.0  # unnormalized cos -> should normalize to (0.6, 0.8)

    torch.save(
        {
            "format": "proj15_lags_ridge_v1",
            "crop_top": 0,
            "lags": [2, 4],
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2,
        },
        head_path,
    )

    step_val = {"v": 0.0}
    fake_dino = SimpleNamespace(
        crop_top=0,
        grid_features=lambda imgs: torch.full((1, 6144), step_val["v"], dtype=torch.float32),
    )
    enc = DinoGridLinearLag15Encoder(head_path=head_path, track=track, device=torch.device("cpu"), dino_encoder=fake_dino)
    enc.reset_episode()

    info = {
        "pos": (0.0, 0.0, 5.0),
        "car": (0.0, 0.0, 90.0),
        "cte": 0.4,
        "vel": (0.25, 0.0, 2.4),
        "gyro": (0.0, -0.6, 0.0),
    }
    dummy_img = np.zeros((120, 160, 3), dtype=np.uint8)
    raw = track.encode(dummy_img, info)

    # Step 0: buffer pads with z_0 = 10 -> 0.5*10 + 0.3*10 + 0.2*10 = 10.0
    step_val["v"] = 10.0
    out0 = enc.encode(dummy_img, info)
    assert out0.shape == (18,)
    assert out0[0] == pytest.approx(10.0)
    assert out0[1] == pytest.approx(0.6) and out0[2] == pytest.approx(0.8)
    # Motion [3:6] must match TrackState.encode untouched
    np.testing.assert_allclose(out0[3:6], raw[3:6])

    # Advance 4 more steps with values 20, 30, 40, 50
    for val in (20.0, 30.0, 40.0, 50.0):
        step_val["v"] = val
        out_last = enc.encode(dummy_img, info)
    # At val=50 (t=4): z_t=50, z_{t-2}=30, z_{t-4}=10 -> 0.5*50 + 0.3*30 + 0.2*10 = 36.0
    assert out_last[0] == pytest.approx(36.0)
    np.testing.assert_allclose(out_last[3:6], raw[3:6])


def test_aligned_residual_schedule_pairs_conditions_and_scales(tmp_path: Path) -> None:
    n = 100
    state = np.zeros((n, 18), dtype=np.float32)
    true_geom = np.zeros((n, 15), dtype=np.float32)
    true_geom[:, 2] = 1.0  # cos(0) = 1

    pred_a = true_geom.copy()
    pred_a[:, 0] = np.linspace(0.01, 0.20, n, dtype=np.float32)
    pred_b = true_geom.copy()
    pred_b[:, 0] = np.linspace(0.05, 0.40, n, dtype=np.float32)

    np.savez(
        tmp_path / "feature_cache.npz",
        state=state,
        run=np.array(["r1"] * n),
        layout=np.array(["1"] * n),
        model=np.array(["m"] * n),
        step=np.arange(1, n + 1, dtype=np.int32),
    )
    np.savez(
        tmp_path / "oof_geometry_predictions_lags.npz",
        valid_lags_mask=np.ones(n, dtype=bool),
        true_geom=true_geom,
        pred_model_a=pred_a,
        pred_model_b=pred_b,
    )

    blocks = load_aligned_residual_blocks(tmp_path, ["pred_model_a", "pred_model_b"], min_block_len=50)
    assert len(blocks) == 1
    sched = build_layout_block_schedule(blocks, total_steps=60, block_len=50, seed=123)

    track = _circle_track()
    enc_1x = PerturbedTrackStateEncoder(track, blocks=blocks, pred_key="pred_model_b", schedule=sched, scale=1.0, delay_geom_steps=0)
    enc_15x = PerturbedTrackStateEncoder(track, blocks=blocks, pred_key="pred_model_b", schedule=sched, scale=1.5, delay_geom_steps=0)
    enc_d1 = PerturbedTrackStateEncoder(track, blocks=blocks, pred_key="pred_model_b", schedule=sched, scale=1.0, delay_geom_steps=1)

    info = {"pos": (0.0, 0.0, 5.0), "car": (0.0, 0.0, 90.0), "cte": 0.0, "vel": (0.1, 0.0, 2.0), "gyro": (0.0, 0.0, 0.0)}
    img = np.zeros((120, 160, 3), dtype=np.uint8)

    for step in range(10):
        info["vel"] = (0.1 * step, 0.0, 2.0 + 0.05 * step)
        o1 = enc_1x.encode(img, info)
        o15 = enc_15x.encode(img, info)
        od1 = enc_d1.encode(img, info)
        assert enc_1x.last_block_id == enc_15x.last_block_id == enc_d1.last_block_id
        assert enc_1x.last_block_offset == enc_15x.last_block_offset == enc_d1.last_block_offset
        assert o15[0] == pytest.approx(1.5 * o1[0], abs=1e-6)
        # Motion [3:6] is never delayed even when delay_geom_steps=1
        np.testing.assert_allclose(o1[3:6], od1[3:6])
        if step > 0:
            assert od1[0] == pytest.approx(prev_o1_cte)  # noqa: F821
        prev_o1_cte = o1[0]


def test_split_assignment_and_holdout_guard(tmp_path: Path) -> None:
    assert assign_layout_split(1, total_layouts=24) == SPLIT_DAGGER_TRAIN_POOL
    assert assign_layout_split(12, total_layouts=24) == SPLIT_DAGGER_TRAIN_POOL
    assert assign_layout_split(13, total_layouts=24) == SPLIT_LOCKED_EVAL_HOLDOUT
    assert assign_layout_split(24, total_layouts=24) == SPLIT_LOCKED_EVAL_HOLDOUT

    train_npz = tmp_path / f"layout_01_vision_{SPLIT_DAGGER_TRAIN_POOL}.npz"
    np.savez(train_npz, split=np.array(SPLIT_DAGGER_TRAIN_POOL), cte=np.zeros(5, dtype=np.float32))
    assert_not_locked_holdout(train_npz)

    holdout_by_name = tmp_path / f"layout_13_vision_{SPLIT_LOCKED_EVAL_HOLDOUT}.npz"
    np.savez(holdout_by_name, split=np.array(SPLIT_DAGGER_TRAIN_POOL), cte=np.zeros(5, dtype=np.float32))
    with pytest.raises(PermissionError, match="locked"):
        assert_not_locked_holdout([train_npz, holdout_by_name])

    holdout_by_payload = tmp_path / "layout_14_vision_renamed.npz"
    np.savez(holdout_by_payload, split=np.array(SPLIT_LOCKED_EVAL_HOLDOUT), cte=np.zeros(5, dtype=np.float32))
    with pytest.raises(PermissionError, match="locked_eval_holdout"):
        assert_not_locked_holdout(holdout_by_payload)


def test_excursion_window_and_pooled_vs_episode_statistics() -> None:
    # Simulate a 40-step trace with a 9-step rate-limit saturation into a -1.428 CTE peak
    cte = np.zeros(40, dtype=np.float32)
    cte[18:27] = np.array([-0.81, -1.04, -1.27, -1.428, -1.423, -1.23, -0.94, -0.62, -0.40], dtype=np.float32)
    eff_cte_err = np.full(40, -0.05, dtype=np.float32)
    eff_cte_err[21] = 0.250
    eff_head_deg = np.full(40, 1.5, dtype=np.float32)
    rl_steer = np.zeros(40, dtype=np.float32)
    rl_steer[21] = 0.996
    rl_steer_delta = np.zeros(40, dtype=np.float32)
    rl_steer_delta[13:22] = 0.19998  # 9 consecutive rate-limited steps
    rl_throttle = np.full(40, 0.35, dtype=np.float32)
    rl_throttle[21] = 0.428

    windows = extract_excursion_windows(
        cte=cte,
        eff_cte_err=eff_cte_err,
        eff_head_deg=eff_head_deg,
        rl_steer=rl_steer,
        rl_steer_delta=rl_steer_delta,
        rl_throttle=rl_throttle,
    )
    assert len(windows) == 1
    w = windows[0]
    assert w["peak_step"] == 22
    assert w["peak_cte_signed"] == pytest.approx(-1.428)
    assert w["eff_cte_err_at_peak"] == pytest.approx(0.250)
    assert w["max_consecutive_rate_limit_steps"] == 9
    assert w["recovered"] is True

    # Verify episode-averaged P95 vs pooled step-level P95 distinction
    episodes = [
        {
            "episode": 1,
            "layout": 1,
            "split": SPLIT_DAGGER_TRAIN_POOL,
            "condition": "baseline",
            "completed": True,
            "steps": 2000,
            "laps": 9,
            "mean_abs_cte": 0.15,
            "max_abs_cte": 0.70,
            "mean_speed": 2.52,
            "eff_cte_err_p95_units": 0.0,
            "eff_head_deg_p95": 0.0,
            "eff_la1_err_p95_pos": 0.0,
            "encoder_ms_p50": 0.10,
            "encoder_ms_p95": 0.13,
        },
        {
            "episode": 2,
            "layout": 1,
            "split": SPLIT_DAGGER_TRAIN_POOL,
            "condition": "vision_proj15_lags_live",
            "completed": True,
            "steps": 2000,
            "laps": 9,
            "mean_abs_cte": 0.18,
            "max_abs_cte": 1.428,
            "mean_speed": 2.55,
            "eff_cte_err_p95_units": 0.135,
            "eff_head_deg_p95": 2.80,
            "eff_la1_err_p95_pos": 0.029,
            "encoder_ms_p50": 4.11,
            "encoder_ms_p95": 4.69,
        },
    ]
    step_pools = {
        "baseline": [
            {
                "step": 1,
                "split": SPLIT_DAGGER_TRAIN_POOL,
                "cte": 0.1,
                "abs_cte": 0.1,
                "eff_cte_err_units": 0.0,
                "eff_head_deg": 0.0,
                "eff_la1_err_pos": 0.0,
                "rl_steer_delta": 0.02,
                "encoder_ms": 0.1,
                "policy_ms": 0.2,
                "decision_obs_to_act_ms": 0.3,
                "step_dt_ms": 25.0,
                "steer_action_gap": 0.0,
                "throttle_action_gap": 0.0,
            }
        ],
        "vision_proj15_lags_live": [
            {
                "step": 1,
                "split": SPLIT_DAGGER_TRAIN_POOL,
                "cte": -1.428,
                "abs_cte": 1.428,
                "eff_cte_err_units": 0.250,
                "eff_head_deg": 2.34,
                "eff_la1_err_pos": 0.03,
                "rl_steer_delta": 0.20,
                "encoder_ms": 4.37,
                "policy_ms": 0.22,
                "decision_obs_to_act_ms": 4.59,
                "step_dt_ms": 25.1,
                "steer_action_gap": 0.08,
                "throttle_action_gap": 0.15,
            }
        ],
    }
    summary = summarize_episodes(["baseline", "vision_proj15_lags_live"], episodes, step_pools)
    assert summary["vision_proj15_lags_live"]["mean_eff_cte_p95_units"] == pytest.approx(0.135)
    assert summary["vision_proj15_lags_live"]["pooled_all_steps"]["eff_cte_err_p95_units"] == pytest.approx(0.250)
    assert summary["vision_proj15_lags_live"]["regime_breakdown"]["large_offset_ge_1_0"]["steps"] == 1
    paired = compute_paired_vs_baseline(episodes, ["baseline", "vision_proj15_lags_live"])
    delta = paired["all_layouts"]["trajectory_deltas_vs_baseline"][0]
    assert delta["mean_abs_cte_diff_units"] == pytest.approx(0.03)
