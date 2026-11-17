import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rl.geometry_encoder import DinoGridLinearLag15Encoder
from tools.train_geometry_recovery import capture_plan, predict_head
from tools.eval_teacher_tolerance import build_conditions, condition_control_config, compute_paired_vs_baseline


def test_capture_partition_groups_conditions_and_never_opens_locked_holdout(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    paths = []
    for layout in range(1, 5):
        for condition in ("baseline", "vision"):
            path = run / f"layout_{layout}_{condition}.npz"
            np.savez(path, split="dagger_train_pool", layout=layout, condition=condition, step=[1, 2])
            paths.append(path.name)
    (run / "split_manifest.json").write_text(json.dumps({"files": {
        "dagger_train_pool": paths, "locked_eval_holdout": ["must_never_be_opened.npz"],
    }}))
    plan = capture_plan([run], validation_layouts=1)
    assert {p["layout"] for p in plan if p["internal_validation"]} == {"4"}
    assert len([p for p in plan if p["internal_validation"]]) == 2
    np.savez(run / paths[0], split="locked_eval_holdout", layout=1, condition="baseline", step=[1])
    with pytest.raises(PermissionError):
        capture_plan([run], validation_layouts=1)


def test_batch_head_predictions_match_stateful_live_lags(tmp_path: Path):
    rng = np.random.default_rng(42)
    grid = rng.normal(size=(12, 6144)).astype(np.float32)
    saved = {"format": "proj15_lags_ridge_v1", "lags": [2, 4], "crop_top": 0,
             "w1": torch.tensor(rng.normal(size=(6144, 15)).astype(np.float32) * 0.001),
             "b1": torch.zeros(15), "w2": torch.tensor(rng.normal(size=(45, 15)).astype(np.float32)),
             "b2": torch.zeros(15)}
    path = tmp_path / "head.pt"
    torch.save(saved, path)
    fake_dino = type("Dino", (), {"crop_top": 0})()
    live = DinoGridLinearLag15Encoder(path, track=None, device=torch.device("cpu"), dino_encoder=fake_dino)
    streamed = np.stack([live.predict_geometry_from_grid_tensor(torch.from_numpy(row[None])) for row in grid])
    indices = np.maximum(np.arange(len(grid))[:, None] - np.array([0, 2, 4])[None], 0)
    batched = predict_head(path, grid, indices, torch.device("cpu"))
    np.testing.assert_allclose(batched, streamed, atol=2e-5, rtol=2e-5)


def test_paired_geometry_heads_keep_identical_control_and_produce_paired_deltas():
    conditions, residual_keys = build_conditions("round_b_vision_paired")
    names = [c[0] for c in conditions]
    assert residual_keys == []
    for name in names:
        assert condition_control_config(name) == {"sync_enabled": True, "reversal_unwind_scale": 1.0}
    eps = [{"condition": name, "layout": 1, "steps": 2000, "laps": 9,
            "completed": True, "mean_abs_cte": cte, "max_abs_cte": 0.8,
            "mean_speed": 2.5, "split": "locked_eval_holdout"}
           for name, cte in zip(names, [0.2, 0.18])]
    paired = compute_paired_vs_baseline(eps, names)
    delta = paired["all_layouts"]["paired_control_deltas"][0]
    assert delta["mean_abs_cte_diff_units"] == pytest.approx(-0.02)
