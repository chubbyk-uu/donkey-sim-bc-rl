from types import SimpleNamespace

import numpy as np

from rl.train_vae_sac import DonkeyVaeSACEnv
from tools.compare_geometry_control import action_metrics, executed_actions, persistent_bias, sequence_segments


def test_counterfactual_mapping_matches_production_at_saturation_and_reversals():
    raw = np.array([[-2, -2], [2, 2], [-1, 0], [0.2, -1], [-0.2, 1]], np.float32)
    history = np.zeros((len(raw), 40), np.float32)
    history[:, -2] = [-0.9, 0.9, 0.8, -0.8, 0.1]
    expected = []
    for action, previous in zip(raw, history[:, -2]):
        env = SimpleNamespace(max_steering=1.0, max_steering_diff=0.1, min_throttle=0.2,
                              max_throttle=0.7, reversal_unwind_scale=1.0, last_steer=float(previous))
        expected.append(DonkeyVaeSACEnv._map_and_clip_action(env, action))
    np.testing.assert_allclose(executed_actions(raw, history), expected, atol=1e-7)


def test_persistent_windows_never_bridge_missing_steps_or_conditions():
    run = np.array(["r"] * 7)
    layout = np.array(["1"] * 7)
    condition = np.array(["a"] * 5 + ["b"] * 2)
    step = np.array([4, 5, 6, 9, 10, 4, 5])
    segments = sequence_segments(run, layout, condition, step)
    assert [list(s) for s in segments] == [[0, 1, 2], [3, 4], [5, 6]]
    result, windows = persistent_bias(np.array([.1, .1, .1, -.9, -.9, .8, .8]), segments, window=3)
    assert np.isclose(result["max_abs_mean_error"], .1)
    assert windows[0][1].tolist() == [0, 1, 2]


def test_direction_and_unwind_metrics_use_teacher_reference_and_deadbands():
    state = np.zeros((4, 18), np.float32)
    state[:, 0] = -0.6
    history = np.zeros((4, 40), np.float32)
    history[:, -2] = [0.5, 0.5, -0.5, 0.1]
    truth = np.array([[.3, .4], [.3, .4], [-.3, .4], [.105, .4]], np.float32)
    candidate = np.array([[.4, .4], [.7, .4], [-.5, .4], [-.1, .4]], np.float32)
    result = action_metrics(candidate, candidate, truth, truth, history, np.ones(4, bool), state)
    assert result["teacher_unwind_frames"] == 3
    assert result["missed_unwind_frames"] == 2
    assert result["teacher_direction_frames"] == 3
    assert result["wrong_direction_frames"] == 1
    identity = action_metrics(truth, truth, truth, truth, history, np.ones(4, bool), state)
    assert identity["steer_mae"] == 0
    assert identity["missed_unwind_frames"] == 0
