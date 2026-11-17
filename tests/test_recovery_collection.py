import numpy as np
import pytest

from rl.recovery_collection import RecoveryCollectionConfig, RecoveryCollector
from tools.eval_teacher_tolerance import build_conditions, condition_control_config


def state(cte=0.0, heading=0.0, speed=2.5):
    out = np.zeros(18, dtype=np.float32)
    out[0] = cte / 2.0
    out[1:3] = np.sin(np.radians(heading)), np.cos(np.radians(heading))
    out[3] = speed / 2.0
    return out


def test_recovery_returns_teacher_control_at_target_and_requires_stable_recovery():
    collector = RecoveryCollector(1, RecoveryCollectionConfig(warmup_steps=60))
    teacher = np.array([-0.2, 0.4], dtype=np.float32)
    original_teacher = teacher.copy()
    applied, note = collector.action(teacher, 59, state())
    np.testing.assert_array_equal(applied, teacher)
    assert note["recovery_phase"] == "cruise"
    applied, note = collector.action(teacher, 60, state())
    assert applied[0] == pytest.approx(0.35)
    assert applied[1] == teacher[1]
    np.testing.assert_array_equal(teacher, original_teacher)
    applied, note = collector.action(teacher, 61, state(cte=0.7))
    np.testing.assert_array_equal(applied, teacher)
    assert note["recovery_phase"] == "recovery"
    assert collector.events[0]["release_reason"] == "target"
    for step in range(62, 70):
        collector.action(teacher, step, state())
    assert collector.events[0]["recovered"] is True
    assert collector.events[0]["peak_abs_cte"] == pytest.approx(0.7)
    assert collector.phase == "cruise"
    _, note = collector.action(teacher, 149, state())
    assert note["target_signed_cte"] == pytest.approx(-0.6)


@pytest.mark.parametrize("observed,release", [(state(heading=21), "heading_guard"), (state(cte=-1.7), "cte_guard")])
def test_recovery_releases_on_guards_and_keeps_failed_geometry(observed, release):
    collector = RecoveryCollector(1, RecoveryCollectionConfig(warmup_steps=60))
    action = np.array([0.1, -0.5], dtype=np.float32)
    collector.action(action, 60, state())
    actual, _ = collector.action(action, 61, observed)
    np.testing.assert_array_equal(actual, action)
    assert collector.events[0]["release_reason"] == release
    collector.finish(62, "cte_limit", -2.1)
    assert collector.events[0]["peak_abs_cte"] == pytest.approx(2.1)
    assert collector.events[0]["recovered"] is False


def test_collection_bounds_duration_and_freezes_teacher_control_configuration():
    collector = RecoveryCollector(6, RecoveryCollectionConfig(warmup_steps=60))
    action = np.array([0.0, -0.2], dtype=np.float32)
    applied, note = collector.action(action, 60, state())
    assert applied[0] < 0 and note["target_signed_cte"] == -1.4
    applied, note = collector.action(action, 64, state())
    np.testing.assert_array_equal(applied, action)
    assert note["recovery_phase"] == "recovery"
    assert collector.events[0]["release_reason"] == "duration"
    specs, keys = build_conditions("round_b_collect")
    assert specs == [("teacher_recovery", None, 0.0, 0)] and keys == []
    assert condition_control_config(specs[0][0]) == {"sync_enabled": True, "reversal_unwind_scale": 1.0}


def test_collection_does_not_interrupt_cornering_or_startup():
    collector = RecoveryCollector(1)
    teacher = np.array([0.1, 0.0], dtype=np.float32)
    _, note = collector.action(teacher, 249, state())
    assert note["recovery_phase"] == "cruise"
    curve = state()
    curve[10] = 0.4
    _, note = collector.action(teacher, 250, curve)
    assert note["recovery_phase"] == "cruise"
    _, note = collector.action(np.array([0.8, 0.0]), 251, state())
    assert note["recovery_phase"] == "cruise"
    _, note = collector.action(teacher, 252, state())
    assert note["recovery_phase"] == "kick"


@pytest.mark.parametrize("layout,target", [(1, 0.6), (2, 1.0), (3, 1.4), (4, 0.6)])
def test_negative_collection_preserves_kick_and_release_bounds(layout, target):
    collector = RecoveryCollector(layout, RecoveryCollectionConfig(directions=(-1,)))
    teacher = np.array([0.1, 0.4], dtype=np.float32)
    applied, note = collector.action(teacher, 250, state())
    assert note["target_signed_cte"] == pytest.approx(-target)
    assert applied[0] == pytest.approx(-0.35)
    assert applied[1] == teacher[1]
    applied, note = collector.action(teacher, 254, state())
    np.testing.assert_array_equal(applied, teacher)
    assert note["recovery_phase"] == "recovery"
    assert collector.events[0]["release_reason"] == "duration"


@pytest.mark.parametrize("directions", [(), (0,), (-1, 2)])
def test_collection_rejects_invalid_directions(directions):
    with pytest.raises(ValueError, match="directions"):
        RecoveryCollectionConfig(directions=directions)
