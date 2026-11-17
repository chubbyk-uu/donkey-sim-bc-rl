import gc
import json
import threading

from rl.timing_diagnostics import GCPauseRecorder
from tools.eval_teacher_tolerance import build_conditions, condition_control_config


def test_gc_pause_recorder_tracks_actual_collection_and_detaches():
    recorder = GCPauseRecorder()
    recorder.step = 123
    recorder.phase = "env_step_and_encoder"
    recorder.install()
    try:
        cycle = []
        cycle.append(cycle)
        del cycle
        gc.collect(2)
        event = recorder.events[-1]
        assert event["generation"] == 2 and event["collected"] >= 1
        assert event["end_mono_s"] >= event["start_mono_s"]
        assert event["duration_ms"] >= 0
        assert event["step"] == 123 and event["phase"] == "env_step_and_encoder"
        assert event["thread_id"] == threading.get_ident()
    finally:
        recorder.close()
    count = len(recorder.events)
    gc.collect(2)
    assert len(recorder.events) == count
    assert recorder.callback not in gc.callbacks


def test_gc_diagnostic_changes_no_head_or_control_configuration():
    conditions, residual_keys = build_conditions("timing_gc_paired")
    assert residual_keys == []
    assert {c[1] for c in conditions} == {"__LIVE_VISION__"}
    for condition, _, _, _ in conditions:
        assert condition_control_config(condition) == {"sync_enabled": True, "reversal_unwind_scale": 1.0}


def test_gc_background_writer_flushes_on_close(tmp_path):
    recorder = GCPauseRecorder()
    path = tmp_path / "gc_events.jsonl"
    recorder.install()
    recorder.start_writer(path)
    try:
        gc.collect(2)
    finally:
        recorder.close()
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert events == recorder.events
    progress = json.loads((tmp_path / "diagnostic_progress.json").read_text())
    assert progress["gc_events"] == len(events)
    assert not recorder._writer.is_alive()
