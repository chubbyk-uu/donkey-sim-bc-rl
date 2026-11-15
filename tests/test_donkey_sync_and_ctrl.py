from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
import socket
import threading
import time

import gymnasium as gym
import numpy as np
from PIL import Image
import pytest

from rl.donkey_sync import (
    LowJitterSyncController,
    add_control_fix_args,
    coalesce_telemetry_messages,
    configure_env_control_fixes,
    split_json_lines_buffer,
)
from rl.experiment import environment_config, validate_model_config
from rl.train_vae_sac import DonkeyVaeSACEnv
from tools.eval_teacher_tolerance import (
    apply_recovery_stress,
    build_conditions,
    compute_paired_vs_baseline,
    condition_control_config,
    summarize_episodes,
)


def _encode_png_b64(value: int) -> str:
    arr = np.full((8, 8, 3), value, dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_split_and_coalesce_telemetry_messages() -> None:
    raw = '{"msg_type":"telemetry","cte":0.1}\n{"msg_type":"telemetry","cte":0.2}\n{"msg_type":"part'
    msgs, rem = split_json_lines_buffer(raw)
    assert len(msgs) == 2
    assert rem == '{"msg_type":"part'

    parsed = [
        {"msg_type": "aborted_check", "tag": 1},
        {"msg_type": "telemetry", "cte": 0.1, "hit": "none"},
        {"msg_type": "telemetry", "cte": 0.5, "hit": "wall"},
        {"msg_type": "lap_timer", "lap": 2},
        {"msg_type": "telemetry", "cte": 0.6, "hit": "none"},
    ]
    ordered, superseded = coalesce_telemetry_messages(parsed)
    assert len(superseded) == 2
    assert [m["cte"] for m in superseded] == [0.1, 0.5]
    assert [m["msg_type"] for m in ordered] == ["aborted_check", "lap_timer", "telemetry"]
    assert ordered[-1]["cte"] == 0.6


class _FakeHandler:
    def __init__(self) -> None:
        self.image_array = np.zeros((8, 8, 3), dtype=np.uint8)
        self.image_array_b = None
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.speed = 0.0
        self.forward_vel = 0.0
        self.gyro_x = 0.0
        self.gyro_y = 0.0
        self.gyro_z = 0.0
        self.accel_x = 0.0
        self.accel_y = 0.0
        self.accel_z = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.vel_z = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.cte = 0.0
        self.hit = "none"
        self.lidar = None
        self.last_lap_time = 0.0
        self.lap_count = 0
        self.over = False
        self.time_received = 0.0
        self.last_received = 0.0
        self.orig_observe_calls = 0
        self.orig_queue_calls = 0
        self.fns = {"telemetry": self.on_telemetry}

    def on_telemetry(self, message: dict) -> None:
        self.cte = message.get("cte", 0.0)
        self.time_received = time.time()

    def observe(self):
        self.orig_observe_calls += 1
        return self.image_array, 0.0, self.over, {"cte": self.cte}

    def reset(self) -> None:
        self.over = False
        self.hit = "none"
        self.cte = 0.0

    def queue_message(self, msg: dict) -> None:
        self.orig_queue_calls += 1

    def determine_episode_over(self) -> None:
        if abs(self.cte) > 2.0 or self.hit != "none":
            self.over = True

    def is_game_over(self) -> bool:
        return self.over

    def calc_reward(self, done: bool) -> float:
        return -1.0 if done else 1.0


class _FakeClient:
    def __init__(self, sock: socket.socket, handler: _FakeHandler) -> None:
        self.s = sock
        self.handler = handler
        self.do_process_msgs = True
        self.aborted = False
        self.poll_socket_sleep_sec = 0.001
        self.msg: str | None = None
        self.th: threading.Thread | None = None

    def on_msg_recv(self, j: dict) -> None:
        fn = self.handler.fns.get(j.get("msg_type"))
        if fn is not None:
            fn(j)


def _tel_line(cte: float, val: int, hit: str = "none") -> bytes:
    payload = {
        "msg_type": "telemetry",
        "image": _encode_png_b64(val),
        "pos_x": float(val),
        "pos_y": 0.0,
        "pos_z": 1.0,
        "speed": 3.0,
        "gyro_x": 0.0,
        "gyro_y": 0.1,
        "gyro_z": 0.0,
        "accel_x": 0.0,
        "accel_y": 0.0,
        "accel_z": 0.0,
        "vel_x": 0.0,
        "vel_y": 0.0,
        "vel_z": 3.0,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "cte": float(cte),
        "hit": hit,
    }
    return (json.dumps(payload) + "\n").encode("utf-8")


def test_low_jitter_sync_controller_coalescing_and_toggle() -> None:
    s_cli, s_sim = socket.socketpair()
    handler = _FakeHandler()
    client = _FakeClient(s_cli, handler)
    sim_ctrl = type("SimCtrl", (), {"handler": handler, "client": client})()
    sync = LowJitterSyncController(sim_ctrl, enabled=True).install()

    try:
        # Queue two telemetry packets back-to-back in one TCP burst; earlier hit must survive!
        burst = _tel_line(0.25, 42, hit="cone") + _tel_line(0.50, 99, hit="none")
        s_sim.sendall(burst)

        obs, _reward, done, info = handler.observe()
        assert int(obs[0, 0, 0]) == 99
        assert info["pos"] == (99.0, 0.0, 1.0)
        assert info["cte"] == pytest.approx(0.50)
        assert done is True
        assert sync.coalesced_dropped == 1
        assert sync.max_burst_seen == 2

        # Pipe wakeup flushes queued command immediately without 1ms polling sleep.
        handler.queue_message({"msg_type": "control", "steering": "0.1", "throttle": "0.2", "brake": "0.0"})
        s_sim.settimeout(1.0)
        recv_cmd = s_sim.recv(4096).decode("utf-8")
        assert '"steering": "0.1"' in recv_cmd

        # Toggling off falls back to stock observe / queue_message paths.
        sync.set_enabled(False)
        handler.reset()
        handler.queue_message({"msg_type": "control"})
        obs2, _r2, _d2, _i2 = handler.observe()
        assert obs2.shape == (8, 8, 3)
        assert handler.orig_queue_calls == 1
        assert handler.orig_observe_calls == 1
    finally:
        client.do_process_msgs = False
        if client.th is not None:
            client.th.join(timeout=1.0)
        sync.close()
        s_cli.close()
        s_sim.close()


class _DummySimEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(8, 8, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.next_gt_cte = 0.0
        self.last_action = np.zeros(2, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros((8, 8, 3), dtype=np.uint8), {"cte": self.next_gt_cte, "speed": 2.5, "lap_count": 0}

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32)
        return (
            np.zeros((8, 8, 3), dtype=np.uint8),
            1.0,
            False,
            False,
            {"cte": self.next_gt_cte, "speed": 2.5, "lap_count": 0},
        )


class _DummyVisionEncoder:
    uses_telemetry = True
    z_size = 18

    def __init__(self) -> None:
        self.last_pred_geom = np.zeros(15, dtype=np.float32)
        self.last_pred_geom[2] = 1.0  # cos(0)

    def set_geom(self, cte: float, head_deg: float) -> None:
        rad = np.radians(head_deg)
        self.last_pred_geom = np.zeros(15, dtype=np.float32)
        self.last_pred_geom[0] = cte / 2.0
        self.last_pred_geom[1] = np.sin(rad)
        self.last_pred_geom[2] = np.cos(rad)

    def encode(self, image: np.ndarray, info: dict) -> np.ndarray:
        z = np.zeros(18, dtype=np.float32)
        z[0] = self.last_pred_geom[0]
        z[1] = self.last_pred_geom[1]
        z[2] = self.last_pred_geom[2]
        return z


def test_env_gated_reversal_unwind_uses_obs_geom_only() -> None:
    raw_env = _DummySimEnv()
    enc = _DummyVisionEncoder()
    env = DonkeyVaeSACEnv(
        raw_env,
        vae=enc,
        min_throttle=0.25,
        max_throttle=0.45,
        max_steering=1.0,
        max_steering_diff=0.1,
        n_command_history=1,
        reward_config=None,
        reversal_unwind_scale=2.0,
        reversal_cte_threshold=0.40,
        reversal_head_deg_threshold=8.0,
    )
    env.reset()

    # 1. Build left steering while centered: unwind gate is OFF.
    enc.set_geom(cte=0.10, head_deg=2.0)
    raw_env.next_gt_cte = 1.80  # ground truth large, obs small -> MUST NOT leak ground truth
    for _ in range(4):
        _, _, _, _, info = env.step(np.array([-0.8, 0.0], dtype=np.float32))
    assert info["rl_steer"] == pytest.approx(-0.79992, abs=1e-4)
    assert info["reversal_unwind_active"] is False

    # 2. Opposite-sign reversal while centered in obs -> stays strictly clamped to ~0.20/step.
    _, _, _, _, info = env.step(np.array([0.8, 0.0], dtype=np.float32))
    assert info["rl_steer"] == pytest.approx(-0.59994, abs=1e-4)
    assert info["reversal_unwind_active"] is False

    # 3. Same-sign increase under excursion -> NEVER boosts turning deeper into the same direction.
    enc.set_geom(cte=0.80, head_deg=12.0)
    env.step(np.array([0.0, 0.0], dtype=np.float32))  # first let _last_geom update with new obs
    prev = env.last_steer
    _, _, _, _, info = env.step(np.array([-1.0, 0.0], dtype=np.float32))
    assert info["rl_steer"] == pytest.approx(prev - 0.19998, abs=1e-4)
    assert info["reversal_unwind_active"] is False

    # 4. Opposite-sign reversal under excursion with large |last_steer| -> unwinds toward 0 faster.
    _, _, _, _, info = env.step(np.array([0.9, 0.0], dtype=np.float32))
    assert info["rl_steer"] == pytest.approx(prev - 0.19998 + 0.39996, abs=1e-4)
    assert info["reversal_unwind_active"] is True

    # 5. Small |last_steer| (< 0.30) must NOT ping-pong across 0 (stays clamped to max_diff).
    env.last_steer = -0.15
    _, _, _, _, info = env.step(np.array([0.9, 0.0], dtype=np.float32))
    assert info["rl_steer"] == pytest.approx(0.04998, abs=1e-4)
    assert info["reversal_unwind_active"] is False

    # 5b. Unwind cap stops at 0.0 (never jumps past 0 in the same unwind step).
    env.last_steer = -0.32
    _, _, _, _, info = env.step(np.array([0.9, 0.0], dtype=np.float32))
    assert info["rl_steer"] == pytest.approx(0.0, abs=1e-5)
    assert info["reversal_unwind_active"] is True

    # 6. Disabling via set_control_fixes reverts to strict baseline clamp.
    env.set_control_fixes(reversal_unwind_scale=1.0)
    env.reset()
    enc.set_geom(cte=0.90, head_deg=20.0)
    env.step(np.array([-0.5, 0.0], dtype=np.float32))
    env.last_steer = -0.60
    _, _, _, _, info = env.step(np.array([0.9, 0.0], dtype=np.float32))
    assert info["rl_steer"] == pytest.approx(-0.40002, abs=1e-4)
    assert info["reversal_unwind_active"] is False


def test_eval_teacher_tolerance_fix_modes_and_metrics() -> None:
    specs, pred_keys = build_conditions("teacher_fix_paired")
    assert pred_keys == []
    assert [c[0] for c in specs] == [
        "baseline",
        "teacher_sync_only",
        "teacher_fixed",
        "vision_fixed_ctrl",
    ]
    assert condition_control_config("baseline") == {"sync_enabled": False, "reversal_unwind_scale": 1.0}
    assert condition_control_config("teacher_sync_only") == {"sync_enabled": True, "reversal_unwind_scale": 1.0}
    assert condition_control_config("teacher_fixed") == {"sync_enabled": True, "reversal_unwind_scale": 2.0}
    assert condition_control_config("vision_fixed_ctrl") == {"sync_enabled": True, "reversal_unwind_scale": 2.0}

    specs_full, pred_keys_full = build_conditions("teacher_fix_paired_full")
    assert pred_keys_full == []
    expected_five = [
        "baseline",
        "teacher_sync_only",
        "teacher_fixed",
        "vision_baseline",
        "vision_fixed_ctrl",
    ]
    assert [c[0] for c in specs_full] == expected_five
    assert condition_control_config("baseline") == {"sync_enabled": False, "reversal_unwind_scale": 1.0}
    assert condition_control_config("vision_baseline") == {"sync_enabled": False, "reversal_unwind_scale": 1.0}
    assert condition_control_config("teacher_sync_only") == {"sync_enabled": True, "reversal_unwind_scale": 1.0}
    assert condition_control_config("teacher_fixed") == {"sync_enabled": True, "reversal_unwind_scale": 2.0}
    assert condition_control_config("vision_fixed_ctrl") == {"sync_enabled": True, "reversal_unwind_scale": 2.0}

    stress_specs, stress_keys = build_conditions("teacher_fix_recovery_stress")
    assert stress_keys == []
    assert [c[0] for c in stress_specs] == expected_five

    act0 = np.array([0.1, 0.5], dtype=np.float32)
    a_k1, s_k1, tag_k1 = apply_recovery_stress(act0, 320, layout_idx=1)
    assert tag_k1 == "kick_1" and s_k1 == 0.0 and a_k1[0] == pytest.approx(0.70)
    _, s_stall, tag_stall = apply_recovery_stress(act0, 900, layout_idx=1, stress_type="ctrl_stall")
    assert tag_stall == "stall_95ms" and s_stall == pytest.approx(0.095)
    _, s_tcp, tag_tcp = apply_recovery_stress(act0, 900, layout_idx=1, stress_type="tcp_backlog")
    assert tag_tcp == "tcp_recv_pause_95ms" and s_tcp == pytest.approx(0.095)
    _, s_both1, tag_both1 = apply_recovery_stress(act0, 600, layout_idx=1, stress_type="both")
    _, s_both2, tag_both2 = apply_recovery_stress(act0, 900, layout_idx=1, stress_type="both")
    assert tag_both1 == "stall_95ms" and s_both1 == pytest.approx(0.095)
    assert tag_both2 == "tcp_recv_pause_95ms" and s_both2 == pytest.approx(0.095)
    a_k2, s_k2, tag_k2 = apply_recovery_stress(act0, 1300, layout_idx=1)
    assert tag_k2 == "kick_2" and s_k2 == 0.0 and a_k2[0] == pytest.approx(-0.70)

    fake_ep = {
        "condition": "teacher_fixed",
        "completed": True,
        "steps": 2000,
        "laps": 9,
        "mean_abs_cte": 0.14,
        "max_abs_cte": 1.1,
        "mean_speed": 2.8,
        "eff_cte_err_p95_units": 0.0,
        "eff_head_deg_p95": 0.0,
        "eff_la1_err_p95_pos": 0.0,
        "encoder_ms_p50": 0.05,
        "encoder_ms_p95": 0.1,
        "bad_dt_steps": 2,
        "coalesced_dropped": 3,
        "max_burst_seen": 4,
        "reversal_unwind_steps": 7,
        "excursions": [
            {"duration_steps": 6, "recovered": True},
            {"duration_steps": 10, "recovered": False},
        ],
    }
    summ = summarize_episodes(["teacher_fixed"], [fake_ep])["teacher_fixed"]
    assert summ["bad_dt_steps_total"] == 2
    assert summ["coalesced_dropped_total"] == 3
    assert summ["max_burst_seen"] == 4
    assert summ["reversal_unwind_steps_total"] == 7
    assert summ["excursion_count"] == 2
    assert summ["excursion_recovered_count"] == 1
    assert summ["excursion_recovery_rate"] == pytest.approx(0.5)
    assert summ["excursion_median_recovery_steps"] == pytest.approx(6.0)

    eps = [
        {"layout": 1, "split": "dagger_train_pool", "condition": "baseline", "completed": True, "steps": 2000, "laps": 8, "mean_abs_cte": 0.16, "max_abs_cte": 0.9, "mean_speed": 2.9},
        {"layout": 1, "split": "dagger_train_pool", "condition": "teacher_sync_only", "completed": True, "steps": 2000, "laps": 8, "mean_abs_cte": 0.15, "max_abs_cte": 0.85, "mean_speed": 2.9},
        {"layout": 1, "split": "dagger_train_pool", "condition": "teacher_fixed", "completed": True, "steps": 2000, "laps": 8, "mean_abs_cte": 0.14, "max_abs_cte": 0.8, "mean_speed": 2.9},
        {"layout": 1, "split": "dagger_train_pool", "condition": "vision_baseline", "completed": False, "steps": 900, "laps": 3, "mean_abs_cte": 0.22, "max_abs_cte": 2.0, "mean_speed": 2.8},
        {"layout": 1, "split": "dagger_train_pool", "condition": "vision_fixed_ctrl", "completed": True, "steps": 2000, "laps": 8, "mean_abs_cte": 0.18, "max_abs_cte": 1.1, "mean_speed": 2.9},
    ]
    paired = compute_paired_vs_baseline(eps, expected_five)["all_layouts"]
    ctrl_pairs = {(d["reference"], d["candidate"]): d for d in paired["paired_control_deltas"]}
    assert ctrl_pairs[("vision_baseline", "vision_fixed_ctrl")]["mean_abs_cte_diff_units"] == pytest.approx(-0.04)
    assert ctrl_pairs[("teacher_sync_only", "teacher_fixed")]["mean_abs_cte_diff_units"] == pytest.approx(-0.01)


def _decode_concat_json(buf: str) -> list[dict]:
    dec = json.JSONDecoder()
    idx = 0
    out: list[dict] = []
    n = len(buf)
    while idx < n:
        while idx < n and buf[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = dec.raw_decode(buf, idx)
        out.append(obj)
        idx = end
    return out


class _PartialSendSocket:
    """Socket wrapper forcing chunked partial sends and BlockingIOError."""

    def __init__(self, sock: socket.socket, max_chunk: int = 9, block_every: int = 2) -> None:
        self._sock = sock
        self._max_chunk = max_chunk
        self._block_every = block_every
        self._calls = 0
        self.partial_calls = 0
        self.wouldblock_calls = 0
        self.bytes_sent = 0

    def fileno(self) -> int:
        return self._sock.fileno()

    def setblocking(self, flag: bool) -> None:
        self._sock.setblocking(flag)

    def recv(self, bufsize: int) -> bytes:
        return self._sock.recv(bufsize)

    def send(self, data) -> int:
        self._calls += 1
        if self._calls % self._block_every == 0:
            self.wouldblock_calls += 1
            time.sleep(0.002)
            raise BlockingIOError("simulated EAGAIN")
        raw = bytes(data)[: self._max_chunk]
        sent = self._sock.send(raw)
        self.partial_calls += 1
        self.bytes_sent += sent
        return sent

    def close(self) -> None:
        self._sock.close()


class _QueueToClientHandler(_FakeHandler):
    def __init__(self, client: _FakeClient) -> None:
        super().__init__()
        self._client_ref = client

    def queue_message(self, msg: dict) -> None:
        super().queue_message(msg)
        self._client_ref.msg = json.dumps(msg)


@pytest.mark.parametrize("sync_enabled", [True, False])
def test_nonblocking_partial_send_never_duplicates_prefix(sync_enabled: bool) -> None:
    s_cli, s_sim = socket.socketpair()
    wrapped_cli = _PartialSendSocket(s_cli, max_chunk=9, block_every=2)
    handler = None
    client = None
    sync = None
    try:
        client = _FakeClient(wrapped_cli, _FakeHandler())
        handler = _QueueToClientHandler(client)
        client.handler = handler
        sim_ctrl = type("SimCtrl", (), {"handler": handler, "client": client})()
        sync = LowJitterSyncController(sim_ctrl, enabled=sync_enabled).install()

        pad_a = "A" * 40
        pad_b = "B" * 40
        msg1 = {"msg_type": "control", "seq": 1, "steering": "0.1234", "pad": pad_a}
        msg2 = {"msg_type": "control", "seq": 2, "steering": "-0.4567", "pad": pad_b}

        handler.queue_message(msg1)
        deadline = time.monotonic() + 2.0
        while wrapped_cli.bytes_sent < 15 and time.monotonic() < deadline:
            time.sleep(0.002)
        assert wrapped_cli.bytes_sent >= 15

        # Enqueue a second message while msg1 is still partially in flight.
        handler.queue_message(msg2)

        expected = len(json.dumps(msg1).encode("utf-8")) + len(json.dumps(msg2).encode("utf-8"))
        s_sim.settimeout(2.0)
        received = bytearray()
        deadline = time.monotonic() + 3.0
        while len(received) < expected and time.monotonic() < deadline:
            chunk = s_sim.recv(4096)
            if not chunk:
                break
            received.extend(chunk)
        # Ensure no extra duplicate tail arrives afterward.
        time.sleep(0.02)
        s_sim.setblocking(False)
        try:
            extra = s_sim.recv(4096)
            received.extend(extra)
        except BlockingIOError:
            pass

        objs = _decode_concat_json(received.decode("utf-8"))
        assert len(objs) == 2
        assert objs[0]["seq"] == 1 and objs[0]["steering"] == "0.1234" and objs[0]["pad"] == pad_a
        assert objs[1]["seq"] == 2 and objs[1]["steering"] == "-0.4567" and objs[1]["pad"] == pad_b
        assert wrapped_cli.wouldblock_calls >= 2
        assert wrapped_cli.partial_calls >= 4
    finally:
        if client is not None:
            client.do_process_msgs = False
            if client.th is not None:
                client.th.join(timeout=1.0)
        if sync is not None:
            sync.close()
        s_cli.close()
        s_sim.close()


def test_real_socket_backpressure_send_and_pause_recv_coalescing() -> None:
    s_cli, s_sim = socket.socketpair()
    s_cli.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    s_sim.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    handler = _FakeHandler()
    client = _FakeClient(s_cli, handler)
    sim_ctrl = type("SimCtrl", (), {"handler": handler, "client": client})()
    sync = LowJitterSyncController(sim_ctrl, enabled=True).install()

    try:
        # 1. Real kernel socket backpressure: send a single 256 KiB message larger than
        # SO_SNDBUF + SO_RCVBUF so kernel send() hits partial writes + EAGAIN until s_sim drains.
        big_pad = "Z" * 262144
        msg1 = {"msg_type": "control", "seq": 1, "steering": "0.15", "pad": big_pad}
        msg2 = {"msg_type": "control", "seq": 2, "steering": "-0.25", "pad": "TAIL"}
        expected_bytes = (
            len(json.dumps(msg1).encode("utf-8"))
            + len(json.dumps(msg2).encode("utf-8"))
        )

        handler.queue_message(msg1)
        # Let _proc_msg_loop pick up msg1 as _inflight_bytes and fill the kernel buffer.
        time.sleep(0.02)
        handler.queue_message(msg2)

        s_sim.settimeout(2.0)
        received = bytearray()
        deadline = time.monotonic() + 3.0
        while len(received) < expected_bytes and time.monotonic() < deadline:
            chunk = s_sim.recv(8192)
            if not chunk:
                break
            received.extend(chunk)
        objs = _decode_concat_json(received.decode("utf-8"))
        assert len(objs) == 2
        assert objs[0]["seq"] == 1 and objs[0]["pad"] == big_pad
        assert objs[1]["seq"] == 2 and objs[1]["pad"] == "TAIL"

        # 2. Real TCP recv pause: hold off recv() so simulator telemetry accumulates in kernel buffer,
        # while outgoing control commands continue flushing without delay.
        sync.pause_recv(0.08)
        handler.queue_message({"msg_type": "control", "steering": "0.22", "throttle": "0.4"})
        out_raw = s_sim.recv(4096).decode("utf-8")
        assert '"steering": "0.22"' in out_raw

        frames = [
            _tel_line(0.10, 10, hit="none"),
            _tel_line(0.20, 20, hit="cone"),
            _tel_line(0.30, 30, hit="none"),
            _tel_line(0.40, 40, hit="none"),
        ]
        for frame in frames:
            s_sim.sendall(frame)
            time.sleep(0.008)

        obs, _reward, done, info = handler.observe()
        assert int(obs[0, 0, 0]) == 40
        assert info["cte"] == pytest.approx(0.40)
        assert done is True
        assert sync.coalesced_dropped >= 2
        assert sync.max_burst_seen >= 3
        sync.reset_stats()
        assert sync.coalesced_dropped == 0
        assert sync.max_burst_seen == 0
    finally:
        client.do_process_msgs = False
        if client.th is not None:
            client.th.join(timeout=1.0)
        sync.close()
        s_cli.close()
        s_sim.close()


def test_opt_in_control_fix_cli_args_and_config_compat() -> None:
    parser = argparse.ArgumentParser()
    add_control_fix_args(parser)

    # Defaults leave behavior strictly stock / disabled.
    default_args = parser.parse_args([])
    assert default_args.low_jitter_sync is False
    assert default_args.reversal_unwind_scale == pytest.approx(1.0)
    assert default_args.reversal_cte_threshold == pytest.approx(0.40)
    assert default_args.reversal_head_deg_threshold == pytest.approx(8.0)
    assert default_args.reversal_min_last_steer == pytest.approx(0.30)

    enabled_args = parser.parse_args([
        "--low-jitter-sync",
        "--reversal-unwind-scale", "2.0",
        "--reversal-cte-threshold", "0.35",
        "--reversal-head-deg-threshold", "6.0",
        "--reversal-min-last-steer", "0.25",
    ])

    s_cli, s_sim = socket.socketpair()
    raw_env = _DummySimEnv()
    handler = _FakeHandler()
    client = _FakeClient(s_cli, handler)
    raw_env.viewer = type("SimCtrl", (), {"handler": handler, "client": client})()
    enc = _DummyVisionEncoder()
    env = DonkeyVaeSACEnv(raw_env, vae=enc)
    try:
        # At defaults, no sync controller is installed and unwind remains 1.0.
        assert configure_env_control_fixes(env, default_args) is None
        assert env.reversal_unwind_scale == pytest.approx(1.0)

        # Opt-in installs sync controller and updates unwind parameters, and env.close() cleans up.
        sync = configure_env_control_fixes(env, enabled_args)
        assert sync is not None and sync.enabled is True
        assert env.reversal_unwind_scale == pytest.approx(2.0)
        assert env.reversal_cte_threshold == pytest.approx(0.35)
        assert env.reversal_head_deg_threshold == pytest.approx(6.0)
        assert env.reversal_min_last_steer == pytest.approx(0.25)

        env.close()
        assert sync._closed is True
    finally:
        client.do_process_msgs = False
        if client.th is not None:
            client.th.join(timeout=1.0)
        s_cli.close()
        s_sim.close()

    # Enabling opt-in control flags never breaks validate_model_config for legacy checkpoints.
    base_ns = argparse.Namespace(
        env_id="donkey-generated-track-v0",
        encoder="resnet18",
        encoder_crop_top=0,
        latent_lags="",
        min_throttle=0.25,
        max_throttle=0.45,
        max_steering=1.0,
        max_steering_diff=0.2,
        n_command_history=1,
        max_cte_error=2.0,
    )
    saved_cfg = environment_config(base_ns)
    fixed_ns = argparse.Namespace(**vars(base_ns), **vars(enabled_args))
    dummy_model = type("Model", (), {"donkey_config": saved_cfg})()
    validate_model_config(dummy_model, environment_config(fixed_ns))



