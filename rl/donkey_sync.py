from __future__ import annotations

"""Event-driven, torn-read-free, burst-coalescing telemetry sync for DonkeyGym."""

import argparse
import base64
import codecs
from io import BytesIO
import json
import logging
import os
import select
import threading
import time
from typing import Any, Callable

import numpy as np
from PIL import Image

from gym_donkeycar.core.util import replace_float_notation

logger = logging.getLogger(__name__)


def split_json_lines_buffer(localbuffer: str) -> tuple[list[str], str]:
    """Extract complete newline-delimited JSON objects from `localbuffer`."""
    n0 = localbuffer.find("{")
    n1 = localbuffer.rfind("}\n")
    if n1 >= 0 and 0 <= n0 < n1:
        msgs = localbuffer[n0 : n1 + 1].split("\n")
        rem = localbuffer[n1 + 2 :] if localbuffer[n1 : n1 + 2] == "}\n" else localbuffer[n1 + 1 :]
        return [m for m in msgs if len(m) > 2], rem
    return [], localbuffer


def coalesce_telemetry_messages(
    parsed_msgs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate non-latest telemetry packets from the final telemetry state in a batch.

    Returns `(ordered_events, superseded_telemetry)` where:
    - `ordered_events` keeps all non-telemetry messages in order plus ONLY the last
      `"telemetry"` message (at its natural position at the end of the telemetry burst).
    - `superseded_telemetry` lists earlier `"telemetry"` packets whose heavy image
      decode can be skipped while still applying their `hit` / `over` flags.
    """
    last_tel_idx = -1
    for i, msg in enumerate(parsed_msgs):
        if msg.get("msg_type") == "telemetry":
            last_tel_idx = i

    if last_tel_idx < 0:
        return list(parsed_msgs), []

    ordered: list[dict[str, Any]] = []
    superseded: list[dict[str, Any]] = []
    for i, msg in enumerate(parsed_msgs):
        if msg.get("msg_type") == "telemetry" and i != last_tel_idx:
            superseded.append(msg)
        else:
            ordered.append(msg)
    return ordered, superseded


def add_control_fix_args(parser: argparse.ArgumentParser) -> None:
    """Register opt-in low-jitter sync & gated reversal-unwind CLI flags (defaults OFF)."""
    parser.add_argument(
        "--low-jitter-sync",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable event-driven sync and TCP telemetry burst coalescing (opt-in; default off).",
    )
    parser.add_argument(
        "--reversal-unwind-scale",
        type=float,
        default=1.0,
        help="Opt-in multiplier for max_steering_diff only when reversing toward 0 under excursion (1.0=disabled).",
    )
    parser.add_argument(
        "--reversal-cte-threshold",
        type=float,
        default=0.40,
        help="Min observed |CTE| to enable gated reversal unwind acceleration.",
    )
    parser.add_argument(
        "--reversal-head-deg-threshold",
        type=float,
        default=8.0,
        help="Min observed |heading| (deg) to enable gated reversal unwind acceleration.",
    )
    parser.add_argument(
        "--reversal-min-last-steer",
        type=float,
        default=0.30,
        help="Min |last_steer| required before gated reversal unwind triggers.",
    )


def maybe_install_low_jitter_sync(env: Any, enabled: bool) -> LowJitterSyncController | None:
    """Install `LowJitterSyncController` on a Donkey gym env if `enabled` and viewer is ready."""
    if not enabled:
        return None
    unwrapped = getattr(env, "unwrapped", env)
    viewer = getattr(unwrapped, "viewer", None)
    if viewer is None or getattr(viewer, "handler", None) is None or getattr(viewer, "client", None) is None:
        return None
    return LowJitterSyncController(viewer, enabled=True).install()


def configure_env_control_fixes(env: Any, args: Any) -> LowJitterSyncController | None:
    """Apply opt-in control fixes from CLI args onto `DonkeyVaeSACEnv` (no-op at defaults)."""
    if hasattr(env, "set_control_fixes"):
        env.set_control_fixes(
            reversal_unwind_scale=float(getattr(args, "reversal_unwind_scale", 1.0)),
            reversal_cte_threshold=float(getattr(args, "reversal_cte_threshold", 0.40)),
            reversal_head_deg_threshold=float(getattr(args, "reversal_head_deg_threshold", 8.0)),
            reversal_min_last_steer=float(getattr(args, "reversal_min_last_steer", 0.30)),
        )
    sync = maybe_install_low_jitter_sync(env, enabled=bool(getattr(args, "low_jitter_sync", False)))
    if sync is not None and hasattr(env, "attach_sync_controller"):
        env.attach_sync_controller(sync)
    return sync


class LowJitterSyncController:
    """Toggleable low-jitter telemetry & command wrapper for `DonkeyUnitySimContoller`.

    When `enabled=False`, delegates to stock `DonkeyUnitySimHandler` / `SDClient`
    behavior (`time.sleep(0.001)` polling, per-frame queue writes, no coalescing).
    When `enabled=True`:
    - Replaces `1 ms` `observe()` polling with `threading.Condition` wakeup.
    - Eliminates torn reads by updating all telemetry attributes and image before
      publishing the new sequence number.
    - Coalesces queued `"telemetry"` bursts within a TCP `recv()` drain so stale
      intermediate frames do not trigger fake rapid back-to-back steps.
    - Uses a pipe wakeup fd + zero-sleep `select()` for sub-millisecond socket sends.
    """

    def __init__(self, sim_controller: Any, *, enabled: bool = False) -> None:
        self.sim_controller = sim_controller
        self.handler = sim_controller.handler
        self.client = sim_controller.client
        self.enabled = bool(enabled)

        self._cond = threading.Condition()
        self._send_lock = threading.Lock()
        self._inflight_bytes: bytes = b""
        self._inflight_offset: int = 0
        self._recv_pause_until: float = 0.0
        self._telemetry_seq = 0
        self._consumed_seq = 0
        self._wake_r, self._wake_w = os.pipe()
        os.set_blocking(self._wake_r, False)
        os.set_blocking(self._wake_w, False)

        self.coalesced_dropped = 0
        self.max_burst_seen = 1

        self._orig_on_telemetry: Callable[[dict[str, Any]], None] = self.handler.on_telemetry
        self._orig_observe: Callable[[], tuple[Any, float, bool, dict[str, Any]]] = self.handler.observe
        self._orig_reset: Callable[[], None] = self.handler.reset
        self._orig_queue_message: Callable[[dict[str, Any]], None] = self.handler.queue_message
        self._installed = False

    def install(self) -> "LowJitterSyncController":
        if self._installed:
            return self
        self._installed = True

        # Swap client proc_msg thread cleanly without closing the connected TCP socket.
        client = self.client
        if client.th is not None and client.th.is_alive():
            client.do_process_msgs = False
            client.th.join(timeout=2.0)

        self.handler.on_telemetry = self._wrapped_on_telemetry
        self.handler.fns["telemetry"] = self._wrapped_on_telemetry
        self.handler.observe = self._wrapped_observe
        self.handler.reset = self._wrapped_reset
        self.handler.queue_message = self._wrapped_queue_message

        client.do_process_msgs = True
        client.th = threading.Thread(target=self._proc_msg_loop, args=(client.s,), daemon=True)
        client.th.start()
        return self

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.coalesced_dropped = 0
        self.max_burst_seen = 1
        self._recv_pause_until = 0.0
        with self._cond:
            self._consumed_seq = self._telemetry_seq
            self._cond.notify_all()
        self._signal_wake()

    def pause_recv(self, duration_sec: float) -> None:
        """Temporarily defer socket recv() draining to simulate TCP receive backlog."""
        if duration_sec <= 0.0:
            return
        self._recv_pause_until = max(self._recv_pause_until, time.monotonic() + float(duration_sec))
        self._signal_wake()

    def reset_stats(self) -> None:
        """Reset per-episode burst coalescing counters."""
        self.coalesced_dropped = 0
        self.max_burst_seen = 0

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        for fd in (self._wake_r, self._wake_w):
            try:
                os.close(fd)
            except OSError:
                pass

    def _signal_wake(self) -> None:
        try:
            os.write(self._wake_w, b"\x01")
        except (BlockingIOError, OSError):
            pass

    def _drain_wake(self) -> None:
        while True:
            try:
                chunk = os.read(self._wake_r, 1024)
                if not chunk:
                    break
            except BlockingIOError:
                break
            except OSError:
                break

    def _wrapped_queue_message(self, msg: dict[str, Any]) -> None:
        if not self.enabled:
            self._orig_queue_message(msg)
            return
        payload = json.dumps(msg)
        with self._send_lock:
            self.client.msg = payload
        self._signal_wake()

    def _wrapped_reset(self) -> None:
        self._orig_reset()
        with self._cond:
            self._consumed_seq = self._telemetry_seq

    def _apply_superseded_telemetry(self, message: dict[str, Any]) -> None:
        """Preserve collision / termination flags from intermediate burst packets."""
        h = self.handler
        if h.over:
            return
        if "cte" in message:
            h.cte = message["cte"]
        if "hit" in message:
            h.hit = message["hit"]
        h.determine_episode_over()

    def _apply_telemetry_atomic(self, message: dict[str, Any]) -> None:
        h = self.handler
        img_string = message["image"]
        image = Image.open(BytesIO(base64.b64decode(img_string)))
        img_arr = np.asarray(image)

        img_arr_b = None
        if "image_b" in message:
            image_b = Image.open(BytesIO(base64.b64decode(message["image_b"])))
            img_arr_b = np.asarray(image_b)

        with self._cond:
            h.image_array = img_arr
            if img_arr_b is not None:
                h.image_array_b = img_arr_b

            if "pos_x" in message:
                h.x = message["pos_x"]
                h.y = message["pos_y"]
                h.z = message["pos_z"]
            if "speed" in message:
                h.speed = message["speed"]
            if "gyro_x" in message:
                h.gyro_x = message["gyro_x"]
                h.gyro_y = message["gyro_y"]
                h.gyro_z = message["gyro_z"]
            if "accel_x" in message:
                h.accel_x = message["accel_x"]
                h.accel_y = message["accel_y"]
                h.accel_z = message["accel_z"]
            if "vel_x" in message:
                h.vel_x = message["vel_x"]
                h.vel_y = message["vel_y"]
                h.vel_z = message["vel_z"]
            if "roll" in message:
                h.roll = message["roll"]
                h.pitch = message["pitch"]
                h.yaw = message["yaw"]
            if "cte" in message:
                h.cte = message["cte"]
            if "lidar" in message:
                h.lidar = h.process_lidar_packet(message["lidar"])

            # Compute forward_vel consistently after vel and orientation are updated.
            from gym_donkeycar.envs.donkey_sim import euler_to_quat, rotate_vec

            e = [h.pitch * np.pi / 180.0, h.yaw * np.pi / 180.0, h.roll * np.pi / 180.0]
            q = euler_to_quat(e)
            forward = rotate_vec(q, [0.0, 0.0, 1.0])
            h.forward_vel = forward[0] * h.vel_x + forward[1] * h.vel_y + forward[2] * h.vel_z

            if not h.over:
                if "hit" in message:
                    h.hit = message["hit"]
                h.determine_episode_over()

            h.time_received = time.time()
            self._telemetry_seq += 1
            self._cond.notify_all()

    def _wrapped_on_telemetry(self, message: dict[str, Any]) -> None:
        if not self.enabled:
            self._orig_on_telemetry(message)
            with self._cond:
                self._telemetry_seq += 1
                self._cond.notify_all()
            return
        self._apply_telemetry_atomic(message)

    def _wrapped_observe(self) -> tuple[Any, float, bool, dict[str, Any]]:
        if not self.enabled:
            return self._orig_observe()

        h = self.handler
        with self._cond:
            while self._consumed_seq == self._telemetry_seq:
                if not self.enabled:
                    break
                if self.client.aborted:
                    raise RuntimeError("Donkey simulator socket aborted while waiting for observation")
                self._cond.wait(timeout=0.05)
            if not self.enabled:
                return self._orig_observe()
            self._consumed_seq = self._telemetry_seq
            h.last_received = h.time_received
            observation = h.image_array
            done = h.is_game_over()
            reward = h.calc_reward(done)
            info = {
                "pos": (h.x, h.y, h.z),
                "cte": h.cte,
                "speed": h.speed,
                "forward_vel": h.forward_vel,
                "hit": h.hit,
                "gyro": (h.gyro_x, h.gyro_y, h.gyro_z),
                "accel": (h.accel_x, h.accel_y, h.accel_z),
                "vel": (h.vel_x, h.vel_y, h.vel_z),
                "lidar": h.lidar,
                "car": (h.roll, h.pitch, h.yaw),
                "last_lap_time": h.last_lap_time,
                "lap_count": h.lap_count,
            }
            if h.image_array_b is not None:
                info["image_b"] = h.image_array_b
        return observation, reward, done, info

    def _has_pending_send(self) -> bool:
        with self._send_lock:
            return self._inflight_offset < len(self._inflight_bytes) or self.client.msg is not None

    def _flush_pending_send(self, sock: Any) -> int:
        total_sent = 0
        with self._send_lock:
            while True:
                if self._inflight_offset >= len(self._inflight_bytes):
                    msg = self.client.msg
                    if msg is None:
                        break
                    self._inflight_bytes = msg.encode("utf-8")
                    self._inflight_offset = 0
                    self.client.msg = None
                view = memoryview(self._inflight_bytes)[self._inflight_offset :]
                try:
                    sent = sock.send(view)
                except BlockingIOError:
                    break
                if sent <= 0:
                    raise ConnectionAbortedError("Socket send returned 0 bytes")
                self._inflight_offset += sent
                total_sent += sent
                if self._inflight_offset >= len(self._inflight_bytes):
                    self._inflight_bytes = b""
                    self._inflight_offset = 0
        return total_sent

    def _proc_msg_loop(self, sock: Any) -> None:
        sock.setblocking(False)
        localbuffer = ""
        utf8_decoder = codecs.getincrementaldecoder("utf-8")("strict")

        while self.client.do_process_msgs:
            paused = time.monotonic() < self._recv_pause_until
            if not self.enabled:
                time.sleep(self.client.poll_socket_sleep_sec)
                try:
                    inputs = [] if paused else [sock]
                    readable, writable, exceptional = select.select(inputs, [sock], [sock], 0.005)
                    for _ in readable:
                        data = utf8_decoder.decode(sock.recv(1024 * 256))
                        localbuffer += data
                        msgs, localbuffer = split_json_lines_buffer(localbuffer)
                        tel_in_batch = 0
                        for m in msgs:
                            m = replace_float_notation(m)
                            try:
                                j = json.loads(m)
                            except Exception:
                                continue
                            if "msg_type" in j:
                                if j.get("msg_type") == "telemetry":
                                    tel_in_batch += 1
                                self.client.on_msg_recv(j)
                        if tel_in_batch > 1:
                            self.max_burst_seen = max(self.max_burst_seen, tel_in_batch)
                    for _ in writable:
                        self._flush_pending_send(sock)
                    if exceptional:
                        logger.error("problems w sockets!")
                except Exception as exc:
                    print("Exception:", exc)
                    self.client.aborted = True
                    self.client.on_msg_recv({"msg_type": "aborted"})
                    break
                continue

            # Sync-enabled event-driven path: zero sleep polling, flush sends immediately,
            # drain all available TCP chunks and coalesce queued telemetry packets.
            try:
                inputs = [self._wake_r] if paused else [sock, self._wake_r]
                outputs = [sock] if self._has_pending_send() else []
                timeout = 0.005
                if paused:
                    timeout = max(0.001, min(0.005, self._recv_pause_until - time.monotonic()))
                readable, writable, exceptional = select.select(inputs, outputs, [sock], timeout)

                if self._wake_r in readable:
                    self._drain_wake()

                if writable:
                    self._flush_pending_send(sock)

                if sock in readable and time.monotonic() >= self._recv_pause_until:
                    while True:
                        try:
                            chunk = sock.recv(1024 * 256)
                        except BlockingIOError:
                            break
                        if not chunk:
                            raise ConnectionAbortedError("Socket closed by simulator")
                        localbuffer += utf8_decoder.decode(chunk)

                    msgs, localbuffer = split_json_lines_buffer(localbuffer)
                    if msgs:
                        parsed: list[dict[str, Any]] = []
                        for m in msgs:
                            m = replace_float_notation(m)
                            try:
                                j = json.loads(m)
                            except Exception:
                                continue
                            if "msg_type" in j:
                                parsed.append(j)
                        ordered, superseded = coalesce_telemetry_messages(parsed)
                        if superseded:
                            self.coalesced_dropped += len(superseded)
                            self.max_burst_seen = max(self.max_burst_seen, len(superseded) + 1)
                            for old_tel in superseded:
                                self._apply_superseded_telemetry(old_tel)
                        for evt in ordered:
                            self.client.on_msg_recv(evt)

                    # Flush any queued send generated immediately before/during recv handling.
                    self._flush_pending_send(sock)

                if exceptional:
                    logger.error("problems w sockets!")
            except Exception as exc:
                print("Exception:", exc)
                self.client.aborted = True
                self.client.on_msg_recv({"msg_type": "aborted"})
                break
