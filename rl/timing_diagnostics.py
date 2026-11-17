"""Low-volume GC pause records; callbacks never perform I/O in the control loop."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import threading
import time


class GCPauseRecorder:
    def __init__(self):
        self.events = []
        self.step = 0
        self.phase = "setup"
        self.started = {}
        self._writer = None
        self._stop = threading.Event()

    def callback(self, phase, info):
        generation = info["generation"]
        now = time.perf_counter()
        if phase == "start":
            self.started[generation] = (now, self.step, self.phase, threading.get_ident())
        elif generation in self.started:
            start, step, location, thread = self.started.pop(generation)
            self.events.append({"start_mono_s": start, "end_mono_s": now,
                                "duration_ms": (now-start)*1000, "generation": generation,
                                "step": step, "phase": location, "thread_id": thread,
                                "collected": info["collected"]})

    def install(self):
        gc.callbacks.append(self.callback)

    def start_writer(self, path):
        """Flush diagnostics from a background thread, never from a GC callback."""
        path = Path(path)
        def write_events():
            cursor = 0
            with path.open("w", buffering=1) as f:
                while True:
                    stopped = self._stop.wait(0.5)
                    pending = self.events[cursor:]
                    for event in pending:
                        f.write(json.dumps(event) + "\n")
                    cursor += len(pending)
                    progress = {"step": self.step, "phase": self.phase, "gc_events": cursor,
                                "largest_gc_pause_ms": max((e["duration_ms"] for e in self.events), default=0)}
                    path.with_name("diagnostic_progress.json").write_text(json.dumps(progress) + "\n")
                    if stopped:
                        break
        self._writer = threading.Thread(target=write_events, daemon=True)
        self._writer.start()

    def close(self):
        if self.callback in gc.callbacks:
            gc.callbacks.remove(self.callback)
        self._stop.set()
        if self._writer is not None:
            self._writer.join(timeout=2)
