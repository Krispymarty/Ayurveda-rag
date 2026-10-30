"""Single-flight bounded initialization. A late worker cannot publish readiness."""
import logging
import threading
import time

logger = logging.getLogger(__name__)


class Readiness:
    def __init__(self, loader, deadline=60):
        self.loader, self.deadline = loader, deadline
        self.lock = threading.Lock()
        self.state, self.reason, self.value = "not_started", "search_initializing", None
        self.started = None
        self.finished = None
        self.timings = {}

    def start(self):
        with self.lock:
            if self.state != "not_started":
                return
            self.state, self.started = "initializing", time.monotonic()
        threading.Thread(target=self._run, daemon=True, name="ayurip-initialize").start()

    def _run(self):
        try:
            value = self.loader(self.record)
            with self.lock:
                if self.state == "initializing" and time.monotonic() - self.started <= self.deadline:
                    self.value, self.state, self.reason = value, "ready", None
                    self.finished = time.monotonic()
                elif self.state == "initializing":
                    self.state, self.reason = "failed", "initialization_timeout"
                    self.finished = time.monotonic()
        except Exception:
            logger.exception("Search initialization failed")
            with self.lock:
                if self.state == "initializing":
                    self.state, self.reason = "failed", "initialization_failed"
                    self.finished = time.monotonic()

    def record(self, stage, seconds):
        with self.lock:
            self.timings[stage] = seconds * 1000

    def status(self):
        with self.lock:
            elapsed = (self.finished or time.monotonic()) - self.started if self.started is not None else 0
            if self.state == "initializing" and elapsed > self.deadline:
                self.state, self.reason = "failed", "initialization_timeout"
                self.finished = time.monotonic()
            return {"state": self.state, "reason": self.reason, "initialization_ms": elapsed * 1000,
                    "stage_ms": dict(self.timings)}

    def require(self):
        status = self.status()
        if status["state"] != "ready":
            raise ValueError(status["reason"] or "search_unavailable")
        return self.value

    def invalidate(self, reason="active_binding_changed"):
        with self.lock:
            self.state, self.reason, self.value = "failed", reason, None

    def close(self):
        self.invalidate("search_stopped")
