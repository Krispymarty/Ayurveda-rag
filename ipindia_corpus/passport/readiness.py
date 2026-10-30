"""Private initialization only; no corpus changes, retries of answers or rebinding."""
import logging
import threading
import time

from ..reliable_answers import MODEL

logger = logging.getLogger(__name__)


class EngineReadiness:
    def __init__(self, *, engine=None, factory=None, clock=time.monotonic):
        self.engine, self.factory, self.clock = engine, factory, clock
        self.lock = threading.Lock()
        self.worker = None
        self.pending = False
        self.started = None
        self.next_retry = 0.0
        self.reason = None if engine else "search_dependency_unavailable"
        self.binding_failed = False

    def _expire(self):
        if self.pending and self.clock() - self.started >= 60:
            self.reason = "search_initialization_timeout"

    def retry(self):
        """Only startup/session can request recovery; one live worker at a time."""
        with self.lock:
            self._expire()
            if self.engine is not None or self.pending or self.clock() < self.next_retry:
                return
            self.pending = True
            self.started = self.clock()
            self.reason = "search_initializing"
            self.worker = threading.Thread(target=self._initialize, name="passport-init", daemon=True)
            self.worker.start()

    def _initialize(self):
        candidate, failure = None, None
        try:
            candidate = self.factory()
            if not candidate.ready():
                raise ValueError("initial_binding_unavailable")
        except Exception as exc:
            # Only an exception class is logged; no paths, settings, intake or text.
            logger.warning("Passport initialization failed (%s)", type(exc).__name__)
            failure = "search_dependency_unavailable"
        with self.lock:
            self._expire()
            if self.reason == "search_initialization_timeout":
                logger.warning("Passport initialization exceeded deadline; result discarded")
            elif failure:
                self.reason = failure
            else:
                self.engine, self.reason = candidate, None
            self.pending = False
            self.next_retry = self.clock() + 5

    def snapshot(self):
        with self.lock:
            self._expire()
            engine = self.engine
            reason = self.reason
            failed = self.binding_failed
        ready = False
        if engine is not None and not failed:
            try:
                ready = engine.ready()
                reason = None if ready else "active_generation_changed"
                if not ready:
                    with self.lock:
                        self.binding_failed = True
            except Exception:
                reason = "search_dependency_unavailable"
        with self.lock:
            # A simultaneous request may already have observed a generation change.
            if self.binding_failed:
                ready, reason = False, "active_generation_changed"
        return {"ready": ready, "reason": reason,
                "generation_id": engine.gid if engine else None, "model": MODEL}

    def available_engine(self):
        return self.engine if self.snapshot()["ready"] else None
