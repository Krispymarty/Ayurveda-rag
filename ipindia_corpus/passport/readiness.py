"""Private initialization only; no corpus changes, retries of answers or rebinding."""
import logging
import threading
import time

from ..inference import configured_identity, provider_identity, provider_binding_valid
from ..llm_experiment import ExperimentFailure

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
        self.binding_reason = None

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
            failure = (str(exc) if isinstance(exc, ExperimentFailure)
                       and str(exc) in {"gemini_key_missing", "gemini_configuration_invalid"}
                       else "search_dependency_unavailable")
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
                binding_reason = getattr(engine, "binding_reason", None)
                reason = (None if ready else binding_reason
                          if binding_reason in {"active_generation_changed", "provider_model_changed"}
                          else "active_generation_changed")
                if not ready:
                    with self.lock:
                        if not self.binding_failed:
                            self.binding_reason = reason
                        self.binding_failed = True
            except Exception:
                reason = "search_dependency_unavailable"
        with self.lock:
            # Keep the first observed binding failure across concurrent requests.
            if self.binding_failed:
                ready, reason = False, self.binding_reason or "active_generation_changed"
        try:
            identity = (provider_identity(getattr(engine, "provider", None))
                        if engine is not None else configured_identity())
        except (ValueError, ExperimentFailure):
            ready, reason = False, "gemini_configuration_invalid"
            identity = {"provider": "unavailable", "model": None,
                        "model_version": None, "model_digest": None,
                        "external_processing": False}
        with self.lock:
            # Identity inspection cannot erase a concurrently latched failure.
            if self.binding_failed:
                ready, reason = False, self.binding_reason or "active_generation_changed"
        fallback = getattr(engine, 'fallback_provider', None) if engine is not None else None
        circuit = getattr(engine, 'circuit', None) if engine is not None else None
        return {"ready": ready, "reason": reason,
                "generation_id": engine.gid if engine else None,
                "model": identity.get("model"), "provider": identity,
                "external_processing": identity.get("external_processing") is True,
                "fallback": {"configured": fallback is not None,
                             "provider": provider_identity(fallback) if fallback is not None else None,
                             "availability": ("binding_failed" if fallback is not None and not provider_binding_valid(fallback)
                                              else "checked_on_use" if fallback is not None else "disabled"),
                             "circuit": circuit.snapshot() if circuit is not None else None}}

    def available_engine(self):
        return self.engine if self.snapshot()["ready"] else None
