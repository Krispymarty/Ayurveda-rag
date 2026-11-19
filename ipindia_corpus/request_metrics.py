"""Request-scoped timings. No questions, passages or provider payloads."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import logging
from time import perf_counter
from uuid import uuid4


logger = logging.getLogger("uvicorn.error")
MEASURED_PATHS = frozenset({"/search", "/answer", "/health"})
TRACE = ContextVar("ayurip_request_metrics", default=None)


@dataclass
class Trace:
    timings: dict[str, list[float]] = field(default_factory=dict)
    outcomes: list[str] = field(default_factory=list)


@contextmanager
def stage(name):
    trace = TRACE.get()
    if trace is None:
        yield
        return
    started = perf_counter()
    try:
        yield
    finally:
        trace.timings.setdefault(name, []).append(round((perf_counter() - started) * 1000, 3))


def outcome(category):
    # Callers provide fixed categories, never exception strings or arguments.
    trace = TRACE.get()
    if trace is not None and len(trace.outcomes) < 12:
        trace.outcomes.append(category)


class RequestMetricsMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        path = scope.get("path")
        if scope.get("type") != "http" or path not in MEASURED_PATHS:
            await self.app(scope, receive, send)
            return
        request_id = uuid4().hex
        trace = Trace()
        token = TRACE.set(trace)
        started = perf_counter()
        status = 500

        async def measured_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = [(key, value) for key, value in message.get("headers", [])
                           if key.lower() != b"x-request-id"]
                message = {**message, "headers": headers + [(b"x-request-id", request_id.encode("ascii"))]}
            await send(message)

        try:
            await self.app(scope, receive, measured_send)
        finally:
            TRACE.reset(token)
            logger.info("ayurip_http %s", json.dumps({
                "request_id": request_id, "route": path, "status": status,
                "elapsed_ms": round((perf_counter() - started) * 1000, 3),
                "stages_ms": trace.timings, "outcomes": trace.outcomes,
            }, separators=(",", ":")))
