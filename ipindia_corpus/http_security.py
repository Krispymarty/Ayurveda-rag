"""Small, process-local HTTP guardrails for the internal evidence service."""

from __future__ import annotations

import logging
import math
import os
import re
import threading
import time
from collections import OrderedDict

from starlette.responses import JSONResponse

from .cloud_config import load_deployment_config

logger = logging.getLogger(__name__)
MAX_POST_BYTES = 8192
MAX_CLIENT_BUCKETS = 2048
ALLOWED_HOST = re.compile(r"(?:localhost|127\.0\.0\.1|api|testserver)(?::([0-9]{1,5}))?", re.I)
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; "
       "font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; "
       "frame-ancestors 'none'; form-action 'self'")
DOCS_CSP = ("default-src 'self'; script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
            "style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
            "img-src 'self' https://fastapi.tiangolo.com; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'")


class ClientBuckets:
    """Bounded token buckets keyed only by the transport peer, never proxy headers."""

    def __init__(self, *, clock=time.monotonic, capacity=MAX_CLIENT_BUCKETS):
        self.clock, self.capacity = clock, capacity
        self.buckets = OrderedDict()
        self.lock = threading.Lock()

    def take(self, client: str, route: str) -> int:
        limit = 60 if route == "search" else 10
        rate, now = limit / 60, self.clock()
        key = (client, route)
        with self.lock:
            # LRU order also orders the most recent access timestamps.
            while self.buckets:
                first = next(iter(self.buckets))
                if now - self.buckets[first][1] < 120:
                    break
                self.buckets.popitem(last=False)
            tokens, previous = self.buckets.pop(key, (float(limit), now))
            tokens = min(float(limit), tokens + max(0, now - previous) * rate)
            retry = math.ceil((1 - tokens) / rate) if tokens < 1 else 0
            if not retry:
                tokens -= 1
            self.buckets[key] = (tokens, now)
            while len(self.buckets) > self.capacity:
                self.buckets.popitem(last=False)
            return max(0, retry)


class SecurityMiddleware:
    def __init__(self, app, *, clock=time.monotonic):
        self.app = app
        self.deployment = load_deployment_config()
        # Read operator configuration once, outside request/source data paths.
        self.rate_enabled = self.deployment.public_research or os.environ.get("AYURIP_RATE_LIMIT_ENABLED", "1") != "0"
        self.buckets = ClientBuckets(clock=clock)
        self.gates = {"search": threading.BoundedSemaphore(2),
                      "answer": threading.BoundedSemaphore(1)}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        csp = DOCS_CSP if not self.deployment.public_research and path in {"/docs", "/redoc"} else CSP
        security_headers = {b"content-security-policy": csp.encode("ascii"),
                            b"x-content-type-options": b"nosniff",
                            b"x-frame-options": b"DENY",
                            b"referrer-policy": b"no-referrer",
                            b"permissions-policy": b"camera=(), microphone=(), geolocation=()"}
        if path.rstrip("/") in {"/answer", "/capabilities"}:
            security_headers[b"cache-control"] = b"no-store"
        if self.deployment.public_research:
            security_headers[b"strict-transport-security"] = b"max-age=31536000"
        response_started = False

        async def guarded_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                message = dict(message)
                message["headers"] = [(key, value) for key, value in message.get("headers", [])
                                      if key.lower() not in security_headers]
                message["headers"].extend(security_headers.items())
            await send(message)

        async def reject(code, reason, retry=None):
            headers = {"Retry-After": str(retry)} if retry is not None else None
            await JSONResponse({"detail": reason}, status_code=code, headers=headers)(scope, receive, guarded_send)

        hosts = [value for key, value in scope.get("headers", []) if key.lower() == b"host"]
        try:
            host = hosts[0].decode("ascii") if len(hosts) == 1 else ""
            if not self.deployment.allows_host(host):
                raise ValueError("untrusted host")
        except (UnicodeDecodeError, ValueError):
            await reject(400, "invalid_host")
            return

        origins = [value for key, value in scope.get("headers", []) if key.lower() == b"origin"]
        if self.deployment.public_research and origins:
            try:
                allowed_origin = len(origins) == 1 and self.deployment.allows_origin(origins[0].decode("ascii"))
            except UnicodeDecodeError:
                allowed_origin = False
            if not allowed_origin:
                await reject(403, "invalid_origin")
                return

        if scope.get("method") == "POST":
            if self.deployment.public_research:
                content_types = [value for key, value in scope.get("headers", []) if key.lower() == b"content-type"]
                if len(content_types) != 1 or content_types[0].split(b";", 1)[0].strip().lower() != b"application/json":
                    await reject(415, "application_json_required")
                    return
            lengths = [value for key, value in scope.get("headers", []) if key.lower() == b"content-length"]
            transfer = any(key.lower() == b"transfer-encoding" for key, _ in scope.get("headers", []))
            if lengths and (len(lengths) != 1 or len(lengths[0]) > 10
                            or not lengths[0].isdigit() or transfer):
                await reject(400, "invalid_content_length")
                return
            if lengths and int(lengths[0]) > MAX_POST_BYTES:
                await reject(413, "request_body_too_large")
                return
            body, more = bytearray(), True
            while more:
                message = await receive()
                if message["type"] != "http.request":
                    await reject(400, "incomplete_request_body")
                    return
                body.extend(message.get("body", b""))
                if len(body) > MAX_POST_BYTES:
                    await reject(413, "request_body_too_large")
                    return
                more = message.get("more_body", False)
            if lengths and len(body) != int(lengths[0]):
                await reject(400, "content_length_mismatch")
                return
            delivered = False

            async def replay_receive():
                nonlocal delivered
                if delivered:
                    return await receive()
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}

            application_receive = replay_receive
        else:
            application_receive = receive

        route = {"/search": "search", "/answer": "answer"}.get(path.rstrip("/"))
        gate = None
        if route:
            client = scope.get("client")
            peer = str(client[0]) if client else "unknown-peer"
            # Hosted answers use durable browser/site quotas and a shared DB
            # lock. The transport peer is the same TLS proxy for many browsers.
            use_peer_limit = self.rate_enabled and not (route == "answer" and self.deployment.hosted_answers)
            retry = self.buckets.take(peer, route) if use_peer_limit else 0
            if retry:
                await reject(429, "rate_limit_exceeded", retry)
                return
            gate = self.gates[route]
            if not gate.acquire(blocking=False):
                await reject(429, "service_busy", 1)
                return
        try:
            await self.app(scope, application_receive, guarded_send)
        except Exception:
            logger.exception("HTTP request failed: %s", path)
            if not response_started:
                await reject(500, "internal_error")
            else:
                raise
        finally:
            if gate is not None:
                gate.release()


def configure_security(app) -> None:
    """Install guardrails once. CORS permissions are intentionally not added."""
    if not any(item.cls is SecurityMiddleware for item in app.user_middleware):
        app.add_middleware(SecurityMiddleware)
