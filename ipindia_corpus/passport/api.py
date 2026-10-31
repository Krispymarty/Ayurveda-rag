"""Opt-in loopback UI/API; never registered on search_api.app.

PowerShell: $env:AYURIP_PASSPORT_DEV='1'; python -m ipindia_corpus.passport.api --serve
"""
import argparse
from contextlib import asynccontextmanager
import ipaddress
import json
import os
from pathlib import Path
import secrets
import threading
import time

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from ..grounded_answers import _unique_object
from ..http_security import ClientBuckets, configure_security
from ..sync_runner import ROOT
from .service import Intake, PassportEngine
from .readiness import EngineReadiness

COOKIE = "ayurip_passport"


class PrivateBoundary:
    def __init__(self, app, token):
        self.app, self.token = app, token
        self.buckets = ClientBuckets()
        self.gate = threading.BoundedSemaphore(1)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def reject(reason, status=403):
            await JSONResponse({"reason": reason}, status_code=status, headers={"Cache-Control": "no-store"})(scope, receive, send)

        async def private_send(message):
            if message["type"] == "http.response.start":
                message = {**message, "headers": [*message.get("headers", []), (b"cache-control", b"no-store")]}
            await send(message)

        peer = scope.get("client")
        try:
            if not peer or not ipaddress.ip_address(peer[0]).is_loopback:
                return await reject("loopback_required")
        except ValueError:
            return await reject("loopback_required")
        headers = scope.get("headers", [])

        def values(key):
            return [value for name, value in headers if name.lower() == key]

        hosts = values(b"host")
        if len(hosts) != 1 or hosts[0].lower() not in {b"127.0.0.1:8003", b"localhost:8003"}:
            return await reject("private_host_required", 400)
        if any(name.lower() in {b"forwarded", b"x-real-ip", b"via"} or name.lower().startswith(b"x-forwarded-") for name, _ in headers):
            return await reject("proxy_headers_not_allowed")
        origin = b"http://" + hosts[0].lower()
        origins = values(b"origin")
        if origins and (len(origins) != 1 or origins[0] != origin):
            return await reject("same_origin_required")
        if any(value not in {b"same-origin", b"none"} for value in values(b"sec-fetch-site")):
            return await reject("same_origin_required")
        acquired = False
        if scope["method"] == "POST" and scope["path"] == "/api/passport":
            if origins != [origin]:
                return await reject("same_origin_required")
            content = values(b"content-type")
            if len(content) != 1 or content[0].split(b";", 1)[0].strip().lower() != b"application/json":
                return await reject("json_required", 415)
            tokens, cookies = values(b"x-passport-token"), values(b"cookie")
            if len(tokens) != 1 or len(cookies) != 1:
                return await reject("private_session_required")
            matches = [part.strip().split(b"=", 1)[1] for part in cookies[0].split(b";")
                       if part.strip().startswith(COOKIE.encode() + b"=")]
            if (len(matches) != 1 or not secrets.compare_digest(tokens[0], self.token.encode())
                    or not secrets.compare_digest(matches[0], self.token.encode())):
                return await reject("private_session_required")
            retry = self.buckets.take(peer[0], "answer")
            if retry:
                return await reject("rate_limit_exceeded", 429)
            if not self.gate.acquire(blocking=False):
                return await reject("service_busy", 429)
            acquired = True
        try:
            await self.app(scope, receive, private_send)
        finally:
            if acquired:
                self.gate.release()


def create_app(*, engine=None, enabled=None, readiness=None):
    if enabled is None:
        enabled = os.environ.get("AYURIP_PASSPORT_DEV") == "1"
    prefetch = os.environ.get("AYURIP_PASSPORT_PREFETCH") == "1"
    state = readiness or EngineReadiness(engine=engine, factory=lambda: PassportEngine.connect(prefetch=prefetch))

    @asynccontextmanager
    async def lifespan(app):
        if enabled:
            state.retry()
        yield

    app = FastAPI(title="AyurIP private Evidence Passport", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    if not enabled:
        return app
    token = secrets.token_urlsafe(32)
    configure_security(app)
    app.add_middleware(PrivateBoundary, token=token)
    ui = Path(__file__).parent / "ui"

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(ui / "index.html")

    @app.get("/assets/{name}", include_in_schema=False)
    def asset(name: str):
        allowed = {"passport.js": ui / "passport.js", "passport.css": ui / "passport.css", "base.css": ROOT / "web/styles.css"}
        if name not in allowed:
            return JSONResponse(status_code=404, content={"reason": "not_found"})
        return FileResponse(allowed[name])

    @app.get("/api/session")
    def session():
        state.retry()
        response = JSONResponse({"csrf_token": token, **state.snapshot()})
        response.set_cookie(COOKIE, token, httponly=True, samesite="strict", path="/")
        return response

    @app.get("/health")
    def health():
        snapshot = state.snapshot()
        return JSONResponse(snapshot, status_code=200 if snapshot["ready"] else 503)

    @app.get("/live")
    def live():
        return {"status": "alive"}

    @app.post("/api/passport")
    async def build(request: Request):
        intake_started = time.monotonic()
        try:
            raw = (await request.body()).decode("utf-8")
            data = json.loads(raw, object_pairs_hook=_unique_object)
        except (UnicodeError, ValueError, TypeError, RecursionError):
            return JSONResponse(status_code=400, content={"reason": "invalid_json"})
        try:
            intake = Intake.model_validate(data)
        except ValidationError as exc:
            return JSONResponse(status_code=422, content={"reason": "invalid_intake", "fields": sorted({e["loc"][0] for e in exc.errors() if e["loc"] and e["loc"][0] in Intake.model_fields})})
        intake_ms = (time.monotonic() - intake_started) * 1000
        ready_started = time.monotonic()
        active = await run_in_threadpool(state.available_engine)
        ready_ms = (time.monotonic() - ready_started) * 1000
        if active is None:
            snapshot = await run_in_threadpool(state.snapshot)
            return JSONResponse(status_code=503, content={"reason": snapshot["reason"]})
        try:
            result = await run_in_threadpool(active.build, intake)
        except Exception:
            # No intake, model completion, archive text or exception details in logs/response.
            return JSONResponse(status_code=503, content={"reason": "passport_unavailable"})
        result["timings_ms"].update(api_intake_validation=round(intake_ms, 3), api_readiness=round(ready_ms, 3))
        serialized = time.monotonic()
        response = JSONResponse(result, status_code=503 if result["status"] == "unavailable" else 200)
        serialization_ms = (time.monotonic() - serialized) * 1000
        response.headers["Server-Timing"] = f"intake;dur={intake_ms:.3f}, readiness;dur={ready_ms:.3f}, serialization;dur={serialization_ms:.3f}"
        return response

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", required=True, action="store_true")
    parser.parse_args()
    if os.environ.get("AYURIP_PASSPORT_DEV") != "1":
        raise SystemExit("Private passport disabled. Set AYURIP_PASSPORT_DEV=1 explicitly.")
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=8003, proxy_headers=False, access_log=False)
