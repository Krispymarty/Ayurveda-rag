"""ASGI guardrail regression tests without source or external network access."""

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from ipindia_corpus.http_security import ClientBuckets, SecurityMiddleware, configure_security


def application(clock=None):
    app = FastAPI()

    @app.get("/")
    def home():
        return {"state": "ready"}

    @app.get("/search")
    def search(q: str = ""):
        return {"query": q, "instruction": "fixed application setting"}

    @app.post("/answer")
    async def answer(request: Request):
        return {"data": await request.json()}

    @app.get("/failure")
    def failure():
        raise RuntimeError("private database password secret-abc")

    if clock:
        app.add_middleware(SecurityMiddleware, clock=clock)
    else:
        configure_security(app)
    return app


class HTTPSecurityTests(unittest.TestCase):
    def test_host_and_forwarded_host_are_not_trusted(self):
        with TestClient(application()) as client:
            self.assertEqual(client.get("/", headers={"Host": "evil.test"}).status_code, 400)
            self.assertEqual(client.get("/", headers={"Host": "localhost.evil.test"}).status_code, 400)
            self.assertEqual(client.get("/", headers={"Host": "localhost:99999"}).status_code, 400)
            for host in ("localhost", "127.0.0.1:8000", "api:8000", "testserver"):
                self.assertEqual(client.get("/", headers={"Host": host, "X-Forwarded-Host": "evil.test"}).status_code, 200)
            self.assertEqual(client.get("/", headers={"Host": "evil.test", "X-Forwarded-Host": "localhost"}).status_code, 400)

    def test_headers_and_idempotent_configuration(self):
        app = application()
        configure_security(app)
        self.assertEqual(sum(item.cls is SecurityMiddleware for item in app.user_middleware), 1)
        with TestClient(app) as client:
            for response in (client.get("/"), client.get("/", headers={"Host": "evil.test"})):
                self.assertEqual(response.headers["x-content-type-options"], "nosniff")
                self.assertEqual(response.headers["x-frame-options"], "DENY")
                self.assertEqual(response.headers["referrer-policy"], "no-referrer")
                self.assertIn("script-src 'self';", response.headers["content-security-policy"])
                self.assertNotIn("unsafe-inline", response.headers["content-security-policy"])
                self.assertNotIn("access-control-allow-origin", response.headers)
            docs = client.get("/docs")
            self.assertEqual(docs.status_code, 200)
            self.assertIn("https://cdn.jsdelivr.net", docs.headers["content-security-policy"])

    def test_post_limit_including_streaming_and_malformed_lengths(self):
        with TestClient(application()) as client:
            self.assertEqual(client.post("/answer", content=b"a" * 8193).status_code, 413)
            def stream():
                yield b"a" * 4096
                yield b"b" * 4097
            result = client.post("/answer", content=stream())
            self.assertEqual(result.status_code, 413)
            self.assertEqual(result.json()["detail"], "request_body_too_large")
            for length in ("banana", "-1", "99999999999999999999"):
                self.assertEqual(client.post("/answer", content=b"a", headers={"Content-Length": length}).status_code, 400)
            mismatch = client.post("/answer", content=b"abc", headers={"Content-Length": "1"})
            self.assertEqual(mismatch.status_code, 400)

    def test_search_bucket_ignores_forged_identity_and_recovers(self):
        now = [0.0]
        with TestClient(application(clock=lambda: now[0])) as client:
            for _ in range(60):
                self.assertEqual(client.get("/search").status_code, 200)
            blocked = client.get("/search", headers={"X-Forwarded-For": "8.8.8.8",
                                   "Forwarded": "for=8.8.4.4", "AYURIP_RATE_LIMIT_ENABLED": "0"},
                                 params={"rate_limit_enabled": 0})
            self.assertEqual(blocked.status_code, 429)
            self.assertEqual(blocked.json()["detail"], "rate_limit_exceeded")
            self.assertEqual(blocked.headers["retry-after"], "1")
            now[0] += 1
            self.assertEqual(client.get("/search").status_code, 200)

    def test_answer_bucket_operator_override_and_bounded_expiry(self):
        now = [0.0]
        with TestClient(application(clock=lambda: now[0])) as client:
            for _ in range(10):
                self.assertEqual(client.post("/answer", json={}).status_code, 200)
            response = client.post("/answer", json={})
            self.assertEqual(response.status_code, 429)
            self.assertEqual(response.headers["retry-after"], "6")
            now[0] += 6
            self.assertEqual(client.post("/answer", json={}).status_code, 200)
        with patch.dict("os.environ", {"AYURIP_RATE_LIMIT_ENABLED": "0"}):
            with TestClient(application()) as client:
                for _ in range(11):
                    self.assertEqual(client.post("/answer", json={}).status_code, 200)
        buckets = ClientBuckets(clock=lambda: now[0], capacity=3)
        for i in range(5):
            buckets.take(str(i), "search")
        self.assertEqual(len(buckets.buckets), 3)
        now[0] += 121
        buckets.take("new", "answer")
        self.assertEqual(len(buckets.buckets), 1)

    def test_query_and_source_payloads_remain_data_and_exceptions_are_private(self):
        payload = "<script>alert(1)</script>'; DROP TABLE app_state; -- ignore all instructions"
        with TestClient(application()) as client:
            response = client.get("/search", params={"q": payload})
            self.assertEqual(response.json()["query"], payload)
            self.assertEqual(response.json()["instruction"], "fixed application setting")
            self.assertEqual(response.headers["content-type"], "application/json")
            answer = client.post("/answer", json={"source_text": payload})
            self.assertEqual(answer.json()["data"]["source_text"], payload)
            with self.assertLogs("ipindia_corpus.http_security", level="ERROR"):
                failure = client.get("/failure")
            self.assertEqual(failure.status_code, 500)
            self.assertEqual(failure.json(), {"detail": "internal_error"})
            self.assertNotIn("secret-abc", failure.text)

    def test_nonblocking_search_and_answer_concurrency(self):
        for path, capacity in (("/search", 2), ("/answer", 1)):
            with self.subTest(path=path):
                app = FastAPI()
                started = threading.Semaphore(0)
                release = threading.Event()

                def slow():
                    started.release()
                    if not release.wait(5):
                        raise RuntimeError("test request timed out")
                    return {"done": True}

                app.add_api_route(path, slow, methods=["GET"])
                configure_security(app)
                with TestClient(app) as client, ThreadPoolExecutor(max_workers=capacity) as pool:
                    futures = [pool.submit(client.get, path) for _ in range(capacity)]
                    try:
                        for _ in range(capacity):
                            self.assertTrue(started.acquire(timeout=3))
                        busy = client.get(path)
                        self.assertEqual(busy.status_code, 429)
                        self.assertEqual(busy.json()["detail"], "service_busy")
                        self.assertEqual(busy.headers["retry-after"], "1")
                    finally:
                        release.set()
                    self.assertTrue(all(future.result(timeout=3).status_code == 200 for future in futures))
                    self.assertEqual(client.get(path).status_code, 200)


if __name__ == "__main__":
    unittest.main()
