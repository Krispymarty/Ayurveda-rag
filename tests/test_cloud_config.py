"""Public deployment boundary fixtures; no cloud approval or source certification."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from ipindia_corpus.cloud_config import load_deployment_config
from ipindia_corpus.http_security import configure_security


class CloudConfigTests(unittest.TestCase):
    def test_default_preserves_local_hosts_and_no_public_origin(self):
        config = load_deployment_config({})
        self.assertFalse(config.public_research)
        for value in ("localhost", "127.0.0.1:8000", "api:8000", "testserver"):
            self.assertTrue(config.allows_host(value))
        for value in ("localhost.evil.test", "localhost:99999", "evil.test"):
            self.assertFalse(config.allows_host(value))

    def test_public_requires_exact_https_origin_and_bounded_host_list(self):
        for origin in ("", "http://research.example", "https://user:pass@research.example",
                       "https://research.example/path", "https://research.example?query=1",
                       "https://research.example#fragment", "https://*.example", "https://research.example:99999",
                       "https://research.example\n", "https://rese\tarch.example"):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                load_deployment_config({"AYURIP_DEPLOYMENT_MODE": "public_research", "AYURIP_PUBLIC_ORIGIN": origin})
        config = load_deployment_config({"AYURIP_DEPLOYMENT_MODE": "public_research",
                                         "AYURIP_PUBLIC_ORIGIN": "https://Research.Example",
                                         "AYURIP_ALLOWED_HOSTS": "localhost:8000,health.internal"})
        self.assertEqual(config.public_origin, "https://research.example")
        for value in ("research.example", "RESEARCH.EXAMPLE:443", "localhost:8000", "health.internal"):
            self.assertTrue(config.allows_host(value))
        for value in ("research.example.evil.test", "research.example:8000", "testserver", "localhost", "health.internal:123"):
            self.assertFalse(config.allows_host(value))
        for extra in ("*", "https://foo.example", "foo.example/path", "foo.example:0", "foo.example,,bar.example"):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                load_deployment_config({"AYURIP_DEPLOYMENT_MODE": "public_research", "AYURIP_PUBLIC_ORIGIN": "https://research.example", "AYURIP_ALLOWED_HOSTS": extra})

    def test_invalid_mode_and_local_cloud_override_fail_closed(self):
        for env in ({"AYURIP_DEPLOYMENT_MODE": "anything"},
                    {"AYURIP_ALLOWED_HOSTS": "evil.test"},
                    {"AYURIP_PUBLIC_ORIGIN": "https://evil.test"}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                load_deployment_config(env)

    def test_public_middleware_host_origin_headers_rate_and_media_type(self):
        env = {"AYURIP_DEPLOYMENT_MODE": "public_research", "AYURIP_PUBLIC_ORIGIN": "https://research.example",
               "AYURIP_RATE_LIMIT_ENABLED": "0"}
        with patch.dict(os.environ, env):
            app = FastAPI()
            @app.get("/search")
            def search():
                return {"search": "fixture"}
            @app.post("/answer")
            async def answer(request: Request):
                return await request.json()
            configure_security(app)
            with TestClient(app, base_url="https://research.example") as client:
                good = client.get("/search", headers={"Origin": "https://research.example"})
                self.assertEqual(good.status_code, 200)
                self.assertEqual(good.headers["strict-transport-security"], "max-age=31536000")
                self.assertNotIn("access-control-allow-origin", good.headers)
                self.assertEqual(client.get("/search", headers={"Host": "evil.test", "X-Forwarded-Host": "research.example"}).status_code, 400)
                self.assertEqual(client.get("/search", headers={"Origin": "https://evil.test"}).status_code, 403)
                self.assertEqual(client.get("/search", headers=[("Origin", "https://research.example"), ("Origin", "https://evil.test")]).status_code, 403)
                self.assertEqual(client.post("/answer", content="{}", headers={"Content-Type": "text/plain"}).status_code, 415)
                self.assertEqual(client.post("/answer", json={}).status_code, 200)
                statuses = [client.get("/search").status_code for _ in range(60)]
                self.assertIn(429, statuses)

    def test_public_api_has_no_docs_or_private_routes(self):
        env = dict(os.environ, AYURIP_DEPLOYMENT_MODE="public_research", AYURIP_PUBLIC_ORIGIN="https://research.example")
        command = [sys.executable, "-B", "-c", "import json; from ipindia_corpus.search_api import app; print(json.dumps([r.path for r in app.routes]))"]
        process = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        paths = json.loads(process.stdout)
        self.assertTrue({"/", "/search", "/sources", "/live", "/health", "/answer"}.issubset(paths))
        self.assertFalse({"/docs", "/redoc", "/openapi.json", "/api/passport", "/api/session"} & set(paths))

    def test_public_sources_use_validated_generation_and_omit_operational_details(self):
        from ipindia_corpus import search_api as api
        config = load_deployment_config({"AYURIP_DEPLOYMENT_MODE": "public_research", "AYURIP_PUBLIC_ORIGIN": "https://research.example"})
        snapshot = {"binding": ("frozen",), "documents": {"d": {"source": "ipindia", "retrieved_at": "2026-01-01T00:00:00Z", "raw_path": "E:/private/secret.pdf"}}}
        with patch.object(api, "DEPLOYMENT", config), patch.object(api, "_ensure_ready", return_value=snapshot):
            result = api.sources_endpoint()
        self.assertEqual(result["active_generation_id"], "frozen")
        self.assertEqual(result["sources"][0]["searchable_document_count"], 1)
        self.assertNotIn("private", json.dumps(result))
        self.assertNotIn("pipeline_failures", json.dumps(result))

    def test_public_mode_never_invokes_answer_provider_even_with_certified_fixture(self):
        from ipindia_corpus import search_api as api
        config = load_deployment_config({"AYURIP_DEPLOYMENT_MODE": "public_research", "AYURIP_PUBLIC_ORIGIN": "https://research.example"})
        with patch.object(api, "DEPLOYMENT", config), patch.object(api, "_answer_capabilities", return_value={"answer_generation_enabled": True, "active_generation_id": "frozen", "provider": {}}), patch.object(api, "generate_answer") as generate:
            response = api.answer_endpoint(api.AnswerRequest(question="Archived research"))
        self.assertEqual(response.status_code, 503)
        self.assertEqual(json.loads(response.body)["reason"], "three_source_not_certified")
        generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
