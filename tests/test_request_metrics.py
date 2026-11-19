from unittest import TestCase

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ipindia_corpus.request_metrics import RequestMetricsMiddleware, outcome, stage


class RequestMetricsTests(TestCase):
    def test_logs_stage_times_from_sync_handler_without_request_text(self):
        app = FastAPI()

        @app.get("/search")
        def search(q: str):
            with stage("db_rank"):
                outcome("search_fixture")
            return {"query": q}

        app.add_middleware(RequestMetricsMiddleware)
        client = TestClient(app)
        self.addCleanup(client.close)
        with self.assertLogs("uvicorn.error", level="INFO") as logs:
            result = client.get("/search", params={"q": "private-text-never-in-logs"},
                                headers={"X-Request-ID": "client-controlled-secret"})
        self.assertEqual(result.status_code, 200)
        self.assertIn(result.headers["x-request-id"], logs.output[0])
        self.assertIn("db_rank", logs.output[0])
        self.assertNotIn("private-text-never-in-logs", logs.output[0])
        self.assertNotIn("client-controlled-secret", logs.output[0])
