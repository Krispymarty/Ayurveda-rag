import os
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from ipindia_corpus import search_api
from ipindia_corpus.http_security import ClientBuckets


class FrontendIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"AYURIP_RATE_LIMIT_ENABLED": "0"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.bucket = patch.object(ClientBuckets, "take", return_value=0)
        self.bucket.start()
        self.addCleanup(self.bucket.stop)
        self.client = TestClient(search_api.app)
        self.addCleanup(self.client.close)

    def test_landing_page_serves_ayuserach(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("AyuSerach", response.text)
        self.assertIn("data-page=\"home\"", response.text)

    def test_research_workspace_serves_ayuserach(self):
        for path in ("/research", "/research.html"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn("text/html", response.headers["content-type"])
                self.assertIn("Start a new research check", response.text)
                self.assertIn("id=\"research-form\"", response.text)
                self.assertIn("id=\"evidence-section\"", response.text)

    def test_static_assets_serve_correct_content_types(self):
        assets = [
            ("/styles.css", "text/css"),
            ("/app.js", "javascript"),
            ("/Images/ayuserach-logo.png", "image/png"),
            ("/Images/book-open-green.svg", "image/svg+xml"),
            ("/Images/book-open-gold.svg", "image/svg+xml"),
            ("/Images/square-3-stack-gold.svg", "image/svg+xml"),
            ("/assets/styles.css", "text/css"),
            ("/assets/app.js", "javascript"),
        ]
        for path, expected_type in assets:
            with self.subTest(path=path):
                res = self.client.get(path)
                self.assertEqual(res.status_code, 200, f"Failed for path {path}")
                self.assertIn(expected_type, res.headers["content-type"])

    def test_security_headers_on_all_pages(self):
        for path in ("/", "/research.html"):
            res = self.client.get(path)
            self.assertEqual(res.headers.get("x-content-type-options"), "nosniff")
            self.assertEqual(res.headers.get("x-frame-options"), "DENY")
            self.assertIn("content-security-policy", res.headers)
            self.assertNotIn("unsafe-inline", res.headers["content-security-policy"])


if __name__ == "__main__":
    unittest.main()
