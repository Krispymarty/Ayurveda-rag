import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from ipindia_corpus import search_api
from ipindia_corpus.http_security import ClientBuckets


class AnswerAPITests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"AYURIP_RATE_LIMIT_ENABLED": "0"})
        self.env.start()
        self.addCleanup(self.env.stop)
        # Keep public-contract tests independent of prior tests sharing app state;
        # dedicated security tests exercise the real limiter and its exhaustion.
        self.bucket = patch.object(ClientBuckets, "take", return_value=0)
        self.bucket.start()
        self.addCleanup(self.bucket.stop)
        self.client = TestClient(search_api.app)
        self.addCleanup(self.client.close)

    def certificate(self, generation="gen1"):
        return {source: {"status": "certified", "search_admission": "accepted", "certified_documents": 10,
                         "certified_generation_id": generation} for source in ("indiacode", "tkdl")}

    def test_home_assets_and_security_headers(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.headers["x-frame-options"], "DENY")
        self.assertNotIn("unsafe-inline", response.headers["content-security-policy"])
        asset = self.client.get("/assets/app.js")
        self.assertEqual(asset.status_code, 200)
        self.assertIn("textContent", asset.text)
        self.assertEqual(self.client.get("/assets/../.env").status_code, 404)
        self.assertEqual(self.client.get("/", headers={"Host": "evil.test"}).status_code, 400)

    def test_capabilities_show_real_active_counts_and_no_model_verification_claim(self):
        with patch.object(search_api, "_active_source_counts", return_value=("gen1", {"ipindia": 307, "indiacode": 0, "tkdl": 0})):
            response = self.client.get("/capabilities")
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["active_source_document_counts"]["ipindia"], 307)
        self.assertFalse(result["answer_generation_enabled"])
        self.assertFalse(result["provider"]["digest_verified"])
        self.assertFalse(result["public_release_ready"])

    def test_uncertified_three_sources_do_not_call_search_or_model(self):
        with patch.object(search_api, "_active_source_counts", return_value=("gen1", {"ipindia": 307, "indiacode": 8, "tkdl": 51})), patch.object(search_api, "generate_answer") as generate, patch.object(search_api, "search") as search:
            response = self.client.post("/answer", json={"question": "What does section 3 say?"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["reason"], "three_source_not_certified")
        self.assertEqual(response.json()["claims"], [])
        generate.assert_not_called()
        search.assert_not_called()

    def test_certification_must_bind_exact_active_generation_and_coverage(self):
        for counts, cert in (({"ipindia": 307, "indiacode": 0, "tkdl": 10}, self.certificate()),
                             ({"ipindia": 307, "indiacode": 10, "tkdl": 10}, self.certificate("old-generation")),
                             ({"ipindia": 307, "indiacode": 11, "tkdl": 10}, self.certificate())):
            with patch.object(search_api, "_active_source_counts", return_value=("gen1", counts)), patch.object(search_api, "_certification_summary", return_value=cert):
                self.assertFalse(self.client.get("/capabilities").json()["answer_generation_enabled"])

    def test_request_bounds_filter_values_and_extra_fields(self):
        invalid = [{"question": "x"}, {"question": " " * 10}, {"question": "a" * 501},
                   {"question": "Valid question", "top_k": 11}, {"question": "Valid question", "top_k": True},
                   {"question": "Valid question", "top_k": "5"}, {"question": "Valid question", "source": "evil"},
                   {"question": "Valid question", "domain": "x" * 81},
                   {"question": "Valid question", "provider": "http://evil.test"}]
        with patch.object(search_api, "generate_answer") as generate:
            for request in invalid:
                with self.subTest(request=request):
                    self.assertEqual(self.client.post("/answer", json=request).status_code, 422)
        generate.assert_not_called()

    def test_certified_route_passes_archive_verifier_and_generation_guard(self):
        answer = {"status": "abstained", "answer_text": "", "claims": [], "reason": "insufficient_relevant_evidence", "notice": "unknown", "generation_id": "gen1", "mode": "verified_extracts", "provider": {}}
        with patch.object(search_api, "_active_source_counts", return_value=("gen1", {"ipindia": 307, "indiacode": 10, "tkdl": 10})), patch.object(search_api, "_certification_summary", return_value=self.certificate()), patch.object(search_api, "generate_answer", return_value=answer) as generate:
            response = self.client.post("/answer", json={"question": "What does section 3 say?", "source": "ipindia", "top_k": 5})
            self.assertEqual(response.status_code, 200)
            self.assertIs(generate.call_args.kwargs["verify_hit_fn"], search_api._verify_answer_hit)
            guarded_search = generate.call_args.args[1]
            with patch.object(search_api, "search", return_value={"generation_id": "other", "hits": []}):
                with self.assertRaises(ValueError):
                    guarded_search("question")

    def test_database_error_capabilities_are_stable_and_private(self):
        with patch.object(search_api, "_active_source_counts", side_effect=RuntimeError("private password")):
            with self.assertLogs("ipindia_corpus.search_api", level="ERROR"):
                response = self.client.get("/capabilities")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["index_available"])
        self.assertNotIn("private password", response.text)

    def test_answer_citation_verifier_binds_original_metadata_and_bytes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            raw = root / "data/raw/ipindia/example.html"
            raw.parent.mkdir(parents=True)
            raw.write_bytes(b"Original archived content")
            text = "Extracted passage"
            expected = {"document_id": "doc", "source_url": "https://ipindia.gov.in/example", "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(), "content_hash": hashlib.sha256(text.encode()).hexdigest(), "document_title": "Title", "section": "3", "rule": None, "page_start": None, "page_end": None}
            document = {"doc": {"source_url": expected["source_url"], "sha256": expected["raw_sha256"], "raw_path": "data/raw/ipindia/example.html"}}
            hit = {**expected, "title": "Title", "chunk_id": "chunk", "text": text}
            with patch.object(search_api, "ROOT", root), patch.object(search_api, "_published_chunks", return_value={"chunk": expected}), patch.object(search_api, "_documents", return_value=document):
                search_api._verify_answer_hit(hit, "gen1")
                with self.assertRaises(ValueError):
                    search_api._verify_answer_hit({**hit, "section": "9"}, "gen1")
                raw.write_bytes(b"tampered")
                with self.assertRaises(ValueError):
                    search_api._verify_answer_hit(hit, "gen1")


if __name__ == "__main__":
    unittest.main()
