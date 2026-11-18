"""Regression tests for the hosted access boundary and evidence handoff."""
from contextlib import nullcontext
from unittest import TestCase
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from ipindia_corpus import hosted_answers, search_api
from ipindia_corpus.cloud_config import load_deployment_config


ORIGIN = "https://research.example"
KEY = bytes(range(32))
CONFIG = load_deployment_config({"AYURIP_DEPLOYMENT_MODE": "public_research",
                                 "AYURIP_PUBLIC_ORIGIN": ORIGIN, "AYURIP_HOSTED_ANSWERS": "1"})
BINDING = ("20261014T204940358650Z", "hash", "revision", "v2")
BODY = {"question": "Quote an archived Patents Act provision", "acknowledged_nonconfidential": True,
        "acknowledged_external_processing": True}


class HostedAnswersTests(TestCase):
    def setUp(self):
        for subject, field, value in (
            (search_api, "DEPLOYMENT", CONFIG),
            (search_api, "_ensure_ready", Mock(return_value={"binding": BINDING})),
            (search_api, "_active_source_counts", Mock(return_value=(BINDING[0], {"ipindia": 307}))),
            (hosted_answers, "settings", Mock(return_value=(KEY, "unused"))),
        ):
            self.enterContext(patch.object(subject, field, value))
        self.providers = self.enterContext(patch.object(hosted_answers, "providers", return_value=(Mock(), Mock())))
        self.reserve = self.enterContext(patch.object(hosted_answers, "reserve", return_value=nullcontext()))
        self.model = self.enterContext(patch.object(search_api.demo_answers, "answer",
            return_value={"status": "answered", "claims": [], "scope": "ipindia_only"}))
        self.client = TestClient(search_api.app, base_url=ORIGIN)
        self.addCleanup(self.client.close)

    def test_cookie_tampering_expiry_and_key_rotation(self):
        token = hosted_answers.issue_cookie(KEY, now=100)
        self.assertIsNotNone(hosted_answers.browser_id(token, KEY, now=101))
        self.assertIsNone(hosted_answers.browser_id(token.replace(".100.", ".101."), KEY, now=102))
        self.assertIsNone(hosted_answers.browser_id(token, KEY, now=100 + hosted_answers.COOKIE_AGE))
        self.assertIsNone(hosted_answers.browser_id(token, b"other key", now=101))

    def test_origin_consent_session_and_policy_precede_quota_and_model(self):
        for headers, data, status in [
            ({}, BODY, 403), ({"Origin": "https://evil.test"}, BODY, 403),
            ({"Origin": ORIGIN}, {**BODY, "acknowledged_nonconfidential": False}, 403),
            ({"Origin": ORIGIN}, BODY, 403),
            ({"Origin": ORIGIN}, {**BODY, "question": "Can I patent my invention?"}, 200),
        ]:
            with self.subTest(headers=headers, data=data):
                response = self.client.post("/answer", json=data, headers={"Host": "testserver", **headers})
                self.assertEqual(response.status_code, status)
        self.reserve.assert_not_called()
        self.providers.assert_not_called()
        self.model.assert_not_called()

    def test_capability_cookie_quota_fail_closed_and_success(self):
        from ipindia_corpus.inference import configured_identity
        with patch("ipindia_corpus.inference.configured_identity", wraps=configured_identity):
            response = self.client.get("/capabilities", headers={"Host": "testserver"})
        self.assertTrue(response.json()["hosted_answers_available"])
        cookie = response.headers["set-cookie"]
        self.assertIn(hosted_answers.COOKIE, cookie)
        self.assertIn("HttpOnly", cookie)
        self.assertIn("Secure", cookie)
        self.assertIn("SameSite=strict", cookie)
        self.assertEqual(response.headers["cache-control"], "no-store")
        for status, reason, retry in [(429, "answer_daily_limit", 24), (503, "answer_quota_unavailable", None)]:
            self.reserve.side_effect = hosted_answers.AdmissionError(reason, status, retry)
            result = self.client.post("/answer", json=BODY, headers={"Host": "testserver", "Origin": ORIGIN})
            self.assertEqual(result.status_code, status)
            self.assertEqual(result.json()["reason"], reason)
            if retry:
                self.assertEqual(result.headers["retry-after"], str(retry))
            self.model.assert_not_called()
        self.reserve.side_effect = None
        result = self.client.post("/answer", json=BODY, headers={"Host": "testserver", "Origin": ORIGIN})
        self.assertEqual(result.status_code, 200)
        self.model.assert_called_once()

    def test_adapted_search_is_scoped_and_echoes_original_question(self):
        original = "पेटेंट अधिनियम की धारा ३(p) में पारंपरिक ज्ञान"
        with patch.object(search_api, "search", return_value={"query": "mapped", "hits": []}) as search:
            result = search_api.search_endpoint(original, 5, "ipindia", "Patent", adapt_hindi=True)
            self.assertEqual(result["query"], original)
            self.assertIn("section 3(p)", search.call_args.args[0])
            self.assertEqual(search.call_args.args[2:], ("ipindia", "Patent"))
            search_api.search_endpoint(original, 5, None, None)
            self.assertEqual(search.call_args.args[0], original)

    def test_rollback_binding_excludes_hosted_answers(self):
        search_api._ensure_ready.return_value = {"binding": ("older", "hash", "revision", "v1")}
        result = self.client.post("/answer", json=BODY, headers={"Host": "testserver", "Origin": ORIGIN})
        self.assertEqual(result.status_code, 503)
        self.reserve.assert_not_called()
        self.model.assert_not_called()
