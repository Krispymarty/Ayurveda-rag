"""Provider-aware readiness and consent checks; no external inference."""
from contextlib import ExitStack
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from ipindia_corpus.llm_experiment import ExperimentFailure
from ipindia_corpus.passport.readiness import EngineReadiness


REMOTE = {"provider": "gemini", "model": "gemini-3.5-flash-lite",
          "model_version": "gemini-3.5-flash-lite-fixture", "model_digest": None,
          "external_processing": True}
LOCAL = {"provider": "ollama", "model": "qwen3.5:4b", "model_version": None,
         "model_digest": "0" * 64, "external_processing": False}


class ProviderReadinessTests(unittest.TestCase):
    def test_active_engine_identity_replaces_configured_model(self):
        class Engine:
            gid = "synthetic-generation"
            provider = object()

            def ready(self):
                return True

        engine = Engine()
        with patch("ipindia_corpus.passport.readiness.provider_identity", return_value=REMOTE) as identity:
            state = EngineReadiness(engine=engine)
            result = state.snapshot()
        identity.assert_called_once_with(engine.provider)
        self.assertTrue(result["ready"])
        self.assertEqual(result["generation_id"], engine.gid)
        self.assertEqual(result["model"], REMOTE["model"])
        self.assertEqual(result["provider"], REMOTE)
        self.assertTrue(result["external_processing"])
        self.assertIsNone(result["provider"]["model_digest"])

    def test_uninitialized_engine_exposes_only_safe_configuration(self):
        with patch("ipindia_corpus.passport.readiness.configured_identity", return_value=REMOTE):
            result = EngineReadiness(factory=lambda: None).snapshot()
        self.assertFalse(result["ready"])
        self.assertEqual(result["model"], REMOTE["model"])
        self.assertTrue(result["external_processing"])

    def test_key_and_configuration_failures_have_stable_reasons(self):
        for reason in ("gemini_key_missing", "gemini_configuration_invalid"):
            with self.subTest(reason=reason), patch(
                "ipindia_corpus.passport.readiness.configured_identity", return_value=REMOTE
            ):
                def fail():
                    raise ExperimentFailure(reason)

                state = EngineReadiness(factory=fail)
                state.retry()
                state.worker.join(2)
                result = state.snapshot()
                self.assertFalse(result["ready"])
                self.assertEqual(result["reason"], reason)
                self.assertIsNone(state.available_engine())

    def test_arbitrary_provider_failures_do_not_leak_details(self):
        def fail():
            raise ExperimentFailure("private-api-key-marker")

        with patch("ipindia_corpus.passport.readiness.configured_identity", return_value=REMOTE):
            state = EngineReadiness(factory=fail)
            state.retry()
            state.worker.join(2)
            result = state.snapshot()
        self.assertEqual(result["reason"], "search_dependency_unavailable")
        self.assertNotIn("private-api-key-marker", str(result))

    def test_invalid_identity_configuration_fails_closed(self):
        with patch("ipindia_corpus.passport.readiness.configured_identity", side_effect=ValueError("private-setting")):
            result = EngineReadiness().snapshot()
        self.assertFalse(result["ready"])
        self.assertEqual(result["reason"], "gemini_configuration_invalid")
        self.assertNotIn("private-setting", str(result))

    def test_provider_drift_reason_latches_without_becoming_generation_failure(self):
        class Engine:
            gid = "synthetic-generation"
            provider = object()
            binding_reason = "provider_model_changed"
            available = False
            ready_calls = 0

            def ready(self):
                self.ready_calls += 1
                return self.available

        engine = Engine()
        with patch("ipindia_corpus.passport.readiness.provider_identity", return_value=REMOTE):
            state = EngineReadiness(engine=engine)
            result = state.snapshot()
            self.assertFalse(result["ready"])
            self.assertEqual(result["reason"], "provider_model_changed")
            engine.available = True
            engine.binding_reason = "active_generation_changed"
            state.retry()
            self.assertEqual(state.snapshot()["reason"], "provider_model_changed")
            self.assertIsNone(state.available_engine())
            self.assertEqual(engine.ready_calls, 1)
        with patch("ipindia_corpus.passport.readiness.provider_identity", side_effect=ValueError("private-setting")):
            result = state.snapshot()
        self.assertEqual(result["reason"], "provider_model_changed")
        self.assertNotIn("private-setting", str(result))

    def test_unknown_binding_reason_is_not_exposed(self):
        class Engine:
            gid = "synthetic-generation"
            binding_reason = "private-provider-marker"

            def ready(self):
                return False

        with patch("ipindia_corpus.passport.readiness.provider_identity", return_value=LOCAL):
            state = EngineReadiness(engine=Engine())
            result = state.snapshot()
        self.assertEqual(result["reason"], "active_generation_changed")
        self.assertNotIn("private-provider-marker", str(result))


@unittest.skipUnless(os.environ.get("AYURIP_RUN_BROWSER_TESTS") == "1", "Opt-in installed-browser test")
class ProviderConsentBrowserTests(unittest.TestCase):
    def test_external_notice_consent_payload_local_compatibility_and_missing_key(self):
        from playwright.sync_api import sync_playwright

        root = Path(__file__).resolve().parents[1]
        ui = root / "ipindia_corpus/passport/ui"
        phase = {"remote": True, "ready": True}
        acknowledgments, errors = [], []

        def serve(route):
            path = route.request.url.split("127.0.0.1:8003", 1)[1]
            if path == "/api/session":
                identity = REMOTE if phase["remote"] else LOCAL
                return route.fulfill(json={"csrf_token": "synthetic-ui-token", "ready": phase["ready"],
                    "reason": None if phase["ready"] else "gemini_key_missing",
                    "model": identity["model"], "provider": identity,
                    "external_processing": identity["external_processing"]})
            if path == "/api/passport":
                intake = route.request.post_data_json
                acknowledgments.append(intake["acknowledged_external_processing"])
                return route.fulfill(json={"status": "needs_clarification", "generation_id": "synthetic-ui-generation",
                    "model": {"name": REMOTE["model"], "provider": "gemini", "model_version": REMOTE["model_version"]}})
            files = {"/": (ui / "index.html", "text/html"),
                     "/assets/passport.js": (ui / "passport.js", "text/javascript"),
                     "/assets/passport.css": (ui / "passport.css", "text/css"),
                     "/assets/base.css": (root / "web/styles.css", "text/css")}
            if path in files:
                file, mime = files[path]
                return route.fulfill(body=file.read_bytes(), content_type=mime + "; charset=utf-8")
            return route.fulfill(status=404)

        with sync_playwright() as playwright, ExitStack() as cleanup:
            executable = os.environ.get("AYURIP_BROWSER_EXECUTABLE")
            if not executable and os.name == "nt":
                executable = "C:/Program Files/Google/Chrome/Application/chrome.exe"
            browser = playwright.chromium.launch(headless=True, **({"executable_path": executable} if executable else {}))
            cleanup.callback(browser.close)
            context = browser.new_context(viewport={"width": 390, "height": 844}, reduced_motion="reduce")
            cleanup.callback(context.close)
            context.route("**/*", serve)
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto("http://127.0.0.1:8003/")
            page.wait_for_function("!document.getElementById('submit-passport').disabled")
            self.assertTrue(page.locator("#external-processing-notice").is_visible())
            self.assertTrue(page.locator("#external-acknowledgment").evaluate("e => e.required"))
            page.locator("#example-fill").click()
            page.locator("#acknowledgment").check()
            page.locator("#submit-passport").click()
            self.assertEqual(acknowledgments, [])
            page.locator("#external-acknowledgment").check()
            page.locator("#submit-passport").click()
            page.locator("#passport-results").wait_for(state="visible")
            self.assertEqual(acknowledgments, [True])
            self.assertIn("Inference provider: gemini", page.locator(".result-meta").inner_text())
            self.assertIn(REMOTE["model_version"], page.locator(".result-meta").inner_text())
            self.assertNotIn("Model digest:", page.locator(".result-meta").inner_text())
            page.locator("#clear-form").click()
            self.assertFalse(page.locator("#external-acknowledgment").is_checked())
            self.assertTrue(page.locator("#external-acknowledgment").evaluate("e => e.required"))

            phase["ready"] = False
            page.reload()
            page.wait_for_function("document.getElementById('session-status').textContent.includes('GEMINI_API_KEY')")
            self.assertTrue(page.locator("#submit-passport").is_disabled())
            self.assertTrue(page.locator("#external-processing-notice").is_visible())

            phase.update(remote=False, ready=True)
            page.reload()
            page.wait_for_function("!document.getElementById('submit-passport').disabled")
            self.assertTrue(page.locator("#external-processing-notice").is_hidden())
            self.assertFalse(page.locator("#external-acknowledgment").evaluate("e => e.required"))
            page.locator("#example-fill").click()
            page.locator("#acknowledgment").check()
            page.locator("#submit-passport").click()
            page.locator("#passport-results").wait_for(state="visible")
            self.assertEqual(acknowledgments, [True, False])
            self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
