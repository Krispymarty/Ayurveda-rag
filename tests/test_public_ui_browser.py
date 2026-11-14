"""Opt-in browser checks using synthetic API responses, never cloud evidence.

Run with AYURIP_RUN_BROWSER_TESTS=1. Uses an already installed browser/Playwright;
does not install packages, contact official sources or invoke any answer model.
"""
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import unittest


@unittest.skipUnless(os.environ.get("AYURIP_RUN_BROWSER_TESTS") == "1", "Opt-in installed-browser test")
class PublicUIBrowserTests(unittest.TestCase):
    def test_research_only_rendering_recovery_and_mobile(self):
        from playwright.sync_api import sync_playwright

        root = Path(__file__).resolve().parents[1]
        output = root / "data/cloud_ui_20261114" / datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
        output.mkdir(parents=True, exist_ok=True)
        requests, errors, checks = [], [], []
        phase = {"ready": False, "search": "hits"}
        source = {"index_available": True, "active_generation_id": "synthetic-ui-fixture",
                  "sources": [{"source_id": "ipindia", "searchable_document_count": 2},
                              {"source_id": "indiacode", "searchable_document_count": 0},
                              {"source_id": "tkdl", "searchable_document_count": 0}]}
        hit = {"title": "Synthetic UI fixture — not legal evidence", "text": '<img src=x onerror="window.fixtureInjection=true"> Synthetic text only.',
               "document_id": "synthetic-document", "chunk_id": "synthetic-passage", "source": "ipindia",
               "legal_domain": "Patent", "evidence_role": "legal_source",
               "source_url": "https://ipindia.gov.in/synthetic.pdf", "page_start": 2,
               "page_end": 2, "section": "3", "rule": None, "status": "unknown",
               "raw_sha256": "0" * 64, "retrieved_at": "2026-11-14T00:00:00+00:00"}
        forged = {**hit, "chunk_id": "synthetic-forged-link", "source_url": "https://ipindia.gov.in.evil.test/forged.pdf"}

        def serve(route):
            path = route.request.url.split("ayurip-ui.test", 1)[1]
            requests.append(path.split("?", 1)[0])
            if path == "/health":
                body = {"status": "ok" if phase["ready"] else "unavailable",
                        "readiness": {"state": "ready" if phase["ready"] else "initializing"}}
                return route.fulfill(status=200 if phase["ready"] else 503, json=body)
            if path == "/sources":
                return route.fulfill(json=source)
            if path.startswith("/search?"):
                self.assertIn("top_k=5", path)
                if phase["search"] == "limited":
                    return route.fulfill(status=429, json={"reason": "rate_limit"})
                if phase["search"] == "unavailable":
                    phase["ready"] = False
                    return route.fulfill(status=503, json={"reason": "initializing"})
                return route.fulfill(json={"hits": [] if phase["search"] == "empty" else [hit, forged],
                                           "notice": "Synthetic browser fixture. Not deployed or legal evidence."})
            files = {"/": ("index.html", "text/html"), "/assets/app.js": ("app.js", "text/javascript"),
                     "/assets/styles.css": ("styles.css", "text/css")}
            if path in files:
                name, mime = files[path]
                return route.fulfill(content_type=mime + "; charset=utf-8", body=(root / "web" / name).read_bytes())
            return route.fulfill(status=404)

        with sync_playwright() as playwright, ExitStack() as browser_cleanup:
            browser_path = os.environ.get("AYURIP_BROWSER_EXECUTABLE")
            if not browser_path and sys.platform == "win32":
                browser_path = "C:/Program Files/Google/Chrome/Application/chrome.exe"
            browser = playwright.chromium.launch(headless=True, **({"executable_path": browser_path} if browser_path else {}))
            browser_cleanup.callback(browser.close)
            context = browser.new_context(viewport={"width": 1365, "height": 900}, reduced_motion="reduce")
            browser_cleanup.callback(context.close)
            context.route("**/*", serve)
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto("http://ayurip-ui.test/")
            page.get_by_text("Preparing evidence search…", exact=True).wait_for()
            self.assertTrue(page.locator("#submit-button").is_disabled())
            page.locator("#question").fill("SYNTHETIC research question")
            phase["ready"] = True
            page.locator("#retry-service").click()
            page.wait_for_function("!document.getElementById('submit-button').disabled")
            self.assertEqual(page.locator("#question").input_value(), "SYNTHETIC research question")
            checks.append("initialization blocks search; recovery preserves input")
            page.locator("#submit-button").click()
            page.locator(".evidence-hit").first.wait_for()
            self.assertEqual(page.locator(".evidence-hit").count(), 2)
            self.assertEqual(page.locator("#results img").count(), 0)
            self.assertFalse(page.evaluate("Boolean(window.fixtureInjection)"))
            self.assertIn('<img src=x onerror="window.fixtureInjection=true">', page.locator(".hit-passage").first.inner_text())
            self.assertEqual(page.locator(".evidence-hit").first.locator("h3 a").get_attribute("href"), "https://ipindia.gov.in/synthetic.pdf#page=2")
            self.assertEqual(page.locator(".evidence-hit").nth(1).locator("a").count(), 0)
            page.locator(".hit-toggle").first.click()
            self.assertEqual(page.locator(".hit-toggle").first.get_attribute("aria-expanded"), "true")
            checks.append("hostile text stays text; forged host has no link; PDF anchor and passage expansion work")
            page.locator(".provenance summary").first.click()
            self.assertIn("0" * 64, page.locator(".provenance").first.inner_text())
            page.locator("#filter-toggle").click()
            self.assertEqual(page.locator("#domain-filter option").all_text_contents(), ["All indexed areas", "Patent", "Trademark", "Design", "GI", "General IP"])
            self.assertEqual(page.locator("#mode-answer").count(), 0)
            page.locator("#source-detail-button").click()
            self.assertFalse(page.locator("#source-panel").is_hidden())
            page.locator("#source-panel-title").press("Escape")
            self.assertTrue(page.locator("#source-panel").is_hidden())
            checks.append("source panel keyboard recovery, archive details and supported filters work")
            page.screenshot(path=str(output / "synthetic-desktop.png"), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), 390)
            self.assertGreaterEqual(page.locator("#submit-button").bounding_box()["height"], 40)
            page.screenshot(path=str(output / "synthetic-mobile.png"), full_page=True)
            checks.append("390px viewport has no horizontal overflow")
            phase["search"] = "limited"
            page.locator("#submit-button").click()
            page.get_by_text("Please pause before trying again.", exact=True).wait_for()
            self.assertEqual(page.locator("#question").input_value(), "SYNTHETIC research question")
            phase["search"] = "empty"
            page.locator("#submit-button").click()
            page.get_by_text("Try a different research phrase.", exact=True).wait_for()
            phase["search"] = "unavailable"
            page.locator("#submit-button").click()
            page.get_by_text("Preparing evidence search…", exact=True).wait_for()
            self.assertTrue(page.locator("#submit-button").is_disabled())
            checks.append("429, empty results and 503 recovery retain input")
            self.assertNotIn("/answer", requests)
            self.assertNotIn("/api/passport", requests)
            self.assertEqual(errors, [])
            report = {"scope": "Synthetic intercepted browser test, not cloud verification or legal evidence",
                      "checks": checks, "page_errors": errors, "request_paths": requests,
                      "answer_model_invoked": False, "result": "passed"}
            (output / "browser_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
