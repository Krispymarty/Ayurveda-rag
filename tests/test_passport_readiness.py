"""Recovery tests use controlled factories, never source capture or model calls."""
import threading
import unittest
from unittest.mock import patch

from ipindia_corpus.passport.readiness import EngineReadiness
from ipindia_corpus.passport.service import PassportEngine
from test_passport import Search, client, engine, intake
from ipindia_corpus.passport.api import create_app


class ReadinessTests(unittest.TestCase):
    def test_one_initialization_with_prompt_unavailable_responses(self):
        entered, release = threading.Event(), threading.Event()
        calls = []

        def factory():
            calls.append(True)
            entered.set()
            release.wait(3)
            return engine()

        state = EngineReadiness(factory=factory)
        try:
            threads = [threading.Thread(target=state.retry) for _ in range(12)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(1)
            self.assertTrue(entered.wait(1))
            self.assertEqual(len(calls), 1)
            self.assertEqual(state.snapshot()["reason"], "search_initializing")
            self.assertIsNone(state.available_engine())
        finally:
            release.set()
            state.worker.join(3)
        self.assertTrue(state.snapshot()["ready"])

    def test_explicit_retry_recovers_failed_initialization(self):
        clock = [0.0]
        calls = []

        def factory():
            calls.append(True)
            if len(calls) == 1:
                raise OSError("private-secret-marker")
            return engine()

        state = EngineReadiness(factory=factory, clock=lambda: clock[0])
        state.retry()
        state.worker.join(2)
        self.assertEqual(state.snapshot()["reason"], "search_dependency_unavailable")
        state.retry()
        self.assertEqual(len(calls), 1)
        clock[0] = 6
        state.retry()
        state.worker.join(2)
        self.assertTrue(state.snapshot()["ready"])
        self.assertNotIn("private-secret", str(state.snapshot()))

    def test_deadline_discards_late_result_and_prevents_overlapping_workers(self):
        clock = [0.0]
        entered, release = threading.Event(), threading.Event()
        calls = []

        def factory():
            calls.append(True)
            entered.set()
            release.wait(3)
            return engine()

        state = EngineReadiness(factory=factory, clock=lambda: clock[0])
        state.retry()
        self.assertTrue(entered.wait(1))
        clock[0] = 61
        self.assertEqual(state.snapshot()["reason"], "search_initialization_timeout")
        state.retry()
        self.assertEqual(len(calls), 1)
        release.set()
        state.worker.join(2)
        self.assertFalse(state.snapshot()["ready"])
        self.assertIsNone(state.available_engine())

    def test_outage_recovers_original_engine_but_generation_change_latches(self):
        e = engine()
        state = EngineReadiness(engine=e)
        with patch.object(e.search, "status", side_effect=OSError("offline")):
            self.assertEqual(state.snapshot()["reason"], "search_dependency_unavailable")
        self.assertTrue(state.snapshot()["ready"])
        e.search.gid = "other"
        self.assertEqual(state.snapshot()["reason"], "active_generation_changed")
        e.search.gid = "g"
        state.retry()
        self.assertEqual(state.snapshot()["reason"], "active_generation_changed")
        self.assertIsNone(state.available_engine())

    def test_constructor_rejects_generation_switch_during_archive_loading(self):
        class Verifier:
            gid = "earlier"
        with self.assertRaisesRegex(ValueError, "generation"):
            PassportEngine(Search(), Verifier())

    def test_api_session_retries_without_model_or_post_initialization(self):
        entered, release = threading.Event(), threading.Event()

        def factory():
            entered.set()
            release.wait(3)
            return engine()

        state = EngineReadiness(factory=factory)
        try:
            with client(create_app(enabled=True, readiness=state)) as c:
                self.assertTrue(entered.wait(1))
                session = c.get("/api/session").json()
                self.assertFalse(session["ready"])
                self.assertEqual(c.get("/live").status_code, 200)
                self.assertEqual(c.get("/health").status_code, 503)
                response = c.post("/api/passport", json=intake(), headers={
                    "Origin": "http://127.0.0.1:8003", "X-Passport-Token": session["csrf_token"]})
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()["reason"], "search_initializing")
        finally:
            release.set()
            state.worker.join(3)


if __name__ == "__main__":
    unittest.main()
