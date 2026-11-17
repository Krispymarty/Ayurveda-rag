import unittest
from ipindia_corpus.llm_experiment import ExperimentFailure
from ipindia_corpus.passport.service import PassportEngine, Intake
from test_gemini_safety import Remote
from test_passport import Search, intake
from test_passport_failover import PLAN, SELECT


class Primary(Remote):
    metadata = {
        "provider": "groq",
        "model": "qwen/qwen3.8-27b",
        "model_version": None,
        "model_digest": None,
        "external_processing": True,
    }


class GroqFailoverTests(unittest.TestCase):
    def test_success_and_transient_failure_route_to_actual_models(self):
        for fail in [None, "provider_unavailable", "timeout", "provider_rate_limited"]:
            p = Primary([PLAN, SELECT, SELECT])
            fallback = Remote([PLAN, SELECT, SELECT])
            if fail:

                def broken(fields):
                    raise ExperimentFailure(fail)

                p.chat = broken
            e = PassportEngine(
                Search(), lambda *args: None, provider=p, fallback_provider=fallback
            )
            r = e.build(Intake(**intake(acknowledged_external_processing=True)))
            self.assertEqual(r["status"], "ready")
            self.assertEqual(r["model"]["provider"], "gemini" if fail else "groq")
            self.assertEqual(r["routing"]["fallback_used"], bool(fail))
            self.assertTrue(r["routing"]["external_processing_attempted"])
            self.assertTrue(
                all(a["calls_attempted"] <= 7 for a in r["routing"]["attempts"])
            )

    def test_no_fallback_on_rejected_request_and_no_calls_without_consent(self):
        for reason in [
            "provider_request_rejected",
            "provider_invalid_selection",
            "provider_safety_blocked",
        ]:
            p = Primary([])
            fallback = Remote([PLAN, SELECT, SELECT])
            p.chat = lambda fields: (_ for _ in ()).throw(ExperimentFailure(reason))
            e = PassportEngine(
                Search(), lambda *args: None, provider=p, fallback_provider=fallback
            )
            self.assertEqual(e.build(Intake(**intake()))["status"], "blocked")
            e.build(Intake(**intake(acknowledged_external_processing=True)))
            self.assertFalse(fallback.calls)


if __name__ == "__main__":
    unittest.main()
