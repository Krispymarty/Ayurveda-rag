"""Offline harness tests. Fixtures are synthetic, never legal quality labels."""
import tempfile
import unittest
from pathlib import Path

from tools.benchmark_gemini import (
    aggregate, model_observation, nearest_rank, output_directory, selected_scenarios,
    stage_totals,
)


class GeminiBenchmarkTests(unittest.TestCase):
    def test_nearest_rank_keeps_failure_latency_and_small_sample(self):
        self.assertEqual(nearest_rank([10, 30, 20], 95), 30)
        self.assertIsNone(nearest_rank([], 95))
        rows = [
            {"case_id": "a", "warmup": False, "http_ms": 10, "http_status": 200,
             "structural_pass": True, "calls_attempted": 2, "stages_ms": {"selection": 4}},
            {"case_id": "a", "warmup": False, "http_ms": 90, "http_status": 503,
             "structural_pass": False, "calls_attempted": 1, "stages_ms": {}},
            {"case_id": "b", "warmup": True, "http_ms": 900, "http_status": 200,
             "structural_pass": True, "calls_attempted": 2, "stages_ms": {}},
        ]
        result = aggregate(rows)
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["unique_questions"], 1)
        self.assertEqual(result["unique_questions_all_attempts_passed"], 0)
        self.assertEqual(result["http_ms"]["p95"], 90)
        self.assertEqual(result["successful_http_ms"]["p95"], 10)
        self.assertEqual(result["warmup_attempts"], 1)
        self.assertFalse(result["five_second_empirical_gate"])

    def test_gemini_observation_uses_version_and_never_fakes_weights_digest(self):
        expected = {"provider": "gemini", "model": "gemini-test"}
        body = {"model": {"provider": "gemini", "name": "gemini-test", "model_version": "gemini-test-001",
                          "digest": None, "calls_completed": 3}}
        self.assertTrue(model_observation(body, expected)["actual_model_run"])
        body["model"]["model_version"] = None
        self.assertFalse(model_observation(body, expected)["actual_model_run"])
        body["model"]["model_version"] = "another-model"
        self.assertFalse(model_observation(body, expected)["actual_model_run"])

    def test_local_identity_requires_original_digest(self):
        expected = {"provider": "ollama", "model": "qwen-fixture", "model_digest": "a" * 64}
        body = {"model": {"name": "qwen-fixture", "digest": "a" * 64, "calls_completed": 1}}
        self.assertTrue(model_observation(body, expected)["actual_model_run"])
        body["model"]["digest"] = "b" * 64
        self.assertFalse(model_observation(body, expected)["actual_model_run"])

    def test_scenarios_are_frozen_unique_and_output_is_confined_fresh(self):
        cases, digest = selected_scenarios(None)
        self.assertEqual(len(cases), 6)
        self.assertEqual(len(digest), 64)
        with self.assertRaises(ValueError):
            selected_scenarios(["novel_process", "novel_process"])
        with self.assertRaises(ValueError):
            selected_scenarios(["invented"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            allowed = root / "data/gemini_20261116/fresh"
            self.assertEqual(output_directory(allowed, root=root), allowed.resolve())
            allowed.mkdir(parents=True)
            with self.assertRaises(ValueError):
                output_directory(allowed, root=root)
            with self.assertRaises(ValueError):
                output_directory(root / "reports/overwrite", root=root)

    def test_stage_sums_keep_planning_retrieval_and_verification_distinct(self):
        body = {"timings_ms": {"issue_identification": 9, "final_passport_archive": 2},
                "performance": {"issues": [
                    {"timings_ms": {"retrieval": 3, "selection": 5, "relevance_verification": 7}},
                    {"timings_ms": {"retrieval": 4, "selection": 6, "relevance_verification": 8}}],
                    "model_requests": [{"provider_timing": {"generation_http_ms": 4, "prompt_tokens": 20}}]}}
        stages, provider = stage_totals(body, {"serialization": 1})
        self.assertEqual(stages["planning"], 9)
        self.assertEqual(stages["retrieval"], 7)
        self.assertEqual(stages["selection"], 11)
        self.assertEqual(stages["verification"], 15)
        self.assertEqual(stages["serialization"], 1)
        self.assertEqual(provider["generation_http_ms"], 4)


if __name__ == "__main__":
    unittest.main()
