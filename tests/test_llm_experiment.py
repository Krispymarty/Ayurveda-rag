import json
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import requests

from ipindia_corpus.llm_experiment import (
    ExperimentFailure, OllamaExperimentClient, canonical, observe, protected_state,
    resource_summary, sha256, summarize, timing_metadata, validate_model, compare, profile,
)


class FakeResponse:
    def __init__(self, body=None, *, status=200, lines=None):
        self.body, self.status_code, self.lines, self.closed = body, status, lines, False

    def iter_content(self, chunk_size):
        yield json.dumps(self.body).encode()

    def iter_lines(self):
        for row in self.lines or []:
            yield json.dumps(row).encode()

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls, self.trust_env = list(responses), [], True

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class NoLiveSampler:
    samples = [{"cpu_percent": 20, "ram_used_bytes": 100, "gpu": {"devices": [{"memory_used_mb": 1000, "gpu_utilization_percent": 90}]}}]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class ManagedClient:
    timeout = 120

    def __init__(self, *, external=False):
        self.models = [{"name": "external:tag", "digest": "e" * 64}] if external else [{"name": "baseline:tag", "digest": "a" * 64}]
        self.unloaded = []

    def receipt(self, model):
        return {"model": model, "model_digest": "a" * 64 if model == "baseline:tag" else "b" * 64,
                "thinking_capable": False}

    def placement(self):
        return {"models": copy.deepcopy(self.models)}

    def request(self, method, path, payload):
        if method != "POST" or path != "/api/generate" or payload["keep_alive"] != 0:
            raise AssertionError("Only owned-model unloads are allowed by this fixture")
        self.unloaded.append(payload["model"])
        self.models = [model for model in self.models if model["name"] != payload["model"]]
        return {"response": {"done": True}}


def benchmark_fixture(root):
    (root / "reports/llm").mkdir(parents=True)
    baseline = {"profiling_completed": True, "protected_state_unchanged": True, "model": "baseline:tag", "model_digest": "a" * 64}
    (root / "reports/llm/baseline_performance.json").write_text(json.dumps(baseline), encoding="utf8")
    (root / "config").mkdir()
    (root / "config/answer_generation.json").write_text(json.dumps({"model": "baseline:tag", "model_digest": "a" * 64, "max_output_tokens": 600}), encoding="utf8")
    manifest = {"id": "fixture", "generation_id": "generation", "files": {}}
    cases = [{"id": "case-one", "category": "exact_extraction", "expected": {"policy": "quote_targets"}}]
    return manifest, cases, "SYSTEM", {}


def completed_fixture_observation(client, receipt, case, system, schema, parameters, **kwargs):
    client.models = [{"name": receipt["model"], "digest": receipt["model_digest"], "size": 1000, "size_vram": 1000}]
    return {"status": "completed", "phase": kwargs["phase"], "category": case["category"], "chat_http_wall_ms": 100,
            "chat_attempt_wall_ms": 101, "timings": {"tokens_per_second": 20},
            "score": {"passed": True, "failures": [], "schema_valid": True, "source_valid": True}}


class LLMExperimentTests(unittest.TestCase):
    def test_native_endpoint_redirect_and_request_bounds(self):
        for endpoint in ("http://evil.test:11434", "http://host.docker.internal:11434", "http://user@localhost:11434", "http://localhost:11434/path"):
            with self.assertRaises(ValueError):
                OllamaExperimentClient(endpoint)
        with self.assertRaises(ValueError):
            OllamaExperimentClient(timeout=121)
        response = FakeResponse({}, status=302)
        session = FakeSession([response])
        client = OllamaExperimentClient(session=session)
        with self.assertRaises(ExperimentFailure):
            client.request("GET", "/api/tags")
        self.assertFalse(session.trust_env)
        self.assertFalse(session.calls[0][2]["allow_redirects"])
        self.assertTrue(response.closed)
        with self.assertRaises(ValueError):
            client.request("POST", "/api/pull", {"model": "anything"})

    def test_timing_nanoseconds_are_converted_and_missing_metrics_are_not_invented(self):
        values = timing_metadata({"load_duration": 2000000000, "eval_count": 30, "eval_duration": 3000000000, "prompt_eval_count": 100, "done": True, "done_reason": "stop"})
        self.assertEqual(values["load_duration_ms"], 2000)
        self.assertEqual(values["tokens_per_second"], 10)
        self.assertNotIn("prompt_eval_duration", values)
        self.assertNotIn("tokens_per_second", timing_metadata({"eval_count": 10, "eval_duration": 0}))

    def test_streaming_buffers_content_ttft_and_drops_raw_thinking(self):
        response = FakeResponse(lines=[{"model": "model:tag", "message": {"thinking": "private reasoning"}, "done": False},
            {"model": "model:tag", "message": {"content": '{"claims":'}, "done": False},
            {"model": "model:tag", "message": {"content": "[]}"}, "done": True, "done_reason": "stop", "eval_count": 3}])
        client = OllamaExperimentClient(session=FakeSession([response]))
        result = client.request("POST", "/api/chat", {}, streaming=True)
        self.assertEqual(result["response"]["message"]["content"], '{"claims":[]}')
        self.assertIsNotNone(result["ttft_content_ms"])
        self.assertLessEqual(result["ttft_any_generated_ms"], result["ttft_content_ms"])
        self.assertNotIn("private reasoning", json.dumps(result))
        self.assertEqual(result["response"]["message"]["thinking_characters"], 17)

    def test_nonstreaming_ttft_is_null_and_timeouts_remain_failure(self):
        client = OllamaExperimentClient(session=FakeSession([FakeResponse({"message": {"content": "{}", "thinking": "secret"}})]))
        response = client.request("POST", "/api/chat", {})
        self.assertIsNone(response["ttft_content_ms"])
        self.assertNotIn("secret", json.dumps(response))
        client = OllamaExperimentClient(session=FakeSession([requests.Timeout("private endpoint")]))
        with self.assertRaisesRegex(ExperimentFailure, "^timeout$"):
            client.request("GET", "/api/tags")

    def test_exact_tag_receipt_no_substitution_and_completion_capability(self):
        session = FakeSession([FakeResponse({"models": [{"name": "other:tag", "digest": "a" * 64}]})])
        with self.assertRaisesRegex(ExperimentFailure, "exact_model_not_installed"):
            OllamaExperimentClient(session=session).receipt("requested:tag")
        session = FakeSession([FakeResponse({"models": [{"name": "model:tag", "digest": "a" * 64}]}), FakeResponse({"capabilities": ["embedding"]})])
        with self.assertRaisesRegex(ExperimentFailure, "chat_capability_unverified"):
            OllamaExperimentClient(session=session).receipt("model:tag")
        for model in ("../../bad", "https://evil", "model;remove", "model with space"):
            with self.assertRaises(ValueError):
                validate_model(model)

    def test_cold_unload_never_evicts_other_models(self):
        client = OllamaExperimentClient(session=FakeSession([]))
        with patch.object(client, "placement", return_value={"models": [{"name": "other:tag"}]}), patch.object(client, "request") as request:
            result = client.unload_for_cold_measurement("model:tag")
        self.assertFalse(result["cold_measurement_available"])
        request.assert_not_called()

    def test_summary_does_not_count_timeout_or_truncation_as_numerical_success(self):
        valid = {"passed": True, "failures": [], "schema_valid": True, "source_valid": True}
        rows = [{"status": "completed", "chat_http_wall_ms": 100, "timings": {"tokens_per_second": 20}, "score": valid},
                {"status": "timeout", "observation_wall_ms": 120000, "score": {"passed": False, "failures": ["TIMEOUT"], "schema_valid": False, "source_valid": False}},
                {"status": "incomplete", "chat_attempt_wall_ms": 3000, "score": {"passed": False, "failures": ["TRUNCATED_OUTPUT"], "schema_valid": True, "source_valid": True}}]
        summary = summarize(rows)
        self.assertEqual(summary["warm_latency_ms"]["p95"], 100)
        self.assertEqual(summary["timeout_requests"], 1)
        self.assertEqual(summary["raw_model_case_pass_rate"], 1 / 3)
        self.assertFalse(summary["zero_fabricated_citations_gate_passed"])
        self.assertEqual(summary["all_attempt_latency_ms"]["p95"], 120000)
        self.assertEqual(summary["attempts_with_latency_measurement"], 3)

    def test_protected_state_detects_model_ranker_or_source_changes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            (root / "config").mkdir()
            path = root / "config/answer_generation.json"
            path.write_text("{}", encoding="utf8")
            before = protected_state(root, "generation")
            path.write_text('{"changed":true}', encoding="utf8")
            self.assertNotEqual(before, protected_state(root, "generation"))

    def test_observation_is_immutable_scored_once_and_does_not_mutate_evidence(self):
        class Client:
            def placement(self):
                return {"models": []}
            def verify_identity(self, receipt):
                return None
            def request(self, *args, **kwargs):
                raw = '{"claims":[{"chunk_id":"c1","quote":"A valid legal provision."}]}'
                return {"response": {"model": "model:tag", "done": True, "done_reason": "stop", "message": {"content": raw}}, "response_bytes_sha256": "a" * 64,
                        "http_wall_ms": 100, "ttft_content_ms": None, "ttft_any_generated_ms": None}
        case = {"id": "fixture", "category": "exact_extraction", "question": "Provision?", "model_input_sha256": "b" * 64, "evidence_sha256": "c" * 64,
                "evidence": [{"chunk_id": "c1", "document_id": "d1", "raw_sha256": "f" * 64, "source": "ipindia", "source_url": "https://ipindia.gov.in/act", "text": "A valid legal provision."}],
                "expected": {"policy": "quote_targets", "target_groups": [{"required_chunk_ids": ["c1"], "required_contains": "valid legal"}]}}
        before = canonical(case)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory, patch("ipindia_corpus.llm_experiment.ResourceSampler", NoLiveSampler):
            root = Path(directory)
            result = observe(Client(), {"model": "model:tag", "model_digest": "d" * 64}, case, "System", {}, {"num_predict": 200, "num_ctx": 4096, "keep_alive": "30m"}, root=root, run_id="fixture", phase="warm")
            self.assertTrue(result["score"]["passed"])
            body = (root / result["immutable_observation_path"]).read_bytes()
            self.assertEqual(sha256(body), result["immutable_observation_sha256"])
            self.assertEqual(before, canonical(case))
        self.assertEqual(resource_summary(NoLiveSampler.samples)["peak_gpu_memory_used_mb"], 1000)

    def test_compare_explicitly_unloads_owned_previous_model_before_next(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            bundle = benchmark_fixture(root)
            client = ManagedClient()
            with patch("tools.freeze_llm_benchmark.load_frozen", return_value=bundle), patch("ipindia_corpus.llm_experiment.observe", side_effect=completed_fixture_observation):
                report = compare(root, models=["baseline:tag", "next:tag"], passes=1, client=client)
            self.assertEqual(client.unloaded, ["baseline:tag", "baseline:tag"])
            self.assertEqual([model["status"] for model in report["models"]], ["measured", "measured"])
            self.assertTrue(all(model["warmup_placement"]["single_model_verified"] for model in report["models"]))
            self.assertEqual(report["models"][1]["warmup_placement"]["reported_vram_ratio"], 1)
            self.assertTrue(report["protected_state_unchanged"])
            self.assertIn("ipindia_corpus/llm_scoring.py", report["code_sha256_before"])

    def test_compare_refuses_unrelated_resident_without_evicting_or_inference(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            bundle = benchmark_fixture(root)
            client = ManagedClient(external=True)
            with patch("tools.freeze_llm_benchmark.load_frozen", return_value=bundle), patch("ipindia_corpus.llm_experiment.observe") as inference:
                report = compare(root, models=["next:tag"], passes=1, client=client)
            self.assertEqual(report["models"][0]["reason"], "external_model_resident")
            self.assertEqual(client.unloaded, [])
            inference.assert_not_called()

    def test_compare_aborts_cohort_when_post_warmup_has_another_resident(self):
        def unexpected(*args, **kwargs):
            row = completed_fixture_observation(*args, **kwargs)
            args[0].models.append({"name": "external:tag", "digest": "e" * 64})
            return row
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            bundle = benchmark_fixture(root)
            client = ManagedClient()
            with patch("tools.freeze_llm_benchmark.load_frozen", return_value=bundle), patch("ipindia_corpus.llm_experiment.observe", side_effect=unexpected) as inference:
                report = compare(root, models=["baseline:tag"], passes=3, client=client)
            self.assertEqual(report["models"][0]["status"], "aborted")
            self.assertEqual(report["models"][0]["reason"], "unexpected_model_concurrency")
            self.assertEqual(inference.call_count, 1)

    def test_profile_without_cold_probe_runs_excluded_warmup_before_timed_warm(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            bundle = benchmark_fixture(root)
            client = ManagedClient()
            with patch("tools.freeze_llm_benchmark.load_frozen", return_value=bundle), patch("ipindia_corpus.llm_experiment.observe", side_effect=completed_fixture_observation) as inference:
                report = profile(root, model="baseline:tag", warm_requests=1, include_cold=False, include_stream=False, client=client)
            phases = [call.kwargs["phase"] for call in inference.call_args_list]
            self.assertEqual(phases, ["baseline_untimed_warmup", "baseline_warm", "candidate_untimed_warmup", "candidate_warm"])
            self.assertEqual(report["summaries"]["baseline"]["observations"], 1)
            self.assertEqual(report["summaries"]["candidate"]["observations"], 1)


if __name__ == "__main__":
    unittest.main()
