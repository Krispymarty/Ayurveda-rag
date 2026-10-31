"""Experimental metadata reuse must never reuse model outputs or weaken checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ipindia_corpus import reliable_answers as extracts
from ipindia_corpus.llm_experiment import ExperimentFailure
from ipindia_corpus.passport.service import Intake, PassportEngine, planning_fields
from tools.passport_call_experiment import ScopedMetadataClient, BoundProvider, http_attempt, run
from test_passport import Search, intake


class Client:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.receipts = self.identities = self.chats = 0
        self.digest = extracts.DIGEST
        self.payloads = []

    def receipt(self, model):
        self.receipts += 1
        return {"model": model, "model_digest": self.digest,
                "thinking_capable": True, "show_response": {"capabilities": ["completion", "thinking"]}}

    def verify_identity(self, receipt):
        self.identities += 1
        if self.digest != receipt["model_digest"]:
            raise ExperimentFailure("model_digest_changed")

    def request(self, method, path, payload):
        self.chats += 1
        self.payloads.append(copy.deepcopy(payload))
        raw = self.outputs.pop(0)
        return {"response": {"model": extracts.MODEL, "done": True,
                             "done_reason": "stop", "prompt_eval_count": 100,
                             "message": {"content": raw}}}


class CallExperimentTests(unittest.TestCase):
    def test_reuse_saves_show_only_and_rechecks_identity_each_later_chat(self):
        client = Client([])
        scoped = ScopedMetadataClient(client)
        left = scoped.receipt(extracts.MODEL)
        left["show_response"]["capabilities"].clear()
        right = scoped.receipt(extracts.MODEL)
        self.assertEqual(client.receipts, 1)
        self.assertEqual(client.identities, 1)
        self.assertTrue(right["show_response"]["capabilities"])

    def test_no_metadata_or_completion_reuse_across_request_scopes(self):
        client = Client(['{"span_ids":[]}', '{"span_ids":["S001"]}'])
        fields = {"messages": [{"role": "user", "content": "same"}]}
        a = BoundProvider(client, reuse_show=True)
        b = BoundProvider(client, reuse_show=True)
        self.assertNotEqual(a.chat(fields)[0], b.chat(fields)[0])
        self.assertEqual((client.receipts, client.chats), (2, 2))

    def test_changed_digest_and_configuration_fail_before_chat(self):
        client = Client(['{"span_ids":[]}'])
        provider = BoundProvider(client, reuse_show=True)
        fields = {"messages": []}
        provider.chat(fields)
        client.digest = "0" * 64
        with self.assertRaisesRegex(ExperimentFailure, "model_digest_changed"):
            provider.chat(fields)
        self.assertEqual(client.chats, 1)
        altered = copy.deepcopy(extracts.configuration())
        altered["profile"]["temperature"] = 0.2
        with self.assertRaisesRegex(ExperimentFailure, "frozen_configuration_changed"):
            BoundProvider(client, configuration=altered).chat(fields)
        self.assertEqual(client.chats, 1)

    def test_frozen_full_fields_reject_changed_verifier_or_evidence(self):
        fields = {"messages": [{"role": "user", "content": "whole original evidence"}], "format": {}}
        expected = [{"fields": fields}]
        for replacement in ("different evidence", "different verifier prompt"):
            client = Client(['{"span_ids":[]}'])
            provider = BoundProvider(client, reuse_show=True, expected=expected)
            with self.assertRaisesRegex(ExperimentFailure, "frozen_call_fields_changed"):
                provider.chat({"messages": [{"role": "user", "content": replacement}], "format": {}})
            self.assertEqual(client.chats, 0)

    def test_frozen_planner_drift_is_retained_and_not_replaced(self):
        values = Intake(**intake())
        fields = planning_fields(values)
        first = '{"issues":[{"topic_id":"traditional_knowledge","context_id":"I001"}],"needs_clarification":false}'
        changed = '{"issues":[{"topic_id":"prior_art","context_id":"I001"}],"needs_clarification":false}'
        client = Client([changed])
        provider = BoundProvider(client, reuse_show=True,
                                 expected=[{"fields": fields, "raw": first}], intake=values)
        with self.assertRaisesRegex(ExperimentFailure, "frozen_planner_changed"):
            provider.chat(fields)
        self.assertEqual(client.chats, 1)
        self.assertEqual(provider.records[0]["raw"], changed)

    def test_frozen_plan_preserves_context_ids_even_if_display_prefix_matches(self):
        values = Intake(**intake(description="A" * 190 + "original", additional_context="A" * 190 + "different"))
        fields = planning_fields(values)
        first = '{"issues":[{"topic_id":"traditional_knowledge","context_id":"I001"}],"needs_clarification":false}'
        changed = first.replace("I001", "I003")
        client = Client([changed])
        provider = BoundProvider(client, expected=[{"fields": fields, "raw": first}], intake=values)
        with self.assertRaisesRegex(ExperimentFailure, "frozen_planner_changed"):
            provider.chat(fields)

    def test_incomplete_result_is_not_served_as_a_cached_completion(self):
        class Incomplete(Client):
            def request(self, *args):
                result = super().request(*args)
                result["response"]["done"] = self.chats > 1
                return result
        client = Incomplete(['{"span_ids":[]}', '{"span_ids":[]}'])
        provider = BoundProvider(client, reuse_show=True)
        with self.assertRaisesRegex(ExperimentFailure, "provider_incomplete_response"):
            provider.chat({"messages": []})
        provider.chat({"messages": []})
        self.assertEqual(client.chats, 2)

    def test_metadata_alternative_preserves_model_payload_and_archive_checks(self):
        plan = json.dumps({"issues": [{"topic_id": "traditional_knowledge", "context_id": "I001"}],
                           "needs_clarification": False})
        outputs = [plan, '{"span_ids":["S001"]}', '{"span_ids":["S001"]}']
        left, right = Client(outputs), Client(outputs)
        verified = [[], []]
        results = []
        for n, (client, reuse) in enumerate(((left, False), (right, True))):
            engine = PassportEngine(Search(), lambda hit, gid: verified[n].append((hit["chunk_id"], gid)),
                                    provider=BoundProvider(client, reuse_show=reuse))
            results.append(engine.build(Intake(**intake())))
        self.assertEqual(left.payloads, right.payloads)
        self.assertEqual(verified[0], verified[1])
        self.assertGreaterEqual(len(verified[0]), 3)
        for key in ("issues", "verified_evidence", "status", "unresolved_questions"):
            self.assertEqual(results[0][key], results[1][key])
        self.assertEqual((right.receipts, right.identities, right.chats), (1, 2, 3))

    def test_experiment_exercises_private_asgi_intake_and_response(self):
        plan = '{"issues":[{"topic_id":"traditional_knowledge","context_id":"I001"}],"needs_clarification":false}'
        client = Client([plan, '{"span_ids":["S001"]}', '{"span_ids":["S001"]}'])
        client.observations = []
        with patch("tools.passport_call_experiment.TimedClient", return_value=client), patch(
                "tools.passport_call_experiment.LocalSearch", return_value=Search()):
            result = http_attempt({"id": "synthetic", "intake": intake()}, archive=lambda *args: None)
        self.assertEqual(result["asgi_http_status"], 200)
        self.assertTrue(result["response"]["verified_evidence"])
        self.assertIn("serialization", result["server_timing"])

    def test_interrupted_capture_retains_already_completed_model_trace(self):
        interrupted = KeyboardInterrupt()
        interrupted.experiment_partial = {"calls": [{"raw": "partial synthetic output", "completed": True}],
                                          "provider_http_requests": [{"path": "/api/chat", "completed": True}]}
        search = Search()
        search.gid = "20261014T204940358650Z"
        catalog = {"cases": [{"id": key, "intake": intake()} for key in
                             ("traditional_ashwagandha_extract", "novel_process")]}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "data/next_stage/nba_fidelity_20261101/passport/interrupted"
            with patch("tools.passport_call_experiment.ROOT", root), patch(
                    "tools.passport_call_experiment.load_scenarios", return_value=(catalog, "synthetic")), patch(
                    "tools.passport_call_experiment.LocalSearch", return_value=search), patch(
                    "tools.passport_call_experiment.PublishedVerifier"), patch(
                    "tools.passport_call_experiment.CitationVerifier"), patch(
                    "tools.passport_call_experiment.http_attempt", side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    run(output)
            saved = json.loads((output / "observations.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "interrupted")
        self.assertEqual(saved["captures"][0]["status"], "interrupted")
        self.assertEqual(saved["captures"][0]["calls"], interrupted.experiment_partial["calls"])


if __name__ == "__main__":
    unittest.main()
