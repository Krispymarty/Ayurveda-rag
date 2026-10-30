from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from ipindia_corpus.llm_experiment import canonical, sha256
from tools.benchmark_reliability import system_score, summarize, run
from tools.freeze_reliability_benchmark import load, VERSION
from test_span_selection import case, hit


class ReliabilityBenchmarkTests(unittest.TestCase):
    def test_new_splits_are_balanced_distinct_and_bound(self):
        root = Path(__file__).resolve().parents[1]
        manifest, cases = load(root, "development")
        self.assertEqual(len(cases), 30)
        self.assertEqual(sum(c["expected"]["policy"] == "quote_targets" for c in cases), 15)
        for source in cases:
            self.assertEqual(sha256(canonical(source["evidence"])), source["evidence_sha256"])
        self.assertEqual(manifest["cases"], 60)
        with self.assertRaisesRegex(ValueError, "freeze_candidate"):
            load(root, "heldout")
        with self.assertRaisesRegex(ValueError, "does_not_bind"):
            load(root, "heldout", candidate_receipt={"unrelated": True})

    def test_tampering_is_rejected_without_reading_heldout_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "data/benchmarks" / VERSION
            folder.mkdir(parents=True)
            payload = canonical({"files": {"development.jsonl": "a" * 64}})
            (folder / "manifest.json").write_bytes(payload)
            (folder / "manifest.sha256").write_text(sha256(payload))
            (folder / "development.jsonl").write_text("altered")
            with self.assertRaisesRegex(ValueError, "cases_changed"):
                load(Path(tmp))

    def test_real_non_target_citation_is_disallowed_not_fabricated(self):
        source = case()
        decoy = hit("another-real-hit")
        result = {"status": "answered", "retrieval_evidence": [decoy], "claims": [{"citations": [
            {"chunk_id": decoy["chunk_id"], "quote": decoy["text"]}]}]}
        score = system_score(source, result)
        self.assertIn("DISALLOWED_CLAIM", score["failures"])
        self.assertNotIn("FABRICATED_CITATION", score["failures"])

    def test_backend_abstention_is_system_success_not_model_credit(self):
        source = case()
        source["expected"] = {"policy": "empty_claims", "target_groups": [], "allowed_claim_chunk_ids": []}
        result = {"status": "abstained", "reason": "current_legal_status_unverified", "claims": []}
        row = {"case_id": source["id"], "pass": 1, "model_called": False, "result": result,
               "system_score": system_score(source, result), "http_or_pipeline_ms": .1}
        summary = summarize([row], [source])
        self.assertEqual(summary["system_case_success"], 1)
        self.assertFalse(summary["model_credit_for_backend_guards"])
        changed = deepcopy(row)
        changed["result"]["status"] = "unavailable"
        self.assertFalse(system_score(source, changed["result"])["passed"])

    def test_interrupted_intent_is_never_sent_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, source, candidate = Path(tmp), case(), {"configuration": "fixture"}
            checksum = sha256(canonical(candidate))
            folder = root / "data/reliability/observations" / checksum / "controlled-development-1passes"
            (folder / "intents").mkdir(parents=True)
            (folder / "intents/1-fixture.json").write_bytes(canonical({"case_id": "fixture", "pass": 1,
                "candidate_sha256": checksum, "case_sha256": sha256(canonical(source))}))
            provider = SimpleNamespace(client=None)
            with patch("tools.benchmark_reliability.binding", return_value=candidate), \
                 patch("tools.benchmark_reliability.load", return_value=({}, [source])), \
                 patch("tools.benchmark_reliability.ArchiveVerifier"), \
                 patch("tools.benchmark_reliability.SpanProvider", return_value=provider), \
                 patch("tools.benchmark_reliability.answer") as model:
                first = run(root, passes=1)
                second = run(root, passes=1)
                model.assert_not_called()
            self.assertEqual(first["summary"], second["summary"])
            self.assertEqual(first["summary"]["attempts"], 1)
            self.assertEqual(first["summary"]["system_case_success"], 0)
