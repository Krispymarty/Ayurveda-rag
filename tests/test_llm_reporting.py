import json
from pathlib import Path
import tempfile
import unittest

from ipindia_corpus.llm_experiment import canonical, sha256
from tools.report_llm_results import quality, verify_observation


class ReportingTests(unittest.TestCase):
    def test_abstention_never_improves_quote_precision(self):
        cases = {"a": {"expected": {"policy": "empty_claims"}, "evidence": []}}
        row = {"case_id": "a", "provider_called": True, "score": {"schema_valid": True, "correct_abstention": True},
               "response_json": {"message": {"content": '{"claims":[]}'}}}
        result = quality([row], cases)
        self.assertEqual(result["correct_abstention_rate"], 1)
        self.assertEqual(result["emitted_claims"], 0)
        self.assertIsNone(result["quote_precision"])

    def test_fake_id_counts_in_emitted_claim_denominator(self):
        cases = {"a": {"expected": {"policy": "quote_targets"}, "evidence": [
            {"chunk_id": "real", "source": "ipindia", "text": "An original exact quotation."}]}}
        row = {"case_id": "a", "provider_called": True, "score": {"schema_valid": True, "passed": False},
               "response_json": {"message": {"content": json.dumps({"claims": [
                   {"chunk_id": "real", "quote": "An original exact quotation."},
                   {"chunk_id": "fake", "quote": "invented"}]})}}}
        result = quality([row], cases)
        self.assertEqual(result["quote_precision"], .5)
        self.assertEqual(result["positive_case_pass_rate"], 0)

    def test_archive_hash_and_report_equality_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "reports/llm/observations/run/one.json"
            path.parent.mkdir(parents=True)
            original = {"model_input_sha256": "input", "evidence_sha256": "evidence"}
            raw = canonical(original)
            path.write_bytes(raw)
            row = {**original, "immutable_observation_path": path.relative_to(root).as_posix(),
                   "immutable_observation_sha256": sha256(raw)}
            verify_observation(root, row, original)
            row["evidence_sha256"] = "tampered"
            with self.assertRaisesRegex(ValueError, "differs"):
                verify_observation(root, row, original)
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash_mismatch"):
                verify_observation(root, row, original)
