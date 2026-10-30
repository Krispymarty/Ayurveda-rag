import json
from pathlib import Path
import tempfile
import unittest

from ipindia_corpus.llm_experiment import canonical, sha256
from tools.resume_llm_comparison import read_rows, pending_cases


class ResumeTests(unittest.TestCase):
    run_id = "compare-" + "a" * 32
    case = {"id": "one", "category": "exact_extraction", "model_input_sha256": "input", "evidence_sha256": "evidence"}
    bindings = {"provider": "unchanged"}

    def row(self, **changes):
        return {"run_id": self.run_id, "case_id": "one", "category": "exact_extraction",
                "model": "ministral-3:3b", "model_digest": "b" * 64,
                "model_input_sha256": "input", "evidence_sha256": "evidence", "code_sha256": self.bindings,
                "profile": {"num_predict": 200, "num_ctx": 4096, "keep_alive": "30m"},
                "cohort_before": {"cohort_safe": True}, "cohort_after": {"cohort_safe": True},
                "phase": "comparison_warm", "pass": 1, "status": "timeout", "score": {"passed": False}, **changes}

    def write(self, root, row, prefix="first"):
        raw = canonical(row)
        path = root / "reports/llm/observations" / self.run_id / (prefix + "-" + sha256(raw) + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return path

    def test_saved_timeout_is_done_and_never_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write(root, self.row())
            rows, inventory = read_rows(root, [self.run_id], {"one": self.case}, self.bindings)
            self.assertEqual(len(inventory), 1)
            self.assertEqual([number for number, _ in pending_cases(rows, "ministral-3:3b", [self.case])], [2, 3])
            self.assertFalse(rows[0]["score"]["passed"])

    def test_duplicate_attempt_is_rejected_even_if_one_passed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write(root, self.row())
            self.write(root, self.row(status="completed", score={"passed": True}), "second")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_rows(root, [self.run_id], {"one": self.case}, self.bindings)

    def test_changed_input_code_profile_or_cohort_is_rejected(self):
        for changes in ({"model_input_sha256": "different"}, {"code_sha256": {}},
                        {"profile": {"num_predict": 600}}, {"cohort_after": {"cohort_safe": False}}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.write(root, self.row(**changes))
                with self.assertRaises(ValueError):
                    read_rows(root, [self.run_id], {"one": self.case}, self.bindings)

    def test_changed_archived_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.write(root, self.row())
            path.write_text(json.dumps(self.row(status="completed")), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash_mismatch"):
                read_rows(root, [self.run_id], {"one": self.case}, self.bindings)

    def test_run_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "run_id"):
                read_rows(Path(temporary), ["../outside"], {"one": self.case}, self.bindings)
