from pathlib import Path
import json
import tempfile
import unittest

from ipindia_corpus.llm_experiment import canonical, sha256, code_bindings
from tools.benchmark_citation_selection import checkpoint_rows, _completed_comparison


class CitationCheckpointTests(unittest.TestCase):
    run_id = "citation-enum-" + "a" * 32
    profile = {"num_predict": 200, "num_ctx": 4096, "keep_alive": "30m"}
    case = {"id": "one", "model_input_sha256": "derived", "evidence_sha256": "evidence"}

    def row(self, **changes):
        return {"run_id": self.run_id, "case_id": "one", "model": "ministral-3:3b",
                "model_digest": "b" * 64, "profile": self.profile,
                "model_input_sha256": "derived", "evidence_sha256": "evidence", "code_sha256": code_bindings(),
                "phase": "citation_enum_warm", "pass": 1, "status": "timeout", "score": {"passed": False},
                "cohort_before": {"cohort_safe": True}, "cohort_after": {"cohort_safe": True}, **changes}

    def write(self, root, row, prefix="one"):
        raw = canonical(row)
        path = root / "reports/llm/observations" / self.run_id / (prefix + "-" + sha256(raw) + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return path

    def read(self, root):
        return checkpoint_rows(root, [self.run_id], {"one": self.case}, model="ministral-3:3b",
                               model_digest="b" * 64, profile=self.profile)

    def test_timeout_is_retained_as_a_failed_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write(root, self.row())
            rows, warmups = self.read(root)
            self.assertEqual(len(rows), 1)
            self.assertFalse(rows[0]["score"]["passed"])
            self.assertEqual(warmups, [])

    def test_duplicate_case_cannot_replace_a_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write(root, self.row())
            self.write(root, self.row(status="completed", score={"passed": True}), "two")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                self.read(root)

    def test_changed_input_digest_profile_or_bytes_are_rejected(self):
        for changes in ({"model_input_sha256": "old-unconstrained-input"}, {"model_digest": "c" * 64},
                        {"profile": {"num_predict": 600}}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.write(root, self.row(**changes))
                with self.assertRaisesRegex(ValueError, "binding"):
                    self.read(root)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.write(root, self.row())
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash_mismatch"):
                self.read(root)

    def test_incomplete_main_model_cannot_unlock_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "reports/llm/model_comparison.json"
            path.parent.mkdir(parents=True)
            result = {"status": "completed", "protected_state_unchanged": True,
                      "cases_per_pass": 1, "passes": 1, "models": [
                          {"model": model, "status": "measured", "observations": [{"pass": 1, "case_id": "one"}]}
                          for model in ("ministral-3:3b", "qwen3.5:4b", "gemma3:4b")]}
            def store():
                raw = canonical(result)
                path.write_bytes(raw)
                path.with_suffix(".json.sha256").write_text(sha256(raw), encoding="ascii")
            store()
            self.assertEqual(_completed_comparison(root)[0], json.loads(path.read_bytes()))
            result["models"][-1]["observations"] = []
            store()
            with self.assertRaisesRegex(ValueError, "finish_main"):
                _completed_comparison(root)
