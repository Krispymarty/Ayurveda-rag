"""Verify the pre-model frozen dataset without loading an inference model."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from ipindia_corpus.indexer import generation
from ipindia_corpus.passage_evaluation import matches
from tools.freeze_llm_benchmark import (GENERATION, LABELS, MAX_EVIDENCE_CHARS, VERSION,
                                       canonical, digest, freeze, load_frozen, model_input)

ROOT = Path(__file__).resolve().parents[1]


class LLMDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.cases, cls.system, cls.schema = load_frozen(ROOT)
        _, documents, chunks = generation(ROOT, generation_id=GENERATION)
        cls.documents = {doc["document_id"]: doc for doc in documents}
        cls.chunks = {chunk["chunk_id"]: chunk for chunk in chunks}
        cls.labels = json.loads((ROOT / LABELS).read_text(encoding="utf-8"))

    def test_frozen_files_and_manifest_are_hash_bound_before_runs(self):
        self.assertTrue(self.manifest["frozen"])
        self.assertTrue(self.manifest["frozen_before_benchmark_model_runs"])
        self.assertEqual(self.manifest["case_count"], 45)
        self.assertEqual(len({case["id"] for case in self.cases}), 45)
        self.assertEqual(self.manifest["generation_id"], GENERATION)
        self.assertEqual(self.manifest["source_labels_sha256"], digest((ROOT / LABELS).read_bytes()))

    def test_thirty_prior_reviewed_questions_and_targets_are_preserved(self):
        archived = [case for case in self.cases if case["review_kind"] == "prior_archived_factual"]
        self.assertEqual(len(archived), 30)
        by_id = {query["id"]: query for query in self.labels["queries"]}
        for case in archived:
            original = by_id[case["source_case_id"]]
            self.assertEqual(case["question"], original["query"])
            self.assertEqual(case["original_reviewed_targets"], original["targets"])
            self.assertEqual(case["expected"]["policy"], "quote_targets")

    def test_archived_evidence_has_exact_published_text_and_identity(self):
        for case in self.cases:
            for evidence in case["evidence"]:
                if evidence["provenance"] != "verified_immutable_archive":
                    continue
                chunk = self.chunks[evidence["chunk_id"]]
                document = self.documents[evidence["document_id"]]
                self.assertEqual(evidence["text"], chunk["text"])
                self.assertEqual(evidence["content_hash"], digest(chunk["text"].encode()))
                self.assertEqual(evidence["raw_sha256"], document["sha256"])
                self.assertEqual(evidence["source_url"], document["source_url"])
                self.assertEqual(evidence["document_id"], chunk["document_id"])
                self.assertEqual(evidence["status"], "unknown")
                for key in ("section", "rule", "page_start", "page_end"):
                    self.assertEqual(evidence[key], chunk.get(key))

    def test_targets_are_alternative_groups_and_decoys_cannot_satisfy_them(self):
        alternatives = 0
        for case in self.cases:
            evidence = {hit["chunk_id"]: hit for hit in case["evidence"]}
            if case["expected"]["policy"] != "quote_targets":
                continue
            expected_ids = set(case["expected"]["allowed_claim_chunk_ids"])
            self.assertEqual(expected_ids, {identifier for group in case["expected"]["target_groups"]
                                            for identifier in group["required_chunk_ids"]})
            for group in case["expected"]["target_groups"]:
                self.assertTrue(group["required_chunk_ids"])
                alternatives += len(group["required_chunk_ids"]) > 1
                for identifier in group["required_chunk_ids"]:
                    self.assertIn(identifier, evidence)
                    self.assertIn(group["required_contains"].casefold(), evidence[identifier]["text"].casefold())
                    self.assertIn(evidence[identifier]["raw_sha256"], group["raw_sha256"])
            decoys = [hit for hit in case["evidence"] if hit["provenance"] == "verified_immutable_archive"
                      and hit["chunk_id"] not in expected_ids]
            self.assertLessEqual(len(decoys), 2)
            if case["review_kind"] == "prior_archived_factual":
                for target, group in zip(case["original_reviewed_targets"], case["expected"]["target_groups"]):
                    self.assertTrue(all(matches(self.chunks[identifier], target) for identifier in group["required_chunk_ids"]))
        self.assertGreater(alternatives, 0)

    def test_model_input_order_system_schema_and_budget_are_identical(self):
        for case in self.cases:
            expected = model_input(case, self.system, self.schema)
            self.assertEqual(digest(canonical(expected)), case["model_input_sha256"])
            self.assertLessEqual(len(canonical(case["evidence"])), MAX_EVIDENCE_CHARS)
            changed = deepcopy(case)
            changed["question"] += " changed"
            self.assertNotEqual(digest(canonical(model_input(changed, self.system, self.schema))), case["model_input_sha256"])
            if len(case["evidence"]) > 1:
                changed = deepcopy(case)
                changed["evidence"].reverse()
                self.assertNotEqual(digest(canonical(model_input(changed, self.system, self.schema))), case["model_input_sha256"])

    def test_adversarial_cases_keep_useful_quote_targets_and_have_no_forged_provenance(self):
        attacks = [case for case in self.cases if case["id"].startswith("adversarial-")]
        self.assertEqual(len(attacks), 3)
        for case in attacks:
            self.assertEqual(case["review_kind"], "synthetic_adversarial")
            self.assertEqual(case["expected"]["policy"], "quote_targets")
            attack = case["evidence"][0]
            self.assertEqual(attack["provenance"], "explicit_synthetic_fixture")
            self.assertIsNone(attack["raw_sha256"])
            self.assertIsNone(attack["source_url"])
            self.assertNotIn(attack["chunk_id"], case["expected"]["allowed_claim_chunk_ids"])

    def test_exploration_and_abstention_are_separate_from_reviewed_legal_ground_truth(self):
        translations = [case for case in self.cases if case["review_kind"] == "exploratory_translation"]
        self.assertEqual(len(translations), 6)
        self.assertTrue(all(any("\u0900" <= character <= "\u097f" for character in case["question"]) for case in translations))
        empty = [case for case in self.cases if case["expected"]["policy"] == "empty_claims"]
        self.assertEqual(len(empty), 5)
        self.assertTrue(all(case["review_kind"] == "synthetic_adversarial" and
                            not case["expected"]["target_groups"] for case in empty))
        short = next(case for case in self.cases if case["id"] == "short-unit-01")
        self.assertEqual(short["review_kind"], "exploratory_archive_probe")
        self.assertTrue(short["evidence"][0]["short_unit"])
        self.assertLess(len(short["evidence"][0]["text"]), 80)

    def test_existing_freeze_cannot_be_overwritten_or_written_outside_project(self):
        directory = ROOT / "data/benchmarks" / VERSION
        before = (directory / "manifest.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "already_frozen"):
            freeze(ROOT)
        with self.assertRaisesRegex(ValueError, "inside_project"):
            freeze(ROOT, ROOT.parent / "forbidden-benchmark-output")
        self.assertEqual(before, (directory / "manifest.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
