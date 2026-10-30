"""Factual benchmark review preserves historical labels and requires archived evidence."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from ipindia_corpus.benchmark_review import create_reviewed, verify_bundle

REPOSITORY = Path(__file__).resolve().parents[1]
GENERATION = "20261014T204940358650Z"
VERSION = "ipindia-reviewed-test-v1"


def digest(data):
    return hashlib.sha256(data).hexdigest()


class BenchmarkReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "tests").mkdir()
        raw_path = self.root / "data/raw/designs-fixture.pdf"
        raw_path.parent.mkdir(parents=True)
        with fitz.open() as pdf:
            for text in ("Designs fixture cover", "Definitions and introductory provisions", "5. Registration of designs upon payment of the prescribed fee."):
                page = pdf.new_page()
                page.insert_text((50, 100), text)
            raw = pdf.tobytes()
        raw_path.write_bytes(raw)
        self.document = {"source": "ipindia", "source_url": "https://ipindia.gov.in/archive/designs-fixture.pdf",
                         "sha256": digest(raw), "raw_path": "data/raw/designs-fixture.pdf", "document_id": "designs-fixture"}
        original = json.loads((REPOSITORY / "tests/retrieval_queries.json").read_text(encoding="utf-8"))
        expanded = json.loads((REPOSITORY / "tests/ipindia_retrieval_expanded.json").read_text(encoding="utf-8"))
        for dataset in (original, expanded):
            for query in dataset["queries"]:
                if query["id"] in {"ipindia-10", "ipindia-28"}:
                    for target in query["targets"]:
                        target["urls"] = [self.document["source_url"]]
                        target["raw_sha256"] = [self.document["sha256"]]
        self.original_path = self.root / "tests/retrieval_queries.json"
        self.expanded_path = self.root / "tests/ipindia_retrieval_expanded.json"
        self.original_path.write_text(json.dumps(original, indent=2), encoding="utf-8")
        expanded["original_label_sha256"] = digest(self.original_path.read_bytes())
        self.expanded_path.write_text(json.dumps(expanded, indent=2), encoding="utf-8")
        self.expanded_path.with_suffix(".sha256").write_text(digest(self.expanded_path.read_bytes()))
        self.historical = {path: path.read_bytes() for path in (self.original_path, self.expanded_path, self.expanded_path.with_suffix(".sha256"))}
        self.chunks = []
        for query in expanded["queries"]:
            for index, target in enumerate(query["targets"]):
                page = 3 if query["id"] in {"ipindia-10", "ipindia-28"} else target.get("page")
                self.chunks.append({"source": target["source"], "source_url": target["urls"][0],
                                    "section": target.get("section"), "rule": target.get("rule"),
                                    "page_start": page, "page_end": page, "text": target.get("contains") or "Fixture legal provision.",
                                    "chunk_id": f"{query['id']}-{index}", "document_id": "designs-fixture"})
        self.bundle = self.root / "data/benchmarks" / VERSION

    def tearDown(self):
        self.temporary.cleanup()

    def create(self):
        with patch("ipindia_corpus.benchmark_review.generation", return_value=(GENERATION, [self.document], self.chunks)):
            return create_reviewed(self.root, generation_id=GENERATION, version=VERSION, reviewer="Fixture factual reviewer")

    def test_review_corrects_physical_page_with_evidence_without_rewriting_history(self):
        manifest = self.create()
        for path, value in self.historical.items():
            self.assertEqual(path.read_bytes(), value)
        reviewed = json.loads((self.bundle / "expanded.json").read_text())
        historical = json.loads(self.historical[self.expanded_path])
        expected = copy.deepcopy(historical["queries"])
        for query in expected:
            if query["id"] in {"ipindia-10", "ipindia-28"}:
                for target in query["targets"]:
                    target["page"] = 3
        self.assertEqual(reviewed["queries"], expected)
        self.assertEqual(reviewed["frozen_generation_id"], GENERATION)
        self.assertTrue(reviewed["frozen"])
        self.assertEqual(len(manifest["corrections"]), 3)
        for correction in manifest["corrections"]:
            self.assertEqual((correction["before"], correction["after"]), (2, 3))
            self.assertEqual(correction["source_url"], self.document["source_url"])
            self.assertEqual(correction["raw_sha256"], self.document["sha256"])
            self.assertIn("prescribed fee", correction["evidence_quote"])
            self.assertEqual(correction["legal_status_conclusion"], "unknown")
        self.assertFalse(manifest["legal_review"])

    def test_reusing_reviewed_version_is_immutable_and_does_not_reprocess_sources(self):
        first = self.create()
        before = {path.relative_to(self.bundle): path.read_bytes() for path in self.bundle.iterdir() if path.is_file()}
        with patch("ipindia_corpus.benchmark_review.generation", side_effect=AssertionError("Should reuse verified bundle")):
            second = create_reviewed(self.root, generation_id=GENERATION, version=VERSION, reviewer="Fixture factual reviewer")
        self.assertEqual(first, second)
        self.assertEqual(before, {path.relative_to(self.bundle): path.read_bytes() for path in self.bundle.iterdir() if path.is_file()})

    def test_changed_historical_inputs_are_rejected(self):
        self.create()
        for path in (self.original_path, self.expanded_path):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.assertRaises(ValueError):
                    verify_bundle(self.root, self.bundle)
                path.write_bytes(original)

    def test_changed_reviewed_passages_are_rejected_on_verification_and_reuse(self):
        self.create()
        for name in ("original.json", "expanded.json"):
            with self.subTest(name=name):
                path = self.bundle / name
                original = path.read_bytes()
                value = json.loads(original)
                value["queries"][0]["targets"][0]["contains"] = "A different provision"
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    verify_bundle(self.root, self.bundle)
                with self.assertRaises(ValueError):
                    create_reviewed(self.root, generation_id=GENERATION, version=VERSION)
                path.write_bytes(original)

    def test_page_correction_requires_observed_pdf_text_and_matching_original_hash(self):
        with self.subTest(reason="source identity changed"):
            actual = self.document["sha256"]
            self.document["sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "review_source_missing"):
                self.create()
            self.document["sha256"] = actual
        with self.subTest(reason="phrase already on annotated page"):
            original = (self.root / self.document["raw_path"]).read_bytes()
            with fitz.open(stream=original, filetype="pdf") as pdf:
                pdf[1].insert_text((50, 150), "prescribed fee")
                raw = pdf.tobytes()
            (self.root / self.document["raw_path"]).write_bytes(raw)
            # Public generation validation normally checks bytes; retain matching identities
            # here to isolate the factual physical-page check under test.
            new_hash = digest(raw)
            self.document["sha256"] = new_hash
            for path in (self.original_path, self.expanded_path):
                content = json.loads(path.read_text())
                for query in content["queries"]:
                    if query["id"] in {"ipindia-10", "ipindia-28"}:
                        query["targets"][0]["raw_sha256"] = [new_hash]
                path.write_text(json.dumps(content))
            expanded = json.loads(self.expanded_path.read_text())
            expanded["original_label_sha256"] = digest(self.original_path.read_bytes())
            self.expanded_path.write_text(json.dumps(expanded))
            self.expanded_path.with_suffix(".sha256").write_text(digest(self.expanded_path.read_bytes()))
            with self.assertRaisesRegex(ValueError, "physical_page_correction_not_verified"):
                self.create()
        self.assertFalse((self.bundle / "manifest.json").exists())

    def test_corrected_anchor_must_resolve_to_candidate_passage_before_bundle_exists(self):
        for chunk in self.chunks:
            if chunk["source_url"] == self.document["source_url"]:
                chunk["page_start"] = chunk["page_end"] = 2
        with self.assertRaisesRegex(ValueError, "corrected_passage_not_in_candidate"):
            self.create()
        self.assertFalse((self.bundle / "manifest.json").exists())
        for path, value in self.historical.items():
            self.assertEqual(path.read_bytes(), value)


if __name__ == "__main__":
    unittest.main()
