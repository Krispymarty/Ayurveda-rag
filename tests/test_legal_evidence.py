import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from ipindia_corpus.legal_evidence import assess_document_status, build_report, validate_annotation


class LegalEvidenceTests(unittest.TestCase):
    def setUp(self):
        import fitz
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        raw = self.root / "data/raw/ipindia/amendment.pdf"
        raw.parent.mkdir(parents=True)
        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_text((45, 70), "These rules amend the Patents Rules, 2003.")
        pdf.save(raw)
        pdf.close()
        digest = hashlib.sha256(raw.read_bytes()).hexdigest()
        (self.root / "documents.jsonl").write_text(json.dumps({"document_id": "test-document", "source": "ipindia", "sha256": digest, "source_url": "https://ipindia.gov.in/test.pdf"}) + "\n", encoding="utf8")
        self.annotation = {"relation": "amends", "target_title": "Patents Rules, 2003", "reviewer": "fixture reviewer", "review_note": "Synthetic exact-text test, not legal review", "evidence": {"document_id": "test-document", "source": "ipindia", "official_url": "https://ipindia.gov.in/test.pdf", "raw_path": "data/raw/ipindia/amendment.pdf", "sha256": digest, "page": 1, "provision": "Rule 1", "quote": "These rules amend the Patents Rules, 2003."}}

    def test_valid_statement_never_sets_current_status(self):
        result = validate_annotation(self.root, self.annotation)
        self.assertTrue(result["verified_archived_statement"])
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["current_force_established"])

    def test_hash_page_quote_identity_url_and_target_are_bound(self):
        mutations = [("sha256", "0" * 64), ("page", 2), ("page", True), ("quote", "Fabricated text"), ("document_id", "other"), ("official_url", "https://ipindia.gov.in/other.pdf"), ("official_url", "https://ipindia.gov.in.evil/a")]
        for field, value in mutations:
            with self.subTest(field=field, value=value):
                annotation = copy.deepcopy(self.annotation)
                annotation["evidence"][field] = value
                with self.assertRaises(ValueError):
                    validate_annotation(self.root, annotation)
        annotation = copy.deepcopy(self.annotation)
        annotation["target_title"] = "Trade Marks Rules, 2017"
        with self.assertRaises(ValueError):
            validate_annotation(self.root, annotation)

    def test_relationship_cannot_be_inferred_from_title_or_unreviewed_assertion(self):
        for field, value in (("relation", "currently_in_force"), ("relation", "repeals"), ("reviewer", "")):
            annotation = copy.deepcopy(self.annotation)
            annotation[field] = value
            with self.assertRaises(ValueError):
                validate_annotation(self.root, annotation)

    def test_temporal_conflict_abstains_from_status(self):
        a = {**self.annotation, "asserted_status": "repealed"}
        b = {**self.annotation, "asserted_status": "in_force"}
        result = assess_document_status([a, b], as_of="2026-10-15")
        self.assertTrue(result["conflicting_assertions"])
        self.assertTrue(result["temporal_uncertainty"])
        self.assertEqual(result["status"], "unknown")

    def configure(self, annotations):
        path = self.root / "config/legal_annotations.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"schema_version": 1, "annotations": annotations}), encoding="utf8")

    def replace_original_text(self, text):
        import fitz
        raw = self.root / self.annotation["evidence"]["raw_path"]
        pdf = fitz.open()
        pdf.new_page().insert_textbox((45, 45, 550, 400), text)
        raw.write_bytes(pdf.tobytes())
        pdf.close()
        digest = hashlib.sha256(raw.read_bytes()).hexdigest()
        self.annotation["evidence"].update(sha256=digest, quote=text)
        (self.root / "documents.jsonl").write_text(json.dumps({"document_id": "test-document", "source": "ipindia", "sha256": digest, "source_url": "https://ipindia.gov.in/test.pdf"}) + "\n", encoding="utf8")

    def test_report_ingests_explicit_annotation_configuration(self):
        self.annotation["annotation_id"] = "fixture-arbitrary-document"
        self.configure([self.annotation])
        report = build_report(self.root)
        self.assertEqual(len(report["verified_archived_relationships"]), 1)
        self.assertEqual(report["verified_archived_relationships"][0]["evidence"]["document_id"], "test-document")
        self.assertTrue(report["annotation_config_sha256"])
        self.assertEqual(report["current_in_force_documents"], 0)
        self.configure([])
        self.assertEqual(build_report(self.root)["verified_archived_relationships"], [])

    def test_duplicate_configuration_ids_and_tampered_quotes_are_rejected(self):
        self.annotation["annotation_id"] = "fixture"
        self.configure([self.annotation, self.annotation])
        with self.assertRaises(ValueError):
            build_report(self.root)
        self.annotation["evidence"]["quote"] = "Invented amendment statement"
        self.configure([self.annotation])
        report = build_report(self.root)
        self.assertEqual(report["verified_archived_relationships"], [])
        self.assertEqual(len(report["unresolved_candidates"]), 1)

    def test_draft_proposals_remain_separate_from_operative_wording(self):
        self.replace_original_text("The following draft rules amend the Patents Rules, 2003 which the Government proposes to make.")
        self.annotation["annotation_id"] = "draft-fixture"
        with self.assertRaises(ValueError):
            validate_annotation(self.root, self.annotation)
        self.annotation["statement_scope"] = "draft_proposal"
        self.configure([self.annotation])
        report = build_report(self.root)
        self.assertEqual(report["verified_archived_relationships"], [])
        self.assertEqual(len(report["verified_draft_proposals"]), 1)
        proposal = report["verified_draft_proposals"][0]
        self.assertFalse(proposal["relationship_effect_established"])
        self.assertEqual(proposal["status"], "unknown")

    def test_publication_uncertainty_requires_bound_qualifying_quote(self):
        self.annotation["statement_scope"] = "publication_uncertain"
        with self.assertRaises(ValueError):
            validate_annotation(self.root, self.annotation)
        self.annotation["qualifying_evidence"] = [{"quote": "Missing notification placeholder", "page": 1}]
        with self.assertRaises(ValueError):
            validate_annotation(self.root, self.annotation)
        self.annotation["qualifying_evidence"] = [{"quote": "These rules", "page": 1}]
        result = validate_annotation(self.root, self.annotation)
        self.assertEqual(result["statement_scope"], "publication_uncertain")
        self.assertFalse(result["current_force_established"])


if __name__ == "__main__":
    unittest.main()
