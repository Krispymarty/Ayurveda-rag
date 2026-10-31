"""Synthetic metadata fixtures; no source or human certification."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from ipindia_corpus.generations import _metadata


class PublicationMetadataTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        raw = b"synthetic original, not legal evidence"
        path = self.root / "data/raw/nba/synthetic.pdf"
        path.parent.mkdir(parents=True)
        path.write_bytes(raw)
        self.title = "Synthetic Biodiversity Regulations, 2025"
        self.row = {"canonical_url": "https://nbaindia.nic.in/synthetic.pdf",
                    "canonical_id": "synthetic", "sha256": hashlib.sha256(raw).hexdigest(),
                    "raw_path": "data/raw/nba/synthetic.pdf", "retrieved_at": "2026-11-01T00:00:00+00:00",
                    "last_checked_at": "2026-11-01T00:00:00+00:00", "verified_version": None}
        self.identity = {"title": self.title, "document_type": "Regulations",
                         "source_url": self.row["canonical_url"], "raw_sha256": self.row["sha256"],
                         "publication_date": "2025-04-30", "notification_date": "2025-04-29",
                         "notification_number": "SYNTHETIC-1", "gazette_identifier": "SYNTHETIC-2",
                         "gazette_issue": "Synthetic fixture only", "issuer": "Synthetic authority",
                         "publisher": "Synthetic publisher", "publication_anchors": {"header": 1}}

    def metadata(self, identity=None, source="nba"):
        row = {**self.row, "metadata_json": json.dumps({"verified_identity": identity or self.identity})}
        return _metadata(row, source, self.title, "[Page 1]\nSynthetic fixture only.", self.root)

    def test_regulations_identity_survives_without_legal_effect_inference(self):
        value = self.metadata({**self.identity, "status": "in_force", "effective_from": "2025-04-30"})
        self.assertEqual(value["document_type"], "Regulations")
        self.assertEqual(value["publication_date"], "2025-04-30")
        self.assertEqual(value["notification_date"], "2025-04-29")
        self.assertEqual(value["gazette_identifier"], "SYNTHETIC-2")
        self.assertEqual(value["publication_anchors"], {"header": 1})
        self.assertEqual(value["status"], "unknown")
        self.assertIsNone(value["effective_from"])
        self.assertIsNone(value["effective_to"])
        self.assertIsNone(value["commencement_date"])
        self.assertTrue(value["metadata_review_required"])
        self.assertEqual(value["identity_review"], "technical_candidate_pending_human_review")
        self.assertNotIn("effective_from", value["verified_identity"])

    def test_mismatched_original_url_or_title_is_rejected(self):
        for key, bad in (("raw_sha256", "0" * 64), ("source_url", "https://evil.test/a.pdf"),
                         ("title", "Another instrument")):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "publication_identity_binding"):
                self.metadata({**self.identity, key: bad})

    def test_dates_and_page_anchors_must_be_precise(self):
        for patch in ({"publication_date": "2025-02-30"}, {"notification_date": "yesterday"},
                      {"publication_anchors": {"header": True}}, {"publication_anchors": {"header": 2}},
                      {"document_type": "Guidance"}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                self.metadata({**self.identity, **patch})

    def test_ipindia_metadata_behavior_is_unchanged(self):
        value = self.metadata(source="ipindia")
        self.assertEqual(value["document_type"], "Guidance")
        self.assertIsNone(value["publication_date"])
        self.assertNotIn("notification_date", value)
        self.assertNotIn("identity_review", value)

    def test_historical_relationship_is_observation_not_legal_effect(self):
        value = self.metadata({**self.identity, "historical_relationships": [
            {"claim": "Synthetic supersession wording for software test only", "page": 1,
             "legal_effect_verified": True}]})
        self.assertFalse(value["historical_relationships"][0]["legal_effect_verified"])
        self.assertIsNone(value["effective_from"])
        self.assertTrue(value["metadata_review_required"])


if __name__ == "__main__":
    unittest.main()
