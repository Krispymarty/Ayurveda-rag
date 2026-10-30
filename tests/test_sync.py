import tempfile
import unittest
from pathlib import Path

from ipindia_corpus.sync import Ledger


class SyncLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        root = Path(self.temp.name)
        self.ledger = Ledger(root / "manifest.sqlite", root / "raw")
        self.url = "https://ipindia.gov.in/resource/example"
        self.run = self.ledger.start("ipindia")

    def tearDown(self):
        self.ledger.close()
        self.temp.cleanup()

    def test_new_unchanged_metadata_and_changed_preserve_versions(self):
        self.assertEqual(self.ledger.observe(self.run, self.url, b"legal text v1"), "NEW_DOCUMENT")
        self.assertEqual(self.ledger.observe(self.run, self.url, b"legal text v1"), "UNCHANGED")
        self.assertEqual(self.ledger.observe(self.run, self.url, b"legal text v1", metadata={"title": "Act"}), "METADATA_CHANGED")
        self.assertEqual(self.ledger.observe(self.run, self.url, b"legal text v2"), "CONTENT_CHANGED")
        self.assertEqual(self.ledger.db.execute("SELECT COUNT(*) FROM document_versions").fetchone()[0], 2)
        self.assertEqual(self.ledger.db.execute("SELECT status FROM document_manifest").fetchone()[0], "unknown")

    def test_explicit_verified_version_and_url_move(self):
        self.ledger.observe(self.run, self.url, b"law v1", verified_version="official 2020")
        canonical_id = self.ledger.db.execute("SELECT canonical_id FROM document_manifest").fetchone()[0]
        self.assertEqual(self.ledger.observe(self.run, self.url, b"law v2", verified_version="official 2025"), "NEW_VERSION")
        moved = "https://ipindia.gov.in/resource/renamed"
        self.assertEqual(self.ledger.observe(self.run, moved, b"law v2", canonical_id_hint=canonical_id), "UNCHANGED")
        self.assertEqual(self.ledger.db.execute("SELECT COUNT(*) FROM document_urls").fetchone()[0], 2)
        self.assertEqual(self.ledger.db.execute("SELECT COUNT(*) FROM sync_changes WHERE change_type='URL_MOVED'").fetchone()[0], 1)

    def test_failure_and_removed_source_keep_history(self):
        self.ledger.observe(self.run, self.url, b"law v1")
        self.ledger.failed(self.run, self.url, "server unavailable")
        with self.assertRaises(ValueError):
            self.ledger.removed_from_source(self.run, self.url, verified_discovery=False)
        self.ledger.removed_from_source(self.run, self.url, verified_discovery=True)
        self.assertEqual(self.ledger.db.execute("SELECT COUNT(*) FROM document_versions").fetchone()[0], 1)
        report = self.ledger.finish(self.run)
        self.assertEqual(report["changes"]["FAILED"], 1)
        self.assertEqual(report["changes"]["REMOVED_FROM_SOURCE"], 1)

    def test_official_domain_is_enforced(self):
        with self.assertRaises(ValueError):
            self.ledger.observe(self.run, "https://ipindia.gov.in.evil.test/law", b"bad")


if __name__ == "__main__":
    unittest.main()
