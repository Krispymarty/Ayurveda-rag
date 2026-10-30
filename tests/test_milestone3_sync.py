"""Real candidate-worker/ledger scenarios with local recorded-response fixtures."""
from datetime import timedelta
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ipindia_corpus.candidate_sync import run_once
from ipindia_corpus.connectors import Snapshot
from ipindia_corpus.sync import Ledger
import test_candidate_sync as worker_fixtures
from test_three_sources import BODY, URL, FakeClient


class SyncScenarioTests(unittest.TestCase):
    def ledger(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        ledger = Ledger(Path(temp.name) / "ledger.sqlite", Path(temp.name) / "raw")
        self.addCleanup(ledger.close)
        return ledger, ledger.start("ipindia"), "https://ipindia.gov.in/resource/fixture"

    def worker(self):
        fixture = worker_fixtures.CandidateSyncTests("test_successful_capture_queues_hash_bound_candidate_without_index")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        fixture.approved()
        return fixture

    def test_NEW_DOCUMENT_is_immutable_and_candidate_only(self):
        fixture = self.worker()
        with patch("ipindia_corpus.indexer.index", side_effect=AssertionError("production indexing forbidden")):
            result = run_once(fixture.root, fixture.now, clients={"tkdl": fixture.client()})
        self.assertFalse(result["production_mutation"])
        self.assertEqual(result["pending_candidates"][0]["state"], "pending_review")
        self.assertEqual(result["results"][0]["changes"]["NEW_DOCUMENT"], 1)
        fixture.assert_pointers()

    def test_UNCHANGED_preserves_bytes_and_conditional_metadata(self):
        ledger, run, url = self.ledger()
        ledger.observe(run, url, b"original", etag='"v1"', last_modified="Wed, 01 Jan 2020 00:00:00 GMT")
        ledger.checked(run, url, etag='"v1"')
        rows = ledger.db.execute("SELECT raw_path,sha256,etag,last_modified FROM document_manifest").fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sha256"], hashlib.sha256(b"original").hexdigest())
        self.assertTrue(rows[0]["last_modified"])
        self.assertEqual(ledger.db.execute("SELECT count(*) FROM document_versions").fetchone()[0], 1)

    def test_CONTENT_CHANGED_retains_old_raw_hash(self):
        ledger, run, url = self.ledger()
        ledger.observe(run, url, b"original")
        self.assertEqual(ledger.observe(run, url, b"changed"), "CONTENT_CHANGED")
        hashes = {r[0] for r in ledger.db.execute("SELECT sha256 FROM document_versions")}
        self.assertEqual(hashes, {hashlib.sha256(b).hexdigest() for b in (b"original", b"changed")})

    def test_URL_MOVED_keeps_verified_alias_and_history(self):
        ledger, run, url = self.ledger()
        ledger.observe(run, url, b"original")
        identity = ledger.db.execute("SELECT canonical_id FROM document_manifest").fetchone()[0]
        ledger.observe(run, url + "-moved", b"original", canonical_id_hint=identity)
        self.assertEqual(ledger.db.execute("SELECT count(*) FROM document_urls").fetchone()[0], 2)
        self.assertEqual(ledger.db.execute("SELECT count(*) FROM sync_changes WHERE change_type='URL_MOVED'").fetchone()[0], 1)

    def test_VERSION_REPLACED_requires_explicit_version_and_does_not_infer_status(self):
        ledger, run, url = self.ledger()
        ledger.observe(run, url, b"v1", verified_version="fixture version 1")
        result = ledger.observe(run, url, b"v2", verified_version="fixture version 2")
        self.assertEqual(result, "NEW_VERSION")
        self.assertEqual(ledger.db.execute("SELECT count(*) FROM document_versions").fetchone()[0], 2)
        self.assertEqual(ledger.db.execute("SELECT status FROM document_manifest").fetchone()[0], "unknown")

    def test_FETCH_FAILED_keeps_checkpoint_and_last_accepted_snapshot(self):
        fixture = self.worker()
        run_once(fixture.root, fixture.now, clients={"tkdl": fixture.client()})
        before = fixture.ledger.db.execute("SELECT snapshot_id FROM accepted_documents").fetchone()[0]
        result = run_once(fixture.root, fixture.now + timedelta(days=1), clients={"tkdl": FakeClient([ValueError("http_403")])})
        self.assertEqual(result["results"][0]["status"], "partial")
        self.assertEqual(fixture.ledger.db.execute("SELECT snapshot_id FROM accepted_documents").fetchone()[0], before)
        self.assertTrue((fixture.root / "data/manifests/checkpoints/tkdl.json").exists())
        fixture.assert_pointers()

    def test_PARSE_FAILED_quarantines_observed_bytes_without_admitting_them(self):
        fixture = self.worker()
        run_once(fixture.root, fixture.now, clients={"tkdl": fixture.client()})
        before = fixture.ledger.db.execute("SELECT snapshot_id FROM accepted_documents").fetchone()[0]
        changed = BODY + b"<!-- new bytes -->"
        with patch("ipindia_corpus.sync_runner.extract", side_effect=ValueError("fixture parse failure")):
            result = run_once(fixture.root, fixture.now + timedelta(days=1), clients={"tkdl": fixture.client(body=changed)})
        self.assertEqual(result["results"][0]["status"], "partial")
        self.assertIn("parse_failed", result["results"][0]["failures"][0]["reason"])
        self.assertEqual(fixture.ledger.db.execute("SELECT snapshot_id FROM accepted_documents").fetchone()[0], before)
        quarantined = list((fixture.root / "data/raw/quarantine/tkdl").glob("*.html"))
        self.assertEqual(quarantined[0].read_bytes(), changed)
        fixture.assert_pointers()

    def test_PARTIAL_CAPTURE_cannot_enter_later_source_candidate(self):
        fixture = self.worker()
        # A completed new page plus a failed second page is not an accepted
        # source snapshot. Later publication must read accepted rows only.
        second = URL.replace("Abouttkdl.asp", "faq.asp")
        import json
        catalog = fixture.root / "config/curated_sources.json"
        config = json.loads(catalog.read_text())
        config["tkdl"]["seeds"].append(second)
        catalog.write_text(json.dumps(config))
        responses = FakeClient([Snapshot(URL, BODY, "text/html", None, None, 200), ValueError("http_429")])
        result = run_once(fixture.root, fixture.now, clients={"tkdl": responses})
        self.assertEqual(result["results"][0]["status"], "partial")
        self.assertFalse(result["pending_candidates"])
        self.assertEqual(fixture.ledger.db.execute("SELECT count(*) FROM accepted_documents").fetchone()[0], 0)
        self.assertEqual(fixture.ledger.db.execute("SELECT count(*) FROM document_versions").fetchone()[0], 1)
        fixture.assert_pointers()


if __name__ == "__main__":
    unittest.main()
