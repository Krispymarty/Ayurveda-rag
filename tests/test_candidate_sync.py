"""Offline-only policy-gated candidate scheduling checks."""

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from ipindia_corpus.candidate_sync import run_once, status
from ipindia_corpus.connectors import Snapshot
from ipindia_corpus.locking import file_lock
from ipindia_corpus.sync import Ledger
from ipindia_corpus.sync_runner import process_source
from support_review import approve_fixture
from test_three_sources import BODY, URL, FakeClient


class CandidateSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.temp.name)
        (self.root / "config").mkdir()
        self.now = datetime.now(timezone.utc)
        self.config = {"mode": "candidate_only", "enabled": True, "poll_seconds": 300,
                       "check_interval_seconds": 86400, "discovery_interval_seconds": 604800,
                       "retry_interval_seconds": 3600,
                       "sources": {"tkdl": {"enabled": True}}}
        self.write_config()
        (self.root / "config/curated_sources.json").write_text(json.dumps({"tkdl": {
            "seeds": [URL], "max_discovered": 5, "evidence_role": "institutional_background"}}))
        (self.root / "config/source_policies.json").write_text(json.dumps({source: {
            "approved": False} for source in ("ipindia", "indiacode", "tkdl")}))
        pointers = self.root / "data/generations"
        pointers.mkdir(parents=True)
        (pointers / "CURRENT").write_text("20200102T000000000000Z")
        (pointers / "ACTIVE").write_text("20200101T000000000000Z")
        baseline = pointers / "20200101T000000000000Z"
        baseline.mkdir()
        (baseline / "manifest.json").write_text(json.dumps({"sources": {"tkdl": 0}}))
        self.ledger = Ledger(self.root / "data/manifests/manifest.sqlite", self.root / "data/raw")

    def write_config(self):
        (self.root / "config/synchronization.json").write_text(json.dumps(self.config))

    def tearDown(self):
        self.ledger.close()
        self.temp.cleanup()

    def approved(self):
        approve_fixture(self.root, "tkdl")

    def client(self, body=BODY, status_code=200):
        return FakeClient([Snapshot(URL, body if status_code != 304 else None,
                                    "text/html", '"test"', None, status_code)])

    def assert_pointers(self):
        self.assertEqual((self.root / "data/generations/CURRENT").read_text(), "20200102T000000000000Z")
        self.assertEqual((self.root / "data/generations/ACTIVE").read_text(), "20200101T000000000000Z")

    def test_blocked_scheduler_does_not_construct_network_client(self):
        with patch("ipindia_corpus.sync_runner.SourceClient", side_effect=AssertionError("network forbidden")):
            report = run_once(self.root, self.now)
        self.assertEqual(report["state"], "access_blocked")
        self.assertEqual(report["eligible_sources"], 0)
        self.assertEqual(report["results"], [])
        self.assertEqual(report["pending_candidates"], [])
        self.assert_pointers()

    def test_successful_capture_queues_hash_bound_candidate_without_index(self):
        self.approved()
        with patch("ipindia_corpus.indexer.index", side_effect=AssertionError("index forbidden")):
            report = run_once(self.root, self.now, clients={"tkdl": self.client()})
        self.assertEqual(report["results"][0]["status"], "success")
        receipt = report["pending_candidates"][0]
        self.assertEqual(receipt["state"], "pending_review")
        manifest = self.root / "data/generations" / receipt["generation_id"] / "manifest.json"
        self.assertEqual(hashlib.sha256(manifest.read_bytes()).hexdigest(), receipt["manifest_sha256"])
        self.assertEqual(json.loads(manifest.read_text())["passage_parser"], "pdf-pages-v4")
        self.assert_pointers()
        publication = self.ledger.db.execute("SELECT last_successful_publication FROM source_registry WHERE source_id='tkdl'").fetchone()[0]
        self.assertIsNone(publication)

    def test_incremental_history_and_independent_due_times(self):
        self.approved()
        first = run_once(self.root, self.now, clients={"tkdl": self.client()})
        discovery = next(source for source in first["sources"] if source["source"] == "tkdl")["next_discovery"]
        second = run_once(self.root, self.now + timedelta(days=1, seconds=1),
                          clients={"tkdl": self.client(status_code=304)})
        self.assertEqual(len(second["pending_candidates"]), 2)
        source = next(source for source in second["sources"] if source["source"] == "tkdl")
        self.assertEqual(source["next_discovery"], discovery)
        third = run_once(self.root, self.now + timedelta(days=2, seconds=2),
                         clients={"tkdl": self.client(body=BODY.replace(b"Traditional Knowledge", b"Updated Traditional Knowledge"))})
        self.assertEqual(len(third["pending_candidates"]), 3)
        self.assertEqual(self.ledger.db.execute("SELECT count(*) FROM document_versions").fetchone()[0], 2)
        self.assert_pointers()

    def test_publication_failure_retains_capture_and_records_retry(self):
        self.approved()
        with patch("ipindia_corpus.generations.publish", side_effect=ValueError("incomplete candidate")):
            report = run_once(self.root, self.now, clients={"tkdl": self.client()})
        self.assertEqual(report["results"][0]["status"], "pipeline_failed")
        self.assertEqual(report["pending_candidates"], [])
        self.assertEqual(self.ledger.db.execute("SELECT count(*) FROM accepted_documents").fetchone()[0], 1)
        event = self.ledger.db.execute("SELECT status FROM pipeline_events WHERE stage='candidate_publication'").fetchone()
        self.assertEqual(event[0], "failed")
        self.assert_pointers()

    def test_failed_fetch_retains_checkpoint_and_accepted_snapshot(self):
        self.approved()
        run_once(self.root, self.now, clients={"tkdl": self.client()})
        report = run_once(self.root, self.now + timedelta(days=1, seconds=1),
                          clients={"tkdl": FakeClient([ValueError("access_restricted:403")])})
        self.assertEqual(report["results"][0]["status"], "partial")
        self.assertEqual(len(report["pending_candidates"]), 1)
        self.assertTrue((self.root / "data/manifests/checkpoints/tkdl.json").exists())
        self.assert_pointers()

    def test_disabled_flags_overlap_and_invalid_mode(self):
        self.approved()
        self.config["sources"]["tkdl"]["enabled"] = False
        self.write_config()
        self.assertEqual(run_once(self.root)["results"], [])
        with file_lock(self.root / "data/locks/candidate-worker.lock"):
            with self.assertRaisesRegex(ValueError, "worker_already_running"):
                run_once(self.root)
        self.config["mode"] = "production"
        self.write_config()
        with self.assertRaisesRegex(ValueError, "candidate_only"):
            status(self.root)

    def test_tampered_review_and_host_rejection_do_not_queue_candidate(self):
        review = approve_fixture(self.root, "tkdl")
        evidence = self.root / review["evidence"][0]["raw_path"]
        original = evidence.read_bytes()
        evidence.write_bytes(b"tampered")
        self.assertEqual(status(self.root)["eligible_sources"], 0)
        evidence.write_bytes(original)
        client = FakeClient([Snapshot("https://evil.test/page", BODY, "text/html", None, None, 200)])
        report = run_once(self.root, clients={"tkdl": client})
        self.assertNotEqual(report["results"][0]["status"], "success")
        self.assertEqual(report["pending_candidates"], [])
        self.assert_pointers()

    def test_direct_process_source_is_also_candidate_only(self):
        self.approved()
        with patch("ipindia_corpus.indexer.index", side_effect=AssertionError("index forbidden")):
            result = process_source("tkdl", root=self.root, client=self.client())
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["candidate"]["mode"], "candidate_only")
        self.assert_pointers()


if __name__ == "__main__":
    unittest.main()
