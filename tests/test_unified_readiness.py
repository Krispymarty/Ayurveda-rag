"""Pure offline fixtures for three-source readiness; no real labels are created."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ipindia_corpus.generations import publish
from ipindia_corpus.indexer import generation
from ipindia_corpus.sync import Ledger
from ipindia_corpus.unified_readiness import assess
from test_three_sources import BODY, URL


def digest(data):
    return hashlib.sha256(data).hexdigest()


class UnifiedReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.temp.name)
        (self.root / "reports").mkdir()
        (self.root / "tests").mkdir()
        ledger = Ledger(self.root / "data/manifests/manifest.sqlite", self.root / "data/raw")
        try:
            for source, url in (("ipindia", "https://ipindia.gov.in/acts/example"),
                                ("indiacode", "https://indiacode.gov.in/act/example"), ("tkdl", URL)):
                run = ledger.start(source)
                ledger.observe(run, url, BODY.replace(b"About TKDL", source.encode()), content_type="text/html")
                ledger.finish(run)
        finally:
            ledger.close()
        candidate = publish(self.root, update_pointer=False)
        self.active = candidate["generation_id"]
        _, self.documents, self.chunks = generation(self.root, generation_id=self.active)
        entries = []
        for doc in self.documents:
            source = doc["source"]
            if source == "ipindia":
                continue
            path = f"staged/{source}.md"
            (self.root / path).parent.mkdir(exist_ok=True)
            transformed = b"Synthetic fidelity fixture; not a real certification."
            (self.root / path).write_bytes(transformed)
            proof_path = f"reports/{source}_fidelity.json"
            proof = json.dumps({"pass": True, "original_sha256": doc["sha256"],
                                "transformed_sha256": digest(transformed)}).encode()
            (self.root / proof_path).write_bytes(proof)
            entries.append({"path": path, "sha256": digest(transformed), "bytes": len(transformed),
                            "source": source, "source_url": doc["source_url"],
                            "certified": True, "search_admission": "accepted",
                            "fidelity_report_path": proof_path, "fidelity_report_sha256": digest(proof),
                            "original_receipt": {"verified": True, "capture_timestamp": doc["retrieved_at"],
                                                 "original_path": doc["raw_path"], "original_sha256": doc["sha256"]}})
        inventory = [{key: entry[key] for key in ("path", "sha256", "bytes")} for entry in entries]
        self.audit = {"inventory_sha256": digest(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()),
                      "documents": entries, "metadata_files": [], "user_supplied_pdf_originals": [],
                      "sources": {source: {"staged_documents": 1, "certified_documents": 1,
                                           "search_admission": "accepted"} for source in ("indiacode", "tkdl")}}
        self.write_audit()
        self.labels = {"frozen": True, "frozen_generation_id": self.active,
                       "frozen_by": "synthetic fixture", "frozen_at": "2026-10-15T00:00:00Z", "queries": []}
        for doc in self.documents:
            for i in range(10):
                self.labels["queries"].append({"id": f"{doc['source']}-{i}", "source": doc["source"],
                    "query": f"Synthetic {doc['source']} query {i}", "reviewed": True,
                    "review_note": "Synthetic fixture physical passage", "targets": [{"source": doc["source"],
                     "urls": [doc["source_url"]], "contains": "public institutional background",
                     "raw_sha256": [doc["sha256"]]}]})
        self.write_labels()

    def tearDown(self):
        self.temp.cleanup()

    def write_audit(self):
        (self.root / "reports/corpus_certification.json").write_text(json.dumps(self.audit))

    def write_labels(self):
        path = self.root / "tests/retrieval_queries.json"
        path.write_text(json.dumps(self.labels))
        report = {"pass": True, "partial": False, "generation_id": self.active,
                  "label_sha256": digest(path.read_bytes()), "recall_at_5": .95, "p95_ms": 500,
                  "observations": 90}
        (self.root / "reports/retrieval_evaluation.json").write_text(json.dumps(report))

    def assess(self, documents=None):
        return assess(self.root, self.active, self.documents if documents is None else documents)

    def test_genuine_bound_fixture_evidence_can_pass_but_answers_public_stay_blocked(self):
        report = self.assess()
        self.assertTrue(report["three_source_ready"])
        self.assertTrue(report["unified_benchmark"]["passage_bindings_verified"])
        self.assertFalse(report["stages"]["answers"]["ready"])
        self.assertFalse(report["stages"]["public"]["ready"])
        self.assertEqual(report["blocked_source_count"], 0)
        self.assertIn("reports/corpus_certification.json", report["artifact_sha256"])

    def test_candidate_generation_never_supplies_missing_active_source_counts(self):
        indexed = [doc for doc in self.documents if doc["source"] == "ipindia"]
        report = self.assess(indexed)
        self.assertFalse(report["three_source_ready"])
        self.assertEqual(report["sources"]["tkdl"]["indexed_documents"], 0)
        self.assertEqual(report["sources"]["indiacode"]["indexed_documents"], 0)
        self.assertIn("no_active_searchable_capture", report["sources"]["tkdl"]["blockers"])

    def test_staged_or_syntax_only_documents_cannot_be_certified(self):
        for entry in self.audit["documents"]:
            entry["certified"] = False
        for summary in self.audit["sources"].values():
            summary.update(certified_documents=0, search_admission="blocked")
        self.write_audit()
        report = self.assess()
        self.assertEqual(report["blocked_source_count"], 2)
        self.assertEqual(report["sources"]["tkdl"]["certified_active_documents"], 0)
        self.assertIn("originals_and_fidelity_not_certified", report["sources"]["tkdl"]["blockers"])

    def test_inventory_drift_and_receipt_fidelity_tampering_fail_closed(self):
        staged = self.root / self.audit["documents"][0]["path"]
        original = staged.read_bytes()
        staged.write_bytes(b"changed")
        report = self.assess()
        self.assertFalse(report["certification_inventory_valid"])
        self.assertFalse(report["three_source_ready"])
        staged.write_bytes(original)
        proof = self.root / self.audit["documents"][0]["fidelity_report_path"]
        proof.write_text(json.dumps({"pass": True}))
        self.assertFalse(self.assess()["three_source_ready"])

    def test_unreviewed_scaffold_is_not_a_frozen_unified_benchmark(self):
        self.labels["frozen"] = False
        for item in self.labels["queries"]:
            if item["source"] != "ipindia":
                item["reviewed"] = False
        self.write_labels()
        before = (self.root / "tests/retrieval_queries.json").read_bytes()
        report = self.assess()
        self.assertEqual(report["unified_benchmark"]["status"], "blocked_not_frozen")
        self.assertEqual(report["unified_benchmark"]["reviewed_by_source"]["tkdl"], 0)
        self.assertFalse(report["three_source_ready"])
        self.assertEqual(before, (self.root / "tests/retrieval_queries.json").read_bytes())

    def test_stale_physical_passage_or_generation_binding_fails(self):
        self.labels["queries"][0]["targets"][0]["contains"] = "invented nonexisting passage"
        self.write_labels()
        self.assertFalse(self.assess()["unified_benchmark"]["ready"])
        self.labels["frozen_generation_id"] = "20200101T000000000000Z"
        self.write_labels()
        self.assertFalse(self.assess()["three_source_ready"])

    def test_missing_database_does_not_use_filesystem_pointer_or_leak_exception(self):
        pointer = self.root / "data/generations/CURRENT"
        pointer.write_text(self.active)
        with patch("ipindia_corpus.unified_readiness._database_active", side_effect=RuntimeError("password=private-secret")):
            report = assess(self.root)
        self.assertFalse(report["active_index"]["available"])
        self.assertFalse(report["three_source_ready"])
        self.assertNotIn("private-secret", json.dumps(report))
        self.assertEqual(pointer.read_text(), self.active)

    def test_untrusted_audit_paths_and_partial_evaluation_do_not_pass(self):
        self.audit["documents"][0]["path"] = "../escape.md"
        self.write_audit()
        self.assertFalse(self.assess()["three_source_ready"])
        evaluation = self.root / "reports/retrieval_evaluation.json"
        data = json.loads(evaluation.read_text())
        data["partial"] = True
        evaluation.write_text(json.dumps(data))
        self.assertFalse(self.assess()["evaluation_ready"])


if __name__ == "__main__":
    unittest.main()
