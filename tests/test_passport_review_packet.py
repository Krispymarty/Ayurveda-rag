"""Offline review preparation must not create human relevance decisions."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import hashlib
from unittest.mock import patch

from tools.passport_review_packet import export_packet, export_priority_packet, main


def fixture():
    evidence = {"evidence_id": "E001", "document_id": "doc", "chunk_id": "chunk",
                "raw_sha256": "a" * 64, "official_url": "https://ipindia.gov.in/fixture",
                "exact_quote": "Archived wording.", "section": "3", "document_status": "unknown"}
    return {"runs": [{"run_id": "chosen", "synthetic": True, "status": "completed",
                     "scenario_file_sha256": "b" * 64,
                     "case_results": [{"case_id": "case", "intake": {"description": "User context"},
                        "response": {"generation_id": "generation", "model": {"name": "pinned", "planner_version": "v2"},
                            "issues": [{"issue_id": "one", "user_context": "Context one", "evidence_ids": ["E001", "E001"]},
                                       {"issue_id": "two", "user_context": "Context two", "evidence_ids": ["E001"]},
                                       {"issue_id": "missing", "evidence_ids": []}],
                            "verified_evidence": [evidence]},
                        "validation": {"evidence_checks": [{"evidence_id": "E001", "integrity_verified": True}]}}]},
                    {"run_id": "failure", "synthetic": True, "status": "interrupted", "case_results": []}]}


class PacketTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "observations.json"
        self.source.write_text(json.dumps(fixture()), encoding="utf-8")
        self.original = self.source.read_bytes()

    def rows(self, folder):
        return [json.loads(line) for line in (folder / "review_queue.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_preserves_source_and_pending_context_bound_rows(self):
        out = self.root / "packet"
        manifest = export_packet(self.source, "chosen", out)
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(manifest["historical_runs_retained"], 2)
        rows = self.rows(out)
        self.assertEqual(len(rows), 3)
        self.assertEqual([r["issue"]["user_context"] for r in rows[:2]], ["Context one", "Context two"])
        self.assertEqual(rows[0]["evidence"]["exact_quote"], "Archived wording.")
        self.assertEqual(rows[2]["kind"], "missing_evidence")
        self.assertEqual(rows[0]["recorded_fidelity_observations"][0]["integrity_verified"], True)
        for row in rows:
            self.assertEqual(row["review_status"], "pending")
            for name in ("decision", "reviewer", "rationale"):
                self.assertIsNone(row[name])
        self.assertEqual(manifest["human_reviews_completed"], 0)
        self.assertEqual(json.loads((out / "selected_run.json").read_text())["run_id"], "chosen")

    def test_packet_is_deterministic_and_refuses_existing_output(self):
        first, second = self.root / "one", self.root / "two"
        self.assertEqual(export_packet(self.source, "chosen", first), export_packet(self.source, "chosen", second))
        self.assertEqual((first / "review_queue.jsonl").read_bytes(), (second / "review_queue.jsonl").read_bytes())
        before = (first / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            export_packet(self.source, "chosen", first)
        self.assertEqual((first / "manifest.json").read_bytes(), before)

    def test_unknown_or_ambiguous_run_fails_before_writing(self):
        for report, identifier in [(fixture(), "absent"), ({"runs": [fixture()["runs"][0]] * 2}, "chosen")]:
            self.source.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                export_packet(self.source, identifier, self.root / "bad")
            self.assertFalse((self.root / "bad").exists())

    def test_rejects_non_synthetic_inputs_and_unknown_evidence_references(self):
        original = fixture()
        changes = [copy.deepcopy(original), copy.deepcopy(original)]
        changes[0]["runs"][0]["synthetic"] = False
        changes[1]["runs"][0]["case_results"][0]["response"]["issues"][0]["evidence_ids"] = ["forged"]
        for report in changes:
            self.source.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                export_packet(self.source, "chosen", self.root / "bad")
            self.assertFalse((self.root / "bad").exists())

    def test_failed_observation_stays_missing_not_removed_or_retried(self):
        report = fixture()
        report["runs"][0]["case_results"] = [{"case_id": "failed", "intake": {"description": "Context"},
                                                "outcome": "transport_error", "error": {"kind": "timeout"}}]
        self.source.write_text(json.dumps(report))
        export_packet(self.source, "chosen", self.root / "packet")
        row = self.rows(self.root / "packet")[0]
        self.assertEqual(row["kind"], "case_without_evidence")
        self.assertEqual(row["recorded_outcome"], "transport_error")
        self.assertIsNone(row["evidence"])

    def test_cli_requires_explicit_run_and_output(self):
        with self.assertRaises(SystemExit) as caught:
            main([])
        self.assertEqual(caught.exception.code, 2)


class PriorityPacketTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.gid = "20261029T000000Z"
        raw = b"original official capture"
        digest = hashlib.sha256(raw).hexdigest()
        raw_path = self.root / "data/raw/ipindia/original.bin"
        raw_path.parent.mkdir(parents=True)
        raw_path.write_bytes(raw)
        doc = {"document_id": "doc", "source": "ipindia", "source_url": "https://ipindia.gov.in/fixture",
               "sha256": digest, "raw_path": "data/raw/ipindia/original.bin"}
        self.chunks = []
        for identifier, text in [("chunk", "पहला. Archived wording. हिन्दी संदर्भ"), ("next", "Adjacent archived provision.")]:
            self.chunks.append({"chunk_id": identifier, "document_id": "doc", "source": "ipindia", "document_title": "Archived Act",
                "source_url": doc["source_url"], "raw_sha256": digest, "section": "3", "rule": "", "page_start": None,
                "page_end": None, "text": text, "content_hash": hashlib.sha256(text.encode()).hexdigest()})
        folder = self.root / "data/generations" / self.gid
        folder.mkdir(parents=True)
        manifest = {"generation_id": self.gid, "documents": 1, "chunks": 2}
        for name, values in [("documents", [doc]), ("chunks", self.chunks)]:
            data = b"".join((json.dumps(v, ensure_ascii=False) + "\n").encode() for v in values)
            (folder / (name + ".jsonl")).write_bytes(data)
            manifest[name + "_sha256"] = hashlib.sha256(data).hexdigest()
        (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        report = fixture()
        case = report["runs"][0]["case_results"][0]
        case["response"]["generation_id"] = self.gid
        case["intake"]["description"] = "हिन्दी research context"
        item = case["response"]["verified_evidence"][0]
        item.update(raw_sha256=digest, document_title="Archived Act", rule="", page_start=None, page_end=None)
        self.source = self.root / "report.json"
        self.source.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        self.base = self.root / "original_packet"
        export_packet(self.source, "chosen", self.base)
        self.old = {p.name: p.read_bytes() for p in self.base.iterdir()}

    def export(self, **changes):
        args = {"report_path": self.source, "run_id": "chosen", "output_dir": self.root / "priority",
                "case_ids": ["case"], "source_packet": self.base, "root": self.root}
        return export_priority_packet(**(args | changes))

    def test_preserves_review_ids_decisions_unicode_and_separates_archived_context(self):
        manifest = self.export()
        source_rows = [json.loads(line) for line in self.old["review_queue.jsonl"].decode().splitlines()]
        out = self.root / "priority"
        rows = [json.loads(line) for line in (out / "review_queue.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["review_id"] for r in rows], [r["review_id"] for r in source_rows])
        self.assertEqual(manifest["human_reviews_completed"], 0)
        self.assertTrue(all(r["review_status"] == "pending" and r["decision"] is None for r in rows))
        case = json.loads((out / "cases/case.json").read_text(encoding="utf-8"))
        self.assertIn("हिन्दी", case["intake"]["description"])
        self.assertFalse(case["historical_model_presented_context"]["available"])
        self.assertFalse(case["retrieval_candidate_pool"]["available"])
        context = case["archive_contexts"][0]
        self.assertEqual(context["archived_chunk"]["text"], self.chunks[0]["text"])
        self.assertEqual(context["adjacent_chunks"][0]["chunk_id"], "next")
        self.assertIn("not_proof_of_historical_model_input", context["basis"])
        self.assertEqual({p.name: p.read_bytes() for p in self.base.iterdir()}, self.old)

    def test_explicit_order_unknown_duplicate_case_and_existing_output(self):
        for identifiers in ([], ["absent"], ["case", "case"], ["../case"]):
            with self.assertRaises(ValueError):
                self.export(case_ids=identifiers)
            self.assertFalse((self.root / "priority").exists())
        self.export()
        with self.assertRaises(FileExistsError):
            self.export()

    def test_tampered_source_packet_report_and_citation_are_rejected(self):
        (self.base / "review_queue.jsonl").write_bytes(self.old["review_queue.jsonl"] + b" ")
        with self.assertRaisesRegex(ValueError, "hash"):
            self.export()
        (self.base / "review_queue.jsonl").write_bytes(self.old["review_queue.jsonl"])
        self.source.write_bytes(self.source.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "report"):
            self.export()
        self.assertFalse((self.root / "priority").exists())

    def test_archive_raw_mutation_is_rejected_before_output(self):
        (self.root / "data/raw/ipindia/original.bin").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.export()
        self.assertFalse((self.root / "priority").exists())

    def test_source_mutation_during_packet_preparation_is_rejected(self):
        from tools.passport_demo import CitationVerifier
        original = CitationVerifier.evidence

        def mutate(verifier, item, documents, chunks):
            errors = original(verifier, item, documents, chunks)
            (self.root / "data/raw/ipindia/original.bin").write_bytes(b"changed during preparation")
            return errors

        with patch.object(CitationVerifier, "evidence", mutate):
            with self.assertRaisesRegex(ValueError, "source changed"):
                self.export()
        self.assertFalse((self.root / "priority").exists())

    def test_invalid_citation_cannot_be_enriched_from_a_coherent_historical_packet(self):
        report = json.loads(self.source.read_text(encoding="utf-8"))
        report["runs"][0]["case_results"][0]["response"]["verified_evidence"][0]["exact_quote"] = "Fabricated quotation."
        self.source.write_text(json.dumps(report), encoding="utf-8")
        alternate = self.root / "alternate_source"
        export_packet(self.source, "chosen", alternate)
        with self.assertRaisesRegex(ValueError, "quote_not_exact"):
            self.export(source_packet=alternate)
        self.assertFalse((self.root / "priority").exists())

    def test_explicit_case_order_and_cli_pairing(self):
        report = json.loads(self.source.read_text(encoding="utf-8"))
        second = copy.deepcopy(report["runs"][0]["case_results"][0])
        second["case_id"] = "second"
        report["runs"][0]["case_results"].append(second)
        self.source.write_text(json.dumps(report), encoding="utf-8")
        alternate = self.root / "alternate_source"
        export_packet(self.source, "chosen", alternate)
        manifest = self.export(case_ids=["second", "case"], source_packet=alternate)
        self.assertEqual(manifest["ordered_case_ids"], ["second", "case"])
        rows = [json.loads(line) for line in (self.root / "priority/review_queue.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["case_id"] for r in rows], ["second"] * 3 + ["case"] * 3)
        with self.assertRaises(SystemExit) as caught:
            main(["--run-id", "chosen", "--output-dir", str(self.root / "cli"), "--case-ids", "case"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
