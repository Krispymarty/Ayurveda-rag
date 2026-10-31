"""Synthetic human-workflow fixtures only; never import real reviewer records."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.milestone3.review import import_decisions, status
from tools.passport_review_packet import encoded, export_packet


CASE_IDS = ["traditional_ashwagandha_extract", "new_formulation", "novel_process", "abs_request", "hindi_intake", "unclear_description"]
RUN = "dc7ecb60-9a82-4b1d-a4b1-fb6133136675"
PACKET = Path("data/next_milestone/review/six-case-priority")
SOURCE = Path("reports/product_mvp/relevance_review/run-dc7ecb60")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def snapshot(root):
    return {p.relative_to(root).as_posix(): digest(p.read_bytes()) for p in root.rglob("*") if p.is_file()}


def make_fixture(root):
    cases = []
    for case_id, count in zip(CASE_IDS + [f"extra_{i}" for i in range(6)], [6, 4, 5, 4, 4, 6, 4, 3, 3, 3, 3, 3]):
        evidence = [{"evidence_id": f"E{i:03}", "document_id": "fixture-doc", "chunk_id": f"fixture-chunk-{i}",
                     "exact_quote": f"हिन्दी fixture source {i}", "raw_sha256": "a" * 64,
                     "official_url": "https://ipindia.gov.in/fixture", "document_status": "unknown"} for i in range(1, count + 1)]
        issues = [{"issue_id": "research", "why_it_may_matter": "Research action, never a legal proposition.", "evidence_ids": [e["evidence_id"] for e in evidence]}]
        if case_id in {"new_formulation", "hindi_intake"}:
            issues.append({"issue_id": "other_topic", "evidence_ids": ["E001"]})
        cases.append({"case_id": case_id, "intake": {"description": "Synthetic non-confidential research intake"},
            "response": {"generation_id": "20261014T204940358650Z", "model": {"name": "fixture-only"}, "issues": issues,
                "verified_evidence": evidence, "unresolved_questions": ["No historical legal propositions retained."]}})
    run = {"run_id": RUN, "synthetic": True, "status": "completed", "scenario_file_sha256": "b" * 64, "case_results": cases}
    report = root / "reports/product_mvp/verified_demo_cases.json"
    report.parent.mkdir(parents=True)
    report.write_bytes(encoded({"runs": [run]}))
    old = export_packet(report, RUN, root / SOURCE)
    old_rows = [json.loads(line) for line in (root / SOURCE / "review_queue.jsonl").read_text(encoding="utf-8").splitlines()]
    selected_rows = [r for case_id in CASE_IDS for r in old_rows if r["case_id"] == case_id]
    folder = root / PACKET
    folder.mkdir(parents=True)
    files = {"review_queue.jsonl": b"".join((json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n").encode() for r in selected_rows),
             "instructions.md": b"Synthetic fixture instructions only.\n"}
    for case in cases[:6]:
        response = case["response"]
        contexts = []
        for item in response["verified_evidence"]:
            text = "Before. " + item["exact_quote"] + " After."
            chunk = {"chunk_id": item["chunk_id"], "document_id": item["document_id"], "text": text,
                     "raw_sha256": item["raw_sha256"], "content_hash": digest(text.encode()), "source_url": item["official_url"]}
            contexts.append({"evidence_id": item["evidence_id"], "archived_chunk": chunk, "adjacent_chunks": [],
                             "document": {"document_id": item["document_id"], "sha256": item["raw_sha256"]}})
        payload = {"case_id": case["case_id"], "run_id": RUN, "intake": case["intake"], "issues": response["issues"],
                   "generation_id": response["generation_id"],
                   "selected_quotations_and_citations": response["verified_evidence"], "archive_contexts": contexts,
                   "unanswered_questions": response["unresolved_questions"],
                   "review_ids": [r["review_id"] for r in selected_rows if r["case_id"] == case["case_id"]],
                   "historical_model_presented_context": {"available": False},
                   "retrieval_candidate_pool": {"available": False, "passages": None}}
        files[f"cases/{case['case_id']}.json"] = encoded(payload)
    manifest = {"schema_version": "passport-priority-review-v1", "selected_run_id": RUN, "ordered_case_ids": CASE_IDS,
        "case_count": 6, "review_unit_count": 31, "preserved_review_ids": [r["review_id"] for r in selected_rows],
        "human_reviews_completed": 0, "source_report_sha256": digest(report.read_bytes()),
        "source_packet_manifest_sha256": digest((root / SOURCE / "manifest.json").read_bytes()),
        "source_packet_payload_sha256": old["files"], "scenario_file_sha256": run["scenario_file_sha256"],
        "files": {name: digest(raw) for name, raw in files.items()}}
    manifest["packet_id"] = digest(encoded(manifest))
    for name, raw in files.items():
        p = folder / name
        p.parent.mkdir(exist_ok=True, parents=True)
        p.write_bytes(raw)
    (folder / "manifest.json").write_bytes(encoded(manifest))
    return manifest, selected_rows


class PriorityReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest, self.rows = make_fixture(self.root)
        self.packet = self.root / PACKET
        self.batch = self.root / "decisions.jsonl"
        self.store = self.root / "data/next_stage/reviews" / self.manifest["packet_id"]

    def decision(self, index=0, **changes):
        row = self.rows[index]
        case_hash = self.manifest["files"][f"cases/{row['case_id']}.json"]
        return {"annotation_version": "synthetic-fixture-v1", "question_id": row["case_id"], "citation_id": row["review_id"],
            "source_hash": row["evidence"]["raw_sha256"], "packet_sha256": case_hash,
            "priority_packet_id": self.manifest["packet_id"], "priority_manifest_sha256": digest((self.packet / "manifest.json").read_bytes()),
            "reviewer": "synthetic-fixture-human", "review_timestamp": "2026-01-01T00:00:00Z", "dimension": "passage_relevance",
            "judgment": "REQUIRED", "supporting_quote": row["evidence"]["exact_quote"], "rationale": "Synthetic workflow test, not a real human assessment.",
            "adjudication_status": "submitted", "human_attestation": True, **changes}

    def import_batch(self, items):
        self.batch.write_bytes(b"".join((json.dumps(item, ensure_ascii=False) + "\n").encode() for item in items))
        return import_decisions(self.batch, self.root, packet=self.packet)

    def test_status_read_only_preserves_31_ids_and_reports_missing_propositions(self):
        before = snapshot(self.root)
        result = status(self.root, packet=self.packet)
        self.assertEqual(snapshot(self.root), before)
        self.assertEqual(result["review_unit_count"], 31)
        self.assertEqual(result["distinct_case_citations"], 29)
        self.assertEqual(result["human_submitted_records"], 0)
        self.assertEqual(result["dimension_availability"]["proposition_support"], "not_assessable_missing_proposition")
        self.assertEqual(result["review_ids"], self.manifest["preserved_review_ids"])
        self.assertFalse(result["certification_allowed"])

    def test_unicode_import_is_idempotent_and_never_completes_missing_dimension(self):
        before = snapshot(self.packet)
        first = self.import_batch([self.decision()])
        second = self.import_batch([self.decision()])
        self.assertEqual(first["human_submitted_records"], 1)
        self.assertEqual(second["human_submitted_records"], 1)
        self.assertEqual(second["passage_relevance_completed_units"], 1)
        self.assertEqual(second["fully_reviewed_units_both_dimensions"], 0)
        self.assertEqual(snapshot(self.packet), before)
        self.assertEqual(len(list((self.store / "judgments").glob("*.json"))), 1)
        self.assertFalse((self.root / "data/milestone3/judgments").exists())

    def test_invalid_batch_writes_no_records_and_rejects_binding_attestation(self):
        for change in ({"human_attestation": False}, {"reviewer": ""}, {"source_hash": "wrong"},
                       {"priority_packet_id": "wrong"}, {"priority_manifest_sha256": "wrong"},
                       {"packet_sha256": "wrong"}, {"supporting_quote": "not in own archived passage"},
                       {"citation_id": "unknown"}, {"review_timestamp": "2026-01-01"},
                       {"judgment": "INCORRECT_LEGAL_INTERPRETATION"}):
            with self.assertRaises(ValueError):
                self.import_batch([self.decision(), self.decision(**change)])
            self.assertFalse((self.store / "judgments").exists())

    def test_missing_proposition_allows_only_unresolved_human_submissions(self):
        with self.assertRaisesRegex(ValueError, "proposition"):
            self.import_batch([self.decision(dimension="proposition_support")])
        result = self.import_batch([self.decision(dimension="proposition_support", judgment="INSUFFICIENT_INFORMATION")])
        self.assertEqual(result["fully_reviewed_units_both_dimensions"], 0)
        self.assertEqual(result["passage_relevance_completed_units"], 0)
        self.assertEqual(result["human_submitted_records"], 1)

    def test_conflicts_append_and_require_independent_adjudicator(self):
        first = self.decision()
        second = self.decision(reviewer="other-synthetic-human", judgment="IRRELEVANT")
        result = self.import_batch([first, second])
        self.assertEqual(len(result["conflicts"]), 1)
        self.assertEqual(result["passage_relevance_completed_units"], 0)
        refs = [p.stem for p in (self.store / "judgments").glob("*.json")]
        with self.assertRaisesRegex(ValueError, "independent"):
            self.import_batch([self.decision(adjudication_status="adjudicated", resolves=refs)])
        resolved = self.import_batch([self.decision(reviewer="independent-synthetic-adjudicator", adjudication_status="adjudicated", resolves=refs)])
        self.assertEqual(resolved["human_submitted_records"], 3)
        self.assertEqual(resolved["passage_relevance_completed_units"], 1)
        self.assertFalse(resolved["conflicts"])

    def test_packet_or_original_source_mutation_rejected_before_import(self):
        case_file = self.packet / "cases/new_formulation.json"
        case_file.write_bytes(case_file.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "hash|mutat"):
            status(self.root, packet=self.packet)
        self.assertFalse(self.store.exists())

    def test_original_report_binding_is_required(self):
        report = self.root / "reports/product_mvp/verified_demo_cases.json"
        report.write_bytes(report.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "report"):
            status(self.root, packet=self.packet)

    def test_case_sha_is_bound_to_exact_bytes_not_reserialized_json(self):
        parsed = json.loads((self.packet / "cases/traditional_ashwagandha_extract.json").read_text(encoding="utf-8"))
        wrong = digest(json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
        self.assertNotEqual(wrong, self.decision()["packet_sha256"])
        with self.assertRaisesRegex(ValueError, "binding"):
            self.import_batch([self.decision(packet_sha256=wrong)])

    def test_support_quote_is_limited_to_the_reviewed_citation_context(self):
        unrelated = self.rows[1]["evidence"]["exact_quote"]
        with self.assertRaisesRegex(ValueError, "supporting_quote"):
            self.import_batch([self.decision(supporting_quote=unrelated)])

    def test_original_packet_hash_and_unknown_review_ids_are_checked(self):
        queue = self.root / SOURCE / "review_queue.jsonl"
        queue.write_bytes(queue.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "hash"):
            status(self.root, packet=self.packet)

    def test_cli_never_regenerates_priority_packet(self):
        from tools.milestone3.review import main
        before = snapshot(self.packet)
        with self.assertRaises(SystemExit) as caught:
            main(["export", "--packet", str(self.packet)])
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(snapshot(self.packet), before)

    def test_mutated_content_addressed_judgment_fails_closed(self):
        self.import_batch([self.decision()])
        path = next((self.store / "judgments").glob("*.json"))
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "record_mutated"):
            status(self.root, packet=self.packet)


if __name__ == "__main__":
    unittest.main()
