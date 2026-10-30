"""Prepare pending HUMAN relevance review from an explicitly selected demo run.

Offline only. Legacy packets copy observations; explicitly selected priority
packets also verify archived context. Neither mode creates relevance judgments.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/product_mvp/verified_demo_cases.json"
VERSION = "passport-human-review-packet-v1"
PRIORITY_VERSION = "passport-priority-review-v1"
DECISIONS = ["necessary", "acceptable", "redundant", "irrelevant", "unsupported", "uncertain"]
INSTRUCTIONS = """# Private Passport relevance review

This packet contains synthetic, legally unreviewed demo observations. No human
review has been completed by this exporter. The source report and every previous
attempt remain unchanged. The run was explicitly selected, not chosen by score.

Review each row in its own intake and investigation-topic context. Identical
quotes under different topics require separate relevance decisions. Inspect the
official source and the archived document/chunk/hash references, surrounding
provisions, version and necessary qualifications. A copied integrity observation
is a historical fidelity check, not a new archive verification or legal review.
Do not infer current legal force from a live official URL. Status stays unknown.

Record decisions separately using review_id, reviewer identity/qualification,
decision, rationale, reviewed_at and source references. Keep this packet immutable.
Allowed evidence decisions: necessary, acceptable, redundant, irrelevant,
unsupported, uncertain. Necessary means required to support this research topic;
acceptable is a relevant alternative or useful qualification; redundant repeats
already adequate support; irrelevant does not address the topic; unsupported
does not establish the attributed proposition; uncertain requires further review.
Do not collapse redundant, irrelevant and unsupported into one failure category.

For missing_evidence/case_without_evidence rows, document missing coverage or the
reason no evidence was available; do not invent a citation. Add missing necessary
qualifications and acceptable alternatives in the separate reviewer record.
Record disagreements explicitly rather than silently replacing earlier decisions.
No record becomes independently legally verified without genuine human review.
These cases are a regression/review resource, not an independent final benchmark.

No model calls, network access, corpus promotion or source synchronization occur.
"""


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def export_packet(report_path, run_id, output_dir):
    """Create a new immutable preparation directory; never overwrite a packet."""
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("an explicit nonempty run_id is required")
    raw = Path(report_path).read_bytes()
    report = json.loads(raw.decode("utf-8-sig"))
    runs = report["runs"]
    matches = [run for run in runs if run.get("run_id") == run_id]
    if len(matches) != 1:
        raise ValueError("run_id must identify exactly one retained run")
    run = matches[0]
    if run.get("synthetic") is not True:
        raise ValueError("only explicitly synthetic demo observations may be exported")
    report_hash = sha(raw)
    packet_id = sha(encoded({"version": VERSION, "report_sha256": report_hash, "run_id": run_id}))
    rows, seen_cases = [], set()
    for case in run["case_results"]:
        case_id = case["case_id"]
        if not isinstance(case_id, str) or not case_id or case_id in seen_cases:
            raise ValueError("missing or duplicate case_id")
        seen_cases.add(case_id)
        response = case.get("response") or {}
        evidence = response.get("verified_evidence") or []
        by_id = {}
        for item in evidence:
            identifier = item.get("evidence_id")
            if not isinstance(identifier, str) or not identifier or identifier in by_id:
                raise ValueError("missing or duplicate evidence_id")
            by_id[identifier] = item
        checks = case.get("validation", {}).get("evidence_checks", [])
        referenced = set()
        case_rows = []

        def add(kind, issue, item):
            identifier = item["evidence_id"] if item else None
            row = {"packet_id": packet_id, "run_id": run_id, "case_id": case_id,
                   "kind": kind, "intake": case.get("intake"), "issue": issue,
                   "evidence": item, "generation_id": response.get("generation_id"),
                   "model": response.get("model"), "response_status": response.get("status"),
                   "request_policy_version": response.get("request_policy_version"),
                   "recorded_outcome": case.get("outcome"),
                   "unresolved_questions": response.get("unresolved_questions", []),
                   "fidelity_basis": "copied_historical_observation_not_reverified",
                   "recorded_fidelity_observations": [c for c in checks if identifier and c.get("evidence_id") == identifier],
                   "review_status": "pending", "decision": None, "reviewer": None,
                   "rationale": None, "allowed_decisions": DECISIONS,
                   "legal_relevance_review": "unreviewed"}
            row["review_id"] = sha(encoded(row))
            case_rows.append(row)

        seen_topics = set()
        for issue in response.get("issues", []):
            topic = issue.get("issue_id")
            if not isinstance(topic, str) or not topic or topic in seen_topics:
                raise ValueError("missing or duplicate issue_id")
            seen_topics.add(topic)
            ids = list(dict.fromkeys(issue.get("evidence_ids") or []))
            if any(identifier not in by_id for identifier in ids):
                raise ValueError("issue references unknown evidence")
            if not ids:
                add("missing_evidence", issue, None)
            for identifier in ids:
                referenced.add(identifier)
                add("evidence_relevance", issue, by_id[identifier])
        for identifier, item in by_id.items():
            if identifier not in referenced:
                add("unassigned_evidence", None, item)
        if not case_rows:
            add("case_without_evidence", None, None)
        rows.extend(case_rows)

    # Freeze payloads before creating output. The manifest is written last so a
    # filesystem failure cannot leave an apparently complete packet.
    files = {"selected_run.json": encoded(run), "instructions.md": INSTRUCTIONS.encode("utf-8"),
             "review_queue.jsonl": b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in rows)}
    manifest = {"schema_version": VERSION, "packet_id": packet_id,
                "input_report_sha256": report_hash, "selected_run_id": run_id,
                "selected_run_status": run.get("status"),
                "scenario_file_sha256": run.get("scenario_file_sha256"),
                "historical_runs_retained": len(runs), "source_report_modified": False,
                "case_count": len(seen_cases), "review_unit_count": len(rows),
                "kind_counts": dict(sorted(Counter(row["kind"] for row in rows).items())),
                "human_reviews_completed": 0, "legal_relevance_review": "pending_human_review",
                "archive_reverification_performed": False,
                "files": {name: sha(content) for name, content in sorted(files.items())}}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        with (output / name).open("xb") as stream:
            stream.write(content)
    with (output / "manifest.json").open("xb") as stream:
        stream.write(encoded(manifest))
    return manifest


def export_priority_packet(report_path, run_id, output_dir, *, case_ids, source_packet, root=ROOT):
    """Enrich explicitly ordered existing review units without changing their IDs.

    Archived chunks/neighbors are review context, never reconstructed model
    inputs. Original fidelity observations remain unchanged and separately named.
    """
    from tools.passport_demo import CitationVerifier
    from ipindia_corpus.paths import archive_path

    if (not isinstance(case_ids, list) or not case_ids or len(set(case_ids)) != len(case_ids)
            or any(not isinstance(c, str) or not re.fullmatch(r"[a-z0-9_]+", c) for c in case_ids)):
        raise ValueError("explicit unique ordered case IDs required")
    raw_report = Path(report_path).read_bytes()
    report = json.loads(raw_report.decode("utf-8-sig"))
    runs = [run for run in report["runs"] if run.get("run_id") == run_id]
    if len(runs) != 1 or runs[0].get("synthetic") is not True:
        raise ValueError("exactly one synthetic retained run required")
    run = runs[0]
    cases = {case["case_id"]: case for case in run["case_results"]}
    if len(cases) != len(run["case_results"]) or any(c not in cases for c in case_ids):
        raise ValueError("unknown or duplicate source case ID")
    source = Path(source_packet)
    raw_manifest = (source / "manifest.json").read_bytes()
    original = json.loads(raw_manifest)
    source_files = original["files"]
    if set(source_files) != {"selected_run.json", "instructions.md", "review_queue.jsonl"}:
        raise ValueError("unexpected source packet files")
    payloads = {name: (source / name).read_bytes() for name in source_files}
    if any(sha(payloads[name]) != digest for name, digest in source_files.items()):
        raise ValueError("source packet hash mismatch")
    if (original.get("selected_run_id") != run_id or original.get("input_report_sha256") != sha(raw_report)
            or payloads["selected_run.json"] != encoded(run)):
        raise ValueError("source report/packet binding mismatch")
    expected_packet_id = sha(encoded({"version": VERSION, "report_sha256": sha(raw_report), "run_id": run_id}))
    if original.get("packet_id") != expected_packet_id:
        raise ValueError("source packet identity mismatch")
    old_rows = [json.loads(line) for line in payloads["review_queue.jsonl"].decode("utf-8").splitlines()]
    row_ids = set()
    for row in old_rows:
        row_id = row.get("review_id")
        if (row_id in row_ids or row_id != sha(encoded({k: v for k, v in row.items() if k != "review_id"}))
                or row.get("run_id") != run_id or row.get("packet_id") != expected_packet_id):
            raise ValueError("source review identity mismatch")
        row_ids.add(row_id)
        if row.get("review_status") != "pending" or any(row.get(k) is not None for k in ("decision", "reviewer", "rationale")):
            raise ValueError("priority preparation requires pending source decisions")
    verifier = CitationVerifier(root)
    files, rows, archive_bindings, referenced_originals = {}, [], {}, {}
    for case_id in case_ids:
        case = cases[case_id]
        response = case.get("response") or {}
        selected_rows = [row for row in old_rows if row["case_id"] == case_id]
        if not selected_rows:
            raise ValueError("selected case has no original review units")
        evidence = response.get("verified_evidence", [])
        if len({item["evidence_id"] for item in evidence}) != len(evidence):
            raise ValueError("duplicate selected evidence ID")
        gid = response.get("generation_id")
        if not isinstance(gid, str) or not gid or any(c not in "0123456789TZ" for c in gid):
            raise ValueError("invalid recorded generation")
        directory = Path(root) / "data/generations" / gid
        if gid not in archive_bindings:
            archive_bindings[gid] = {name: sha((directory / name).read_bytes()) for name in ("manifest.json", "documents.jsonl", "chunks.jsonl")}
        documents, chunks = verifier.generation(gid)
        contexts = []
        for item in evidence:
            errors = verifier.evidence(item, documents, chunks)
            if errors:
                raise ValueError("citation/archive mismatch: " + ",".join(errors))
            chunk = chunks[item["chunk_id"]]
            document = documents[item["document_id"]]
            referenced_originals[document["raw_path"]] = document["sha256"]
            same_document = [c for c in chunks.values() if c["document_id"] == item["document_id"]]
            position = next(i for i, c in enumerate(same_document) if c["chunk_id"] == chunk["chunk_id"])
            neighbors = same_document[max(0, position - 1):position] + same_document[position + 1:position + 2]
            contexts.append({"evidence_id": item["evidence_id"], "archived_chunk": chunk,
                "adjacent_chunks": neighbors, "document": document,
                "basis": "verified_current_archived_context_not_proof_of_historical_model_input",
                "adjacency": "one_previous_and_one_next_chunk_in_same_document_generation_order",
                "legal_relevance_review": "unreviewed", "document_status": "unknown"})
        # Never describe a retained retrieval pool as the exact packed model input.
        retained = response.get("retrieval_evidence")
        case_payload = {"case_id": case_id, "run_id": run_id, "intake": case.get("intake"),
            "issues": response.get("issues", []), "selected_quotations_and_citations": evidence,
            "unanswered_questions": response.get("unresolved_questions", []),
            "unsupported_domains": response.get("unsupported_domains", []), "response_status": response.get("status"),
            "model": response.get("model"), "generation_id": gid, "review_ids": [r["review_id"] for r in selected_rows],
            "historical_model_presented_context": {"available": False, "reason": "not_retained_in_product_observation"},
            "retrieval_candidate_pool": {"available": isinstance(retained, list), "passages": retained if isinstance(retained, list) else None,
                "basis": "recorded_retrieval_evidence" if isinstance(retained, list) else "not_retained_in_product_observation"},
            "archive_contexts": contexts, "review_status": "pending", "legal_relevance_review": "unreviewed"}
        files[f"cases/{case_id}.json"] = encoded(case_payload)
        rows.extend(selected_rows)
    files["review_queue.jsonl"] = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in rows)
    files["instructions.md"] = (INSTRUCTIONS + "\n## Priority batch context\n\n"
        "Review cases in the manifest's explicit order. Original review IDs and pending decisions are preserved.\n"
        "The cases include freshly verified full archived chunks and one neighbor on each side where available.\n"
        "These surrounding passages are review context, not proof of what the historical model saw.\n"
        "Historical candidate pools and packed model inputs are unavailable when the observation did not retain them.\n").encode("utf-8")
    manifest = {"schema_version": PRIORITY_VERSION, "selected_run_id": run_id, "ordered_case_ids": case_ids,
        "case_count": len(case_ids), "review_unit_count": len(rows), "preserved_review_ids": [r["review_id"] for r in rows],
        "human_reviews_completed": 0, "legal_relevance_review": "pending_human_review", "source_report_sha256": sha(raw_report),
        "source_packet_manifest_sha256": sha(raw_manifest), "source_packet_payload_sha256": source_files,
        "scenario_file_sha256": run.get("scenario_file_sha256"), "archive_reverification_performed": True,
        "verified_generation_files": archive_bindings, "source_packet_modified": False,
        "files": {name: sha(content) for name, content in sorted(files.items())}}
    manifest["packet_id"] = sha(encoded(manifest))
    if any(sha((Path(root) / "data/generations" / gid / name).read_bytes()) != digest
           for gid, bindings in archive_bindings.items() for name, digest in bindings.items()):
        raise ValueError("generation changed during preparation")
    if any(sha(archive_path(root, path).read_bytes()) != digest for path, digest in referenced_originals.items()):
        raise ValueError("original source changed during preparation")
    if Path(report_path).read_bytes() != raw_report or (source / "manifest.json").read_bytes() != raw_manifest or any((source / name).read_bytes() != raw for name, raw in payloads.items()):
        raise ValueError("source changed during preparation")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
    with (output / "manifest.json").open("xb") as stream:
        stream.write(encoded(manifest))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case-ids", nargs="+", help="Explicit ordered subset for an enriched priority packet")
    parser.add_argument("--source-packet", type=Path, help="Immutable original review packet; required with --case-ids")
    parser.add_argument("--archive-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if bool(args.case_ids) != bool(args.source_packet):
        parser.error("--case-ids and --source-packet must be provided together")
    if args.case_ids:
        manifest = export_priority_packet(args.report, args.run_id, args.output_dir, case_ids=args.case_ids,
                                         source_packet=args.source_packet, root=args.archive_root)
    else:
        manifest = export_packet(args.report, args.run_id, args.output_dir)
    print(json.dumps({key: manifest[key] for key in ("packet_id", "selected_run_id", "case_count", "review_unit_count", "human_reviews_completed")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
