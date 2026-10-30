"""Offline assessment of actual three-source index and benchmark readiness.

Never treats CURRENT, ACTIVE or an unindexed candidate as the searchable index.
Explicit active_* arguments are reserved for trusted callers supplying a database
snapshot; omitting them reads PostgreSQL without changing any database state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .sync_runner import ROOT, atomic_json

SOURCES = ("ipindia", "indiacode", "tkdl")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if Path(relative).is_absolute() or not candidate.is_relative_to(root.resolve()):
        raise ValueError("artifact_path_outside_project")
    return candidate


def _load(root: Path, relative: str, artifacts: dict) -> dict:
    data = _path(root, relative).read_bytes()
    artifacts[relative] = _digest(data)
    return json.loads(data)


def _verify_inventory(root: Path, audit: dict) -> list[str]:
    failures, inventory = [], []
    records = audit.get("documents", []) + audit.get("user_supplied_pdf_originals", [])
    records += audit.get("metadata_files", [])
    seen = set()
    for record in records:
        try:
            relative = record["path"]
            data = _path(root, relative).read_bytes()
            if relative in seen:
                raise ValueError("duplicate_inventory_path")
            seen.add(relative)
            if _digest(data) != record["sha256"] or len(data) != record["bytes"]:
                raise ValueError("artifact_hash_or_length_changed")
            inventory.append({key: record[key] for key in ("path", "sha256", "bytes")})
        except (OSError, ValueError, KeyError, TypeError):
            failures.append(f"audit_inventory_invalid:{record.get('path', 'unknown')}")
    calculated = _digest(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode())
    if calculated != audit.get("inventory_sha256"):
        failures.append("audit_inventory_digest_mismatch")
    return failures


def _database_active() -> tuple[str, list[dict], dict]:
    from .indexer import _connect
    connection = _connect()
    try:
        with connection.transaction():
            connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            row = connection.execute("SELECT value FROM app_state WHERE key='active_generation'").fetchone()
            if not row:
                raise ValueError("no_active_searchable_generation")
            identifier = row[0]
            documents = [row[0] for row in connection.execute(
                "SELECT metadata FROM corpus_documents WHERE generation_id=%s", (identifier,)).fetchall()]
            index = connection.execute(
                "SELECT manifest_sha256,documents,chunks,model_revision FROM index_generations WHERE generation_id=%s",
                (identifier,)).fetchone()
            if not index:
                raise ValueError("active_index_manifest_missing")
            chunks = connection.execute("SELECT count(*) FROM corpus_chunks WHERE generation_id=%s",
                                        (identifier,)).fetchone()[0]
            embedded = connection.execute(
                "SELECT count(*) FROM corpus_chunks c JOIN embedding_cache e ON c.content_hash=e.content_hash "
                "AND e.model_revision=%s WHERE c.generation_id=%s", (index[3], identifier)).fetchone()[0]
            if index[1] != len(documents) or index[2] != chunks or embedded != chunks:
                raise ValueError("active_index_incomplete")
            return identifier, documents, {"authority": "postgresql_active_index",
                                           "manifest_sha256": index[0], "indexed_chunks": chunks,
                                           "embedding_model_revision": index[3]}
    finally:
        connection.close()


def _certified(root: Path, source: str, audit: dict, inventory_failures: list[str],
               documents: list[dict], artifacts: dict) -> tuple[int, list[str]]:
    summary = audit.get("sources", {}).get(source, {})
    entries = [entry for entry in audit.get("documents", []) if entry.get("source") == source]
    verified, blockers = [], []
    if inventory_failures:
        return 0, ["certification_audit_stale_or_invalid"]
    for entry in entries:
        if not entry.get("certified") or entry.get("search_admission") != "accepted":
            continue
        receipt = entry.get("original_receipt", {})
        try:
            if not receipt.get("verified") or not receipt.get("capture_timestamp"):
                raise ValueError("verified_original_receipt_required")
            original_path = receipt["original_path"]
            if not original_path.replace("\\", "/").startswith("data/raw/"):
                raise ValueError("immutable_archived_original_required")
            raw = _path(root, original_path).read_bytes()
            if _digest(raw) != receipt["original_sha256"]:
                raise ValueError("archived_original_hash_mismatch")
            # A provenance receipt alone cannot certify transformed passage fidelity.
            proof_path = entry["fidelity_report_path"]
            proof = _load(root, proof_path, artifacts)
            if (artifacts[proof_path] != entry["fidelity_report_sha256"] or not proof.get("pass")
                    or proof.get("original_sha256") != receipt["original_sha256"]
                    or proof.get("transformed_sha256") != entry["sha256"]):
                raise ValueError("fidelity_evidence_missing_or_invalid")
            matches = [doc for doc in documents if doc.get("source") == source
                       and doc.get("source_url") == entry.get("source_url")
                       and doc.get("sha256") == receipt["original_sha256"]]
            if not matches:
                raise ValueError("certified_original_not_in_active_index")
            verified.append(entry)
        except (OSError, ValueError, KeyError, TypeError):
            blockers.append("original_capture_or_fidelity_evidence_missing_or_invalid")
    if summary.get("certified_documents") != len(verified):
        blockers.append("certification_count_does_not_match_verified_active_documents")
    if not verified:
        blockers.append("originals_and_fidelity_not_certified")
    if summary.get("search_admission") != "accepted":
        blockers.append("source_search_admission_blocked")
    active_count = sum(doc.get("source") == source for doc in documents)
    if active_count != len(verified):
        blockers.append("active_source_documents_not_fully_certified")
    return len(verified), sorted(set(blockers))


def _benchmark(root: Path, active: str | None, documents: list[dict], chunks: list[dict],
               artifacts: dict) -> dict:
    failures, reviewed, counts = [], {}, {}
    result = {"path": "tests/retrieval_queries.json", "ready": False, "frozen": False,
              "reviewed_by_source": {source: 0 for source in SOURCES}, "passage_bindings_verified": False}
    try:
        labels = _load(root, result["path"], artifacts)
        queries = labels["queries"]
        result["frozen"] = bool(labels.get("frozen"))
        result["queries"] = len(queries)
        counts = Counter(item.get("source") for item in queries)
        reviewed = Counter(item.get("source") for item in queries if item.get("reviewed"))
        result["queries_by_source"] = {source: counts[source] for source in SOURCES}
        result["reviewed_by_source"] = {source: reviewed[source] for source in SOURCES}
        if len({item["id"] for item in queries}) != len(queries):
            failures.append("duplicate_query_identifiers")
        if not result["frozen"]:
            failures.append("unified_benchmark_not_frozen")
        if any(reviewed[source] < 10 for source in SOURCES):
            failures.append("ten_reviewed_physical_passage_queries_per_source_required")
        if not active or labels.get("frozen_generation_id") != active:
            failures.append("benchmark_not_bound_to_active_generation")
        if not labels.get("frozen_by") or not labels.get("frozen_at"):
            failures.append("benchmark_freeze_review_identity_required")
        if not failures:
            from .passage_evaluation import validate_labels
            validate_labels(labels, documents, chunks)
            result["passage_bindings_verified"] = True
    except (OSError, ValueError, KeyError, TypeError):
        failures.append("unified_benchmark_missing_or_physical_passage_binding_invalid")
    result.update(ready=not failures, blockers=failures,
                  status="ready" if not failures else "blocked_not_frozen" if not result["frozen"] else "blocked")
    return result


def assess(root: Path = ROOT, active_generation_id: str | None = None,
           active_documents: list[dict] | None = None) -> dict:
    """Assess trusted active-index metadata; no writes, capture or activation."""
    root = Path(root)
    artifacts, failures = {}, []
    documents, chunks, active = [], [], None
    index = {"authority": "unavailable", "available": False, "counts": {source: None for source in SOURCES}}
    try:
        if active_generation_id is None and active_documents is None:
            active, documents, details = _database_active()
        elif active_generation_id and active_documents is not None:
            active, documents, details = active_generation_id, active_documents, {"authority": "trusted_caller_active_index"}
        else:
            raise ValueError("active_context_incomplete")
        index.update(details, available=True, generation_id=active,
                     counts={source: sum(doc.get("source") == source for doc in documents) for source in SOURCES})
        from .indexer import generation
        _, archived_documents, chunks = generation(root, generation_id=active)
        manifest_path = f"data/generations/{active}/manifest.json"
        _load(root, manifest_path, artifacts)
        if index.get("manifest_sha256", artifacts[manifest_path]) != artifacts[manifest_path]:
            raise ValueError("active_manifest_hash_mismatch")
        by_id = {doc["document_id"]: doc for doc in documents}
        if (len(by_id) != len(documents) or len(documents) != len(archived_documents)
                or any(by_id.get(doc["document_id"]) != doc for doc in archived_documents)):
            raise ValueError("active_document_metadata_mismatch")
        index["archive_bindings_verified"] = True
    except Exception:
        # Database exception details, credentials and host names stay out of reports.
        failures.append("active_searchable_index_unavailable_or_unverified")
        index["archive_bindings_verified"] = False
        chunks = []
    try:
        audit = _load(root, "reports/corpus_certification.json", artifacts)
        inventory_failures = _verify_inventory(root, audit)
    except (OSError, ValueError, KeyError, TypeError):
        audit, inventory_failures = {}, ["certification_audit_unavailable"]
    sources = {}
    for source in SOURCES:
        count = index["counts"][source]
        summary = audit.get("sources", {}).get(source, {})
        certified, source_failures = (count or 0, []) if source == "ipindia" else _certified(
            root, source, audit, inventory_failures, documents, artifacts)
        if count is None:
            source_failures.append("active_source_count_unavailable")
        elif not count:
            source_failures.append("no_active_searchable_capture")
        if not index.get("archive_bindings_verified"):
            source_failures.append("active_archive_bindings_unverified")
        sources[source] = {"indexed_documents": count, "staged_documents": summary.get("staged_documents", 0),
                           "certified_active_documents": certified if index.get("archive_bindings_verified") else 0,
                           "ready": not source_failures, "blockers": sorted(set(source_failures)),
                           "actions": summary.get("actions", []) if source != "ipindia" else []}
    benchmark = _benchmark(root, active, documents, chunks, artifacts)
    evaluation_ready = False
    try:
        report = _load(root, "reports/retrieval_evaluation.json", artifacts)
        evaluation_ready = bool(benchmark["ready"] and report.get("pass") and not report.get("partial")
                                and report.get("generation_id") == active
                                and report.get("label_sha256") == artifacts.get("tests/retrieval_queries.json")
                                and report.get("recall_at_5", 0) >= .9 and report.get("p95_ms", float("inf")) <= 2000
                                and report.get("observations", 0) >= benchmark.get("queries", 30) * 3)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    corpus_ready = not failures and all(item["ready"] for item in sources.values())
    evidence_ready = corpus_ready and benchmark["ready"] and evaluation_ready
    blocked = [source for source in SOURCES if not sources[source]["ready"]]
    return {"schema_version": 1, "checked_at": datetime.now(timezone.utc).isoformat(),
            "mode": "offline_readiness_assessment", "active_index": index, "sources": sources,
            "blocked_sources": blocked, "blocked_source_count": len(blocked),
            "certification_inventory_valid": not inventory_failures,
            "certification_inventory_failures": inventory_failures, "artifact_sha256": artifacts,
            "unified_corpus_ready": corpus_ready, "unified_benchmark": benchmark,
            "three_source_ready": evidence_ready, "evaluation_ready": evaluation_ready,
            "stages": {"evidence": {"ready": evidence_ready,
                                       "blockers": failures + (["unified_corpus_incomplete"] if not corpus_ready else [])
                                       + benchmark["blockers"] + ([] if evaluation_ready else ["unfiltered_three_pass_evaluation_required"])},
                       "answers": {"ready": False, "blockers": ["three_source_evidence_required"] if not evidence_ready else
                                   ["live_answer_claim_citation_abstention_and_latency_release_evaluation_required"]},
                       "public": {"ready": False, "blockers": ["public_security_and_performance_release_evidence_required"]}},
            "limitations": ["Candidate generations and filesystem pointers never establish active searchable coverage.",
                            "Staged transformed files and official URL syntax do not prove original-byte fidelity.",
                            "This assessment never invents labels, freezes a benchmark or changes legal status.",
                            "TKDL remains institutional background; it cannot establish a binding legal rule."]}


def write_report(root: Path, report: dict) -> None:
    atomic_json(root / "reports/unified_readiness.json", report)
    lines = ["# Unified three-source readiness", "",
             f"Evidence ready: **{str(report['three_source_ready']).lower()}**.", "",
             f"Active index authority: `{report['active_index']['authority']}`.", "",
             "| Source | Active documents | Certified active documents | Ready |",
             "|---|---:|---:|---|"]
    for source, item in report["sources"].items():
        count = item["indexed_documents"] if item["indexed_documents"] is not None else "unknown"
        lines.append(f"| {source} | {count} | {item['certified_active_documents']} | {item['ready']} |")
    lines += ["", "## Required work", ""]
    for source, item in report["sources"].items():
        for blocker in item["blockers"]:
            lines.append(f"- {source}: {blocker}.")
        for action in item["actions"]:
            lines.append(f"- {source}: {action}")
    benchmark = report["unified_benchmark"]
    lines += ["", f"Unified benchmark status: **{benchmark['status']}**; reviewed queries by source: "
              + ", ".join(f"{source}={count}" for source, count in benchmark["reviewed_by_source"].items()) + ".", "",
              "The historical scaffold is preserved. No new source labels or unified captures were invented.", "",
              "Answer release and public release remain blocked pending their separate integrated evaluations.", "",
              "Hashes of assessed artifacts and inventory drift findings are recorded in the JSON report."]
    (root / "reports/unified_readiness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = assess(args.root)
    if args.write_report:
        write_report(args.root, report)
    print(json.dumps(report, indent=2))
    return 0 if report["three_source_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
