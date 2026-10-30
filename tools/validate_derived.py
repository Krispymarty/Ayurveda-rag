"""Validate published derived exports against their immutable local archives."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    documents = load(ROOT / "documents.jsonl")
    chunks = load(ROOT / "chunks.jsonl")
    by_id = {item["document_id"]: item for item in documents}
    errors = []
    if len(by_id) != len(documents):
        errors.append("duplicate_document_id")
    chunk_ids = set()
    for item in documents:
        url = item["source_url"]
        if url != item["canonical_url"] or urlsplit(url).hostname not in {"ipindia.gov.in", "www.ipindia.gov.in"}:
            errors.append(f"invalid_source_url:{item['document_id']}")
        path = Path(item["raw_path"])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            errors.append(f"raw_hash_mismatch:{item['document_id']}")
        if item["status"] != "unknown" or not item["metadata_review_required"]:
            errors.append(f"unverified_temporal_claim:{item['document_id']}")
    citation_matches = 0
    page_anchors = 0
    for item in chunks:
        identifier = item["chunk_id"]
        if identifier in chunk_ids:
            errors.append(f"duplicate_chunk_id:{identifier}")
        chunk_ids.add(identifier)
        parent = by_id.get(item["document_id"])
        if parent is None:
            errors.append(f"orphan_chunk:{identifier}")
        elif item["source_url"] == parent["source_url"]:
            citation_matches += 1
        else:
            errors.append(f"citation_mismatch:{identifier}")
        if hashlib.sha256(item["text"].encode()).hexdigest() != item["content_hash"]:
            errors.append(f"chunk_hash_mismatch:{identifier}")
        if item["page_start"] is not None:
            page_anchors += 1
            if item["page_end"] is None or item["page_end"] < item["page_start"]:
                errors.append(f"invalid_page_span:{identifier}")
    result = {
        "documents": len(documents), "chunks": len(chunks),
        "source_counts": dict(Counter(item["source"] for item in documents)),
        "domain_counts": dict(Counter(item["legal_domain"] for item in documents)),
        "citation_matches": citation_matches, "citation_readiness_percent": round(100 * citation_matches / len(chunks), 1) if chunks else 0,
        "page_anchored_chunks": page_anchors,
        "status_counts": dict(Counter(item["status"] for item in documents)),
        "duplicate_chunk_content_hash_groups": sum(count > 1 for count in Counter(item["content_hash"] for item in chunks).values()),
        "staged_tkdl_pages_not_in_retrieval": len(load(ROOT / "data" / "documents" / "tkdl_candidates.jsonl")),
        "errors": errors,
    }
    destination = ROOT / "reports" / "derived_validation.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("documents", "chunks", "citation_matches", "page_anchored_chunks", "errors")}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
