"""Build canonical derived exports from audited local archives; no network access."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipindia_corpus.pipeline import Fetch, build_document, extract, structure_chunks  # noqa: E402
from ipindia_corpus.sync import Ledger  # noqa: E402

CORPUS = ROOT / "corpus"
DATA = ROOT / "data"


def jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_ipindia() -> tuple[list[dict], list[dict], list[dict], Counter]:
    documents: list[dict] = []
    chunks: list[dict] = []
    review: list[dict] = []
    counts: Counter = Counter()
    cache_root = (CORPUS / "cache").resolve()
    for prior in jsonl(CORPUS / "documents.jsonl"):
        document_id = prior["document_id"]
        actual_url = prior.get("download_url") or prior.get("source_url")
        path = Path(prior["raw_file_path"]).resolve()
        if not path.is_relative_to(cache_root) or not path.is_file():
            review.append({"source": "ipindia", "document_id": document_id, "url": actual_url, "reason": "raw_path_missing_or_outside_cache"})
            continue
        raw = path.read_bytes()
        raw_hash = sha256(raw)
        if raw_hash != prior["content_hash"]:
            review.append({"source": "ipindia", "document_id": document_id, "url": actual_url, "reason": "raw_hash_mismatch"})
            continue
        cache_meta_path = path.with_suffix(".json")
        if not cache_meta_path.is_file():
            review.append({"source": "ipindia", "document_id": document_id, "url": actual_url, "reason": "cache_metadata_missing"})
            continue
        cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
        if cache_meta.get("url") != actual_url:
            review.append({"source": "ipindia", "document_id": document_id, "url": actual_url, "reason": "cache_url_mismatch"})
            continue
        domain = prior.get("ip_domain") or "General IP"
        folder = {
            "Patent": "patents", "Trademark": "trademarks", "GI": "gi", "Design": "designs",
            "AYUSH": "ayush_tk", "Traditional Knowledge": "ayush_tk",
        }.get(domain, "general")
        suffix = Path(urlsplit(actual_url).path).suffix.lower() or (".html" if cache_meta["content_type"].startswith("text/html") else ".bin")
        raw_path = DATA / "raw" / "ipindia" / folder / f"{raw_hash}{suffix}"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if raw_path.exists():
            if sha256(raw_path.read_bytes()) != raw_hash:
                raise ValueError(f"immutable raw collision: {raw_path}")
        else:
            shutil.copyfile(path, raw_path)
        fetch = Fetch(actual_url, raw, cache_meta["content_type"], cache_meta["retrieved_at"], raw_path)
        try:
            _, content, _ = extract(fetch)
        except (ValueError, OSError, ImportError) as exc:
            review.append({"source": "ipindia", "document_id": document_id, "url": actual_url, "reason": f"reextract_failed:{exc}"})
            continue
        if len(content.strip()) < 80:
            review.append({"source": "ipindia", "document_id": document_id, "url": actual_url, "reason": "too_little_text"})
            continue
        discovered_from = prior.get("discovered_from_url") or prior.get("source_url")
        document = build_document(fetch, prior["title"], content, discovered_from)
        document["document_type"] = prior["document_type"]
        document["ip_domain"] = domain
        document["legal_domain"] = domain
        document["related_ip_domains"] = prior.get("related_ip_domains", [])
        document["alternate_urls"] = prior.get("alternate_urls") or prior.get("alternate_download_urls", [])
        document["etag"] = cache_meta.get("etag")
        document["last_modified"] = cache_meta.get("last_modified")
        # A cached acquisition is not a new live check.
        document["last_checked_at"] = cache_meta["retrieved_at"]
        if document["document_id"] != document_id:
            raise ValueError(f"document identity changed: {document_id}")
        normalized_path = DATA / "normalized" / "ipindia" / f"{document_id}_{raw_hash}.txt"
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        normalized_path.write_text(content, encoding="utf-8")
        document["normalized_path"] = str(normalized_path.resolve())
        local_chunks = structure_chunks(content, document)
        if not local_chunks:
            review.append({"source": "ipindia", "document_id": document_id, "url": actual_url, "reason": "no_chunks"})
            continue
        for chunk in local_chunks:
            if chunk["source_url"] != actual_url:
                raise ValueError(f"citation mismatch: {document_id}")
        documents.append(document)
        chunks.extend(local_chunks)
        counts["parsed"] += 1
        counts["chunks"] += len(local_chunks)
    return documents, chunks, review, counts


def main() -> None:
    documents, chunks, review, counts = build_ipindia()
    write_jsonl(DATA / "documents" / "documents.jsonl", documents)
    write_jsonl(DATA / "chunks" / "chunks.jsonl", chunks)
    write_jsonl(ROOT / "documents.jsonl", documents)
    write_jsonl(ROOT / "chunks.jsonl", chunks)
    with (ROOT / "sources.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source_id", "domain", "enabled", "sync_frequency", "last_successful_sync", "next_sync"])
        writer.writeheader()
        writer.writerows([
            {"source_id": "ipindia", "domain": "ipindia.gov.in", "enabled": "false", "sync_frequency": "manual_until_review", "last_successful_sync": "", "next_sync": ""},
            {"source_id": "tkdl", "domain": "tkdl.res.in", "enabled": "false", "sync_frequency": "manual_until_review", "last_successful_sync": "", "next_sync": ""},
            {"source_id": "indiacode", "domain": "indiacode.nic.in", "enabled": "false", "sync_frequency": "not_configured", "last_successful_sync": "", "next_sync": ""},
        ])
    prior_review = list(jsonl(CORPUS / "review_queue.jsonl")) if (CORPUS / "review_queue.jsonl").exists() else []
    with (ROOT / "manual_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source", "document_id", "url", "reason"])
        writer.writeheader()
        writer.writerows(review)
        writer.writerows({"source": "ipindia", "document_id": "", "url": x.get("url", ""), "reason": x.get("reason", "")} for x in prior_review)
    with (ROOT / "failed_urls.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["url", "reason"])
        writer.writeheader()
        latest_audit = {item.get("url", ""): item for item in jsonl(CORPUS / "audit.jsonl")}
        writer.writerows({"url": item["url"], "reason": item.get("reason", "")} for item in latest_audit.values() if item.get("status") == "error")
    ledger = Ledger(DATA / "manifests" / "manifest.sqlite", DATA / "raw")
    try:
        existing_ids = {row[0] for row in ledger.db.execute("SELECT canonical_id FROM document_manifest")}
        new_documents = [document for document in documents if document["canonical_id"] not in existing_ids]
        if new_documents:
            run_id = ledger.start("ipindia")
            with ledger.db:
                for document in new_documents:
                    raw_path = Path(document["raw_path"])
                    if sha256(raw_path.read_bytes()) != document["sha256"]:
                        raise ValueError(f"archive hash changed: {raw_path}")
                    metadata_hash = sha256(json.dumps({"title": document["title"], "document_type": document["document_type"]}, sort_keys=True, ensure_ascii=False).encode())
                    ledger.db.execute("INSERT INTO document_manifest(canonical_id,source_id,canonical_url,sha256,metadata_sha256,etag,last_modified,retrieved_at,last_checked_at,raw_path,verified_version) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                                      (document["canonical_id"], "ipindia", document["canonical_url"], document["sha256"], metadata_hash, document["etag"], document["last_modified"], document["retrieved_at"], document["last_checked_at"], str(raw_path), None))
                    ledger.db.execute("INSERT INTO document_versions(canonical_id,sha256,raw_path,retrieved_at,verified_version) VALUES (?,?,?,?,?)",
                                      (document["canonical_id"], document["sha256"], str(raw_path), document["retrieved_at"], None))
                    for url in [document["canonical_url"], *document["alternate_urls"]]:
                        ledger.db.execute("INSERT OR IGNORE INTO document_urls(url,canonical_id) VALUES (?,?)", (url, document["canonical_id"]))
                    ledger._change(run_id, document["canonical_id"], document["canonical_url"], "NEW_DOCUMENT", current=document["sha256"], detail="offline baseline import; not a live synchronization")
            ledger.finish(run_id, "imported")
        with ledger.db:
            for document in documents:
                current = ledger.db.execute("SELECT sha256 FROM document_manifest WHERE canonical_id=?", (document["canonical_id"],)).fetchone()
                if current is None or current["sha256"] != document["sha256"]:
                    raise ValueError(f"offline baseline changed; use sync ledger for {document['canonical_id']}")
                ledger.db.execute("UPDATE document_manifest SET raw_path=? WHERE canonical_id=?", (document["raw_path"], document["canonical_id"]))
                ledger.db.execute("UPDATE document_versions SET raw_path=? WHERE canonical_id=? AND sha256=?", (document["raw_path"], document["canonical_id"], document["sha256"]))
                for url in [document["canonical_url"], *document["alternate_urls"]]:
                    ledger.db.execute("INSERT OR IGNORE INTO document_urls(url,canonical_id) VALUES (?,?)", (url, document["canonical_id"]))
    finally:
        ledger.close()
    print(json.dumps({"documents": len(documents), "chunks": len(chunks), "manual_review": len(review) + len(prior_review), "source": "offline_cache"}, indent=2))


if __name__ == "__main__":
    main()
