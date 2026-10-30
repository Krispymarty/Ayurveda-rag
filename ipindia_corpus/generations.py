"""Build and atomically publish source-verified corpus generations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
from contextlib import closing, nullcontext
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .connectors import official_url
from .paths import archive_path, portable_archive_path
from .pipeline import Fetch, extract, structure_chunks
from .sync_runner import ROOT, _is_discovery
from .sync import Ledger
from .locking import file_lock

MAX_CHARS = 2000
MIN_CHARS = 80


def _jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _domain(source: str, title: str) -> str:
    if source == "tkdl":
        return "Traditional Knowledge"
    value = title.lower()
    for needle, label in (("patent", "Patent"), ("biodivers", "Biodiversity / ABS"), ("biological diversity", "Biodiversity / ABS"),
                          ("geographical indication", "GI"), ("drug", "AYUSH regulation"),
                          ("cosmetic", "AYUSH regulation"), ("trade mark", "Trademark"),
                          ("design", "Design")):
        if needle in value:
            return label
    if re.search(r"\bgi\b", value):
        return "GI"
    return "General IP" if source == "ipindia" else "India law"


def _known_title(source: str, url: str, extracted: str) -> str:
    if source != "indiacode":
        return extracted
    for marker, title in (
        ("bd_rules_2024", "Biological Diversity Rules, 2024"),
        ("GI+Rules.pdf", "Geographical Indications of Goods Rules, 2002"),
        ("/1392/", "The Patents Act, 1970"),
        ("/2046/", "The Biological Diversity Act, 2002"),
        ("/1981/", "The Geographical Indications of Goods Act, 1999"),
        ("/2409/", "The Drugs and Cosmetics Act, 1940"),
    ):
        if marker.lower() in url.lower():
            return title
    return extracted


def _metadata(row, source: str, title: str, text: str, root: Path) -> dict:
    url = row["canonical_url"]
    category = _domain(source, title + " " + url)
    role = "institutional_background" if source == "tkdl" else "legal_source"
    record_metadata = json.loads(row.get("metadata_json", "{}"))
    return {
        "document_id": row["canonical_id"], "canonical_id": row["canonical_id"],
        "source": source, "source_domain": url.split("/")[2].lower(),
        "authority": {"ipindia": "IP India / CGPDTM", "indiacode": "India Code / Government of India",
                      "tkdl": "TKDL public website / CSIR"}[source],
        "evidence_role": role, "title": title, "jurisdiction": "India",
        "legal_domain": category, "ip_domain": category,
        "document_type": "Public information page" if source == "tkdl" else ("Rules" if "rules" in title.lower() else "Act" if "act" in title.lower() else "Guidance"),
        "source_url": url, "canonical_url": url, "retrieved_at": row["retrieved_at"],
        "last_checked_at": row["last_checked_at"], "sha256": row["sha256"],
        "raw_path": portable_archive_path(root, archive_path(root, row["raw_path"])),
        "status": "unknown", "metadata_review_required": True,
        "publication_date": None, "effective_from": None, "effective_to": None,
        "version": row["verified_version"], "language": "hi" if sum("\u0900" <= c <= "\u097f" for c in text) > 100 else "en",
        "official_record_url": record_metadata.get("official_record_url"),
        "verified_identity": record_metadata.get("verified_identity"),
        "parent_identity": record_metadata.get("parent_identity"),
        "metadata_item_url": record_metadata.get("metadata_item_url"),
    }


def _split(text: str, limit: int = MAX_CHARS) -> list[str]:
    result, current = [], ""
    for word in re.findall(r"\S+\s*", text):
        while len(word) > limit:
            if current:
                result.append(current.strip())
                current = ""
            result.append(word[:limit].strip())
            word = word[limit:]
        if current and len(current) + len(word) > limit:
            result.append(current.strip())
            current = ""
        current += word
    if current.strip():
        result.append(current.strip())
    if len(result) > 1 and len(result[-1]) < MIN_CHARS and len(result[-2]) + len(result[-1]) + 1 <= limit:
        result[-2] += " " + result.pop()
    return result


def _rechunk(chunks: list[dict], document: dict) -> list[dict]:
    """Keep legal anchors while splitting long units and absorbing heading fragments."""
    chunks = [dict(chunk) for chunk in chunks]
    for index, item in enumerate(chunks[:-1]):
        body = re.sub(r"\s+", " ", item.get("text", "")).strip()
        heading = re.sub(r"\s+", " ", item.get("heading") or "").strip()
        stripped = re.sub(r"^(?:SECTION|RULE|CHAPTER)\s+[0-9A-Za-z]+\.?\s*", "", body)
        following = chunks[index + 1]
        same_provision = all(item.get(key) == following.get(key) for key in ("section", "rule"))
        same_page = item.get("page_start") == following.get("page_start") and item.get("page_end") == following.get("page_end")
        if heading and stripped == heading and stripped != body and same_provision and same_page:
            following["text"] = body + "\n" + following.get("text", "")
            if item.get("page_start") is not None:
                following["page_start"] = min(item["page_start"], following.get("page_start") or item["page_start"])
            item["text"] = ""
    grouped = []
    for chunk in chunks:
        body = re.sub(r"\s+", " ", chunk.get("text", "")).strip()
        if not body:
            continue
        item = dict(chunk)
        item["text"] = body
        if len(body) < MIN_CHARS and grouped:
            previous = grouped[-1]
            same_anchor = all(previous.get(k) == item.get(k) for k in ("section", "rule", "subsection", "page_start", "page_end"))
            if same_anchor and len(previous["text"]) + len(body) + 1 <= MAX_CHARS:
                previous["text"] += " " + body
                previous["page_end"] = item.get("page_end") or previous.get("page_end")
                continue
        grouped.append(item)
    result = []
    for item in grouped:
        for part in _split(item["text"]):
            record = {k: item.get(k) for k in ("chapter", "section", "subsection", "rule", "heading", "page_start", "page_end", "parser_version", "heading_origin", "heading_page", "heading_bbox")}
            content_hash = _digest(part.encode("utf-8"))
            record.update({"chunk_id": f"{document['document_id']}_{len(result):05d}_{content_hash[:8]}",
                           "document_id": document["document_id"], "source": document["source"],
                           "document_title": document["title"], "retrieved_at": document["retrieved_at"],
                           "authority": document["authority"],
                           "legal_domain": document["legal_domain"], "evidence_role": document["evidence_role"],
                           "source_url": document["source_url"], "raw_sha256": document["sha256"],
                           "status": "unknown", "text": part, "content_hash": content_hash,
                           "short_unit": len(part) < MIN_CHARS})
            result.append(record)
    return result


def accepted_rows(root: Path) -> list[dict]:
    """Migrate only verified previously published bytes, never a partial manifest."""
    ledger = Ledger(root / "data" / "manifests" / "manifest.sqlite", root / "data" / "raw")
    try:
        parent = root / "data" / "generations"
        pointer = parent / "ACTIVE"
        if not pointer.exists():
            pointer = parent / "CURRENT"  # migration of the existing validated release
        if pointer.exists():
            identifier = pointer.read_text(encoding="ascii").strip()
            if not identifier or any(char not in "0123456789TZ" for char in identifier):
                raise ValueError("invalid legacy generation pointer")
            manifest = json.loads((parent / identifier / "manifest.json").read_text(encoding="utf-8"))
            existing = {row[0] for row in ledger.db.execute("SELECT DISTINCT source_id FROM accepted_documents")}
            missing = {source for source, count in manifest["sources"].items() if count and source not in existing}
            if missing:
                from .indexer import generation
                _, documents, _ = generation(root, generation_id=identifier)
                grouped = defaultdict(list)
                for doc in documents:
                    if doc["source"] not in missing:
                        continue
                    row = ledger.db.execute("SELECT * FROM document_manifest WHERE canonical_id=?", (doc["document_id"],)).fetchone()
                    if not row:
                        raise ValueError("published document missing from ledger")
                    record = dict(row)
                    record.update(canonical_url=doc["source_url"], sha256=doc["sha256"],
                                  raw_path=doc["raw_path"], retrieved_at=doc["retrieved_at"],
                                  last_checked_at=doc.get("last_checked_at") or doc["retrieved_at"],
                                  verified_version=doc.get("version"), metadata_json=json.dumps({"title": doc["title"]}))
                    grouped[doc["source"]].append(record)
                with ledger.db:
                    for source, records in grouped.items():
                        ledger.accept(source, records, origin="legacy_verified_generation")
        return [json.loads(row[0]) for row in ledger.db.execute("SELECT record_json FROM accepted_documents ORDER BY source_id,canonical_id")]
    finally:
        ledger.close()


def build(root: Path = ROOT, *, repair_pdf=False) -> tuple[list[dict], list[dict], list[dict]]:
    root = Path(root)
    baseline = {doc["document_id"]: doc for doc in _jsonl(root / "data" / "documents" / "documents.jsonl") or []}
    old_chunks = defaultdict(list)
    for chunk in _jsonl(root / "data" / "chunks" / "chunks.jsonl") or []:
        old_chunks[chunk["document_id"]].append(chunk)
    connection = sqlite3.connect(root / "data" / "manifests" / "manifest.sqlite")
    connection.row_factory = sqlite3.Row
    documents, chunks, review = [], [], []
    try:
        rows = accepted_rows(root)
        for row in rows:
            source, url = row["source_id"], row["canonical_url"]
            if _is_discovery(source, url):
                continue
            official_url(source, url)
            raw_path = archive_path(root, row["raw_path"])
            if not raw_path.is_file():
                raise ValueError(f"missing archived bytes: {url}")
            raw = raw_path.read_bytes()
            if _digest(raw) != row["sha256"]:
                raise ValueError(f"raw hash mismatch: {url}")
            prior = baseline.get(row["canonical_id"])
            if prior and prior.get("sha256") == row["sha256"] and old_chunks.get(row["canonical_id"]):
                document = dict(prior)
                document["evidence_role"] = "institutional_background" if source == "tkdl" else "legal_source"
                document["last_checked_at"] = row["last_checked_at"]
                document["raw_path"] = portable_archive_path(root, raw_path)
                document["status"] = "unknown"
                source_chunks = old_chunks[row["canonical_id"]]
            else:
                kind = "application/pdf" if raw[:4] == b"%PDF" else "text/html"
                fetch = Fetch(url, raw, kind, row["retrieved_at"], raw_path)
                try:
                    title, text, _ = extract(fetch)
                except (ValueError, OSError, ImportError) as exc:
                    raise ValueError(f"cannot publish {url}: {exc}") from exc
                if not text.strip():
                    raise ValueError(f"empty accepted document: {url}")
                if not title or re.fullmatch(r"[A-Za-z]?\d{4}[-_ ]?\d+|[a-f0-9-]{16,}", title.strip(), re.I):
                    title = next((line.strip() for line in text.splitlines()
                                  if 12 <= len(line.strip()) <= 150 and not line.startswith("[Page ")), url.rsplit("/", 1)[-1])
                title = json.loads(row.get("metadata_json", "{}")).get("title") or _known_title(source, url, title)
                document = _metadata(row, source, title, text, root)
                source_chunks = structure_chunks(text, document)
            document["source_url"] = url
            document["canonical_url"] = url
            document["sha256"] = row["sha256"]
            document["raw_sha256"] = row["sha256"]
            # These identities come from the curated official paths and archived full Acts.
            if source == "ipindia":
                for marker, title in (("/acts/patent-act-1970/", "The Patents Act, 1970"),
                                      ("1_43_1_trade-marks-act.pdf", "The Trade Marks Act, 1999"),
                                      ("1_58_1_design_act_1_.pdf", "The Designs Act, 2000")):
                    if marker in url:
                        document["title"], document["document_type"] = title, "Act"
            if repair_pdf and raw.startswith(b"%PDF"):
                from .pdf_passages import pdf_passages, sidebar_headings
                _, text, _ = extract(Fetch(url, raw, "application/pdf", row["retrieved_at"], raw_path))
                source_chunks = sidebar_headings(pdf_passages(text, document), raw, document)
            official_url(source, document["source_url"])
            local_chunks = _rechunk(source_chunks, document)
            if not local_chunks:
                raise ValueError(f"no chunks for accepted document: {url}")
            if any(chunk["source_url"] != url or chunk["raw_sha256"] != row["sha256"] for chunk in local_chunks):
                raise ValueError(f"citation mismatch: {url}")
            documents.append(document)
            chunks.extend(local_chunks)
    finally:
        connection.close()
    if not documents or not chunks:
        raise ValueError("empty generation")
    if len({item["chunk_id"] for item in chunks}) != len(chunks):
        raise ValueError("duplicate chunk identifiers")
    return documents, chunks, review


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return _digest(path.read_bytes())


def publish(root: Path = ROOT, *, lock_held=False, repair_pdf=False, update_pointer=True) -> dict:
    lock = nullcontext() if lock_held else file_lock(Path(root) / "data" / "locks" / "maintenance.lock")
    with lock:
        with file_lock(Path(root) / "data" / "locks" / "publication.lock"):
            return _publish(root, repair_pdf=repair_pdf, update_pointer=update_pointer)


def _publish(root: Path = ROOT, *, repair_pdf=False, update_pointer=True) -> dict:
    documents, chunks, review = build(root, repair_pdf=repair_pdf)
    root = Path(root)
    parent = root / "data" / "generations"
    parent.mkdir(parents=True, exist_ok=True)
    generation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    staging = parent / f".{generation}.staging"
    final = parent / generation
    staging.mkdir()
    try:
        doc_hash = _write_jsonl(staging / "documents.jsonl", documents)
        chunk_hash = _write_jsonl(staging / "chunks.jsonl", chunks)
        manifest = {"generation_id": generation, "created_at": datetime.now(timezone.utc).isoformat(),
                    "documents": len(documents), "chunks": len(chunks),
                    "sources": {source: sum(doc["source"] == source for doc in documents) for source in ("ipindia", "indiacode", "tkdl")},
                    "documents_sha256": doc_hash, "chunks_sha256": chunk_hash, "review": review,
                    "source_snapshots": {row["source_id"]: row["snapshot_id"] for row in _snapshot_ids(root)},
                    "passage_parser": "pdf-pages-v4" if repair_pdf else "legacy-compatible"}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(staging, final)
        if update_pointer:
            pointer = parent / "CURRENT.tmp"
            pointer.write_text(generation, encoding="ascii")
            os.replace(pointer, parent / "CURRENT")
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _snapshot_ids(root: Path):
    with closing(sqlite3.connect(root / "data" / "manifests" / "manifest.sqlite")) as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute("SELECT source_id,max(snapshot_id) snapshot_id FROM source_snapshots GROUP BY source_id").fetchall()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-pdf", action="store_true")
    parser.add_argument("--candidate", action="store_true", help="Leave CURRENT/ACTIVE unchanged")
    args = parser.parse_args()
    print(json.dumps(publish(repair_pdf=args.repair_pdf, update_pointer=not args.candidate), indent=2))
