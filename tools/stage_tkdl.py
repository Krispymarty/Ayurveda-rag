"""Stage the audited TKDL public Markdown ZIP without promoting it to legal RAG."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "tkdl_corpus_final.zip"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    archive_bytes = ARCHIVE.read_bytes()
    archive_hash = sha256(archive_bytes)
    raw_path = ROOT / "data" / "raw" / "tkdl" / f"{archive_hash}.zip"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        if sha256(raw_path.read_bytes()) != archive_hash:
            raise ValueError("immutable TKDL archive collision")
    else:
        raw_path.write_bytes(archive_bytes)
    candidates = []
    by_text_hash = {}
    with ZipFile(ARCHIVE) as archive:
        names = set(archive.namelist())
        lookup = {name.casefold(): name for name in names}
        metadata = []
        for name in sorted(n for n in names if n.endswith("/metadata.json")):
            metadata.extend(json.loads(archive.read(name).decode("utf-8-sig")))
        for item in metadata:
            url = item.get("url", "")
            if urlsplit(url).scheme != "https" or urlsplit(url).hostname != "tkdl.res.in":
                continue
            expected = "tkdl_corpus/" + item.get("file", "").replace("\\", "/")
            member = expected if expected in names else lookup.get(expected.casefold())
            if not member:
                continue
            body = archive.read(member)
            text_hash = sha256(body)
            if text_hash in by_text_hash:
                by_text_hash[text_hash]["alternate_urls"].append(url)
                continue
            document_id = sha256(url.encode())[:20]
            record = {
                "document_id": document_id, "canonical_id": document_id,
                "source": "tkdl", "source_domain": "tkdl.res.in",
                "authority": "TKDL public website (unverified extraction)", "jurisdiction": "India",
                "legal_domain": "Traditional Knowledge", "document_type": "Public information page",
                "title": item.get("title", ""), "language": "en",
                "publication_date": None, "effective_from": None, "effective_to": None,
                "notification_number": None, "version": None, "status": "unknown",
                "amends": [], "amended_by": [], "supersedes": [], "superseded_by": [],
                "canonical_url": url, "source_url": url, "download_url": url,
                "alternate_urls": [], "retrieved_at": None, "last_checked_at": None,
                "etag": None, "last_modified": None, "sha256": text_hash,
                "official_source": False, "topics": [], "raw_path": str(raw_path.resolve()),
                "zip_member": member, "source_snapshot_type": "transformed_markdown_only",
                "metadata_review_required": True, "security_flags": [],
            }
            by_text_hash[text_hash] = record
            candidates.append(record)
    target = ROOT / "data" / "documents" / "tkdl_candidates.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in candidates), encoding="utf-8")
    print(json.dumps({"metadata_rows": len(metadata), "unique_staged_pages": len(candidates), "promoted_to_retrieval": 0}))


if __name__ == "__main__":
    main()
