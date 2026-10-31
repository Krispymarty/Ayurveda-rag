"""Offline-safe corpus version ledger shared by future source connectors.

This module never crawls. Connectors must enforce robots, terms, and access rules
before passing verified public response bytes to ``observe``.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit
from .paths import archive_path, immutable_write

SOURCES = {
    "ipindia": "ipindia.gov.in",
    "tkdl": "tkdl.res.in",
    "indiacode": "indiacode.gov.in",
    "nba": "nbaindia.nic.in",
    "gazette": "egazette.gov.in",
}
EXTRA_OFFICIAL_HOSTS = {"indiacode": {"upload.indiacode.nic.in", "indiacode.nic.in", "www.indiacode.nic.in"}}
CHANGE_TYPES = {
    "NEW_DOCUMENT", "CONTENT_CHANGED", "METADATA_CHANGED", "NEW_VERSION",
    "SUPERSEDED", "URL_MOVED", "REMOVED_FROM_SOURCE", "UNCHANGED", "FAILED",
}
TKDL_RESTRICTED = re.compile(r"(?:login|logon|signin|authenticate|formulation|password|database)|(?:^|[/_.?&=\-])search(?:$|[/_.?&=\-])", re.I)


def public_tkdl_destination(url: str) -> bool:
    """Use the same bounded decode policy before fetching and certification."""
    try:
        parsed = urlsplit(url)
        destination = parsed.path + "?" + parsed.query
        for _ in range(8):
            decoded = unquote(destination)
            if decoded == destination:
                return not TKDL_RESTRICTED.search(destination)
            destination = decoded
        return False  # Excessive nested encoding is ambiguous, never public evidence.
    except ValueError:
        return False


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_url(source: str, url: str) -> None:
    if source not in SOURCES:
        raise ValueError("unknown source")
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    allowed = {SOURCES[source], "www." + SOURCES[source]} | EXTRA_OFFICIAL_HOSTS.get(source, set())
    if parsed.scheme != "https" or host not in allowed or parsed.port not in (None, 443) or parsed.username or parsed.password:
        raise ValueError("source URL is not on the configured official host")
    if "\\" in url:
        raise ValueError("ambiguous source URL")
    path = parsed.path
    for _ in range(8):
        decoded = unquote(path)
        if decoded == path:
            break
        path = decoded
    else:
        raise ValueError("ambiguous source path encoding")
    if "\\" in path or any(part in {".", ".."} for part in path.split("/")) or any(ord(c) < 32 or ord(c) == 127 for c in path):
        raise ValueError("ambiguous source path")
    if source == "tkdl" and not public_tkdl_destination(url):
        raise ValueError("restricted_tkdl_endpoint")


def capture_receipt(raw_root: Path, source: str, *, requested_url: str, final_url: str,
                    body: bytes, content_type: str, redirects: list[dict], etag=None,
                    last_modified=None, access_review=None) -> dict:
    """Archive raw bytes and one immutable observation; approval is never inferred."""
    validate_url(source, requested_url)
    validate_url(source, final_url)
    review = access_review or {}
    review_bytes = json.dumps(review, sort_keys=True, ensure_ascii=False).encode("utf-8")
    review_hash = sha256(review_bytes)
    review_relative = f"receipts/{source}/review_{review_hash}.json"
    immutable_write(Path(raw_root) / review_relative, review_bytes)
    digest = sha256(body)
    suffix = {"text/html": ".html", "application/xhtml+xml": ".html", "application/pdf": ".pdf",
              "application/json": ".json", "application/hal+json": ".json"}.get(content_type, ".bin")
    raw_relative = f"{source}/{digest}{suffix}"
    immutable_write(Path(raw_root) / raw_relative, body)
    receipt = {"schema_version": 1, "source": source, "requested_url": requested_url,
               "final_url": final_url, "redirect_chain": redirects, "retrieved_at": now(),
               "status": 200, "content_type": content_type, "byte_length": len(body),
               "sha256": digest, "raw_path": "data/raw/" + raw_relative,
               "etag": etag, "last_modified": last_modified,
               "access_review": {"approved": review.get("approved") is True,
                                 "reviewer": review.get("reviewer"), "reviewed_at": review.get("reviewed_at"),
                                 "sha256": review_hash, "raw_path": "data/raw/" + review_relative}}
    payload = json.dumps(receipt, sort_keys=True, ensure_ascii=False).encode("utf-8")
    receipt_hash = sha256(payload)
    relative = f"receipts/{source}/{receipt_hash}.json"
    immutable_write(Path(raw_root) / relative, payload)
    return {"receipt_path": "data/raw/" + relative, "receipt_sha256": receipt_hash}


def verify_capture_receipt(root: Path, record: dict, source: str, final_url: str) -> dict:
    payload = archive_path(root, record["receipt_path"]).read_bytes()
    if sha256(payload) != record["receipt_sha256"]:
        raise ValueError("capture_receipt_hash_mismatch")
    receipt = json.loads(payload)
    if (receipt.get("schema_version") != 1 or receipt["source"] != source
            or receipt["final_url"] != final_url or receipt["status"] != 200
            or receipt["content_type"] not in {"text/html", "application/xhtml+xml", "application/pdf", "application/json", "application/hal+json"}
            or type(receipt["byte_length"]) is not int or receipt["byte_length"] <= 0
            or not isinstance(receipt["redirect_chain"], list) or len(receipt["redirect_chain"]) > 4):
        raise ValueError("capture_receipt_identity_mismatch")
    for url in (receipt["requested_url"], final_url):
        validate_url(source, url)
    previous = receipt["requested_url"]
    for step in receipt["redirect_chain"]:
        if step["from_url"] != previous or step["status"] not in (301, 302, 303, 307, 308):
            raise ValueError("capture_redirect_chain_invalid")
        validate_url(source, step["to_url"])
        previous = step["to_url"]
    if previous != final_url:
        raise ValueError("capture_redirect_chain_incomplete")
    stamp = datetime.fromisoformat(receipt["retrieved_at"].replace("Z", "+00:00"))
    if stamp.utcoffset() != timedelta(0):
        raise ValueError("capture_timestamp_not_utc")
    raw = archive_path(root, receipt["raw_path"]).read_bytes()
    if len(raw) != receipt["byte_length"] or sha256(raw) != receipt["sha256"]:
        raise ValueError("capture_original_hash_mismatch")
    evidence = receipt["access_review"]
    review_raw = archive_path(root, evidence["raw_path"]).read_bytes()
    if sha256(review_raw) != evidence["sha256"]:
        raise ValueError("capture_access_review_hash_mismatch")
    review = json.loads(review_raw)
    if (review.get("approved") is not True or not review.get("reviewer")
            or evidence["approved"] is not True or evidence["reviewer"] != review["reviewer"]
            or evidence["reviewed_at"] != review.get("reviewed_at")):
        raise ValueError("capture_access_review_unapproved")
    from .access_review import verify_review
    verify_review(root, source, review)
    if not all(any(urlsplit(url).path.startswith(prefix) for prefix in review["permitted_prefixes"])
               for url in [receipt["requested_url"], *(s["to_url"] for s in receipt["redirect_chain"])]):
        raise ValueError("capture_path_not_approved")
    return receipt


class Ledger:
    def __init__(self, database: Path, raw_root: Path):
        self.database = Path(database)
        self.raw_root = Path(raw_root)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.database)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS source_registry (
          source_id TEXT PRIMARY KEY, domain TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 0,
          sync_frequency TEXT NOT NULL DEFAULT 'manual_until_review',
          last_successful_sync TEXT, next_sync TEXT
        );
        CREATE TABLE IF NOT EXISTS sync_runs (
          run_id INTEGER PRIMARY KEY AUTOINCREMENT, source_id TEXT NOT NULL,
          started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL,
          FOREIGN KEY(source_id) REFERENCES source_registry(source_id)
        );
        CREATE TABLE IF NOT EXISTS document_manifest (
          canonical_id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
          canonical_url TEXT NOT NULL UNIQUE, sha256 TEXT NOT NULL, metadata_sha256 TEXT NOT NULL,
          etag TEXT, last_modified TEXT, retrieved_at TEXT NOT NULL,
          last_checked_at TEXT NOT NULL, raw_path TEXT NOT NULL,
          verified_version TEXT, status TEXT NOT NULL DEFAULT 'unknown',
          metadata_review_required INTEGER NOT NULL DEFAULT 1,
          FOREIGN KEY(source_id) REFERENCES source_registry(source_id)
        );
        CREATE TABLE IF NOT EXISTS document_urls (
          url TEXT PRIMARY KEY, canonical_id TEXT NOT NULL,
          FOREIGN KEY(canonical_id) REFERENCES document_manifest(canonical_id)
        );
        CREATE TABLE IF NOT EXISTS document_versions (
          canonical_id TEXT NOT NULL, sha256 TEXT NOT NULL, raw_path TEXT NOT NULL,
          retrieved_at TEXT NOT NULL, verified_version TEXT,
          PRIMARY KEY(canonical_id, sha256),
          FOREIGN KEY(canonical_id) REFERENCES document_manifest(canonical_id)
        );
        CREATE TABLE IF NOT EXISTS capture_receipts (
          canonical_id TEXT NOT NULL, raw_sha256 TEXT NOT NULL,
          receipt_sha256 TEXT NOT NULL, receipt_path TEXT NOT NULL,
          PRIMARY KEY(canonical_id, receipt_sha256),
          FOREIGN KEY(canonical_id) REFERENCES document_manifest(canonical_id)
        );
        CREATE TABLE IF NOT EXISTS sync_changes (
          change_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL,
          canonical_id TEXT, url TEXT NOT NULL, change_type TEXT NOT NULL,
          previous_sha256 TEXT, current_sha256 TEXT, at TEXT NOT NULL, detail TEXT,
          FOREIGN KEY(run_id) REFERENCES sync_runs(run_id)
        );
        CREATE TABLE IF NOT EXISTS source_snapshots (
          snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT, source_id TEXT NOT NULL,
          run_id INTEGER, accepted_at TEXT NOT NULL, origin TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS accepted_documents (
          canonical_id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
          snapshot_id INTEGER NOT NULL, record_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS snapshot_documents (
          snapshot_id INTEGER NOT NULL, canonical_id TEXT NOT NULL, record_json TEXT NOT NULL,
          PRIMARY KEY(snapshot_id,canonical_id)
        );
        CREATE TABLE IF NOT EXISTS pipeline_events (
          event_id INTEGER PRIMARY KEY AUTOINCREMENT, source_id TEXT NOT NULL,
          stage TEXT NOT NULL, status TEXT NOT NULL, at TEXT NOT NULL,
          generation_id TEXT, detail TEXT NOT NULL DEFAULT ''
        );
        """)
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(source_registry)")}
        for column in ("next_check", "next_discovery", "last_successful_discovery", "last_successful_publication"):
            if column not in columns:
                self.db.execute(f"ALTER TABLE source_registry ADD COLUMN {column} TEXT")
        if "metadata_json" not in {row[1] for row in self.db.execute("PRAGMA table_info(document_manifest)")}:
            self.db.execute("ALTER TABLE document_manifest ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'")
        if "discovery" not in {row[1] for row in self.db.execute("PRAGMA table_info(sync_runs)")}:
            self.db.execute("ALTER TABLE sync_runs ADD COLUMN discovery INTEGER NOT NULL DEFAULT 0")
        with self.db:
            for source, domain in SOURCES.items():
                self.db.execute("INSERT OR IGNORE INTO source_registry(source_id,domain) VALUES (?,?)", (source, domain))
                self.db.execute("UPDATE source_registry SET domain=? WHERE source_id=?", (domain, source))

    def recover_abandoned(self, source: str):
        """Call only while owning the source worker lock."""
        with self.db:
            runs = self.db.execute("SELECT run_id FROM sync_runs WHERE source_id=? AND status='running'", (source,)).fetchall()
            self.db.execute("UPDATE sync_runs SET status='failed',finished_at=? WHERE source_id=? AND status='running'", (now(), source))
        if runs:
            self.event(source, "capture", "failed", "interrupted_worker_recovered; checkpoints and raw versions retained")

    def close(self) -> None:
        self.db.close()

    def start(self, source: str, *, discovery: bool = False) -> int:
        if source not in SOURCES:
            raise ValueError("unknown source")
        with self.db:
            cursor = self.db.execute("INSERT INTO sync_runs(source_id,started_at,status,discovery) VALUES (?,?,?,?)", (source, now(), "running", int(discovery)))
        return int(cursor.lastrowid)

    def _run_source(self, run_id: int) -> str:
        row = self.db.execute("SELECT source_id FROM sync_runs WHERE run_id=? AND status='running'", (run_id,)).fetchone()
        if row is None:
            raise ValueError("run is missing or closed")
        return row["source_id"]

    def _change(self, run_id: int, canonical_id: str | None, url: str, kind: str,
                previous: str | None = None, current: str | None = None, detail: str = "") -> None:
        if kind not in CHANGE_TYPES:
            raise ValueError("unknown change type")
        self.db.execute("INSERT INTO sync_changes(run_id,canonical_id,url,change_type,previous_sha256,current_sha256,at,detail) VALUES (?,?,?,?,?,?,?,?)",
                        (run_id, canonical_id, url, kind, previous, current, now(), detail))

    def observe(self, run_id: int, url: str, data: bytes, *, content_type: str = "application/octet-stream",
                etag: str | None = None, last_modified: str | None = None,
                metadata: dict | None = None, verified_version: str | None = None,
                canonical_id_hint: str | None = None, capture: dict | None = None) -> str:
        source = self._run_source(run_id)
        validate_url(source, url)
        if not data:
            raise ValueError("empty response must be recorded as a failure")
        raw_hash = sha256(data)
        if capture is not None:
            receipt = verify_capture_receipt(self.raw_root.parent.parent, capture, source, url)
            if receipt["sha256"] != raw_hash or receipt["content_type"] != content_type:
                raise ValueError("capture_observation_mismatch")
        metadata_hash = sha256(json.dumps(metadata or {}, sort_keys=True, ensure_ascii=False).encode())
        suffix = {"application/pdf": ".pdf", "text/html": ".html", "application/xhtml+xml": ".html",
                  "application/json": ".json", "application/hal+json": ".json"}.get(content_type, ".bin")
        raw_path = self.raw_root / source / f"{raw_hash}{suffix}"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        immutable_write(raw_path, data)
        checked_at = now()
        row = self.db.execute("SELECT m.* FROM document_urls u JOIN document_manifest m USING(canonical_id) WHERE u.url=?", (url,)).fetchone()
        if row is None and canonical_id_hint:
            row = self.db.execute("SELECT * FROM document_manifest WHERE canonical_id=? AND source_id=?", (canonical_id_hint, source)).fetchone()
        with self.db:
            if row is None:
                canonical_id = sha256(f"{source}\n{url}".encode())[:24]
                self.db.execute("INSERT INTO document_manifest(canonical_id,source_id,canonical_url,sha256,metadata_sha256,etag,last_modified,retrieved_at,last_checked_at,raw_path,verified_version) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                                (canonical_id, source, url, raw_hash, metadata_hash, etag, last_modified, checked_at, checked_at, str(raw_path), verified_version))
                self.db.execute("INSERT INTO document_urls(url,canonical_id) VALUES (?,?)", (url, canonical_id))
                self.db.execute("INSERT INTO document_versions(canonical_id,sha256,raw_path,retrieved_at,verified_version) VALUES (?,?,?,?,?)",
                                (canonical_id, raw_hash, str(raw_path), checked_at, verified_version))
                kind = "NEW_DOCUMENT"
                previous = None
            else:
                canonical_id = row["canonical_id"]
                previous = row["sha256"]
                moved = url != row["canonical_url"] and self.db.execute("SELECT 1 FROM document_urls WHERE url=?", (url,)).fetchone() is None
                if moved:
                    self.db.execute("INSERT INTO document_urls(url,canonical_id) VALUES (?,?)", (url, canonical_id))
                    self.db.execute("UPDATE document_manifest SET canonical_url=? WHERE canonical_id=?", (url, canonical_id))
                if previous != raw_hash:
                    kind = "NEW_VERSION" if verified_version and verified_version != row["verified_version"] else "CONTENT_CHANGED"
                    self.db.execute("INSERT OR IGNORE INTO document_versions(canonical_id,sha256,raw_path,retrieved_at,verified_version) VALUES (?,?,?,?,?)",
                                    (canonical_id, raw_hash, str(raw_path), checked_at, verified_version))
                    self.db.execute("UPDATE document_manifest SET sha256=?,metadata_sha256=?,etag=?,last_modified=?,retrieved_at=?,last_checked_at=?,raw_path=?,verified_version=?,metadata_review_required=1 WHERE canonical_id=?",
                                    (raw_hash, metadata_hash, etag, last_modified, checked_at, checked_at, str(raw_path), verified_version, canonical_id))
                elif metadata_hash != row["metadata_sha256"]:
                    kind = "METADATA_CHANGED"
                    self.db.execute("UPDATE document_manifest SET metadata_sha256=?,etag=?,last_modified=?,last_checked_at=?,metadata_review_required=1 WHERE canonical_id=?",
                                    (metadata_hash, etag, last_modified, checked_at, canonical_id))
                else:
                    kind = "UNCHANGED"
                    self.db.execute("UPDATE document_manifest SET etag=?,last_modified=?,last_checked_at=? WHERE canonical_id=?",
                                    (etag or row["etag"], last_modified or row["last_modified"], checked_at, canonical_id))
                if moved:
                    self._change(run_id, canonical_id, url, "URL_MOVED", previous, raw_hash, "confirmed by canonical_id_hint")
            self._change(run_id, canonical_id, url, kind, previous, raw_hash)
            self.db.execute("UPDATE document_manifest SET metadata_json=? WHERE canonical_id=?",
                            (json.dumps(metadata or {}, ensure_ascii=False), canonical_id))
            if capture is not None:
                self.db.execute("INSERT OR IGNORE INTO capture_receipts VALUES (?,?,?,?)",
                                (canonical_id, raw_hash, capture["receipt_sha256"], capture["receipt_path"]))
        return kind

    def accept(self, source: str, records: list[dict], *, run_id: int | None = None, origin: str = "sync") -> int:
        """Accept a complete source view, retaining prior accepted documents and history."""
        cursor = self.db.execute("INSERT INTO source_snapshots(source_id,run_id,accepted_at,origin) VALUES (?,?,?,?)",
                                 (source, run_id, now(), origin))
        snapshot = cursor.lastrowid
        merged = {row["canonical_id"]: json.loads(row["record_json"]) for row in self.db.execute(
            "SELECT * FROM accepted_documents WHERE source_id=?", (source,))}
        merged.update({row["canonical_id"]: dict(row) for row in records})
        for identifier, record in merged.items():
            payload = json.dumps(record, ensure_ascii=False)
            self.db.execute("INSERT INTO snapshot_documents VALUES (?,?,?)", (snapshot, identifier, payload))
            self.db.execute("INSERT INTO accepted_documents VALUES (?,?,?,?) ON CONFLICT(canonical_id) DO UPDATE SET snapshot_id=excluded.snapshot_id,record_json=excluded.record_json",
                            (identifier, source, snapshot, payload))
        return snapshot

    def event(self, source: str, stage: str, status: str, detail: str = "", generation_id: str | None = None):
        with self.db:
            self.db.execute("INSERT INTO pipeline_events(source_id,stage,status,at,generation_id,detail) VALUES (?,?,?,?,?,?)",
                            (source, stage, status, now(), generation_id, detail))
            if stage == "index" and status == "success":
                self.db.execute("UPDATE source_registry SET last_successful_publication=? WHERE source_id=?", (now(), source))

    def failed(self, run_id: int, url: str, reason: str) -> None:
        source = self._run_source(run_id)
        validate_url(source, url)
        with self.db:
            self._change(run_id, None, url, "FAILED", detail=reason)

    def checked(self, run_id: int, url: str, *, etag: str | None = None,
                last_modified: str | None = None) -> None:
        """Record a 304 without replacing the archived bytes or retrieval time."""
        source = self._run_source(run_id)
        validate_url(source, url)
        row = self.db.execute("SELECT canonical_id FROM document_urls WHERE url=?", (url,)).fetchone()
        if row is None:
            raise ValueError("304 for an unarchived URL")
        with self.db:
            self.db.execute("UPDATE document_manifest SET last_checked_at=?,etag=COALESCE(?,etag),last_modified=COALESCE(?,last_modified) WHERE canonical_id=?",
                            (now(), etag, last_modified, row["canonical_id"]))
            self._change(run_id, row["canonical_id"], url, "UNCHANGED", detail="HTTP 304")

    def removed_from_source(self, run_id: int, url: str, *, verified_discovery: bool) -> None:
        source = self._run_source(run_id)
        validate_url(source, url)
        if not verified_discovery:
            raise ValueError("removal requires a complete verified discovery pass")
        row = self.db.execute("SELECT canonical_id FROM document_urls WHERE url=?", (url,)).fetchone()
        with self.db:
            self._change(run_id, row["canonical_id"] if row else None, url, "REMOVED_FROM_SOURCE", detail="historical snapshot retained; manual status review required")

    def finish(self, run_id: int, status: str = "success", report_dir: Path | None = None,
               *, accepted_urls: list[str] | None = None, discovery: bool = False) -> dict:
        source = self._run_source(run_id)
        if status == "success" and self.db.execute("SELECT 1 FROM sync_changes WHERE run_id=? AND change_type='FAILED'", (run_id,)).fetchone():
            status = "partial"
        with self.db:
            self.db.execute("UPDATE sync_runs SET finished_at=?,status=? WHERE run_id=?", (now(), status, run_id))
            if status == "success":
                if accepted_urls is None:
                    records = self.db.execute("SELECT DISTINCT m.* FROM document_manifest m JOIN sync_changes c USING(canonical_id) WHERE c.run_id=? AND m.source_id=?", (run_id, source)).fetchall()
                else:
                    records = []
                    for url in accepted_urls:
                        record = self.db.execute("SELECT m.* FROM document_manifest m JOIN document_urls u USING(canonical_id) WHERE u.url=? AND m.source_id=?", (url, source)).fetchone()
                        if record is None:
                            raise ValueError("cannot accept missing completed URL")
                        records.append(record)
                self.accept(source, records, run_id=run_id)
                self.db.execute("UPDATE source_registry SET last_successful_sync=?,next_sync=? WHERE source_id=?",
                                (now(), (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(), source))
                self.db.execute("UPDATE source_registry SET next_check=? WHERE source_id=?", ((datetime.now(timezone.utc) + timedelta(days=1)).isoformat(), source))
                if discovery:
                    self.db.execute("UPDATE source_registry SET last_successful_discovery=?,next_discovery=? WHERE source_id=?",
                                    (now(), (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(), source))
        counts = {row["change_type"]: row["n"] for row in self.db.execute("SELECT change_type,COUNT(*) n FROM sync_changes WHERE run_id=? GROUP BY change_type", (run_id,))}
        result = {"run_id": run_id, "source": source, "status": status, "changes": counts}
        if report_dir is not None:
            report_dir = Path(report_dir)
            report_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            stem = f"sync_{stamp}_{run_id}"
            (report_dir / f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            lines = [f"# {source} sync run {run_id}", "", f"Status: {status}", "", "| Change | Count |", "|---|---:|"]
            lines.extend(f"| {kind} | {count} |" for kind, count in sorted(counts.items()))
            (report_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return result
