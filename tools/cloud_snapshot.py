"""Export only the protected retrieval release; restore into an empty isolated DB.

Credentials are read from environment variables and never written to snapshots.
Archive files must be provisioned separately at the same portable paths.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

from ipindia_corpus.indexer import DIMENSIONS, MODEL_ID, SCHEMA, generation, model_revision
from ipindia_corpus.sync_runner import ROOT

ACTIVE = "20261014T204940358650Z"
ROLLBACK = "20261014T190100599320Z"
GENERATIONS = (ACTIVE, ROLLBACK)
STATE_KEYS = ("active_generation", "ranking_version", "retrieval_rollback")
TABLES = {
    "corpus_documents": ("generation_id", "document_id", "metadata"),
    "corpus_chunks": ("generation_id", "chunk_id", "document_id", "content_hash", "text", "metadata"),
    "index_generations": ("generation_id", "model_revision", "documents", "chunks", "manifest_sha256", "activated_at"),
    "embedding_cache": ("content_hash", "model_revision", "embedding"),
    "app_state": ("key", "value"),
}
VERSION = "ayurip-retrieval-snapshot-v1"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def parse(raw):
    return json.loads(raw, object_pairs_hook=_unique, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))


def connection_identity(dsn):
    """No credentials in the returned endpoint fingerprint input."""
    from psycopg.conninfo import conninfo_to_dict
    info = conninfo_to_dict(dsn)
    host = info.get("host", "localhost").lower()
    if host in {"127.0.0.1", "::1", "localhost"}:
        host = "localhost"
    if "," in host or not info.get("dbname"):
        raise ValueError("single_explicit_database_required")
    if host != "localhost" and info.get("sslmode") != "verify-full":
        raise ValueError("remote_database_requires_ssl_verify_full")
    return {"host": host, "port": str(info.get("port", "5432")), "database": info["dbname"]}


def endpoint_hash(dsn):
    return sha(canonical(connection_identity(dsn)))


def validate_state(rows):
    if len(rows) != 3 or {r["key"] for r in rows} != set(STATE_KEYS):
        raise ValueError("unexpected_release_state_keys")
    state = {r["key"]: r["value"] for r in rows}
    if state["active_generation"] != ACTIVE or state["ranking_version"] != "v2":
        raise ValueError("protected_active_binding_mismatch")
    if parse(state["retrieval_rollback"]) != {"generation_id": ROLLBACK, "ranking_version": "v1"}:
        raise ValueError("protected_rollback_binding_mismatch")
    return state


def local_evidence(root):
    root = Path(root)
    revision = model_revision(root)
    evidence = {}
    for gid in GENERATIONS:
        _, docs, chunks = generation(root, generation_id=gid)
        if any(d["source"] != "ipindia" or d["status"] != "unknown" for d in docs):
            raise ValueError("snapshot_must_be_archived_ipindia_unknown_status")
        folder = root / "data/generations" / gid
        evidence[gid] = {"documents": {d["document_id"]: d for d in docs},
                         "chunks": {c["chunk_id"]: c for c in chunks},
                         "files": {name: sha((folder / name).read_bytes()) for name in ("manifest.json", "documents.jsonl", "chunks.jsonl")}}
    return revision, evidence


def validate_rows(rows, revision, evidence):
    if set(rows) != set(TABLES):
        raise ValueError("unexpected_snapshot_tables")
    for table, fields in TABLES.items():
        if any(set(row) != set(fields) for row in rows[table]):
            raise ValueError("unexpected_snapshot_columns")
    validate_state(rows["app_state"])
    docs = {(r["generation_id"], r["document_id"]): r["metadata"] for r in rows["corpus_documents"]}
    chunks = {(r["generation_id"], r["chunk_id"]): r for r in rows["corpus_chunks"]}
    if len(docs) != len(rows["corpus_documents"]) or len(chunks) != len(rows["corpus_chunks"]):
        raise ValueError("duplicate_snapshot_identity")
    expected_docs = {(gid, key): value for gid, item in evidence.items() for key, value in item["documents"].items()}
    expected_chunks = {(gid, key): value for gid, item in evidence.items() for key, value in item["chunks"].items()}
    if docs != expected_docs or set(chunks) != set(expected_chunks):
        raise ValueError("database_archive_document_or_chunk_mismatch")
    hashes = set()
    for key, row in chunks.items():
        metadata = expected_chunks[key]
        if (row["metadata"] != metadata or row["text"] != metadata["text"]
                or row["document_id"] != metadata["document_id"] or row["content_hash"] != metadata["content_hash"]
                or sha(row["text"].encode("utf-8")) != row["content_hash"]):
            raise ValueError("database_archive_passage_mismatch")
        hashes.add(row["content_hash"])
    indexed = {r["generation_id"]: r for r in rows["index_generations"]}
    if len(indexed) != 2 or set(indexed) != set(GENERATIONS):
        raise ValueError("index_generation_coverage_mismatch")
    for gid, entry in indexed.items():
        expected = evidence[gid]
        if (entry["model_revision"] != revision or entry["documents"] != len(expected["documents"])
                or entry["chunks"] != len(expected["chunks"]) or entry["manifest_sha256"] != expected["files"]["manifest.json"]):
            raise ValueError("index_generation_binding_mismatch")
        if datetime.fromisoformat(entry["activated_at"]).tzinfo is None:
            raise ValueError("index_timestamp_timezone_required")
    embeddings = rows["embedding_cache"]
    if len(embeddings) != len(hashes) or {e["content_hash"] for e in embeddings} != hashes:
        raise ValueError("embedding_coverage_mismatch")
    for entry in embeddings:
        if entry["model_revision"] != revision:
            raise ValueError("embedding_model_binding_mismatch")
        vector = parse(entry["embedding"])
        if (not isinstance(vector, list) or len(vector) != DIMENSIONS
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in vector)):
            raise ValueError("invalid_embedding_dimensions_or_values")


def _read_tables(db, revision):
    result = {}
    for table in ("corpus_documents", "corpus_chunks", "index_generations"):
        fields = TABLES[table]
        selected = ",".join(fields)
        records = db.execute(f"SELECT {selected} FROM {table} WHERE generation_id=ANY(%s) ORDER BY 1,2", (list(GENERATIONS),)).fetchall()
        result[table] = [dict(zip(fields, record)) for record in records]
    for row in result["index_generations"]:
        if isinstance(row["activated_at"], datetime):
            row["activated_at"] = row["activated_at"].astimezone(timezone.utc).isoformat()
    records = db.execute("SELECT e.content_hash,e.model_revision,e.embedding::text FROM embedding_cache e WHERE e.model_revision=%s AND EXISTS (SELECT 1 FROM corpus_chunks c WHERE c.content_hash=e.content_hash AND c.generation_id=ANY(%s)) ORDER BY e.content_hash,e.model_revision", (revision, list(GENERATIONS))).fetchall()
    result["embedding_cache"] = [dict(zip(TABLES["embedding_cache"], row)) for row in records]
    result["app_state"] = [dict(zip(TABLES["app_state"], row)) for row in db.execute("SELECT key,value FROM app_state WHERE key=ANY(%s) ORDER BY key", (list(STATE_KEYS),)).fetchall()]
    return result


def encode_table(rows):
    return b"".join(canonical(row) + b"\n" for row in rows)


def export_snapshot(output, root=ROOT):
    import psycopg
    dsn = os.environ.get("SOURCE_DATABASE_URL")
    if not dsn:
        raise ValueError("SOURCE_DATABASE_URL_required")
    endpoint = endpoint_hash(dsn)
    revision, evidence = local_evidence(root)
    output = Path(output)
    if output.exists():
        raise ValueError("snapshot_output_must_be_new")
    with psycopg.connect(dsn, connect_timeout=10) as db:
        db.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        db.execute("SET LOCAL search_path = public")
        rows = _read_tables(db, revision)
        validate_rows(rows, revision, evidence)
    output.mkdir(parents=True)
    files = {}
    for table, entries in rows.items():
        raw = encode_table(entries)
        with (output / f"{table}.jsonl").open("xb") as handle:
            handle.write(raw)
        files[table] = {"sha256": sha(raw), "rows": len(entries), "bytes": len(raw)}
    manifest = {"version": VERSION, "source_endpoint_sha256": endpoint, "model_id": MODEL_ID,
                "model_revision": revision, "dimensions": DIMENSIONS,
                "generation_files": {gid: item["files"] for gid, item in evidence.items()},
                "tables": files, "active_generation": ACTIVE, "ranking_version": "v2",
                "rollback_generation": ROLLBACK, "rollback_ranker": "v1"}
    with (output / "manifest.json").open("xb") as handle:
        handle.write(canonical(manifest))
    return {"status": "exported", "manifest_sha256": sha(canonical(manifest)), "tables": files,
            "active_generation": ACTIVE, "rollback_generation": ROLLBACK}


def load_snapshot(folder, root=ROOT):
    folder = Path(folder).resolve()
    manifest = parse((folder / "manifest.json").read_bytes())
    revision, evidence = local_evidence(root)
    if (manifest.get("version") != VERSION or manifest.get("model_id") != MODEL_ID
            or manifest.get("model_revision") != revision or manifest.get("dimensions") != DIMENSIONS
            or manifest.get("active_generation") != ACTIVE or manifest.get("ranking_version") != "v2"
            or manifest.get("rollback_generation") != ROLLBACK or manifest.get("rollback_ranker") != "v1"
            or manifest.get("generation_files") != {gid: item["files"] for gid, item in evidence.items()}
            or set(manifest.get("tables", {})) != set(TABLES)
            or not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("source_endpoint_sha256", "")))):
        raise ValueError("snapshot_release_binding_mismatch")
    rows = {}
    for table in TABLES:
        path = (folder / f"{table}.jsonl").resolve()
        if not path.is_relative_to(folder):
            raise ValueError("snapshot_file_containment_failure")
        raw = path.read_bytes()
        expected = manifest["tables"][table]
        if len(raw) != expected["bytes"] or sha(raw) != expected["sha256"]:
            raise ValueError("snapshot_file_integrity_failure")
        rows[table] = [parse(line) for line in raw.decode("utf-8").splitlines()]
        if len(rows[table]) != expected["rows"]:
            raise ValueError("snapshot_row_count_failure")
    validate_rows(rows, revision, evidence)
    return manifest, rows, evidence


def restore_snapshot(folder, *, allow_empty_target=False, root=ROOT):
    import psycopg
    from psycopg.types.json import Jsonb
    if not allow_empty_target:
        raise ValueError("explicit_empty_target_authorization_required")
    dsn = os.environ.get("TARGET_DATABASE_URL")
    if not dsn:
        raise ValueError("TARGET_DATABASE_URL_required")
    target = endpoint_hash(dsn)
    manifest, rows, evidence = load_snapshot(folder, root)
    source = os.environ.get("SOURCE_DATABASE_URL")
    if target == manifest["source_endpoint_sha256"] or (source and target == endpoint_hash(source)):
        raise ValueError("restore_into_source_database_forbidden")
    with psycopg.connect(dsn, connect_timeout=10) as db:
        db.execute("SET LOCAL search_path = public")
        # This is intentionally stricter than an empty corpus: no existing user tables.
        existing = db.execute("SELECT tablename FROM pg_tables WHERE schemaname='public'").fetchall()
        if existing:
            raise ValueError("target_database_must_have_no_public_tables")
        db.execute(SCHEMA)
        for table in ("corpus_documents", "corpus_chunks", "index_generations", "embedding_cache", "app_state"):
            fields = TABLES[table]
            casts = ["%s::vector" if key == "embedding" else "%s" for key in fields]
            sql = f"INSERT INTO {table} ({','.join(fields)}) VALUES ({','.join(casts)})"
            with db.cursor() as cursor:
                cursor.executemany(sql, [[Jsonb(row[key]) if key == "metadata" else row[key] for key in fields] for row in rows[table]])
        observed = _read_tables(db, manifest["model_revision"])
        validate_rows(observed, manifest["model_revision"], evidence)
        for table in TABLES:
            if sha(encode_table(observed[table])) != manifest["tables"][table]["sha256"]:
                raise ValueError("restored_table_digest_mismatch")
    return {"status": "restored_verified", "active_generation": ACTIVE, "ranking_version": "v2",
            "rollback_generation": ROLLBACK, "table_counts": {key: len(value) for key, value in rows.items()},
            "source_database_modified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "verify", "restore"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--allow-empty-target", action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "export":
            if not args.output:
                raise ValueError("output_required")
            result = export_snapshot(args.output, args.root)
        elif args.action == "verify":
            if not args.snapshot:
                raise ValueError("snapshot_required")
            manifest, rows, _ = load_snapshot(args.snapshot, args.root)
            result = {"status": "verified", "table_counts": {k: len(v) for k, v in rows.items()}, "model_revision": manifest["model_revision"]}
        else:
            if not args.snapshot:
                raise ValueError("snapshot_required")
            result = restore_snapshot(args.snapshot, allow_empty_target=args.allow_empty_target, root=args.root)
        print(json.dumps(result, ensure_ascii=True))
        return 0
    except Exception as exc:
        # Driver exception messages can contain endpoint/user/password details.
        reason = str(exc) if type(exc) is ValueError and re.fullmatch(r"[A-Za-z0-9_]+", str(exc)) else "snapshot_operation_failed"
        print(json.dumps({"status": "failed", "reason": reason, "error_class": type(exc).__name__}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
