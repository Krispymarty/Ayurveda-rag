"""Load a validated generation into PostgreSQL/pgvector and switch it atomically."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .connectors import official_url
from .paths import ArchiveResolver
from .sync_runner import ROOT
from .locking import file_lock
from .sync import Ledger

MODEL_ID = "intfloat/multilingual-e5-small"
DIMENSIONS = 384
SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS corpus_documents (
    generation_id text NOT NULL, document_id text NOT NULL, metadata jsonb NOT NULL,
    PRIMARY KEY (generation_id, document_id)
);
CREATE TABLE IF NOT EXISTS corpus_chunks (
    generation_id text NOT NULL, chunk_id text NOT NULL, document_id text NOT NULL,
    content_hash text NOT NULL, text text NOT NULL, metadata jsonb NOT NULL,
    lexical tsvector GENERATED ALWAYS AS (to_tsvector('simple', coalesce(metadata->>'document_title','') || ' ' || text)) STORED,
    PRIMARY KEY (generation_id, chunk_id),
    FOREIGN KEY (generation_id, document_id) REFERENCES corpus_documents(generation_id, document_id)
);
CREATE INDEX IF NOT EXISTS corpus_chunks_lexical_idx ON corpus_chunks USING gin(lexical);
CREATE INDEX IF NOT EXISTS corpus_chunks_source_idx ON corpus_chunks(generation_id, (metadata->>'source'));
CREATE INDEX IF NOT EXISTS corpus_chunks_document_idx ON corpus_chunks(generation_id,document_id);
ALTER TABLE corpus_chunks ADD COLUMN IF NOT EXISTS body_lexical tsvector
  GENERATED ALWAYS AS (to_tsvector('english',text)) STORED;
ALTER TABLE corpus_chunks ADD COLUMN IF NOT EXISTS heading_lexical tsvector
  GENERATED ALWAYS AS (to_tsvector('english',coalesce(metadata->>'heading','') || ' ' || left(text,350))) STORED;
CREATE INDEX IF NOT EXISTS corpus_chunks_body_idx ON corpus_chunks USING gin(body_lexical);
CREATE TABLE IF NOT EXISTS embedding_cache (
    content_hash text NOT NULL, model_revision text NOT NULL, embedding vector(384) NOT NULL,
    PRIMARY KEY (content_hash, model_revision)
);
CREATE TABLE IF NOT EXISTS app_state (key text PRIMARY KEY, value text NOT NULL);
CREATE TABLE IF NOT EXISTS index_generations (
  generation_id text PRIMARY KEY, model_revision text NOT NULL,
  documents integer NOT NULL, chunks integer NOT NULL, manifest_sha256 text NOT NULL,
  activated_at timestamptz NOT NULL DEFAULT now()
);
"""


def model_revision(root: Path = ROOT, *, resolve: bool = False) -> str:
    path = Path(root) / "data" / "model.lock.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("model_id") != MODEL_ID or not data.get("revision"):
            raise ValueError("invalid model lock")
        return data["revision"]
    if not resolve:
        raise ValueError("model lock missing; run the indexer first")
    from huggingface_hub import HfApi
    revision = HfApi().model_info(MODEL_ID).sha
    if not revision:
        raise ValueError("could not resolve model revision")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"model_id": MODEL_ID, "revision": revision}, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return revision


def _verify_archive(doc: dict, resolver: ArchiveResolver) -> None:
    if hashlib.sha256(resolver.resolve(doc["raw_path"]).read_bytes()).hexdigest() != doc["sha256"]:
        raise ValueError("document archive hash mismatch")


def generation(root: Path = ROOT, *, generation_id: str | None = None, raw_workers: int = 4) -> tuple[str, list[dict], list[dict]]:
    if type(raw_workers) is not int or not 1 <= raw_workers <= 4:
        raise ValueError("raw verification workers must be between one and four")
    parent = Path(root) / "data" / "generations"
    generation_id = generation_id or (parent / "CURRENT").read_text(encoding="ascii").strip()
    if not generation_id or any(char not in "0123456789TZ" for char in generation_id):
        raise ValueError("invalid generation pointer")
    directory = parent / generation_id
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["generation_id"] != generation_id:
        raise ValueError("generation ID mismatch")
    files = ("documents.jsonl", "chunks.jsonl")
    captured = []
    for file, key in zip(files, ("documents_sha256", "chunks_sha256")):
        raw = (directory / file).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest[key]:
            raise ValueError(f"generation file hash mismatch: {file}")
        captured.append([json.loads(line) for line in raw.decode("utf-8").splitlines() if line])
    documents, chunks = captured
    if len(documents) != manifest["documents"] or len(chunks) != manifest["chunks"]:
        raise ValueError("generation counts mismatch")
    by_id = {doc["document_id"]: doc for doc in documents}
    if len(by_id) != len(documents):
        raise ValueError("duplicate documents")
    for doc in documents:
        official_url(doc["source"], doc["source_url"])
        for field in ("official_record_url", "metadata_item_url"):
            if doc.get(field):
                official_url(doc["source"], doc[field])
    resolver = ArchiveResolver(root)
    if raw_workers == 1:
        for doc in documents:
            _verify_archive(doc, resolver)
    else:
        # Consume every future in document order, including after an error.
        # Executor.map cancels pending jobs when iteration raises, so use explicit
        # futures to ensure the full verification cohort finishes before return.
        failure = None
        with ThreadPoolExecutor(max_workers=raw_workers, thread_name_prefix="archive-verify") as pool:
            futures = [pool.submit(_verify_archive, doc, resolver) for doc in documents]
            for future in futures:
                try:
                    future.result()
                except Exception as exc:
                    if failure is None:
                        failure = exc
        if failure is not None:
            raise failure
    for chunk in chunks:
        doc = by_id.get(chunk["document_id"])
        if not doc or chunk["source_url"] != doc["source_url"] or chunk["raw_sha256"] != doc["sha256"]:
            raise ValueError("chunk citation/provenance mismatch")
        if hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest() != chunk["content_hash"]:
            raise ValueError("chunk text hash mismatch")
        start, end = chunk.get("page_start"), chunk.get("page_end")
        if (start is None) != (end is None) or (start is not None and (start < 1 or end < start)):
            raise ValueError("invalid page anchor")
    if len({chunk["chunk_id"] for chunk in chunks}) != len(chunks):
        raise ValueError("duplicate chunks")
    if {chunk["document_id"] for chunk in chunks} != set(by_id):
        raise ValueError("document/chunk coverage mismatch")
    return generation_id, documents, chunks


def _connect():
    import psycopg
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise ValueError("DATABASE_URL is required")
    return psycopg.connect(dsn, autocommit=True, connect_timeout=5)


def index(root: Path = ROOT, *, batch_size: int = 32, generation_id: str | None = None,
          publish_hint=True, record_events=True, activate=True) -> dict:
    with file_lock(Path(root) / "data" / "locks" / "index.lock"):
        return _index(root, batch_size=batch_size, generation_id=generation_id,
                      publish_hint=publish_hint, record_events=record_events, activate=activate)


def _index(root: Path = ROOT, *, batch_size: int = 32, generation_id: str | None = None,
           publish_hint=True, record_events=True, activate=True) -> dict:
    generation_id, documents, chunks = generation(root, generation_id=generation_id)
    revision = model_revision(root, resolve=True)
    model = None
    connection = _connect()
    try:
        manifest_hash = hashlib.sha256((Path(root) / "data" / "generations" / generation_id / "manifest.json").read_bytes()).hexdigest()
        connection.execute(SCHEMA)
        existing_generation = connection.execute("SELECT manifest_sha256 FROM index_generations WHERE generation_id=%s", (generation_id,)).fetchone()
        if existing_generation and existing_generation[0] != manifest_hash:
            raise ValueError("immutable_generation_index_mismatch")
        with connection.transaction():
            for doc in documents:
                connection.execute("INSERT INTO corpus_documents(generation_id,document_id,metadata) VALUES (%s,%s,%s::jsonb) ON CONFLICT DO NOTHING",
                                   (generation_id, doc["document_id"], json.dumps(doc, ensure_ascii=False)))
            for chunk in chunks:
                connection.execute("INSERT INTO corpus_chunks(generation_id,chunk_id,document_id,content_hash,text,metadata) VALUES (%s,%s,%s,%s,%s,%s::jsonb) ON CONFLICT DO NOTHING",
                                   (generation_id, chunk["chunk_id"], chunk["document_id"], chunk["content_hash"],
                                    chunk["text"], json.dumps(chunk, ensure_ascii=False)))
        unique = {chunk["content_hash"]: chunk["text"] for chunk in chunks}
        hashes = list(unique)
        created = 0
        for start in range(0, len(hashes), batch_size):
            group = hashes[start:start + batch_size]
            existing = {row[0] for row in connection.execute(
                "SELECT content_hash FROM embedding_cache WHERE model_revision=%s AND content_hash=ANY(%s)",
                (revision, group)).fetchall()}
            missing = [content_hash for content_hash in group if content_hash not in existing]
            if not missing:
                continue
            if model is None:
                import torch
                torch.set_num_threads(4)
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(MODEL_ID, revision=revision)
                if model.get_sentence_embedding_dimension() != DIMENSIONS:
                    raise ValueError("embedding dimensions changed")
            vectors = model.encode(["passage: " + unique[content_hash] for content_hash in missing],
                                   normalize_embeddings=True, batch_size=batch_size)
            with connection.transaction():
                for content_hash, vector in zip(missing, vectors):
                    value = "[" + ",".join(str(float(v)) for v in vector) + "]"
                    connection.execute("INSERT INTO embedding_cache(content_hash,model_revision,embedding) VALUES (%s,%s,%s::vector) ON CONFLICT DO NOTHING",
                                       (content_hash, revision, value))
                    created += 1
        for table in ("corpus_documents", "corpus_chunks", "embedding_cache"):
            connection.execute("ANALYZE " + table)
        count = connection.execute("SELECT count(*) FROM corpus_chunks c JOIN embedding_cache e ON e.content_hash=c.content_hash AND e.model_revision=%s WHERE c.generation_id=%s",
                                   (revision, generation_id)).fetchone()[0]
        if count != len(chunks):
            raise ValueError("incomplete embedding index")
        # Revalidate the files after expensive embedding work, before switching readers.
        generation(root, generation_id=generation_id)
        manifest_hash = hashlib.sha256((Path(root) / "data" / "generations" / generation_id / "manifest.json").read_bytes()).hexdigest()
        with connection.transaction():
            connection.execute("INSERT INTO index_generations(generation_id,model_revision,documents,chunks,manifest_sha256) VALUES (%s,%s,%s,%s,%s) ON CONFLICT(generation_id) DO NOTHING",
                               (generation_id, revision, len(documents), len(chunks), manifest_hash))
            if activate:
                connection.execute("INSERT INTO app_state(key,value) VALUES ('active_generation',%s) ON CONFLICT (key) DO UPDATE SET value=excluded.value", (generation_id,))
        # ACTIVE is a recovery hint. PostgreSQL remains authoritative for API readers.
        parent = Path(root) / "data" / "generations"
        temporary = parent / "ACTIVE.tmp"
        try:
            if publish_hint and activate:
                temporary.write_text(generation_id, encoding="ascii")
                os.replace(temporary, parent / "ACTIVE")
        except OSError:
            logging.getLogger(__name__).exception("Index activated; recovery pointer could not be updated")
        try:
            if record_events and activate:
                ledger = Ledger(Path(root) / "data" / "manifests" / "manifest.sqlite", Path(root) / "data" / "raw")
                try:
                    for source in {doc["source"] for doc in documents}:
                        ledger.event(source, "index", "success", "database_active_generation", generation_id)
                finally:
                    ledger.close()
        except Exception:
            logging.getLogger(__name__).exception("Index activated; source publication status could not be recorded")
        return {"generation_id": generation_id, "documents": len(documents), "chunks": len(chunks),
                "new_embeddings": created, "model_revision": revision}
    finally:
        connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-id")
    args = parser.parse_args()
    print(json.dumps(index(generation_id=args.generation_id), indent=2))
