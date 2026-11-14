"""Synthetic release bindings only. No network or cloud provisioning."""
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from tools.cloud_snapshot import (
    ACTIVE, ROLLBACK, TABLES, VERSION, canonical, connection_identity, endpoint_hash,
    load_snapshot, parse, restore_snapshot, sha, validate_rows, validate_state,
)


def fixture():
    text = "Synthetic archived provision; not legal evidence."
    digest = sha(text.encode())
    doc = {"document_id": "synthetic", "source": "ipindia", "status": "unknown", "sha256": "a" * 64,
           "source_url": "https://ipindia.gov.in/synthetic.pdf"}
    chunk = {"chunk_id": "synthetic-1", "document_id": "synthetic", "text": text, "content_hash": digest,
             "raw_sha256": doc["sha256"], "source_url": doc["source_url"]}
    evidence = {gid: {"documents": {"synthetic": doc}, "chunks": {"synthetic-1": chunk},
                      "files": {"manifest.json": "b" * 64}} for gid in (ACTIVE, ROLLBACK)}
    rows = {
        "corpus_documents": [{"generation_id": gid, "document_id": "synthetic", "metadata": doc} for gid in (ACTIVE, ROLLBACK)],
        "corpus_chunks": [{"generation_id": gid, "chunk_id": "synthetic-1", "document_id": "synthetic", "content_hash": digest,
                           "text": text, "metadata": chunk} for gid in (ACTIVE, ROLLBACK)],
        "index_generations": [{"generation_id": gid, "model_revision": "revision", "documents": 1, "chunks": 1,
                               "manifest_sha256": "b" * 64, "activated_at": datetime.now(timezone.utc).isoformat()} for gid in (ACTIVE, ROLLBACK)],
        "embedding_cache": [{"content_hash": digest, "model_revision": "revision", "embedding": json.dumps([0.0] * 384)}],
        "app_state": [{"key": "active_generation", "value": ACTIVE}, {"key": "ranking_version", "value": "v2"},
                      {"key": "retrieval_rollback", "value": json.dumps({"generation_id": ROLLBACK, "ranking_version": "v1"})}],
    }
    return rows, evidence


class CloudSnapshotTests(unittest.TestCase):
    def test_only_exact_release_state_and_rollback_are_accepted(self):
        rows, _ = fixture()
        validate_state(rows["app_state"])
        for edit in (lambda x: x.append({"key": "secret", "value": "synthetic-private"}),
                     lambda x: x[0].update(value=ROLLBACK),
                     lambda x: x[2].update(value=json.dumps({"generation_id": ACTIVE, "ranking_version": "v1"}))):
            value = copy.deepcopy(rows["app_state"])
            edit(value)
            with self.assertRaises(ValueError):
                validate_state(value)

    def test_credentials_are_not_in_identity_and_remote_requires_verified_tls(self):
        a = "postgresql://user:synthetic-password@localhost:5432/source"
        b = "postgresql://different:other@127.0.0.1:5432/source"
        self.assertEqual(endpoint_hash(a), endpoint_hash(b))
        self.assertNotIn("password", str(connection_identity(a)))
        with self.assertRaisesRegex(ValueError, "ssl_verify_full"):
            connection_identity("postgresql://user:password@managed.example/db?sslmode=require")
        self.assertEqual(connection_identity("postgresql://managed.example/db?sslmode=verify-full")["host"], "managed.example")
        with self.assertRaises(ValueError):
            connection_identity("host=one,two dbname=target sslmode=verify-full")

    def test_duplicate_json_keys_and_nonfinite_values_fail_closed(self):
        for text in ('{"x":1,"x":2}', '[NaN]', '[Infinity]'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse(text)

    def test_archive_identity_quote_vector_and_model_tampering_rejected(self):
        rows, evidence = fixture()
        validate_rows(rows, "revision", evidence)
        mutations = [
            lambda x: x["corpus_chunks"][0].update(text="altered quotation"),
            lambda x: x["corpus_chunks"][0].update(document_id="forged-id"),
            lambda x: x["corpus_documents"][0].update(metadata={"source": "nba"}),
            lambda x: x["embedding_cache"][0].update(embedding="[0.0]"),
            lambda x: x["embedding_cache"][0].update(model_revision="other"),
            lambda x: x["embedding_cache"].clear(),
            lambda x: x["index_generations"][0].update(manifest_sha256="c" * 64),
            lambda x: x["index_generations"][0].update(activated_at="2026-11-14T11:00:00"),
            lambda x: x["corpus_chunks"].append(copy.deepcopy(x["corpus_chunks"][0])),
        ]
        for mutate in mutations:
            changed = copy.deepcopy(rows)
            mutate(changed)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                validate_rows(changed, "revision", evidence)

    def test_restore_requires_explicit_authorization_before_connection(self):
        with patch("psycopg.connect") as connect, self.assertRaisesRegex(ValueError, "authorization"):
            restore_snapshot(Path("not-read"))
        connect.assert_not_called()

    def test_restore_never_opens_source_database(self):
        dsn = "postgresql://user:synthetic@localhost/source"
        manifest = {"source_endpoint_sha256": endpoint_hash(dsn)}
        with patch.dict(os.environ, {"TARGET_DATABASE_URL": dsn}), patch("tools.cloud_snapshot.load_snapshot", return_value=(manifest, {}, {})), patch("psycopg.connect") as connect:
            with self.assertRaisesRegex(ValueError, "source_database_forbidden"):
                restore_snapshot(Path("synthetic"), allow_empty_target=True)
            connect.assert_not_called()

    def test_restore_rejects_existing_tables_without_schema_or_insert(self):
        manifest = {"source_endpoint_sha256": "a" * 64}
        connection = MagicMock()
        connection.execute.return_value.fetchall.return_value = [("existing_application",)]
        with patch.dict(os.environ, {"TARGET_DATABASE_URL": "postgresql://localhost/disposable", "SOURCE_DATABASE_URL": "postgresql://localhost/source"}), patch("tools.cloud_snapshot.load_snapshot", return_value=(manifest, {}, {})), patch("psycopg.connect") as connect:
            connect.return_value.__enter__.return_value = connection
            with self.assertRaisesRegex(ValueError, "no_public_tables"):
                restore_snapshot(Path("synthetic"), allow_empty_target=True)
        queries = [call.args[0] for call in connection.execute.call_args_list]
        self.assertEqual(queries[0], "SET LOCAL search_path = public")
        self.assertFalse(any("CREATE" in query or "INSERT" in query or "DROP" in query for query in queries))

    def test_changed_snapshot_bytes_or_count_cannot_be_restored(self):
        rows, evidence = fixture()
        from tools.cloud_snapshot import encode_table
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            files = {}
            for table, entries in rows.items():
                raw = encode_table(entries)
                (path / f"{table}.jsonl").write_bytes(raw)
                files[table] = {"sha256": sha(raw), "bytes": len(raw), "rows": len(entries)}
            manifest = {"version": VERSION, "model_id": "intfloat/multilingual-e5-small", "model_revision": "revision", "dimensions": 384,
                        "active_generation": ACTIVE, "ranking_version": "v2", "rollback_generation": ROLLBACK, "rollback_ranker": "v1",
                        "source_endpoint_sha256": "a" * 64, "generation_files": {gid: item["files"] for gid, item in evidence.items()}, "tables": files}
            (path / "manifest.json").write_bytes(canonical(manifest))
            with patch("tools.cloud_snapshot.local_evidence", return_value=("revision", evidence)):
                load_snapshot(path)
                manifest["tables"]["corpus_documents"]["rows"] += 1
                (path / "manifest.json").write_bytes(canonical(manifest))
                with self.assertRaisesRegex(ValueError, "row_count"):
                    load_snapshot(path)
                manifest["tables"]["corpus_documents"]["rows"] -= 1
                (path / "manifest.json").write_bytes(canonical(manifest))
                (path / "corpus_chunks.jsonl").write_bytes(b"truncated")
                with self.assertRaisesRegex(ValueError, "integrity"):
                    load_snapshot(path)
        self.assertEqual(set(rows), set(TABLES))


if __name__ == "__main__":
    unittest.main()
