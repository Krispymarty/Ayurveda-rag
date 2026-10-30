"""Run only in an explicitly supplied disposable PostgreSQL database."""
import hashlib
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from ipindia_corpus.generations import publish
from ipindia_corpus.indexer import MODEL_ID, ROOT, SCHEMA, _connect, generation, index
from ipindia_corpus.sync import Ledger
from ipindia_corpus.retrieval import SEARCH_SQL, query_parameters


@unittest.skipUnless(os.environ.get("AYURIP_TEST_DATABASE_URL"), "disposable PostgreSQL URL not supplied")
class PostgreSQLIntegrationTests(unittest.TestCase):
    def test_passage_ranking_deduplicates_within_document_and_keeps_versions(self):
        with patch.dict(os.environ, {"DATABASE_URL": os.environ["AYURIP_TEST_DATABASE_URL"]}):
            connection = _connect()
            try:
                connection.execute(SCHEMA)
                revision = "test-ranking-v2"
                vector = "[" + ",".join(["1"] + ["0"] * 383) + "]"
                identifier = "ranking-fixture"
                documents = [("a", "https://ipindia.gov.in/acts/test-a", "Design"),
                             ("b", "https://ipindia.gov.in/acts/test-b", "Design"),
                             ("c", "https://ipindia.gov.in/acts/test-c", "Patent")]
                for document_id, url, domain in documents:
                    document = {"title": "Designs Act 2000", "document_type": "Act", "source_url": url}
                    connection.execute("INSERT INTO corpus_documents VALUES (%s,%s,%s::jsonb)", (identifier, document_id, json.dumps(document)))
                    for number, body in (("5", "5. (1) Application for registration. Pay the prescribed fee."),
                                         ("6", "6. (1) Classification of registered designs.")):
                        content_hash = hashlib.sha256(body.encode()).hexdigest()
                        metadata = {"document_id": document_id, "source": "ipindia", "source_url": url,
                                    "legal_domain": domain, "section": number, "heading": "Application for registration" if number == "5" else "Classification"}
                        connection.execute("INSERT INTO embedding_cache VALUES (%s,%s,%s::vector) ON CONFLICT DO NOTHING", (content_hash, revision, vector))
                        for duplicate in range(2):
                            connection.execute("INSERT INTO corpus_chunks(generation_id,chunk_id,document_id,content_hash,text,metadata) VALUES (%s,%s,%s,%s,%s,%s::jsonb)",
                                               (identifier, f"{document_id}-{number}-{duplicate}", document_id, content_hash, body, json.dumps(metadata)))
                arguments = {"generation_id": identifier, "source": None, "domain": None,
                             "embedding": vector, "revision": revision, "limit": 20,
                             **query_parameters("Designs Act 2000 sections 5 and 6 registration classification")}
                rows = connection.execute(SEARCH_SQL, arguments).fetchall()
                self.assertEqual(len(rows), 4)
                self.assertEqual({r[2]['document_id'] for r in rows}, {'a', 'b'})
                self.assertEqual({r[2]['section'] for r in rows}, {'5', '6'})
                arguments['source'] = 'tkdl'
                self.assertEqual(connection.execute(SEARCH_SQL, arguments).fetchall(), [])
                arguments['source'] = 'ipindia'
                arguments['domain'] = 'Patent'
                self.assertEqual(connection.execute(SEARCH_SQL, arguments).fetchall(), [])
            finally:
                connection.close()

    def test_failed_embedding_keeps_active_index_and_sources_ignore_candidate(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory, patch.dict(os.environ, {"DATABASE_URL": os.environ["AYURIP_TEST_DATABASE_URL"]}):
            root = Path(directory)
            (root / "data").mkdir()
            (root / "config").mkdir()
            (root / "config/source_policies.json").write_text(json.dumps({key: {"approved": False} for key in ("ipindia", "indiacode", "tkdl")}))
            lock = json.loads((ROOT / "data/model.lock.json").read_text())
            (root / "data/model.lock.json").write_text(json.dumps(lock))
            ledger = Ledger(root / "data/manifests/manifest.sqlite", root / "data/raw")
            try:
                url = "https://ipindia.gov.in/acts/integration-test"
                body = b"<html><title>Test Act</title><main><p>Archived evidence for the disposable integration database. This paragraph verifies citation integrity and source history.</p></main></html>"
                run = ledger.start("ipindia")
                ledger.observe(run, url, body, content_type="text/html")
                ledger.finish(run)
                first = publish(root)["generation_id"]
                _, docs, chunks = generation(root)
                connection = _connect()
                try:
                    connection.execute(SCHEMA)
                    for chunk in chunks:
                        connection.execute("INSERT INTO embedding_cache VALUES (%s,%s,%s::vector)",
                                           (chunk["content_hash"], lock["revision"], "[" + ",".join(["0"] * 384) + "]"))
                finally:
                    connection.close()
                index(root)
                run = ledger.start("tkdl")
                ledger.observe(run, "https://tkdl.res.in/tkdl/langdefault/common/Abouttkdl.asp", body.replace(b"disposable", b"candidate"), content_type="text/html")
                ledger.finish(run)
                candidate = publish(root)["generation_id"]
                fake = types.SimpleNamespace(SentenceTransformer=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected model failure")))
                with patch.dict("sys.modules", {"sentence_transformers": fake}):
                    with self.assertRaisesRegex(RuntimeError, "injected model failure"):
                        index(root)
                connection = _connect()
                try:
                    self.assertEqual(connection.execute("SELECT value FROM app_state WHERE key='active_generation'").fetchone()[0], first)
                    self.assertGreater(connection.execute("SELECT count(*) FROM corpus_documents WHERE generation_id=%s", (candidate,)).fetchone()[0], 0)
                finally:
                    connection.close()
                from ipindia_corpus import search_api
                from ipindia_corpus.readiness import Readiness
                ready = Readiness(lambda record: None)
                with patch.object(search_api, "ROOT", root), patch.object(search_api, "readiness", ready), patch.object(search_api, "_load_model", return_value=object()):
                    ready.value = search_api._initialize(lambda *args: None)
                    ready.state = "ready"
                    status = search_api.sources_endpoint()
                    self.assertEqual(status["active_generation_id"], first)
                    self.assertEqual(next(row for row in status["sources"] if row["source_id"] == "tkdl")["searchable_document_count"], 0)
                    self.assertEqual(search_api.health_endpoint()["status"], "ok")
                    search_api.verify_hit(chunks[0], chunks[0]["text"], {doc["document_id"]: doc for doc in docs}, root)
                    with self.assertRaisesRegex(ValueError, "indexed_text_hash"):
                        search_api.verify_hit(chunks[0], chunks[0]["text"] + "changed", {doc["document_id"]: doc for doc in docs}, root)
                    forged = {**chunks[0], "text": "Fabricated evidence", "content_hash": hashlib.sha256(b"Fabricated evidence").hexdigest()}
                    with self.assertRaisesRegex(ValueError, "published_chunk_mismatch"):
                        search_api.verify_hit(forged, forged["text"], {doc["document_id"]: doc for doc in docs}, root, expected_chunk=chunks[0])
                    self.assertEqual(docs[0]["sha256"], hashlib.sha256(body).hexdigest())
                manifest_path = root / "data/generations" / first / "manifest.json"
                original_manifest = manifest_path.read_bytes()
                altered_manifest = json.loads(original_manifest)
                altered_manifest["unexpected_mutation"] = True
                manifest_path.write_text(json.dumps(altered_manifest), encoding="utf-8")
                try:
                    with self.assertRaisesRegex(ValueError, "immutable_generation_index_mismatch"):
                        index(root, generation_id=first)
                finally:
                    manifest_path.write_bytes(original_manifest)
                self.assertEqual(MODEL_ID, lock["model_id"])
                # An inactive index build must not change readers, and the paired
                # generation/ranker selection must be restored by real rollback.
                _, candidate_docs, candidate_chunks = generation(root, generation_id=candidate)
                connection = _connect()
                try:
                    for chunk in candidate_chunks:
                        connection.execute("INSERT INTO embedding_cache VALUES (%s,%s,%s::vector) ON CONFLICT DO NOTHING",
                                           (chunk["content_hash"], lock["revision"], "[" + ",".join(["0"] * 384) + "]"))
                    index(root, generation_id=candidate, activate=False)
                    self.assertEqual(connection.execute("SELECT value FROM app_state WHERE key='active_generation'").fetchone()[0], first)
                    connection.execute("INSERT INTO app_state VALUES ('retrieval_rollback',%s)", (json.dumps({"generation_id": first, "ranking_version": "v1"}),))
                    connection.execute("INSERT INTO app_state VALUES ('ranking_version','v2')")
                    connection.execute("UPDATE app_state SET value=%s WHERE key='active_generation'", (candidate,))
                    from ipindia_corpus import retrieval_release
                    with patch.object(retrieval_release, "ROOT", root):
                        result = retrieval_release.rollback()
                    self.assertEqual(result, {"generation_id": first, "ranking_version": "v1"})
                    self.assertEqual(connection.execute("SELECT value FROM app_state WHERE key='active_generation'").fetchone()[0], first)
                    self.assertEqual(connection.execute("SELECT value FROM app_state WHERE key='ranking_version'").fetchone()[0], "v1")
                finally:
                    connection.close()
            finally:
                ledger.close()
