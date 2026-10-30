"""Integrity-preserving startup I/O regressions; no live sources or models."""
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from ipindia_corpus import indexer
from ipindia_corpus.paths import ArchiveResolver


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class StartupIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.gid = "20261029T010101Z"
        self.directory = self.root / "data/generations" / self.gid
        self.directory.mkdir(parents=True)
        self.documents, self.chunks = [], []
        for i in range(12):
            raw = self.root / f"data/raw/ipindia/{i}.html"
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_bytes(f"<main>Original source {i}</main>".encode())
            url = f"https://ipindia.gov.in/acts/test-{i}"
            self.documents.append({"document_id": str(i), "source": "ipindia", "source_url": url,
                                   "raw_path": raw.relative_to(self.root).as_posix(), "sha256": digest(raw.read_bytes())})
            text = f"Verified provision {i}"
            self.chunks.append({"chunk_id": f"c{i}", "document_id": str(i), "source_url": url,
                                "raw_sha256": self.documents[-1]["sha256"], "text": text,
                                "content_hash": digest(text.encode())})
        self.write_generation()

    def write_generation(self):
        manifest = {"generation_id": self.gid, "documents": len(self.documents), "chunks": len(self.chunks)}
        for name, values in (("documents", self.documents), ("chunks", self.chunks)):
            raw = "".join(json.dumps(v) + "\n" for v in values).encode()
            (self.directory / f"{name}.jsonl").write_bytes(raw)
            manifest[f"{name}_sha256"] = digest(raw)
        (self.directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def load(self, workers=1):
        return indexer.generation(self.root, generation_id=self.gid, raw_workers=workers)

    def test_hashes_and_parses_exact_same_jsonl_bytes_once(self):
        reads = []
        original = Path.read_bytes

        def read(path):
            reads.append(path)
            result = original(path)
            if path.name == "chunks.jsonl":
                path.write_bytes(b"changed after read")
            return result

        with patch.object(Path, "read_bytes", read):
            self.assertEqual(self.load()[2], self.chunks)
        for name in ("documents.jsonl", "chunks.jsonl"):
            self.assertEqual(sum(p.name == name for p in reads), 1)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.load()

    def test_serial_and_parallel_results_match_and_every_raw_is_rechecked(self):
        self.assertEqual(self.load(1), self.load(4))
        path = self.root / self.documents[-1]["raw_path"]
        path.write_bytes(b"corrupt last source")
        for workers in (1, 4):
            with self.assertRaisesRegex(ValueError, "archive hash mismatch"):
                self.load(workers)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.load(4)

    def test_workers_are_bounded_and_all_jobs_complete_before_failure_returns(self):
        lock = threading.Lock()
        active = peak = completed = 0
        original = indexer._verify_archive

        def verify(doc, resolver):
            nonlocal active, peak, completed
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(.01)
                return original(doc, resolver)
            finally:
                with lock:
                    active -= 1
                    completed += 1

        (self.root / self.documents[0]["raw_path"]).write_bytes(b"corrupt first source")
        with patch.object(indexer, "_verify_archive", verify):
            with self.assertRaisesRegex(ValueError, "archive hash mismatch"):
                self.load(4)
        self.assertLessEqual(peak, 4)
        self.assertGreater(peak, 1)
        self.assertEqual(completed, len(self.documents))
        self.assertEqual(active, 0)

    def test_invalid_workers_duplicate_ids_and_invalid_text_fail_closed(self):
        for workers in (0, 5, True):
            with self.assertRaises(ValueError):
                self.load(workers)
        self.chunks[-1]["chunk_id"] = self.chunks[0]["chunk_id"]
        self.write_generation()
        with self.assertRaisesRegex(ValueError, "duplicate chunks"):
            self.load(4)
        self.chunks[-1]["chunk_id"] = "last"
        self.chunks[-1]["text"] += "altered"
        self.write_generation()
        with self.assertRaisesRegex(ValueError, "text hash mismatch"):
            self.load(4)

    def test_parallel_errors_are_reported_in_document_order(self):
        def verify(doc, resolver):
            if doc["document_id"] == "0":
                time.sleep(.03)
                raise ValueError("first-document")
            if doc["document_id"] == "1":
                raise ValueError("second-document")
        with patch.object(indexer, "_verify_archive", verify):
            with self.assertRaisesRegex(ValueError, "first-document"):
                self.load(4)

    def test_resolver_checks_each_candidate_and_rejects_escape(self):
        resolver = ArchiveResolver(self.root)
        self.assertEqual(resolver.resolve(r"E:\SIH\data\raw\ipindia\1.html"), self.root / "data/raw/ipindia/1.html")
        for stored in ("data/raw/../secret", "outside", r"E:\SIH\data\raw\..\secret"):
            with self.assertRaises(ValueError):
                resolver.resolve(stored)
        outside = self.root / "outside"
        outside.mkdir()
        link = self.root / "data/raw/escape"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("symlink creation unavailable on this host; execute Linux fixture too")
        with self.assertRaisesRegex(ValueError, "escapes"):
            resolver.resolve("data/raw/escape/secret")

    def test_resolver_rechecks_symlinks_between_candidates(self):
        resolver = ArchiveResolver(self.root)
        link = self.root / "data/raw/current"
        try:
            link.symlink_to(self.root / "data/raw/ipindia", target_is_directory=True)
        except OSError:
            self.skipTest("symlink creation unavailable on this host; execute Linux fixture too")
        self.assertEqual(resolver.resolve("data/raw/current/1.html"), self.root / "data/raw/ipindia/1.html")
        link.unlink()
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes"):
            resolver.resolve("data/raw/current/1.html")


if __name__ == "__main__":
    unittest.main()
