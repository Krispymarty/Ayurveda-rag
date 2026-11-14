"""Small synthetic release integrity tests; never source or legal certification."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.cloud_serve import runtime_port, verify_release


class ReleaseSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        (self.root / "evidence.txt").write_bytes(b"synthetic evidence")
        self.data = {"version": "ayurip-readonly-release-v1", "active_generation": "20261014T204940358650Z",
                     "ranking_version": "v2", "rollback_generation": "20261014T190100599320Z", "rollback_ranker": "v1",
                     "public_ai_enabled": False, "synchronization_enabled": False,
                     "files": {"evidence.txt": {"bytes": 18, "sha256": hashlib.sha256(b"synthetic evidence").hexdigest()}}}

    def manifest(self):
        raw = json.dumps(self.data).encode()
        (self.root / "release-manifest.json").write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def test_exact_snapshot_accepts_then_changed_or_missing_bytes_reject(self):
        digest = self.manifest()
        verify_release(self.root, digest)
        (self.root / "evidence.txt").write_bytes(b"modified evidence!")
        with self.assertRaisesRegex(ValueError, "integrity"):
            verify_release(self.root, digest)
        (self.root / "evidence.txt").unlink()
        with self.assertRaisesRegex(ValueError, "containment"):
            verify_release(self.root, digest)

    def test_wrong_manifest_hash_and_ai_or_sync_changes_reject(self):
        self.manifest()
        with self.assertRaisesRegex(ValueError, "manifest_binding"):
            verify_release(self.root, "0" * 64)
        for key in ("public_ai_enabled", "synchronization_enabled"):
            self.data[key] = True
            with self.assertRaisesRegex(ValueError, "policy_binding"):
                verify_release(self.root, self.manifest())
            self.data[key] = False

    def test_traversal_and_absolute_path_reject_even_if_manifest_is_bound(self):
        for name in ("../escape", str(self.root.parent / "absolute.txt")):
            self.data["files"] = {name: {"sha256": "0" * 64, "bytes": 0}}
            with self.assertRaisesRegex(ValueError, "containment"):
                verify_release(self.root, self.manifest())

    def test_entrypoint_requires_public_mode_verified_tls_offline_and_valid_port(self):
        env = {"AYURIP_DEPLOYMENT_MODE": "public_research", "AYURIP_PUBLIC_ORIGIN": "https://research.test",
               "DATABASE_URL": "postgresql://app:synthetic@database.test/research?sslmode=verify-full",
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PORT": "10000"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(runtime_port(), 10000)
        for key, value in (("AYURIP_DEPLOYMENT_MODE", "local_internal"),
                           ("DATABASE_URL", "postgresql://app:synthetic@database.test/research?sslmode=require"),
                           ("HF_HUB_OFFLINE", "0"), ("TRANSFORMERS_OFFLINE", "0"), ("PORT", "65536")):
            with self.subTest(key=key), patch.dict(os.environ, {**env, key: value}, clear=True), self.assertRaises(ValueError):
                runtime_port()
