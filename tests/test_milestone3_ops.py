"""Frozen proposal bundles must survive export-tool revisions unchanged."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ipindia_corpus.llm_experiment import canonical
from tools.milestone3.common import DATA, sha
from tools.milestone3.preparation import proposals


class FrozenProposalTests(unittest.TestCase):
    def bundle(self, root):
        folder = root / DATA / "evaluation_proposals_v1"
        folder.mkdir(parents=True)
        raw = b'{"id":"synthetic-frozen-proposal"}\n'
        (folder / "pending_cases.jsonl").write_bytes(raw)
        manifest = {"cases_sha256": sha(raw), "independently_reviewed": False}
        (folder / "manifest.json").write_bytes(canonical(manifest))
        return folder, manifest

    def test_repeat_export_reuses_sealed_bytes_instead_of_reauthoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder, manifest = self.bundle(root)
            original = (folder / "pending_cases.jsonl").read_bytes()
            with patch("tools.milestone3.preparation.Sources", side_effect=AssertionError("no reauthoring")):
                self.assertEqual(proposals(root), manifest)
            self.assertEqual((folder / "pending_cases.jsonl").read_bytes(), original)

    def test_corrupt_bundle_rejected_before_report_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder, _ = self.bundle(root)
            (folder / "pending_cases.jsonl").write_bytes(b"altered")
            with self.assertRaisesRegex(ValueError, "frozen_proposal_bundle_changed"):
                proposals(root)


if __name__ == "__main__":
    unittest.main()
