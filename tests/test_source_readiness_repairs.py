"""Source-policy parity and immutable receipt regressions; no live fetches."""
import json
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.robotparser import RobotFileParser

from ipindia_corpus.connectors import SourceClient, official_url
from ipindia_corpus.corpus_certification import official_public_url
from ipindia_corpus.sync import Ledger, capture_receipt, verify_capture_receipt
from ipindia_corpus.access_review import verify_review
from test_three_sources import BODY, URL, FakeHTTPResponse, FakeSession
from support_review import approve_fixture


class SourceReadinessRepairs(unittest.TestCase):
    def test_restricted_encoded_destinations_fail_before_network(self):
        with tempfile.TemporaryDirectory() as folder:
            session = FakeSession([])
            c = SourceClient("tkdl", Path(folder), session=session)
            for path in ("/%64atabase/index.asp", "/%2564atabase/index.asp",
                         "/about?next=%252Fdatabase%252Flogin", "/about?login=1", "/logon"):
                url = "https://www.tkdl.res.in" + path
                with self.subTest(path=path):
                    self.assertFalse(official_public_url("tkdl", url))
                    with self.assertRaisesRegex(ValueError, "restricted_tkdl"):
                        c.get(url)
                    with self.assertRaises(ValueError):
                        official_url("tkdl", url)
            self.assertEqual(session.responses, [])
            self.assertTrue(official_public_url("tkdl", "https://tkdl.res.in/tkdl/langdefault/Unani/Una_Research.asp?GL=Eng"))

    @patch("ipindia_corpus.connectors.time.sleep")
    def test_redirect_restriction_is_checked_before_following(self, _sleep):
        with tempfile.TemporaryDirectory() as folder:
            session = FakeSession([FakeHTTPResponse(302, headers={"Location": "/%64atabase/login"})])
            c = SourceClient("tkdl", Path(folder), session=session)
            robots = RobotFileParser()
            robots.parse([])
            c.robots["tkdl.res.in"] = robots
            with self.assertRaisesRegex(ValueError, "restricted_tkdl"):
                c.get(URL)

    @patch("ipindia_corpus.connectors.time.sleep")
    def test_complete_receipt_is_hash_bound_and_tampering_fails(self, _sleep):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config").mkdir()
            (root / "config/source_policies.json").write_text(json.dumps({"tkdl": {"approved": True}}), encoding="utf-8")
            approve_fixture(root, "tkdl")
            review = json.loads((root / "config/source_policies.json").read_text())["tkdl"]
            session = FakeSession([FakeHTTPResponse(302, headers={"Location": URL}),
                                   FakeHTTPResponse(200, BODY, {"Content-Type": "text/html", "ETag": "v1", "Last-Modified": "Wed, 14 Oct 2026 00:00:00 GMT"})])
            c = SourceClient("tkdl", root / "data/raw", session=session, access_review=review)
            robots = RobotFileParser()
            robots.parse([])
            c.robots["tkdl.res.in"] = robots
            snapshot = c.get("https://tkdl.res.in/old-public-page")
            record = snapshot.capture_receipt
            receipt = verify_capture_receipt(root, record, "tkdl", URL)
            self.assertEqual(receipt["requested_url"], "https://tkdl.res.in/old-public-page")
            self.assertEqual(receipt["final_url"], URL)
            self.assertEqual(len(receipt["redirect_chain"]), 1)
            self.assertEqual(receipt["byte_length"], len(BODY))
            self.assertEqual(receipt["etag"], "v1")
            self.assertEqual(receipt["access_review"]["reviewer"], review["reviewer"])
            path = root / record["receipt_path"]
            old = path.read_bytes()
            path.write_bytes(old.replace(b'"status": 200', b'"status": 201'))
            with self.assertRaisesRegex(ValueError, "receipt_hash"):
                verify_capture_receipt(root, record, "tkdl", URL)

    @patch("ipindia_corpus.connectors.time.sleep")
    def test_304_and_substantive_timeout_footer_do_not_fabricate_new_original(self, _sleep):
        with tempfile.TemporaryDirectory() as folder:
            text = b"<html><title>Public TKDL background</title><body>" + b"Substantive historical information. " * 30 + b"Session Timeout</body></html>"
            session = FakeSession([FakeHTTPResponse(200, text), FakeHTTPResponse(304)])
            c = SourceClient("tkdl", Path(folder) / "data/raw", session=session)
            robots = RobotFileParser()
            robots.parse([])
            c.robots["tkdl.res.in"] = robots
            self.assertTrue(c.get(URL).body)
            unchanged = c.get(URL)
            self.assertIsNone(unchanged.body)
            self.assertIsNone(unchanged.capture_receipt)

    def test_receipt_history_does_not_turn_unchanged_bytes_into_metadata_change(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        (root / "config").mkdir()
        review = approve_fixture(root, "tkdl")
        ledger = Ledger(root / "data/manifests/test.sqlite", root / "data/raw")
        self.addCleanup(ledger.close)
        run = ledger.start("tkdl")
        captures = [capture_receipt(root / "data/raw", "tkdl", requested_url=URL, final_url=URL,
                                   body=BODY, content_type="text/html", redirects=[], access_review=review) for _ in range(2)]
        first = ledger.observe(run, URL, BODY, content_type="text/html", capture=captures[0])
        second = ledger.observe(run, URL, BODY, content_type="text/html", capture=captures[1])
        self.assertEqual((first, second), ("NEW_DOCUMENT", "UNCHANGED"))
        self.assertEqual(ledger.db.execute("SELECT count(*) FROM capture_receipts").fetchone()[0], 2)
        self.assertEqual(ledger.db.execute("SELECT count(*) FROM document_versions").fetchone()[0], 1)
        self.assertEqual(ledger.db.execute("SELECT count(*) FROM accepted_documents").fetchone()[0], 0)
        original = root / verify_capture_receipt(root, captures[0], "tkdl", URL)["raw_path"]
        original.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "original_hash"):
            verify_capture_receipt(root, captures[0], "tkdl", URL)

    @patch("ipindia_corpus.connectors.time.sleep")
    def test_html_error_is_rejected_after_mime_sniffing(self, _sleep):
        with tempfile.TemporaryDirectory() as folder:
            session = FakeSession([FakeHTTPResponse(200, b"<html><title>Session Timeout</title><p>Session expired</p></html>", {"Content-Type": "application/octet-stream"})])
            c = SourceClient("tkdl", Path(folder), session=session)
            robots = RobotFileParser()
            robots.parse([])
            c.robots["tkdl.res.in"] = robots
            with self.assertRaisesRegex(ValueError, "error_or_session_page"):
                c.get(URL)

    def test_encoded_dot_segments_cannot_escape_permitted_prefix(self):
        with tempfile.TemporaryDirectory() as folder:
            c = SourceClient("indiacode", Path(folder), session=FakeSession([]), permitted_prefixes=["/public/"])
            for suffix in ("../private", "%2e%2e/private", "%252e%252e/private", "%2e%2e%2fprivate"):
                with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, "ambiguous source path"):
                    c.get("https://indiacode.gov.in/public/" + suffix)

    def test_unapproved_review_and_tampered_review_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            record = capture_receipt(root / "data/raw", "tkdl", requested_url=URL, final_url=URL,
                                     body=BODY, content_type="text/html", redirects=[])
            with self.assertRaisesRegex(ValueError, "unapproved"):
                verify_capture_receipt(root, record, "tkdl", URL)
            receipt = json.loads((root / record["receipt_path"]).read_text(encoding="utf-8"))
            (root / receipt["access_review"]["raw_path"]).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "review_hash"):
                verify_capture_receipt(root, record, "tkdl", URL)

    def test_access_review_binds_all_archived_observation_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config").mkdir()
            review = approve_fixture(root, "tkdl")
            verify_review(root, "tkdl", review)
            changes = {"final_url": "https://www.tkdl.res.in/robots.txt",
                       "requested_url": "https://www.tkdl.res.in/robots.txt",
                       "redirect_chain": [{"to_url": "https://www.tkdl.res.in/robots.txt"}],
                       "at": "2020-01-01T00:00:00Z", "retrieved_at": "2020-01-01T00:00:00Z",
                       "byte_length": 1, "etag": "altered", "last_modified": "altered"}
            for key, value in changes.items():
                tampered = copy.deepcopy(review)
                tampered["evidence"][0][key] = value
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "receipt mismatch"):
                    verify_review(root, "tkdl", tampered)
            record = review["evidence"][0]
            path = root / record["receipt_path"]
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "receipt hash mismatch"):
                verify_review(root, "tkdl", review)

    def test_legacy_access_receipt_schema_remains_supported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config").mkdir()
            review = approve_fixture(root, "tkdl")
            for record in review["evidence"]:
                legacy = {key: value for key, value in record.items() if key in {
                    "url", "kind", "at", "final_url", "status", "sha256", "raw_path", "content_type"}}
                payload = json.dumps(legacy, sort_keys=True).encode("utf-8")
                path = root / "data/raw/access/tkdl" / (hashlib.sha256(payload).hexdigest() + ".json")
                path.write_bytes(payload)
                record.clear()
                record.update(legacy, receipt_path=path.relative_to(root).as_posix())
            verify_review(root, "tkdl", review)


if __name__ == "__main__":
    unittest.main()
