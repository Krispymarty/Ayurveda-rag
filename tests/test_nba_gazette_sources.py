"""Candidate-only fixture coverage; synthetic receipts are not live permission."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ipindia_corpus.connectors import Snapshot, official_url
from ipindia_corpus.generations import publish
from ipindia_corpus.indexer import generation
from ipindia_corpus.nba_gazette import MetadataOnlySession, listing_links, manual_queue
from ipindia_corpus.sync import Ledger, capture_receipt
from ipindia_corpus.sync_runner import sync_source
from support_review import approve_fixture
from test_three_sources import FakeClient


class NBAGazetteTests(unittest.TestCase):
    def test_curated_queue_excludes_unrelated_recruitment(self):
        links = listing_links("nba", "https://nbaindia.nic.in/public-information/notification-guidelines",
                              b'<a href="/guidelines.pdf">ABS procedural guidelines</a><a href="/nba-recruitment.pdf">Recruitment regulations</a>')
        queue = manual_queue({"nba": {"metadata_links": links}})
        urls = [item["url"] for group in queue for item in group["official_listing_candidates"]]
        self.assertIn("https://nbaindia.nic.in/guidelines.pdf", urls)
        self.assertFalse(any("recruitment" in url for url in urls))

    def test_encoded_private_or_document_destinations_stop_before_network(self):
        session = MetadataOnlySession()
        for path in ("/%6cogin", "/page?next=%6cogin", "/a%2epdf", "/page?file=a%2epdf", "/a%252epdf"):
            with self.subTest(path=path), patch("requests.Session.get") as network:
                with self.assertRaisesRegex(ValueError, "metadata_only"):
                    session.get("https://nbaindia.nic.in" + path)
                network.assert_not_called()
        session.close()
        links = listing_links("nba", "https://nbaindia.nic.in/", b'<a href="/%6cogin">ABS</a><a href="/a%2epdf">ABS rules</a>')
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]["kind"], "document_link")

    def test_encoded_redirect_cannot_fetch_an_original(self):
        from ipindia_corpus.access_review import capture
        response = type("Redirect", (), {"status_code": 302, "headers": {"Location": "/a%2epdf"}, "close": lambda self: None,
                                        "iter_content": lambda self, size: iter(())})()
        with tempfile.TemporaryDirectory() as folder, patch("requests.Session.get", return_value=response) as network:
            session = MetadataOnlySession()
            result = capture(Path(folder), "nba", "https://nbaindia.nic.in/public", "listing_or_policy", session)
            session.close()
            self.assertEqual(network.call_count, 1)
            self.assertIn("metadata_only", result["error"])
            self.assertNotIn("raw_path", result)

    def test_exact_hosts_and_pdf_metadata_only(self):
        for source, host in (("nba", "nbaindia.nic.in"), ("gazette", "egazette.gov.in")):
            official_url(source, f"https://{host}/")
            for bad in (f"https://{host}.evil.test/a", f"https://evil@{host}/a", f"http://{host}/", f"https://{host}/a/../b"):
                with self.assertRaises(ValueError):
                    official_url(source, bad)
        links = listing_links("nba", "https://nbaindia.nic.in/", b'<a href="/upload/abs.pdf">ABS Regulations</a><a href="https://evil.test/a.pdf">bad</a>')
        self.assertEqual(links[0]["url"], "https://nbaindia.nic.in/upload/abs.pdf")
        self.assertFalse(links[0]["downloaded"])
        self.assertEqual(len(links), 1)

    def test_unapproved_sources_stop_before_client(self):
        for source, host in (("nba", "nbaindia.nic.in"), ("gazette", "egazette.gov.in")):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / "config").mkdir()
                (root / "config/source_policies.json").write_text(json.dumps({source: {"approved": False}}))
                (root / "config/curated_sources.json").write_text(json.dumps({source: {"seeds": [f"https://{host}/"], "evidence_role": "legal_source"}}))
                with patch("ipindia_corpus.sync_runner.SourceClient") as client:
                    with self.assertRaisesRegex(ValueError, "policy review"):
                        sync_source(source, root)
                    client.assert_not_called()
                    self.assertFalse((root / "data/generations").exists())

    def test_fixture_capture_parse_and_candidate_publish_keep_publisher(self):
        for source, host, publisher in (("nba", "nbaindia.nic.in", "National Biodiversity Authority"),
                                        ("gazette", "egazette.gov.in", "Gazette of India / Department of Publication")):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / "config").mkdir()
                url = f"https://{host}/public/fixture.html"
                body = b"<html><title>Biological Diversity Rules fixture</title><main><h1>Biological Diversity Rules fixture</h1><p>" + b"Synthetic archived biodiversity research provision for technical tests only. " * 6 + b"</p></main></html>"
                review = approve_fixture(root, source)
                record = capture_receipt(root / "data/raw", source, requested_url=url, final_url=url,
                                         body=body, content_type="text/html", redirects=[], access_review=review)
                (root / "config/curated_sources.json").write_text(json.dumps({source: {"seeds": [url], "evidence_role": "legal_source", "max_discovered": 1}}))
                result = sync_source(source, root, client=FakeClient([Snapshot(url, body, "text/html", None, None, 200, record)]))
                self.assertEqual(result["status"], "success")
                manifest = publish(root, update_pointer=False)
                self.assertEqual(manifest["sources"][source], 1)
                self.assertFalse((root / "data/generations/CURRENT").exists())
                _, documents, chunks = generation(root, generation_id=manifest["generation_id"])
                self.assertEqual(documents[0]["authority"], publisher)
                self.assertEqual(documents[0]["source"], source)
                self.assertEqual(documents[0]["status"], "unknown")
                self.assertTrue(chunks)
                ledger = Ledger(root / "data/manifests/manifest.sqlite", root / "data/raw")
                try:
                    self.assertEqual(ledger.db.execute("SELECT count(*) FROM capture_receipts").fetchone()[0], 1)
                finally:
                    ledger.close()


if __name__ == "__main__":
    unittest.main()
