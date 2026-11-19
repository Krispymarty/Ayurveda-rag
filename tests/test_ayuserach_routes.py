"""Integration checks for the new landing page and research entry point."""
from html.parser import HTMLParser
from unittest import TestCase

from fastapi.testclient import TestClient
from ipindia_corpus.search_api import app


class StaticReferences(HTMLParser):
    def __init__(self):
        super().__init__()
        self.assets = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"script", "img", "link"}:
            value = attrs.get("src") or attrs.get("href")
            if value:
                self.assets.add(value)


class AyuSerachRoutesTests(TestCase):
    def test_pages_keep_asset_delivery_same_origin(self):
        client = TestClient(app)
        self.addCleanup(client.close)
        for path in ("/", "/research", "/research.html"):
            with self.subTest(path=path):
                page = client.get(path)
                self.assertEqual(page.status_code, 200)
                self.assertEqual(page.headers["cache-control"], "no-store")
                self.assertIn("connect-src 'self'", page.headers["content-security-policy"])
                self.assertIn("AyurIP", page.text)
                refs = StaticReferences()
                refs.feed(page.text)
                self.assertIn("/assets/ayuserach.css", refs.assets)
                for asset in refs.assets:
                    self.assertTrue(asset.startswith("/assets/"), asset)
                    self.assertEqual(client.get(asset).status_code, 200, asset)
