import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from ipindia_corpus.pipeline import (
    BASE, Client, Fetch, build_document, canonical, html_content,
    classify_domains, is_index_page, normalized_document_title, score, select_export_records, structure_chunks, topics,
)


class PipelineTests(unittest.TestCase):
    def test_canonical_allows_only_official_https_hosts(self):
        self.assertEqual(canonical("/resource/patents-resources-act#top"), BASE + "resource/patents-resources-act")
        self.assertIsNone(canonical("https://search.ipindia.gov.in/private"))
        self.assertIsNone(canonical("http://ipindia.gov.in/resource"))
        self.assertIsNone(canonical("https://ipindia.gov.in.evil.test/"))

    def test_focus_filter(self):
        self.assertGreater(score(BASE + "resource/patents-resources-guidelines"), 0)
        self.assertLess(score(BASE + "staff/recruitment-roster"), 0)
        self.assertIn("Traditional Knowledge", topics("TKDL and traditional knowledge"))
        self.assertGreater(
            score(BASE + "storage/file.pdf", "Guidelines for Traditional Knowledge"),
            score(BASE + "acts/patent-act-1970/section-48", "Section 48"),
        )
        self.assertEqual(classify_domains("Trade Marks Rules, 2017", BASE + "storage/file.pdf", BASE + "resource/patents-resources-rules")[0], "Trademark")
        self.assertEqual(classify_domains("GI Rules", BASE + "storage/file.pdf", BASE)[0], "GI")

    def test_html_extracts_content_and_discovery_links(self):
        html = b"<html><head><title>Patent Rules</title></head><body><nav><a href='/resource/patents-resources-rules'>Rules</a></nav><main><h1>Rules</h1><p>Rule 1. Short title</p></main><footer>Private details</footer></body></html>"
        title, content, links = html_content(html)
        self.assertEqual(title, "Patent Rules")
        self.assertIn("Rule 1", content)
        self.assertNotIn("Private details", content)
        self.assertEqual(links[0][0], "/resource/patents-resources-rules")

    def test_legal_page_extracts_section_heading_and_omits_sidebar(self):
        html = b"<html><head><title>Patent Act 1970</title></head><body><div class='leftSideBar'>Section 1 Section 2</div><div class='actsAreaBox'><h2><span>CHAPTER 2</span></h2><div class='chapterHeadinBox'>SECTION 3. WHAT ARE NOT INVENTIONS</div><div class='section-body'><p>(p) traditional knowledge</p></div></div></body></html>"
        _, content, _ = html_content(html)
        self.assertIn("SECTION 3. WHAT ARE NOT INVENTIONS", content)
        self.assertNotIn("Section 1", content)
        self.assertTrue(is_index_page(BASE + "resource/patents-resources-act"))

    def test_pdf_link_uses_resource_row_title(self):
        html = b"<table><tr><td>14</td><td>Guidelines for Traditional Knowledge</td><td><a href='/storage/file.pdf'>Format: PDF Size: 300 KB</a></td></tr></table>"
        _, _, links = html_content(html)
        self.assertEqual(links[0][1], "Guidelines for Traditional Knowledge")

    def test_numbered_rules_keep_rule_number(self):
        fetch = Fetch(BASE + "frontend/pdf/patents/rules/patent-rules-2003.pdf", b"example", "application/pdf", "2026-01-01T00:00:00+00:00", Path("x"))
        doc = build_document(fetch, "The Patents Rules, 2003", "1. Short title and commencement", fetch.url)
        chunks = structure_chunks("1. Short title and commencement\n(1) These rules may be called the Patents Rules\n2. Definitions\nIn these rules", doc)
        self.assertEqual([chunk["rule"] for chunk in chunks], ["1", "1", "2"])

    def test_export_deduplicates_identical_content(self):
        entries = []
        for identifier in ("a" * 20, "b" * 20):
            entries.append(({
                "document_id": identifier, "title": "Patent Rules", "content_hash": "same",
                "download_url": BASE + identifier + ".pdf", "source_url": BASE,
            }, [{"document_id": identifier, "text": "Rule 1"}]))
        selected = select_export_records(entries)
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(selected[0][0]["alternate_download_urls"]), 1)

    def test_section_three_clause_boundaries(self):
        fetch = Fetch(BASE + "acts/patent-act-1970/section-3", b"example", "text/html", "2026-01-01T00:00:00+00:00", Path("x"))
        doc = build_document(fetch, "The Patents Act, 1970", "SECTION 3. What are not inventions", fetch.url)
        text = "CHAPTER II\nSECTION 3. What are not inventions\n(c) discovery of a scientific principle (d) a known substance (p) traditional knowledge"
        chunks = structure_chunks(text, doc)
        clauses = [chunk for chunk in chunks if chunk["subsection"]]
        self.assertEqual([chunk["subsection"] for chunk in clauses], ["c", "d", "p"])
        self.assertTrue(all(chunk["section"] == "3" for chunk in clauses))
        self.assertTrue(all(chunk["source_url"] == fetch.url for chunk in clauses))
        self.assertEqual(normalized_document_title("IP India - Intellectual Property India", fetch.url), "The Patents Act, 1970")

    def test_source_url_is_the_document_and_parent_is_provenance(self):
        fetch = Fetch(BASE + "frontend/pdf/patents/example.pdf", b"raw", "application/pdf", "2026-01-01T00:00:00+00:00", Path("x"))
        document = build_document(fetch, "Patent manual", "Visible text", BASE + "resource/patents-resources-manuals")
        self.assertEqual(document["source_url"], fetch.url)
        self.assertEqual(document["discovered_from_url"], BASE + "resource/patents-resources-manuals")
        self.assertEqual(document["status"], "unknown")
        self.assertIsNone(document["effective_from"])
        chunks = structure_chunks("[Page 2]\nRule 3. Provisions\nText\n[Page 3]\nMore text", document)
        self.assertEqual(chunks[0]["page_end"], 2)
        self.assertEqual(chunks[1]["page_start"], 3)

    def test_hidden_and_active_html_are_not_extracted(self):
        html = b"<main><p>Visible legal wording</p><p hidden>Secret</p><div style='display:none'><p>Hidden</p></div><iframe>Active</iframe><script>Bad</script></main>"
        _, content, _ = html_content(html)
        self.assertIn("Visible legal wording", content)
        self.assertNotIn("Secret", content)
        self.assertNotIn("Hidden", content)
        self.assertNotIn("Active", content)
        self.assertNotIn("Bad", content)

    def test_conditional_revalidation_reports_304_without_overwriting_cache(self):
        client = Client(Path("unused"), delay=1)
        client.allowed = lambda url: True
        response = Mock()
        response.status_code = 304
        response.headers = {"ETag": '"v1"'}
        client.session.get = Mock(return_value=response)
        result = client.revalidate(BASE + "acts/example", etag='"v1"')
        self.assertEqual(result.status_code, 304)
        self.assertIsNone(result.data)
        self.assertEqual(client.session.get.call_args.kwargs["headers"]["If-None-Match"], '"v1"')

    def test_retry_after_server_error_and_minimum_delay(self):
        client = Client(Path("unused"), delay=1)
        client.allowed = lambda url: True
        failed = Mock(status_code=500, headers={})
        success = Mock(status_code=200, headers={})
        client.session.get = Mock(side_effect=[failed, success])
        client.last_request = 100.0
        with patch("ipindia_corpus.pipeline.time.monotonic", return_value=100.0), patch("ipindia_corpus.pipeline.time.sleep") as sleeping:
            self.assertIs(client._request(BASE + "acts/example"), success)
        self.assertEqual(client.session.get.call_count, 2)
        failed.close.assert_called_once()
        self.assertTrue(any(call.args and call.args[0] >= 1 for call in sleeping.call_args_list))


if __name__ == "__main__":
    unittest.main()
