"""Conservative font repair fixtures; no synthetic legal review or admission."""
import hashlib
from pathlib import Path
import unittest

from ipindia_corpus.nba_fidelity import (
    ORIGINAL_SHA256, build_derivation, parse_to_unicode, repair_mapping,
)


class NBAFontFidelityTests(unittest.TestCase):
    def test_parse_bfchar_ranges_and_ligatures(self):
        data = (b"2 beginbfchar <0001> <0041> <0002> <0915094D> endbfchar "
                b"2 beginbfrange <0003> <0004> <0030> "
                b"<0005> <0006> [<093E> <091C>] endbfrange")
        self.assertEqual(parse_to_unicode(data),
                         {1: "A", 2: "\u0915\u094d", 3: "0", 4: "1", 5: "\u093e", 6: "\u091c"})

    def test_invalid_or_ambiguous_cmaps_fail_closed(self):
        for data in (b"2 beginbfchar <0001> <0041> endbfchar",
                     b"1 beginbfchar <0001> <D800> endbfchar",
                     b"1 beginbfrange <FFFF> <0000> <0041> endbfrange",
                     b"1 beginbfrange <0001> <0002> [<0041>] endbfrange",
                     b"1 beginbfrange <0001> <FFFF> <" + b"0041" * 100 + b"> endbfrange",
                     b"2 beginbfchar <0001> <0041> <0001> <0042> endbfchar"):
            with self.subTest(data=data), self.assertRaises(ValueError):
                parse_to_unicode(data)

    def test_mapping_changes_only_unambiguous_existing_scalar(self):
        original = {20: "4", 23: "1", 1439: "\u091c", 5: " ", 6: "\u0915\u094d", 7: "x"}
        expected = {20: {ord("1")}, 23: {ord("4")}, 1439: {0x093F},
                    5: {32, 160}, 6: {0x0915}, 7: {ord("y"), ord("z")}, 8: {ord("q")}}
        repaired, changes = repair_mapping(original, expected)
        self.assertEqual({k: repaired[k] for k in (20, 23, 1439)}, {20: "1", 23: "4", 1439: "\u093f"})
        self.assertEqual(repaired[5], " ")
        self.assertEqual(repaired[6], "\u0915\u094d")
        self.assertEqual(repaired[7], "x")
        self.assertNotIn(8, repaired)
        self.assertEqual({c["cid"] for c in changes}, {20, 23, 1439})
        self.assertEqual(original[20], "4")

    def test_other_pdf_bytes_cannot_use_document_specific_derivation(self):
        with self.assertRaisesRegex(ValueError, "original_hash_mismatch"):
            build_derivation(b"%PDF-1.7 synthetic unrelated bytes")

    def test_original_document_derivation_is_quarantined_and_render_identical(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "data/next_stage/candidates/nba_abs_20261031/data/raw/nba" / (ORIGINAL_SHA256 + ".pdf")
        if not path.exists():
            self.skipTest("receipt-bound local original unavailable; unit fixtures remain executable")
        raw = path.read_bytes()
        result, derived = build_derivation(raw)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), ORIGINAL_SHA256)
        self.assertEqual(result["original_sha256"], ORIGINAL_SHA256)
        self.assertEqual(hashlib.sha256(derived).hexdigest(), result["derived_pdf_sha256"])
        self.assertEqual(len(result["pages"]), 28)
        self.assertFalse(result["certified"])
        self.assertFalse(result["indexing_allowed"])
        self.assertFalse(result["activation_allowed"])
        self.assertTrue(all(p["render_pixels_unchanged"] for p in result["pages"]))
        first = result["pages"][0]
        self.assertIn("4917", first["original_text"])
        self.assertNotIn("4917", first["cmap_recovered_text"])
        self.assertIn("1947", first["cmap_recovered_text"])
        self.assertIn("prebase_matra_logical_order_unverified", first["unresolved"])
        correction = next(c for c in result["font_corrections"] if c["font_xref"] == 202)
        self.assertIn({"cid": 1439, "glyph_id": 1439, "before": "\u091c", "after": "\u093f"}, correction["changes"])
        self.assertTrue(correction["embedded_font_sha256"])
        self.assertTrue(correction["original_cmap_sha256"])


if __name__ == "__main__":
    unittest.main()
