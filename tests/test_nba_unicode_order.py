"""Synthetic glyph fixtures and the pinned local original; no legal judgments."""
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

from ipindia_corpus.nba_unicode_order import _font_evidence, build_candidate, reorder_span


def glyphs(text):
    return [{"char": c, "glyph_id": 1439 if c == "\u093f" else 100 + i,
             "origin": [float(i), 10.0], "bbox": [float(i), 0.0, float(i + 1), 12.0],
             "trace_refs": [[0, i]]} for i, c in enumerate(text)]


class UnicodeOrderTests(unittest.TestCase):
    def test_known_glyph_before_contiguous_consonant_is_permutation(self):
        for before, expected in (("अिधिनयम", "अधिनियम"), ("प्रािधकरण", "प्राधिकरण"),
                                 ("िविनयम", "विनियम"), ("िवकल्प", "विकल्प")):
            with self.subTest(before=before):
                source = glyphs(before)
                result = reorder_span(source, font_verified=True, size=10)
                self.assertEqual(result["text"], expected)
                self.assertEqual(sorted(result["source_order"]), list(range(len(source))))
                self.assertEqual(sorted(result["text"]), sorted(before))
                self.assertEqual("".join(c["char"] for c in source), before)
                self.assertTrue(result["changes"])
                second = reorder_span([source[i] for i in result["source_order"]], font_verified=True, size=10)
                self.assertEqual(second["text"], expected)
                self.assertFalse(second["changes"])

    def test_conjunct_expansion_and_nukta_stay_together(self):
        for before, expected in (("िक्ष", "क्षि"), ("िक़", "क़ि")):
            source = glyphs(before)
            for i in range(2, len(source)):
                source[i].update(glyph_id=-1, origin=[2.0, 10.0], bbox=[2.0, 0.0, 2.0, 12.0])
            self.assertEqual(reorder_span(source, font_verified=True, size=10)["text"], expected)

    def test_zero_width_logical_continuation_cannot_jump_to_another_cell(self):
        source = glyphs("िक्ष")
        source[2].update(glyph_id=-1, origin=[50.0, 10.0], bbox=[50.0, 0.0, 50.0, 12.0])
        source[3].update(glyph_id=-1, origin=[50.0, 10.0], bbox=[50.0, 0.0, 50.0, 12.0])
        result = reorder_span(source, font_verified=True, size=10)
        self.assertEqual(result["text"], "िक्ष")
        self.assertFalse(result["changes"])

    def test_unbound_or_wrong_glyph_never_triggers_reordering(self):
        source = glyphs("िव")
        self.assertEqual(reorder_span(source, font_verified=False, size=10)["text"], "िव")
        source[0]["glyph_id"] = 2222
        result = reorder_span(source, font_verified=True, size=10)
        self.assertEqual(result["text"], "िव")
        self.assertTrue(result["unresolved"])

    def test_overprinted_or_missing_trace_is_not_unique_provenance(self):
        for refs in ([], [[0, 0], [1, 0]]):
            source = glyphs("िव")
            source[0]["trace_refs"] = refs
            result = reorder_span(source, font_verified=True, size=10)
            self.assertEqual(result["text"], "िव")
            self.assertFalse(result["changes"])

    def test_empty_alias_font_glyph_cannot_trigger_reorder(self):
        source = glyphs("िव")
        source[0]["bbox"][2] = 0
        source[1].update(origin=[0.0, 10.0], bbox=[0.0, 0.0, 1.0, 12.0])
        self.assertEqual(reorder_span(source, font_verified=True, size=10)["text"], "िव")

    def test_unknown_same_family_nonempty_or_advancing_glyph_blocks_font_scope(self):
        class Fonts:
            def get_fonts(self, full):
                return [(1, "ttf", "Type0", "Arial Unicode MS"),
                        (2, "ttf", "Type0", "Arial Unicode MS")]

        class Document:
            def extract_font(self, xref):
                return (None, None, None, b"known" if xref == 1 else b"other")

            def xref_get_key(self, xref, key):
                return ("null", "null")

        known = hashlib.sha256(b"known").hexdigest()
        other = hashlib.sha256(b"other").hexdigest()
        for outline, advance in ((True, 0), (False, 12)):
            programs = {known: {"unicode_candidates": [0x093F], "glyph_1439_has_outline": True,
                                "glyph_1439_advance": 654},
                        other: {"unicode_candidates": [], "glyph_1439_has_outline": outline,
                                "glyph_1439_advance": advance}}
            with patch("ipindia_corpus.nba_unicode_order.FONT_SHA256", known):
                evidence, scope = _font_evidence(Document(), Document(), Fonts(), programs)
            self.assertFalse(evidence)
            self.assertFalse(scope["visible_glyph_program_unique"])

    def test_whitespace_number_other_vowel_joiner_and_geometry_are_not_crossed(self):
        for text in ("ि व", "ि2व", "िवु", "िक्\u200dष", "ििव", "िक्", "िर्क"):
            with self.subTest(text=text):
                result = reorder_span(glyphs(text), font_verified=True, size=10)
                self.assertEqual(result["text"], text)
                self.assertFalse(result["changes"])
        source = glyphs("िव")
        source[1]["origin"][0] = 50
        self.assertEqual(reorder_span(source, font_verified=True, size=10)["text"], "िव")
        source = glyphs("िव")
        source[1]["origin"][1] = 30
        self.assertEqual(reorder_span(source, font_verified=True, size=10)["text"], "िव")

    def test_already_logical_prebase_geometry_does_not_attach_to_next_base(self):
        source = glyphs("विकल्प")
        # In logical text the vowel belongs to व and its visual glyph lies left
        # of that consonant, not next to the following क target.
        source[0].update(origin=[1.0, 10.0], bbox=[1.0, 0.0, 2.0, 12.0])
        source[1].update(origin=[0.0, 10.0], bbox=[0.0, 0.0, 1.0, 12.0])
        result = reorder_span(source, font_verified=True, size=10)
        self.assertEqual(result["text"], "विकल्प")
        self.assertFalse(result["changes"])

    def test_digits_and_already_logical_or_integrated_matra_stay_unchanged(self):
        source = glyphs("1947 0.2% कि")
        source[-1]["glyph_id"] = -1
        result = reorder_span(source, font_verified=True, size=10)
        self.assertEqual(result["text"], "1947 0.2% कि")
        self.assertFalse(result["changes"])

    def test_changed_original_or_derivative_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "original_hash"):
            build_candidate(b"wrong", b"wrong")

    def test_pinned_local_candidate_preserves_originals_and_remains_blocked(self):
        root = Path(__file__).resolve().parents[1]
        original = root / "data/next_stage/candidates/nba_abs_20261031/data/raw/nba/15beed19974f0948db417a6a0be39d147f8ff5b1871824888a2062b37a4f20e0.pdf"
        derivative = root / "data/next_stage/nba_fidelity_20261101/font_analysis/derived-unreviewed-cmap.pdf"
        if not original.exists() or not derivative.exists():
            self.skipTest("pinned local source/derivative unavailable")
        a, b = original.read_bytes(), derivative.read_bytes()
        result = build_candidate(a, b)
        self.assertEqual(len(result["pages"]), 28)
        self.assertFalse(result["certified"])
        self.assertFalse(result["indexing_allowed"])
        self.assertTrue(result["changes"])
        self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), result["original_sha256"])
        self.assertEqual(hashlib.sha256(derivative.read_bytes()).hexdigest(), result["cmap_pdf_sha256"])
        text = "\n".join(p["candidate_text"] for p in result["pages"])
        for word in ("अधिनियम", "प्राधिकरण", "विनियम", "विकल्प"):
            self.assertIn(word, text)
        for page in result["pages"]:
            self.assertTrue(page["character_inventory_preserved"])
            self.assertEqual(page["human_review_status"], "pending")
            self.assertIn("nontext_regions", page)
            for block in page["blocks"]:
                for line in block["lines"]:
                    for span in line["spans"]:
                        again = reorder_span([span["glyphs"][i] for i in span["source_order"]],
                                             font_verified=bool(span["font_evidence"]), size=span["font_size"])
                        self.assertEqual(again["text"], span["after"])
                        self.assertFalse(again["changes"])


if __name__ == "__main__":
    unittest.main()
