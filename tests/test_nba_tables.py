"""Synthetic geometry plus the pinned local original; no legal certification."""
import hashlib
from pathlib import Path
import unittest

from ipindia_corpus.nba_tables import (
    ORIGINAL_SHA256, _benefit_table, build_table_evidence, numeric_tokens, validate_grid,
)


class NBATableTests(unittest.TestCase):
    def test_rectangular_grid_has_distinct_rows_and_columns(self):
        rows = [[(0, 0, 10, 10), (10, 0, 25, 10)],
                [(0, 10, 10, 20), (10, 10, 25, 20)]]
        self.assertEqual(validate_grid(rows), {"columns": 2, "rows": 2})

    def test_ambiguous_missing_overlapping_or_nonfinite_grid_rejected(self):
        for rows in ([[None]], [[(0, 0, 10, 10), (9, 0, 20, 10)]],
                     [[(0, 0, float("inf"), 10)]], [[(0, 0, 10, 10)], [(0, 9, 10, 20)]],
                     [[(False, 0, 10, 10)]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_grid(rows)

    def test_numeric_tokens_do_not_conflate_percentage_with_turnover(self):
        self.assertEqual(numeric_tokens("Above 5 crore to 50 crore / 0.2%"), ["5", "50", "0.2%"])
        self.assertNotEqual(numeric_tokens("5%"), numeric_tokens("5"))

    def test_wrong_original_or_derivative_hash_stops_before_parsing(self):
        with self.assertRaisesRegex(ValueError, "original_hash"):
            build_table_evidence(b"synthetic wrong original", b"wrong derivative")

    def test_correct_original_with_tampered_derivative_is_rejected(self):
        root = Path(__file__).resolve().parents[1]
        original = root / "data/next_stage/candidates/nba_abs_20261031/data/raw/nba" / (ORIGINAL_SHA256 + ".pdf")
        if not original.exists():
            self.skipTest("Pinned local original absent")
        with self.assertRaisesRegex(ValueError, "cmap_derivative_hash"):
            build_table_evidence(original.read_bytes(), b"tampered derivative")

    def test_same_number_bag_with_rates_assigned_to_wrong_rows_fails(self):
        source = [["heading", "heading", "heading"], ["(1)", "(2)", "(3)"],
                  ["1.", "Up to 5 crore", "Nil"], ["2.", "Above 5 crore to 50 crore", "0.4%"],
                  ["3.", "Above 50 crore to 250 crore", "0.2%"], ["4.", "Above 250 crore", "0.6%"]]
        rectangles = [[(c * 10, r * 10, (c + 1) * 10, (r + 1) * 10) for c in range(3)] for r in range(6)]
        lookup = {rect: source[r][c] for r, row in enumerate(rectangles) for c, rect in enumerate(row)}
        class Page:
            def get_textbox(self, rectangle):
                return lookup.get(tuple(rectangle), "")
        table = type("Table", (), {"rows": [type("Row", (), {"cells": row})() for row in rectangles],
                                    "bbox": (0, 0, 30, 60)})()
        with self.assertRaisesRegex(ValueError, "numeric_identity"):
            _benefit_table(Page(), Page(), table, 17)

    def test_source_table_and_form_structure_is_candidate_only(self):
        root = Path(__file__).resolve().parents[1]
        original = root / "data/next_stage/candidates/nba_abs_20261031/data/raw/nba" / (ORIGINAL_SHA256 + ".pdf")
        derivative = root / "data/next_stage/nba_fidelity_20261101/font_analysis/derived-unreviewed-cmap.pdf"
        if not original.exists() or not derivative.exists():
            self.skipTest("Pinned local original/derivative absent")
        before = hashlib.sha256(original.read_bytes()).hexdigest()
        evidence = build_table_evidence(original.read_bytes(), derivative.read_bytes())
        self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), before)
        self.assertEqual([t["page"] for t in evidence["benefit_tables"]], [2, 4, 17, 18])
        self.assertTrue(all(t["numeric_identity_passed"] for t in evidence["benefit_tables"]))
        self.assertTrue(all(t["row_column_geometry_passed"] for t in evidence["benefit_tables"]))
        for table in evidence["benefit_tables"]:
            self.assertEqual([r["cells"][2]["numeric_tokens_original"] for r in table["rows"][2:]],
                             [[], ["0.2%"], ["0.4%"], ["0.6%"]])
        self.assertEqual([t["resource_grid"]["columns"] for t in evidence["form_a"]], [8, 8])
        self.assertTrue(all(t["blank_entry_cells_preserved"] for t in evidence["form_a"]))
        forms = {f["form"]: f for f in evidence["forms"]}
        self.assertEqual(forms["C"]["hi_pages"], [12])
        self.assertEqual(forms["D"]["hi_pages"], [13])
        self.assertEqual(forms["C"]["en_pages"], [25])
        self.assertEqual(forms["D"]["en_pages"], [26])
        hindi_f16 = forms["F"]["representations"][0]["pages"][1]
        self.assertLess(hindi_f16["content_bbox"][3], 300)
        self.assertFalse(any("NATIONAL BIODIVERSITY AUTHORITY" in line["original_text"]
                             or "Short title and commencement" in line["original_text"]
                             for line in hindi_f16["lines"]))
        hindi_a8 = forms["A"]["representations"][0]["pages"][0]
        self.assertGreater(hindi_a8["content_bbox"][1], 570)
        self.assertFalse(evidence["certified"])
        self.assertFalse(evidence["indexing_allowed"])
        self.assertEqual(evidence["human_reviews_completed"], 0)


if __name__ == "__main__":
    unittest.main()
