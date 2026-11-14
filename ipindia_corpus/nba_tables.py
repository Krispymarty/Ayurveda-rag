"""Pinned NBA ruling/cell evidence, kept outside admission and serving.

Cell geometry, numeric identity, language fidelity and legal meaning are separate
checks. This module supplies quarantined evidence, never certified passages.
"""
from __future__ import annotations

import hashlib
import math
import re

from .nba_fidelity import ORIGINAL_SHA256


CMAP_SHA256 = "0aa9d079734f9a6e14e93d339d94aefa57275970a25fe7ddf432dc8eab4e1656"
VERSION = "nba-original-ruling-cells-v1"
# Measured ruling bounds in this exact hash-pinned PDF, not text-derived guesses.
RULING_REGIONS = {2: (58, 608, 537, 775), 4: (58, 82, 537, 263),
                  17: (58, 261, 537, 410), 18: (58, 320, 537, 469),
                  9: (58, 59, 537, 530), 22: (58, 163, 537, 620)}
FORM_PAGES = {"A": ([8, 9], [22]), "B": ([10, 11], [23, 24]),
              "C": ([12], [25]), "D": ([13], [26]),
              "E": ([14], [27]), "F": ([15, 16], [28])}


def numeric_tokens(text):
    """Literal ASCII numerals; a percent is not interchangeable with an amount."""
    return re.findall(r"(?<![\d.])\d+(?:\.\d+)?%?(?!\d)", text)


class _CellPage:
    """One function-local TextPage for repeated cells, never a persistent cache."""
    def __init__(self, page):
        self.page = page
        self.textpage = page.get_textpage()

    def get_textbox(self, rectangle):
        return self.page.get_textbox(rectangle, textpage=self.textpage)


def _rect(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 4
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)
            or value[0] >= value[2] or value[1] >= value[3]):
        raise ValueError("invalid_cell_geometry")
    return list(value)


def validate_grid(rows, tolerance=0.05):
    """Require an actual rectangular ruling, not a guessed flattened column map."""
    if not rows or not rows[0]:
        raise ValueError("empty_cell_grid")
    width = len(rows[0])
    previous = None
    for row in rows:
        if len(row) != width:
            raise ValueError("nonrectangular_cell_grid")
        cells = [_rect(cell) for cell in row]
        for left, right in zip(cells, cells[1:]):
            if abs(left[2] - right[0]) > tolerance:
                raise ValueError("ambiguous_column_gap_or_overlap")
        if any(abs(c[1] - cells[0][1]) > tolerance or abs(c[3] - cells[0][3]) > tolerance for c in cells):
            raise ValueError("ambiguous_row_bounds")
        if previous:
            if abs(previous[0][3] - cells[0][1]) > tolerance:
                raise ValueError("ambiguous_row_gap_or_overlap")
            if any(abs(a[0] - b[0]) > tolerance or abs(a[2] - b[2]) > tolerance
                   for a, b in zip(previous, cells)):
                raise ValueError("columns_change_between_rows")
        previous = cells
    return {"columns": width, "rows": len(rows)}


def _cell(page, recovered, rectangle, row, column):
    rectangle = _rect(rectangle)
    before = page.get_textbox(rectangle).strip()
    after = recovered.get_textbox(rectangle).strip()
    return {"row": row, "column": column, "bbox": rectangle,
            "original_sha256": ORIGINAL_SHA256, "cmap_sha256": CMAP_SHA256,
            "original_text": before, "cmap_recovered_text": after,
            "numeric_tokens_original": numeric_tokens(before),
            "numeric_tokens_cmap": numeric_tokens(after),
            "blank": not before and not after,
            "text_fidelity": "pending_human_unicode_and_layout_review"}


def _grid(page, recovered, rows):
    shape = validate_grid(rows)
    return {**shape, "row_cells": [
        {"row": r + 1, "cells": [_cell(page, recovered, rect, r + 1, c + 1)
                                 for c, rect in enumerate(row)]}
        for r, row in enumerate(rows)]}


def _physical_rows(page, recovered, table_rows):
    # Form field cells can merge across columns. Preserve those rectangles rather
    # than assigning nonexistent columns to a detector's union-of-layouts grid.
    return [{"physical_row": r + 1, "cells": [
        {**_cell(page, recovered, rect, r + 1, None), "detector_slot": c + 1,
         "column_assignment": "not_inferred_for_merged_form_cells"}
        for c, rect in enumerate(row.cells) if rect is not None]}
        for r, row in enumerate(table_rows)]


def _context(page, recovered, bbox):
    x0, y0, x1, y1 = bbox
    return {"above": {"bbox": [x0, 55, x1, max(55.01, y0)],
                       "original_text": page.get_textbox((x0, 55, x1, y0)),
                       "cmap_recovered_text": recovered.get_textbox((x0, 55, x1, y0))},
            "below": {"bbox": [x0, y1, x1, 800],
                       "original_text": page.get_textbox((x0, y1, x1, 800)),
                       "cmap_recovered_text": recovered.get_textbox((x0, y1, x1, 800))},
            "qualification_completeness": "pending; inspect linked preceding/following pages"}


def _benefit_table(page, recovered, table, number):
    grid = _grid(page, recovered, [row.cells for row in table.rows])
    if grid["rows"] != 6 or grid["columns"] != 3:
        raise ValueError("unexpected_benefit_table_shape")
    rows = grid["row_cells"]
    bands = [["5"], ["5", "50"], ["50", "250"], ["250"]]
    rates = [[], ["0.2%"], ["0.4%"], ["0.6%"]]
    checks = []
    for serial, (row, band, rate) in enumerate(zip(rows[2:], bands, rates), 1):
        cells = row["cells"]
        expected = [[str(serial)], band, rate]
        checks.append(all(c["numeric_tokens_original"] == wanted == c["numeric_tokens_cmap"]
                          for c, wanted in zip(cells, expected)))
    if not all(checks):
        raise ValueError("benefit_table_numeric_identity_mismatch")
    return {"page": number, "language": "hi" if number < 16 else "en", "bbox": list(table.bbox),
            "rows": rows, "columns": 3, "row_column_geometry_passed": True,
            "numeric_identity_passed": True,
            "numeric_check_basis": "Literal original/CMap cell tokens match rendered-original assistant spot checks; not human approval",
            "first_rate_literal": rows[2]["cells"][2]["cmap_recovered_text"],
            "first_rate_translation": None, "context": _context(page, recovered, table.bbox),
            "text_fidelity": "pending_human_unicode_and_qualification_review",
            "legal_effect_verified": False, "indexing_allowed": False}


def _form_a(page, recovered, table, number):
    found = []
    for index, row in enumerate(table.rows):
        real = [cell for cell in row.cells if cell is not None]
        labels = [re.sub(r"\s+", "", page.get_textbox(cell)) for cell in real]
        if labels == [f"({n})" for n in range(1, 9)]:
            found.append(index)
    if len(found) != 1 or found[0] < 1 or found[0] + 1 >= len(table.rows):
        raise ValueError("form_a_numbered_eight_column_ruling_not_unique")
    index = found[0]
    source_rows = [[cell for cell in row.cells if cell is not None]
                   for row in table.rows[index - 1:index + 2]]
    resource = _grid(page, recovered, source_rows)
    if resource["columns"] != 8:
        raise ValueError("form_a_resource_columns_changed")
    blank = all(cell["blank"] for cell in resource["row_cells"][2]["cells"])
    if not blank:
        raise ValueError("form_a_blank_entry_row_changed")
    formula = resource["row_cells"][0]["cells"][6]["original_text"]
    if "7=(5)-(6)" not in re.sub(r"\s+", "", formula):
        raise ValueError("form_a_printed_formula_unresolved")
    return {"page": number, "language": "hi" if number < 16 else "en",
            "resource_grid": resource, "blank_entry_cells_preserved": True,
            "applicant_field_rows": _physical_rows(page, recovered, table.rows[:index - 1]),
            "trailing_spanning_blank_rows": _physical_rows(page, recovered, table.rows[index + 2:]),
            "printed_formula_cell": formula, "formula_interpretation": None,
            "declaration_context": _context(page, recovered, table.bbox)["below"],
            "detector_union_column_count": table.col_count,
            "warning": "The whole-page detector unions 3-column and 8-column layouts; its 10 columns are not a real Form A schema.",
            "human_review_completed": False, "indexing_allowed": False}


def _literal_heading_y(page, literal):
    hits = [line["bbox"][1] for block in page.get_text("dict", sort=True)["blocks"]
            for line in block.get("lines", [])
            if "".join(span["text"] for span in line["spans"]).strip() == literal]
    if len(hits) != 1:
        raise ValueError("form_or_language_boundary_not_unique")
    return hits[0]


def _page_lines(page, recovered, number, form):
    before = page.get_text("dict", sort=True)["blocks"]
    after = recovered.get_text("dict", sort=True)["blocks"]
    old = [(line["bbox"], "".join(s["text"] for s in line["spans"]))
           for block in before for line in block.get("lines", [])]
    new = [(line["bbox"], "".join(s["text"] for s in line["spans"]))
           for block in after for line in block.get("lines", [])]
    if len(old) != len(new) or any(a[0] != b[0] for a, b in zip(old, new)):
        raise ValueError("form_page_line_geometry_changed")
    start, end = 55.0, 800.0
    if form == "A" and number in (8, 22):
        start = _literal_heading_y(page, "प्रारूप क" if number == 8 else "FORM A")
    if form == "F" and number == 16:
        end = _literal_heading_y(page, "NATIONAL BIODIVERSITY AUTHORITY")
    lines = [{"bbox": list(a[0]), "original_text": a[1], "cmap_recovered_text": b[1]}
             for a, b in zip(old, new) if a[0][1] >= start - .01 and a[0][3] < end]
    return {"page": number, "content_bbox": [0, start, page.rect.width, end], "lines": lines,
            "text_fidelity": "pending", "preserve_as_form_not_independent_obligations": True}


def build_table_evidence(original_bytes, cmap_bytes):
    """Return immutable-source-bound draft evidence. Does no I/O or admission."""
    if hashlib.sha256(original_bytes).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("original_hash_mismatch")
    if hashlib.sha256(cmap_bytes).hexdigest() != CMAP_SHA256:
        raise ValueError("cmap_derivative_hash_mismatch")
    import fitz
    with fitz.open(stream=original_bytes, filetype="pdf") as original, fitz.open(stream=cmap_bytes, filetype="pdf") as cmap:
        if len(original) != 28 or len(cmap) != 28:
            raise ValueError("original_page_count_changed")
        benefit = []
        for number in (2, 4, 17, 18):
            page, recovered = original[number - 1], cmap[number - 1]
            tables = page.find_tables(clip=RULING_REGIONS[number]).tables
            if len(tables) != 1:
                raise ValueError("benefit_table_ruling_not_unique")
            benefit.append(_benefit_table(_CellPage(page), _CellPage(recovered), tables[0], number))
        form_a = []
        for number in (9, 22):
            page, recovered = original[number - 1], cmap[number - 1]
            tables = page.find_tables(clip=RULING_REGIONS[number]).tables
            if len(tables) != 1:
                raise ValueError("form_a_ruling_not_unique")
            form_a.append(_form_a(_CellPage(page), _CellPage(recovered), tables[0], number))
        forms = []
        for label, (hi, en) in FORM_PAGES.items():
            forms.append({"form": label, "hi_pages": hi, "en_pages": en,
                          "representations": [
                              {"language": language, "pages": [_page_lines(original[n - 1], cmap[n - 1], n, label) for n in numbers]}
                              for language, numbers in (("hi", hi), ("en", en))],
                          "same_instrument": ORIGINAL_SHA256, "not_independent_source_versions": True,
                          "human_review_completed": False})
    return {"version": VERSION, "original_sha256": ORIGINAL_SHA256, "cmap_sha256": CMAP_SHA256,
            "benefit_tables": benefit, "form_a": form_a, "forms": forms,
            "parallel_renditions": [{"hi_page": 2, "en_page": 17, "observed_table_context": "regulation4"},
                                    {"hi_page": 4, "en_page": 18, "observed_table_context": "regulation5(7)"}],
            "form_boundary_correction": "Hindi Form C is page12; page13 starts Form D, not a Form C continuation. English C25/D26.",
            "bilingual_boundary": "Page16 contains the Hindi Form F continuation before the English notification; do not double-count renditions.",
            "certified": False, "human_reviews_completed": 0, "indexing_allowed": False,
            "activation_allowed": False, "legal_status": "unknown"}
