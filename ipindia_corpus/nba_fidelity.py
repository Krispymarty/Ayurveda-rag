"""Quarantined, original-bound NBA font decoding; never an admission mechanism.

Only unambiguous embedded-font/CMap contradictions are repaired. Hindi shaping,
table semantics and legal relevance still require visual/human review. This module
is deliberately absent from service, publication and indexing imports.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import io
import re


ORIGINAL_SHA256 = "15beed19974f0948db417a6a0be39d147f8ff5b1871824888a2062b37a4f20e0"
VERSION = "nba-embedded-cmap-quarantined-v1"


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _unicode(value):
    try:
        raw = bytes.fromhex(value)
        if not raw or len(raw) % 2 or len(raw) > 64:
            raise ValueError("invalid_utf16")
        return raw.decode("utf-16-be", errors="strict")
    except (UnicodeError, ValueError) as exc:
        raise ValueError("invalid_cmap_unicode") from exc


def parse_to_unicode(raw):
    """Parse the bounded bfchar/bfrange subset, retaining multi-codepoint glyphs."""
    if not isinstance(raw, bytes) or len(raw) > 1_000_000:
        raise ValueError("cmap_size_limit")
    text = re.sub(r"%[^\r\n]*", "", raw.decode("ascii", errors="strict"))
    result = {}

    def add(cid, value):
        if not 0 <= cid <= 65535 or cid in result:
            raise ValueError("ambiguous_cmap_code")
        result[cid] = value

    for block in re.finditer(r"(\d+)\s+begin(bfchar|bfrange)\b(.*?)end\2\b", text, re.S):
        count, kind, body = int(block[1]), block[2], block[3]
        if count > 65536:
            raise ValueError("cmap_range_limit")
        tokens = re.findall(r"<[0-9A-Fa-f]+>|\[|\]", body)
        if re.sub(r"<[0-9A-Fa-f]+>|\[|\]|\s+", "", body):
            raise ValueError("unsupported_cmap_tokens")
        index, rows = 0, 0

        def hex_value():
            nonlocal index
            if index >= len(tokens) or not tokens[index].startswith("<"):
                raise ValueError("truncated_cmap")
            value = tokens[index][1:-1]
            index += 1
            return value

        while index < len(tokens):
            start = int(hex_value(), 16)
            if kind == "bfchar":
                add(start, _unicode(hex_value()))
            else:
                end = int(hex_value(), 16)
                if not 0 <= start <= end <= 65535:
                    raise ValueError("cmap_range_limit")
                if index < len(tokens) and tokens[index] == "[":
                    index += 1
                    for cid in range(start, end + 1):
                        add(cid, _unicode(hex_value()))
                    if index >= len(tokens) or tokens[index] != "]":
                        raise ValueError("cmap_array_length_mismatch")
                    index += 1
                else:
                    base = hex_value()
                    _unicode(base)
                    width = len(base) // 2
                    for offset, cid in enumerate(range(start, end + 1)):
                        try:
                            value = (int(base, 16) + offset).to_bytes(width, "big")
                        except OverflowError as exc:
                            raise ValueError("cmap_unicode_overflow") from exc
                        add(cid, _unicode(value.hex()))
            rows += 1
        if rows != count:
            raise ValueError("cmap_declared_count_mismatch")
    if not result:
        raise ValueError("unsupported_or_empty_cmap")
    return result


def repair_mapping(mapping, unicode_by_gid):
    """Identity CID→GID only: no lexicon, character substitution list or LLM."""
    repaired, changes = dict(mapping), []
    for cid, before in sorted(mapping.items()):
        candidates = unicode_by_gid.get(cid, set())
        # Keep multi-codepoint shaping mappings and ambiguous Unicode aliases.
        if len(before) != 1 or len(candidates) != 1 or ord(before) in candidates:
            continue
        after = chr(next(iter(candidates)))
        if 0xD800 <= ord(after) <= 0xDFFF:
            continue
        repaired[cid] = after
        changes.append({"cid": cid, "glyph_id": cid, "before": before, "after": after})
    return repaired, changes


def _cmap_bytes(mapping):
    parts = ["/CIDInit /ProcSet findresource begin", "12 dict begin", "begincmap",
             "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def",
             "/CMapName /AyurIP-Derived-Unreviewed def", "/CMapType 2 def",
             "1 begincodespacerange", "<0000> <FFFF>", "endcodespacerange"]
    items = sorted(mapping.items())
    for offset in range(0, len(items), 100):
        batch = items[offset:offset + 100]
        parts.append(f"{len(batch)} beginbfchar")
        parts.extend(f"<{cid:04X}> <{value.encode('utf-16-be').hex().upper()}>" for cid, value in batch)
        parts.append("endbfchar")
    parts.extend(["endcmap", "CMapName currentdict /CMap defineresource pop", "end", "end"])
    return ("\n".join(parts) + "\n").encode("ascii")


def _descendant(document, font_xref):
    kind, value = document.xref_get_key(font_xref, "DescendantFonts")
    if kind == "xref":
        value = document.xref_object(int(value.split()[0]))
    refs = re.findall(r"(\d+)\s+0\s+R", value)
    if len(refs) != 1:
        raise ValueError("ambiguous_descendant_font")
    return int(refs[0])


def _font_key(name):
    return re.sub(r"\s+", "", re.sub(r"^[A-Z]{6}\+", "", name)).lower()


def _spans(page):
    return [(block_index, line_index, span_index, span)
            for block_index, block in enumerate(page.get_text("dict", sort=True)["blocks"])
            for line_index, line in enumerate(block.get("lines", []))
            for span_index, span in enumerate(line["spans"])]


def build_derivation(raw):
    """Return explicit review data and an in-memory derived PDF; write nothing.

    The derived PDF changes extraction metadata, not glyph outlines. Its signature
    cannot authenticate the modified PDF; the unmodified original remains authority.
    """
    if _sha(raw) != ORIGINAL_SHA256:
        raise ValueError("original_hash_mismatch")
    # Optional local analysis dependencies; no imports/dependencies added to serving.
    import fitz
    from fontTools.ttLib import TTFont

    with fitz.open(stream=raw, filetype="pdf") as original, fitz.open(stream=raw, filetype="pdf") as derived:
        if len(original) != 28 or original.is_encrypted:
            raise ValueError("unexpected_original_structure")
        fonts = {font[0]: font for page in original for font in page.get_fonts(full=True)}
        corrections, skipped = [], []
        patched_streams = {}
        for xref, font in sorted(fonts.items()):
            if font[2] != "Type0":
                continue
            try:
                if original.xref_get_key(xref, "Encoding") != ("name", "/Identity-H"):
                    raise ValueError("unsupported_font_encoding")
                child = _descendant(original, xref)
                if original.xref_get_key(child, "CIDToGIDMap") != ("name", "/Identity"):
                    raise ValueError("nonidentity_cid_to_gid")
                kind, value = original.xref_get_key(xref, "ToUnicode")
                if kind != "xref":
                    raise ValueError("missing_to_unicode")
                cmap_xref = int(value.split()[0])
                cmap = original.xref_stream(cmap_xref)
                mapping = parse_to_unicode(cmap)
                embedded = original.extract_font(xref)[3]
                unicode_by_gid = defaultdict(set)
                with TTFont(io.BytesIO(embedded), lazy=False) as table:
                    for subtable in table["cmap"].tables:
                        if subtable.isUnicode():
                            for codepoint, glyph in subtable.cmap.items():
                                unicode_by_gid[table.getGlyphID(glyph)].add(codepoint)
                repaired, changes = repair_mapping(mapping, unicode_by_gid)
                if not changes:
                    continue
                candidate = _cmap_bytes(repaired)
                if cmap_xref in patched_streams and patched_streams[cmap_xref] != candidate:
                    raise ValueError("shared_cmap_conflicting_repairs")
                patched_streams[cmap_xref] = candidate
                derived.update_stream(cmap_xref, candidate)
                corrections.append({"font_xref": xref, "font_name": font[3], "descendant_xref": child,
                                    "cid_to_gid": "Identity", "cmap_xref": cmap_xref,
                                    "embedded_font_sha256": _sha(embedded), "original_cmap_sha256": _sha(cmap),
                                    "derived_cmap_sha256": _sha(candidate), "changes": changes})
            except (KeyError, ValueError, UnicodeError) as exc:
                skipped.append({"font_xref": xref, "font_name": font[3], "reason": str(exc)})
        pdf_bytes = derived.tobytes(garbage=0, deflate=True, no_new_id=True)
        pages = []
        for number, before in enumerate(original, 1):
            after = derived[number - 1]
            before_text, after_text = before.get_text(sort=True), after.get_text(sort=True)
            old_pixels, new_pixels = before.get_pixmap(alpha=False), after.get_pixmap(alpha=False)
            before_hash, after_hash = _sha(old_pixels.samples), _sha(new_pixels.samples)
            font_xrefs = {font[0] for font in before.get_fonts(full=True)}
            local = [item for item in corrections if item["font_xref"] in font_xrefs]
            span_changes = []
            old_spans, new_spans = _spans(before), _spans(after)
            if len(old_spans) == len(new_spans):
                for old, new in zip(old_spans, new_spans):
                    block_index, line_index, span_index, a = old
                    z = new[3]
                    if a["text"] != z["text"]:
                        span_changes.append({"block": block_index, "line": line_index, "span": span_index, "font": a["font"],
                                             "bbox": list(a["bbox"]), "before": a["text"], "after": z["text"],
                                             "font_correction_xrefs": [f["font_xref"] for f in local
                                                if _font_key(f["font_name"]) == _font_key(a["font"])]})
            unresolved = ["mandatory_human_page_fidelity_review_pending", "legal_relevance_unreviewed"]
            if re.search(r"[\u0900-\u097f]", after_text):
                unresolved.extend(["hindi_font_ligatures_and_conjuncts_unverified",
                                   "prebase_matra_logical_order_unverified"])
            if re.search(r"\b(?:table|form|schedule)\b|%|\brupees\b|crore", after_text, re.I):
                unresolved.append("table_amounts_percentages_and_row_associations_unverified")
            if len(old_spans) != len(new_spans):
                unresolved.append("span_segmentation_changed_compare_full_page")
            pages.append({"page": number, "original_text": before_text, "cmap_recovered_text": after_text,
                          "original_text_sha256": _sha(before_text.encode("utf-8")),
                          "cmap_recovered_text_sha256": _sha(after_text.encode("utf-8")),
                          "original_render_pixel_sha256": before_hash, "derived_render_pixel_sha256": after_hash,
                          "render_pixels_unchanged": before_hash == after_hash,
                          "font_correction_xrefs": [f["font_xref"] for f in local],
                          "span_changes": span_changes, "unresolved": unresolved,
                          "human_review_completed": False})
        return {"version": VERSION, "original_sha256": ORIGINAL_SHA256, "derived_pdf_sha256": _sha(pdf_bytes),
                "certified": False, "indexing_allowed": False, "activation_allowed": False,
                "parser_is_original_legacy_or_v4": False, "human_reviews_completed": 0,
                "method": "Existing scalar ToUnicode contradictions corrected only from unambiguous embedded Unicode cmap under Identity CIDToGIDMap; no dictionary/LLM/word replacements or added missing mappings.",
                "limitation": "CMap recovery is not Hindi logical-order, table, legal or human fidelity certification. No original bytes or glyph outlines changed. The modified PDF is derived and its original digital signature is not valid evidence for the derivative.",
                "font_corrections": corrections, "fonts_not_repaired": skipped, "pages": pages}, pdf_bytes
