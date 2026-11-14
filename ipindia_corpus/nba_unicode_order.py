"""Pinned NBA glyph-order candidate; never a publication or admission mechanism.

The document draws the known pre-base i glyph before its consonant cluster.
Only that glyph in the verified embedded font may be permuted. No dictionaries,
word substitutions, inferred characters, normalisation or LLM output are used.
"""
from collections import Counter, defaultdict
import hashlib
import io
import math
import re

from .nba_fidelity import ORIGINAL_SHA256, _descendant, parse_to_unicode

CMAP_PDF_SHA256 = "0aa9d079734f9a6e14e93d339d94aefa57275970a25fe7ddf432dc8eab4e1656"
FONT_SHA256 = "1e6f84730ee3c9fdc40ce71d27329f94c3a9d13a05747411cf7a8fc9febedfb9"
VERSION = "nba-bound-prebase-i-span-permutation-v2"
I_MATRA = "\u093f"
VIRAMA = "\u094d"
NUKTA = "\u093c"
JOINERS = {"\u200c", "\u200d"}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _font_key(name):
    return re.sub(r"\s+", "", re.sub(r"^[A-Z]{6}\+", "", name)).lower()


def _consonant(char):
    return "\u0915" <= char <= "\u0939" or "\u0958" <= char <= "\u095f" or "\u0978" <= char <= "\u097f"


def _dependent_vowel(char):
    return char in "\u093a\u093b\u093e\u093f\u0940\u0941\u0942\u0943\u0944\u0945\u0946\u0947\u0948\u0949\u094a\u094b\u094c\u094e\u094f\u0955\u0956\u0957\u0962\u0963"


def _numbers(text):
    return re.findall(r"[0-9\u0966-\u096f]+(?:[.,][0-9\u0966-\u096f]+)*(?:%)?", text)


def _continuation_geometry(char, previous, tolerance):
    return (abs(char["bbox"][2] - char["bbox"][0]) < 0.001
            and abs(char["origin"][0] - previous["bbox"][2]) <= tolerance
            and abs(char["bbox"][0] - char["origin"][0]) <= tolerance)


def _cluster_end(chars, start):
    if start >= len(chars) or not _consonant(chars[start]["char"]):
        return None, "prebase_i_without_following_consonant"
    end = start + 1
    if chars[start]["char"] == "\u0930" and end < len(chars) and chars[end]["char"] == VIRAMA:
        return None, "leading_ra_virama_reph_ambiguous"
    while True:
        if end < len(chars) and chars[end]["char"] == NUKTA:
            end += 1
        if end < len(chars) and chars[end]["char"] in JOINERS:
            return None, "joiner_cluster_requires_visual_review"
        if end >= len(chars) or chars[end]["char"] != VIRAMA:
            break
        end += 1
        if end < len(chars) and chars[end]["char"] in JOINERS:
            return None, "joiner_cluster_requires_visual_review"
        if end >= len(chars) or not _consonant(chars[end]["char"]):
            return None, "incomplete_virama_cluster"
        end += 1
    if end < len(chars) and _dependent_vowel(chars[end]["char"]):
        # Another physical pre-base i may start the NEXT syllable; all other
        # competing vowels are ambiguous and must not be resolved by spelling.
        if not (chars[end]["char"] == I_MATRA and chars[end]["glyph_id"] == 1439
                and end + 1 < len(chars) and _consonant(chars[end + 1]["char"])):
            return None, "competing_dependent_vowel"
    return end, None


def reorder_span(chars, *, font_verified, size):
    """Return a reversible source-index permutation confined to one PDF span."""
    if type(font_verified) is not bool or not isinstance(chars, list) or len(chars) > 20000:
        raise ValueError("invalid_span_contract")
    if type(size) not in (int, float) or not math.isfinite(size) or not 0 < size <= 1000:
        raise ValueError("invalid_span_size")
    for item in chars:
        if (not isinstance(item, dict) or not isinstance(item.get("char"), str) or len(item["char"]) != 1
                or item.get("glyph_id") is not None and type(item["glyph_id"]) is not int
                or len(item.get("origin", [])) != 2 or len(item.get("bbox", [])) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v)
                       for v in item["origin"] + item["bbox"])):
            raise ValueError("invalid_glyph_contract")
    order, changes, unresolved = list(range(len(chars))), [], []
    for index, item in enumerate(chars):
        if item["char"] != I_MATRA:
            continue
        reason = None
        if not font_verified or item["glyph_id"] != 1439:
            reason = "unbound_or_integrated_i_glyph_not_reordered"
        elif item["bbox"][2] - item["bbox"][0] <= 0.001:
            reason = "empty_alias_font_glyph_not_reordered"
        elif index and chars[index - 1]["char"] == I_MATRA:
            reason = "adjacent_i_marks_ambiguous"
        elif index and _consonant(chars[index - 1]["char"]) and chars[index - 1]["origin"][0] >= item["origin"][0]:
            reason = "already_logical_or_backtracking_glyph_order"
        else:
            end, reason = _cluster_end(chars, index + 1)
            if reason is None:
                selected = chars[index:end]
                if any(c["glyph_id"] is None or len(c.get("trace_refs", [])) != 1 for c in selected):
                    reason = "missing_unique_glyph_trace"
                else:
                    physical = [c for c in selected if c["glyph_id"] >= 0]
                    tolerance = max(0.15, size * 0.08)
                    if any(abs(c["origin"][1] - item["origin"][1]) > tolerance for c in selected):
                        reason = "baseline_or_line_boundary"
                    elif selected[1]["glyph_id"] < 0 or any(
                            c["glyph_id"] == -1 and not _continuation_geometry(c, selected[j - 1], tolerance)
                            for j, c in enumerate(selected) if j):
                        reason = "logical_continuation_geometry_ambiguous"
                    elif any(abs(b["origin"][0] - a["bbox"][2]) > tolerance
                             for a, b in zip(physical, physical[1:])):
                        reason = "noncontiguous_glyph_or_cell_boundary"
                if reason is None:
                    moved = list(range(index + 1, end)) + [index]
                    order.remove(index)
                    order.insert(order.index(end - 1) + 1, index)
                    box = [min(c["bbox"][0] for c in selected), min(c["bbox"][1] for c in selected),
                           max(c["bbox"][2] for c in selected), max(c["bbox"][3] for c in selected)]
                    changes.append({"source_indices": list(range(index, end)), "candidate_source_order": moved,
                                    "before": "".join(c["char"] for c in selected),
                                    "after": "".join(chars[i]["char"] for i in moved), "bbox": box,
                                    "trigger_glyph_id": 1439, "kind": "bound_prebase_i_cluster_permutation",
                                    "validation_status": "mechanical_candidate_pending_visual_review"})
        if reason:
            unresolved.append({"source_index": index, "bbox": item["bbox"], "reason": reason})
    text = "".join(chars[i]["char"] for i in order)
    if sorted(order) != list(range(len(chars))) or Counter(text) != Counter(c["char"] for c in chars):
        raise ValueError("source_permutation_integrity_failed")
    return {"text": text, "source_order": order, "changes": changes, "unresolved": unresolved}


def _font_program(embedded):
    from fontTools.ttLib import TTFont
    with TTFont(io.BytesIO(embedded), lazy=False) as table:
        name = table.getGlyphName(1439)
        candidates = {cp for subtable in table["cmap"].tables if subtable.isUnicode()
                      for cp, glyph in subtable.cmap.items() if table.getGlyphID(glyph) == 1439}
        return {"unicode_candidates": sorted(candidates), "glyph_1439_advance": table["hmtx"][name][0],
                "glyph_1439_has_outline": table["glyf"][name].numberOfContours != 0}


def _font_evidence(original, derived, page, programs):
    evidence = defaultdict(list)
    resources, visible_unique = [], True
    for font in page.get_fonts(full=True):
        if _font_key(font[3]) != "arialunicodems":
            continue
        embedded = original.extract_font(font[0])[3]
        digest = _sha(embedded)
        if digest not in programs:
            programs[digest] = _font_program(embedded)
        info = programs[digest]
        cmap_kind, cmap_ref = original.xref_get_key(font[0], "ToUnicode")
        cmap_raw = original.xref_stream(int(cmap_ref.split()[0])) if cmap_kind == "xref" else None
        mapped_i = [cid for cid, value in parse_to_unicode(cmap_raw).items() if I_MATRA in value] if cmap_raw else []
        resources.append({"font_xref": font[0], "font_type": font[2], "embedded_font_sha256": digest,
                          "pdf_encoding": list(original.xref_get_key(font[0], "Encoding")),
                          "resource_to_unicode_sha256": _sha(cmap_raw) if cmap_raw else None,
                          "resource_cids_mapping_i": mapped_i, **info})
        if _sha(derived.extract_font(font[0])[3]) != digest or digest != FONT_SHA256 and (
                info["glyph_1439_has_outline"] or info["glyph_1439_advance"] != 0 or mapped_i):
            visible_unique = False
    scope = {"visible_glyph_program_unique": visible_unique, "resources": resources,
             "basis": "All same-family resources inspected; non-bound glyph1439 must have empty outline and zero advance"}
    if not visible_unique:
        return evidence, scope
    for font in page.get_fonts(full=True):
        xref = font[0]
        if font[2] != "Type0" or _font_key(font[3]) != "arialunicodems":
            continue
        embedded = original.extract_font(xref)[3]
        if _sha(embedded) != FONT_SHA256 or _sha(derived.extract_font(xref)[3]) != FONT_SHA256:
            continue
        if original.xref_get_key(xref, "Encoding") != ("name", "/Identity-H"):
            continue
        child = _descendant(original, xref)
        if original.xref_get_key(child, "CIDToGIDMap") != ("name", "/Identity"):
            continue
        cmap = int(original.xref_get_key(xref, "ToUnicode")[1].split()[0])
        old = parse_to_unicode(original.xref_stream(cmap))
        new = parse_to_unicode(derived.xref_stream(cmap))
        if (old.get(1439) != "\u091c" or new.get(1439) != I_MATRA
                or programs[FONT_SHA256]["unicode_candidates"] != [0x093F]):
            continue
        evidence[_font_key(font[3])].append({"font_xref": xref, "font_name": font[3], "cmap_xref": cmap,
                                           "embedded_font_sha256": FONT_SHA256, "glyph_id": 1439,
                                           "original_cmap_sha256": _sha(original.xref_stream(cmap)),
                                           "derived_cmap_sha256": _sha(derived.xref_stream(cmap))})
    return evidence, scope


def _glyph_lookup(page):
    lookup = defaultdict(list)
    traces = page.get_texttrace()
    for span_index, span in enumerate(traces):
        for char_index, (cp, gid, origin, _) in enumerate(span["chars"]):
            key = (_font_key(span["font"]), round(origin[0], 3), round(origin[1], 3), chr(cp))
            lookup[key].append((gid, span_index, char_index))
    return lookup, traces


def build_candidate(original_bytes, cmap_pdf_bytes):
    """Create an in-memory, immutable-serialisable candidate, not certified text."""
    if _sha(original_bytes) != ORIGINAL_SHA256:
        raise ValueError("original_hash_mismatch")
    if _sha(cmap_pdf_bytes) != CMAP_PDF_SHA256:
        raise ValueError("cmap_derivative_hash_mismatch")
    import fitz
    pages, all_changes, programs = [], [], {}
    with fitz.open(stream=original_bytes, filetype="pdf") as original, fitz.open(stream=cmap_pdf_bytes, filetype="pdf") as derived:
        if len(original) != 28 or len(derived) != 28:
            raise ValueError("unexpected_page_count")
        for n, page in enumerate(derived, 1):
            fonts, font_scope = _font_evidence(original, derived, original[n - 1], programs)
            lookup, traces = _glyph_lookup(page)
            blocks, page_changes, page_unresolved, before_lines, after_lines = [], [], [], [], []
            nontext_regions = []
            for bi, block in enumerate(page.get_text("rawdict", sort=True)["blocks"]):
                if "lines" not in block:
                    nontext_regions.append({"block": bi, "type": block.get("type"), "bbox": list(block["bbox"]),
                                            "image_bytes_sha256": _sha(block["image"]) if isinstance(block.get("image"), bytes) else None,
                                            "status": "untranscribed_region_requires_visual_review"})
                    continue
                lines = []
                for li, line in enumerate(block["lines"]):
                    spans, before_line, after_line = [], [], []
                    horizontal = abs(line["dir"][0] - 1) < 0.001 and abs(line["dir"][1]) < 0.001
                    for si, span in enumerate(line["spans"]):
                        chars = []
                        for char in span["chars"]:
                            key = (_font_key(span["font"]), round(char["origin"][0], 3), round(char["origin"][1], 3), char["c"])
                            trace = lookup.get(key, [])
                            # rawdict positions zero-width logical continuation
                            # characters at the physical glyph's advance end;
                            # texttrace anchors them at that glyph's origin.
                            # Bind only the immediately following trace char,
                            # never a nearest-coordinate/character guess.
                            if (not trace and chars and len(chars[-1]["trace_refs"]) == 1
                                    and _continuation_geometry({"origin": char["origin"], "bbox": char["bbox"]},
                                                               chars[-1], max(0.15, span["size"] * 0.08))):
                                ti, ci = chars[-1]["trace_refs"][0]
                                candidates = traces[ti]["chars"]
                                if ci + 1 < len(candidates):
                                    cp, gid, origin, _ = candidates[ci + 1]
                                    if (gid == -1 and chr(cp) == char["c"]
                                            and abs(origin[1] - char["origin"][1]) < 0.01
                                            and _font_key(traces[ti]["font"]) == _font_key(span["font"])):
                                        trace = [(gid, ti, ci + 1)]
                            gids = {g for g, _, _ in trace}
                            chars.append({"char": char["c"], "glyph_id": next(iter(gids)) if len(gids) == 1 else None,
                                          "origin": list(char["origin"]), "bbox": list(char["bbox"]),
                                          "trace_refs": [[s, c] for _, s, c in trace]})
                        result = reorder_span(chars, font_verified=bool(fonts.get(_font_key(span["font"]))) and horizontal,
                                              size=span["size"])
                        before = "".join(c["char"] for c in chars)
                        record = {"block": bi, "line": li, "span": si, "font": span["font"], "font_size": span["size"], "bbox": list(span["bbox"]),
                                  "font_evidence": fonts.get(_font_key(span["font"]), []), "before": before,
                                  "after": result["text"], "source_order": result["source_order"], "glyphs": chars,
                                  "before_sha256": _sha(before.encode()), "after_sha256": _sha(result["text"].encode()),
                                  "changes": result["changes"], "unresolved": result["unresolved"]}
                        spans.append(record)
                        before_line.append(before)
                        after_line.append(result["text"])
                        for change in result["changes"]:
                            page_changes.append({**change, "page": n, "block": bi, "line": li, "span": si,
                                                 "font_evidence": record["font_evidence"],
                                                 "original_sha256": ORIGINAL_SHA256, "cmap_pdf_sha256": CMAP_PDF_SHA256})
                        page_unresolved.extend({**item, "page": n, "block": bi, "line": li, "span": si}
                                               for item in result["unresolved"])
                    lines.append({"line": li, "bbox": list(line["bbox"]), "spans": spans})
                    before_lines.append("".join(before_line))
                    after_lines.append("".join(after_line))
                blocks.append({"block": bi, "bbox": list(block["bbox"]), "lines": lines})
            before, after = "\n".join(before_lines), "\n".join(after_lines)
            if Counter(before) != Counter(after) or _numbers(before) != _numbers(after):
                raise ValueError("page_inventory_integrity_failed")
            pages.append({"page": n, "original_text": original[n - 1].get_text(sort=True),
                          "cmap_recovered_text": page.get_text(sort=True), "span_flow_before": before,
                          "candidate_text": after, "candidate_text_sha256": _sha(after.encode()),
                          "span_flow_before_sha256": _sha(before.encode()), "blocks": blocks,
                          "font_scope": font_scope,
                          "nontext_regions": nontext_regions,
                          "changes": page_changes, "unresolved_spans": page_unresolved,
                          "character_inventory_preserved": True, "numeric_tokens_before": _numbers(before),
                          "numeric_tokens_after": _numbers(after), "numeric_token_sequence_preserved": True,
                          "inventory_scope": "extracted_text_spans_only; not proof of all printed/raster content",
                          "human_review_status": "pending", "layout_status": "block_line_span_flow_not_verified_paragraph_or_table_reading_order",
                          "remaining_checks": ["all_page_fidelity_review_pending", "ligatures_reph_half_forms_unverified",
                                               "tables_forms_continuations_and_qualifications_unverified"],
                          "language_representation": "Hindi/English renditions of one instrument, never independent corroboration"})
            all_changes.extend(page_changes)
    return {"version": VERSION, "original_sha256": ORIGINAL_SHA256, "cmap_pdf_sha256": CMAP_PDF_SHA256,
            "pages": pages, "changes": all_changes, "human_reviews_completed": 0,
            "certified": False, "indexing_allowed": False, "activation_allowed": False,
            "method": "Font/glyph-bound pre-base i permutation within contiguous consonant clusters in one span; no characters added/deleted or lexical substitutions.",
            "warning": "Mechanical candidate only. Glyph and numeric inventory preservation does not prove wording, paragraph/table semantics or legal relevance."}
