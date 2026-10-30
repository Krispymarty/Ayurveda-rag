"""Bounded question-reference parser; it never writes corpus anchors.

Offsets refer to the normalized QUESTION, never to archived evidence. A query
reference is a retrieval/coverage hypothesis, not a legal-applicability finding.
"""
from dataclasses import dataclass
import re
import unicodedata

VERSION = "question-references-v3-1"
MAX_RANGE = 12
LABEL = re.compile(r"\b(sections?|rules?|subsections?)\s+", re.I)
ROOT = re.compile(r"(\d{1,3}[a-z]?)(?![\w])", re.I)
CLAUSE = re.compile(r"\s*\(([\da-z\u0900-\u097f]{1,8})\)", re.I)
JOIN = re.compile(r"\s*(?:,\s*(?:(?:and|or)\s+)?|(?:and|or|&)\s+)", re.I)


@dataclass(frozen=True)
class Context:
    document_id: str
    raw_sha256: str
    kind: str
    root: str
    page: int
    reference_page: int
    anchor_verified: bool = False
    continuation_verified: bool = False
    following_proviso_verified: bool = False
    heading_bbox: tuple | None = None

    def usable(self):
        return (self.anchor_verified and bool(self.document_id)
                and bool(re.fullmatch(r"[a-f0-9]{64}", self.raw_sha256))
                and self.kind in {"section", "rule"} and bool(ROOT.fullmatch(self.root))
                and self.page > 0 and self.reference_page > 0
                and (self.page == self.reference_page
                     or self.reference_page == self.page + 1 and self.continuation_verified))


def normalize(question):
    value = unicodedata.normalize("NFKC", question)
    value = "".join(str(unicodedata.digit(c)) if c.isdigit() else c for c in value)
    for pattern, label in ((r"उप[- ]?धार(?:ाओं|ा|ाएँ|ाएं)", "subsections"),
                           (r"धार(?:ाओं|ा|ाएँ|ाएं)|\bdhara\b", "sections"),
                           (r"नियमों|नियम|\bniyam\b", "rules"), (r"और", "and"), (r"से", "to")):
        value = re.sub(pattern, label, value, flags=re.I)
    return value


def parse(question, context=None):
    value = normalize(question)
    refs, unresolved = [], []

    def add(kind, root, clauses, origin, start, end):
        item = {"kind": kind, "root": root.lower(), "clauses": clauses,
                "origin": origin, "normalized_question_start": start, "normalized_question_end": end}
        if origin.startswith("context"):
            item["context_provenance"] = {"document_id": context.document_id, "raw_sha256": context.raw_sha256,
                                          "page": context.page, "reference_page": context.reference_page,
                                          "heading_bbox": context.heading_bbox}
        if not any((r["kind"], r["root"], r["clauses"]) == (kind, root.lower(), clauses) for r in refs):
            refs.append(item)

    for label in LABEL.finditer(value):
        kind = label[1].lower().rstrip("s")
        position = label.end()
        if kind == "subsection":
            if not context or not context.usable():
                unresolved.append({"text": value[label.start():label.end() + 45], "reason": "subsection_parent_unverified"})
                continue
            root, kind, origin = context.root, context.kind, "context_subsection"
        else:
            number = ROOT.match(value, position)
            if not number:
                continue
            root, position, origin = number[1], number.end(), "explicit"
        clauses = []
        while match := CLAUSE.match(value, position):
            clauses.append(match[1].lower())
            position = match.end()
        if origin == "context_subsection" and not clauses:
            unresolved.append({"text": value[label.start():position], "reason": "missing_subsection_identifier"})
            continue
        add(kind, root, clauses, origin, label.start(), position)
        tail = re.match(r"\s*(?:to|through|[-–—])\s*(\d{1,3})(?![\w])", value[position:], re.I)
        if tail and root.isdigit() and not clauses:
            end = int(tail[1])
            start = int(root)
            if start <= end < start + MAX_RANGE:
                for n in range(start + 1, end + 1):
                    add(kind, str(n), [], "bounded_range", label.start(), position + tail.end())
            else:
                unresolved.append({"text": tail[0], "reason": "range_out_of_bounds"})
            continue
        for _ in range(MAX_RANGE - 1):
            join = JOIN.match(value, position)
            if not join:
                # Narrow coordinated topical form: "Rules 15 drawings and 16
                # models". Counts/amounts/durations are not inherited.
                if label[1].casefold() not in {"rules", "sections"}:
                    break
                join = re.match(r"\s+[\w -]{1,70}?\s+(?:and|&)\s+", value[position:], re.I)
                if not join:
                    break
                next_position = position + join.end()
                next_number = ROOT.match(value, next_position)
                if not next_number or re.match(r"\s*(?:days?|years?|months?|copies|times|rupees|percent|%)\b", value[next_number.end():], re.I):
                    break
                origin = "implicit_coordinated"
            else:
                next_position = join.end()
                origin = "coordinated"
            number = ROOT.match(value, next_position)
            if number:
                if re.match(r"\s*(?:days?|years?|months?|copies|times|rupees|percent|%)\b", value[number.end():], re.I):
                    break
                root, position, clauses = number[1], number.end(), []
            else:
                position, clauses = next_position, clauses[:-1]
                if not CLAUSE.match(value, position):
                    break
            while match := CLAUSE.match(value, position):
                clauses.append(match[1].lower())
                position = match.end()
            if label[1].casefold().startswith("subsection"):
                origin = "context_" + origin
            add(kind, root, clauses, origin, label.start(), position)
    for pattern, proviso in ((r"\bthe above section\b", False), (r"\bthe following proviso\b", True)):
        for match in re.finditer(pattern, value, re.I):
            if context and context.usable() and (not proviso or context.following_proviso_verified):
                add(context.kind, context.root, ["proviso"] if proviso else [], "context_anaphora", match.start(), match.end())
            else:
                unresolved.append({"text": match[0], "reason": "anaphora_context_unverified"})
    if not refs and re.search(r"\([\da-z]+\)", value, re.I) and not unresolved:
        unresolved.append({"text": value, "reason": "bare_clause_without_parent"})
    return {"version": VERSION, "references": refs, "unresolved": unresolved,
            "offset_scope": "normalized_question_unicode_codepoints", "corpus_anchors_modified": False}


def roots(question):
    result = {"sections": [], "rules": []}
    for reference in parse(question)["references"]:
        key = reference["kind"] + "s"
        if reference["root"] not in result[key]:
            result[key].append(reference["root"])
    return result
