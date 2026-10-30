"""Lossless page-local PDF units, with conservative printed provision anchors."""
import re

from .pipeline import CHAPTER, RULE, SECTION

PARSER_VERSION = "pdf-pages-v4"


def pdf_passages(text, document):
    state = {"chapter": "", "section": "", "rule": "", "subsection": "", "heading": ""}
    units, lines = [], []
    page = None
    contents = False

    def flush():
        if lines:
            units.append({**state, "page_start": page, "page_end": page, "text": "\n".join(lines),
                          "parser_version": PARSER_VERSION})
            lines.clear()

    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        marker = re.fullmatch(r"\[Page (\d+)\]", line)
        if marker:
            flush()
            page = int(marker[1])
            continue
        if not line:
            continue
        if re.search(r"^(?:TABLE OF CONTENTS|CONTENTS|ARRANGEMENT OF SECTIONS)\b", line, re.I):
            flush()
            contents = True
            state.update(section="", rule="", subsection="", heading="")
        chapter = CHAPTER.match(line)
        explicit = SECTION.match(line) if document["document_type"] == "Act" else RULE.match(line)
        # Bare lists in a table of contents are not provision evidence. Printed (1)
        # or an enactment dash is necessary for an unlabelled Act provision.
        numbered = re.match(r"^(\d+[A-Za-z]?)\.?\s+(\(1\).*)$", line)
        dash_heading = re.match(r"^(\d+[A-Za-z]?)\.\s+([A-Z].{2,120}?)[—–]", line)
        rule_heading = re.match(r"^(\d+[A-Za-z]?)\.\s+([A-Z][^\n]{2,120}?)(?:[.—–�]\s*(?:\(1\)|$))", line)
        if chapter:
            flush()
            state.update(chapter=chapter[1], section="", rule="", subsection="", heading=chapter[2])
            # Chapter headings occur in contents too; only provision evidence
            # or the end of a contents page clears the contents marker.
        elif not contents and (explicit or numbered or dash_heading or (document["document_type"] == "Rules" and rule_heading)):
            flush()
            match = explicit or numbered or dash_heading or rule_heading
            key = "rule" if document["document_type"] == "Rules" else "section"
            heading = explicit[2] if explicit else (dash_heading[2] if dash_heading else rule_heading[2] if rule_heading else "")
            state.update(section="", rule="", subsection="", heading=heading)
            state[key] = match[1]
        elif re.match(r"^\(\d+\)\s", line) and (state["section"] or state["rule"]):
            flush()
            state["subsection"] = re.match(r"^\((\d+)\)", line)[1]
        elif chapter is None and line.isupper() and len(line) < 130 and len(line.split()) > 1 and not contents:
            # Retain the printed heading in the text as well as metadata.
            flush()
            state["heading"] = line
        lines.append(line)
        if contents and re.search(r"(?:BE it enacted|Be it enacted|\(1\) These rules may be called)", line):
            contents = False
    flush()
    return units


def sidebar_headings(units, raw, document):
    """Attach only a unique, vertically aligned printed margin heading to an Act."""
    if document['document_type'] != 'Act':
        return units
    import fitz
    headings = {}
    with fitz.open(stream=raw, filetype='pdf') as pdf:
        for number, page in enumerate(pdf, 1):
            blocks = page.get_text('blocks')
            margins = [b for b in blocks if b[0] >= page.rect.width * .78
                       and b[2] - b[0] <= page.rect.width * .22
                       and 2 <= len(' '.join(b[4].split())) <= 140]
            for block in blocks:
                if block[0] >= page.rect.width * .78:
                    continue
                text = ' '.join(block[4].split())
                anchor = re.match(r'^(\d+[A-Za-z]?)\.?\s+\(1\)', text)
                if not anchor:
                    continue
                aligned = [b for b in margins if abs(b[1] - block[1]) <= 12 and re.search(r'[A-Za-z]', b[4])]
                if len(aligned) == 1:
                    heading = aligned[0]
                    headings[number, anchor[1]] = (' '.join(heading[4].split()), list(heading[:4]))
    for unit in units:
        match = headings.get((unit['page_start'], unit['section']))
        if match and not unit.get('heading'):
            unit.update(heading=match[0], heading_origin='pdf_sidebar', heading_page=unit['page_start'], heading_bbox=match[1])
    return units
