"""Bounded official Patent Rules HTML ingestion from the verified table of contents."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipindia_corpus.pipeline import Pipeline, build_document, canonical, extract, structure_chunks  # noqa: E402

INDEX = "https://ipindia.gov.in/acts/patent-rules-2003"
RULE_PATH = re.compile(r"^/acts/patent-rules-2003/rule-[\da-z-]+$", re.I)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "corpus")
    parser.add_argument("--max-rules", type=int, default=160)
    parser.add_argument("--terms-reviewed", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_rules <= 160:
        raise ValueError("max-rules must be between 1 and 160")
    pipeline = Pipeline(args.out, 1.5, 20_000_000)
    if not pipeline.preflight(args.terms_reviewed):
        return 2
    index = pipeline.client.fetch(INDEX)
    _, _, links = extract(index)
    selected = {}
    for href, label in links:
        url = canonical(href, INDEX)
        if url and RULE_PATH.match(urlsplit(url).path):
            selected.setdefault(url, label)
    pipeline.audit_event(INDEX, "discovery_only", f"focused official rule links: {len(selected)}")
    ingested = 0
    for url, label in list(selected.items())[:args.max_rules]:
        try:
            fetch = pipeline.client.fetch(url)
            _, content, _ = extract(fetch)
            if len(content.strip()) < 80:
                raise ValueError("manual_review_required: too_little_rule_text")
            title = f"The Patents Rules, 2003 - {label}"
            document = build_document(fetch, title, content, INDEX)
            document["document_type"] = "Rules"
            chunks = structure_chunks(content, document)
            if not chunks:
                raise ValueError("manual_review_required: no_rule_chunks")
            target = args.out / "documents" / document["document_id"]
            target.mkdir(parents=True, exist_ok=True)
            (target / "metadata.json").write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
            (target / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
            pipeline.audit_event(url, "ingested", f"focused rule page; {len(chunks)} chunks")
            ingested += 1
        except (ValueError, OSError, ImportError) as exc:
            pipeline.audit_event(url, "manual_review_required", str(exc))
    pipeline.export()
    print(json.dumps({"rule_links": len(selected), "attempted": min(len(selected), args.max_rules), "ingested": ingested}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
