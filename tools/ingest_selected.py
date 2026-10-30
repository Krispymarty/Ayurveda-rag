"""Compliant, explicitly selected IP India URL ingestion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipindia_corpus.pipeline import Pipeline, build_document, canonical, extract, score, structure_chunks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--discovered-from", required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "corpus")
    parser.add_argument("--terms-reviewed", action="store_true")
    args = parser.parse_args()
    url = canonical(args.url)
    parent = canonical(args.discovered_from)
    if not url or not parent or score(url, args.title) <= 0:
        raise ValueError("selected URL must be an in-scope official HTTPS resource")
    pipeline = Pipeline(args.out, 1.5, 20_000_000)
    if not pipeline.preflight(args.terms_reviewed):
        return 2
    fetch = pipeline.client.fetch(url)
    _, content, _ = extract(fetch)
    if len(content.strip()) < 80:
        pipeline.audit_event(url, "manual_review_required", "too_little_extractable_text")
        return 2
    document = build_document(fetch, args.title, content, parent)
    chunks = structure_chunks(content, document)
    if not chunks:
        pipeline.audit_event(url, "manual_review_required", "no_usable_chunks")
        return 2
    target = args.out / "documents" / document["document_id"]
    target.mkdir(parents=True, exist_ok=True)
    (target / "metadata.json").write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    pipeline.audit_event(url, "ingested", f"selected official resource; {len(chunks)} chunks")
    pipeline.export()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
