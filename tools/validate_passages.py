"""Verify that publication preserves all non-whitespace extracted source text."""
import hashlib
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from ipindia_corpus.indexer import generation
from ipindia_corpus.paths import archive_path
from ipindia_corpus.pipeline import Fetch, extract, structure_chunks

ROOT = Path(__file__).resolve().parents[1]


def stream_hash(units):
    return hashlib.sha256("".join(re.sub(r"\s+", "", item["text"]) for item in units).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-id")
    parser.add_argument("--out", default="passage_validation.json")
    args = parser.parse_args()
    identifier, documents, chunks = generation(ROOT, generation_id=args.generation_id)
    baseline_documents = {doc["document_id"]: doc for doc in (json.loads(line) for line in (ROOT / "data/documents/documents.jsonl").read_text(encoding="utf-8").splitlines())}
    before, after = defaultdict(list), defaultdict(list)
    for line in (ROOT / "data/chunks/chunks.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        before[item["document_id"]].append(item)
    for chunk in chunks:
        after[chunk["document_id"]].append(chunk)
    mismatches, page_errors, legacy_changes, pdf_count = [], [], [], 0
    for doc in documents:
        old = baseline_documents.get(doc["document_id"])
        if not old or old["sha256"] != doc["sha256"] or not before[doc["document_id"]]:
            path = archive_path(ROOT, doc["raw_path"])
            raw = path.read_bytes()
            _, text, _ = extract(Fetch(doc["source_url"], raw, "application/pdf" if raw.startswith(b"%PDF") else "text/html", doc["retrieved_at"], path))
            before[doc["document_id"]] = structure_chunks(text, doc)
        expected_hash = stream_hash(before[doc["document_id"]])
        if (after[doc["document_id"]][0].get("parser_version") or "").startswith("pdf-pages-v"):
            pdf_count += 1
            path = archive_path(ROOT, doc["raw_path"])
            _, source_text, _ = extract(Fetch(doc["source_url"], path.read_bytes(), "application/pdf", doc["retrieved_at"], path))
            source_pages = {}
            page = None
            for line in source_text.splitlines():
                marker = re.fullmatch(r"\[Page (\d+)\]", line.strip())
                if marker:
                    page = int(marker[1])
                    source_pages[page] = []
                elif page is not None:
                    source_pages[page].append(line)
            actual_pages = defaultdict(list)
            for chunk in after[doc["document_id"]]:
                if chunk["page_start"] != chunk["page_end"]:
                    page_errors.append(chunk["chunk_id"])
                actual_pages[chunk["page_start"]].append(chunk)
            for number, source_lines in source_pages.items():
                if stream_hash([{"text": "\n".join(source_lines)}]) != stream_hash(actual_pages[number]):
                    page_errors.append(f"{doc['document_id']}:page:{number}")
            expected_hash = stream_hash([{"text": "\n".join('\n'.join(v) for v in source_pages.values())}])
            if stream_hash(before[doc["document_id"]]) != stream_hash(after[doc["document_id"]]):
                legacy_changes.append(doc["document_id"])
        if expected_hash != stream_hash(after[doc["document_id"]]):
            mismatches.append(doc["document_id"])
    report = {"generation_id": identifier, "documents": len(documents), "chunks": len(chunks),
              "pass": not mismatches and not page_errors, "text_stream_mismatches": mismatches,
              "pdf_documents_checked_against_raw": pdf_count, "page_errors": page_errors,
              "legacy_stream_changes": legacy_changes,
              "short_units_retained": sum(chunk.get("short_unit", False) for chunk in chunks),
              "max_chunk_chars": max(len(chunk["text"]) for chunk in chunks),
              "comparison": "Repaired PDFs checked page-by-page against archived extraction including headers; other documents checked against baseline. Whitespace normalized."}
    (ROOT / "reports" / args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
