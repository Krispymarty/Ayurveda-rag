"""Review archived passage targets and freeze labels before retrieval tuning."""
import argparse
import json
from datetime import datetime, timezone
from .indexer import generation
from .passage_evaluation import matches, validate_labels
from .sync_runner import ROOT, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["review", "freeze"])
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--query-id")
    parser.add_argument("--url", action="append")
    parser.add_argument("--section")
    parser.add_argument("--rule")
    parser.add_argument("--page", type=int)
    parser.add_argument("--contains")
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--note", required=True)
    args = parser.parse_args()
    path = ROOT / "tests" / "retrieval_queries.json"
    dataset = json.loads(path.read_text())
    _, documents, chunks = generation(ROOT, generation_id=args.generation_id)
    if not args.reviewer.strip() or not args.note.strip():
        raise ValueError("reviewer_and_source_passage_note_required")
    if args.command == "review":
        if dataset.get("frozen"):
            raise ValueError("frozen_labels_require_a_separate_new_evaluation_set")
        item = next((q for q in dataset["queries"] if q["id"] == args.query_id), None)
        if not item or not args.url:
            raise ValueError("query_id_and_official_urls_required")
        if dataset.get("partial_labels_frozen") and item.get("reviewed"):
            raise ValueError("existing_partial_labels_frozen_create_new_evaluation_set")
        target = {"source": item["source"], "urls": args.url, "section": args.section,
                  "rule": args.rule, "page": args.page, "contains": args.contains}
        candidates = [chunk for chunk in chunks if matches(chunk, target)]
        if not candidates:
            raise ValueError("reviewed_passage_not_present_in_generation")
        identifiers = {chunk["document_id"] for chunk in candidates}
        target["raw_sha256"] = sorted({doc["sha256"] for doc in documents if doc["document_id"] in identifiers})
        item.update(targets=[target], reviewed=True, reviewer=args.reviewer,
                    review_note=args.note, reviewed_at=datetime.now(timezone.utc).isoformat())
    else:
        dataset["frozen"] = True
        validate_labels(dataset, documents, chunks)
        dataset.update(frozen_generation_id=args.generation_id, frozen_by=args.reviewer,
                       freeze_note=args.note, frozen_at=datetime.now(timezone.utc).isoformat())
    atomic_json(path, dataset)
    print(json.dumps({"frozen": dataset["frozen"], "reviewed": sum(item["reviewed"] for item in dataset["queries"])}))


if __name__ == "__main__":
    main()
