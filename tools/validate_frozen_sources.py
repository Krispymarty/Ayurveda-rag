"""Check every frozen passage phrase directly against its archived original."""
import json
import argparse
import re
from pathlib import Path

from ipindia_corpus.indexer import generation
from ipindia_corpus.paths import archive_path
from ipindia_corpus.pipeline import Fetch, extract

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, default=ROOT / "tests/ipindia_retrieval_expanded.json")
    parser.add_argument('--out', default='frozen_source_validation.json')
    args = parser.parse_args()
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    _, documents, _ = generation(generation_id=labels.get("frozen_generation_id") or labels["baseline_generation"])
    by_url = {doc["source_url"]: doc for doc in documents}
    cache, failures, count = {}, [], 0
    for query in labels["queries"]:
        for target in query["targets"]:
            count += 1
            found = False
            for url in target["urls"]:
                doc = by_url.get(url)
                if not doc or doc["sha256"] not in target["raw_sha256"]:
                    continue
                if url not in cache:
                    path = archive_path(ROOT, doc["raw_path"])
                    raw = path.read_bytes()
                    _, text, _ = extract(Fetch(url, raw, "application/pdf" if raw.startswith(b"%PDF") else "text/html", doc["retrieved_at"], path))
                    cache[url] = text
                text = cache[url]
                if target.get("page"):
                    text = text.split(f"[Page {target['page']}]\n", 1)[-1].split("[Page ", 1)[0]
                found |= target["contains"].casefold() in re.sub(r"\s+", " ", text).casefold()
            if not found:
                failures.append({"query": query["id"], "target": target})
    report = {"pass": not failures, "queries": len(labels["queries"]), "passages": count,
              "failures": failures, "method": "Phrase and page checked against archived original extraction; no live requests."}
    (ROOT / "reports" / args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
