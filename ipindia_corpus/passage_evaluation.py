"""Frozen passage relevance evaluation through the running search API."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit
import requests
from .indexer import generation, model_revision
from .sync_runner import ROOT


def comparable(url):
    parts = urlsplit(url)
    return ((parts.hostname or "").removeprefix("www."), parts.path,
            tuple(sorted(parse_qsl(parts.query, keep_blank_values=True))))


def matches(hit, target):
    if hit["source"] != target["source"]:
        return False
    if comparable(hit["source_url"]) not in {comparable(url) for url in target["urls"]}:
        return False
    for key in ("section", "rule"):
        if target.get(key) is not None and str(hit.get(key, "")).lower() != str(target[key]).lower():
            return False
    if target.get("page") is not None and (hit.get("page_start") is None or not hit["page_start"] <= target["page"] <= hit["page_end"]):
        return False
    if target.get("contains") and target["contains"].casefold() not in hit["text"].casefold():
        return False
    return bool(target.get("section") or target.get("rule") or target.get("page") or target.get("contains"))


def validate_labels(dataset, documents, chunks, *, partial=False, issues=None):
    queries = dataset["queries"]
    if not partial and (not dataset.get("frozen") or any(not item.get("reviewed") for item in queries)):
        raise ValueError("passage_labels_not_frozen_and_reviewed")
    selected = [item for item in queries if item.get("reviewed")]
    if not partial and (len(selected) < 30 or any(sum(item["source"] == source for item in selected) < 10 for source in ("ipindia", "indiacode", "tkdl"))):
        raise ValueError("at_least_ten_reviewed_queries_per_source_required")
    for item in selected:
        if not item.get("targets") or not item.get("review_note"):
            raise ValueError("missing_passage_review")
        for target in item["targets"]:
            candidates = [chunk for chunk in chunks if matches(chunk, target)]
            if not candidates:
                # Offline diagnostic evaluations count unresolved targets as misses.
                # They must not certify a release or silently modify frozen labels.
                matching_docs = [doc for doc in documents if doc["source"] == target["source"]
                                 and comparable(doc["source_url"]) in {comparable(url) for url in target["urls"]}]
                if partial and issues is not None and matching_docs and all(doc["sha256"] in target.get("raw_sha256", []) for doc in matching_docs):
                    issues.append({"id": item["id"], "reason": "annotated_passage_missing", "target": target})
                    continue
                raise ValueError(f"annotated_passage_missing:{item['id']}")
            hashes = {doc["sha256"] for doc in documents if doc["document_id"] in {c["document_id"] for c in candidates}}
            if not hashes.issubset(set(target.get("raw_sha256", []))):
                raise ValueError(f"passage_labels_stale:{item['id']}")
    if not selected:
        raise ValueError("no_reviewed_passage_queries")
    return selected


def evaluate(root: Path = ROOT, *, api_url="http://127.0.0.1:8000", partial=False,
             labels_path=None, report_name=None):
    root = Path(root)
    path = Path(labels_path) if labels_path else root / "tests" / "retrieval_queries.json"
    frozen_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    dataset = json.loads(path.read_text(encoding="utf-8"))
    session = requests.Session()
    response = session.get(api_url.rstrip("/") + "/health", timeout=15)
    response.raise_for_status()
    active = response.json()["generation_id"]
    _, documents, chunks = generation(root, generation_id=active)
    coverage = {source: sum(doc["source"] == source for doc in documents) for source in ("ipindia", "indiacode", "tkdl")}
    ipindia_only = dataset.get("scope") in ("ipindia_only", "ipindia_reviewed_subset")
    if ipindia_only:
        receipt = path.with_suffix(".sha256")
        if not dataset.get("frozen") or not receipt.exists() or receipt.read_text().strip().lower() != frozen_hash:
            raise ValueError("ipindia_labels_not_frozen")
        minimum = 10 if dataset.get('scope') == 'ipindia_reviewed_subset' else 30
        if len(dataset["queries"]) < minimum or any(q["source"] != "ipindia" or not q.get("reviewed") for q in dataset["queries"]):
            raise ValueError("ipindia_requires_reviewed_queries")
    if not partial and not ipindia_only and any(not count for count in coverage.values()):
        raise ValueError(f"three_source_coverage_incomplete:{coverage}")
    label_issues = []
    queries = validate_labels(dataset, documents, chunks, partial=partial or ipindia_only, issues=label_issues)
    observed_ranking, observed_revisions = set(), set()

    def retrieve(item, filtered=False):
        parameters = {"q": item["query"], "top_k": 5}
        if filtered:
            parameters["source"] = item["source"]
        start = time.perf_counter()
        response = session.get(api_url.rstrip("/") + "/search", params=parameters, timeout=120)
        elapsed = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        result = response.json()
        observed_ranking.add(result.get("ranking_version", "v1"))
        observed_revisions.add(result.get("ranking_revision", ""))
        if len(observed_ranking) > 1 or len(observed_revisions) > 1:
            raise ValueError("ranking_changed_during_evaluation")
        if result["generation_id"] != active:
            raise ValueError("active_generation_changed_during_evaluation")
        return result, elapsed

    for item in queries:
        retrieve(item)
    rows, durations = [], []
    for repetition in range(3):
        for item in queries:
            result, elapsed = retrieve(item)
            recalled = [any(matches(hit, target) for hit in result["hits"]) for target in item["targets"]]
            durations.append(elapsed)
            rows.append({"id": item["id"], "source": item["source"], "query": item["query"], "pass": repetition + 1,
                         "recall_at_5": sum(recalled) / len(recalled), "hit_at_5": any(recalled),
                         "latency_ms": round(elapsed, 2), "returned_chunk_ids": [hit["chunk_id"] for hit in result["hits"]],
                         "returned_urls": [hit["source_url"] for hit in result["hits"]]})
    recall = sum(row["recall_at_5"] for row in rows) / len(rows)
    p95 = sorted(durations)[math.ceil(len(durations) * .95) - 1]
    diagnostics = []
    for item in queries:
        result, _ = retrieve(item, filtered=True)
        diagnostics.append({"id": item["id"], "source": item["source"], "recall_at_5": sum(any(matches(hit, target) for hit in result["hits"]) for target in item["targets"]) / len(item["targets"])})
    hardware_path = root / "reports" / "hardware.json"
    hardware = json.loads(hardware_path.read_text(encoding="utf-8-sig")) if hardware_path.exists() else {"physical_hardware": "not_recorded"}
    report = {"generation_id": active, "label_sha256": frozen_hash, "queries": len(queries), "observations": len(rows),
              "recall_at_5": recall, "hit_at_5": sum(row["hit_at_5"] for row in rows) / len(rows), "p95_ms": p95,
              "coverage": coverage, "pass": not partial and not ipindia_only and recall >= .90 and p95 <= 2000 and bool(hardware_path.exists()),
              "scope_pass": not label_issues and recall >= .90 and p95 <= 2000 and bool(hardware_path.exists()),
              "label_issues": label_issues,
              "scope": "ipindia_only" if ipindia_only else "reviewed_subset" if partial else "three_sources",
              "ranking_version": next(iter(observed_ranking)),
              "ranking_revision": next(iter(observed_revisions)),
              "cpu_inference_threads": 4,
              "partial": partial, "mode": "unfiltered HTTP hybrid passage retrieval", "model": "intfloat/multilingual-e5-small",
              "model_revision": model_revision(root), "hardware": hardware,
              "software": {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in ("requests", "fastapi", "psycopg", "sentence-transformers", "torch", "numpy")}},
              "per_source": {source: {"queries": sum(item["source"] == source for item in queries), "recall_at_5": sum(row["recall_at_5"] for row in rows if row["source"] == source) / max(1, sum(row["source"] == source for row in rows))} for source in coverage},
              "misses": [row for row in rows if row["recall_at_5"] < 1], "results": rows, "source_filtered_diagnostics": diagnostics}
    if hashlib.sha256(path.read_bytes()).hexdigest() != frozen_hash:
        raise ValueError("labels_changed_during_evaluation")
    target = root / "reports" / (report_name or ("retrieval_evaluation_ipindia_expanded.json" if ipindia_only else "retrieval_evaluation_partial.json" if partial else "retrieval_evaluation.json"))
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    archive = root / "reports" / "evaluations" / f"{active}_{hashlib.sha256(serialized.encode()).hexdigest()}.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        archive.write_text(serialized, encoding="utf-8")
    target.write_text(serialized, encoding="utf-8")
    return report
