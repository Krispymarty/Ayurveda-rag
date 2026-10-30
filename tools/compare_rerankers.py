"""Paired local hybrid versus cross-encoder DEVELOPMENT benchmark, no activation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from ipindia_corpus.indexer import _connect, generation, model_revision
from ipindia_corpus.passage_evaluation import matches, validate_labels
from ipindia_corpus.paths import immutable_write
from ipindia_corpus.reranker import CONFIG, configuration, load_lock, rerank, sha256
from ipindia_corpus.sync_runner import ROOT


def selected_state(dsn=None):
    import psycopg
    with (psycopg.connect(dsn) if dsn else _connect()) as connection:
        values = dict(connection.execute("SELECT key,value FROM app_state WHERE key IN ('active_generation','ranking_version')").fetchall())
    if not values.get("active_generation"):
        raise ValueError("no_active_index")
    return values


def metrics(hits, targets):
    ranks = [next((index + 1 for index, hit in enumerate(hits) if matches(hit, target)), None) for target in targets]
    recalled = sum(rank is not None for rank in ranks)
    first = min((rank for rank in ranks if rank is not None), default=None)
    return {"recall_at_5": recalled / len(targets), "reciprocal_rank_at_5": 1 / first if first else 0,
            "target_ranks": ranks, "chunk_ids": [hit["chunk_id"] for hit in hits]}


def percentile(values, p=.95):
    return sorted(values)[math.ceil(len(values) * p) - 1]


def compare(root=ROOT, *, labels_path=None, preview=False):
    from ipindia_corpus.search_api import search
    from ipindia_corpus import retrieval, retrieval_legacy
    root = Path(root)
    labels_path = Path(labels_path or root / "data/benchmarks/ipindia-reviewed-20261015-v1/expanded.json")
    if not labels_path.is_absolute():
        labels_path = root / labels_path
    frozen_hash = sha256(labels_path)
    receipt = labels_path.with_suffix(".sha256")
    if not receipt.exists() or receipt.read_text().strip() != frozen_hash:
        raise ValueError("benchmark_receipt_mismatch")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if not labels.get("frozen") or labels.get("scope") != "ipindia_only" or len(labels.get("queries", [])) < 30:
        raise ValueError("reviewed_frozen_ipindia_benchmark_required")
    if any(not q.get("reviewed") or q.get("source") != "ipindia" for q in labels["queries"]):
        raise ValueError("reviewed_ipindia_queries_required")
    production_dsn = os.environ["DATABASE_URL"]
    production_before = selected_state()
    if preview:
        from ipindia_corpus.retrieval_preview import STATE, preview_dsn
        os.environ["DATABASE_URL"] = preview_dsn(json.loads(STATE.read_text())["database"])
    try:
        from ipindia_corpus.cli_readiness import initialize_for_cli
        initialize_for_cli()
        before = selected_state()
        active = before["active_generation"]
        manifest = root / "data/generations" / active / "manifest.json"
        manifest_hash = sha256(manifest)
        _, documents, chunks = generation(root, generation_id=active)
        label_issues = []
        queries = validate_labels(labels, documents, chunks, partial=True, issues=label_issues)
        if label_issues:
            raise ValueError("unresolved_benchmark_targets")
        if labels.get("frozen_generation_id") != active:
            raise ValueError("benchmark_generation_mismatch")
        load_lock(root)
        config = configuration(root)
        lock_hash = sha256(root / "data/reranker.lock.json")
        ranker = retrieval if before.get("ranking_version", "v1") == "v2" else retrieval_legacy
        ranker_hash = sha256(ranker.__file__)
        candidate_config = {"generation_id": active, "manifest_sha256": manifest_hash,
                            "ranking_version": before.get("ranking_version", "v1"), "ranking_revision": ranker_hash,
                            "label_sha256": frozen_hash, "candidate_count": CONFIG["max_candidates"],
                            "embedding_revision": model_revision(root), "source_filter": None, "domain_filter": None}
        candidate_config["fingerprint"] = hashlib.sha256(json.dumps(candidate_config, sort_keys=True).encode()).hexdigest()
        for item in queries:
            result = search(item["query"], top_k=CONFIG["max_candidates"])
            if result["generation_id"] != active:
                raise ValueError("generation_changed_during_warmup")
            rerank(item["query"], result["hits"], root=root)
        rows = []
        candidate_fingerprints = {}
        for repetition in range(3):
            for item in queries:
                started = time.perf_counter()
                result = search(item["query"], top_k=CONFIG["max_candidates"])
                retrieved_at = time.perf_counter()
                if result["generation_id"] != active or result["ranking_revision"] != ranker_hash:
                    raise ValueError("retrieval_changed_during_comparison")
                hits = result["hits"]
                candidate_hash = hashlib.sha256(json.dumps(hits, sort_keys=True).encode()).hexdigest()
                if item["id"] in candidate_fingerprints and candidate_fingerprints[item["id"]] != candidate_hash:
                    raise ValueError("candidates_changed_between_passes")
                candidate_fingerprints[item["id"]] = candidate_hash
                reranked, details = rerank(item["query"], hits, root=root)
                ended = time.perf_counter()
                if details["fingerprint"] != config["fingerprint"]:
                    raise ValueError("reranker_configuration_changed")
                baseline = metrics(hits[:5], item["targets"])
                reranked_metrics = metrics(reranked, item["targets"])
                rows.append({"id": item["id"], "query": item["query"], "source": item["source"], "pass": repetition + 1,
                             "baseline": baseline, "reranked": reranked_metrics,
                             "candidate_target_recall_at_20": metrics(hits, item["targets"])["recall_at_5"],
                             "candidate_sha256": candidate_hash, "actual_candidates": len(hits),
                             "retrieval_ms": (retrieved_at - started) * 1000,
                             "rerank_ms": (ended - retrieved_at) * 1000, "total_ms": (ended - started) * 1000,
                             "regression": reranked_metrics["recall_at_5"] < baseline["recall_at_5"]})
        if selected_state() != before or sha256(manifest) != manifest_hash or sha256(labels_path) != frozen_hash:
            raise ValueError("benchmark_state_changed")
        if sha256(root / "data/reranker.lock.json") != lock_hash or sha256(ranker.__file__) != ranker_hash:
            raise ValueError("model_or_ranker_changed")
        load_lock(root)
        generation(root, generation_id=active)
        summary = {}
        for name, latency in (("baseline", "retrieval_ms"), ("reranked", "total_ms")):
            summary[name] = {"recall_at_5": statistics.mean(row[name]["recall_at_5"] for row in rows),
                             "mrr_at_5": statistics.mean(row[name]["reciprocal_rank_at_5"] for row in rows),
                             "warm_p95_ms": percentile([row[latency] for row in rows]),
                             "mean_ms": statistics.mean(row[latency] for row in rows)}
        report = {"scope": "English IP India DEVELOPMENT benchmark; not held-out or three-source certification",
                  "checked_at": datetime.now(timezone.utc).isoformat(), "preview_database": preview,
                  "queries": len(queries), "passes": 3, "observations": len(rows),
                  "generation_id": active, "label_path": labels_path.relative_to(root).as_posix(), "label_sha256": frozen_hash,
                  "model": config, "model_lock_sha256": lock_hash, "candidate_configuration": candidate_config,
                  "summary": summary, "rerank_warm_p95_ms": percentile([row["rerank_ms"] for row in rows]),
                  "regressions": [row for row in rows if row["regression"]],
                  "misses": {name: [row for row in rows if row[name]["recall_at_5"] < 1] for name in ("baseline", "reranked")},
                  "per_source": {"ipindia": summary}, "results": rows,
                  "hardware": json.loads((root / "reports/hardware.json").read_text(encoding="utf-8-sig")),
                  "runtime": {"platform": platform.platform(), "cpu_count": os.cpu_count(), "inference_device": "cpu",
                              "threads": CONFIG["cpu_threads"], "python": platform.python_version()},
                  "software": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "sentence-transformers", "huggingface-hub", "numpy", "psycopg")},
                  "adopted": False, "production_ranking_changed": False}
        report["recommendation"] = "retain_current_hybrid_pending_review"
        if summary["reranked"]["recall_at_5"] > summary["baseline"]["recall_at_5"] and summary["reranked"]["warm_p95_ms"] <= 2000 and not report["regressions"]:
            report["recommendation"] = "quality_improvement_observed_review_latency_and_independent_queries_before_adoption"
        if selected_state(production_dsn) != production_before:
            raise ValueError("production_selection_changed_during_benchmark")
        report["production_selection_unchanged"] = True
        report["latency_measurement"] = "warm in-process search of 20 candidates; reranked total adds local CPU inference; excludes HTTP and serialization"
        serialized = json.dumps(report, indent=2, ensure_ascii=False).encode()
        archive = root / "reports/evaluations" / ("reranker_" + hashlib.sha256(serialized).hexdigest() + ".json")
        immutable_write(archive, serialized)
        (root / "reports/reranker_comparison.json").write_bytes(serialized)
    finally:
        os.environ["DATABASE_URL"] = production_dsn
    if selected_state() != production_before:
        raise ValueError("production_selection_changed_during_benchmark")
    return {"report": "reports/reranker_comparison.json", "archive": archive.relative_to(root).as_posix(),
            "candidate_fingerprint": candidate_config["fingerprint"], "summary": summary,
            "recommendation": report["recommendation"], "adopted": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--labels", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(labels_path=args.labels, preview=args.preview), indent=2))
