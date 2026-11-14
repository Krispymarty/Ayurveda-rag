"""HTTP checks against a real research service; local runs are explicitly labelled."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from urllib.parse import urlsplit

import requests

from ipindia_corpus.indexer import generation
from ipindia_corpus.passage_evaluation import matches
from ipindia_corpus.sync_runner import ROOT


def check(base, *, root=ROOT, allow_local=False, baseline=None, host=None, trace=None):
    parsed = urlsplit(base)
    local = parsed.hostname in {"localhost", "127.0.0.1"}
    if parsed.scheme != "https" and not (local and allow_local and parsed.scheme == "http"):
        raise ValueError("https_required_except_explicit_loopback_test")
    if parsed.username or parsed.password or parsed.query or parsed.fragment or (host and not local):
        raise ValueError("invalid_smoke_target")
    root = Path(root)
    session = requests.Session()
    if host:
        session.headers["Host"] = host
    _, documents, chunks = generation(root, generation_id="20261014T204940358650Z")
    by_chunk = {x["chunk_id"]: x for x in chunks}
    by_doc = {x["document_id"]: x for x in documents}
    cases_path = root / "data/benchmarks/ipindia-reviewed-20261015-v1/expanded.json"
    raw = cases_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != cases_path.with_suffix(".sha256").read_text().strip():
        raise ValueError("historical_benchmark_hash_changed")
    cases = json.loads(raw)["queries"]
    checks, observations, baseline_rows = {}, [], []

    def request(method, path, **kwargs):
        return session.request(method, base.rstrip("/") + path, timeout=15, **kwargs)

    home = request("GET", "/")
    checks["homepage"] = home.status_code == 200 and "AyurIP" in home.text
    checks["assets"] = all(request("GET", p).status_code == 200 for p in ("/assets/app.js", "/assets/styles.css"))
    health = request("GET", "/health")
    checks["readiness"] = health.status_code == 200 and health.json().get("generation_id") == "20261014T204940358650Z"
    checks["liveness"] = request("GET", "/live").status_code == 200
    sources = request("GET", "/sources")
    source_body = sources.text
    checks["sources_no_internal_paths"] = sources.status_code == 200 and not any(x in source_body for x in ("raw_path", "E:\\\\", "/app/", "postgresql://", "password"))
    for path in ("/docs", "/openapi.json", "/api/passport", "/data/raw", "/assets/%2e%2e/.env"):
        checks["unavailable_" + path] = request("GET", path).status_code == 404
    checks["host_rejected"] = request("GET", "/live", headers={"Host": "evil.invalid"}).status_code == 400
    checks["origin_rejected"] = request("GET", "/live", headers={"Origin": "https://evil.invalid"}).status_code == 403
    checks["oversized_query"] = request("GET", "/search", params={"q": "x" * 501}).status_code in {413, 422}
    answer = request("POST", "/answer", json={"question": "Archived patent source research"})
    checks["public_answer_blocked"] = answer.status_code == 503 and answer.json().get("reason") == "three_source_not_certified"

    def retrieve(url, item):
        start = time.perf_counter()
        try:
            result = session.get(url.rstrip("/") + "/search", params={"q": item["query"], "top_k": 5}, timeout=15)
        except requests.RequestException as exc:
            row = {"id": item["id"], "http": None, "latency_ms": (time.perf_counter() - start) * 1000, "error": type(exc).__name__}
            if trace:
                trace.write(json.dumps({"target": url, **row}) + "\n")
                trace.flush()
            return row
        elapsed = (time.perf_counter() - start) * 1000
        if result.status_code != 200:
            row = {"id": item["id"], "http": result.status_code, "latency_ms": elapsed, "error": "http_failure"}
            if trace:
                trace.write(json.dumps({"target": url, **row}) + "\n")
                trace.flush()
            return row
        body = result.json()
        hits = body["hits"]
        integrity = True
        for hit in hits:
            archived = by_chunk.get(hit["chunk_id"])
            doc = by_doc.get(hit["document_id"])
            integrity &= bool(archived and doc and hit["text"] == archived["text"] and hit["document_id"] == archived["document_id"]
                              and hit["source_url"] == archived["source_url"] == doc["source_url"]
                              and hit["raw_sha256"] == archived["raw_sha256"] == doc["sha256"] and hit["status"] == "unknown")
        row = {"id": item["id"], "http": 200, "latency_ms": elapsed, "integrity": integrity,
                "generation_id": body["generation_id"], "ranking_version": body["ranking_version"],
                "chunk_ids": [h["chunk_id"] for h in hits],
                "recall_at_5": sum(any(matches(h, target) for h in hits) for target in item["targets"]) / len(item["targets"])}
        if trace:
            trace.write(json.dumps({"target": url, **row}) + "\n")
            trace.flush()
        return row

    if baseline:
        for item in cases:
            baseline_rows.append(retrieve(baseline, item))
            time.sleep(1.02)  # Outside request timing; stay inside public request quota.
    for item in cases[:3]:
        retrieve(base, item)
        time.sleep(1.02)
    for repetition in range(3):
        for item in cases:
            row = retrieve(base, item)
            row["pass"] = repetition + 1
            observations.append(row)
            time.sleep(1.02)
    valid = [r for r in observations if r.get("http") == 200]
    durations = sorted(r["latency_ms"] for r in observations)
    checks["all_attempts_completed"] = len(valid) == len(observations)
    checks["all_citations_match_published_archive"] = all(r.get("integrity") for r in observations)
    checks["protected_generation_ranker"] = all(r.get("generation_id") == "20261014T204940358650Z" and r.get("ranking_version") == "v2" for r in observations)
    equivalent = None
    if baseline_rows:
        before = {r["id"]: r for r in baseline_rows}
        equivalent = all(r.get("http") == 200 and r.get("chunk_ids") == before[r["id"]].get("chunk_ids") for r in observations)
        checks["exact_baseline_result_identity"] = equivalent
    # Exercise cheap validation requests, preserving any rate rejection as an observation.
    rate_codes = [request("GET", "/search", params={"q": "x"}).status_code for _ in range(70)]
    checks["rate_limit_observed"] = 429 in rate_codes
    return {"scope": "local release integration" if local else "external HTTPS deployment", "cloud": not local,
            "target": base, "checks": checks, "health": health.json(), "queries": len(cases),
            "label_sha256": hashlib.sha256(raw).hexdigest(), "independent_legal_review": False,
            "observations": observations, "baseline_observations": baseline_rows,
            "recall_at_5": sum(r.get("recall_at_5", 0) for r in observations) / len(observations),
            "warm_p95_ms": durations[math.ceil(len(durations) * .95) - 1], "passes": 3,
            "strict_result_identity": equivalent, "rate_statuses": rate_codes,
            "all_checks_pass": all(checks.values())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--baseline-url")
    parser.add_argument("--allow-local", action="store_true")
    parser.add_argument("--host")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.with_suffix(".attempts.jsonl").open("x", encoding="utf-8") as trace:
        result = check(args.base_url, allow_local=args.allow_local, baseline=args.baseline_url, host=args.host, trace=trace)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ("observations", "baseline_observations", "rate_statuses")}))
    raise SystemExit(0 if result["all_checks_pass"] else 1)
